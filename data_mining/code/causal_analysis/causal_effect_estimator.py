"""
Causal Effect Estimation using DoWhy framework.

For each direct cause of is_finally_correct identified by causal discovery,
estimate the Average Treatment Effect (ATE) using multiple methods
(backdoor linear, propensity score, IPW) and run refutation tests.
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import dowhy
from dowhy import CausalModel

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_data_from_path

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="dowhy")


# --- ICML-style plot settings ---
def setup_plot_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["legend.title_fontsize"] = 18
    plt.rcParams["legend.fontsize"] = 16


class CausalEffectEstimator:
    """Estimate causal effects using DoWhy with multiple estimation methods and refutation."""

    def __init__(
        self,
        data_path: str = "data_mining/data/merged_datasets.csv",
        feature_list_path: str = "data_mining/results_causal_complete/feature_selection/selected_features.csv",
        causes_path: str = "data_mining/results_causal_complete/causal_discovery/direct_causes.json",
        edges_path: str = "data_mining/results_causal_complete/causal_discovery/all_edges.csv",
        output_dir: str = "data_mining/results_causal_complete/causal_effects",
        max_sample: int = 15000,
    ):
        self.data_path = Path(data_path)
        self.feature_list_path = Path(feature_list_path)
        self.causes_path = Path(causes_path)
        self.edges_path = Path(edges_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_sample = max_sample
        self.target = "is_finally_correct"

    def load_data(self) -> Tuple[pd.DataFrame, List[str], Dict]:
        """Load data, feature list, and causal discovery results."""
        # Feature list
        feat_df = pd.read_csv(self.feature_list_path)
        selected = feat_df["feature"].tolist()

        # Direct causes
        with open(self.causes_path) as f:
            causes_info = json.load(f)

        # Full data
        df = load_data_from_path(self.data_path)
        cols = [c for c in selected if c in df.columns] + [self.target]
        cols = list(dict.fromkeys(cols))  # deduplicate
        sub = df[cols].dropna()

        if len(sub) > self.max_sample:
            sub = sub.sample(n=self.max_sample, random_state=42)
            logger.info("Subsampled to %d rows", self.max_sample)

        logger.info("Loaded %d rows, %d columns", *sub.shape)
        return sub, selected, causes_info

    def build_causal_graph_gml(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str],
        edges_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """Build a GML graph string for DoWhy.

        Conservative approach: use all other selected features as common causes
        (potential confounders) unless we have specific edge information.
        """
        nodes = list(set([treatment, outcome] + confounders))
        lines = ["graph [directed 1"]
        for node in nodes:
            lines.append(f'  node [id "{node}" label "{node}"]')

        # Treatment -> Outcome
        lines.append(f'  edge [source "{treatment}" target "{outcome}"]')

        # Confounders -> Treatment and Confounders -> Outcome
        for c in confounders:
            if c != treatment and c != outcome:
                lines.append(f'  edge [source "{c}" target "{treatment}"]')
                lines.append(f'  edge [source "{c}" target "{outcome}"]')

        lines.append("]")
        return "\n".join(lines)

    def binarize_treatment(self, data: pd.DataFrame, treatment: str) -> pd.DataFrame:
        """Binarize continuous treatment at median for methods that need it."""
        df = data.copy()
        median_val = df[treatment].median()
        col_name = f"{treatment}_binary"
        df[col_name] = (df[treatment] > median_val).astype(int)
        return df, col_name

    def estimate_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        selected_features: List[str],
    ) -> Dict:
        """Estimate causal effect of treatment on outcome with multiple methods."""
        outcome = self.target
        confounders = [
            f
            for f in selected_features
            if f != treatment and f != outcome and f in data.columns
        ]

        # Limit confounders to reduce computational cost
        # Use top confounders by correlation with treatment
        if len(confounders) > 15:
            corrs = (
                data[confounders]
                .corrwith(data[treatment])
                .abs()
                .sort_values(ascending=False)
            )
            confounders = corrs.head(15).index.tolist()

        results = {
            "treatment": treatment,
            "outcome": outcome,
            "n_confounders": len(confounders),
        }

        # Build causal graph
        gml = self.build_causal_graph_gml(treatment, outcome, confounders)

        try:
            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                graph=gml,
            )

            # Identify estimand
            identified_estimand = model.identify_effect(
                proceed_when_unidentifiable=True
            )
            results["estimand"] = str(identified_estimand)

            # --- Method 1: Linear regression (backdoor) ---
            try:
                estimate_lr = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.linear_regression",
                    confidence_intervals=True,
                    test_significance=True,
                )
                results["linear_regression"] = {
                    "ate": float(estimate_lr.value),
                    "p_value": (
                        float(
                            estimate_lr.test_stat_significance().get("p_value", np.nan)
                        )
                        if estimate_lr.test_stat_significance()
                        else np.nan
                    ),
                }
                logger.info("  LR ATE for %s: %.6f", treatment, estimate_lr.value)
            except Exception as e:
                logger.warning("  LR estimation failed for %s: %s", treatment, e)
                results["linear_regression"] = {"ate": np.nan, "error": str(e)}

            # --- Method 2: Propensity Score Stratification ---
            # Need binary treatment
            data_bin, bin_col = self.binarize_treatment(data, treatment)
            try:
                model_bin = CausalModel(
                    data=data_bin,
                    treatment=bin_col,
                    outcome=outcome,
                    graph=self.build_causal_graph_gml(bin_col, outcome, confounders),
                )
                estimand_bin = model_bin.identify_effect(
                    proceed_when_unidentifiable=True
                )
                estimate_ps = model_bin.estimate_effect(
                    estimand_bin,
                    method_name="backdoor.propensity_score_stratification",
                )
                results["propensity_score"] = {
                    "ate": float(estimate_ps.value),
                }
                logger.info("  PS ATE for %s: %.6f", treatment, estimate_ps.value)
            except Exception as e:
                logger.warning("  PS estimation failed for %s: %s", treatment, e)
                results["propensity_score"] = {"ate": np.nan, "error": str(e)}

            # --- Method 3: IPW ---
            try:
                estimate_ipw = model_bin.estimate_effect(
                    estimand_bin,
                    method_name="backdoor.propensity_score_weighting",
                )
                results["ipw"] = {
                    "ate": float(estimate_ipw.value),
                }
                logger.info("  IPW ATE for %s: %.6f", treatment, estimate_ipw.value)
            except Exception as e:
                logger.warning("  IPW estimation failed for %s: %s", treatment, e)
                results["ipw"] = {"ate": np.nan, "error": str(e)}

            # --- Refutation tests (on linear regression estimate) ---
            if "error" not in results.get("linear_regression", {}):
                results["refutations"] = self.run_refutations(
                    model, identified_estimand, estimate_lr
                )

        except Exception as e:
            logger.error("DoWhy model failed for %s: %s", treatment, e)
            results["error"] = str(e)

        return results

    def run_refutations(self, model, estimand, estimate) -> Dict:
        """Run refutation tests to validate causal estimate."""
        refutations = {}

        # 1. Random Common Cause
        try:
            ref_rcc = model.refute_estimate(
                estimand,
                estimate,
                method_name="random_common_cause",
                num_simulations=50,
            )
            refutations["random_common_cause"] = {
                "new_effect": float(ref_rcc.new_effect),
                "refutation_result": (
                    str(ref_rcc.refutation_result)
                    if hasattr(ref_rcc, "refutation_result")
                    else "N/A"
                ),
            }
            logger.info("    RCC refutation: new_effect=%.6f", ref_rcc.new_effect)
        except Exception as e:
            refutations["random_common_cause"] = {"error": str(e)}

        # 2. Placebo Treatment
        try:
            ref_placebo = model.refute_estimate(
                estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=50,
            )
            refutations["placebo_treatment"] = {
                "new_effect": float(ref_placebo.new_effect),
                "refutation_result": (
                    str(ref_placebo.refutation_result)
                    if hasattr(ref_placebo, "refutation_result")
                    else "N/A"
                ),
            }
            logger.info(
                "    Placebo refutation: new_effect=%.6f", ref_placebo.new_effect
            )
        except Exception as e:
            refutations["placebo_treatment"] = {"error": str(e)}

        # 3. Data Subset
        try:
            ref_subset = model.refute_estimate(
                estimand,
                estimate,
                method_name="data_subset_refuter",
                subset_fraction=0.8,
                num_simulations=50,
            )
            refutations["data_subset"] = {
                "new_effect": float(ref_subset.new_effect),
                "refutation_result": (
                    str(ref_subset.refutation_result)
                    if hasattr(ref_subset, "refutation_result")
                    else "N/A"
                ),
            }
            logger.info("    Subset refutation: new_effect=%.6f", ref_subset.new_effect)
        except Exception as e:
            refutations["data_subset"] = {"error": str(e)}

        return refutations

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot_ate_comparison(self, all_results: List[Dict]):
        """Bar chart comparing ATE estimates across methods for each treatment."""
        treatments = []
        methods = ["linear_regression", "propensity_score", "ipw"]
        method_labels = ["Linear Regression", "Propensity Score", "IPW"]
        plot_data = []

        for res in all_results:
            t = res["treatment"]
            for method, label in zip(methods, method_labels):
                ate = res.get(method, {}).get("ate", np.nan)
                if not np.isnan(ate):
                    plot_data.append({"treatment": t, "method": label, "ATE": ate})

        if not plot_data:
            logger.warning("No ATE data to plot")
            return

        df_plot = pd.DataFrame(plot_data)

        setup_plot_style()
        color_map = {
            "Linear Regression": "#D73027",
            "Propensity Score": "#4575B4",
            "IPW": "#91BFD8",
        }
        with sns.plotting_context("paper", font_scale=1.4):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=df_plot,
                x="treatment",
                y="ATE",
                hue="method",
                ax=ax,
                palette=color_map,
                edgecolor="white",
                linewidth=0.8,
                saturation=0.9,
                width=0.65,
            )
            # ax.set_title(
            #     "ATE Estimates by Method", fontsize=20, fontweight="bold", pad=15
            # )
            ax.set_xlabel("Treatment Variable", fontsize=18)
            ax.set_ylabel("Average Treatment Effect", fontsize=18)
            ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
            # Use symlog scale to handle extreme values while keeping small values visible
            ax.set_yscale("symlog", linthresh=0.5)
            plt.xticks(rotation=25, ha="right", fontsize=9.5)
            ax.legend(title="Method", fontsize=14, title_fontsize=16, frameon=False)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            sns.despine(ax=ax, top=True, right=True)
            plt.tight_layout()
            out_path = self.output_dir / "ate_comparison.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved ATE comparison to %s", out_path)

    def plot_refutation_summary(self, all_results: List[Dict]):
        """Heatmap of refutation results."""
        records = []
        for res in all_results:
            t = res["treatment"]
            refs = res.get("refutations", {})
            original_ate = res.get("linear_regression", {}).get("ate", np.nan)
            for ref_name, ref_data in refs.items():
                new_effect = ref_data.get("new_effect", np.nan)
                if (
                    not np.isnan(original_ate)
                    and not np.isnan(new_effect)
                    and original_ate != 0
                ):
                    change_pct = (
                        abs(new_effect - original_ate) / abs(original_ate) * 100
                    )
                else:
                    change_pct = np.nan
                records.append(
                    {
                        "treatment": t,
                        "refutation": ref_name,
                        "original_ate": original_ate,
                        "new_effect": new_effect,
                        "change_pct": change_pct,
                    }
                )

        if not records:
            return

        df = pd.DataFrame(records)
        pivot = df.pivot_table(
            index="treatment", columns="refutation", values="change_pct"
        )

        setup_plot_style()
        with sns.plotting_context("paper", font_scale=1.4):
            fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 1.0)))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                ax=ax,
                annot_kws={"fontsize": 14},
                cbar_kws={"label": "% Change from Original ATE"},
            )
            # ax.set_title(
            #     "Refutation Test Results (% ATE Change)",
            #     fontsize=20,
            #     fontweight="bold",
            #     pad=15,
            # )
            ax.tick_params(axis="both", labelsize=13)
            plt.tight_layout()
            out_path = self.output_dir / "refutation_summary.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved refutation summary to %s", out_path)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_report(self, all_results: List[Dict]) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("CAUSAL EFFECT ESTIMATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Target variable: {self.target}")
        lines.append(f"Framework: DoWhy")
        lines.append(
            f"Estimation methods: Linear Regression, Propensity Score Stratification, IPW"
        )
        lines.append(
            f"Refutation tests: Random Common Cause, Placebo Treatment, Data Subset"
        )
        lines.append("")

        for res in all_results:
            t = res["treatment"]
            lines.append("-" * 80)
            lines.append(f"TREATMENT: {t}")
            lines.append(f"  Confounders: {res.get('n_confounders', 'N/A')}")
            lines.append("")

            # ATE estimates
            for method in ["linear_regression", "propensity_score", "ipw"]:
                method_res = res.get(method, {})
                ate = method_res.get("ate", "N/A")
                p_val = method_res.get("p_value", "N/A")
                err = method_res.get("error", None)
                if err:
                    lines.append(f"  {method}: ERROR - {err}")
                else:
                    line = (
                        f"  {method}: ATE = {ate:.6f}"
                        if isinstance(ate, float)
                        else f"  {method}: ATE = {ate}"
                    )
                    if isinstance(p_val, float) and not np.isnan(p_val):
                        line += f", p-value = {p_val:.4e}"
                    lines.append(line)

            # Refutations
            refs = res.get("refutations", {})
            if refs:
                lines.append("")
                lines.append("  Refutation Tests:")
                original_ate = res.get("linear_regression", {}).get("ate", np.nan)
                for ref_name, ref_data in refs.items():
                    new_effect = ref_data.get("new_effect", "N/A")
                    err = ref_data.get("error", None)
                    if err:
                        lines.append(f"    {ref_name}: ERROR - {err}")
                    else:
                        if (
                            isinstance(new_effect, float)
                            and isinstance(original_ate, float)
                            and original_ate != 0
                        ):
                            change = (
                                abs(new_effect - original_ate) / abs(original_ate) * 100
                            )
                            lines.append(
                                f"    {ref_name}: new_effect = {new_effect:.6f} (change = {change:.1f}%)"
                            )
                        else:
                            lines.append(f"    {ref_name}: new_effect = {new_effect}")
            lines.append("")

        # Summary
        lines.append("=" * 80)
        lines.append("SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        robust_causes = []
        for res in all_results:
            t = res["treatment"]
            ate_lr = res.get("linear_regression", {}).get("ate", np.nan)
            if np.isnan(ate_lr):
                continue
            # Check if refutations passed
            refs = res.get("refutations", {})
            placebo_ok = True
            for ref_name, ref_data in refs.items():
                if ref_name == "placebo_treatment":
                    new_e = ref_data.get("new_effect", np.nan)
                    if isinstance(new_e, float) and abs(new_e) < abs(ate_lr) * 0.5:
                        placebo_ok = True
                    elif isinstance(new_e, float):
                        placebo_ok = abs(new_e) < abs(ate_lr) * 0.5

            if placebo_ok and abs(ate_lr) > 0.001:
                robust_causes.append(t)

        lines.append(f"Robust causal factors: {robust_causes}")
        lines.append("")

        report = "\n".join(lines)
        out_path = self.output_dir / "causal_effect_report.txt"
        out_path.write_text(report)
        logger.info("Saved report to %s", out_path)
        return report

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self) -> List[Dict]:
        """Run causal effect estimation for all identified causes."""
        logger.info("=" * 60)
        logger.info("Starting causal effect estimation")
        logger.info("=" * 60)

        data, selected_features, causes_info = self.load_data()

        # Use all_causes (union of PC and FCI)
        treatments = causes_info.get("all_causes", [])
        if not treatments:
            logger.warning("No direct causes found. Using top features from selection.")
            treatments = selected_features[:5]

        # Also add a few key entropy features that may be interesting even if
        # not identified as direct causes (they might have indirect effects)
        extra_treatments = []
        key_entropy_features = [
            "sample_avg_entropy_per_token",
            "base_sample_avg_entropy_per_token",
            "answer_token_entropy_change",
            "sample_round_2_q3_agent_max_entropy",
        ]
        for f in key_entropy_features:
            if f in data.columns and f not in treatments:
                extra_treatments.append(f)

        all_treatments = treatments + extra_treatments
        logger.info("Treatments to analyze: %s", all_treatments)

        all_results = []
        for i, treatment in enumerate(all_treatments):
            logger.info(
                "[%d/%d] Estimating effect of %s...",
                i + 1,
                len(all_treatments),
                treatment,
            )
            result = self.estimate_effect(data, treatment, selected_features)
            result["is_direct_cause"] = treatment in treatments
            all_results.append(result)

        # Save raw results
        with open(self.output_dir / "causal_effects_raw.json", "w") as f:
            # Convert NaN to None for JSON
            def clean(obj):
                if isinstance(obj, float) and np.isnan(obj):
                    return None
                if isinstance(obj, dict):
                    return {k: clean(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [clean(v) for v in obj]
                return obj

            json.dump(clean(all_results), f, indent=2)

        # Visualizations
        self.plot_ate_comparison(all_results)
        self.plot_refutation_summary(all_results)

        # Report
        self.generate_report(all_results)

        logger.info("Causal effect estimation complete. Results in %s", self.output_dir)
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Causal Effect Estimation with DoWhy")
    parser.add_argument("--data-path", default="data_mining/data/merged_datasets.csv")
    parser.add_argument(
        "--feature-list",
        default="data_mining/results_causal_complete/feature_selection/selected_features.csv",
    )
    parser.add_argument(
        "--causes-path",
        default="data_mining/results_causal_complete/causal_discovery/direct_causes.json",
    )
    parser.add_argument(
        "--edges-path",
        default="data_mining/results_causal_complete/causal_discovery/all_edges.csv",
    )
    parser.add_argument(
        "--output-dir", default="data_mining/results_causal_complete/causal_effects"
    )
    parser.add_argument("--max-sample", type=int, default=15000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    estimator = CausalEffectEstimator(
        data_path=args.data_path,
        feature_list_path=args.feature_list,
        causes_path=args.causes_path,
        edges_path=args.edges_path,
        output_dir=args.output_dir,
        max_sample=args.max_sample,
    )
    results = estimator.run()
    print(f"\nAnalyzed {len(results)} treatment variables.")
    for r in results:
        ate = r.get("linear_regression", {}).get("ate", "N/A")
        print(f"  {r['treatment']}: LR ATE = {ate}")


if __name__ == "__main__":
    main()
