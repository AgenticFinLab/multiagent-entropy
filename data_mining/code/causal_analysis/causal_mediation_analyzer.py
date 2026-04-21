"""
Causal Mediation Analysis for MAS Entropy Features.

Analyzes mediation pathways:
1. architecture -> entropy features -> is_finally_correct
2. base_model_accuracy -> entropy change -> is_finally_correct
3. round_1_entropy -> round_2_entropy -> is_finally_correct

Uses Baron-Kenny method with Bootstrap confidence intervals.
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
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_data_from_path

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


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


class MediationPath:
    """Represents a single mediation path: Treatment -> Mediator -> Outcome."""

    def __init__(
        self, treatment: str, mediator: str, outcome: str, covariates: List[str] = None
    ):
        self.treatment = treatment
        self.mediator = mediator
        self.outcome = outcome
        self.covariates = covariates or []


class CausalMediationAnalyzer:
    """Analyze mediation effects using Baron-Kenny method with Bootstrap CI."""

    def __init__(
        self,
        data_path: str = "data_mining/data/merged_datasets.csv",
        feature_list_path: str = "data_mining/results_causal_complete/feature_selection/selected_features.csv",
        output_dir: str = "data_mining/results_causal_complete/mediation",
        n_bootstrap: int = 1000,
        max_sample: int = 20000,
    ):
        self.data_path = Path(data_path)
        self.feature_list_path = Path(feature_list_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_bootstrap = n_bootstrap
        self.max_sample = max_sample
        self.target = "is_finally_correct"

    def load_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load and prepare data."""
        feat_df = pd.read_csv(self.feature_list_path)
        selected = feat_df["feature"].tolist()

        df = load_data_from_path(self.data_path)

        # Also need architecture column for mediation path 1
        cols = [c for c in selected if c in df.columns] + [self.target]
        if "architecture" in df.columns:
            cols.append("architecture")
        cols = list(dict.fromkeys(cols))
        sub = df[cols].dropna()

        if len(sub) > self.max_sample:
            sub = sub.sample(n=self.max_sample, random_state=42)

        # Encode architecture as numeric for regression
        if "architecture" in sub.columns:
            # Binary: single(0) vs multi-agent(1)
            sub["is_multi_agent"] = (sub["architecture"] != "single").astype(int)

        logger.info("Loaded %d rows for mediation analysis", len(sub))
        return sub, selected

    def define_mediation_paths(
        self, data: pd.DataFrame, selected: List[str]
    ) -> List[MediationPath]:
        """Define the mediation paths to analyze based on available features."""
        paths = []

        # Helper to check if feature exists in data
        def has(f):
            return f in data.columns

        # ---- Path Type 1: Architecture -> Entropy -> Correctness ----
        if has("is_multi_agent"):
            entropy_mediators = [
                f
                for f in selected
                if "entropy" in f
                and f != self.target
                and has(f)
                and "base_" not in f  # Exclude base model features
            ]
            # Pick top mediator candidates
            for m in entropy_mediators[:5]:
                covariates = [f for f in ["base_model_accuracy"] if has(f)]
                paths.append(
                    MediationPath("is_multi_agent", m, self.target, covariates)
                )

        # ---- Path Type 2: Base Model Accuracy -> Entropy Change -> Correctness ----
        if has("base_model_accuracy"):
            entropy_change_mediators = [
                f
                for f in selected
                if has(f)
                and f != self.target
                and f != "base_model_accuracy"
                and any(kw in f for kw in ["change", "diff", "reduction", "ratio"])
            ]
            for m in entropy_change_mediators[:3]:
                paths.append(MediationPath("base_model_accuracy", m, self.target, []))

        # ---- Path Type 3: Round 1 Entropy -> Round 2 Entropy -> Correctness ----
        r1_features = [
            f for f in selected if "round_1" in f and "entropy" in f and has(f)
        ]
        r2_features = [
            f for f in selected if "round_2" in f and "entropy" in f and has(f)
        ]

        for r1 in r1_features[:3]:
            for r2 in r2_features[:2]:
                covariates = [f for f in ["base_model_accuracy"] if has(f)]
                paths.append(MediationPath(r1, r2, self.target, covariates))

        # ---- Path Type 4: Base Entropy -> Sample Entropy -> Correctness ----
        if has("base_sample_avg_entropy_per_token"):
            sample_entropy_mediators = [
                "sample_avg_entropy_per_token",
                "sample_mean_answer_token_entropy",
                "sample_max_entropy",
            ]
            for m in sample_entropy_mediators:
                if has(m):
                    paths.append(
                        MediationPath(
                            "base_sample_avg_entropy_per_token", m, self.target, []
                        )
                    )

        logger.info("Defined %d mediation paths to analyze", len(paths))
        return paths

    def baron_kenny_test(
        self,
        data: pd.DataFrame,
        treatment: str,
        mediator: str,
        outcome: str,
        covariates: List[str],
    ) -> Dict:
        """Run Baron-Kenny mediation test.

        Step 1: Total effect  (Treatment -> Outcome)
        Step 2: Path a        (Treatment -> Mediator)
        Step 3: Path b + c'   (Treatment + Mediator -> Outcome)
        Indirect effect = a * b
        Direct effect = c'
        """
        result = {
            "treatment": treatment,
            "mediator": mediator,
            "outcome": outcome,
            "n_covariates": len(covariates),
        }

        try:
            # Standardize continuous variables for comparable coefficients
            scaler = StandardScaler()
            continuous_cols = [treatment, mediator] + covariates
            continuous_cols = [c for c in continuous_cols if c in data.columns]
            data_std = data.copy()
            data_std[continuous_cols] = scaler.fit_transform(data_std[continuous_cols])

            X_base = (
                data_std[covariates].values
                if covariates
                else np.zeros((len(data_std), 0))
            )
            T = data_std[treatment].values.reshape(-1, 1)
            M = data_std[mediator].values.reshape(-1, 1)
            Y = data_std[outcome].values

            # Step 1: Total effect (c): T -> Y
            X_total = np.hstack([T, X_base]) if X_base.shape[1] > 0 else T
            model_total = LogisticRegression(max_iter=1000, solver="lbfgs")
            model_total.fit(X_total, Y)
            c_total = model_total.coef_[0][0]

            # Step 2: Path a: T -> M
            X_a = np.hstack([T, X_base]) if X_base.shape[1] > 0 else T
            model_a = LinearRegression()
            model_a.fit(X_a, M.ravel())
            a_coef = model_a.coef_[0]

            # Step 3: Path b and c': T + M -> Y
            X_full = (
                np.hstack([T, M, X_base]) if X_base.shape[1] > 0 else np.hstack([T, M])
            )
            model_full = LogisticRegression(max_iter=1000, solver="lbfgs")
            model_full.fit(X_full, Y)
            c_prime = model_full.coef_[0][0]  # direct effect
            b_coef = model_full.coef_[0][1]  # mediator effect

            # Indirect effect = a * b
            indirect = a_coef * b_coef
            # Proportion mediated
            total_via_coefs = c_prime + indirect
            prop_mediated = (
                indirect / total_via_coefs if abs(total_via_coefs) > 1e-10 else 0.0
            )

            result.update(
                {
                    "total_effect_c": float(c_total),
                    "path_a": float(a_coef),
                    "path_b": float(b_coef),
                    "direct_effect_cprime": float(c_prime),
                    "indirect_effect_ab": float(indirect),
                    "proportion_mediated": float(prop_mediated),
                }
            )

            # Bootstrap CI for indirect effect
            boot_indirect = self._bootstrap_indirect(
                data_std, treatment, mediator, outcome, covariates
            )
            if boot_indirect is not None:
                ci_lower = np.percentile(boot_indirect, 2.5)
                ci_upper = np.percentile(boot_indirect, 97.5)
                # Check if CI excludes zero
                significant = (ci_lower > 0 and ci_upper > 0) or (
                    ci_lower < 0 and ci_upper < 0
                )
                result.update(
                    {
                        "indirect_ci_lower": float(ci_lower),
                        "indirect_ci_upper": float(ci_upper),
                        "indirect_significant": bool(significant),
                        "boot_indirect_mean": float(np.mean(boot_indirect)),
                        "boot_indirect_std": float(np.std(boot_indirect)),
                    }
                )

            # Sobel test
            se_a = np.sqrt(
                np.sum((M.ravel() - model_a.predict(X_a)) ** 2) / (len(M) - 2)
            )
            se_b_approx = abs(b_coef) * 0.1  # approximate SE
            sobel_se = np.sqrt(a_coef**2 * se_b_approx**2 + b_coef**2 * se_a**2)
            if sobel_se > 0:
                sobel_z = indirect / sobel_se
                sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
                result.update(
                    {
                        "sobel_z": float(sobel_z),
                        "sobel_p": float(sobel_p),
                    }
                )

        except Exception as e:
            logger.warning(
                "Mediation test failed for %s -> %s -> %s: %s",
                treatment,
                mediator,
                outcome,
                e,
            )
            result["error"] = str(e)

        return result

    def _bootstrap_indirect(
        self,
        data: pd.DataFrame,
        treatment: str,
        mediator: str,
        outcome: str,
        covariates: List[str],
    ) -> Optional[np.ndarray]:
        """Bootstrap the indirect effect (a*b)."""
        n = len(data)
        indirect_effects = []

        X_base_cols = covariates if covariates else []

        try:
            for _ in range(self.n_bootstrap):
                idx = np.random.choice(n, size=n, replace=True)
                boot = data.iloc[idx]

                T = boot[treatment].values.reshape(-1, 1)
                M = boot[mediator].values.reshape(-1, 1)
                Y = boot[outcome].values
                X_base = (
                    boot[X_base_cols].values
                    if X_base_cols
                    else np.zeros((len(boot), 0))
                )

                # Path a
                X_a = np.hstack([T, X_base]) if X_base.shape[1] > 0 else T
                model_a = LinearRegression()
                model_a.fit(X_a, M.ravel())
                a = model_a.coef_[0]

                # Path b
                X_full = (
                    np.hstack([T, M, X_base])
                    if X_base.shape[1] > 0
                    else np.hstack([T, M])
                )
                model_full = LogisticRegression(max_iter=500, solver="lbfgs")
                model_full.fit(X_full, Y)
                b = model_full.coef_[0][1]

                indirect_effects.append(a * b)

            return np.array(indirect_effects)
        except Exception as e:
            logger.warning("Bootstrap failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot_mediation_effects(self, results: List[Dict]):
        """Bar chart of indirect effects with CI."""
        sig_results = [
            r for r in results if "indirect_effect_ab" in r and "error" not in r
        ]
        if not sig_results:
            return

        # Sort by absolute indirect effect
        sig_results.sort(key=lambda r: abs(r["indirect_effect_ab"]), reverse=True)
        top = sig_results[:20]

        labels = [f"{r['treatment'][:20]}\n-> {r['mediator'][:20]}" for r in top]
        effects = [r["indirect_effect_ab"] for r in top]
        ci_lower = [r.get("indirect_ci_lower", r["indirect_effect_ab"]) for r in top]
        ci_upper = [r.get("indirect_ci_upper", r["indirect_effect_ab"]) for r in top]
        significant = [r.get("indirect_significant", False) for r in top]

        setup_plot_style()
        with sns.plotting_context("paper", font_scale=1.4):
            fig, ax = plt.subplots(figsize=(14, max(8, len(top) * 0.6)))
            colors = ["#D73027" if s else "#D3D3D3" for s in significant]
            y_pos = range(len(top))

            ax.barh(
                y_pos,
                effects,
                color=colors,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            )

            # Error bars
            xerr_lower = [e - l for e, l in zip(effects, ci_lower)]
            xerr_upper = [u - e for e, u in zip(effects, ci_upper)]
            ax.errorbar(
                effects,
                y_pos,
                xerr=[xerr_lower, xerr_upper],
                fmt="none",
                ecolor="black",
                capsize=4,
                linewidth=1.5,
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=12)
            ax.invert_yaxis()
            ax.axvline(x=0, color="black", linestyle="--", alpha=0.3)
            ax.set_xlabel("Indirect Effect (a × b)", fontsize=18)
            # ax.set_title(
            #     "Mediation Effects (red = significant, gray = non-significant)",
            #     fontsize=20,
            #     fontweight="bold",
            #     pad=15,
            # )
            ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            sns.despine(ax=ax, top=True, right=True)

            plt.tight_layout()
            out_path = self.output_dir / "mediation_effects.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved mediation effects to %s", out_path)

    def plot_path_diagram(self, results: List[Dict]):
        """Plot top significant mediation path diagrams."""
        sig = [
            r
            for r in results
            if r.get("indirect_significant", False) and "error" not in r
        ]
        sig.sort(key=lambda r: abs(r["indirect_effect_ab"]), reverse=True)
        top = sig[:6]

        if not top:
            logger.warning("No significant mediation paths to plot")
            return

        setup_plot_style()
        n_paths = len(top)
        with sns.plotting_context("paper", font_scale=1.4):
            fig, axes = plt.subplots(2, 3, figsize=(24, 14))
            axes = axes.flatten()

            for idx, (r, ax) in enumerate(zip(top, axes)):
                self._draw_single_path(ax, r)

            # Hide unused axes
            for idx in range(n_paths, len(axes)):
                axes[idx].set_visible(False)

            # fig.suptitle(
            #     "Top Significant Mediation Paths",
            #     fontsize=22,
            #     fontweight="bold",
            #     y=1.02,
            # )
            plt.tight_layout()
            out_path = self.output_dir / "path_diagrams.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved path diagrams to %s", out_path)

    def _draw_single_path(self, ax, result: Dict):
        """Draw a single mediation path diagram."""
        t_name = result["treatment"][:30]
        m_name = result["mediator"][:30]
        o_name = "is_correct"

        a = result.get("path_a", 0)
        b = result.get("path_b", 0)
        c = result.get("total_effect_c", 0)
        cprime = result.get("direct_effect_cprime", 0)
        ab = result.get("indirect_effect_ab", 0)
        prop = result.get("proportion_mediated", 0)

        # Positions
        pos_t = (0.1, 0.3)
        pos_m = (0.5, 0.85)
        pos_o = (0.9, 0.3)

        # Draw nodes
        for pos, label in [(pos_t, t_name), (pos_m, m_name), (pos_o, o_name)]:
            bbox = dict(
                boxstyle="round,pad=0.5", facecolor="#4575B4", alpha=0.85, linewidth=2
            )
            ax.text(
                pos[0],
                pos[1],
                label,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white",
                bbox=bbox,
                transform=ax.transAxes,
            )

        # Draw arrows with coefficients
        # T -> M (path a)
        ax.annotate(
            "",
            xy=(0.42, 0.82),
            xytext=(0.18, 0.42),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#D73027", lw=2.5),
        )
        ax.text(
            0.22,
            0.65,
            f"a={a:.3f}",
            fontsize=15,
            color="#D73027",
            fontweight="bold",
            transform=ax.transAxes,
        )

        # M -> O (path b)
        ax.annotate(
            "",
            xy=(0.82, 0.42),
            xytext=(0.58, 0.82),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#D73027", lw=2.5),
        )
        ax.text(
            0.72,
            0.65,
            f"b={b:.3f}",
            fontsize=15,
            color="#D73027",
            fontweight="bold",
            transform=ax.transAxes,
        )

        # T -> O (direct, path c')
        ax.annotate(
            "",
            xy=(0.82, 0.3),
            xytext=(0.18, 0.3),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="->", color="#2c3e50", lw=1.5, linestyle="dashed"
            ),
        )
        ax.text(
            0.5,
            0.15,
            f"c'={cprime:.3f}",
            fontsize=15,
            color="#2c3e50",
            ha="center",
            transform=ax.transAxes,
        )

        # Summary text
        sig_mark = "*" if result.get("indirect_significant", False) else ""
        ax.text(
            0.5,
            0.02,
            f"Indirect(a×b)={ab:.4f}{sig_mark} | Prop={prop:.1%}",
            fontsize=14,
            ha="center",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def plot_effect_decomposition(self, results: List[Dict]):
        """Stacked bar chart showing total effect decomposition."""
        sig = [r for r in results if "error" not in r and "direct_effect_cprime" in r]
        if not sig:
            return

        # Group by treatment
        treatment_groups = {}
        for r in sig:
            t = r["treatment"]
            treatment_groups.setdefault(t, []).append(r)

        setup_plot_style()
        with sns.plotting_context("paper", font_scale=1.4):
            fig, ax = plt.subplots(figsize=(16, max(8, len(treatment_groups) * 1.8)))
        y_labels = []
        y_pos = 0
        bar_data = []

        for treatment, paths in treatment_groups.items():
            for r in sorted(
                paths, key=lambda x: abs(x.get("indirect_effect_ab", 0)), reverse=True
            )[:3]:
                label = f"{treatment[:20]} via {r['mediator'][:20]}"
                direct = r.get("direct_effect_cprime", 0)
                indirect = r.get("indirect_effect_ab", 0)
                bar_data.append(
                    {"label": label, "direct": direct, "indirect": indirect, "y": y_pos}
                )
                y_labels.append(label)
                y_pos += 1

        if not bar_data:
            return

        positions = [d["y"] for d in bar_data]
        directs = [d["direct"] for d in bar_data]
        indirects = [d["indirect"] for d in bar_data]

        ax.barh(
            positions,
            directs,
            color="#4575B4",
            alpha=0.9,
            label="Direct Effect (c')",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.barh(
            positions,
            indirects,
            left=directs,
            color="#D73027",
            alpha=0.9,
            label="Indirect Effect (a×b)",
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_yticks(positions)
        ax.set_yticklabels(y_labels, fontsize=12)
        ax.invert_yaxis()
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("Effect Size (standardized)", fontsize=18)
        # ax.set_title(
        #     "Effect Decomposition: Direct vs Indirect",
        #     fontsize=20,
        #     fontweight="bold",
        #     pad=15,
        # )
        ax.legend(loc="lower right", fontsize=16, frameon=False)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
        plt.tight_layout()
        out_path = self.output_dir / "effect_decomposition.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved effect decomposition to %s", out_path)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_report(self, results: List[Dict]) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("CAUSAL MEDIATION ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(
            f"Method: Baron-Kenny with Bootstrap CI (n_bootstrap={self.n_bootstrap})"
        )
        lines.append(f"Total paths analyzed: {len(results)}")
        sig_count = sum(1 for r in results if r.get("indirect_significant", False))
        lines.append(f"Significant mediation paths: {sig_count}")
        lines.append("")

        # Group by path type
        path_types = {
            "Architecture -> Entropy -> Correctness": [],
            "Base Accuracy -> Entropy Change -> Correctness": [],
            "Round 1 Entropy -> Round 2 Entropy -> Correctness": [],
            "Base Entropy -> Sample Entropy -> Correctness": [],
        }

        for r in results:
            t = r.get("treatment", "")
            if t == "is_multi_agent":
                path_types["Architecture -> Entropy -> Correctness"].append(r)
            elif t == "base_model_accuracy":
                path_types["Base Accuracy -> Entropy Change -> Correctness"].append(r)
            elif "round_1" in t:
                path_types["Round 1 Entropy -> Round 2 Entropy -> Correctness"].append(
                    r
                )
            elif "base_" in t:
                path_types["Base Entropy -> Sample Entropy -> Correctness"].append(r)
            else:
                path_types.setdefault("Other", []).append(r)

        for path_type, paths in path_types.items():
            if not paths:
                continue
            lines.append("-" * 80)
            lines.append(f"PATH TYPE: {path_type}")
            lines.append("-" * 80)

            for r in sorted(
                paths, key=lambda x: abs(x.get("indirect_effect_ab", 0)), reverse=True
            ):
                if "error" in r:
                    lines.append(
                        f"  {r['treatment']} -> {r['mediator']}: ERROR - {r['error']}"
                    )
                    continue

                sig = "*" if r.get("indirect_significant", False) else ""
                lines.append(
                    f"  {r['treatment']} -> {r['mediator']} -> {r['outcome']} {sig}"
                )
                lines.append(
                    f"    Total effect (c):     {r.get('total_effect_c', 'N/A'):.4f}"
                )
                lines.append(f"    Path a (T->M):        {r.get('path_a', 'N/A'):.4f}")
                lines.append(f"    Path b (M->Y):        {r.get('path_b', 'N/A'):.4f}")
                lines.append(
                    f"    Direct effect (c'):   {r.get('direct_effect_cprime', 'N/A'):.4f}"
                )
                lines.append(
                    f"    Indirect effect (ab): {r.get('indirect_effect_ab', 'N/A'):.4f}"
                )
                ci_l = r.get("indirect_ci_lower", "N/A")
                ci_u = r.get("indirect_ci_upper", "N/A")
                if isinstance(ci_l, float):
                    lines.append(f"    Bootstrap 95% CI:     [{ci_l:.4f}, {ci_u:.4f}]")
                lines.append(
                    f"    Proportion mediated:  {r.get('proportion_mediated', 0):.1%}"
                )
                if "sobel_z" in r:
                    lines.append(
                        f"    Sobel test:           z={r['sobel_z']:.3f}, p={r['sobel_p']:.4e}"
                    )
                lines.append("")

        # Key findings
        lines.append("=" * 80)
        lines.append("KEY FINDINGS")
        lines.append("=" * 80)
        lines.append("")

        sig_paths = [r for r in results if r.get("indirect_significant", False)]
        if sig_paths:
            sig_paths.sort(
                key=lambda r: abs(r.get("indirect_effect_ab", 0)), reverse=True
            )
            lines.append(f"Top significant mediation paths (by |indirect effect|):")
            for r in sig_paths[:10]:
                lines.append(
                    f"  {r['treatment'][:30]} -> {r['mediator'][:30]}: "
                    f"indirect={r['indirect_effect_ab']:.4f}, "
                    f"prop_mediated={r.get('proportion_mediated', 0):.1%}"
                )
        else:
            lines.append(
                "No statistically significant mediation paths found at 95% CI."
            )

        lines.append("")
        report = "\n".join(lines)
        out_path = self.output_dir / "mediation_report.txt"
        out_path.write_text(report)
        logger.info("Saved report to %s", out_path)
        return report

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self) -> List[Dict]:
        logger.info("=" * 60)
        logger.info("Starting causal mediation analysis")
        logger.info("=" * 60)

        data, selected = self.load_data()
        paths = self.define_mediation_paths(data, selected)

        all_results = []
        for i, path in enumerate(paths):
            logger.info(
                "[%d/%d] Analyzing: %s -> %s -> %s",
                i + 1,
                len(paths),
                path.treatment,
                path.mediator,
                path.outcome,
            )
            result = self.baron_kenny_test(
                data, path.treatment, path.mediator, path.outcome, path.covariates
            )
            all_results.append(result)

        # Save raw results
        with open(self.output_dir / "mediation_results.json", "w") as f:

            def clean(obj):
                if isinstance(obj, (np.floating, float)) and np.isnan(obj):
                    return None
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: clean(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [clean(v) for v in obj]
                return obj

            json.dump(clean(all_results), f, indent=2)

        # Visualizations
        self.plot_mediation_effects(all_results)
        self.plot_path_diagram(all_results)
        self.plot_effect_decomposition(all_results)

        # Report
        self.generate_report(all_results)

        logger.info("Mediation analysis complete. Results in %s", self.output_dir)
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Causal Mediation Analysis")
    parser.add_argument("--data-path", default="data_mining/data/merged_datasets.csv")
    parser.add_argument(
        "--feature-list",
        default="data_mining/results_causal_complete/feature_selection/selected_features.csv",
    )
    parser.add_argument(
        "--output-dir", default="data_mining/results_causal_complete/mediation"
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--max-sample", type=int, default=20000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    analyzer = CausalMediationAnalyzer(
        data_path=args.data_path,
        feature_list_path=args.feature_list,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        max_sample=args.max_sample,
    )
    results = analyzer.run()

    sig = [r for r in results if r.get("indirect_significant", False)]
    print(f"\nAnalyzed {len(results)} mediation paths, {len(sig)} significant.")
    for r in sig[:10]:
        print(
            f"  {r['treatment']} -> {r['mediator']}: indirect={r['indirect_effect_ab']:.4f}"
        )


if __name__ == "__main__":
    main()
