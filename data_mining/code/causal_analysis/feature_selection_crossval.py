"""
Multi-method Cross-validation Feature Selection.

Combines rankings from RFE, Tree/LR importance, and statistical tests (Chi2, MI, F-classif)
using Borda Count weighted fusion, then applies correlation-based redundancy removal
and semantic layer constraints to produce an optimal feature subset for causal inference.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Allow running from repo root or from code/
import sys

# causal_analysis/ is inside code/, so add parent (code/) to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import (
    BASE_MODEL_METRICS_EXPERIMENT_LEVEL,
    BASE_MODEL_METRICS_SAMPLE_LEVEL,
    SAMPLE_BASELINE_ENTROPY,
    EXPERIMENT_STATISTICS,
    ROUND_STATISTICS,
    SAMPLE_STATISTICS,
    SAMPLE_DISTRIBUTION_SHAPE,
    AGGREGATION_OVER_AGENTS,
    SAMPLE_ROUND_WISE_AGGREGATED,
    CROSS_ROUND_AGGREGATED,
    INTRA_ROUND_AGENT_DISTRIBUTION,
    CROSS_ROUND_AGENT_SPREAD_CHANGE,
    SAMPLE_ROUND1_AGENT_STATISTICS,
    SAMPLE_ROUND2_AGENT_STATISTICS,
    DEFAULT_EXCLUDE_COLUMNS,
)
from utils import load_data_from_path, prepare_features

logger = logging.getLogger(__name__)


# --- ICML-style plot settings (ref: analyze_accuracy.py) ---
def setup_plot_style():
    """Configure matplotlib/seaborn for ICML-quality figures."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["legend.title_fontsize"] = 18
    plt.rcParams["legend.fontsize"] = 16


# Features that are NOT entropy-related and must be excluded from selection
NON_ENTROPY_FEATURES = [
    "base_model_accuracy",
    "base_model_format_compliance_rate",
    "base_model_is_finally_correct",
    "base_model_format_compliance",
    "round_1_total_time",
    "round_2_total_time",
    "round_1_num_inferences",
    "round_2_num_inferences",
    "exp_total_time",
    "exp_total_token",
    "round_1_total_token",
    "round_2_total_token",
    "round_1_2_change_tokens",
    "sample_answer_token_count",
    "base_model_answer_token_count",
    "base_sample_token_count",
    "sample_all_agents_token_count",
    "sample_num_agents",
    "sample_round_1_all_agents_total_token",
    "sample_round_2_all_agents_total_token",
    "sample_round_all_agents_total_token_first_last_diff",
    "sample_round_all_agents_total_token_first_last_ratio",
    "sample_round_1_2_change_tokens",
    "exp_total_token",
]

# Semantic feature groups for layer constraints (entropy-related only)
SEMANTIC_GROUPS: Dict[str, List[str]] = {
    "baseline_entropy": SAMPLE_BASELINE_ENTROPY,
    "experiment_statistics": EXPERIMENT_STATISTICS,
    "round_statistics": ROUND_STATISTICS,
    "sample_statistics": SAMPLE_STATISTICS,
    "sample_distribution_shape": SAMPLE_DISTRIBUTION_SHAPE,
    "aggregation_over_agents": AGGREGATION_OVER_AGENTS,
    "round_wise_aggregated": SAMPLE_ROUND_WISE_AGGREGATED,
    "cross_round_aggregated": CROSS_ROUND_AGGREGATED,
    "intra_round_agent_distribution": INTRA_ROUND_AGENT_DISTRIBUTION,
    "cross_round_agent_spread_change": CROSS_ROUND_AGENT_SPREAD_CHANGE,
    "round1_agent_statistics": SAMPLE_ROUND1_AGENT_STATISTICS,
    "round2_agent_statistics": SAMPLE_ROUND2_AGENT_STATISTICS,
}


class FeatureSelectionCrossValidator:
    """Cross-validate feature importance across multiple methods and select optimal subset."""

    def __init__(
        self,
        data_path: str = "data_mining/data/merged_datasets.csv",
        results_dir: str = "data_mining/results_ablation/feature_ablation",
        output_dir: str = "data_mining/results_causal_complete/feature_selection",
        corr_threshold: float = 0.85,
        max_features: int = 30,
        min_features: int = 15,
    ):
        self.data_path = Path(data_path)
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.corr_threshold = corr_threshold
        self.max_features = max_features
        self.min_features = min_features

        # Method weights for Borda fusion (higher = more trusted)
        self.method_weights = {
            "combined": 3.0,  # Tree + LR combined ranking
            "chi2": 1.0,
            "mutual_info": 1.5,
            "f_classif": 1.0,
        }

    # ------------------------------------------------------------------
    # Loading rankings
    # ------------------------------------------------------------------
    def load_combined_ranking(self) -> pd.DataFrame:
        """Load the combined (Tree + LR) feature ranking."""
        path = self.results_dir / "feature_rankings_combined.csv"
        df = pd.read_csv(path)
        logger.info("Loaded combined ranking: %d features", len(df))
        return df

    def load_statistical_ranking(self, method: str) -> pd.DataFrame:
        """Load statistical selection ranking (chi2 / mutual_info / f_classif)."""
        path = self.results_dir / f"statistical_selection_{method}.csv"
        df = pd.read_csv(path)
        # Add rank column (already sorted by score descending)
        df["Rank"] = range(1, len(df) + 1)
        logger.info("Loaded %s ranking: %d features", method, len(df))
        return df

    def load_all_rankings(self) -> Dict[str, pd.Series]:
        """Load all rankings and return {method_name: Series(feature -> rank)}."""
        rankings: Dict[str, pd.Series] = {}

        # Combined ranking
        combined = self.load_combined_ranking()
        rankings["combined"] = combined.set_index("Feature")["Combined_Rank"]

        # Statistical rankings
        for method in ["chi2", "mutual_info", "f_classif"]:
            stat_df = self.load_statistical_ranking(method)
            rankings[method] = stat_df.set_index("Feature")["Rank"]

        return rankings

    # ------------------------------------------------------------------
    # Borda count fusion
    # ------------------------------------------------------------------
    def compute_borda_scores(self, rankings: Dict[str, pd.Series]) -> pd.DataFrame:
        """Compute weighted Borda scores across all methods.

        Uses reciprocal rank weighting: score = weight * (1 / rank).
        """
        all_features = set()
        for r in rankings.values():
            all_features.update(r.index)
        all_features = sorted(all_features)

        n_features = len(all_features)
        scores = pd.DataFrame(index=all_features)

        for method, rank_series in rankings.items():
            weight = self.method_weights.get(method, 1.0)
            # Reciprocal rank score
            method_scores = []
            for feat in all_features:
                if feat in rank_series.index:
                    rank = rank_series[feat]
                    method_scores.append(weight * (1.0 / rank))
                else:
                    # Feature not ranked by this method -> lowest score
                    method_scores.append(weight * (1.0 / (n_features + 1)))
            scores[f"score_{method}"] = method_scores

        scores["total_score"] = scores.sum(axis=1)
        scores = scores.sort_values("total_score", ascending=False)
        scores["borda_rank"] = range(1, len(scores) + 1)

        logger.info("Computed Borda scores for %d features", len(scores))
        return scores

    # ------------------------------------------------------------------
    # Correlation-based redundancy removal
    # ------------------------------------------------------------------
    def remove_redundant_features(
        self,
        candidate_features: List[str],
        borda_scores: pd.DataFrame,
        data: pd.DataFrame,
    ) -> List[str]:
        """Remove highly correlated features, keeping the one with higher Borda score."""
        # Compute Spearman correlation matrix
        feat_data = data[candidate_features].copy()
        corr_matrix = feat_data.corr(method="spearman").abs()

        # Identify pairs above threshold
        removed = set()
        n = len(candidate_features)
        for i in range(n):
            if candidate_features[i] in removed:
                continue
            for j in range(i + 1, n):
                if candidate_features[j] in removed:
                    continue
                if corr_matrix.iloc[i, j] > self.corr_threshold:
                    fi, fj = candidate_features[i], candidate_features[j]
                    si = (
                        borda_scores.loc[fi, "total_score"]
                        if fi in borda_scores.index
                        else 0
                    )
                    sj = (
                        borda_scores.loc[fj, "total_score"]
                        if fj in borda_scores.index
                        else 0
                    )
                    drop = fj if si >= sj else fi
                    removed.add(drop)
                    logger.debug(
                        "Removed %s (corr=%.3f with %s)",
                        drop,
                        corr_matrix.iloc[i, j],
                        fi if drop == fj else fj,
                    )

        selected = [f for f in candidate_features if f not in removed]
        logger.info(
            "Correlation filtering: %d -> %d features (removed %d)",
            len(candidate_features),
            len(selected),
            len(removed),
        )
        return selected

    # ------------------------------------------------------------------
    # Semantic layer constraints
    # ------------------------------------------------------------------
    def get_feature_group(self, feature: str) -> Optional[str]:
        """Return the semantic group name for a feature, or None."""
        for group_name, features in SEMANTIC_GROUPS.items():
            if feature in features:
                return group_name
        return None

    def apply_layer_constraints(
        self,
        selected_features: List[str],
        borda_scores: pd.DataFrame,
        all_numeric_features: List[str],
    ) -> List[str]:
        """Ensure each major semantic group is represented; add top feature from
        under-represented groups if possible."""
        # Check current coverage
        group_coverage: Dict[str, List[str]] = {}
        for feat in selected_features:
            grp = self.get_feature_group(feat)
            if grp:
                group_coverage.setdefault(grp, []).append(feat)

        covered_groups = set(group_coverage.keys())
        all_groups = set(SEMANTIC_GROUPS.keys())
        missing_groups = all_groups - covered_groups

        additions = []
        for grp in sorted(missing_groups):
            group_feats = SEMANTIC_GROUPS[grp]
            # Find the best-scoring feature in this group that exists in data
            candidates = [
                f
                for f in group_feats
                if f in borda_scores.index
                and f in all_numeric_features
                and f not in NON_ENTROPY_FEATURES
            ]
            if candidates:
                best = max(candidates, key=lambda f: borda_scores.loc[f, "total_score"])
                if best not in selected_features:
                    additions.append(best)
                    logger.info("Layer constraint: added %s for group '%s'", best, grp)

        final = selected_features + additions
        logger.info(
            "Layer constraints: %d -> %d features (+%d from missing groups)",
            len(selected_features),
            len(final),
            len(additions),
        )
        return final

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot_method_ranking_comparison(
        self, rankings: Dict[str, pd.Series], top_n: int = 30
    ):
        """Heatmap comparing feature ranks across methods."""
        # Get union of top-N from each method
        top_features = set()
        for method, rank_series in rankings.items():
            top_features.update(rank_series.nsmallest(top_n).index)
        top_features = sorted(top_features)

        # Build rank matrix
        rank_matrix = pd.DataFrame(index=top_features, columns=list(rankings.keys()))
        for method, rank_series in rankings.items():
            for feat in top_features:
                rank_matrix.loc[feat, method] = (
                    rank_series[feat] if feat in rank_series.index else len(rank_series)
                )
        rank_matrix = rank_matrix.astype(float)

        # Sort by mean rank
        rank_matrix["mean_rank"] = rank_matrix.mean(axis=1)
        rank_matrix = rank_matrix.sort_values("mean_rank")
        rank_matrix = rank_matrix.drop(columns=["mean_rank"])

        setup_plot_style()
        with sns.plotting_context("paper", font_scale=1.4):
            fig, ax = plt.subplots(figsize=(12, max(10, len(rank_matrix) * 0.35)))
            sns.heatmap(
                rank_matrix.head(40),
                annot=True,
                fmt=".0f",
                cmap="YlOrRd_r",
                linewidths=0.5,
                ax=ax,
                annot_kws={"fontsize": 12},
                cbar_kws={"label": "Rank (lower = more important)"},
            )
            # ax.set_title(
            #     "Feature Ranking Comparison Across Methods",
            #     fontsize=20,
            #     fontweight="bold",
            #     pad=15,
            # )
            ax.set_xlabel("Method", fontsize=18)
            ax.set_ylabel("Feature", fontsize=18)
            ax.tick_params(axis="y", labelsize=12)
            ax.tick_params(axis="x", labelsize=14)
            plt.tight_layout()
            out_path = self.output_dir / "ranking_comparison_heatmap.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved ranking comparison heatmap to %s", out_path)

    def plot_correlation_matrix(self, features: List[str], data: pd.DataFrame):
        """Plot Spearman correlation matrix for selected features."""
        setup_plot_style()
        corr = data[features].corr(method="spearman")
        with sns.plotting_context("paper", font_scale=1.4):
            fig, ax = plt.subplots(
                figsize=(max(12, len(features) * 0.55), max(10, len(features) * 0.45))
            )
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            sns.heatmap(
                corr,
                mask=mask,
                annot=len(features) <= 25,
                fmt=".2f" if len(features) <= 25 else "",
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                ax=ax,
                annot_kws={"fontsize": 10},
                cbar_kws={"label": "Spearman Correlation"},
            )
            # ax.set_title(
            #     "Selected Features Correlation Matrix",
            #     fontsize=20,
            #     fontweight="bold",
            #     pad=15,
            # )
            ax.tick_params(axis="both", labelsize=11)
            plt.tight_layout()
            out_path = self.output_dir / "selected_features_correlation.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved correlation matrix to %s", out_path)

    def plot_borda_scores(self, borda_scores: pd.DataFrame, top_n: int = 40):
        """Bar chart of top-N Borda scores with method breakdown."""
        setup_plot_style()
        top = borda_scores.head(top_n).copy()
        score_cols = [c for c in top.columns if c.startswith("score_")]

        color_map = ["#D73027", "#FC8D59", "#FEE090", "#4575B4"]
        with sns.plotting_context("paper", font_scale=1.4):
            fig, ax = plt.subplots(figsize=(14, max(10, top_n * 0.35)))
            bottom = np.zeros(len(top))

            for idx, col in enumerate(score_cols):
                method_name = col.replace("score_", "")
                ax.barh(
                    range(len(top)),
                    top[col].values,
                    left=bottom,
                    label=method_name,
                    color=color_map[idx % len(color_map)],
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottom += top[col].values

            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.index, fontsize=12)
            ax.invert_yaxis()
            ax.set_xlabel("Weighted Reciprocal Rank Score", fontsize=18)
            # ax.set_title(
            #     "Top Features by Borda Count Fusion",
            #     fontsize=20,
            #     fontweight="bold",
            #     pad=15,
            # )
            ax.legend(loc="lower right", fontsize=16, frameon=False)
            ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            sns.despine(ax=ax, top=True, right=True)
            plt.tight_layout()
            out_path = self.output_dir / "borda_scores_breakdown.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved Borda scores chart to %s", out_path)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def generate_report(
        self,
        borda_scores: pd.DataFrame,
        selected_features: List[str],
        rankings: Dict[str, pd.Series],
    ):
        """Generate text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("MULTI-METHOD CROSS-VALIDATION FEATURE SELECTION REPORT")
        lines.append("=" * 80)
        lines.append("")

        lines.append("METHODS USED")
        lines.append("-" * 80)
        for method, weight in self.method_weights.items():
            lines.append(f"  {method}: weight = {weight}")
        lines.append(f"  Fusion: Weighted Reciprocal Rank (Borda Count)")
        lines.append(f"  Correlation threshold: {self.corr_threshold}")
        lines.append("")

        lines.append("BORDA FUSION TOP 30")
        lines.append("-" * 80)
        for i, (feat, row) in enumerate(borda_scores.head(30).iterrows()):
            lines.append(f"  {i+1:3d}. {feat}: total_score = {row['total_score']:.4f}")
        lines.append("")

        # Group coverage
        lines.append("SEMANTIC GROUP COVERAGE")
        lines.append("-" * 80)
        group_counts: Dict[str, List[str]] = {}
        ungrouped = []
        for feat in selected_features:
            grp = self.get_feature_group(feat)
            if grp:
                group_counts.setdefault(grp, []).append(feat)
            else:
                ungrouped.append(feat)
        for grp in sorted(SEMANTIC_GROUPS.keys()):
            feats = group_counts.get(grp, [])
            lines.append(f"  {grp}: {len(feats)} feature(s)")
            for f in feats:
                lines.append(f"    - {f}")
        if ungrouped:
            lines.append(f"  (ungrouped): {len(ungrouped)} feature(s)")
            for f in ungrouped:
                lines.append(f"    - {f}")
        lines.append("")

        lines.append("FINAL SELECTED FEATURES")
        lines.append("-" * 80)
        lines.append(f"  Total: {len(selected_features)} features")
        lines.append("")
        for i, feat in enumerate(selected_features):
            score = (
                borda_scores.loc[feat, "total_score"]
                if feat in borda_scores.index
                else 0
            )
            lines.append(f"  {i+1:3d}. {feat} (score={score:.4f})")
        lines.append("")

        report = "\n".join(lines)
        out_path = self.output_dir / "feature_selection_report.txt"
        out_path.write_text(report)
        logger.info("Saved report to %s", out_path)
        return report

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self) -> List[str]:
        """Execute the full feature selection pipeline."""
        logger.info("=" * 60)
        logger.info("Starting multi-method cross-validation feature selection")
        logger.info("=" * 60)

        # 1. Load data for correlation computation
        logger.info("[1/6] Loading data...")
        df = load_data_from_path(self.data_path)
        X, y = prepare_features(df, target_column="is_finally_correct")
        all_numeric_features = list(X.columns)

        # 2. Load all rankings
        logger.info("[2/6] Loading rankings from ablation results...")
        rankings = self.load_all_rankings()

        # 3. Compute Borda scores
        logger.info("[3/6] Computing Borda fusion scores...")
        borda_scores = self.compute_borda_scores(rankings)

        # 4. Initial candidate set (top max_features by Borda)
        # Only keep features that exist in the actual data
        borda_features = [
            f
            for f in borda_scores.index
            if f in all_numeric_features and f not in NON_ENTROPY_FEATURES
        ]
        candidates = borda_features[
            : self.max_features + 20
        ]  # extra buffer for corr removal
        logger.info("[4/6] Initial candidates: %d features", len(candidates))

        # 5. Correlation-based redundancy removal
        logger.info(
            "[5/6] Removing correlated features (threshold=%.2f)...",
            self.corr_threshold,
        )
        filtered = self.remove_redundant_features(candidates, borda_scores, X)

        # Trim to max_features
        if len(filtered) > self.max_features:
            filtered = filtered[: self.max_features]

        # 6. Semantic layer constraints
        logger.info("[6/6] Applying semantic layer constraints...")
        selected = self.apply_layer_constraints(
            filtered, borda_scores, all_numeric_features
        )

        # Sort by Borda score
        selected = sorted(
            selected,
            key=lambda f: (
                borda_scores.loc[f, "total_score"] if f in borda_scores.index else 0
            ),
            reverse=True,
        )

        logger.info("Final selected features: %d", len(selected))

        # --- Outputs ---
        # Save feature list
        feat_df = pd.DataFrame(
            {
                "feature": selected,
                "borda_score": [
                    borda_scores.loc[f, "total_score"] if f in borda_scores.index else 0
                    for f in selected
                ],
                "semantic_group": [
                    self.get_feature_group(f) or "ungrouped" for f in selected
                ],
            }
        )
        feat_df.to_csv(self.output_dir / "selected_features.csv", index=False)

        # Visualizations
        self.plot_method_ranking_comparison(rankings)
        self.plot_borda_scores(borda_scores)
        self.plot_correlation_matrix(selected, X)

        # Report
        self.generate_report(borda_scores, selected, rankings)

        logger.info("Feature selection complete. Results in %s", self.output_dir)
        return selected


def main():
    parser = argparse.ArgumentParser(description="Multi-method Feature Selection")
    parser.add_argument(
        "--data-path",
        default="data_mining/data/merged_datasets.csv",
        help="Path to merged dataset CSV",
    )
    parser.add_argument(
        "--results-dir",
        default="data_mining/results_ablation/feature_ablation",
        help="Directory with ablation analysis results",
    )
    parser.add_argument(
        "--output-dir",
        default="data_mining/results_causal_complete/feature_selection",
        help="Output directory",
    )
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    parser.add_argument("--max-features", type=int, default=30)
    parser.add_argument("--min-features", type=int, default=15)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    selector = FeatureSelectionCrossValidator(
        data_path=args.data_path,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        corr_threshold=args.corr_threshold,
        max_features=args.max_features,
        min_features=args.min_features,
    )
    selected = selector.run()
    print(f"\nSelected {len(selected)} features:")
    for i, f in enumerate(selected):
        print(f"  {i+1}. {f}")


if __name__ == "__main__":
    main()
