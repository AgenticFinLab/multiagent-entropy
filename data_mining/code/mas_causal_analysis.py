"""MAS Causal Separation Control Experiment Analysis."""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

MODEL_NAME_MAP = {
    "llama_3_1_8b_instruct": "LLaMA-3.1-8B-Instruct",
    "llama_3_2_3b_instruct": "LLaMA-3.2-3B-Instruct",
    "qwen3_0_6b": "Qwen3-0.6B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}

DATASET_NAME_MAP = {
    "gsm8k": "GSM8K",
    "math500": "MATH500",
    "humaneval": "HumanEval",
    "mmlu": "MMLU",
    "aime2024_16384": "AIME 2024",
    "aime2025_16384": "AIME 2025",
}

MODEL_FAMILIES = {
    "LLaMA": ["llama_3_1_8b_instruct", "llama_3_2_3b_instruct"],
    "Qwen": ["qwen3_0_6b", "qwen3_4b", "qwen3_8b"],
}

ALL_MODELS = [
    "llama_3_1_8b_instruct",
    "llama_3_2_3b_instruct",
    "qwen3_0_6b",
    "qwen3_4b",
    "qwen3_8b",
]

ALL_DATASETS = sorted(
    ["gsm8k", "math500", "humaneval", "mmlu", "aime2024_16384", "aime2025_16384"]
)

MAS_ARCHITECTURES = ["centralized", "debate", "hybrid", "sequential"]


class MASCausalAnalyzer:
    """Analyzer for MAS causal separation control experiments."""

    def __init__(self, data_path: str, output_dir: str, entropy_col: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading data from %s", data_path)
        self.df = pd.read_csv(data_path)
        logger.info("Loaded %d rows, %d columns", *self.df.shape)

        # Auto-detect entropy columns
        self.r1_entropy_col = entropy_col or self._detect_entropy_col("round_1")
        self.r2_entropy_col = self._detect_entropy_col("round_2")
        logger.info(
            "Using entropy columns: R1=%s, R2=%s",
            self.r1_entropy_col,
            self.r2_entropy_col,
        )

    def _detect_entropy_col(self, round_prefix: str) -> str:
        """Detect the best *sample-level* entropy column for a given round

        Priority order:
        1. sample_{round_prefix}_mean_agent_mean_entropy  (sample-level, preferred)
        2. sample_{round_prefix}_mean_agent_total_entropy  (sample-level, fallback)
        3. Any sample-level column matching the round
        4. Legacy experiment-level column (round_X_infer_avg_entropy) — last resort
        """
        # --- sample-level candidates (preferred) ---
        sample_preferred = f"sample_{round_prefix}_mean_agent_mean_entropy"
        if sample_preferred in self.df.columns:
            return sample_preferred

        sample_fallback = f"sample_{round_prefix}_mean_agent_total_entropy"
        if sample_fallback in self.df.columns:
            return sample_fallback

        # Any sample-level column for this round
        sample_candidates = [
            c
            for c in self.df.columns
            if f"sample_{round_prefix}" in c and "entropy" in c
        ]
        if sample_candidates:
            return sorted(sample_candidates)[0]

        # --- legacy experiment-level fallback ---
        legacy = f"{round_prefix}_infer_avg_entropy"
        if legacy in self.df.columns:
            logger.warning(
                "Falling back to experiment-level column %s — plots will have "
                "very few unique data points",
                legacy,
            )
            return legacy

        candidates = [
            c for c in self.df.columns if round_prefix in c and "entropy" in c
        ]
        if candidates:
            return sorted(candidates)[0]
        raise ValueError(f"No entropy column found for {round_prefix}")

    # ------------------------------------------------------------------
    # 1. Pair SAS and MAS samples
    # ------------------------------------------------------------------
    def pair_sas_mas_samples(self) -> pd.DataFrame:
        """Pair Single-Agent and Multi-Agent samples by (dataset, model_name, sample_id)."""
        single = self.df[self.df["architecture"] == "single"].copy()
        mas = self.df[self.df["architecture"].isin(MAS_ARCHITECTURES)].copy()

        single_key = single.set_index(["dataset", "model_name", "sample_id"])
        records = []

        for _, row in mas.iterrows():
            key = (row["dataset"], row["model_name"], row["sample_id"])
            if key not in single_key.index:
                continue
            sas_row = single_key.loc[key]
            # Handle duplicate index (take first)
            if isinstance(sas_row, pd.DataFrame):
                sas_row = sas_row.iloc[0]

            records.append(
                {
                    "dataset": row["dataset"],
                    "model_name": row["model_name"],
                    "sample_id": row["sample_id"],
                    "mas_architecture": row["architecture"],
                    "sas_entropy": sas_row[self.r1_entropy_col],
                    "mas_r1_entropy": row[self.r1_entropy_col],
                    "mas_r2_entropy": row[self.r2_entropy_col],
                    "sas_correct": sas_row["is_finally_correct"],
                    "mas_correct": row["is_finally_correct"],
                }
            )

        paired = pd.DataFrame(records)
        logger.info("Paired %d SAS-MAS sample pairs", len(paired))
        return paired

    # ------------------------------------------------------------------
    # 2. SAS vs MAS Round 1 statistical tests
    # ------------------------------------------------------------------
    def analyze_sas_vs_mas_r1(self, paired: pd.DataFrame = None) -> pd.DataFrame:
        """Wilcoxon signed-rank test comparing SAS entropy vs MAS R1 entropy."""
        if paired is None:
            paired = self.pair_sas_mas_samples()

        results = []
        for model in ALL_MODELS:
            for dataset in ALL_DATASETS:
                sub = paired[
                    (paired["model_name"] == model) & (paired["dataset"] == dataset)
                ]
                if len(sub) < 5:
                    continue

                sas_e = sub["sas_entropy"].values
                mas_e = sub["mas_r1_entropy"].values
                diff = mas_e - sas_e

                mean_diff = np.nanmean(diff)
                std_diff = np.nanstd(diff, ddof=1)
                cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

                # Wilcoxon signed-rank test
                valid = diff[~np.isnan(diff)]
                valid = valid[valid != 0]
                if len(valid) >= 5:
                    stat, pval = stats.wilcoxon(valid)
                else:
                    stat, pval = np.nan, np.nan

                results.append(
                    {
                        "model": MODEL_NAME_MAP.get(model, model),
                        "model_key": model,
                        "dataset": DATASET_NAME_MAP.get(dataset, dataset),
                        "dataset_key": dataset,
                        "n_pairs": len(sub),
                        "sas_entropy_mean": np.nanmean(sas_e),
                        "mas_r1_entropy_mean": np.nanmean(mas_e),
                        "mean_diff": mean_diff,
                        "cohens_d": cohens_d,
                        "wilcoxon_stat": stat,
                        "p_value": pval,
                        "significant": pval < 0.05 if not np.isnan(pval) else False,
                    }
                )

        result_df = pd.DataFrame(results)
        out_path = self.output_dir / "sas_vs_mas_r1_stats.csv"
        result_df.to_csv(out_path, index=False)
        logger.info("Saved SAS vs MAS R1 stats to %s", out_path)
        return result_df

    # ------------------------------------------------------------------
    # 3. Entropy change vs accuracy analysis
    # ------------------------------------------------------------------
    def analyze_entropy_change_vs_accuracy(
        self, paired: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Analyze Round 1->2 entropy change direction vs accuracy change."""
        if paired is None:
            paired = self.pair_sas_mas_samples()

        results = []
        for model in ALL_MODELS:
            for dataset in ALL_DATASETS:
                for arch in MAS_ARCHITECTURES:
                    sub = paired[
                        (paired["model_name"] == model)
                        & (paired["dataset"] == dataset)
                        & (paired["mas_architecture"] == arch)
                    ]
                    if len(sub) == 0:
                        continue

                    entropy_diff = sub["mas_r2_entropy"] - sub["mas_r1_entropy"]
                    acc_diff = sub["mas_correct"].astype(float) - sub[
                        "sas_correct"
                    ].astype(float)

                    n = len(sub)
                    eps = 1e-8
                    n_entropy_down = (entropy_diff < -eps).sum()
                    n_entropy_same = (
                        (entropy_diff >= -eps) & (entropy_diff <= eps)
                    ).sum()
                    n_entropy_up = (entropy_diff > eps).sum()

                    # Cross-tabulation
                    genuine_improve = ((entropy_diff < -eps) & (acc_diff > 0)).sum()
                    possible_anchor = ((entropy_diff < -eps) & (acc_diff <= 0)).sum()
                    entropy_up_acc_up = ((entropy_diff > eps) & (acc_diff > 0)).sum()
                    entropy_up_acc_down = ((entropy_diff > eps) & (acc_diff < 0)).sum()

                    results.append(
                        {
                            "model": MODEL_NAME_MAP.get(model, model),
                            "model_key": model,
                            "dataset": DATASET_NAME_MAP.get(dataset, dataset),
                            "dataset_key": dataset,
                            "architecture": arch,
                            "n_samples": n,
                            "pct_entropy_down": n_entropy_down / n,
                            "pct_entropy_same": n_entropy_same / n,
                            "pct_entropy_up": n_entropy_up / n,
                            "sas_accuracy": sub["sas_correct"].mean(),
                            "mas_accuracy": sub["mas_correct"].mean(),
                            "pct_genuine_improve": genuine_improve / n,
                            "pct_possible_anchor": possible_anchor / n,
                            "pct_entropy_up_acc_up": entropy_up_acc_up / n,
                            "pct_entropy_up_acc_down": entropy_up_acc_down / n,
                            "mean_entropy_change": entropy_diff.mean(),
                            "mean_acc_change": acc_diff.mean(),
                        }
                    )

        result_df = pd.DataFrame(results)
        out_path = self.output_dir / "entropy_change_analysis.csv"
        result_df.to_csv(out_path, index=False)
        logger.info("Saved entropy change analysis to %s", out_path)
        return result_df

    # ------------------------------------------------------------------
    # 4. Three-way comparison plot (SAS / MAS R1 / MAS R2)
    # ------------------------------------------------------------------
    def plot_three_way_comparison(self, paired: pd.DataFrame = None):
        """5x6 violin/box plot grid: SAS entropy, MAS R1, MAS R2."""
        if paired is None:
            paired = self.pair_sas_mas_samples()

        # Aggregate MAS architectures (mean across archs per sample)
        agg = (
            paired.groupby(["dataset", "model_name", "sample_id"])
            .agg(
                sas_entropy=("sas_entropy", "first"),
                mas_r1_entropy=("mas_r1_entropy", "mean"),
                mas_r2_entropy=("mas_r2_entropy", "mean"),
            )
            .reset_index()
        )

        fig, axes = plt.subplots(
            len(ALL_MODELS),
            len(ALL_DATASETS),
            figsize=(20, 16),
            squeeze=False,
        )
        fig.suptitle(
            "Three-Way Entropy Comparison: SAS vs MAS Round 1 vs MAS Round 2",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        for i, model in enumerate(ALL_MODELS):
            for j, dataset in enumerate(ALL_DATASETS):
                ax = axes[i, j]
                sub = agg[(agg["model_name"] == model) & (agg["dataset"] == dataset)]

                if len(sub) == 0:
                    ax.set_visible(False)
                    continue

                plot_data = pd.DataFrame(
                    {
                        "SAS": sub["sas_entropy"].values,
                        "MAS R1": sub["mas_r1_entropy"].values,
                        "MAS R2": sub["mas_r2_entropy"].values,
                    }
                )
                melted = plot_data.melt(var_name="Stage", value_name="Entropy")

                sns.violinplot(
                    data=melted,
                    x="Stage",
                    y="Entropy",
                    ax=ax,
                    palette=["#4ECDC4", "#FF6B6B", "#45B7D1"],
                    inner="box",
                    cut=0,
                    scale="width",
                )

                # Annotate means
                for k, col in enumerate(["SAS", "MAS R1", "MAS R2"]):
                    mean_val = plot_data[col].mean()
                    ax.text(
                        k,
                        ax.get_ylim()[1] * 0.95,
                        f"μ={mean_val:.2f}",
                        ha="center",
                        va="top",
                        fontsize=6,
                        fontweight="bold",
                    )

                # p-value for SAS vs MAS R1
                sas_vals = sub["sas_entropy"].dropna().values
                mas_r1_vals = sub["mas_r1_entropy"].dropna().values
                diff = mas_r1_vals - sas_vals
                diff = diff[diff != 0]
                if len(diff) >= 5:
                    _, pval = stats.wilcoxon(diff)
                    pstr = f"p={pval:.1e}" if pval < 0.001 else f"p={pval:.3f}"
                    ax.set_title(
                        f"{MODEL_NAME_MAP.get(model, model)}\n{DATASET_NAME_MAP.get(dataset, dataset)} ({pstr})",
                        fontsize=7,
                    )
                else:
                    ax.set_title(
                        f"{MODEL_NAME_MAP.get(model, model)}\n{DATASET_NAME_MAP.get(dataset, dataset)}",
                        fontsize=7,
                    )

                ax.tick_params(labelsize=6)
                ax.set_xlabel("")
                if j > 0:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel("Entropy", fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = self.output_dir / "three_way_comparison.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved three-way comparison to %s", out_path)

    # ------------------------------------------------------------------
    # 5. Entropy change direction stacked bar
    # ------------------------------------------------------------------
    def plot_entropy_change_direction(self, paired: pd.DataFrame = None):
        """Dual-panel heatmap: entropy change magnitude vs accuracy change.

        Since nearly all samples show entropy decrease, the original stacked
        bar chart lacked discriminative power.  This replacement shows *how
        much* entropy changed and whether that correlates with accuracy,
        using side-by-side heatmaps for easy visual comparison.
        """
        if paired is None:
            paired = self.pair_sas_mas_samples()

        paired = paired.copy()
        paired["entropy_diff"] = paired["mas_r2_entropy"] - paired["mas_r1_entropy"]
        paired["acc_diff"] = paired["mas_correct"].astype(float) - paired[
            "sas_correct"
        ].astype(float)

        # Build a summary table: one row per (model, architecture), one col per dataset
        records = []
        for model in ALL_MODELS:
            for arch in MAS_ARCHITECTURES:
                for dataset in ALL_DATASETS:
                    sub = paired[
                        (paired["model_name"] == model)
                        & (paired["dataset"] == dataset)
                        & (paired["mas_architecture"] == arch)
                    ]
                    if len(sub) == 0:
                        records.append(
                            {
                                "model": model,
                                "architecture": arch,
                                "dataset": dataset,
                                "mean_entropy_change": np.nan,
                                "acc_change": np.nan,
                            }
                        )
                        continue
                    records.append(
                        {
                            "model": model,
                            "architecture": arch,
                            "dataset": dataset,
                            "mean_entropy_change": sub["entropy_diff"].mean(),
                            "acc_change": sub["acc_diff"].mean(),
                        }
                    )

        summary = pd.DataFrame(records)

        # Friendly labels for rows and columns
        summary["row_label"] = summary.apply(
            lambda r: f"{MODEL_NAME_MAP.get(r['model'], r['model'])}  |  {r['architecture'].capitalize()}",
            axis=1,
        )
        summary["col_label"] = summary["dataset"].map(
            lambda d: DATASET_NAME_MAP.get(d, d)
        )

        # Pivot into matrices
        row_order = []
        for model in ALL_MODELS:
            for arch in MAS_ARCHITECTURES:
                label = f"{MODEL_NAME_MAP.get(model, model)}  |  {arch.capitalize()}"
                if label in summary["row_label"].values:
                    row_order.append(label)
        col_order = [DATASET_NAME_MAP.get(d, d) for d in ALL_DATASETS]

        entropy_pivot = summary.pivot_table(
            index="row_label",
            columns="col_label",
            values="mean_entropy_change",
            aggfunc="first",
        )
        acc_pivot = summary.pivot_table(
            index="row_label",
            columns="col_label",
            values="acc_change",
            aggfunc="first",
        )

        # Reindex to enforce order
        row_order = [r for r in row_order if r in entropy_pivot.index]
        col_order = [c for c in col_order if c in entropy_pivot.columns]
        entropy_pivot = entropy_pivot.reindex(index=row_order, columns=col_order)
        acc_pivot = acc_pivot.reindex(index=row_order, columns=col_order)

        # --- Draw dual-panel heatmap ---
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(22, max(10, len(row_order) * 0.55)), sharey=True
        )
        fig.suptitle(
            "Entropy Change Magnitude & Accuracy Change: MAS Round 1 → Round 2",
            fontsize=15,
            fontweight="bold",
            y=1.01,
        )

        # Symmetric color limits for entropy
        e_abs_max = max(
            abs(np.nanmin(entropy_pivot.values)),
            abs(np.nanmax(entropy_pivot.values)),
        )
        a_abs_max = max(
            abs(np.nanmin(acc_pivot.values)),
            abs(np.nanmax(acc_pivot.values)),
        )

        # Panel 1: Entropy change
        sns.heatmap(
            entropy_pivot,
            ax=ax1,
            cmap="RdBu",
            center=0,
            vmin=-e_abs_max,
            vmax=e_abs_max,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Mean Entropy Change (R2 − R1)", "shrink": 0.8},
            annot_kws={"fontsize": 7},
        )
        ax1.set_title(
            "Entropy Change (R2 − R1)", fontsize=12, fontweight="bold", pad=10
        )
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        ax1.tick_params(axis="y", labelsize=8)
        ax1.tick_params(axis="x", labelsize=9, rotation=30)

        # Panel 2: Accuracy change
        sns.heatmap(
            acc_pivot,
            ax=ax2,
            cmap="RdYlGn",
            center=0,
            vmin=-a_abs_max,
            vmax=a_abs_max,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Accuracy Change (MAS − SAS)", "shrink": 0.8},
            annot_kws={"fontsize": 7},
        )
        ax2.set_title(
            "Accuracy Change (MAS − SAS)", fontsize=12, fontweight="bold", pad=10
        )
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ax2.tick_params(axis="x", labelsize=9, rotation=30)

        # Add model-group separators (horizontal lines between model families)
        rows_per_model = len(MAS_ARCHITECTURES)
        for k in range(1, len(ALL_MODELS)):
            y_pos = k * rows_per_model
            if y_pos <= len(row_order):
                ax1.axhline(y=y_pos, color="black", linewidth=1.5)
                ax2.axhline(y=y_pos, color="black", linewidth=1.5)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = self.output_dir / "entropy_change_direction.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved entropy change direction (dual heatmap) to %s", out_path)

    # ------------------------------------------------------------------
    # 6. Paired scatter plot
    # ------------------------------------------------------------------
    def plot_paired_scatter(self, paired: pd.DataFrame = None):
        """Scatter: MAS R1 entropy vs MAS R2 entropy, colored by correctness."""
        if paired is None:
            paired = self.pair_sas_mas_samples()

        # Aggregate across MAS architectures
        agg = (
            paired.groupby(["dataset", "model_name", "sample_id"])
            .agg(
                mas_r1_entropy=("mas_r1_entropy", "mean"),
                mas_r2_entropy=("mas_r2_entropy", "mean"),
                mas_correct=("mas_correct", "max"),
            )
            .reset_index()
        )

        datasets = sorted(agg["dataset"].unique())
        n_cols = 3
        n_rows = int(np.ceil(len(datasets) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False
        )
        fig.suptitle(
            "Paired Entropy Scatter: MAS Round 1 vs Round 2",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        palette = {0: "#e74c3c", 1: "#2ecc71"}

        for idx, dataset in enumerate(datasets):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            sub = agg[agg["dataset"] == dataset]

            for correct_val, label, marker in [
                (0, "Incorrect", "x"),
                (1, "Correct", "o"),
            ]:
                mask = sub["mas_correct"] == correct_val
                ax.scatter(
                    sub.loc[mask, "mas_r1_entropy"],
                    sub.loc[mask, "mas_r2_entropy"],
                    c=palette[correct_val],
                    marker=marker,
                    alpha=0.3,
                    s=10,
                    label=label,
                )

            # Diagonal line
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            ax.set_title(DATASET_NAME_MAP.get(dataset, dataset), fontsize=11)
            ax.set_xlabel("MAS Round 1 Entropy", fontsize=9)
            ax.set_ylabel("MAS Round 2 Entropy", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")
            ax.tick_params(labelsize=7)

        # Hide unused axes
        for idx in range(len(datasets), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = self.output_dir / "paired_entropy_scatter.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved paired entropy scatter to %s", out_path)

    # ------------------------------------------------------------------
    # 7. Generate markdown report
    # ------------------------------------------------------------------
    def generate_report(
        self,
        paired: pd.DataFrame = None,
        r1_stats: pd.DataFrame = None,
        change_stats: pd.DataFrame = None,
    ):
        """Generate comprehensive markdown analysis report."""
        if paired is None:
            paired = self.pair_sas_mas_samples()
        if r1_stats is None:
            r1_stats = self.analyze_sas_vs_mas_r1(paired)
        if change_stats is None:
            change_stats = self.analyze_entropy_change_vs_accuracy(paired)

        lines = []
        lines.append("# MAS Causal Separation Analysis Report\n")

        # Executive Summary
        lines.append("## Executive Summary\n")
        n_total = len(paired)
        n_significant = r1_stats["significant"].sum() if len(r1_stats) > 0 else 0
        n_tests = len(r1_stats)
        lines.append(
            f"This analysis examines the causal relationship between multi-agent "
            f"system (MAS) interaction and entropy changes. We paired **{n_total:,}** "
            f"SAS-MAS sample pairs across {len(ALL_MODELS)} models and "
            f"{len(ALL_DATASETS)} datasets.\n"
        )
        lines.append(
            f"- **{n_significant}/{n_tests}** model-dataset combinations show "
            f"statistically significant differences between SAS and MAS Round 1 entropy "
            f"(Wilcoxon signed-rank test, p < 0.05).\n"
        )

        # SAS vs MAS R1
        lines.append("## SAS vs MAS Round 1 Entropy Comparison\n")
        lines.append(
            "The difference between SAS and MAS Round 1 entropy is itself a meaningful "
            "finding: MAS Round 1 agents operate with different role/context prompts, "
            "which can shift the entropy distribution even before inter-agent interaction.\n"
        )

        # --- Figure 1: Three-Way Comparison ---
        lines.append(
            "### Figure 1: Three-Way Entropy Comparison "
            "(SAS vs MAS Round 1 vs MAS Round 2)\n"
        )
        lines.append("![Three-Way Comparison](three_way_comparison.png)\n")
        lines.append(
            "This figure presents a **5×6 grid of violin plots** comparing token "
            "entropy distributions across three conditions for every model–dataset "
            "combination. Each row corresponds to one of the five models "
            "(LLaMA-3.1-8B-Instruct, LLaMA-3.2-3B-Instruct, Qwen3-0.6B, Qwen3-4B, "
            "Qwen3-8B) and each column to one of the six benchmark datasets "
            "(AIME 2024, AIME 2025, GSM8K, HumanEval, MATH500, MMLU).\n"
        )
        lines.append(
            "Within each subplot, three violin plots are shown side by side:\n\n"
            "- **SAS** (teal): The entropy distribution when a single agent answers "
            "the question independently.\n"
            "- **MAS R1** (red): The entropy distribution at Round 1 of multi-agent "
            "interaction, where agents have been assigned roles and context but have "
            "not yet exchanged messages.\n"
            "- **MAS R2** (blue): The entropy distribution at Round 2, after agents "
            "have discussed and refined their answers.\n\n"
            "Each violin includes an embedded box plot showing the median and "
            "interquartile range, and the mean (μ) is annotated above each violin. "
            "The subplot title also includes the Wilcoxon signed-rank test p-value "
            "for the SAS vs MAS R1 comparison where sufficient data exists.\n"
        )
        lines.append("**How to interpret:**\n")
        lines.append(
            "- A **leftward shift** from SAS to MAS R1 indicates that multi-agent "
            "role prompts alone reduce entropy (more confident initial predictions).\n"
            "- A **further shift** from MAS R1 to MAS R2 indicates that inter-agent "
            "discussion drives additional entropy reduction (consensus formation).\n"
            "- The **width** of each violin reflects the density of samples at that "
            "entropy level; wider regions indicate more samples.\n"
            "- Significant p-values (p < 0.05) in the subplot title confirm that "
            "the SAS→MAS R1 difference is statistically reliable.\n"
        )
        if len(r1_stats) > 0:
            n_sig = r1_stats["significant"].sum()
            avg_diff = r1_stats["mean_diff"].mean()
            lines.append(
                f"**Key observations:** {n_sig}/{len(r1_stats)} model-dataset "
                f"combinations show statistically significant SAS vs MAS R1 "
                f"differences (p < 0.05). The average entropy difference "
                f"(MAS R1 − SAS) across all combinations is {avg_diff:.4f}.\n"
            )
        lines.append(
            "| Model | Dataset | N | SAS Mean | MAS R1 Mean | Diff | Cohen's d | p-value | Sig. |"
        )
        lines.append(
            "|-------|---------|---|----------|-------------|------|-----------|---------|------|"
        )
        for _, row in r1_stats.iterrows():
            sig_mark = "***" if row.get("significant", False) else ""
            pval_str = (
                f"{row['p_value']:.1e}" if not np.isnan(row["p_value"]) else "N/A"
            )
            lines.append(
                f"| {row['model']} | {row['dataset']} | {row['n_pairs']} | "
                f"{row['sas_entropy_mean']:.3f} | {row['mas_r1_entropy_mean']:.3f} | "
                f"{row['mean_diff']:.3f} | {row['cohens_d']:.3f} | "
                f"{pval_str} | {sig_mark} |"
            )
        lines.append("")

        # Entropy change vs accuracy
        lines.append("## Round 1→2 Entropy Change vs Accuracy\n")
        lines.append(
            "We categorize each MAS sample by whether entropy decreased, stayed the "
            "same, or increased from Round 1 to Round 2, and examine accuracy changes.\n"
        )

        # --- Figure 2: Entropy Change Direction Heatmap ---
        lines.append(
            "### Figure 2: Entropy Change Magnitude & Accuracy Change Heatmap\n"
        )
        lines.append("![Entropy Change Direction](entropy_change_direction.png)\n")
        lines.append(
            "This figure displays a **dual-panel heatmap** visualizing the "
            "relationship between entropy change and accuracy change across all "
            "model–architecture–dataset combinations.\n"
        )
        lines.append(
            "**Left panel — Entropy Change (R2 − R1):** Each cell shows the mean "
            "entropy change from MAS Round 1 to Round 2. The color scale uses a "
            "diverging **RdBu** palette centered at zero: blue cells indicate "
            "entropy decreased (model became more confident), while red cells "
            "indicate entropy increased (model became less certain). Rows are "
            "organized as *Model | Architecture* combinations (5 models × 4 MAS "
            "architectures = up to 20 rows), and columns represent the 6 datasets. "
            "Black horizontal lines separate different models.\n"
        )
        lines.append(
            "**Right panel — Accuracy Change (MAS − SAS):** Each cell shows the "
            "mean accuracy difference between MAS and SAS outcomes. The **RdYlGn** "
            "diverging palette is used: green indicates accuracy improvement, "
            "red indicates degradation, and yellow indicates no change.\n"
        )
        lines.append("**How to interpret:**\n")
        lines.append(
            "- Look for cells that are **blue on the left and green on the right** "
            "— these are *genuine improvement* cases where entropy reduction "
            "co-occurs with accuracy gains.\n"
            "- Cells that are **blue on the left but red/yellow on the right** "
            "suggest *anchoring/copying* behavior: entropy decreases without "
            "accuracy benefit.\n"
            "- Compare rows within the same model to see which MAS architectures "
            "(centralized, debate, hybrid, sequential) yield the best outcomes.\n"
            "- The numeric annotations in each cell provide the exact magnitude "
            "for precise comparison.\n"
        )

        if len(change_stats) > 0:
            avg_genuine = change_stats["pct_genuine_improve"].mean()
            avg_anchor = change_stats["pct_possible_anchor"].mean()
            lines.append(
                f"- Average **genuine improvement** rate (entropy ↓ + accuracy ↑): "
                f"**{avg_genuine:.1%}**\n"
            )
            lines.append(
                f"- Average **possible anchoring** rate (entropy ↓ + accuracy ↓/=): "
                f"**{avg_anchor:.1%}**\n"
            )

        lines.append("### Breakdown by Architecture\n")
        for arch in MAS_ARCHITECTURES:
            arch_data = change_stats[change_stats["architecture"] == arch]
            if len(arch_data) == 0:
                continue
            lines.append(f"**{arch.capitalize()}:**")
            lines.append(
                f"- Entropy decrease: {arch_data['pct_entropy_down'].mean():.1%} "
                f"of samples"
            )
            lines.append(
                f"- Genuine improvement: {arch_data['pct_genuine_improve'].mean():.1%}"
            )
            lines.append(
                f"- Possible anchoring: {arch_data['pct_possible_anchor'].mean():.1%}"
            )
            lines.append(
                f"- Mean accuracy change: {arch_data['mean_acc_change'].mean():.3f}\n"
            )

        # Genuinely Improved vs Anchoring Evidence
        lines.append("## Evidence: Genuine Improvement vs Anchoring/Copying\n")
        lines.append(
            "If MAS interactions genuinely improve reasoning, we expect entropy "
            "decreases to co-occur with accuracy improvements. If agents merely "
            "anchor or copy, we expect entropy decreases without accuracy gains.\n"
        )

        # --- Figure 3: Paired Entropy Scatter ---
        lines.append(
            "### Figure 3: Paired Entropy Scatter " "(MAS Round 1 vs Round 2)\n"
        )
        lines.append("![Paired Entropy Scatter](paired_entropy_scatter.png)\n")
        lines.append(
            "This figure shows a **faceted scatter plot** with one subplot per "
            "dataset (arranged in a 2×3 grid). In each subplot, every point "
            "represents a single question (sample), with the x-axis showing "
            "**MAS Round 1 entropy** and the y-axis showing **MAS Round 2 entropy**. "
            "Points are aggregated across all MAS architectures per sample.\n"
        )
        lines.append(
            "Points are color-coded and shaped by final correctness:\n\n"
            "- **Green circles (○):** Correctly answered questions.\n"
            "- **Red crosses (×):** Incorrectly answered questions.\n\n"
            "A **dashed diagonal line** (y = x) serves as a reference: points "
            "below this line indicate entropy *decreased* from Round 1 to Round 2 "
            "(the model became more confident after discussion), while points above "
            "indicate entropy *increased*.\n"
        )
        lines.append("**How to interpret:**\n")
        lines.append(
            "- If most points fall **below the diagonal**, multi-agent discussion "
            "systematically reduces entropy (consensus effect).\n"
            "- If **green points cluster below the diagonal** and **red points "
            "scatter above**, it suggests that entropy reduction is associated "
            "with genuine improvement.\n"
            "- If both correct and incorrect answers show similar entropy reduction "
            "patterns, it may indicate anchoring rather than genuine reasoning "
            "improvement.\n"
            "- The **spread of the point cloud** around the diagonal indicates how "
            "variable the entropy change is across samples.\n"
        )

        if len(change_stats) > 0:
            # Per-model summary
            for model in ALL_MODELS:
                model_data = change_stats[change_stats["model_key"] == model]
                if len(model_data) == 0:
                    continue
                gi = model_data["pct_genuine_improve"].mean()
                pa = model_data["pct_possible_anchor"].mean()
                ratio = gi / pa if pa > 0 else float("inf")
                lines.append(
                    f"- **{MODEL_NAME_MAP.get(model, model)}**: "
                    f"Genuine={gi:.1%}, Anchoring={pa:.1%}, Ratio={ratio:.2f}"
                )
            lines.append("")

        # Key Findings
        lines.append("## Key Findings\n")
        lines.append(
            "1. **SAS vs MAS R1 Difference**: The entropy shift at Round 1 reflects "
            "the impact of role/context prompts in MAS, constituting a meaningful "
            "baseline difference.\n"
        )
        lines.append(
            "2. **Round 1→2 Entropy Dynamics**: The multi-round interaction in MAS "
            "predominantly leads to entropy reduction, consistent with consensus "
            "formation.\n"
        )
        lines.append(
            "3. **Genuine vs Anchoring**: The ratio of genuine improvement to possible "
            "anchoring provides evidence for whether MAS interactions add value beyond "
            "mere agreement.\n"
        )
        lines.append(
            "4. **Implications**: These findings support/qualify the paper's conclusions "
            "about MAS entropy dynamics and their relationship to task performance.\n"
        )

        report = "\n".join(lines)
        out_path = self.output_dir / "mas_causal_analysis_report.md"
        out_path.write_text(report)
        logger.info("Saved report to %s", out_path)
        return report

    # ------------------------------------------------------------------
    # Run all analyses
    # ------------------------------------------------------------------
    def run_all(self):
        """Execute all analyses and generate outputs."""
        logger.info("Starting MAS causal analysis...")

        logger.info("[1/7] Pairing SAS-MAS samples...")
        paired = self.pair_sas_mas_samples()

        logger.info("[2/7] Analyzing SAS vs MAS R1...")
        r1_stats = self.analyze_sas_vs_mas_r1(paired)

        logger.info("[3/7] Analyzing entropy change vs accuracy...")
        change_stats = self.analyze_entropy_change_vs_accuracy(paired)

        logger.info("[4/7] Plotting three-way comparison...")
        self.plot_three_way_comparison(paired)

        logger.info("[5/7] Plotting entropy change direction...")
        self.plot_entropy_change_direction(paired)

        logger.info("[6/7] Plotting paired scatter...")
        self.plot_paired_scatter(paired)

        logger.info("[7/7] Generating report...")
        self.generate_report(paired, r1_stats, change_stats)

        logger.info("All analyses complete. Output in %s", self.output_dir)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="MAS Causal Separation Control Experiment Analysis"
    )
    parser.add_argument(
        "--data-path",
        default="data_mining/data/merged_datasets.csv",
        help="Path to merged_datasets.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data_mining/results_causal",
        help="Output directory for results",
    )
    parser.add_argument(
        "--entropy-col",
        default=None,
        help="Primary entropy column name (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    analyzer = MASCausalAnalyzer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        entropy_col=args.entropy_col,
    )
    analyzer.run_all()


if __name__ == "__main__":
    main()
