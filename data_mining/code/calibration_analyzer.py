"""Model Calibration Analysis for Multi-Agent Entropy."""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

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

ModelNameMap = {
    "llama_3_1_8b_instruct": "LLaMA-3.1-8B-Instruct",
    "llama_3_2_3b_instruct": "LLaMA-3.2-3B-Instruct",
    "qwen3_0_6b": "Qwen3-0.6B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}

ALL_DATASETS = sorted(
    [
        "gsm8k",
        "math500",
        "humaneval",
        "mmlu",
        "aime2024_16384",
        "aime2025_16384",
    ]
)

DatasetNameMap = {
    "gsm8k": "GSM8K",
    "math500": "MATH500",
    "humaneval": "HumanEval",
    "mmlu": "MMLU",
    "aime2024_16384": "AIME 2024",
    "aime2025_16384": "AIME 2025",
}

MODEL_PARAMS = {
    "llama_3_2_3b_instruct": 3,
    "qwen3_0_6b": 0.6,
    "qwen3_4b": 4,
    "llama_3_1_8b_instruct": 8,
    "qwen3_8b": 8,
}


def entropy_to_confidence(values, method="inverse"):
    """Convert entropy to [0,1] confidence. inverse: 1/(1+entropy)"""
    arr = np.array(values, dtype=float)
    if method == "inverse":
        return 1.0 / (1.0 + arr)
    elif method == "minmax":
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mx == mn:
            return np.ones_like(arr)
        return 1.0 - (arr - mn) / (mx - mn)
    return 1.0 / (1.0 + arr)


def compute_ece(confidence, correctness, n_bins=10):
    """Compute Expected Calibration Error.

    Returns:
        Tuple of (ece, bin_accs, bin_confs, bin_counts, bin_edges)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confidence >= lo) & (confidence < hi)
        else:
            mask = (confidence >= lo) & (confidence <= hi)
        cnt = mask.sum()
        bin_counts[i] = cnt
        if cnt > 0:
            bin_accs[i] = correctness[mask].mean()
            bin_confs[i] = confidence[mask].mean()
    total = confidence.shape[0]
    ece = sum(
        bin_counts[i] / total * abs(bin_accs[i] - bin_confs[i])
        for i in range(n_bins)
        if bin_counts[i] > 0
    )
    return ece, bin_accs, bin_confs, bin_counts, bin_edges


# ---------------------------------------------------------------------------
# CalibrationAnalyzer
# ---------------------------------------------------------------------------


class CalibrationAnalyzer:
    """Calibration analysis for multi-agent entropy predictions."""

    def __init__(self, data_path, output_dir, n_bins=10, entropy_metrics=None):
        """
        Initialize the CalibrationAnalyzer.

        Args:
            data_path: Path to the merged dataset CSV file.
            output_dir: Directory to save analysis results.
            n_bins: Number of bins for calibration analysis.
            entropy_metrics: List of entropy column names to analyze.
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.n_bins = n_bins
        self.entropy_metrics = entropy_metrics or [
            "sample_mean_entropy",
            "sample_mean_answer_token_entropy",
        ]
        self.df = None
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    def _display_name(self, name, name_map):
        """Return friendly display name from a mapping, fallback to raw name."""
        return name_map.get(name, name)

    def load_data(self):
        """Load and validate data from CSV."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} samples, columns: {self.df.shape[1]}")
        return self

    def run(self):
        """Execute the full calibration analysis pipeline."""
        self.load_data()
        all_ece_results = []

        for metric in self.entropy_metrics:
            if metric not in self.df.columns:
                logger.warning(f"Column {metric} not found, skipping")
                continue

            metric_dir = self.output_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"=== Analyzing metric: {metric} ===")

            ece_df = self.compute_ece_table(metric, metric_dir)
            all_ece_results.append(ece_df.assign(entropy_metric=metric))

            self.plot_reliability_diagrams(metric, metric_dir)
            self.plot_reliability_diagrams_per_model(metric, metric_dir)
            self.analyze_quadrants(metric, metric_dir)
            self.plot_confidence_distribution(metric, metric_dir)

        if all_ece_results:
            summary = pd.concat(all_ece_results, ignore_index=True)
            summary.to_csv(self.output_dir / "summary_report.csv", index=False)
            logger.info(f"Summary saved to {self.output_dir / 'summary_report.csv'}")

        self.generate_report()

    # ------------------------------------------------------------------
    # Part 3: ECE table
    # ------------------------------------------------------------------

    def compute_ece_table(self, entropy_col, output_dir):
        """Compute ECE for each model x dataset combination."""
        records = []
        for model in sorted(self.df["model_name"].unique()):
            for ds in sorted(self.df["dataset"].unique()):
                subset = self.df[
                    (self.df["model_name"] == model) & (self.df["dataset"] == ds)
                ]
                if len(subset) < 10:
                    continue
                conf = entropy_to_confidence(subset[entropy_col].values)
                corr = subset["is_finally_correct"].astype(float).values
                ece, _, _, _, _ = compute_ece(conf, corr, self.n_bins)
                records.append(
                    {
                        "model_name": model,
                        "dataset": ds,
                        "ece": round(ece, 4),
                        "n_samples": len(subset),
                        "accuracy": round(corr.mean(), 4),
                    }
                )
        df_ece = pd.DataFrame(records)
        df_ece.to_csv(output_dir / "ece_table.csv", index=False)
        # Log summary
        if not df_ece.empty:
            pivot = df_ece.pivot_table(
                index="model_name", columns="dataset", values="ece"
            )
            logger.info(f"ECE Table:\n{pivot.to_string()}")
        return df_ece

    # ------------------------------------------------------------------
    # Part 4: Reliability Diagram – by model family (2 rows x 6 cols)
    # ------------------------------------------------------------------

    def plot_reliability_diagrams(self, entropy_col, output_dir):
        """Plot reliability diagrams grouped by model family (2 rows x 6 cols)."""
        families = list(MODEL_FAMILIES.keys())  # LLaMA, Qwen
        datasets = sorted(self.df["dataset"].unique())
        n_families = len(families)
        n_datasets = len(datasets)

        fig, axes = plt.subplots(
            n_families,
            n_datasets,
            figsize=(24, 8),
            squeeze=False,
        )

        for row_idx, family in enumerate(families):
            family_models = MODEL_FAMILIES[family]
            for col_idx, ds in enumerate(datasets):
                ax = axes[row_idx, col_idx]
                # Aggregate all models in this family for this dataset
                mask = self.df["model_name"].isin(family_models) & (
                    self.df["dataset"] == ds
                )
                subset = self.df[mask]

                ds_display = self._display_name(ds, DatasetNameMap)

                if len(subset) < 10:
                    ax.set_title(f"{family} / {ds_display}\n(n<10)", fontsize=9)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    continue

                conf = entropy_to_confidence(subset[entropy_col].values)
                corr = subset["is_finally_correct"].astype(float).values
                ece, bin_accs, bin_confs, bin_counts, bin_edges = compute_ece(
                    conf,
                    corr,
                    self.n_bins,
                )

                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = bin_edges[1] - bin_edges[0]

                ax.bar(
                    bin_centers,
                    bin_accs,
                    width=bin_width * 0.85,
                    alpha=0.7,
                    color="steelblue",
                    edgecolor="white",
                )
                ax.plot([0, 1], [0, 1], "r--", linewidth=1.0, label="Perfect")
                ax.set_title(
                    f"{family} / {ds_display}\nECE={ece:.4f}  n={len(subset)}",
                    fontsize=9,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                # Annotate bin counts
                for k in range(self.n_bins):
                    if bin_counts[k] > 0:
                        ax.text(
                            bin_centers[k],
                            bin_accs[k] + 0.02,
                            str(int(bin_counts[k])),
                            ha="center",
                            va="bottom",
                            fontsize=6,
                        )

                if row_idx == n_families - 1:
                    ax.set_xlabel("Confidence", fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel("Accuracy", fontsize=8)

        fig.suptitle(
            f"Reliability Diagrams by Model Family ({entropy_col})",
            fontsize=13,
            y=1.01,
        )
        plt.tight_layout()
        save_path = output_dir / "reliability_diagram.png"
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Saved reliability diagram to {save_path}")

    # ------------------------------------------------------------------
    # Part 5: Reliability Diagram – per model (5 rows x 6 cols)
    # ------------------------------------------------------------------

    def plot_reliability_diagrams_per_model(self, entropy_col, output_dir):
        """Plot reliability diagrams for each individual model (5 rows x 6 cols)."""
        models = sorted(self.df["model_name"].unique())
        datasets = sorted(self.df["dataset"].unique())
        n_models = len(models)
        n_datasets = len(datasets)

        fig, axes = plt.subplots(
            n_models,
            n_datasets,
            figsize=(24, 18),
            squeeze=False,
        )

        for row_idx, model in enumerate(models):
            model_display = self._display_name(model, ModelNameMap)
            for col_idx, ds in enumerate(datasets):
                ax = axes[row_idx, col_idx]
                ds_display = self._display_name(ds, DatasetNameMap)
                subset = self.df[
                    (self.df["model_name"] == model) & (self.df["dataset"] == ds)
                ]

                if len(subset) < 10:
                    ax.set_title(f"{model_display}\n{ds_display} (n<10)", fontsize=8)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    continue

                conf = entropy_to_confidence(subset[entropy_col].values)
                corr = subset["is_finally_correct"].astype(float).values
                ece, bin_accs, bin_confs, bin_counts, bin_edges = compute_ece(
                    conf,
                    corr,
                    self.n_bins,
                )

                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = bin_edges[1] - bin_edges[0]

                ax.bar(
                    bin_centers,
                    bin_accs,
                    width=bin_width * 0.85,
                    alpha=0.7,
                    color="steelblue",
                    edgecolor="white",
                )
                ax.plot([0, 1], [0, 1], "r--", linewidth=1.0, label="Perfect")
                ax.set_title(
                    f"{model_display}\n{ds_display}  ECE={ece:.4f}  n={len(subset)}",
                    fontsize=8,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                # Annotate bin counts
                for k in range(self.n_bins):
                    if bin_counts[k] > 0:
                        ax.text(
                            bin_centers[k],
                            bin_accs[k] + 0.02,
                            str(int(bin_counts[k])),
                            ha="center",
                            va="bottom",
                            fontsize=5,
                        )

                if row_idx == n_models - 1:
                    ax.set_xlabel("Confidence", fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel("Accuracy", fontsize=8)

        fig.suptitle(
            f"Reliability Diagrams per Model ({entropy_col})",
            fontsize=13,
            y=1.01,
        )
        plt.tight_layout()
        save_path = output_dir / "reliability_diagram_per_model.png"
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Saved per-model reliability diagram to {save_path}")

    # ------------------------------------------------------------------
    # Part 6: Quadrant analysis
    # ------------------------------------------------------------------

    def analyze_quadrants(self, entropy_col, output_dir):
        """Perform quadrant analysis based on entropy median threshold.

        Quadrants:
          - Confidently Correct   (low entropy, correct)
          - Confidently Wrong     (low entropy, wrong)
          - Uncertain Correct     (high entropy, correct)
          - Uncertain Wrong       (high entropy, wrong)
        """
        records = []

        # Helper to compute quadrant proportions for a subset
        def _quadrant_props(sub, label_dict):
            if len(sub) < 5:
                return None
            median_ent = sub[entropy_col].median()
            low = sub[entropy_col] <= median_ent
            high = ~low
            correct = sub["is_finally_correct"].astype(bool)

            n = len(sub)
            conf_correct = (low & correct).sum() / n
            conf_wrong = (low & ~correct).sum() / n
            unc_correct = (high & correct).sum() / n
            unc_wrong = (high & ~correct).sum() / n

            row = {
                **label_dict,
                "n_samples": n,
                "median_entropy": round(median_ent, 4),
                "confidently_correct": round(conf_correct, 4),
                "confidently_wrong": round(conf_wrong, 4),
                "uncertain_correct": round(unc_correct, 4),
                "uncertain_wrong": round(unc_wrong, 4),
            }
            return row

        # Global
        row = _quadrant_props(self.df, {"model_name": "ALL", "dataset": "ALL"})
        if row:
            records.append(row)

        # Per model
        for model in sorted(self.df["model_name"].unique()):
            sub = self.df[self.df["model_name"] == model]
            row = _quadrant_props(sub, {"model_name": model, "dataset": "ALL"})
            if row:
                records.append(row)

        # Per dataset
        for ds in sorted(self.df["dataset"].unique()):
            sub = self.df[self.df["dataset"] == ds]
            row = _quadrant_props(sub, {"model_name": "ALL", "dataset": ds})
            if row:
                records.append(row)

        # Per model x dataset
        for model in sorted(self.df["model_name"].unique()):
            for ds in sorted(self.df["dataset"].unique()):
                sub = self.df[
                    (self.df["model_name"] == model) & (self.df["dataset"] == ds)
                ]
                row = _quadrant_props(sub, {"model_name": model, "dataset": ds})
                if row:
                    records.append(row)

        quad_df = pd.DataFrame(records)
        quad_df.to_csv(output_dir / "quadrant_analysis.csv", index=False)
        logger.info(f"Quadrant analysis saved ({len(quad_df)} rows)")

        # Heatmap: Confidently Wrong by model x dataset
        model_ds = quad_df[
            (quad_df["model_name"] != "ALL") & (quad_df["dataset"] != "ALL")
        ]
        if model_ds.empty:
            return quad_df

        pivot = model_ds.pivot_table(
            index="model_name",
            columns="dataset",
            values="confidently_wrong",
        )

        # Rename axis labels to friendly names
        pivot.index = [self._display_name(m, ModelNameMap) for m in pivot.index]
        pivot.columns = [self._display_name(d, DatasetNameMap) for d in pivot.columns]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="Reds",
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            vmin=0,
            vmax=pivot.values.max() * 1.1 if pivot.values.max() > 0 else 0.5,
        )
        ax.set_title(
            f"Confidently Wrong Proportion ({entropy_col})",
            fontsize=12,
        )
        ax.set_ylabel("Model")
        ax.set_xlabel("Dataset")
        plt.tight_layout()
        save_path = output_dir / "quadrant_heatmap.png"
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Quadrant heatmap saved to {save_path}")

        return quad_df

    # ------------------------------------------------------------------
    # Part 7: Confidence distribution
    # ------------------------------------------------------------------

    def plot_confidence_distribution(self, entropy_col, output_dir):
        """Plot confidence distributions split by correct/incorrect, faceted by dataset."""
        datasets = sorted(self.df["dataset"].unique())
        n_ds = len(datasets)
        n_cols = min(n_ds, 3)
        n_rows = (n_ds + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            squeeze=False,
        )

        for idx, ds in enumerate(datasets):
            r, c = divmod(idx, n_cols)
            ax = axes[r][c]
            sub = self.df[self.df["dataset"] == ds].copy()
            sub["confidence"] = entropy_to_confidence(sub[entropy_col].values)

            correct = sub[sub["is_finally_correct"].astype(bool)]["confidence"]
            wrong = sub[~sub["is_finally_correct"].astype(bool)]["confidence"]

            bins = np.linspace(0, 1, 21)
            ax.hist(
                correct,
                bins=bins,
                alpha=0.6,
                color="green",
                label=f"Correct (n={len(correct)})",
                density=True,
            )
            ax.hist(
                wrong,
                bins=bins,
                alpha=0.6,
                color="red",
                label=f"Wrong (n={len(wrong)})",
                density=True,
            )
            ax.set_title(f"{self._display_name(ds, DatasetNameMap)}", fontsize=10)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)

        # Hide unused axes
        for idx in range(n_ds, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r][c].set_visible(False)

        fig.suptitle(
            f"Confidence Distribution by Correctness ({entropy_col})",
            fontsize=13,
            y=1.01,
        )
        plt.tight_layout()
        save_path = output_dir / "confidence_distribution.png"
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Confidence distribution saved to {save_path}")

    # ------------------------------------------------------------------
    # Part 8: Generate Markdown report
    # ------------------------------------------------------------------

    def generate_report(self, output_path=None):
        """Generate a comprehensive Markdown analysis report."""
        if output_path is None:
            output_path = self.output_dir / "calibration_analysis_report.md"

        # Collect ECE and quadrant data from all metrics
        all_ece = []
        all_quad = []
        for metric in self.entropy_metrics:
            metric_dir = self.output_dir / metric
            ece_path = metric_dir / "ece_table.csv"
            quad_path = metric_dir / "quadrant_analysis.csv"
            if ece_path.exists():
                df = pd.read_csv(ece_path)
                df["entropy_metric"] = metric
                all_ece.append(df)
            if quad_path.exists():
                df = pd.read_csv(quad_path)
                df["entropy_metric"] = metric
                all_quad.append(df)

        if not all_ece:
            logger.warning("No ECE data found, skipping report generation")
            return

        ece_df = pd.concat(all_ece, ignore_index=True)
        quad_df = pd.concat(all_quad, ignore_index=True) if all_quad else pd.DataFrame()

        # Use first metric as primary for the report
        primary_metric = self.entropy_metrics[0]
        ece_primary = ece_df[ece_df["entropy_metric"] == primary_metric].copy()
        quad_primary = (
            quad_df[quad_df["entropy_metric"] == primary_metric].copy()
            if not quad_df.empty
            else pd.DataFrame()
        )

        lines = []
        lines.append("# Model Calibration Analysis Report\n")
        lines.append(
            f"> Auto-generated report based on entropy metric: " f"`{primary_metric}`\n"
        )

        # ---- Section 1: Overall Calibration Summary ----
        lines.append("## 1. Overall Calibration Summary\n")

        # --- Figure: Reliability Diagram by Model Family ---
        lines.append("### Figure 1: Reliability Diagrams by Model Family\n")
        lines.append(
            f"![Reliability Diagram by Family]"
            f"({primary_metric}/reliability_diagram.png)\n"
        )
        lines.append(
            "This figure presents a **2×6 grid of reliability diagrams** "
            "(also known as calibration plots), with one row per model family "
            "(LLaMA and Qwen) and one column per dataset (AIME 2024, AIME 2025, "
            "GSM8K, HumanEval, MATH500, MMLU). Each subplot aggregates all "
            "models within the family for the given dataset.\n"
        )
        lines.append(
            "In each subplot, **blue bars** represent the observed accuracy within "
            "each confidence bin (x-axis: entropy-derived confidence via "
            "`1/(1+entropy)`, y-axis: fraction correct). The **red dashed diagonal** "
            "line represents perfect calibration — where predicted confidence equals "
            "actual accuracy. Numbers above each bar indicate the sample count in "
            "that bin. The subplot title shows the family, dataset, ECE value, and "
            "sample size.\n"
        )
        lines.append("**How to interpret:**\n")
        lines.append(
            "- Bars **above the diagonal** indicate the model is *under-confident* "
            "(accuracy exceeds predicted confidence).\n"
            "- Bars **below the diagonal** indicate the model is *over-confident* "
            "(predicted confidence exceeds actual accuracy).\n"
            "- A **lower ECE** value means better calibration (bars closer to the "
            "diagonal).\n"
            "- Empty or low-count bins suggest the model rarely produces confidence "
            "values in that range.\n"
        )

        # --- Figure: Reliability Diagram per Model ---
        lines.append("### Figure 2: Reliability Diagrams per Individual Model\n")
        lines.append(
            f"![Reliability Diagram per Model]"
            f"({primary_metric}/reliability_diagram_per_model.png)\n"
        )
        lines.append(
            "This figure presents a **5×6 grid of reliability diagrams**, with "
            "one row per individual model (LLaMA-3.1-8B-Instruct, "
            "LLaMA-3.2-3B-Instruct, Qwen3-0.6B, Qwen3-4B, Qwen3-8B) and one "
            "column per dataset. Unlike Figure 1 which aggregates by model family, "
            "this plot provides per-model granularity.\n"
        )
        lines.append(
            "The layout and visual encoding are identical to Figure 1: blue bars "
            "for observed accuracy per confidence bin, red dashed diagonal for "
            "perfect calibration, bin counts annotated above bars. This allows "
            "direct comparison of calibration quality across models of different "
            "scales (0.6B to 8B parameters) and architectures (LLaMA vs Qwen).\n"
        )
        lines.append("**How to interpret:**\n")
        lines.append(
            "- Compare rows to assess how model scale affects calibration within "
            "the same family.\n"
            "- Compare columns to see which datasets are harder to calibrate.\n"
            "- Models with consistently low ECE across datasets are better "
            "calibrated overall.\n"
        )
        if not ece_primary.empty:
            sorted_ece = ece_primary.sort_values("ece")
            global_avg_ece = ece_primary["ece"].mean()
            lines.append(f"**Global Average ECE:** {global_avg_ece:.4f}\n")

            # Per-model average ECE
            lines.append("### Average ECE by Model\n")
            lines.append("| Model | Avg ECE | Num Datasets |")
            lines.append("|-------|---------|-------------|")
            model_avg = (
                ece_primary.groupby("model_name")["ece"]
                .agg(["mean", "count"])
                .sort_values("mean")
            )
            for model, row in model_avg.iterrows():
                m_disp = self._display_name(model, ModelNameMap)
                lines.append(f"| {m_disp} | {row['mean']:.4f} | {int(row['count'])} |")
            lines.append("")

            # Per-dataset average ECE
            lines.append("### Average ECE by Dataset\n")
            lines.append("| Dataset | Avg ECE | Num Models |")
            lines.append("|---------|---------|-----------|")
            ds_avg = (
                ece_primary.groupby("dataset")["ece"]
                .agg(["mean", "count"])
                .sort_values("mean")
            )
            for ds, row in ds_avg.iterrows():
                d_disp = self._display_name(ds, DatasetNameMap)
                lines.append(f"| {d_disp} | {row['mean']:.4f} | {int(row['count'])} |")
            lines.append("")

            # Full sorted table
            lines.append("### All Model-Dataset Combinations (sorted by ECE)\n")
            lines.append("| Rank | Model | Dataset | ECE | Accuracy | N |")
            lines.append("|------|-------|---------|-----|----------|---|")
            for rank, (_, r) in enumerate(sorted_ece.iterrows(), 1):
                m_disp = self._display_name(r["model_name"], ModelNameMap)
                d_disp = self._display_name(r["dataset"], DatasetNameMap)
                lines.append(
                    f"| {rank} | {m_disp} | {d_disp} | "
                    f"{r['ece']:.4f} | {r['accuracy']:.4f} | "
                    f"{int(r['n_samples'])} |"
                )
            lines.append("")

        # ---- Section 2: Best and Worst Calibrated ----
        lines.append("## 2. Best and Worst Calibrated Configurations\n")
        if not ece_primary.empty:
            sorted_ece = ece_primary.sort_values("ece")
            top5_best = sorted_ece.head(5)
            top5_worst = sorted_ece.tail(5).iloc[::-1]

            lines.append("### Top 5 Best Calibrated (Lowest ECE)\n")
            lines.append("| Model | Dataset | ECE | Accuracy | Params |")
            lines.append("|-------|---------|-----|----------|--------|")
            for _, r in top5_best.iterrows():
                m_disp = self._display_name(r["model_name"], ModelNameMap)
                d_disp = self._display_name(r["dataset"], DatasetNameMap)
                params = MODEL_PARAMS.get(r["model_name"], "?")
                lines.append(
                    f"| {m_disp} | {d_disp} | {r['ece']:.4f} | "
                    f"{r['accuracy']:.4f} | {params}B |"
                )
            lines.append("")

            lines.append("### Top 5 Worst Calibrated (Highest ECE)\n")
            lines.append("| Model | Dataset | ECE | Accuracy | Params |")
            lines.append("|-------|---------|-----|----------|--------|")
            for _, r in top5_worst.iterrows():
                m_disp = self._display_name(r["model_name"], ModelNameMap)
                d_disp = self._display_name(r["dataset"], DatasetNameMap)
                params = MODEL_PARAMS.get(r["model_name"], "?")
                lines.append(
                    f"| {m_disp} | {d_disp} | {r['ece']:.4f} | "
                    f"{r['accuracy']:.4f} | {params}B |"
                )
            lines.append("")

            # Analysis
            lines.append("### Analysis\n")
            best_row = sorted_ece.iloc[0]
            worst_row = sorted_ece.iloc[-1]
            best_params = MODEL_PARAMS.get(best_row["model_name"], "?")
            worst_params = MODEL_PARAMS.get(worst_row["model_name"], "?")
            lines.append(
                f"- The best calibrated configuration is "
                f"**{self._display_name(best_row['model_name'], ModelNameMap)}** "
                f"on **{self._display_name(best_row['dataset'], DatasetNameMap)}** "
                f"(ECE={best_row['ece']:.4f}, {best_params}B parameters)."
            )
            lines.append(
                f"- The worst calibrated configuration is "
                f"**{self._display_name(worst_row['model_name'], ModelNameMap)}** "
                f"on **{self._display_name(worst_row['dataset'], DatasetNameMap)}** "
                f"(ECE={worst_row['ece']:.4f}, {worst_params}B parameters)."
            )
            # Check if harder datasets tend to be worse
            hard_datasets = {"aime2024_16384", "aime2025_16384"}
            worst5_ds = set(top5_worst["dataset"].values)
            if worst5_ds & hard_datasets:
                lines.append(
                    "- Challenging competition-level datasets (AIME) tend to "
                    "appear among the worst calibrated, suggesting that task "
                    "difficulty significantly impacts calibration quality."
                )
            lines.append("")

        # ---- Section 3: Confidently Wrong Analysis ----
        lines.append("## 3. Confidently Wrong Analysis\n")

        # --- Figure: Quadrant Heatmap ---
        lines.append("### Figure 3: Quadrant Analysis — Confidently Wrong Heatmap\n")
        lines.append(
            f"![Quadrant Heatmap]" f"({primary_metric}/quadrant_heatmap.png)\n"
        )
        lines.append(
            "This figure shows a **model × dataset heatmap** of the "
            "*Confidently Wrong* proportion — the fraction of samples where the "
            "model has low entropy (high confidence) yet answers incorrectly. "
            "The heatmap uses a **Reds** color scale: darker red indicates a "
            "higher proportion of confidently wrong predictions.\n"
        )
        lines.append(
            "Rows represent the 5 individual models (with friendly display names) "
            "and columns represent the 6 datasets. Each cell is annotated with "
            "the exact proportion value. The quadrant analysis splits samples by "
            "the median entropy threshold into four categories:\n\n"
            "- **Confidently Correct** (low entropy, correct)\n"
            "- **Confidently Wrong** (low entropy, incorrect) — shown in this plot\n"
            "- **Uncertain Correct** (high entropy, correct)\n"
            "- **Uncertain Wrong** (high entropy, incorrect)\n"
        )
        lines.append("**How to interpret:**\n")
        lines.append(
            "- **Darker cells** are problematic: the model is confident but wrong, "
            "posing deployment risks.\n"
            "- Compare across rows to identify which models are most prone to "
            "overconfident errors.\n"
            "- Compare across columns to identify which datasets elicit the most "
            "overconfident wrong predictions.\n"
            "- Ideally, all cells should be light (low confidently wrong rate).\n"
        )

        # --- Figure: Confidence Distribution ---
        lines.append("### Figure 4: Confidence Distribution by Correctness\n")
        lines.append(
            f"![Confidence Distribution]"
            f"({primary_metric}/confidence_distribution.png)\n"
        )
        lines.append(
            "This figure shows **overlapping histograms** of entropy-derived "
            "confidence values, split by whether the model's answer was correct "
            "(green) or incorrect (red). The plot is faceted by dataset in a "
            "grid layout (up to 2×3). Each histogram is density-normalized to "
            "allow comparison between groups of different sizes.\n"
        )
        lines.append(
            "The x-axis represents confidence (computed as `1/(1+entropy)`, "
            "ranging from 0 to 1), and the y-axis represents probability density. "
            "The legend in each subplot indicates the sample count for correct "
            "and wrong predictions.\n"
        )
        lines.append("**How to interpret:**\n")
        lines.append(
            "- **Good calibration** would show the green (correct) distribution "
            "shifted toward higher confidence values and the red (wrong) "
            "distribution shifted toward lower values, with minimal overlap.\n"
            "- **Heavy overlap** between the two distributions suggests the "
            "entropy-based confidence measure has limited discriminative power "
            "for that dataset.\n"
            "- A **peak of red at high confidence** indicates a problematic "
            "tendency toward overconfident wrong predictions.\n"
            "- Compare across datasets to see where entropy-based confidence "
            "is most/least informative.\n"
        )
        if not quad_primary.empty:
            # Global
            global_row = quad_primary[
                (quad_primary["model_name"] == "ALL")
                & (quad_primary["dataset"] == "ALL")
            ]
            if not global_row.empty:
                cw_global = global_row.iloc[0]["confidently_wrong"]
                lines.append(
                    f"**Global Confidently Wrong Proportion:** "
                    f"{cw_global:.4f} ({cw_global*100:.2f}%)\n"
                )

            # Per model
            model_rows = quad_primary[
                (quad_primary["model_name"] != "ALL")
                & (quad_primary["dataset"] == "ALL")
            ].sort_values("confidently_wrong", ascending=False)
            if not model_rows.empty:
                lines.append("### Confidently Wrong by Model\n")
                lines.append("| Model | Confidently Wrong |")
                lines.append("|-------|-------------------|")
                for _, r in model_rows.iterrows():
                    m_disp = self._display_name(r["model_name"], ModelNameMap)
                    lines.append(
                        f"| {m_disp} | {r['confidently_wrong']:.4f} "
                        f"({r['confidently_wrong']*100:.2f}%) |"
                    )
                lines.append("")

            # Per dataset
            ds_rows = quad_primary[
                (quad_primary["model_name"] == "ALL")
                & (quad_primary["dataset"] != "ALL")
            ].sort_values("confidently_wrong", ascending=False)
            if not ds_rows.empty:
                lines.append("### Confidently Wrong by Dataset\n")
                lines.append("| Dataset | Confidently Wrong |")
                lines.append("|---------|-------------------|")
                for _, r in ds_rows.iterrows():
                    d_disp = self._display_name(r["dataset"], DatasetNameMap)
                    lines.append(
                        f"| {d_disp} | {r['confidently_wrong']:.4f} "
                        f"({r['confidently_wrong']*100:.2f}%) |"
                    )
                lines.append("")

            # Correlation with ECE
            if not ece_primary.empty:
                model_ds_quad = quad_primary[
                    (quad_primary["model_name"] != "ALL")
                    & (quad_primary["dataset"] != "ALL")
                ].copy()
                merged = model_ds_quad.merge(
                    ece_primary[["model_name", "dataset", "ece"]],
                    on=["model_name", "dataset"],
                    how="inner",
                )
                if len(merged) > 3:
                    corr = merged["confidently_wrong"].corr(merged["ece"])
                    lines.append("### Correlation with ECE\n")
                    lines.append(
                        f"Pearson correlation between Confidently Wrong "
                        f"proportion and ECE: **r = {corr:.4f}**\n"
                    )
                    if abs(corr) > 0.7:
                        lines.append(
                            "This indicates a strong correlation: "
                            "configurations with higher ECE also tend to have "
                            "more confidently wrong predictions.\n"
                        )
                    elif abs(corr) > 0.4:
                        lines.append(
                            "This indicates a moderate correlation between "
                            "miscalibration and confident errors.\n"
                        )
                    else:
                        lines.append(
                            "The correlation is relatively weak, suggesting "
                            "that high ECE does not always coincide with "
                            "confident wrong predictions.\n"
                        )

        # ---- Section 4: Calibration vs Model Scale/Family ----
        lines.append("## 4. Calibration vs Model Scale/Family\n")
        if not ece_primary.empty:
            # Family comparison
            lines.append("### LLaMA vs Qwen Family Comparison\n")
            lines.append("| Family | Models | Avg ECE |")
            lines.append("|--------|--------|---------|")
            for family, members in MODEL_FAMILIES.items():
                fam_ece = ece_primary[ece_primary["model_name"].isin(members)]
                if not fam_ece.empty:
                    avg = fam_ece["ece"].mean()
                    member_names = ", ".join(
                        self._display_name(m, ModelNameMap) for m in members
                    )
                    lines.append(f"| {family} | {member_names} | {avg:.4f} |")
            lines.append("")

            # Scale analysis
            lines.append("### Model Scale vs Calibration\n")
            lines.append("| Model | Parameters | Avg ECE | Avg Accuracy |")
            lines.append("|-------|-----------|---------|-------------|")
            model_stats = (
                ece_primary.groupby("model_name")
                .agg({"ece": "mean", "accuracy": "mean"})
                .reset_index()
            )
            model_stats["params"] = model_stats["model_name"].map(MODEL_PARAMS)
            model_stats = model_stats.sort_values("params")
            for _, r in model_stats.iterrows():
                m_disp = self._display_name(r["model_name"], ModelNameMap)
                lines.append(
                    f"| {m_disp} | {r['params']}B | "
                    f"{r['ece']:.4f} | {r['accuracy']:.4f} |"
                )
            lines.append("")

            # Per-family dataset breakdown
            lines.append("### Per-Family Dataset Calibration\n")
            for family, members in MODEL_FAMILIES.items():
                fam_data = ece_primary[ece_primary["model_name"].isin(members)]
                if fam_data.empty:
                    continue
                ds_avg = fam_data.groupby("dataset")["ece"].mean().sort_values()
                lines.append(f"**{family} family:**\n")
                for ds, ece_val in ds_avg.items():
                    d_disp = self._display_name(ds, DatasetNameMap)
                    lines.append(f"- {d_disp}: ECE = {ece_val:.4f}")
                lines.append("")

        # ---- Section 5: Key Findings ----
        lines.append("## 5. Key Findings and Implications\n")
        lines.append("### Key Findings\n")

        if not ece_primary.empty:
            global_avg = ece_primary["ece"].mean()
            lines.append(
                f"1. **Overall calibration quality:** The global average ECE "
                f"is {global_avg:.4f}, indicating "
                + (
                    "relatively well-calibrated models overall."
                    if global_avg < 0.1
                    else (
                        "moderate miscalibration across configurations."
                        if global_avg < 0.2
                        else "significant miscalibration that warrants attention."
                    )
                )
            )

            # Best/worst family
            family_eces = {}
            for family, members in MODEL_FAMILIES.items():
                fam_data = ece_primary[ece_primary["model_name"].isin(members)]
                if not fam_data.empty:
                    family_eces[family] = fam_data["ece"].mean()
            if len(family_eces) == 2:
                sorted_fam = sorted(family_eces.items(), key=lambda x: x[1])
                lines.append(
                    f"2. **Family comparison:** {sorted_fam[0][0]} "
                    f"(avg ECE={sorted_fam[0][1]:.4f}) shows "
                    f"{'better' if sorted_fam[0][1] < sorted_fam[1][1] else 'comparable'} "
                    f"calibration than {sorted_fam[1][0]} "
                    f"(avg ECE={sorted_fam[1][1]:.4f})."
                )

            # Scale insight
            if not model_stats.empty:
                corr_scale = model_stats["params"].corr(model_stats["ece"])
                lines.append(
                    f"3. **Scale effect:** The correlation between model "
                    f"parameters and ECE is {corr_scale:.4f}, suggesting that "
                    + (
                        "larger models tend to be better calibrated."
                        if corr_scale < -0.3
                        else (
                            "larger models tend to be worse calibrated."
                            if corr_scale > 0.3
                            else "model scale has limited impact on calibration."
                        )
                    )
                )

            # Dataset difficulty
            ds_avg_all = ece_primary.groupby("dataset")["ece"].mean().sort_values()
            easiest = ds_avg_all.index[0]
            hardest = ds_avg_all.index[-1]
            lines.append(
                f"4. **Dataset difficulty:** "
                f"{self._display_name(easiest, DatasetNameMap)} "
                f"(avg ECE={ds_avg_all.iloc[0]:.4f}) is the easiest to "
                f"calibrate, while "
                f"{self._display_name(hardest, DatasetNameMap)} "
                f"(avg ECE={ds_avg_all.iloc[-1]:.4f}) is the hardest."
            )

        if not quad_primary.empty:
            global_cw = quad_primary[
                (quad_primary["model_name"] == "ALL")
                & (quad_primary["dataset"] == "ALL")
            ]
            if not global_cw.empty:
                cw_val = global_cw.iloc[0]["confidently_wrong"]
                lines.append(
                    f"5. **Confident errors:** {cw_val*100:.2f}% of all "
                    f"predictions are confidently wrong (low entropy but "
                    f"incorrect), highlighting potential risks in deployment."
                )

        lines.append("")
        lines.append("### Implications for the Paper\n")
        lines.append(
            "- Entropy-based confidence measures show varying calibration "
            "quality across model families and datasets."
        )
        lines.append(
            "- The presence of confidently wrong predictions suggests that "
            "entropy alone may not be sufficient for reliable uncertainty "
            "estimation in all scenarios."
        )
        lines.append(
            "- Multi-agent agreement metrics could complement entropy-based "
            "measures to improve calibration."
        )
        lines.append("")
        lines.append("### Suggested Follow-up Analyses\n")
        lines.append(
            "- Investigate temperature scaling or other post-hoc calibration "
            "methods."
        )
        lines.append(
            "- Analyze calibration differences across problem difficulty "
            "levels within each dataset."
        )
        lines.append(
            "- Compare entropy-based calibration with multi-agent "
            "agreement-based calibration."
        )
        lines.append("")

        report_text = "\n".join(lines)
        Path(output_path).write_text(report_text, encoding="utf-8")
        logger.info(f"Calibration analysis report saved to {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Model Calibration Analysis")
    parser.add_argument("--data-path", default="data_mining/data/merged_datasets.csv")
    parser.add_argument("--output-dir", default="data_mining/results_calibration")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--entropy-metrics",
        nargs="+",
        default=["sample_mean_entropy", "sample_mean_answer_token_entropy"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    analyzer = CalibrationAnalyzer(
        args.data_path,
        args.output_dir,
        args.n_bins,
        args.entropy_metrics,
    )
    analyzer.run()
    logger.info("Calibration analysis complete.")


if __name__ == "__main__":
    main()
