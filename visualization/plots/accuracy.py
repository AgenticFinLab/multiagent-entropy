"""Accuracy bar plot + Single-vs-MAS superiority report.

Refactored from results_plot/accuracy/analyze_accuracy.py into the
`visualization` package layout.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization.base import (
    ARCH_COLORS,
    ARCH_ORDER_WITH_BASE,
    BaseVisualizer,
)
from visualization.base.data_loaders import load_csv


MAS_ARCHITECTURES = ["centralized", "debate", "hybrid", "sequential"]


def report_accuracy(df: pd.DataFrame) -> None:
    """Print a Single-vs-MAS comparison report (was analyze_accuracy)."""
    total_scenarios = 0
    single_is_best_count = 0
    single_is_superior_to_any_count = 0
    single_is_best_better_than_all_count = 0

    improvements_over_avg: List[float] = []
    improvements_over_best_mas: List[float] = []
    win_details = []

    print("--- Accuracy Analysis Report ---")

    for (dataset, model), group in df.groupby(["dataset", "model"]):
        arch_data = group.set_index("architecture")["accuracy"].to_dict()
        if "single" not in arch_data:
            continue

        single_acc = arch_data["single"]
        mas_accs = {k: v for k, v in arch_data.items() if k in MAS_ARCHITECTURES}
        if not mas_accs:
            continue

        total_scenarios += 1
        max_mas_acc = max(mas_accs.values())
        min_mas_acc = min(mas_accs.values())
        avg_mas_acc = sum(mas_accs.values()) / len(mas_accs)

        if single_acc > max_mas_acc:
            single_is_best_better_than_all_count += 1
            improvements_over_best_mas.append(single_acc - max_mas_acc)
            win_details.append(
                {"dataset": dataset, "model": model, "type": "Better than all MAS"}
            )

        all_accs = list(arch_data.values())
        if single_acc == max(all_accs):
            single_is_best_count += 1
            improvements_over_avg.append(single_acc - avg_mas_acc)

        if single_acc > min_mas_acc:
            single_is_superior_to_any_count += 1

    if total_scenarios == 0:
        print("No scenarios found.")
        return

    print(f"Total Scenarios analyzed (Dataset + Model combinations): {total_scenarios}")
    print(
        f"Cases where Single is strictly better than ALL 4 MAS architectures: "
        f"{single_is_best_better_than_all_count} "
        f"({single_is_best_better_than_all_count/total_scenarios*100:.2f}%)"
    )
    print(
        f"Cases where Single is at least as good as the best MAS (Tied or Best): "
        f"{single_is_best_count} ({single_is_best_count/total_scenarios*100:.2f}%)"
    )
    print(
        f"Cases where Single is better than at least one MAS architecture: "
        f"{single_is_superior_to_any_count} "
        f"({single_is_superior_to_any_count/total_scenarios*100:.2f}%)"
    )

    if improvements_over_best_mas:
        print(
            f"Average accuracy improvement when Single is strictly better than all "
            f"MAS: {np.mean(improvements_over_best_mas)*100:.2f} percentage points"
        )
    if improvements_over_avg:
        print(
            f"Average accuracy improvement over MAS average when Single is among the "
            f"best: {np.mean(improvements_over_avg)*100:.2f} percentage points"
        )

    if win_details:
        win_df = pd.DataFrame(win_details)
        print("\n--- Environments where Single outperforms ALL MAS ---")
        print("By Dataset:")
        print(win_df["dataset"].value_counts())
        print("\nBy Model:")
        print(win_df["model"].value_counts())


class AccuracyPlot(BaseVisualizer):
    """Bar plot: per-dataset accuracy by model × architecture."""

    def __init__(
        self,
        csv_path: Path | str,
        output_dir: Path | str,
        num_rows: int = 1,
    ) -> None:
        super().__init__(output_dir=output_dir, base_font_size=16)
        self.csv_path = Path(csv_path)
        self.num_rows = num_rows
        self.df = load_csv(self.csv_path)

    def _augment_with_base_rows(self, dataset_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        base_rows = []
        for model in dataset_df["model"].unique():
            row = dataset_df[dataset_df["model"] == model].iloc[0]
            base_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "architecture": "base",
                    "accuracy": row["base model accuracy"] / 100.0,
                }
            )
        base_df = pd.DataFrame(base_rows)
        return pd.concat([base_df, dataset_df], ignore_index=True)

    def _draw_bar(self, ax: plt.Axes, combined_df: pd.DataFrame, title: str) -> None:
        sns.barplot(
            data=combined_df,
            x="model",
            y="accuracy",
            hue="architecture",
            hue_order=ARCH_ORDER_WITH_BASE,
            ax=ax,
            palette=ARCH_COLORS,
            edgecolor="white",
            linewidth=0.8,
            saturation=0.9,
        )
        ax.set_title(title, fontsize=20, pad=15)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
        ax.set_ylim(bottom=0)

    def compose(self, filename: str = "accuracy_comparison.pdf", save_individual: bool = True) -> None:
        datasets = sorted(self.df["dataset"].unique())
        n_datasets = len(datasets)
        n_cols = math.ceil(n_datasets / self.num_rows)

        fig_width = 4.5 * n_cols
        fig_height = (4 if self.num_rows == 2 else 5) * self.num_rows

        fig, axes = plt.subplots(self.num_rows, n_cols, figsize=(fig_width, fig_height))

        if self.num_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).flatten()

        last_row_start = n_cols * (self.num_rows - 1)

        with sns.plotting_context("paper", font_scale=1.4):
            for i, dataset in enumerate(datasets):
                dataset_df = self.df[self.df["dataset"] == dataset].sort_values("model")
                combined_df = self._augment_with_base_rows(dataset_df, dataset)

                ax = axes[i]
                self._draw_bar(ax, combined_df, dataset)

                ax.set_ylabel("Accuracy" if i % n_cols == 0 else "", fontsize=18)
                ax.set_xlabel("Model" if i >= last_row_start else "", fontsize=18)
                ax.get_legend().remove()

        for j in range(len(datasets), len(axes)):
            fig.delaxes(axes[j])

        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.02, 0.6),
            title="",
            frameon=False,
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0, 1, 1], pad=1.5)
        self.save_figure(fig, filename)
        plt.close(fig)

        if save_individual:
            self._save_individual_subplots(datasets)

    def _save_individual_subplots(self, datasets: List[str]) -> None:
        with sns.plotting_context("paper", font_scale=1.4):
            for dataset in datasets:
                dataset_df = self.df[self.df["dataset"] == dataset].sort_values("model")
                combined_df = self._augment_with_base_rows(dataset_df, dataset)

                fig, ax = plt.subplots(figsize=(6, 5))
                self._draw_bar(ax, combined_df, dataset)
                ax.set_ylabel("Accuracy", fontsize=18)
                ax.set_xlabel("Model", fontsize=18)

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    labels,
                    loc="best",
                    title="",
                    frameon=False,
                    fontsize=12,
                )
                plt.tight_layout()
                self.save_subplot(fig, dataset)
                plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parents[2] / "results_plot" / "accuracy"
    csv_path = here / "accuracy.csv"
    output_dir = Path(__file__).resolve().parents[2] / "visualization" / "outputs" / "accuracy"

    df = load_csv(csv_path)
    report_accuracy(df)

    plotter = AccuracyPlot(csv_path=csv_path, output_dir=output_dir, num_rows=1)
    plotter.compose()


if __name__ == "__main__":
    main()
