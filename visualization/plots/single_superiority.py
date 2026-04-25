"""Single-vs-MAS superiority bar plot with arrow annotations.

Refactored from results_plot/accuracy/analyze_single_superiority.py.
All previously-hardcoded Chinese labels are dropped (English only).
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from visualization.base import (
    ARCH_COLORS,
    ARCH_ORDER_WITH_BASE,
    BaseVisualizer,
)
from visualization.base.data_loaders import load_csv

MAS_ARCHITECTURES = ["centralized", "debate", "hybrid", "sequential"]
SUPERIORITY_COLOR = "#2E8B57"


def analyze_single_superiority(df: pd.DataFrame) -> Dict[Tuple[str, str], dict]:
    """Identify (dataset, model) cells where single architecture is the best."""
    superiority_info: Dict[Tuple[str, str], dict] = {}
    total = best = strict = 0
    improvements_avg = []
    improvements_best = []
    win_details = []

    print("=" * 60)
    print("Single Architecture Superiority Analysis Report")
    print("=" * 60)

    for (dataset, model), group in df.groupby(["dataset", "model"]):
        arch_data = group.set_index("architecture")["accuracy"].to_dict()
        if "single" not in arch_data:
            continue

        single_acc = arch_data["single"]
        mas_accs = {k: v for k, v in arch_data.items() if k in MAS_ARCHITECTURES}
        if not mas_accs:
            continue

        total += 1
        max_mas = max(mas_accs.values())
        avg_mas = sum(mas_accs.values()) / len(mas_accs)

        info = {
            "single_acc": single_acc,
            "mas_accs": mas_accs,
            "avg_mas_acc": avg_mas,
            "max_mas_acc": max_mas,
            "is_best": False,
            "improvement_over_avg": 0,
        }

        if single_acc >= max_mas:
            best += 1
            improvement = single_acc - avg_mas
            improvements_avg.append(improvement)
            info["is_best"] = True
            info["improvement_over_avg"] = improvement

            if single_acc > max_mas:
                strict += 1
                improvements_best.append(single_acc - max_mas)

            win_details.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "improvement": improvement * 100,
                    "is_tied": single_acc == max_mas,
                }
            )

        superiority_info[(dataset, model)] = info

    if total == 0:
        print("No scenarios found.")
        return superiority_info

    print(f"\nTotal Scenarios Analyzed: {total}")
    print("\nSingle Architecture Performance:")
    print(f"  - Best or tied for best: {best} ({best/total*100:.1f}%)")
    print(f"  - Strictly better than ALL MAS: {strict} ({strict/total*100:.1f}%)")

    if improvements_best:
        print("\nAverage Improvement (when Single is best):")
        print(f"  - Over best MAS: {np.mean(improvements_best)*100:.2f} percentage points")
        print(f"  - Over MAS average: {np.mean(improvements_avg)*100:.2f} percentage points")

    if win_details:
        win_df = pd.DataFrame(win_details)
        print("\n" + "-" * 40)
        print("Scenarios where Single outperforms ALL MAS:")
        print("-" * 40)
        print("\nBy Dataset:")
        print(win_df["dataset"].value_counts().to_string())
        print("\nBy Model:")
        print(win_df["model"].value_counts().to_string())

    print("=" * 60)
    return superiority_info


class SingleSuperiorityPlot(BaseVisualizer):
    """2x3 bar plot annotated with arrows where Single beats all MAS."""

    N_ARCHS = 6  # base + 4 MAS + single
    N_ROWS = 2
    N_COLS = 3

    def __init__(self, csv_path: Path | str, output_dir: Path | str) -> None:
        super().__init__(output_dir=output_dir, base_font_size=14)
        self.csv_path = Path(csv_path)
        self.df = load_csv(self.csv_path)
        self.superiority_info = analyze_single_superiority(self.df)
        self.individual_dir = self.output_dir / "individual_subplots_superiority"

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

    def _draw_bar(self, ax: plt.Axes, combined_df: pd.DataFrame, dataset: str) -> None:
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
        ax.set_title(dataset, fontsize=16, pad=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

        y_top = 1.15 if "gsm8k" in dataset.lower() else 1.0
        ax.set_ylim(bottom=0, top=y_top)

    def _annotate_superiority(self, ax: plt.Axes, dataset: str, dataset_df: pd.DataFrame) -> None:
        bar_width = 0.8 / self.N_ARCHS
        for j, model in enumerate(sorted(dataset_df["model"].unique())):
            info = self.superiority_info.get((dataset, model))
            if not info or not info["is_best"]:
                continue
            improvement = info["improvement_over_avg"] * 100
            single_acc = info["single_acc"]
            x_center = j + (5 - (self.N_ARCHS - 1) / 2) * bar_width
            arrow_y = single_acc + 0.02

            ax.annotate(
                "",
                xy=(x_center, arrow_y + 0.06),
                xytext=(x_center, arrow_y),
                arrowprops=dict(arrowstyle="->", color=SUPERIORITY_COLOR, lw=2.5),
            )
            ax.text(
                x_center,
                arrow_y + 0.08,
                f"+{improvement:.1f}%",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="black",
            )

    def compose(
        self,
        filename: str = "single_superiority_analysis.pdf",
        save_individual: bool = True,
    ) -> None:
        datasets = sorted(self.df["dataset"].unique())

        fig, axes = plt.subplots(
            self.N_ROWS, self.N_COLS, figsize=(4.5 * self.N_COLS, 4.5 * self.N_ROWS)
        )
        axes = axes.flatten()

        last_row_start = self.N_COLS * (self.N_ROWS - 1)

        with sns.plotting_context("paper", font_scale=1.3):
            for i, dataset in enumerate(datasets):
                if i >= len(axes):
                    break
                ax = axes[i]
                dataset_df = self.df[self.df["dataset"] == dataset].sort_values("model")
                combined_df = self._augment_with_base_rows(dataset_df, dataset)

                self._draw_bar(ax, combined_df, dataset)
                self._annotate_superiority(ax, dataset, dataset_df)

                ax.set_ylabel("Accuracy" if i % self.N_COLS == 0 else "", fontsize=15)
                ax.set_xlabel("Model" if i >= last_row_start else "", fontsize=15)

                if i > 0 and ax.get_legend():
                    ax.get_legend().remove()

        for j in range(len(datasets), len(axes)):
            fig.delaxes(axes[j])

        handles, labels = axes[0].get_legend_handles_labels()
        arrow_legend = Line2D(
            [0], [0],
            marker="^",
            color=SUPERIORITY_COLOR,
            linestyle="None",
            markersize=10,
            label="Single best",
        )
        handles.append(arrow_legend)
        labels.append("Single best")

        if axes[0].get_legend():
            axes[0].get_legend().remove()
        axes[0].legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=1,
            title="",
            frameon=False,
            fontsize=13,
        )

        plt.tight_layout()
        self.save_figure(fig, filename, dpi=300)
        plt.close(fig)

        if save_individual:
            self._save_individual_subplots(datasets)

    def _save_individual_subplots(self, datasets) -> None:
        with sns.plotting_context("paper", font_scale=1.3):
            for dataset in datasets:
                dataset_df = self.df[self.df["dataset"] == dataset].sort_values("model")
                combined_df = self._augment_with_base_rows(dataset_df, dataset)

                fig, ax = plt.subplots(figsize=(7, 5.5))
                self._draw_bar(ax, combined_df, dataset)
                self._annotate_superiority(ax, dataset, dataset_df)
                ax.set_ylabel("Accuracy", fontsize=15)
                ax.set_xlabel("Model", fontsize=15)

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    labels,
                    loc="best",
                    title="",
                    frameon=False,
                    fontsize=10,
                )
                plt.tight_layout()
                self.save_subplot(fig, f"{dataset}_superiority")
                plt.close(fig)


def main() -> None:
    csv_path = Path(__file__).resolve().parents[2] / "results_plot" / "accuracy" / "accuracy.csv"
    output_dir = Path(__file__).resolve().parents[2] / "visualization" / "outputs" / "accuracy"
    plotter = SingleSuperiorityPlot(csv_path=csv_path, output_dir=output_dir)
    plotter.compose()


if __name__ == "__main__":
    main()
