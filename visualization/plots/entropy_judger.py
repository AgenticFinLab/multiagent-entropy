"""
Entropy Judger result visualization (LLaMA + Qwen combined).

Produces three figures:
  1. judger_best_of_k.png  — 2 rows (K=1/K=3) × 6 dataset subplots; X=arch,
                             grouped bars by method; dark=LLaMA, light=Qwen
  2. judger_delta_heatmap.png — Judger−Random delta heatmap, LLaMA/Qwen side-by-side
  3. judger_early_stop.png — accuracy vs avg runs used; LLaMA solid, Qwen dashed

Usage (from repo root):
    python visualization/plots/entropy_judger.py \
        --llama-cache entropy_judger/results_llama/all_runs_scored.csv \
        --qwen-cache  entropy_judger/results_qwen/all_runs_scored.csv  \
        --output-dir  entropy_judger/figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))

from visualization.base import BaseVisualizer, ARCH_COLORS, ARCH_ORDER

# ── colour scheme ─────────────────────────────────────────────────────────────
# Each method: (LLaMA-dark, Qwen-light)
METHOD_COLORS: dict[str, tuple[str, str]] = {
    "Random":  ("#888888", "#CCCCCC"),
    "MajVote": ("#4575B4", "#91BFD8"),
    "Judger":  ("#D73027", "#FC8D59"),
    "Oracle":  ("#2D004B", "#9970AB"),
}
METHOD_ORDER = ["Random", "MajVote", "Judger", "Oracle"]

MODEL_FAMILIES: dict[str, list[str]] = {
    "llama": ["llama_3_1_8b_instruct", "llama_3_2_3b_instruct"],
    "qwen":  ["qwen3_4b", "qwen3_8b"],
}
FAMILY_LABEL = {"llama": "LLaMA", "qwen": "Qwen"}

DATASET_ORDER = ["gsm8k", "math500", "humaneval", "mmlu", "aime2024", "aime2025"]
DATASET_LABELS = {
    "gsm8k":    "GSM8K",
    "math500":  "MATH500",
    "humaneval":"HumanEval",
    "mmlu":     "MMLU",
    "aime2024": "AIME2024",
    "aime2025": "AIME2025",
}


def _family(model_name: str) -> str:
    for fam, members in MODEL_FAMILIES.items():
        if model_name in members:
            return fam
    return "llama"


def _ax_style(ax: plt.Axes) -> None:
    """Apply shared axis style (mirrors existing plot convention)."""
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    ax.set_ylim(bottom=0)


# ── metric computation ────────────────────────────────────────────────────────

def _majority_vote(sg: pd.DataFrame, k: int) -> float:
    return float(sg.head(k)["is_correct"].sum() > k / 2)


def compute_bestk(df: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    rows = []
    for (dataset, model, arch), grp in df.groupby(["dataset", "model_name", "architecture"]):
        sample_groups = {sid: g for sid, g in grp.groupby("sample_id")}
        for k in k_values:
            rands, majs, judgs, oracs = [], [], [], []
            for sg in sample_groups.values():
                sg_k = sg.sort_values("run_k").head(k)
                if sg_k.empty:
                    continue
                rands.append(sg_k["is_correct"].mean())
                majs.append(_majority_vote(sg_k, k))
                judgs.append(float(sg_k.loc[sg_k["prob_correct"].idxmax(), "is_correct"]))
                oracs.append(float(sg_k["is_correct"].any()))
            if rands:
                rows.append(dict(
                    dataset=dataset, model_name=model, architecture=arch, K=k,
                    family=_family(model),
                    Random=np.mean(rands), MajVote=np.mean(majs),
                    Judger=np.mean(judgs), Oracle=np.mean(oracs),
                    Delta=np.mean(judgs) - np.mean(rands),
                ))
    return pd.DataFrame(rows)


def compute_early(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for (dataset, model, arch), grp in df.groupby(["dataset", "model_name", "architecture"]):
        sample_groups = {sid: g for sid, g in grp.groupby("sample_id")}
        K_max = grp["run_k"].max()
        for theta in thresholds:
            corrects, runs_used = [], []
            for sg in sample_groups.values():
                sg_s = sg.sort_values("run_k")
                stopped = False
                for _, r in sg_s.iterrows():
                    if r["prob_correct"] >= theta:
                        corrects.append(float(r["is_correct"]))
                        runs_used.append(int(r["run_k"]))
                        stopped = True
                        break
                if not stopped:
                    last = sg_s.iloc[-1]
                    corrects.append(float(last["is_correct"]))
                    runs_used.append(K_max)
            if corrects:
                rows.append(dict(
                    dataset=dataset, model_name=model, architecture=arch,
                    family=_family(model),
                    theta=theta, Accuracy=np.mean(corrects),
                    Avg_Runs=np.mean(runs_used),
                ))
    return pd.DataFrame(rows)


# ── main visualizer class ─────────────────────────────────────────────────────

class EntropyJudgerPlot(BaseVisualizer):
    """Visualize Entropy Judger pass@K results across LLaMA and Qwen models."""

    def __init__(
        self,
        llama_cache: Path | str,
        qwen_cache: Path | str,
        output_dir: Path | str,
        k_values: list[int] | None = None,
        k_heatmap: int = 3,
        thresholds: list[float] | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir, base_font_size=12)
        self.k_values = k_values or [1, 3]
        self.k_heatmap = k_heatmap
        self.thresholds = thresholds or [0.5, 0.6, 0.7, 0.8, 0.9]

        dfs = []
        for path in [llama_cache, qwen_cache]:
            p = Path(path)
            if p.exists():
                dfs.append(pd.read_csv(p))
                print(f"Loaded {len(dfs[-1])} rows from {p}")
            else:
                print(f"[WARN] Not found: {p}")
        if not dfs:
            raise FileNotFoundError("No scored-runs CSV files found.")
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Combined: {len(self.df)} rows, {self.df['model_name'].nunique()} models")

    # ── Figure 1: Best-of-K grouped bar ──────────────────────────────────────

    def _draw_bestk_subplot(
        self,
        ax: plt.Axes,
        agg: pd.DataFrame,
        dataset: str,
        k: int,
        show_title: bool = True,
        show_ylabel: bool = True,
    ) -> None:
        datasets_in = agg[agg["dataset"] == dataset]
        x = np.arange(len(ARCH_ORDER))
        n_methods = len(METHOD_ORDER)
        n_families = 2
        bar_w = 0.75 / (n_methods * n_families)

        for m_idx, method in enumerate(METHOD_ORDER):
            for f_idx, family in enumerate(["llama", "qwen"]):
                color = METHOD_COLORS[method][f_idx]
                offset = (m_idx * n_families + f_idx - (n_methods * n_families - 1) / 2) * bar_w
                vals = []
                for arch in ARCH_ORDER:
                    row = datasets_in[
                        (datasets_in["architecture"] == arch) &
                        (datasets_in["family"] == family)
                    ]
                    vals.append(float(row[method].iloc[0]) if not row.empty else 0.0)
                ax.bar(x + offset, vals, bar_w, color=color,
                       edgecolor="white", linewidth=0.5)

        if show_title:
            ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=10, fontsize=11)
        if show_ylabel:
            ax.set_ylabel(f"K={k}  Accuracy", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [a[:3].capitalize() for a in ARCH_ORDER],
            rotation=30, ha="right", fontsize=8,
        )
        _ax_style(ax)

    def plot_best_of_k(self, filename: str = "judger_best_of_k.png") -> None:
        df_bestk = compute_bestk(self.df, self.k_values)
        agg = (
            df_bestk
            .groupby(["dataset", "architecture", "family", "K"])[METHOD_ORDER]
            .mean()
            .reset_index()
        )

        datasets = [d for d in DATASET_ORDER if d in agg["dataset"].unique()]
        n_cols = len(datasets)
        n_rows = len(self.k_values)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.8 * n_cols, 3.8 * n_rows),
            sharey="row",
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row_idx, k in enumerate(self.k_values):
            sub = agg[agg["K"] == k]
            for col_idx, dataset in enumerate(datasets):
                self._draw_bestk_subplot(
                    axes[row_idx, col_idx], sub, dataset, k,
                    show_title=(row_idx == 0),
                    show_ylabel=(col_idx == 0),
                )

        # Legend
        handles = []
        for method in METHOD_ORDER:
            for f_idx, family in enumerate(["llama", "qwen"]):
                color = METHOD_COLORS[method][f_idx]
                handles.append(mpatches.Patch(
                    color=color,
                    label=f"{method} ({FAMILY_LABEL[family]})",
                ))
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=9)
        fig.suptitle("Best-of-K: Random / MajVote / Judger / Oracle", y=1.01, fontsize=13)
        plt.tight_layout()
        self.save_figure(fig, filename, dpi=200)

    # ── Figure 2: Judger Δ heatmap ────────────────────────────────────────────

    def plot_delta_heatmap(self, filename: str = "judger_delta_heatmap.png") -> None:
        df_bestk = compute_bestk(self.df, [self.k_heatmap])
        agg = (
            df_bestk
            .groupby(["dataset", "architecture", "family"])["Delta"]
            .mean()
            .reset_index()
        )

        datasets_ordered = [d for d in DATASET_ORDER if d in agg["dataset"].unique()]
        archs_ordered = [a for a in ARCH_ORDER if a in agg["architecture"].unique()]

        fig, axes = plt.subplots(
            1, 2,
            figsize=(len(archs_ordered) * 1.6 + 2, len(datasets_ordered) * 0.85 + 1.5),
        )

        for ax, family in zip(axes, ["llama", "qwen"]):
            pivot = (
                agg[agg["family"] == family]
                .pivot(index="dataset", columns="architecture", values="Delta")
                .reindex(index=datasets_ordered, columns=archs_ordered)
            )
            pivot.index = [DATASET_LABELS.get(d, d) for d in pivot.index]

            vals = pivot.values[~np.isnan(pivot.values)]
            vmax = max(abs(vals).max(), 0.01) if len(vals) else 0.1

            sns.heatmap(
                pivot, ax=ax, annot=True, fmt=".3f",
                cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Judger − Random", "shrink": 0.7},
            )
            ax.set_title(f"{FAMILY_LABEL[family]}  (K={self.k_heatmap})", pad=10, fontsize=12)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels(
                [a.capitalize() for a in archs_ordered], rotation=30, ha="right",
            )

        plt.tight_layout()
        self.save_figure(fig, filename, dpi=200)

    # ── Figure 3: Early-Stop accuracy vs cost ─────────────────────────────────

    def plot_early_stop(self, filename: str = "judger_early_stop.png") -> None:
        df_early = compute_early(self.df, self.thresholds)
        agg = (
            df_early
            .groupby(["dataset", "architecture", "family", "theta"])[["Accuracy", "Avg_Runs"]]
            .mean()
            .reset_index()
        )

        datasets = [d for d in DATASET_ORDER if d in agg["dataset"].unique()]
        n_cols = min(3, len(datasets))
        n_rows = (len(datasets) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows))
        axes = np.array(axes).flatten()

        linestyles = {"llama": "-", "qwen": "--"}

        for ax_idx, dataset in enumerate(datasets):
            ax = axes[ax_idx]
            ds = agg[agg["dataset"] == dataset]

            for arch in ARCH_ORDER:
                color = ARCH_COLORS.get(arch, "grey")
                for family in ["llama", "qwen"]:
                    line_data = (
                        ds[(ds["architecture"] == arch) & (ds["family"] == family)]
                        .sort_values("Avg_Runs")
                    )
                    if line_data.empty:
                        continue
                    ax.plot(
                        line_data["Avg_Runs"], line_data["Accuracy"],
                        linestyle=linestyles[family], marker="o",
                        markersize=4, linewidth=1.5, color=color,
                    )

            ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=10)
            ax.set_xlabel("Avg Runs Used")
            ax.set_ylabel("Accuracy")
            _ax_style(ax)

        for ax in axes[len(datasets):]:
            ax.set_visible(False)

        # Legend: arch colour + family linestyle
        arch_handles = [
            plt.Line2D([0], [0], color=ARCH_COLORS.get(a, "grey"),
                       linewidth=2, label=a.capitalize())
            for a in ARCH_ORDER
        ]
        family_handles = [
            plt.Line2D([0], [0], color="black", linewidth=2,
                       linestyle=linestyles[f], label=FAMILY_LABEL[f])
            for f in ["llama", "qwen"]
        ]
        fig.legend(
            handles=arch_handles + family_handles,
            loc="lower center", ncol=len(ARCH_ORDER) + 2,
            frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=9,
        )
        fig.suptitle("Early-Stop: accuracy vs avg runs used", y=1.01, fontsize=13)
        plt.tight_layout()
        self.save_figure(fig, filename, dpi=200)

    # ── compose ───────────────────────────────────────────────────────────────

    def compose(self) -> None:
        print("Plotting Best-of-K bar chart ...")
        self.plot_best_of_k()
        print("Plotting delta heatmap ...")
        self.plot_delta_heatmap()
        print("Plotting Early-Stop curve ...")
        self.plot_early_stop()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Entropy Judger results")
    parser.add_argument("--llama-cache",
                        default="entropy_judger/results_llama/all_runs_scored.csv")
    parser.add_argument("--qwen-cache",
                        default="entropy_judger/results_qwen/all_runs_scored.csv")
    parser.add_argument("--output-dir", default="visualization/outputs/entropy_judger/")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--k-heatmap", type=int, default=3)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.5, 0.6, 0.7, 0.8, 0.9])
    args = parser.parse_args()

    plotter = EntropyJudgerPlot(
        llama_cache=args.llama_cache,
        qwen_cache=args.qwen_cache,
        output_dir=args.output_dir,
        k_values=args.k_values,
        k_heatmap=args.k_heatmap,
        thresholds=args.thresholds,
    )
    plotter.compose()
    print("Done.")


if __name__ == "__main__":
    main()
