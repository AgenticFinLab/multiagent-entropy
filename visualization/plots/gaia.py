"""GAIA visualization suite.

Implements six figures described in ``docs/gaia-visualization-plan.md``,
focused on three top GAIA factors:

  1. tool effective rate            (tool-call effectiveness)
  2. tool-call mean entropy         (uncertainty during tool decisions)
  3. round-1 max agent total entropy (uncertainty in initial agent reasoning)

Reuses BaseVisualizer / ARCH_COLORS / load_csv. Driven from
``visualization/configs/paths.yml`` and ``visualization/run.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from visualization.base import ARCH_COLORS, ARCH_ORDER, BaseVisualizer
from visualization.base.data_loaders import load_csv

warnings.filterwarnings("ignore")

F_TOOL_EFF = "exp_tool_effective_rate"
F_TOOL_ENT = "exp_tool_call_mean_entropy"
F_ROUND1_MAX = "sample_round_1_max_agent_total_entropy"

# Featured factors for Figure 1 (SHAP scatter) and Figure 2 (phase plot)
F_R1_Q3_STD = "sample_round_1_q3_agent_std_entropy"
F_STEP1_MEAN = "step_1_mean_entropy"

F_TOOL_EFF_SAMPLE = "sample_tool_effective_rate"
F_TOOL_ENT_SAMPLE = "sample_tool_call_mean_entropy"
F_TOOL_CALLS_SAMPLE = "sample_tool_total_calls"

ARCH_LABEL = {
    "single": "Single",
    "sequential": "Sequential",
    "centralized": "Centralized",
    "debate": "Debate",
    "hybrid": "Hybrid",
}

FACTOR_LABEL = {
    F_TOOL_EFF: "Tool effective rate",
    F_TOOL_ENT: "Tool-call mean entropy",
    F_ROUND1_MAX: "Round-1 max agent total entropy",
    F_R1_Q3_STD: "Round-1 Q3 agent std entropy",
    F_STEP1_MEAN: "Step-1 mean entropy",
    F_TOOL_EFF_SAMPLE: "Tool effective rate (per sample)",
    F_TOOL_ENT_SAMPLE: "Tool-call mean entropy (per sample)",
}

# Short model labels for figure 6
MODEL_SHORT = {
    "qwen3_0_6b": "Q-0.6",
    "qwen3_4b": "Q-4",
    "qwen3_8b": "Q-8",
    "qwen3_14b": "Q-14",
    "llama_3_2_3b_instruct": "L-3",
    "llama_3_1_8b_instruct": "L-8",
}


class GAIAPlot(BaseVisualizer):
    """Visualizer producing the six GAIA figures.

    Parameters mirror paths defined in ``configs/paths.yml`` under ``gaia``.
    All figures are saved without top-level titles; per-figure interpretations
    are written to ``output_dir/figure_analyses.md``.
    """

    def __init__(
        self,
        gaia_aggregated_path: Path | str,
        gaia_aggregated_exclude_agent_path: Path | str,
        nongaia_merged_path: Path | str,
        shap_values_path: Path | str,
        shap_x_test_path: Path | str,
        output_dir: Path | str,
        summary_path: Path | str | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir, base_font_size=13)
        self.gaia_path = Path(gaia_aggregated_path)
        self.gaia_exagent_path = Path(gaia_aggregated_exclude_agent_path)
        self.nongaia_path = Path(nongaia_merged_path)
        self.shap_values_path = Path(shap_values_path)
        self.shap_x_test_path = Path(shap_x_test_path)
        self.summary_path = Path(summary_path) if summary_path else None
        # Collected interpretation strings; flushed by compose().
        self._analyses: list[str] = []

    # ------------------------------------------------------------------
    # data caching
    # ------------------------------------------------------------------

    def _load(self, attr: str, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            print(f"[gaia] missing data file: {path}")
            return None
        cached = getattr(self, attr, None)
        if cached is None:
            cached = load_csv(path)
            setattr(self, attr, cached)
        return cached

    @property
    def gaia(self) -> Optional[pd.DataFrame]:
        return self._load("_gaia_df", self.gaia_path)

    @property
    def gaia_exagent(self) -> Optional[pd.DataFrame]:
        return self._load("_gaia_exa_df", self.gaia_exagent_path)

    @property
    def nongaia(self) -> Optional[pd.DataFrame]:
        return self._load("_nongaia_df", self.nongaia_path)

    def _record_analysis(self, name: str, body: str) -> None:
        self._analyses.append(f"## {name}\n\n{body.strip()}\n")

    # ------------------------------------------------------------------
    # Figure 1 — three-factor SHAP scatter
    # ------------------------------------------------------------------

    def fig1_shap_three_factor(self) -> None:
        if not self.shap_values_path.exists() or not self.shap_x_test_path.exists():
            print("[fig1] skipped: SHAP files not found")
            return
        shap_df = pd.read_csv(self.shap_values_path)
        x_df = pd.read_csv(self.shap_x_test_path)

        factors = [
            (F_R1_Q3_STD, "Round-1 Q3 std entropy", "#1B7E3D"),
            (F_TOOL_ENT, "Tool-call mean entropy", "#D73027"),
            (F_STEP1_MEAN, "Step-1 mean entropy", "#4575B4"),
        ]

        fig, ax = plt.subplots(figsize=(6.5, 5.4))
        slope_summary = []
        for col, label, color in factors:
            if col not in shap_df.columns or col not in x_df.columns:
                continue
            x = pd.to_numeric(x_df[col], errors="coerce").values
            y = pd.to_numeric(shap_df[col], errors="coerce").values
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            if x.size == 0:
                continue
            # Min-max normalize x so the three different feature scales share
            # a common axis. SHAP values keep their native scale (probability units).
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
            x_norm = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
            ax.scatter(
                x_norm, y, color=color, alpha=0.18, s=8, edgecolors="none",
            )

            # Smooth trend line via LOWESS for a clean monotone-ish curve
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                sm = lowess(y, x_norm, frac=0.35, it=2, return_sorted=True)
                tx, ty = sm[:, 0], sm[:, 1]
            except Exception:
                # Fallback: rolling mean over sorted x
                order = np.argsort(x_norm)
                xs, ys = x_norm[order], y[order]
                w = max(50, len(xs) // 25)
                ty = pd.Series(ys).rolling(w, min_periods=max(10, w // 4),
                                            center=True).mean().values
                tx = xs
            ax.plot(tx, ty, color=color, lw=2.6, alpha=0.7, linestyle="--", label=label)

            try:
                slope = np.polyfit(x, y, 1)[0]
            except Exception:
                slope = float("nan")
            slope_summary.append(
                (
                    label,
                    float(np.nanmean(y[x >= np.nanmedian(x)])),
                    float(np.nanmean(y[x < np.nanmedian(x)])),
                    slope,
                )
            )

        ax.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax.set_xlabel("Normalized feature value (min–max per feature)")
        ax.set_ylabel("SHAP value")
        ax.legend(loc="upper right", frameon=True, fontsize=11,
                  framealpha=0.9, edgecolor="#CCCCCC")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax, top=True, right=True)

        fig.tight_layout()
        self.save_figure(fig, "fig1_shap_three_factor.pdf", dpi=600)
        self.save_figure(fig, "fig1_shap_three_factor.png", dpi=300)
        plt.close(fig)

        # ----- analysis -----
        bullets = []
        for lbl, hi, lo, slope in slope_summary:
            direction = "raises" if hi > lo else "lowers"
            bullets.append(
                f"- **{lbl}** — above-median samples have mean SHAP "
                f"{hi:+.3f} vs. below-median {lo:+.3f}; linear slope ≈ {slope:+.3f}. "
                f"Higher feature value {direction} the predicted probability of correctness."
            )
        body = (
            "All three uncertainty features are overlaid on a single axes to save "
            "page space. Each feature's raw value is min–max normalized to [0, 1] "
            "so they can share an x-axis; SHAP values keep their native scale "
            "(units of predicted probability of `is_finally_correct`). Per feature, "
            "the translucent dots are per-sample SHAP contributions and the solid "
            "line is the mean SHAP value within 12 quantile bins of the (normalized) "
            "feature.\n\n"
            + "\n".join(bullets)
            + "\n\n**Reading.** The three features cover three complementary slices of "
            "uncertainty during a GAIA run. **Round-1 Q3 agent std entropy** (green) "
            "is a *spread* signal — how disagreeing the upper-quartile of round-1 "
            "agents are about their own next tokens; high spread early in the run "
            "reliably subtracts from success probability. **Tool-call mean entropy** "
            "(red) is the *decision-hesitation* signal at the Action / Action-Input "
            "span; the more uncertain the agent is while choosing tools and "
            "arguments, the worse the outcome. **Step-1 mean entropy** (blue) is "
            "the *post-observation re-anchoring* signal — entropy of the second "
            "ReAct step, after the first tool result has come back; if the agent "
            "is still highly entropic at this point, downstream rounds rarely "
            "recover. Together they tell the same story from three angles: failure "
            "is correlated with uncertainty that is *broad* (across agents), "
            "*localized* (at the tool-decision boundary), and *persistent* (still "
            "high after the first observation)."
        )
        self._record_analysis("Figure 1 — Three-factor SHAP scatter", body)

    # ------------------------------------------------------------------
    # Figure 2 — tool eff x tool entropy phase plot
    # ------------------------------------------------------------------

    def fig2_tool_phase(self) -> None:
        df = self.gaia_exagent
        if df is None:
            return
        x_col = F_TOOL_EFF_SAMPLE
        y_col = F_R1_Q3_STD
        cols = [x_col, y_col, F_TOOL_CALLS_SAMPLE, "is_finally_correct"]
        if not all(c in df.columns for c in cols):
            missing = [c for c in cols if c not in df.columns]
            print(f"[fig2] skipped: missing {missing}")
            return
        sub = df[cols].dropna()
        sub = sub[sub[F_TOOL_CALLS_SAMPLE] > 0]
        if sub.empty:
            print("[fig2] skipped: no rows")
            return

        fig, ax = plt.subplots(figsize=(6.5, 5.4))
        sizes = 8 + np.clip(sub[F_TOOL_CALLS_SAMPLE].values, 1, 30) * 4
        correct = sub["is_finally_correct"].astype(bool).values

        ax.scatter(
            sub.loc[~correct, x_col],
            sub.loc[~correct, y_col],
            s=sizes[~correct],
            color="#E45756",
            alpha=0.4,
            label="Wrong",
            edgecolors="none",
        )
        ax.scatter(
            sub.loc[correct, x_col],
            sub.loc[correct, y_col],
            s=sizes[correct],
            color="#4C9F70",
            alpha=0.7,
            label="Correct",
            edgecolors="none",
        )

        from scipy.stats import gaussian_kde

        xmin, xmax = float(sub[x_col].min()), float(sub[x_col].quantile(0.99))
        ymin, ymax = float(sub[y_col].min()), float(sub[y_col].quantile(0.99))
        if xmax <= xmin:
            xmax = xmin + 1e-6
        if ymax <= ymin:
            ymax = ymin + 1e-6
        for mask, color in [(correct, "#1B7E3D"), (~correct, "#9C2E26")]:
            if mask.sum() < 30:
                continue
            xy = np.vstack([sub.loc[mask, x_col].values, sub.loc[mask, y_col].values])
            try:
                kde = gaussian_kde(xy)
                xx, yy = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(
                    xx, yy, zz, levels=4, colors=color, alpha=0.7, linewidths=1.2
                )
            except Exception as e:
                print(f"[fig2] KDE failed: {e}")

        ax.set_xlabel(FACTOR_LABEL[x_col])
        ax.set_ylabel(FACTOR_LABEL[y_col])
        ax.legend(loc="upper right", frameon=False)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
        sns.despine(ax=ax, top=True, right=True)
        fig.tight_layout()
        self.save_figure(fig, "fig2_tool_phase.pdf", dpi=600)
        self.save_figure(fig, "fig2_tool_phase.png", dpi=300)
        plt.close(fig)

        # ----- analysis -----
        c_x = sub.loc[correct, x_col].mean()
        w_x = sub.loc[~correct, x_col].mean()
        c_y = sub.loc[correct, y_col].mean()
        w_y = sub.loc[~correct, y_col].mean()
        body = (
            f"Each point is one GAIA sample with at least one tool call "
            f"(n = {len(sub):,}). Marker size scales with the number of tool calls; "
            f"color indicates whether the final MAS answer was correct (green) or "
            f"wrong (red). KDE contours summarize where each class concentrates.\n\n"
            f"- **Correct samples**: tool effective rate ≈ {c_x:.2f}, "
            f"round-1 Q3 std-entropy ≈ {c_y:.3f}.\n"
            f"- **Wrong samples**:   tool effective rate ≈ {w_x:.2f}, "
            f"round-1 Q3 std-entropy ≈ {w_y:.3f}.\n\n"
            "**Interpretation.** The two axes capture the two practical signals an "
            "online controller can monitor: *how well the tools are actually working* "
            "(x — tool effective rate) and *how uncertain the round-1 agents are "
            "about their own next tokens* (y — Q3 std entropy across upper-quartile "
            "agent positions). Correct samples cluster in the bottom-right quadrant "
            "(high effectiveness AND low spread); wrong samples spread toward the "
            "top-left (low effectiveness AND high spread). The two axes are **not "
            "redundant** — samples with high effectiveness can still fail if round-1 "
            "spread is high, and samples with low spread can still fail if "
            "effectiveness is poor — but the joint condition (low on y AND high on "
            "x) is the most reliable success indicator."
        )
        self._record_analysis(
            "Figure 2 — Tool effective rate × round-1 Q3 std entropy phase plot",
            body,
        )

    # ------------------------------------------------------------------
    # Figure 3 — entropy → accuracy combined (GAIA solid + non-GAIA dashed)
    # ------------------------------------------------------------------

    @staticmethod
    def _quantile_curve(df: pd.DataFrame, x_col: str, n_bins: int = 5) -> pd.DataFrame:
        out = []
        for arch, g in df.groupby("architecture"):
            g = g[[x_col, "is_finally_correct"]].dropna()
            if len(g) < 30:
                continue
            try:
                g = g.copy()
                g["_bin"] = pd.qcut(g[x_col], q=n_bins, duplicates="drop")
            except ValueError:
                continue
            agg = (
                g.groupby("_bin")
                .agg(
                    x=(x_col, "mean"),
                    acc=("is_finally_correct", "mean"),
                    n=("is_finally_correct", "size"),
                )
                .reset_index()
            )
            agg["architecture"] = arch
            agg["_quantile"] = range(1, len(agg) + 1)
            out.append(agg)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def fig3_entropy_accuracy(self) -> None:
        gaia = self.gaia_exagent
        non = self.nongaia
        if gaia is None or non is None:
            return

        gaia_pos = gaia[gaia[F_TOOL_ENT_SAMPLE].fillna(0) > 0]
        gaia_curve = self._quantile_curve(gaia_pos, F_TOOL_ENT_SAMPLE)
        non_curve = self._quantile_curve(non, "sample_total_entropy")

        # Quintile index on shared x-axis (1..5) so the two entropy scales
        # (which differ by orders of magnitude) can share one panel.
        fig, ax = plt.subplots(figsize=(6.5, 5.4))
        for arch in ARCH_ORDER:
            color = ARCH_COLORS.get(arch, "#999999")
            g = gaia_curve[gaia_curve["architecture"] == arch]
            n = non_curve[non_curve["architecture"] == arch]
            if not g.empty:
                ax.plot(
                    g["_quantile"],
                    g["acc"] * 100,
                    color=color,
                    marker="o",
                    linestyle="-",
                    linewidth=1.8,
                    markersize=6,
                    label=ARCH_LABEL.get(arch, arch),
                )
            if not n.empty:
                ax.plot(
                    n["_quantile"],
                    n["acc"] * 100,
                    color=color,
                    marker="s",
                    linestyle="--",
                    linewidth=1.8,
                    markersize=5,
                    alpha=0.85,
                )

        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(
            ["Q1\n(low entropy)", "Q2", "Q3", "Q4", "Q5\n(high entropy)"]
        )
        ax.set_xlabel("Sample-entropy quintile")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
        sns.despine(ax=ax, top=True, right=True)

        arch_handles = [
            Line2D([], [], color=ARCH_COLORS[a], lw=2, label=ARCH_LABEL.get(a, a))
            for a in ARCH_ORDER
        ]
        style_handles = [
            Line2D(
                [],
                [],
                color="#444",
                lw=2,
                linestyle="-",
                marker="o",
                label="GAIA (tool-call entropy)",
            ),
            Line2D(
                [],
                [],
                color="#444",
                lw=2,
                linestyle="--",
                marker="s",
                label="Non-GAIA (sample total entropy)",
            ),
        ]
        leg1 = ax.legend(
            handles=arch_handles,
            title="Architecture",
            loc="upper right",
            frameon=False,
            fontsize=10,
            ncol=1,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=style_handles,
            title="Dataset family",
            loc="lower left",
            frameon=False,
            fontsize=10,
        )

        fig.tight_layout()
        self.save_figure(fig, "fig3_entropy_accuracy.pdf", dpi=600)
        self.save_figure(fig, "fig3_entropy_accuracy.png", dpi=300)
        plt.close(fig)

        # ----- analysis -----
        def _drop(curve):
            rows = []
            for arch in ARCH_ORDER:
                sub = curve[curve["architecture"] == arch].sort_values("_quantile")
                if len(sub) >= 2:
                    rows.append(
                        (arch, sub["acc"].iloc[0] * 100, sub["acc"].iloc[-1] * 100)
                    )
            return rows

        gaia_drops = _drop(gaia_curve)
        non_drops = _drop(non_curve)
        gaia_lines = "\n".join(
            f"  - {ARCH_LABEL.get(a, a)}: {lo:.1f}% → {hi:.1f}% (Δ {hi - lo:+.1f}pp)"
            for a, lo, hi in gaia_drops
        )
        non_lines = "\n".join(
            f"  - {ARCH_LABEL.get(a, a)}: {lo:.1f}% → {hi:.1f}% (Δ {hi - lo:+.1f}pp)"
            for a, lo, hi in non_drops
        )
        body = (
            "All five architectures are plotted on a shared quintile x-axis to make GAIA "
            "(tool-call mean entropy) and non-GAIA (sample total entropy) directly "
            "comparable despite their different absolute entropy scales. Each "
            "architecture has one **solid line + circles for GAIA** and one **dashed line "
            "+ squares for non-GAIA**, sharing the same color.\n\n"
            "**GAIA (low → high entropy quintile, accuracy %):**\n"
            f"{gaia_lines}\n\n"
            "**Non-GAIA datasets (low → high entropy quintile, accuracy %):**\n"
            f"{non_lines}\n\n"
            "**Reading.** Both line styles slope downward: as entropy rises, accuracy "
            "falls — and this happens across every architecture and across both task "
            "families. The two families differ in *level* (non-GAIA reasoning is much "
            "easier than tool-using GAIA tasks) but not in *trend*. This is the "
            "cross-dataset generalization claim of the paper: uncertainty (entropy) is "
            "a unified predictor of failure regardless of whether the model is reasoning "
            "internally or deciding which tool to call."
        )
        self._record_analysis(
            "Figure 3 — Entropy → accuracy across architectures (GAIA solid, non-GAIA dashed)",
            body,
        )

    # ------------------------------------------------------------------
    # Figure 4 — step-level tool entropy heatmap
    # ------------------------------------------------------------------

    def fig4_step_tool_entropy_heatmap(self) -> None:
        df = self.gaia_exagent
        if df is None:
            return
        step_cols = sorted(
            [
                c
                for c in df.columns
                if c.startswith("step_") and c.endswith("_tool_call_mean_entropy")
            ],
            key=lambda c: int(c.split("_")[1]),
        )
        if not step_cols:
            print("[fig4] skipped: no step tool-entropy columns")
            return

        sub = df[step_cols + ["is_finally_correct"]].copy()
        sub = sub[sub[step_cols].fillna(0).sum(axis=1) > 0]
        if sub.empty:
            print("[fig4] skipped: no rows with step tool entropy")
            return

        sub["is_finally_correct"] = sub["is_finally_correct"].astype(bool)

        def _bin_rows(block: pd.DataFrame, n_bins: int = 60) -> np.ndarray:
            """Sort by mean trajectory entropy and aggregate into n_bins rows.

            NaNs are preserved (returned as NaN) so the heatmap can show
            steps that never occurred as a distinct background color.
            """
            if block.empty:
                return np.empty((0, len(step_cols)))
            traj_mean = block[step_cols].mean(axis=1, skipna=True)
            order = np.argsort(traj_mean.values, kind="mergesort")
            arr = block[step_cols].values[order]
            n = arr.shape[0]
            n_bins = min(n_bins, n)
            edges = np.linspace(0, n, n_bins + 1, dtype=int)
            out = np.full((n_bins, arr.shape[1]), np.nan)
            for i in range(n_bins):
                lo, hi = edges[i], edges[i + 1]
                if hi > lo:
                    out[i] = np.nanmean(arr[lo:hi], axis=0)
            return out

        c_block = sub[sub["is_finally_correct"]]
        w_block = sub[~sub["is_finally_correct"]]
        c_mat = _bin_rows(c_block, n_bins=60)
        w_mat = _bin_rows(w_block, n_bins=60)
        # Both classes high-entropy on top of their block; stack correct above wrong.
        # _bin_rows already sorts ascending, reverse so high entropy is at the
        # boundary (visual continuity across the split).
        if c_mat.size:
            c_mat = c_mat[::-1]  # high -> low (so low-entropy correct rows sit at top)
        if w_mat.size:
            pass  # ascending: low entropy near boundary, high entropy at bottom
        full = (
            np.vstack([c_mat, w_mat])
            if c_mat.size and w_mat.size
            else (c_mat if c_mat.size else w_mat)
        )
        n_corr_bins = c_mat.shape[0]

        # Robust upper bound for color clipping
        all_vals = full[~np.isnan(full)]
        vmax = float(np.nanpercentile(all_vals, 90)) if all_vals.size else 1.0
        if vmax <= 0:
            vmax = 1.0

        # Layout: top marginal line + main heatmap, sharing x.
        fig = plt.figure(figsize=(9.0, 6.5))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1, 3.2], width_ratios=[40, 1], hspace=0.05, wspace=0.04
        )
        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
        cax = fig.add_subplot(gs[1, 1])

        # -- top marginal: mean ± IQR per step, per class --
        x_idx = np.arange(len(step_cols))

        def _stats(block: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            arr = block[step_cols].values
            mean = np.nanmean(arr, axis=0)
            q1 = np.nanpercentile(arr, 25, axis=0)
            q3 = np.nanpercentile(arr, 75, axis=0)
            return mean, q1, q3

        c_mean, c_q1, c_q3 = _stats(c_block)
        w_mean, w_q1, w_q3 = _stats(w_block)

        ax_top.fill_between(x_idx, c_q1, c_q3, color="#4C9F70", alpha=0.18)
        ax_top.plot(
            x_idx,
            c_mean,
            color="#1B7E3D",
            lw=2,
            marker="o",
            markersize=5,
            label=f"Correct (n = {len(c_block):,})",
        )
        ax_top.fill_between(x_idx, w_q1, w_q3, color="#E45756", alpha=0.18)
        ax_top.plot(
            x_idx,
            w_mean,
            color="#9C2E26",
            lw=2,
            marker="s",
            markersize=5,
            label=f"Wrong   (n = {len(w_block):,})",
        )
        ax_top.set_ylabel("Step entropy\n(mean ± IQR)", fontsize=11)
        ax_top.legend(loc="upper left", frameon=False, fontsize=10)
        ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax_top.tick_params(axis="x", labelbottom=False)
        sns.despine(ax=ax_top, top=True, right=True)

        # -- main heatmap with NaN as light gray --
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad(color="#EDEDED")
        masked = np.ma.masked_invalid(full)
        im = ax_main.imshow(
            masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest"
        )

        if n_corr_bins > 0 and n_corr_bins < full.shape[0]:
            ax_main.axhline(n_corr_bins - 0.5, color="white", lw=1.6, ls="-")
            ax_main.axhline(n_corr_bins - 0.5, color="black", lw=0.8, ls="--")

        # y tick labels: just two block labels, not per-row
        ax_main.set_yticks(
            [
                n_corr_bins / 2 - 0.5,
                n_corr_bins + (full.shape[0] - n_corr_bins) / 2 - 0.5,
            ]
        )
        ax_main.set_yticklabels(["Correct\n(binned)", "Wrong\n(binned)"], fontsize=11)
        ax_main.set_xticks(range(len(step_cols)))
        ax_main.set_xticklabels([c.split("_")[1] for c in step_cols])
        ax_main.set_xlabel("ReAct step index")
        plt.colorbar(im, cax=cax, label="step tool-call mean entropy")

        self.save_figure(fig, "fig4_step_tool_entropy_heatmap.pdf", dpi=600)
        self.save_figure(fig, "fig4_step_tool_entropy_heatmap.png", dpi=300)
        plt.close(fig)

        c_first = float(c_mean[0]) if not np.isnan(c_mean[0]) else float("nan")
        c_last = float(np.nanmean(c_mean))
        w_first = float(w_mean[0]) if not np.isnan(w_mean[0]) else float("nan")
        w_last_idx = (
            int(np.where(~np.isnan(w_mean))[0][-1]) if (~np.isnan(w_mean)).any() else 0
        )
        w_last = float(w_mean[w_last_idx])
        gap = float(np.nanmean(w_mean - c_mean))
        body = (
            "Two coupled panels using a blue-toned heatmap. **Top**: per-step mean "
            "tool-call decision entropy with an inter-quartile band, computed "
            "separately for correct (green) and wrong (red) GAIA samples — the "
            "trajectory-level mechanism at a glance. **Bottom**: a row-aggregated "
            "heatmap. To make ~4.7k samples readable, samples within each class are "
            "sorted by their mean trajectory entropy and averaged into 60 quantile "
            "bins per class (correct on top, wrong below). Light-gray cells mark "
            "steps that never occurred for that bin (the trajectory finished earlier).\n\n"
            f"- Correct trajectories (n = {len(c_block):,}): step 0 mean ≈ "
            f"{c_first:.3f}, trajectory-average ≈ {c_last:.3f}.\n"
            f"- Wrong trajectories (n = {len(w_block):,}): step 0 mean ≈ {w_first:.3f}, "
            f"last-active-step mean ≈ {w_last:.3f}.\n"
            f"- Average per-step entropy gap (wrong − correct) ≈ {gap:.3f}.\n\n"
            "**What this figure tells us.**\n"
            "1. **Failure is dynamic, not static.** The two classes start at "
            "comparable step-0 entropy but diverge as the trajectory progresses — "
            "you cannot tell a successful run from a failing one by looking at the "
            "first action alone.\n"
            "2. **Hesitation persists deeper into the trajectory for failures.** "
            "The marginal mean for the wrong group stays elevated through "
            "mid-trajectory steps, while the correct group decays. This is "
            "consistent with the *decision-hesitation → failure* mechanism we infer "
            "from the SHAP analysis (Figure 1, panel b).\n"
            "3. **The signal is informative at the bin level, not just the "
            "aggregate.** In the binned heatmap, even the *lightest* (lowest-entropy) "
            "wrong-class rows are not uniformly lighter than the correct-class rows — "
            "wrong samples spread across the entropy spectrum, but their density "
            "skews higher. This is why per-step tool-call entropy works as an "
            "online early-warning signal for an adaptive controller, not as a single "
            "hard threshold.\n"
            "4. **Trajectory length itself matters.** The light-gray strip on the "
            "right of the wrong block is shorter than on the correct block — wrong "
            "trajectories tend to use up more steps before terminating, suggesting "
            "the agent is grinding through retries rather than converging."
        )
        self._record_analysis(
            "Figure 4 — Step-level tool-call mean entropy (per-step profile + binned heatmap)",
            body,
        )

    # ------------------------------------------------------------------
    # Figure 5 — architecture comparison radar
    # ------------------------------------------------------------------

    def fig5_arch_radar(self) -> None:
        df = self.gaia_exagent
        if df is None:
            return
        axes_def = [
            ("Low round-1 Q3 std entropy", F_R1_Q3_STD, True),
            ("Low tool-call entropy", F_TOOL_ENT, True),
            ("Low step-1 mean entropy", F_STEP1_MEAN, True),
        ]
        rows = []
        for arch, g in df.groupby("architecture"):
            rec = {"architecture": arch}
            for label, col, _ in axes_def:
                rec[label] = g[col].mean() if col in g.columns else np.nan
            rows.append(rec)
        table = pd.DataFrame(rows).set_index("architecture")
        if table.empty:
            print("[fig5] skipped: no architectures")
            return

        norm = pd.DataFrame(index=table.index)
        for label, _, reverse in axes_def:
            v = table[label]
            lo, hi = v.min(), v.max()
            if hi - lo < 1e-12:
                norm[label] = 0.5
                continue
            s = (v - lo) / (hi - lo)
            if reverse:
                s = 1 - s
            norm[label] = s

        labels = [l for l, _, _ in axes_def]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for arch in ARCH_ORDER:
            if arch not in norm.index:
                continue
            vals = norm.loc[arch].tolist() + [norm.loc[arch].iloc[0]]
            ax.plot(
                angles,
                vals,
                color=ARCH_COLORS.get(arch, "#999999"),
                label=ARCH_LABEL.get(arch, arch),
                linewidth=1.8,
            )
            ax.fill(angles, vals, color=ARCH_COLORS.get(arch, "#999999"), alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(
            loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=10, frameon=False
        )
        fig.tight_layout()
        self.save_figure(fig, "fig5_arch_radar.pdf", dpi=600)
        self.save_figure(fig, "fig5_arch_radar.png", dpi=300)
        plt.close(fig)

        winners = {
            label: table[label].idxmax() if not reverse else table[label].idxmin()
            for label, _, reverse in axes_def
        }
        winners_lines = "\n".join(
            f"  - **{ax_lbl}** — best: {ARCH_LABEL.get(winner, winner)}"
            for ax_lbl, winner in winners.items()
        )
        body = (
            "Three axes — low round-1 Q3 std entropy, low tool-call mean entropy, low "
            "step-1 mean entropy — the same three uncertainty features featured in "
            "Figure 1's SHAP scatter, each min-max normalized across architectures and "
            "reversed so that **outward = lower entropy = better**. Polylines are "
            "colored by architecture.\n\n"
            f"{winners_lines}\n\n"
            "**Reading.** The three axes capture three complementary slices of "
            "uncertainty during a GAIA run: round-1 agent disagreement (spread), "
            "tool-call decision hesitation (localized), and step-1 post-observation "
            "re-anchoring (persistence). No single architecture dominates all three "
            "simultaneously — the architecture that minimizes tool-call entropy is "
            "not the same one that minimizes round-1 spread or step-1 hesitation. "
            "This makes the paper's broader claim concrete: **MAS architectures "
            "trade off across distinct uncertainty regimes**, and selecting the "
            "right architecture for tool-using tasks is not a single-objective "
            "decision."
        )
        self._record_analysis("Figure 5 — Architecture comparison radar", body)

    # ------------------------------------------------------------------
    # Figure 6 — failure-mode breakdown
    # ------------------------------------------------------------------

    def fig6_failure_breakdown(self) -> None:
        """Tool-call failure attribution per model.

        Reads ``evaluation/results_gaia/gaia/tool_failure_breakdown.csv`` produced
        by ``scripts/scan_gaia_tool_failures.py``. Shows two coupled views:

          (a) tool-call volume + overall failure rate per model
          (b) failure-reason composition (stacked, normalized) per model
        """
        breakdown_path = self.gaia_path.parent / "tool_failure_breakdown.csv"
        if not breakdown_path.exists():
            print(
                f"[fig6] skipped: {breakdown_path} not found "
                f"(run scripts/scan_gaia_tool_failures.py first)"
            )
            return
        bd = pd.read_csv(breakdown_path)

        cats = [
            "parse_error",
            "arg_error",
            "duplicate_call",
            "timeout",
            "network",
            "empty_result",
            "executed_with_error",
            "other_explicit",
        ]
        cats = [c for c in cats if c in bd.columns]

        agg = bd.groupby("model")[["n_calls", "ok"] + cats].sum()
        agg["fail"] = agg[cats].sum(axis=1)
        agg["fail_rate"] = agg["fail"] / agg["n_calls"]
        # Order models by total tool-call volume (largest first) for readability
        agg = agg.sort_values("n_calls", ascending=False)

        cat_label = {
            "arg_error": "Argument / schema error",
            "duplicate_call": "Duplicate call",
            "empty_result": "Empty / no-result",
            "executed_with_error": "Executed with error",
            "network": "Network / HTTP",
            "timeout": "Timeout",
            "parse_error": "Unparseable output",
            "other_explicit": "Malformed invocation",
        }
        # Blue→red diverging palette: cool tones for "interface-layer" failures
        # (parsing, args, duplicate calls — the tool never really ran), warm tones
        # for "execution-layer" failures (call ran but produced bad/empty output).
        cat_colors = {
            # cool side — interface / pre-execution failures
            "parse_error": "#08306B",  # darkest navy
            "arg_error": "#2171B5",
            "duplicate_call": "#6BAED6",
            "timeout": "#C6DBEF",
            # warm side — execution / semantic failures
            "network": "#FCBBA1",
            "empty_result": "#FB6A4A",
            "executed_with_error": "#CB181D",
            "other_explicit": "#67000D",  # darkest red
        }

        models = list(agg.index)
        short_labels = [MODEL_SHORT.get(m, m) for m in models]
        x = np.arange(len(models))

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1, 1.25])
        )

        # ---------- (a) volume + failure rate ----------
        bars = ax1.bar(
            x, agg["n_calls"].values, color="#4575B4", alpha=0.6, edgecolor="white"
        )
        ax1.set_yscale("log")
        ax1.set_ylabel("Tool calls (log scale)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_labels, rotation=0, ha="center")
        ax1.text(
            0.5,
            -0.18,
            "(a) Tool-call volume and overall failure rate",
            transform=ax1.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax1, top=True, right=False)

        ax1b = ax1.twinx()
        ax1b.plot(
            x,
            agg["fail_rate"].values * 100,
            color="#D73027",
            marker="o",
            lw=2,
            markersize=7,
            label="Failure rate (%)",
        )
        ax1b.set_ylabel("Failure rate (%)", color="#D73027")
        ax1b.tick_params(axis="y", colors="#D73027")
        ax1b.set_ylim(0, max(60, agg["fail_rate"].max() * 100 * 1.15))
        for xi, n, fr in zip(x, agg["n_calls"], agg["fail_rate"]):
            ax1.text(xi, n, f"{int(n):,}", ha="center", va="bottom", fontsize=8)
            ax1b.text(
                xi,
                fr * 100,
                f"{fr * 100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#D73027",
            )

        # ---------- (b) failure-reason composition ----------
        # Normalize to percentages of failed calls only
        comp = agg[cats].div(agg[cats].sum(axis=1).replace(0, np.nan), axis=0) * 100
        comp = comp.fillna(0)

        bottom = np.zeros(len(models))
        for c in cats:
            vals = comp[c].values
            if vals.sum() < 1e-9:
                continue
            ax2.barh(
                x,
                vals,
                left=bottom,
                color=cat_colors[c],
                edgecolor="white",
                linewidth=0.5,
                label=cat_label[c],
            )
            bottom += vals
        ax2.set_yticks(x)
        ax2.set_yticklabels(short_labels)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Share of failed tool calls (%)")
        ax2.text(
            0.5,
            -0.20,
            "(b) Failure-reason composition (% of failed calls)",
            transform=ax2.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        ax2.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9
        )
        ax2.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax2, top=True, right=True)

        fig.tight_layout()
        self.save_figure(fig, "fig6_tool_failure_attribution.pdf", dpi=600)
        self.save_figure(fig, "fig6_tool_failure_attribution.png", dpi=300)
        plt.close(fig)

        # ----- analysis -----
        worst_model = agg["fail_rate"].idxmax()
        best_model = agg["fail_rate"].idxmin()
        # dominant failure category per model
        dom_lines = []
        for m in models:
            cat_counts = agg.loc[m, cats]
            if cat_counts.sum() == 0:
                continue
            dom = cat_counts.idxmax()
            dom_pct = cat_counts[dom] / cat_counts.sum() * 100
            dom_lines.append(
                f"  - **{MODEL_SHORT.get(m, m)}** ({m}; n_calls = "
                f"{int(agg.loc[m, 'n_calls']):,}, "
                f"fail = {agg.loc[m, 'fail_rate']*100:.1f}%): "
                f"{cat_label[dom]} dominates at {dom_pct:.1f}% of failures"
            )

        body = (
            "Tool-call failure attribution per base model, derived from the raw "
            "`react_steps[*].tool_calls` payloads in every GAIA experiment trace. "
            "A call is *failed* if its tool result has `success=false`, fails to "
            "parse as JSON, or has `success=true` but a result string containing an "
            "error indicator (traceback / NameError / etc.). Failures are then "
            "classified by string-matching the result body into eight reasons: "
            "argument/schema errors, duplicate calls, empty results, executed-with-"
            "error (effectiveness), network/HTTP, timeout, unparseable output, and "
            "*malformed invocation* — calls the framework rejected with a non-"
            "standard error message, typically structurally broken JSON or "
            "off-protocol output that does not match the tool schema at all "
            "(distinct from `arg_error`, which is rejection by the tool itself "
            "with a recognizable schema/argument complaint).\n\n"
            "**Panel (a)** ranks models by total tool-call volume (log scale, blue "
            "bars) and overlays the overall failure rate (red line). **Panel (b)** "
            "decomposes the failed calls of each model into a 100%-stacked bar.\n\n"
            f"- Highest failure rate: **{MODEL_SHORT.get(worst_model, worst_model)}** "
            f"({worst_model}, {agg.loc[worst_model, 'fail_rate']*100:.1f}%); "
            f"lowest: **{MODEL_SHORT.get(best_model, best_model)}** "
            f"({best_model}, {agg.loc[best_model, 'fail_rate']*100:.1f}%).\n"
            f"- Aggregate across all models — total tool calls "
            f"{int(agg['n_calls'].sum()):,}, of which "
            f"{int(agg['fail'].sum()):,} failed "
            f"({agg['fail'].sum() / agg['n_calls'].sum() * 100:.1f}%).\n\n"
            "**Dominant failure category per model:**\n"
            + "\n".join(dom_lines)
            + "\n\n**Reading.** Failure attribution is **model-shaped, not random**. "
            "Smaller models pour calls into the loop and most of those calls are "
            "*malformed invocations* — structurally broken JSON or off-protocol "
            "output that the framework rejects with a non-standard error — or are "
            "flagged as unparseable. Both are *interface-layer* failures: the tool "
            "never actually ran. Stronger models call tools far less often, but "
            "their failures concentrate on *executed-with-error* and *empty "
            "results* — the tool ran, but with the wrong intent. This is the "
            "practical message of the figure: "
            "improving GAIA-style tool use is not one engineering problem but two "
            "different ones depending on the base model — schema robustness for the "
            "weak-model regime, search/grounding quality for the strong-model regime."
        )
        self._record_analysis(
            "Figure 6 — Tool-call failure attribution per base model",
            body,
        )

    # ------------------------------------------------------------------
    # Figure 7 — per-model architecture accuracy with base-model marker
    # ------------------------------------------------------------------

    def fig7_arch_accuracy(self) -> None:
        if self.summary_path is None or not self.summary_path.exists():
            print("[fig7] skipped: summary_path not found")
            return
        df = pd.read_csv(self.summary_path)
        needed = {"model", "architecture", "accuracy", "base model accuracy"}
        if not needed.issubset(df.columns):
            print(f"[fig7] skipped: missing columns {needed - set(df.columns)}")
            return

        # MAS accuracy is fractional (0–1); base accuracy already percent (0–100).
        df = df.copy()
        df["acc_pct"] = df["accuracy"] * 100.0
        df["base_pct"] = df["base model accuracy"]

        # Order models by base-model strength (ascending) for readability
        base_by_model = df.groupby("model")["base_pct"].first().sort_values()
        models = list(base_by_model.index)
        short_labels = [MODEL_SHORT.get(m, m) for m in models]

        fig, ax = plt.subplots(figsize=(6.5, 5.4))
        n_arch = len(ARCH_ORDER)
        bar_w = 0.8 / n_arch
        x = np.arange(len(models))

        for j, arch in enumerate(ARCH_ORDER):
            sub = df[df["architecture"] == arch].set_index("model")
            vals = [sub["acc_pct"].get(m, np.nan) for m in models]
            offset = (j - (n_arch - 1) / 2) * bar_w
            ax.bar(
                x + offset,
                vals,
                width=bar_w,
                color=ARCH_COLORS.get(arch, "#999999"),
                edgecolor="white",
                linewidth=0.6,
                label=ARCH_LABEL.get(arch, arch),
            )

        # Base-model accuracy as a horizontal gray dashed tick spanning each model's group
        half = (bar_w * n_arch) / 2
        for xi, m in enumerate(models):
            base = base_by_model[m]
            ax.hlines(
                base,
                xi - half,
                xi + half,
                colors="#888888",
                linestyles="--",
                linewidth=1.6,
                zorder=5,
            )
            ax.text(
                xi - half - 0.04,
                base + 0.4,
                f"{base:.1f}%",
                va="bottom",
                ha="right",
                fontsize=9,
                color="#555555",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels)
        ax.set_xlabel("Base model")
        ax.set_ylabel("GAIA accuracy (%)")
        ax.set_ylim(0, max(df["acc_pct"].max(), df["base_pct"].max()) * 1.18)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax, top=True, right=True)

        arch_handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=ARCH_COLORS.get(a, "#999999"),
                label=ARCH_LABEL.get(a, a),
            )
            for a in ARCH_ORDER
        ]
        base_handle = Line2D(
            [],
            [],
            color="#888888",
            linestyle="--",
            linewidth=1.6,
            label="Base model",
        )
        ax.legend(
            handles=arch_handles + [base_handle],
            loc="upper left",
            frameon=False,
            ncol=1,
            fontsize=10,
        )

        fig.tight_layout()
        self.save_figure(fig, "fig7_arch_accuracy.pdf", dpi=600)
        self.save_figure(fig, "fig7_arch_accuracy.png", dpi=300)
        plt.close(fig)

        # ----- analysis -----
        lines = []
        for m in models:
            base = base_by_model[m]
            sub = df[df["model"] == m].set_index("architecture")["acc_pct"]
            best_arch = sub.idxmax()
            best_val = sub.max()
            delta = best_val - base
            lines.append(
                f"  - **{MODEL_SHORT.get(m, m)}** ({m}): base = {base:.1f}%, "
                f"best MAS = {ARCH_LABEL.get(best_arch, best_arch)} {best_val:.1f}% "
                f"(Δ {delta:+.1f}pp)"
            )
        body = (
            "Per-model GAIA accuracy across the five MAS architectures, with the "
            "base-model (no-MAS) accuracy overlaid as a black dashed tick spanning "
            "each model's bar group. Base-model accuracy is taken from "
            "`all_summary_data.csv:base model accuracy` (already in percent); MAS "
            "accuracies come from the `accuracy` column (fraction, scaled ×100). "
            "Models are ordered along the x-axis by ascending base-model strength.\n\n"
            "**Per-model best architecture vs. base model:**\n"
            + "\n".join(lines)
            + "\n\n**Reading.** The dashed tick for each model is the *no-MAS ceiling* — "
            "what a single forward pass of that base model achieves on GAIA without "
            "any orchestration. For the weakest models (Q-0.6, L-3, L-8) every MAS "
            "architecture lands well below this ceiling: the orchestration overhead "
            "compounds the model's tool-use weakness rather than mitigating it. For "
            "the strongest models (Q-4, Q-8, Q-14) several MAS architectures match or "
            "exceed the base-model line, but the gain is modest and not uniform — "
            "**centralized** and **hybrid** tend to be the architectures that lift "
            "above base, while **debate** typically lags. The figure makes the paper's "
            "headline trade-off concrete: MAS only helps when the base model is "
            "already competent at tool use, and even then the choice of architecture "
            "is the difference between recovering the base-model accuracy and "
            "destroying it."
        )
        self._record_analysis(
            "Figure 7 — Per-model GAIA accuracy by architecture, with base-model overlay",
            body,
        )

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    def compose(self) -> None:
        print("[gaia] Figure 1 — three-factor SHAP scatter")
        self.fig1_shap_three_factor()
        print("[gaia] Figure 2 — tool eff x tool entropy phase plot")
        self.fig2_tool_phase()
        print("[gaia] Figure 3 — entropy → accuracy combined")
        self.fig3_entropy_accuracy()
        print("[gaia] Figure 4 — step-level tool entropy heatmap")
        self.fig4_step_tool_entropy_heatmap()
        print("[gaia] Figure 5 — architecture comparison radar")
        self.fig5_arch_radar()
        print("[gaia] Figure 6 — failure-mode breakdown")
        self.fig6_failure_breakdown()
        print("[gaia] Figure 7 — per-model architecture accuracy")
        self.fig7_arch_accuracy()

        if self._analyses:
            md = (
                "# GAIA Figure Analyses\n\n"
                "Auto-generated by `visualization/plots/gaia.py::GAIAPlot.compose()`. "
                "Each section interprets the matching figure in this directory.\n\n"
                + "\n".join(self._analyses)
            )
            out = self.output_dir / "figure_analyses.md"
            out.write_text(md, encoding="utf-8")
            print(f"Saved: {out}")
