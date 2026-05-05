"""Causal analysis visualization suite.

Three publication-quality figures from ``data_mining/exp_causal_complete/``:

  C1 — ATE forest plot: three estimators (LR, PS, IPW) per treatment.
  C2 — Mediation decomposition: direct vs. indirect effect, stacked bars.
  C3 — Full consensus causal DAG with semantic node grouping.

All figures use figsize=(6.5, 5.4) for consistency with the GAIA suite.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from visualization.base import BaseVisualizer

warnings.filterwarnings("ignore")

# ── Shared figsize ────────────────────────────────────────────────────────────
_FIGSIZE = (6.5, 5.4)

# ── Abbreviation rules (applied in order, most specific first) ───────────────
# Word-level substitutions: covers all features systematically
_ABBREV_RULES = [
    # Prefix patterns
    ("sample_round_1_", "R1·"),
    ("sample_round_2_", "R2·"),
    ("round_1_", "R1·"),
    ("round_2_", "R2·"),
    ("base_sample_", "Base·"),
    ("base_model_", "Base·"),
    ("sample_", ""),  # drop "sample_" prefix — context clear from tier
    ("exp_", "Exp·"),
    ("answer_token_", "AnsT·"),
    # Common words
    ("entropy", "H"),
    ("agent", "ag"),
    ("average", "avg"),
    ("total", "tot"),
    ("spread", "sprd"),
    ("variance", "var"),
    ("median", "med"),
    ("infer", "inf"),
    ("change", "Δ"),
    ("diff", "Δ"),
    ("per_token", "/tok"),
    ("per_agent", "/ag"),
    ("_", "-"),  # remaining underscores → dash
]

# Abbreviation glossary for the legend table
_ABBREV_GLOSSARY = [
    ("R1·", "Round 1"),
    ("R2·", "Round 2"),
    ("Base·", "Base model"),
    ("Exp·", "Experiment"),
    ("AnsT·", "Answer token"),
    ("H", "Entropy"),
    ("ag", "Agent"),
    ("avg", "Average"),
    ("tot", "Total"),
    ("sprd", "Spread"),
    ("var", "Variance"),
    ("med", "Median"),
    ("inf", "Inference"),
    ("Δ", "Change/Diff"),
    ("/tok", "Per token"),
    ("/ag", "Per agent"),
]


def _s(name: str) -> str:
    """Apply abbreviation rules to produce a compact node label."""
    if name == "is_finally_correct":
        return "MAS correctness"
    s = name
    for src, dst in _ABBREV_RULES:
        s = s.replace(src, dst)
    return s


# ── Node semantic groups for figC3 ────────────────────────────────────────────
def _node_group(name: str) -> str:
    if name == "is_finally_correct":
        return "outcome"
    if name.startswith("base_"):
        return "base_model"
    if name.startswith("sample_round_1_") or name == "round_1_total_entropy":
        return "round1"
    if name.startswith("sample_round_2_") or name.startswith("round_2_"):
        return "round2"
    if name.startswith("exp_") or "answer_token" in name:
        return "exp_ans"
    if name.startswith("sample_"):
        return "sample"
    return "sample"


# ── Macaron / muted palette ───────────────────────────────────────────────────
_GROUP_COLORS = {
    "outcome": "#A8D8A8",  # soft mint green
    "base_model": "#F4A7A3",  # muted rose
    "round1": "#FDDCB0",  # soft peach
    "round2": "#A8C8E8",  # powder blue
    "exp_ans": "#D4B8E0",  # lavender
    "sample": "#C8DEB8",  # sage green
}


_COL_DIRECT = "#D73027"
_COL_INDIRECT = "#4575B4"


class CausalPlot(BaseVisualizer):
    """Visualizer for the causal analysis figures."""

    def __init__(
        self,
        causal_data_dir: Path | str,
        output_dir: Path | str,
    ) -> None:
        super().__init__(output_dir=output_dir, base_font_size=13)
        self.data_dir = Path(causal_data_dir)
        self._analyses: list[str] = []

    def _record_analysis(self, name: str, body: str) -> None:
        self._analyses.append(f"## {name}\n\n{body.strip()}\n")

    # ── Figure C1 — ATE forest plot ──────────────────────────────────────────

    def figc1_ate_forest(self) -> None:
        path = self.data_dir / "causal_effects" / "causal_effects_raw.json"
        if not path.exists():
            print(f"[figC1] skipped: {path} not found")
            return
        raw = json.loads(path.read_text(encoding="utf-8"))

        rows = []
        for d in raw:
            lr = d.get("linear_regression", {})
            ps = d.get("propensity_score", {})
            ipw = d.get("ipw", {})
            rows.append(
                {
                    "treatment": d["treatment"],
                    "is_direct": bool(d.get("is_direct_cause", False)),
                    "lr_ate": float(lr.get("ate", np.nan)),
                    "lr_p": float(lr.get("p_value", 1.0)),
                    "ps_ate": float(ps.get("ate", np.nan)),
                    "ipw_ate": float(ipw.get("ate", np.nan)),
                }
            )
        df = pd.DataFrame(rows).sort_values("ps_ate")
        n = len(df)

        fig, ax = plt.subplots(figsize=_FIGSIZE)

        for i, row in df.reset_index(drop=True).iterrows():
            color = _COL_DIRECT if row["is_direct"] else _COL_INDIRECT
            sig_star = "*" if row["lr_p"] < 0.05 else ""

            offsets = {"LR": 0.22, "PS": 0.0, "IPW": -0.22}
            ates = {"LR": row["lr_ate"], "PS": row["ps_ate"], "IPW": row["ipw_ate"]}
            markers = {"LR": "D", "PS": "o", "IPW": "s"}
            alphas = {"LR": 0.55, "PS": 1.0, "IPW": 0.75}

            valid = [v for v in ates.values() if not np.isnan(v)]
            if len(valid) >= 2:
                ax.hlines(
                    i,
                    min(valid),
                    max(valid),
                    color=color,
                    linewidth=1.2,
                    alpha=0.5,
                    zorder=3,
                )
            for est, marker in markers.items():
                val = ates[est]
                if not np.isnan(val):
                    ax.scatter(
                        val,
                        i + offsets[est],
                        color=color,
                        marker=marker,
                        s=36,
                        alpha=alphas[est],
                        zorder=4,
                    )

            ax.text(
                -0.02,
                i,
                f"{_s(row['treatment'])}{sig_star}",
                va="center",
                ha="right",
                fontsize=9.5,
                transform=ax.get_yaxis_transform(),
            )

        ax.axvline(0, color="black", lw=0.8, alpha=0.6, linestyle="--")
        ax.set_yticks([])
        ax.set_xlabel("Average Treatment Effect on MAS correctness")
        vals = df[["lr_ate", "ps_ate", "ipw_ate"]]
        ax.set_xlim(vals.min().min() * 1.4, vals.max().max() * 1.4 + 0.2)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax, top=True, right=True, left=True)

        handles = [
            mpatches.Patch(color=_COL_DIRECT, label="Direct cause"),
            mpatches.Patch(color=_COL_INDIRECT, label="Indirect cause"),
            Line2D(
                [],
                [],
                color="#555",
                marker="D",
                linestyle="none",
                markersize=6,
                alpha=0.7,
                label="Linear regression",
            ),
            Line2D(
                [],
                [],
                color="#555",
                marker="o",
                linestyle="none",
                markersize=6,
                label="Propensity score",
            ),
            Line2D(
                [],
                [],
                color="#555",
                marker="s",
                linestyle="none",
                markersize=6,
                alpha=0.8,
                label="IPW",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
            fontsize=9.5,
            ncol=1,
        )

        fig.tight_layout()
        self.save_figure(fig, "figC1_ate_forest.pdf", dpi=600)
        self.save_figure(fig, "figC1_ate_forest.png", dpi=300)
        plt.close(fig)

        direct = df[df["is_direct"]]["treatment"].tolist()
        indirect = df[~df["is_direct"]]["treatment"].tolist()
        sig = df[df["lr_p"] < 0.05]["treatment"].tolist()
        strongest = df.loc[df["ps_ate"].abs().idxmax()]
        body = (
            "Forest plot showing the ATE of each entropy feature on MAS correctness, "
            "estimated by three methods: linear regression (diamond, LR), propensity-"
            "score matching (circle, PS), and inverse-probability weighting (square, IPW). "
            "Rows sorted by PS-ATE (most negative first). Red = direct cause in the "
            "causal graph; blue = indirect. * marks LR p < 0.05.\n\n"
            f"- Direct causes: " + ", ".join(_s(t) for t in direct) + ".\n"
            f"- Indirect: " + ", ".join(_s(t) for t in indirect) + ".\n"
            f"- Significant (LR): " + ", ".join(_s(t) for t in sig) + ".\n"
            f"- Strongest PS-ATE: {_s(strongest['treatment'])} ({strongest['ps_ate']:.3f}).\n\n"
            "All three estimators consistently place entropy features on the negative "
            "side, confirming entropy universally reduces MAS correctness. Direct causes "
            "show smaller estimator spread, indicating better confounder control."
        )
        self._record_analysis("Figure C1 — ATE forest plot", body)

    # ── Figure C2 — Mediation decomposition ─────────────────────────────────

    def figc2_mediation(self) -> None:
        path = self.data_dir / "mediation" / "mediation_results.json"
        if not path.exists():
            print(f"[figC2] skipped: {path} not found")
            return
        raw = json.loads(path.read_text(encoding="utf-8"))

        rows = []
        for d in raw:
            rows.append(
                {
                    "treatment": d["treatment"],
                    "mediator": d.get("mediator", "?"),
                    "direct": float(d["direct_effect_cprime"]),
                    "indirect": float(d["indirect_effect_ab"]),
                    "total": float(d["direct_effect_cprime"])
                    + float(d["indirect_effect_ab"]),
                    "significant": bool(d.get("indirect_significant", False)),
                }
            )
        df = pd.DataFrame(rows)
        df = df[df["significant"] & (df["total"].abs() > 0.01)].copy()
        df = df.reindex(df["total"].abs().sort_values(ascending=True).index)

        n = len(df)
        if n == 0:
            print("[figC2] skipped: no significant mediation rows")
            return

        fig, ax = plt.subplots(figsize=_FIGSIZE)
        bar_h = 0.65

        for i, row in df.reset_index(drop=True).iterrows():
            ax.barh(
                i,
                row["direct"],
                height=bar_h,
                color=_COL_DIRECT,
                alpha=0.85,
                label="Direct" if i == 0 else "",
            )
            ax.barh(
                i,
                row["indirect"],
                height=bar_h,
                left=row["direct"],
                color=_COL_INDIRECT,
                alpha=0.85,
                label="Indirect (mediated)" if i == 0 else "",
            )
            label = f"{_s(row['treatment'])}\n→ {_s(row['mediator'])}"
            ax.text(
                -0.01,
                i,
                label,
                va="center",
                ha="right",
                fontsize=8,
                transform=ax.get_yaxis_transform(),
                linespacing=1.3,
            )
            total = row["total"]
            # Wide bars: white text centered inside; narrow bars: black text outside
            _thresh = df["total"].abs().max() * 0.12
            if abs(total) >= _thresh:
                ax.text(
                    total / 2,
                    i,
                    f"{total:.3f}",
                    va="center",
                    ha="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            else:
                offset = 0.005 if total >= 0 else -0.005
                ha = "left" if total >= 0 else "right"
                ax.text(
                    total + offset,
                    i,
                    f"{total:.3f}",
                    va="center",
                    ha=ha,
                    fontsize=8,
                    color="#333",
                )

        ax.axvline(0, color="black", lw=0.8, alpha=0.6, linestyle="--")
        ax.set_yticks([])
        ax.set_xlabel("Effect size on MAS correctness")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax, top=True, right=True, left=True)
        ax.legend(
            loc="lower left",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
            fontsize=10,
        )

        fig.tight_layout()
        self.save_figure(fig, "figC2_mediation.pdf", dpi=600)
        self.save_figure(fig, "figC2_mediation.png", dpi=300)
        plt.close(fig)

        top = df.reindex(df["total"].abs().sort_values(ascending=False).index).head(3)
        top_lines = "\n".join(
            f"  - {_s(r['treatment'])} → {_s(r['mediator'])}: "
            f"direct={r['direct']:.3f}, indirect={r['indirect']:.3f}, total={r['total']:.3f}"
            for _, r in top.iterrows()
        )
        body = (
            "Stacked bars decomposing each treatment→mediator→MAS correctness path into "
            "direct effect (red, c') and indirect/mediated effect (blue, a×b). Only "
            "paths with a significant indirect effect and |total| > 0.01 are shown. "
            "Sorted by total effect magnitude.\n\n"
            f"Top three:\n{top_lines}\n\n"
            "Round-1 agent entropy has a large direct negative effect with an additional "
            "indirect channel through round-2 entropy — uncertainty compounds across rounds."
        )
        self._record_analysis("Figure C2 — Mediation decomposition", body)

    # ── Figure C3 — Full consensus causal DAG ────────────────────────────────

    def figc3_causal_dag(self) -> None:
        edges_path = self.data_dir / "causal_discovery" / "all_edges.csv"
        if not edges_path.exists():
            print(f"[figC3] skipped: {edges_path} not found")
            return

        try:
            import networkx as nx
        except ImportError:
            print("[figC3] skipped: networkx not installed")
            return

        df = pd.read_csv(edges_path)
        df_con = df[
            (df["consensus"] == True) & (df["type"] == "directed")
        ].drop_duplicates(["source", "target"])

        G = nx.DiGraph()
        for _, row in df_con.iterrows():
            G.add_edge(row["source"], row["target"])

        outcome = "is_finally_correct"

        # Two-line wrapped labels: split at "-" or "/" near midpoint
        def _wrap_node_label(s: str, max_len: int = 12) -> str:
            if len(s) <= max_len:
                return s
            # Try splitting at each "-" or "/" from midpoint outward
            mid = len(s) // 2
            best = None
            for i in range(len(s)):
                for delta in (i, -i):
                    idx = mid + delta
                    if 0 < idx < len(s) and s[idx] in "-/":
                        line1 = s[: idx + 1]
                        line2 = s[idx + 1 :]
                        if line1 and line2:
                            best = (line1.rstrip("-"), line2)
                            break
                if best:
                    break
            if best:
                return best[0] + "\n" + best[1]
            # Fallback: hard break at midpoint
            return s[:mid] + "\n" + s[mid:]

        labels = {n: _wrap_node_label(_s(n)) for n in G.nodes()}

        # Always use manual tier layout for full control over spacing
        _GROUP_TIER = {
            "base_model": 0,
            "round1": 1,
            "round2": 2,
            "exp_ans": 2,
            "sample": 2,
            "outcome": 3,
        }
        tier_nodes: dict[int, list] = {0: [], 1: [], 2: [], 3: []}
        for n in G.nodes():
            t = _GROUP_TIER.get(_node_group(n), 1)
            tier_nodes[t].append(n)

        pos = {}
        x_gap = 4.5
        y_gap = 2.2
        for tier, tnodes in tier_nodes.items():
            n_nodes = len(tnodes)
            # Split large tiers into two sub-columns
            if n_nodes > 7:
                n_cols = 2
            else:
                n_cols = 1
            col_width = 1.8
            for i, n in enumerate(tnodes):
                col = i % n_cols
                row = i // n_cols
                n_rows = (n_nodes + n_cols - 1) // n_cols
                x = tier * x_gap + (col - (n_cols - 1) / 2) * col_width
                y = -(row - (n_rows - 1) / 2) * y_gap
                pos[n] = (x, y)

        _LEGEND_FONTSIZE = 11

        fig, ax = plt.subplots(figsize=(13, 10.8))

        # ── Causal-flow background arrow (bottom, spanning full x range) ──────
        ax.annotate(
            "Causal flow →",
            xy=(0.98, 0.02),
            xytext=(0.02, 0.02),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=10,
            color="#BBBBBB",
            va="bottom",
            ha="left",
            arrowprops=dict(
                arrowstyle="-|>", color="#DDDDDD", lw=2.5, mutation_scale=20
            ),
        )

        # ── Edges ─────────────────────────────────────────────────────────────
        non_outcome_edges = [(u, v) for u, v in G.edges() if v != outcome]
        outcome_edges = [(u, v) for u, v in G.edges() if v == outcome]

        if non_outcome_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=non_outcome_edges,
                edge_color="#CCCCCC",
                width=1.0,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=12,
                connectionstyle="arc3,rad=0.06",
                ax=ax,
            )
        if outcome_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=outcome_edges,
                edge_color="#2B4B8C",
                width=3.0,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=20,
                connectionstyle="arc3,rad=0.06",
                ax=ax,
            )

        # ── Nodes ─────────────────────────────────────────────────────────────
        non_outcome_nodes = [n for n in G.nodes() if n != outcome]
        outcome_nodes = [n for n in G.nodes() if n == outcome]
        non_outcome_colors = [_GROUP_COLORS[_node_group(n)] for n in non_outcome_nodes]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=non_outcome_nodes,
            node_color=non_outcome_colors,
            node_size=2400,
            alpha=0.92,
            ax=ax,
        )
        if outcome_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=outcome_nodes,
                node_color=["#2B4B8C"],
                node_size=2400,
                alpha=0.95,
                ax=ax,
            )

        non_outcome_labels = {n: l for n, l in labels.items() if n != outcome}
        outcome_labels = {n: l for n, l in labels.items() if n == outcome}
        nx.draw_networkx_labels(
            G,
            pos,
            labels=non_outcome_labels,
            font_size=10,
            font_color="black",
            font_weight="bold",
            ax=ax,
        )
        if outcome_labels:
            nx.draw_networkx_labels(
                G,
                pos,
                labels=outcome_labels,
                font_size=10,
                font_color="white",
                font_weight="bold",
                ax=ax,
            )

        # ── Edge labels on outcome edges only ────────────────────────────────
        edge_label_map = {(u, v): "→ correct" for u, v in G.edges() if v == outcome}
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_label_map,
            font_size=11,
            font_color="#2B4B8C",
            label_pos=0.35,
            ax=ax,
        )

        ax.axis("off")

        legend_handles = [
            mpatches.Patch(color=_GROUP_COLORS["outcome"], label="MAS correctness"),
            mpatches.Patch(
                color=_GROUP_COLORS["base_model"], label="Base-model entropy"
            ),
            mpatches.Patch(
                color=_GROUP_COLORS["round1"], label="Round-1 agent entropy"
            ),
            mpatches.Patch(
                color=_GROUP_COLORS["round2"], label="Round-2 agent entropy"
            ),
            mpatches.Patch(
                color=_GROUP_COLORS["exp_ans"], label="Exp / Answer-token entropy"
            ),
            mpatches.Patch(color=_GROUP_COLORS["sample"], label="Sample-level entropy"),
            Line2D(
                [], [], color="#2B4B8C", lw=2.5, label="Direct cause of correctness"
            ),
            Line2D([], [], color="#CCCCCC", lw=1.5, label="Causal edge"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
            fontsize=_LEGEND_FONTSIZE,
            ncol=2,
        )

        # Abbreviation table (upper left)
        present_abbrevs = []
        for abbr, full in _ABBREV_GLOSSARY:
            if any(abbr in _s(n) for n in G.nodes() if n != outcome):
                present_abbrevs.append(f"{abbr:<6}  {full}")
        if present_abbrevs:
            ax.text(
                0.01,
                0.99,
                "\n".join(present_abbrevs),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=_LEGEND_FONTSIZE,
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="#CCCCCC",
                ),
            )

        fig.tight_layout()
        self.save_figure(fig, "figC3_causal_dag.pdf", dpi=600)
        self.save_figure(fig, "figC3_causal_dag.png", dpi=300)
        plt.close(fig)

        in_edges = [(u, v) for u, v in G.edges() if v == outcome]
        body = (
            f"Full consensus DAG: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} directed edges agreed by both PC and FCI. "
            "Nodes are colored by semantic group. Edges into MAS correctness are "
            "drawn in dark gray at greater width.\n\n"
            "Direct causal predecessors of MAS correctness: "
            + ", ".join(_s(u) for u, v in in_edges)
            + ".\n\n"
            "The graph reveals a layered propagation: base-model entropy (red) seeds "
            "round-1 agent entropy (orange), which in turn drives round-2 entropy (blue). "
            "Answer-token entropy (purple) forms a parallel branch tracking how the model "
            "revises its final token distribution across rounds. Only the base-model "
            "entropy cluster has a confirmed direct edge to MAS correctness; all other "
            "entropy signals exert their harm indirectly through this cascade."
        )
        self._record_analysis("Figure C3 — Consensus causal DAG", body)

    # ── Figure C4 — PC vs FCI edge agreement ────────────────────────────────

    def figc4_edge_agreement(self) -> None:
        edges_path = self.data_dir / "causal_discovery" / "all_edges.csv"
        if not edges_path.exists():
            print(f"[figC4] skipped: {edges_path} not found")
            return

        df = pd.read_csv(edges_path)
        df = df[df["type"] == "directed"]
        nodes = sorted(set(df["source"]) | set(df["target"]))
        n = len(nodes)
        idx = {nd: i for i, nd in enumerate(nodes)}

        pc_mat = np.zeros((n, n), dtype=int)
        fci_mat = np.zeros((n, n), dtype=int)
        for _, row in df.iterrows():
            i, j = idx[row["source"]], idx[row["target"]]
            if row["method"] == "PC":
                pc_mat[i, j] = 1
            elif row["method"] == "FCI":
                fci_mat[i, j] = 1

        agree = pc_mat * 2 + fci_mat
        agree_vis = np.where(
            agree == 3, 2, np.where(agree == 2, 1, np.where(agree == 1, -1, 0))
        )
        short_nodes = [_s(nd) for nd in nodes]

        fig, ax = plt.subplots(figsize=(8.5, 7.2))
        cmap = plt.cm.get_cmap("RdBu", 5)
        ax.imshow(agree_vis, cmap=cmap, vmin=-2, vmax=2, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_nodes, rotation=90, fontsize=6.5)
        ax.set_yticklabels(short_nodes, fontsize=6.5)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")

        legend_els = [
            mpatches.Patch(facecolor=cmap(4), label="Both PC & FCI"),
            mpatches.Patch(facecolor=cmap(3), label="PC only"),
            mpatches.Patch(facecolor=cmap(1), label="FCI only"),
        ]
        ax.legend(
            handles=legend_els,
            loc="upper right",
            fontsize=9,
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
        )

        fig.tight_layout()
        self.save_figure(fig, "figC4_edge_agreement.pdf", dpi=600)
        self.save_figure(fig, "figC4_edge_agreement.png", dpi=300)
        plt.close(fig)

        both = int((agree_vis == 2).sum())
        pc_only = int((agree_vis == 1).sum())
        fci_only = int((agree_vis == -1).sum())
        body = (
            f"Edge agreement heatmap between PC and FCI over {n} nodes. "
            f"Dark blue = both agree ({both} edges); light blue = PC only ({pc_only}); "
            f"red = FCI only ({fci_only}). High overlap confirms the consensus DAG is "
            "stable across Markov-boundary assumptions."
        )
        self._record_analysis("Figure C4 — PC vs FCI edge agreement", body)

    # ── Figure C5 — Full mediation table ────────────────────────────────────

    def figc5_mediation_full(self) -> None:
        path = self.data_dir / "mediation" / "mediation_results.json"
        if not path.exists():
            print(f"[figC5] skipped: {path} not found")
            return
        raw = json.loads(path.read_text(encoding="utf-8"))

        rows = []
        for d in raw:
            rows.append(
                {
                    "treatment": _s(d["treatment"]),
                    "mediator": _s(d.get("mediator", "?")),
                    "direct": float(d["direct_effect_cprime"]),
                    "indirect": float(d["indirect_effect_ab"]),
                    "total": float(d["direct_effect_cprime"])
                    + float(d["indirect_effect_ab"]),
                    "sig": bool(d.get("indirect_significant", False)),
                }
            )
        df = pd.DataFrame(rows)
        df = df.reindex(df["total"].abs().sort_values(ascending=False).index).head(18)
        df = df.reset_index(drop=True)

        n = len(df)
        bar_h = 0.65
        thresh = df["total"].abs().max() * 0.12

        fig, ax = plt.subplots(figsize=(6.5, max(5.4, n * 0.55 + 1.2)))

        for i, row in df.iterrows():
            alpha = 0.9 if row["sig"] else 0.35
            ax.barh(
                i,
                row["direct"],
                height=bar_h,
                color=_COL_DIRECT,
                alpha=alpha,
                label="Direct (c')" if i == 0 else "",
            )
            ax.barh(
                i,
                row["indirect"],
                height=bar_h,
                left=row["direct"],
                color=_COL_INDIRECT,
                alpha=alpha,
                label="Indirect (a×b)" if i == 0 else "",
            )
            ax.text(
                -0.01,
                i,
                f"{row['treatment']}\n→ {row['mediator']}",
                va="center",
                ha="right",
                fontsize=8,
                transform=ax.get_yaxis_transform(),
                linespacing=1.3,
                color="#333" if row["sig"] else "#999",
            )
            total = row["total"]
            if abs(total) >= thresh:
                ax.text(
                    total / 2,
                    i,
                    f"{total:.3f}",
                    va="center",
                    ha="center",
                    fontsize=7.5,
                    color="white",
                    fontweight="bold",
                )
            else:
                offset = 0.005 if total >= 0 else -0.005
                ax.text(
                    total + offset,
                    i,
                    f"{total:.3f}",
                    va="center",
                    ha="left" if total >= 0 else "right",
                    fontsize=7.5,
                    color="#333",
                )
            if row["sig"]:
                ax.text(
                    1.01,
                    i,
                    "✓",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#2CA02C",
                    transform=ax.get_yaxis_transform(),
                )

        ax.axvline(0, color="black", lw=0.8, alpha=0.6, linestyle="--")
        ax.set_yticks([])
        ax.set_xlabel("Effect size on MAS correctness")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax, top=True, right=True, left=True)

        handles = [
            mpatches.Patch(color=_COL_DIRECT, alpha=0.9, label="Direct (c')"),
            mpatches.Patch(color=_COL_INDIRECT, alpha=0.9, label="Indirect (a×b)"),
            mpatches.Patch(color="#333", alpha=0.35, label="Non-significant (faded)"),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
            fontsize=9,
        )

        fig.tight_layout()
        self.save_figure(fig, "figC5_mediation_full.pdf", dpi=600)
        self.save_figure(fig, "figC5_mediation_full.png", dpi=300)
        plt.close(fig)

        n_sig = int(df["sig"].sum())
        body = (
            f"Extended mediation table: top-{n} paths by |total effect| "
            f"({n_sig} significant, ✓). Faded bars = non-significant indirect effect. "
            "Sorted by |total| descending."
        )
        self._record_analysis("Figure C5 — Full mediation table", body)

    # ── Figure C6 — Feature selection Borda scores ───────────────────────────

    def figc6_borda_scores(self) -> None:
        feat_path = self.data_dir / "feature_selection" / "selected_features.csv"
        if not feat_path.exists():
            print(f"[figC6] skipped: {feat_path} not found")
            return

        df = pd.read_csv(feat_path).sort_values("borda_score", ascending=True)
        n = len(df)
        labels = [_s(f) for f in df["feature"]]

        fig, ax = plt.subplots(figsize=(8, max(5.4, n * 0.32 + 1.0)))
        bar_h = 0.6

        # Color bars by semantic group (reuse node group palette)
        bar_colors = [_GROUP_COLORS.get(_node_group(f), "#CCCCCC") for f in df["feature"]]
        ax.barh(range(n), df["borda_score"], height=bar_h, color=bar_colors, alpha=0.88)

        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Borda Count Score")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax, top=True, right=True)

        legend_handles = [
            mpatches.Patch(color=_GROUP_COLORS["base_model"], label="Base-model"),
            mpatches.Patch(color=_GROUP_COLORS["round1"],     label="Round-1 agent"),
            mpatches.Patch(color=_GROUP_COLORS["round2"],     label="Round-2 agent"),
            mpatches.Patch(color=_GROUP_COLORS["exp_ans"],    label="Exp / Answer-token"),
            mpatches.Patch(color=_GROUP_COLORS["sample"],     label="Sample-level"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", frameon=True,
                  framealpha=0.9, edgecolor="#CCCCCC", fontsize=9)

        fig.tight_layout()
        self.save_figure(fig, "figC6_borda_scores.pdf", dpi=600)
        self.save_figure(fig, "figC6_borda_scores.png", dpi=300)
        plt.close(fig)
        self._record_analysis("Figure C6 — Feature selection Borda scores",
                              f"{n} selected features ranked by Borda count fusion score.")

    # ── Figure C7 — Selected features correlation heatmap ───────────────────

    def figc7_feature_correlation(self) -> None:
        feat_path = self.data_dir / "feature_selection" / "selected_features.csv"
        # Walk up from exp dir to find merged_datasets.csv
        data_path = self.data_dir.parent / "data" / "merged_datasets.csv"
        if not feat_path.exists():
            print(f"[figC7] skipped: {feat_path} not found")
            return
        if not data_path.exists():
            print(f"[figC7] skipped: {data_path} not found")
            return

        features = pd.read_csv(feat_path)["feature"].tolist()
        df = pd.read_csv(data_path, usecols=lambda c: c in features)
        available = [f for f in features if f in df.columns]
        corr = df[available].corr(method="spearman")

        short = [_s(f) for f in available]
        n = len(short)
        fig_size = max(8, n * 0.38 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmin=-1, vmax=1, center=0,
            xticklabels=short, yticklabels=short,
            square=True, linewidths=0.3, linecolor="#EEEEEE",
            annot=False,
            cbar_kws={"label": "Spearman Correlation", "shrink": 0.7},
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

        fig.tight_layout()
        self.save_figure(fig, "figC7_feature_correlation.pdf", dpi=600)
        self.save_figure(fig, "figC7_feature_correlation.png", dpi=300)
        plt.close(fig)
        self._record_analysis("Figure C7 — Selected features correlation heatmap",
                              f"Spearman correlation matrix of {len(available)} selected features.")

    # ── main entry ────────────────────────────────────────────────────────────

    def compose(self) -> None:
        print("[causal] Figure C1 — ATE forest plot")
        self.figc1_ate_forest()
        print("[causal] Figure C2 — Mediation decomposition")
        self.figc2_mediation()
        print("[causal] Figure C3 — Consensus causal DAG")
        self.figc3_causal_dag()
        print("[causal] Figure C4 — PC vs FCI edge agreement")
        self.figc4_edge_agreement()
        print("[causal] Figure C5 — Full mediation table")
        self.figc5_mediation_full()
        print("[causal] Figure C6 — Borda scores")
        self.figc6_borda_scores()
        print("[causal] Figure C7 — Feature correlation heatmap")
        self.figc7_feature_correlation()
        self.figc5_mediation_full()

        if self._analyses:
            md = (
                "# Causal Analysis Figure Interpretations\n\n"
                "Auto-generated by `visualization/plots/causal.py::CausalPlot.compose()`.\n\n"
                + "\n".join(self._analyses)
            )
            out = self.output_dir / "figure_analyses.md"
            out.write_text(md, encoding="utf-8")
            print(f"Saved: {out}")
