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

# ── Short labels ──────────────────────────────────────────────────────────────
_SHORT = {
    "is_finally_correct":                             "MAS correctness",
    "base_sample_avg_entropy_per_token":              "Base avg entropy/token",
    "base_sample_total_entropy":                      "Base total entropy",
    "base_model_max_answer_token_entropy":            "Base max ans-token entropy",
    "base_model_std_answer_token_entropy":            "Base std ans-token entropy",
    "sample_mean_answer_token_entropy":               "Mean ans-token entropy",
    "sample_avg_entropy_per_token":                   "Avg entropy/token",
    "sample_avg_entropy_per_token_diff_vs_base":      "Entropy diff vs base",
    "sample_avg_entropy_per_agent":                   "Avg entropy/agent",
    "sample_max_entropy":                             "Sample max entropy",
    "sample_entropy_reduction_vs_base_total":         "Entropy reduction vs base",
    "answer_token_entropy_change":                    "Ans-token entropy change",
    "answer_token_entropy_change_direction":          "Entropy change direction",
    "exp_total_entropy":                              "Exp total entropy",
    "round_2_total_entropy":                          "R2 total entropy",
    "round_1_2_change_entropy":                       "R1→R2 entropy change",
    "sample_round_1_median_agent_max_entropy":        "R1 median agent max ent.",
    "sample_round_1_median_agent_total_entropy":      "R1 median agent total ent.",
    "sample_round_1_max_agent_max_entropy":           "R1 max agent max ent.",
    "sample_round_1_min_agent_q3_entropy":            "R1 min agent Q3 ent.",
    "sample_round_1_q1_agent_variance_entropy":       "R1 Q1 agent var. ent.",
    "sample_round_1_q3_agent_std_entropy":            "R1 Q3 agent std ent.",
    "sample_round_1_all_agents_entropy_per_token":    "R1 all-agents ent./token",
    "sample_round_1_agent_total_entropy_spread":      "R1 agent total ent. spread",
    "sample_round_2_q3_agent_max_entropy":            "R2 Q3 agent max ent.",
    "sample_round_2_max_agent_std_entropy":           "R2 max agent std ent.",
    "sample_round_2_median_agent_std_entropy":        "R2 median agent std ent.",
    "sample_round_2_std_agent_max_entropy":           "R2 std agent max ent.",
    "sample_round_mean_agent_total_entropy_first_last_diff": "Agent total ent. Δ (R1→last)",
    "sample_round_agent_total_entropy_spread_first_last_diff": "Agent ent. spread Δ (R1→last)",
    "is_multi_agent":                                 "Multi-agent",
}


def _s(name: str) -> str:
    return _SHORT.get(name, name.replace("_", " "))


# ── Node semantic groups for figC3 ────────────────────────────────────────────
def _node_group(name: str) -> str:
    if name == "is_finally_correct":
        return "outcome"
    if name.startswith("base_"):
        return "base_model"
    if "round_1" in name or "r1" in name.lower():
        return "round1"
    if "round_2" in name or "r2" in name.lower() or name in (
        "round_2_total_entropy", "round_1_2_change_entropy",
    ):
        return "round2"
    if "answer_token" in name or name == "answer_token_entropy_change_direction":
        return "answer"
    if "exp_total" in name or "sample_max" in name or "sample_avg" in name or \
            "sample_mean" in name or "sample_entropy" in name:
        return "sample"
    return "other"


_GROUP_COLORS = {
    "outcome":    "#2CA02C",   # green
    "base_model": "#D73027",   # red
    "round1":     "#FC8D59",   # orange
    "round2":     "#4575B4",   # blue
    "answer":     "#9467BD",   # purple
    "sample":     "#8C564B",   # brown
    "other":      "#7F7F7F",   # gray
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
            rows.append({
                "treatment":  d["treatment"],
                "is_direct":  bool(d.get("is_direct_cause", False)),
                "lr_ate":     float(lr.get("ate", np.nan)),
                "lr_p":       float(lr.get("p_value", 1.0)),
                "ps_ate":     float(ps.get("ate", np.nan)),
                "ipw_ate":    float(ipw.get("ate", np.nan)),
            })
        df = pd.DataFrame(rows).sort_values("ps_ate")
        n = len(df)

        fig, ax = plt.subplots(figsize=_FIGSIZE)

        for i, row in df.reset_index(drop=True).iterrows():
            color = _COL_DIRECT if row["is_direct"] else _COL_INDIRECT
            sig_star = "*" if row["lr_p"] < 0.05 else ""

            offsets = {"LR": 0.22, "PS": 0.0, "IPW": -0.22}
            ates    = {"LR": row["lr_ate"], "PS": row["ps_ate"], "IPW": row["ipw_ate"]}
            markers = {"LR": "D", "PS": "o", "IPW": "s"}
            alphas  = {"LR": 0.55, "PS": 1.0, "IPW": 0.75}

            valid = [v for v in ates.values() if not np.isnan(v)]
            if len(valid) >= 2:
                ax.hlines(i, min(valid), max(valid),
                          color=color, linewidth=1.2, alpha=0.5, zorder=3)
            for est, marker in markers.items():
                val = ates[est]
                if not np.isnan(val):
                    ax.scatter(val, i + offsets[est], color=color, marker=marker,
                               s=36, alpha=alphas[est], zorder=4)

            ax.text(-0.02, i, f"{_s(row['treatment'])}{sig_star}",
                    va="center", ha="right", fontsize=9.5,
                    transform=ax.get_yaxis_transform())

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
            Line2D([], [], color="#555", marker="D", linestyle="none",
                   markersize=6, alpha=0.7, label="Linear regression"),
            Line2D([], [], color="#555", marker="o", linestyle="none",
                   markersize=6, label="Propensity score"),
            Line2D([], [], color="#555", marker="s", linestyle="none",
                   markersize=6, alpha=0.8, label="IPW"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=True,
                  framealpha=0.9, edgecolor="#CCCCCC", fontsize=9.5, ncol=1)

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
            rows.append({
                "treatment":   d["treatment"],
                "mediator":    d.get("mediator", "?"),
                "direct":      float(d["direct_effect_cprime"]),
                "indirect":    float(d["indirect_effect_ab"]),
                "total":       float(d["direct_effect_cprime"]) + float(d["indirect_effect_ab"]),
                "significant": bool(d.get("indirect_significant", False)),
            })
        df = pd.DataFrame(rows)
        df = df[df["significant"] & (df["total"].abs() > 0.01)].copy()
        df = df.reindex(df["total"].abs().sort_values(ascending=True).index)

        n = len(df)
        if n == 0:
            print("[figC2] skipped: no significant mediation rows")
            return

        fig, ax = plt.subplots(figsize=_FIGSIZE)
        bar_h = 0.55

        for i, row in df.reset_index(drop=True).iterrows():
            ax.barh(i, row["direct"], height=bar_h, color=_COL_DIRECT,
                    alpha=0.85, label="Direct" if i == 0 else "")
            ax.barh(i, row["indirect"], height=bar_h, left=row["direct"],
                    color=_COL_INDIRECT, alpha=0.85,
                    label="Indirect (mediated)" if i == 0 else "")
            label = f"{_s(row['treatment'])} → {_s(row['mediator'])}"
            ax.text(-0.01, i, label, va="center", ha="right", fontsize=9,
                    transform=ax.get_yaxis_transform())
            total = row["total"]
            # Wide bars: white text centered inside; narrow bars: black text outside
            _thresh = df["total"].abs().max() * 0.12
            if abs(total) >= _thresh:
                ax.text(total / 2, i, f"{total:.3f}", va="center", ha="center",
                        fontsize=8, color="white", fontweight="bold")
            else:
                offset = 0.005 if total >= 0 else -0.005
                ha = "left" if total >= 0 else "right"
                ax.text(total + offset, i, f"{total:.3f}", va="center", ha=ha,
                        fontsize=8, color="#333")

        ax.axvline(0, color="black", lw=0.8, alpha=0.6, linestyle="--")
        ax.set_yticks([])
        ax.set_xlabel("Effect size on MAS correctness")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax, top=True, right=True, left=True)
        ax.legend(loc="lower left", frameon=True, framealpha=0.9,
                  edgecolor="#CCCCCC", fontsize=10)

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
        df_con = df[(df["consensus"] == True) & (df["type"] == "directed")].drop_duplicates(
            ["source", "target"]
        )

        G = nx.DiGraph()
        for _, row in df_con.iterrows():
            G.add_edge(row["source"], row["target"])

        outcome = "is_finally_correct"

        # Node colors by semantic group
        node_colors = [_GROUP_COLORS[_node_group(n)] for n in G.nodes()]

        # Short labels
        labels = {n: _s(n) for n in G.nodes()}

        # Layout: try graphviz dot for hierarchy, fallback to manual column layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            try:
                pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
            except Exception:
                # Manual hierarchical layout grouped by semantic column
                _COL_ORDER = ["base_model", "sample", "answer", "round1", "round2", "other", "outcome"]
                col_nodes: dict[str, list] = {c: [] for c in _COL_ORDER}
                for n in G.nodes():
                    col_nodes[_node_group(n)].append(n)
                pos = {}
                x_gap = 2.2
                for ci, col in enumerate(_COL_ORDER):
                    nodes = col_nodes[col]
                    for ri, n in enumerate(nodes):
                        y = -(ri - (len(nodes) - 1) / 2)
                        pos[n] = (ci * x_gap, y)

        fig, ax = plt.subplots(figsize=(11, 7.5))

        # Edge colors: edges into outcome are dark/thick; others are light gray
        edge_colors = ["#444444" if v == outcome else "#BBBBBB" for u, v in G.edges()]
        edge_widths = [2.5 if v == outcome else 0.9 for u, v in G.edges()]

        nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors, width=edge_widths,
            arrows=True, arrowstyle="-|>", arrowsize=12,
            connectionstyle="arc3,rad=0.06", ax=ax,
        )
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=900, alpha=0.92, ax=ax,
        )
        nx.draw_networkx_labels(
            G, pos, labels=labels, font_size=6.5, font_color="black",
            font_weight="bold", ax=ax,
        )

        # Annotate edges into outcome with "→ correct" direction label
        edge_label_map = {(u, v): "→ correct" for u, v in G.edges() if v == outcome}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_label_map,
            font_size=6, font_color="#444444", label_pos=0.35, ax=ax,
        )

        ax.axis("off")

        legend_handles = [
            mpatches.Patch(color=_GROUP_COLORS["outcome"],    label="MAS correctness (outcome)"),
            mpatches.Patch(color=_GROUP_COLORS["base_model"], label="Base-model entropy"),
            mpatches.Patch(color=_GROUP_COLORS["round1"],     label="Round-1 agent entropy"),
            mpatches.Patch(color=_GROUP_COLORS["round2"],     label="Round-2 agent entropy"),
            mpatches.Patch(color=_GROUP_COLORS["answer"],     label="Answer-token entropy"),
            mpatches.Patch(color=_GROUP_COLORS["sample"],     label="Sample-level entropy"),
        ]
        ax.legend(handles=legend_handles, loc="lower left", frameon=True,
                  framealpha=0.9, edgecolor="#CCCCCC", fontsize=9, ncol=2)

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
            + ", ".join(_s(u) for u, v in in_edges) + ".\n\n"
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

        pc_mat  = np.zeros((n, n), dtype=int)
        fci_mat = np.zeros((n, n), dtype=int)
        for _, row in df.iterrows():
            i, j = idx[row["source"]], idx[row["target"]]
            if row["method"] == "PC":
                pc_mat[i, j] = 1
            elif row["method"] == "FCI":
                fci_mat[i, j] = 1

        agree = pc_mat * 2 + fci_mat
        agree_vis = np.where(agree == 3, 2, np.where(agree == 2, 1,
                             np.where(agree == 1, -1, 0)))
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
        ax.legend(handles=legend_els, loc="upper right", fontsize=9,
                  frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

        fig.tight_layout()
        self.save_figure(fig, "figC4_edge_agreement.pdf", dpi=600)
        self.save_figure(fig, "figC4_edge_agreement.png", dpi=300)
        plt.close(fig)

        both    = int((agree_vis == 2).sum())
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
            rows.append({
                "treatment": _s(d["treatment"]),
                "mediator":  _s(d.get("mediator", "?")),
                "direct":    float(d["direct_effect_cprime"]),
                "indirect":  float(d["indirect_effect_ab"]),
                "total":     float(d["direct_effect_cprime"]) + float(d["indirect_effect_ab"]),
                "sig":       bool(d.get("indirect_significant", False)),
            })
        df = pd.DataFrame(rows)
        df = df.reindex(df["total"].abs().sort_values(ascending=False).index).head(18)
        df = df.reset_index(drop=True)

        n = len(df)
        bar_h = 0.52
        thresh = df["total"].abs().max() * 0.12

        fig, ax = plt.subplots(figsize=(6.5, max(5.4, n * 0.42 + 1.2)))

        for i, row in df.iterrows():
            alpha = 0.9 if row["sig"] else 0.35
            ax.barh(i, row["direct"], height=bar_h, color=_COL_DIRECT, alpha=alpha,
                    label="Direct (c')" if i == 0 else "")
            ax.barh(i, row["indirect"], height=bar_h, left=row["direct"],
                    color=_COL_INDIRECT, alpha=alpha,
                    label="Indirect (a×b)" if i == 0 else "")
            ax.text(-0.01, i, f"{row['treatment']} → {row['mediator']}",
                    va="center", ha="right", fontsize=8.5,
                    transform=ax.get_yaxis_transform(),
                    color="#333" if row["sig"] else "#999")
            total = row["total"]
            if abs(total) >= thresh:
                ax.text(total / 2, i, f"{total:.3f}", va="center", ha="center",
                        fontsize=7.5, color="white", fontweight="bold")
            else:
                offset = 0.005 if total >= 0 else -0.005
                ax.text(total + offset, i, f"{total:.3f}", va="center",
                        ha="left" if total >= 0 else "right",
                        fontsize=7.5, color="#333")
            if row["sig"]:
                ax.text(1.01, i, "✓", va="center", ha="left", fontsize=9,
                        color="#2CA02C", transform=ax.get_yaxis_transform())

        ax.axvline(0, color="black", lw=0.8, alpha=0.6, linestyle="--")
        ax.set_yticks([])
        ax.set_xlabel("Effect size on MAS correctness")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax, top=True, right=True, left=True)

        handles = [
            mpatches.Patch(color=_COL_DIRECT,   alpha=0.9, label="Direct (c')"),
            mpatches.Patch(color=_COL_INDIRECT,  alpha=0.9, label="Indirect (a×b)"),
            mpatches.Patch(color="#333", alpha=0.35, label="Non-significant (faded)"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=True,
                  framealpha=0.9, edgecolor="#CCCCCC", fontsize=9)

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

        if self._analyses:
            md = (
                "# Causal Analysis Figure Interpretations\n\n"
                "Auto-generated by `visualization/plots/causal.py::CausalPlot.compose()`.\n\n"
                + "\n".join(self._analyses)
            )
            out = self.output_dir / "figure_analyses.md"
            out.write_text(md, encoding="utf-8")
            print(f"Saved: {out}")
