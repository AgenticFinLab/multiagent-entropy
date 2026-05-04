"""
Causal Discovery: Learn causal graph structure from selected features.

Uses PC and FCI algorithms from causal-learn library with domain knowledge
constraints (temporal ordering, forbidden edges) to discover the causal DAG
relating entropy features to MAS correctness.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz, chisq, kci
from causallearn.utils.GraphUtils import GraphUtils

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_data_from_path, prepare_features

import seaborn as sns

logger = logging.getLogger(__name__)


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


# ---------- Temporal tiers for background knowledge ----------
# Tier 0: Pre-existing / exogenous (base model properties)
# Tier 1: Round 1 features
# Tier 2: Round 2 / cross-round features
# Tier 3: Outcome
TIER_KEYWORDS = {
    0: ["base_model_", "base_sample_"],
    1: ["round_1_", "sample_round_1_"],
    2: [
        "round_2_",
        "sample_round_2_",
        "round_1_2_change",
        "cross_round",
        "first_last",
        "answer_token_entropy_change",
        "slope_per_round",
        "volatility",
        "spread_first_last",
    ],
    # Tier 3 is the outcome only
}

# Sample-level aggregate features (no clear temporal order, treated as Tier 1.5)
AGGREGATE_KEYWORDS = [
    "sample_mean_",
    "sample_max_",
    "sample_min_",
    "sample_std_",
    "sample_median_",
    "sample_q1_",
    "sample_q3_",
    "sample_variance_",
    "sample_total_",
    "sample_avg_",
    "sample_entropy_",
    "sample_answer_",
    "sample_num_agents",
    "sample_all_agents",
    "exp_total",
]


def assign_tier(feature: str) -> int:
    """Assign a temporal tier to a feature based on naming conventions."""
    if feature == "is_finally_correct":
        return 3
    for tier in [0, 2, 1]:  # check tier 0 and 2 first (more specific)
        for kw in TIER_KEYWORDS[tier]:
            if kw in feature:
                return tier
    # Check aggregate keywords -> post-execution sample-level, same tier as Round 2
    for kw in AGGREGATE_KEYWORDS:
        if feature.startswith(kw):
            return 2
    return 1  # default


class CausalDiscovery:
    """Discover causal structure using PC and FCI algorithms."""

    def __init__(
        self,
        data_path: str = "data_mining/data/merged_datasets.csv",
        feature_list_path: str = "data_mining/results_causal_complete/feature_selection/selected_features.csv",
        output_dir: str = "data_mining/results_causal_complete/causal_discovery",
        alpha: float = 0.01,
        max_sample: int = 10000,
    ):
        self.data_path = Path(data_path)
        self.feature_list_path = Path(feature_list_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.max_sample = max_sample

    def load_data(self) -> Tuple[np.ndarray, List[str]]:
        """Load data with selected features + target."""
        feat_df = pd.read_csv(self.feature_list_path)
        selected = feat_df["feature"].tolist()

        df = load_data_from_path(self.data_path)
        target = "is_finally_correct"
        cols = [c for c in selected if c in df.columns] + [target]
        sub = df[cols].dropna()

        # Subsample for computational efficiency
        if len(sub) > self.max_sample:
            sub = sub.sample(n=self.max_sample, random_state=42)
            logger.info("Subsampled to %d rows for causal discovery", self.max_sample)

        node_names = list(sub.columns)
        data = sub.values.astype(float)
        logger.info("Data for causal discovery: %d rows, %d variables", *data.shape)
        return data, node_names

    def build_background_knowledge(self, node_names: List[str]):
        """Build background knowledge from temporal tiers."""
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

        bk = BackgroundKnowledge()
        tiers = {name: assign_tier(name) for name in node_names}

        # Forbidden edges: higher tier cannot cause lower tier
        for i, ni in enumerate(node_names):
            for j, nj in enumerate(node_names):
                if i == j:
                    continue
                ti, tj = tiers[ni], tiers[nj]
                # Forbid: outcome -> anything
                if ni == "is_finally_correct":
                    from causallearn.graph.GraphNode import GraphNode

                    # We'll handle this via tier constraint below
                    pass
                # Forbid higher tier causing lower tier
                if ti > tj:
                    bk.add_forbidden_by_node(
                        self._make_node(ni, node_names),
                        self._make_node(nj, node_names),
                    )

        logger.info(
            "Background knowledge: tiers = %s", {n: tiers[n] for n in node_names}
        )
        return bk, tiers

    @staticmethod
    def _make_node(name: str, node_names: List[str]):
        """Create a GraphNode compatible with causal-learn BK API."""
        from causallearn.graph.GraphNode import GraphNode

        return GraphNode(name)

    # ------------------------------------------------------------------
    # PC algorithm
    # ------------------------------------------------------------------
    def run_pc(self, data: np.ndarray, node_names: List[str], bk=None):
        """Run PC algorithm."""
        logger.info("Running PC algorithm (alpha=%.3f)...", self.alpha)
        cg = pc(
            data,
            alpha=self.alpha,
            indep_test=fisherz,
            stable=True,
            uc_rule=0,  # default orientation rule
            uc_priority=2,
            background_knowledge=bk,
            node_names=node_names,
        )
        logger.info("PC algorithm completed")
        return cg

    # ------------------------------------------------------------------
    # FCI algorithm
    # ------------------------------------------------------------------
    def run_fci(self, data: np.ndarray, node_names: List[str], bk=None):
        """Run FCI algorithm."""
        logger.info("Running FCI algorithm (alpha=%.3f)...", self.alpha)
        g, edges = fci(
            data,
            independence_test_method=fisherz,
            alpha=self.alpha,
            background_knowledge=bk,
            node_names=node_names,
        )
        logger.info("FCI algorithm completed")
        return g, edges

    # ------------------------------------------------------------------
    # Extract adjacency
    # ------------------------------------------------------------------
    def extract_adjacency(self, graph, node_names: List[str]) -> pd.DataFrame:
        """Extract adjacency matrix from causal-learn graph object."""
        adj = graph.graph  # numpy array
        adj_df = pd.DataFrame(adj, index=node_names, columns=node_names)
        return adj_df

    def extract_edges(self, adj_df: pd.DataFrame, method: str) -> List[Dict]:
        """Extract directed edges from adjacency matrix.

        causal-learn encoding: adj[i,j] = -1 and adj[j,i] = 1 means i -> j
        """
        edges = []
        node_names = list(adj_df.index)
        n = len(node_names)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i -> j: adj[i,j] == -1 and adj[j,i] == 1
                if adj_df.iloc[i, j] == -1 and adj_df.iloc[j, i] == 1:
                    edges.append(
                        {
                            "source": node_names[i],
                            "target": node_names[j],
                            "type": "directed",
                            "method": method,
                        }
                    )
                # i -- j (undirected): adj[i,j] == -1 and adj[j,i] == -1
                elif adj_df.iloc[i, j] == -1 and adj_df.iloc[j, i] == -1 and i < j:
                    edges.append(
                        {
                            "source": node_names[i],
                            "target": node_names[j],
                            "type": "undirected",
                            "method": method,
                        }
                    )
                # i <-> j (bidirected / latent): adj[i,j] == 1 and adj[j,i] == 1
                elif adj_df.iloc[i, j] == 1 and adj_df.iloc[j, i] == 1 and i < j:
                    edges.append(
                        {
                            "source": node_names[i],
                            "target": node_names[j],
                            "type": "bidirected",
                            "method": method,
                        }
                    )
        return edges

    # ------------------------------------------------------------------
    # Compare PC and FCI
    # ------------------------------------------------------------------
    def compare_results(
        self, pc_edges: List[Dict], fci_edges: List[Dict]
    ) -> pd.DataFrame:
        """Compare edges found by PC and FCI, mark consensus."""

        # Normalize to frozensets for comparison
        def edge_key(e):
            if e["type"] == "directed":
                return (e["source"], e["target"], "directed")
            return (tuple(sorted([e["source"], e["target"]])), e["type"])

        pc_set = {edge_key(e) for e in pc_edges}
        fci_set = {edge_key(e) for e in fci_edges}
        consensus = pc_set & fci_set

        all_edges = []
        for e in pc_edges + fci_edges:
            ek = edge_key(e)
            e_copy = e.copy()
            e_copy["consensus"] = ek in consensus
            all_edges.append(e_copy)

        # Deduplicate
        seen = set()
        unique = []
        for e in all_edges:
            key = (e["source"], e["target"], e["type"], e["method"])
            if key not in seen:
                seen.add(key)
                unique.append(e)

        df = pd.DataFrame(unique)
        logger.info(
            "Edges: PC=%d, FCI=%d, consensus=%d",
            len(pc_edges),
            len(fci_edges),
            len(consensus),
        )
        return df

    # ------------------------------------------------------------------
    # Identify direct causes of outcome
    # ------------------------------------------------------------------
    def find_direct_causes(
        self, adj_df: pd.DataFrame, target: str = "is_finally_correct"
    ) -> List[str]:
        """Find variables with a directed edge into the target."""
        if target not in adj_df.columns:
            return []
        causes = []
        for node in adj_df.index:
            if node == target:
                continue
            # node -> target: adj[node, target] == -1 and adj[target, node] == 1
            if adj_df.loc[node, target] == -1 and adj_df.loc[target, node] == 1:
                causes.append(node)
        return causes

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot_causal_graph(
        self,
        edges_df: pd.DataFrame,
        node_names: List[str],
        tiers: Dict[str, int],
        title: str = "Causal Graph",
        filename: str = "causal_graph.png",
        filter_method: Optional[str] = None,
    ):
        """Plot the causal graph using matplotlib."""
        if filter_method:
            plot_edges = edges_df[edges_df["method"] == filter_method]
        else:
            # Use consensus edges, plus unique directed edges
            plot_edges = edges_df[
                (edges_df["consensus"] == True) | (edges_df["type"] == "directed")
            ].drop_duplicates(subset=["source", "target", "type"])

        if len(plot_edges) == 0:
            logger.warning("No edges to plot for %s", title)
            return

        # --- Only keep nodes that appear in edges ---
        edge_nodes = set()
        for _, edge in plot_edges.iterrows():
            edge_nodes.add(edge["source"])
            edge_nodes.add(edge["target"])
        active_nodes = [n for n in node_names if n in edge_nodes]

        # Group active nodes by tier
        tier_nodes: Dict[int, List[str]] = {}
        for name in active_nodes:
            t = tiers.get(name, 1)
            tier_nodes.setdefault(t, []).append(name)

        # --- Compact grid layout ---
        # Tiers go left-to-right; within each tier, nodes arranged in a grid
        tier_colors = {0: "#4575B4", 1: "#91BFD8", 2: "#FC8D59", 3: "#D73027"}
        node_colors = {}
        pos = {}

        # Calculate tier x-positions with adequate spacing
        n_tiers = len(tier_nodes)
        tier_x_gap = 6.0  # horizontal gap between tiers
        node_y_gap = 1.4  # vertical gap between nodes

        # For large tiers, use multi-column sub-layout
        max_tier_rows = 0

        for tier in sorted(tier_nodes.keys()):
            nodes = tier_nodes[tier]
            n_nodes = len(nodes)
            # Adaptive columns: use more columns for large tiers
            if n_nodes <= 4:
                n_cols = 1
            elif n_nodes <= 10:
                n_cols = 2
            else:
                n_cols = 3
            n_rows = int(np.ceil(n_nodes / n_cols))
            max_tier_rows = max(max_tier_rows, n_rows)

            x_base = tier * tier_x_gap
            col_width = 3.0  # sub-column width within a tier

            for i, node in enumerate(nodes):
                col = i % n_cols
                row = i // n_cols
                x = x_base + (col - (n_cols - 1) / 2.0) * col_width
                y = -(row - (n_rows - 1) / 2.0) * node_y_gap
                pos[node] = (x, y)
                node_colors[node] = tier_colors.get(tier, "#95a5a6")

        setup_plot_style()

        # --- Figure size: compact, similar ratio to analyze_accuracy.py ---
        fig_w = max(16, n_tiers * 6.5)
        fig_h = max(8, max_tier_rows * 1.6 + 2)
        # Cap to reasonable bounds
        fig_w = min(fig_w, 30)
        fig_h = min(fig_h, 16)

        with sns.plotting_context("paper", font_scale=1.4):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            # Draw edges
            for _, edge in plot_edges.iterrows():
                src, tgt = edge["source"], edge["target"]
                if src not in pos or tgt not in pos:
                    continue
                x0, y0 = pos[src]
                x1, y1 = pos[tgt]

                style = "->"
                color = "#2c3e50"
                lw = 1.5
                if edge["type"] == "undirected":
                    style = "-"
                    color = "#7f8c8d"
                    lw = 1.0
                elif edge["type"] == "bidirected":
                    style = "<->"
                    color = "#8e44ad"
                    lw = 1.0

                if edge.get("consensus", False):
                    lw = 2.5
                    color = "#D73027" if edge["type"] == "directed" else color

                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle=style,
                        color=color,
                        lw=lw,
                        connectionstyle="arc3,rad=0.15",
                        mutation_scale=18,
                    ),
                )

            # Draw nodes
            def _wrap_label(name: str, max_chars: int = 28) -> str:
                """Wrap long node labels for readability."""
                if len(name) <= max_chars:
                    return name
                # Try to break at underscore near midpoint
                parts = name.split("_")
                line1, line2 = "", ""
                for p in parts:
                    candidate = (line1 + "_" + p) if line1 else p
                    if len(candidate) <= max_chars:
                        line1 = candidate
                    else:
                        line2 = (line2 + "_" + p) if line2 else p
                return line1 + "\n" + line2 if line2 else line1

            for node, (x, y) in pos.items():
                color = node_colors.get(node, "#95a5a6")
                bbox = dict(
                    boxstyle="round,pad=0.35",
                    facecolor=color,
                    alpha=0.88,
                    edgecolor="white",
                    linewidth=1.5,
                )
                display_name = _wrap_label(node)
                ax.text(
                    x,
                    y,
                    display_name,
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                    bbox=bbox,
                )

            # Legend
            legend_patches = [
                mpatches.Patch(color=tier_colors[0], label="Tier 0: Base Entropy"),
                mpatches.Patch(color=tier_colors[1], label="Tier 1: Round 1 / Sample"),
                mpatches.Patch(
                    color=tier_colors[2], label="Tier 2: Round 2 / Cross-round"
                ),
                mpatches.Patch(color=tier_colors[3], label="Tier 3: Outcome"),
            ]
            ax.legend(
                handles=legend_patches,
                loc="upper left",
                fontsize=14,
                frameon=False,
                bbox_to_anchor=(0.0, 1.0),
            )

            # ax.set_title(title, fontsize=20, fontweight="bold", pad=15)
            # Auto-fit limits with padding
            all_x = [p[0] for p in pos.values()]
            all_y = [p[1] for p in pos.values()]
            x_margin = 2.0
            y_margin = 1.2
            ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
            ax.axis("off")
            plt.tight_layout()
            out_path = self.output_dir / filename
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        logger.info("Saved causal graph to %s", out_path)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_report(
        self,
        node_names: List[str],
        tiers: Dict[str, int],
        pc_causes: List[str],
        fci_causes: List[str],
        edges_df: pd.DataFrame,
    ) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("CAUSAL DISCOVERY REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Variables: {len(node_names)}")
        lines.append(f"Significance level (alpha): {self.alpha}")
        lines.append(f"Algorithms: PC, FCI (Fisher-Z independence test)")
        lines.append("")

        lines.append("TEMPORAL TIER ASSIGNMENT")
        lines.append("-" * 80)
        for tier in sorted(set(tiers.values())):
            feats = [n for n, t in tiers.items() if t == tier]
            lines.append(f"  Tier {tier}: {len(feats)} variables")
            for f in feats:
                lines.append(f"    - {f}")
        lines.append("")

        lines.append("DIRECT CAUSES OF is_finally_correct")
        lines.append("-" * 80)
        lines.append(f"  PC algorithm: {pc_causes}")
        lines.append(f"  FCI algorithm: {fci_causes}")
        consensus_causes = sorted(set(pc_causes) & set(fci_causes))
        lines.append(f"  Consensus (both methods): {consensus_causes}")
        lines.append("")

        lines.append("EDGE SUMMARY")
        lines.append("-" * 80)
        if len(edges_df) > 0:
            for method in edges_df["method"].unique():
                method_edges = edges_df[edges_df["method"] == method]
                directed = method_edges[method_edges["type"] == "directed"]
                undirected = method_edges[method_edges["type"] == "undirected"]
                bidirected = method_edges[method_edges["type"] == "bidirected"]
                lines.append(
                    f"  {method}: {len(directed)} directed, {len(undirected)} undirected, {len(bidirected)} bidirected"
                )

            consensus_edges = edges_df[edges_df["consensus"] == True]
            lines.append(f"  Consensus edges: {len(consensus_edges)}")
        lines.append("")

        lines.append("ALL DIRECTED EDGES (sorted by target)")
        lines.append("-" * 80)
        directed = edges_df[edges_df["type"] == "directed"].sort_values(
            ["target", "source"]
        )
        for _, row in directed.iterrows():
            consensus_mark = " [CONSENSUS]" if row.get("consensus", False) else ""
            lines.append(
                f"  {row['source']} -> {row['target']} ({row['method']}){consensus_mark}"
            )
        lines.append("")

        report = "\n".join(lines)
        out_path = self.output_dir / "causal_discovery_report.txt"
        out_path.write_text(report)
        logger.info("Saved report to %s", out_path)
        return report

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """Run causal discovery pipeline."""
        logger.info("=" * 60)
        logger.info("Starting causal discovery")
        logger.info("=" * 60)

        # 1. Load data
        data, node_names = self.load_data()

        # 2. Build background knowledge
        bk, tiers = self.build_background_knowledge(node_names)

        # 3. Run PC
        pc_result = self.run_pc(data, node_names, bk=bk)
        pc_adj = self.extract_adjacency(pc_result.G, node_names)
        pc_edges = self.extract_edges(pc_adj, method="PC")
        pc_causes = self.find_direct_causes(pc_adj)
        logger.info("PC direct causes of outcome: %s", pc_causes)

        # 4. Run FCI
        fci_graph, _ = self.run_fci(data, node_names, bk=bk)
        fci_adj = self.extract_adjacency(fci_graph, node_names)
        fci_edges = self.extract_edges(fci_adj, method="FCI")
        fci_causes = self.find_direct_causes(fci_adj)
        logger.info("FCI direct causes of outcome: %s", fci_causes)

        # 5. Compare
        edges_df = self.compare_results(pc_edges, fci_edges)
        edges_df.to_csv(self.output_dir / "all_edges.csv", index=False)

        # 6. Save adjacency matrices
        pc_adj.to_csv(self.output_dir / "pc_adjacency.csv")
        fci_adj.to_csv(self.output_dir / "fci_adjacency.csv")

        # 7. Save direct causes
        consensus_causes = sorted(set(pc_causes) & set(fci_causes))
        all_causes = sorted(set(pc_causes) | set(fci_causes))
        causes_info = {
            "pc_causes": pc_causes,
            "fci_causes": fci_causes,
            "consensus_causes": consensus_causes,
            "all_causes": all_causes,
        }
        with open(self.output_dir / "direct_causes.json", "w") as f:
            json.dump(causes_info, f, indent=2)

        # 8. Visualizations
        self.plot_causal_graph(
            edges_df,
            node_names,
            tiers,
            title="Causal Graph (PC Algorithm)",
            filename="causal_graph_pc.png",
            filter_method="PC",
        )
        self.plot_causal_graph(
            edges_df,
            node_names,
            tiers,
            title="Causal Graph (FCI Algorithm)",
            filename="causal_graph_fci.png",
            filter_method="FCI",
        )
        self.plot_causal_graph(
            edges_df,
            node_names,
            tiers,
            title="Causal Graph (Consensus)",
            filename="causal_graph_consensus.png",
        )

        # 9. Report
        self.generate_report(node_names, tiers, pc_causes, fci_causes, edges_df)

        logger.info("Causal discovery complete. Results in %s", self.output_dir)
        return causes_info


def main():
    parser = argparse.ArgumentParser(description="Causal Discovery")
    parser.add_argument("--data-path", default="data_mining/data/merged_datasets.csv")
    parser.add_argument(
        "--feature-list",
        default="data_mining/results_causal_complete/feature_selection/selected_features.csv",
    )
    parser.add_argument(
        "--output-dir", default="data_mining/results_causal_complete/causal_discovery"
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max-sample", type=int, default=10000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    disc = CausalDiscovery(
        data_path=args.data_path,
        feature_list_path=args.feature_list,
        output_dir=args.output_dir,
        alpha=args.alpha,
        max_sample=args.max_sample,
    )
    causes = disc.run()
    print(f"\nDirect causes of is_finally_correct:")
    print(f"  PC:        {causes['pc_causes']}")
    print(f"  FCI:       {causes['fci_causes']}")
    print(f"  Consensus: {causes['consensus_causes']}")


if __name__ == "__main__":
    main()
