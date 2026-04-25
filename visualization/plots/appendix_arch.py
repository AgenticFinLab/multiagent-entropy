"""Per-model token-level entropy visualization (appendix).

Refactored from results_plot/appendix-arch/analyze_appendix_arch.py into the
`visualization` package layout.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from visualization.base import ARCH_COLORS, BaseVisualizer

warnings.filterwarnings("ignore")


class AppendixArchPlot(BaseVisualizer):
    """Token-level entropy visualization across datasets and architectures."""

    DATASETS = [
        "aime2024_16384", "aime2025_16384", "gsm8k",
        "humaneval", "math500", "mmlu",
    ]
    DATASET_DISPLAY = {
        "aime2024_16384": "AIME24",
        "aime2025_16384": "AIME25",
        "gsm8k": "GSM8K",
        "humaneval": "HumanEval",
        "math500": "Math500",
        "mmlu": "MMLU",
    }
    MODELS = ["qwen3_0_6b", "qwen3_4b", "qwen3_8b", "qwen_2_5_7b_simplerl_zoo"]
    MODEL_DISPLAY = {
        "qwen3_0_6b": "Qwen3-0.6B",
        "qwen3_4b": "Qwen3-4B",
        "qwen3_8b": "Qwen3-8B",
        "qwen_2_5_7b_simplerl_zoo": "Qwen2.5-7B-RL",
    }
    ARCHITECTURES = ["single", "centralized", "debate", "hybrid", "sequential"]
    ARCH_DISPLAY = {
        "single": "Single",
        "centralized": "Centralized",
        "debate": "Debate",
        "hybrid": "Hybrid",
        "sequential": "Sequential",
    }

    def __init__(self, data_dir: Path | str, output_dir: Path | str, eval_dir: Path | str) -> None:
        super().__init__(output_dir=output_dir, base_font_size=14)
        self.data_dir = Path(data_dir)
        self.eval_dir = Path(eval_dir)
        self._eval_cache: Dict[str, Optional[dict]] = {}

        # Round backgrounds: Round 1 = white, Round 2 = light gray.
        self.round_colors = ["#FFFFFF", "#F2F2F2"]

        # Override certain rcParams to match original appendix style.
        plt.rcParams["axes.linewidth"] = 1.0
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["ytick.labelsize"] = 12
        plt.rcParams["legend.fontsize"] = 12

    # ---------- file lookup helpers ----------

    def _find_experiment_dir(self, dataset: str, model: str, arch: str) -> Optional[Path]:
        model_dir = self.data_dir / dataset / model
        if not model_dir.exists():
            return None
        for exp_dir in model_dir.iterdir():
            if exp_dir.is_dir() and f"_{arch}_agent_" in exp_dir.name:
                return exp_dir
        return None

    def _load_eval_data(self, dataset: str) -> Optional[dict]:
        if dataset in self._eval_cache:
            return self._eval_cache[dataset]
        eval_path = self.eval_dir / dataset / "all_metrics.json"
        if not eval_path.exists():
            self._eval_cache[dataset] = None
            return None
        try:
            with open(eval_path, "r") as f:
                data = json.load(f)
            self._eval_cache[dataset] = data
            return data
        except Exception:
            self._eval_cache[dataset] = None
            return None

    def _get_sample_correctness(
        self, dataset: str, model: str, exp_name: str, sample_id: int
    ) -> Optional[bool]:
        eval_data = self._load_eval_data(dataset)
        if eval_data is None:
            return None
        try:
            experiments = eval_data["models"].get(model, {}).get("experiments", {})
            exp_data = experiments.get(exp_name)
            if exp_data is None:
                return None
            samples = exp_data.get("samples", {})
            sample_data = samples.get(f"ID{sample_id}") or samples.get(str(sample_id))
            if sample_data is None:
                return None
            return sample_data.get("is_finally_correct")
        except Exception:
            return None

    def _get_first_sample_id_and_keys(self, result_block_path: Path) -> Tuple[Optional[str], List[str]]:
        """Parse a Result_block file and return the first sample's prefix + agent keys."""
        try:
            with open(result_block_path, "r") as f:
                data = json.load(f)
            sample_prefixes: Dict[str, int] = {}
            for key in data.keys():
                # Match both formats: Result_60-Agent or Result_ID100-Agent
                match = re.match(r"(Result_(?:ID)?(\d+))-", key)
                if match:
                    prefix = match.group(1)
                    num = int(match.group(2))
                    if prefix not in sample_prefixes:
                        sample_prefixes[prefix] = num
            if not sample_prefixes:
                return None, []
            first_prefix = min(sample_prefixes.keys(), key=lambda p: sample_prefixes[p])
            agent_keys = [k for k in data.keys() if k.startswith(first_prefix + "-")]
            return first_prefix, agent_keys
        except Exception as e:
            print(f"Error reading {result_block_path}: {e}")
            return None, []

    def _load_token_entropy(self, tensor_path: Path) -> Optional[np.ndarray]:
        try:
            tensor = torch.load(tensor_path, map_location="cpu")
            return tensor.numpy()
        except Exception:
            return None

    def _smooth_entropy(self, entropy: np.ndarray, window_size: int = 30) -> np.ndarray:
        if len(entropy) < window_size:
            return entropy
        kernel = np.ones(window_size) / window_size
        return np.convolve(entropy, kernel, mode="valid")

    def _normalize_to_length(self, data: np.ndarray, target_length: int = 1000) -> np.ndarray:
        if len(data) == 0:
            return np.zeros(target_length)
        if len(data) == target_length:
            return data
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, target_length)
        return np.interp(x_new, x_old, data)

    def _parse_agent_info(self, key: str, arch: str) -> Dict:
        """Parse agent info; round counting depends on architecture topology.

        - single: 1 agent per round
        - sequential / hybrid / centralized: 4 agents per round
        - debate: 3 agents per round (orchestrator runs at end)
        """
        match = re.match(r"Result_(?:ID)?\d+-(.+)-(\d+)_sample_\d+", key)
        if match:
            agent_type = match.group(1)
            order = int(match.group(2))
            if arch == "single":
                round_num = order
            elif arch == "sequential":
                round_num = (order - 1) // 4 + 1
            elif arch == "debate":
                round_num = (order - 1) // 3 + 1
            elif arch in ["hybrid", "centralized"]:
                round_num = (order - 1) // 4 + 1
            else:
                round_num = (order - 1) // 4 + 1
            return {
                "agent_type": agent_type,
                "order": order,
                "round": round_num,
                "short_name": f"{agent_type[:4]}-{order}",
            }
        return {"agent_type": "Unknown", "order": 0, "round": 1, "short_name": "Unk"}

    def _get_agent_data_for_sample(
        self, exp_dir: Path, sample_prefix: str, agent_keys: List[str], arch: str
    ) -> List[Dict]:
        tensors_dir = exp_dir / "traces" / "tensors"
        agent_data = []
        for key in agent_keys:
            agent_info = self._parse_agent_info(key, arch)
            tensor_file = tensors_dir / f"{key}_extras_entropy.pt"
            if tensor_file.exists():
                entropy = self._load_token_entropy(tensor_file)
                if entropy is not None:
                    agent_data.append({
                        "info": agent_info,
                        "entropy": entropy,
                        "length": len(entropy),
                        "key": key,
                    })
        agent_data.sort(key=lambda x: x["info"]["order"])
        return agent_data

    # ---------- plotting ----------

    def plot_model_entropy(
        self,
        ax,
        model: str,
        dataset: str,
        arch: str,
        target_length: int = 1000,
        show_agent_labels: bool = True,
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        exp_dir = self._find_experiment_dir(dataset, model, arch)
        if exp_dir is None:
            return False, None, None

        exp_name = exp_dir.name
        result_block = exp_dir / "traces" / "Result_block_0.json"
        if not result_block.exists():
            return False, None, None

        sample_prefix, agent_keys = self._get_first_sample_id_and_keys(result_block)
        if sample_prefix is None or not agent_keys:
            return False, None, None

        match = re.match(r"Result_(?:ID)?(\d+)", sample_prefix)
        sample_id = int(match.group(1)) if match else None

        agent_data = self._get_agent_data_for_sample(exp_dir, sample_prefix, agent_keys, arch)
        if not agent_data:
            return False, exp_name, sample_id

        total_length = sum(d["length"] for d in agent_data)
        combined_entropy = np.concatenate([d["entropy"] for d in agent_data])
        smoothed = self._smooth_entropy(combined_entropy, window_size=30)
        normalized = self._normalize_to_length(smoothed, target_length)

        x = np.arange(target_length)
        ax.plot(x, normalized, color=ARCH_COLORS[arch], linewidth=1.2, alpha=0.9)

        cumsum = 0
        current_round = 0
        round_start = 0

        for idx, d in enumerate(agent_data):
            start_norm = int((cumsum / total_length) * target_length)
            cumsum += d["length"]
            end_norm = int((cumsum / total_length) * target_length)
            agent_info = d["info"]

            if agent_info["round"] != current_round:
                if current_round > 0:
                    round_color = self.round_colors[(current_round - 1) % len(self.round_colors)]
                    ax.axvspan(round_start, start_norm, alpha=1.0, color=round_color, zorder=0)
                current_round = agent_info["round"]
                round_start = start_norm

            if idx > 0:
                ax.axvline(x=start_norm, color="black", linestyle="--", linewidth=1, alpha=0.7)

            if show_agent_labels:
                region_width = end_norm - start_norm
                # Only label region when it's wide enough (>=5% of axis) to avoid overlap.
                if region_width >= 50:
                    mid_x = (start_norm + end_norm) / 2
                    ax.text(mid_x, -0.05, agent_info["short_name"],
                            ha="center", va="top", fontsize=12,
                            rotation=90, alpha=0.8, transform=ax.get_xaxis_transform())

        if current_round > 0:
            round_color = self.round_colors[(current_round - 1) % len(self.round_colors)]
            ax.axvspan(round_start, target_length, alpha=1.0, color=round_color, zorder=0)

        ax.set_xlim(0, target_length)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        return True, exp_name, sample_id

    def generate_model_figure(self, model: str) -> None:
        """6 rows (datasets) × 5 columns (architectures) for one model."""
        print(f"Generating figure for model: {self.MODEL_DISPLAY[model]}...")

        fig, axes = plt.subplots(6, 5, figsize=(18, 22))

        for i, dataset in enumerate(self.DATASETS):
            for j, arch in enumerate(self.ARCHITECTURES):
                ax = axes[i, j]
                show_labels = i < 5
                has_data, exp_name, sample_id = self.plot_model_entropy(
                    ax, model, dataset, arch, show_agent_labels=show_labels
                )

                if not has_data:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=13, color="gray")
                else:
                    is_correct = self._get_sample_correctness(dataset, model, exp_name, sample_id)
                    if is_correct is not None:
                        indicator = "✓" if is_correct else "✗"
                        color = "#2E8B57" if is_correct else "#DC143C"
                        ax.text(0.97, 0.95, indicator, transform=ax.transAxes,
                                ha="right", va="top", fontsize=18, fontweight="bold",
                                color=color, zorder=10)

                if j == 0:
                    ax.set_ylabel(f"{self.DATASET_DISPLAY[dataset]}", fontsize=14, fontweight="bold")
                if i == 0:
                    ax.set_title(f"{self.ARCH_DISPLAY[arch]}", fontsize=15, fontweight="bold")
                if i == 5:
                    ax.set_xlabel("Normalized Token Position", fontsize=13)
                else:
                    ax.tick_params(axis="x", labelbottom=False)

        legend_elements = [
            Patch(facecolor=self.round_colors[0], edgecolor="black", linewidth=1, label="Round 1"),
            Patch(facecolor=self.round_colors[1], edgecolor="black", linewidth=1, label="Round 2"),
            Line2D([0], [0], color="#EDEDED", linestyle="--", linewidth=1.5, label="Agent Boundary"),
            Line2D([0], [0], marker="$✓$", color="#2E8B57", markersize=12,
                   linestyle="None", label="Correct"),
            Line2D([0], [0], marker="$✗$", color="#DC143C", markersize=12,
                   linestyle="None", label="Wrong"),
        ]
        fig.legend(handles=legend_elements, loc="upper center", ncol=5,
                   bbox_to_anchor=(0.5, 0.97), fontsize=14, frameon=True)

        fig.suptitle(f"Token-Level Entropy Dynamics: {self.MODEL_DISPLAY[model]}",
                     fontsize=20, fontweight="bold", y=0.99)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        self.save_figure(fig, f"token_entropy_{model}.pdf")
        plt.close(fig)

    def compose(self) -> None:
        for model in self.MODELS:
            self.generate_model_figure(model)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "experiments" / "results" / "raw"
    output_dir = root / "visualization" / "outputs" / "appendix_arch"
    eval_dir = root / "evaluation" / "results_qwen"

    plotter = AppendixArchPlot(
        data_dir=data_dir,
        output_dir=output_dir,
        eval_dir=eval_dir,
    )
    plotter.compose()


if __name__ == "__main__":
    main()
