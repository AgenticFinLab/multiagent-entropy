"""
Entropy analyzer for multi-agent experiments.

This module provides functionality to analyze entropy values from experiments,
including visualization, correlation analysis, and comparison between agents.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


logger = logging.getLogger(__name__)


class EntropyAnalyzer:
    """Analyzer for entropy values in multi-agent experiments."""

    def __init__(self, experiment_dir: str):
        """
        Initialize the entropy analyzer.

        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = experiment_dir
        self.entropy_data = {}
        self.agent_names = set()

    def load_entropy_data(self) -> Dict[str, Any]:
        """
        Load entropy data from experiment directory.

        Returns:
            Dictionary containing entropy data
        """
        tensors_dir = os.path.join(self.experiment_dir, "traces", "tensors")
        if not os.path.exists(tensors_dir):
            logger.warning(f"Tensors directory not found: {tensors_dir}")
            return {}

        entropy_files = self._find_entropy_files(tensors_dir)
        if not entropy_files:
            logger.warning(f"No entropy files found in: {tensors_dir}")
            return {}

        entropy_data = {
            "agents": {},
            "samples": {},
            "rounds": {},
            "agent_sequences": {},
        }

        for entropy_file in entropy_files:
            file_info = self._parse_entropy_filename(entropy_file)
            if not file_info:
                continue

            sample_id = file_info["sample_id"]
            agent_name = file_info["agent_name"]
            agent_sequence = file_info["agent_sequence"]

            try:
                import torch

                entropy_tensor = torch.load(entropy_file)
                entropy_values = entropy_tensor.cpu().numpy().tolist()

                if agent_name not in entropy_data["agents"]:
                    entropy_data["agents"][agent_name] = []

                if sample_id not in entropy_data["samples"]:
                    entropy_data["samples"][sample_id] = {}

                if agent_name not in entropy_data["samples"][sample_id]:
                    entropy_data["samples"][sample_id][agent_name] = {}

                if agent_sequence not in entropy_data["samples"][sample_id][agent_name]:
                    entropy_data["samples"][sample_id][agent_name][agent_sequence] = []

                entropy_data["samples"][sample_id][agent_name][agent_sequence].extend(
                    entropy_values
                )
                entropy_data["agents"][agent_name].extend(entropy_values)

                if agent_sequence not in entropy_data["agent_sequences"]:
                    entropy_data["agent_sequences"][agent_sequence] = []

                entropy_data["agent_sequences"][agent_sequence].extend(entropy_values)
                self.agent_names.add(agent_name)

            except Exception as e:
                logger.warning(f"Failed to load entropy file {entropy_file}: {e}")

        self.entropy_data = entropy_data
        return entropy_data

    def _find_entropy_files(self, tensors_dir: str) -> List[str]:
        """
        Find all entropy files in the tensors directory.

        Args:
            tensors_dir: Path to tensors directory

        Returns:
            List of entropy file paths
        """
        entropy_files = []

        for filename in os.listdir(tensors_dir):
            if filename.endswith("_extras_entropy.pt"):
                entropy_files.append(os.path.join(tensors_dir, filename))

        return sorted(entropy_files)

    def _parse_entropy_filename(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Parse entropy filename to extract sample_id, agent_name, round, and agent_sequence.

        Args:
            filepath: Path to entropy file

        Returns:
            Dictionary with sample_id, agent_name, round, agent_sequence, and main_id, or None if parsing fails
        """
        filename = os.path.basename(filepath)

        if not filename.endswith("_extras_entropy.pt"):
            return None

        filename = filename[: -len("_extras_entropy.pt")]

        parts = filename.split("_")
        if len(parts) < 4:
            return None

        if parts[0] != "Result":
            return None

        sample_id = parts[1]
        if parts[2] != "sample":
            return None

        try:
            sample_num = int(parts[3])
        except ValueError:
            return None

        dash_parts = sample_id.split("-")
        if len(dash_parts) < 3:
            return None

        main_id = dash_parts[0]
        agent_name = "-".join(dash_parts[1:-1])

        try:
            agent_sequence = int(dash_parts[-1])
        except ValueError:
            return None

        return {
            "sample_id": f"Result_{sample_id}_sample_{sample_num}",
            "agent_name": agent_name,
            "sample_num": sample_num,
            "main_id": main_id,
            "agent_sequence": agent_sequence,
        }

    def _extract_entropy_values(self, output: Dict[str, Any]) -> List[float]:
        """
        Extract entropy values from agent output.

        Args:
            output: Agent output dictionary

        Returns:
            List of entropy values
        """
        entropy_values = []

        if "entropy" in output and isinstance(output["entropy"], (int, float)):
            entropy_values.append(float(output["entropy"]))

        for key, value in output.items():
            if key.startswith("entropy_") and isinstance(value, (int, float)):
                entropy_values.append(float(value))

        return entropy_values

    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate entropy statistics for each agent.

        Returns:
            Dictionary of statistics per agent
        """
        if not self.entropy_data:
            self.load_entropy_data()

        statistics = {}

        for agent_name, entropies in self.entropy_data.get("agents", {}).items():
            if not entropies:
                continue

            entropies_array = np.array(entropies)
            statistics[agent_name] = {
                "mean": np.mean(entropies_array),
                "std": np.std(entropies_array),
                "min": np.min(entropies_array),
                "max": np.max(entropies_array),
                "median": np.median(entropies_array),
                "q25": np.percentile(entropies_array, 25),
                "q75": np.percentile(entropies_array, 75),
                "count": len(entropies_array),
            }

        return statistics

    def calculate_step_statistics(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Calculate entropy statistics for each agent at each agent sequence step.

        Returns:
            Dictionary of statistics per agent per agent sequence
            Structure: {agent_name: {agent_sequence: {mean, std, min, max, median, count}}}
        """
        if not self.entropy_data:
            self.load_entropy_data()

        step_statistics = {}

        samples = self.entropy_data.get("samples", {})

        for sample_id, sample_data in samples.items():
            for agent_name, sequence_data in sample_data.items():
                if agent_name not in step_statistics:
                    step_statistics[agent_name] = {}

                for agent_sequence, entropies in sequence_data.items():
                    if agent_sequence not in step_statistics[agent_name]:
                        step_statistics[agent_name][agent_sequence] = []

                    step_statistics[agent_name][agent_sequence].extend(entropies)

        for agent_name, sequences in step_statistics.items():
            for agent_sequence, entropies in sequences.items():
                if not entropies:
                    continue

                entropies_array = np.array(entropies)
                step_statistics[agent_name][agent_sequence] = {
                    "mean": np.mean(entropies_array),
                    "std": np.std(entropies_array),
                    "min": np.min(entropies_array),
                    "max": np.max(entropies_array),
                    "median": np.median(entropies_array),
                    "q25": np.percentile(entropies_array, 25),
                    "q75": np.percentile(entropies_array, 75),
                    "count": len(entropies_array),
                }

        return step_statistics

    def calculate_round_statistics(
        self, agents_per_round: int = 4
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Calculate entropy statistics for each agent at each round.

        Args:
            agents_per_round: Number of agents per round (default: 4 for MathAgent, ScienceAgent, CodeAgent, OrchestratorAgent)

        Returns:
            Dictionary of statistics per agent per round
            Structure: {agent_name: {round: {mean, std, min, max, median, count}}}
        """
        if not self.entropy_data:
            self.load_entropy_data()

        round_statistics = {}

        samples = self.entropy_data.get("samples", {})

        for sample_id, sample_data in samples.items():
            for agent_name, sequence_data in sample_data.items():
                if agent_name not in round_statistics:
                    round_statistics[agent_name] = {}

                for agent_sequence, entropies in sequence_data.items():
                    round_num = (agent_sequence - 1) // agents_per_round + 1

                    if round_num not in round_statistics[agent_name]:
                        round_statistics[agent_name][round_num] = []

                    round_statistics[agent_name][round_num].extend(entropies)

        for agent_name, rounds in round_statistics.items():
            for round_num, entropies in rounds.items():
                if not entropies:
                    continue

                entropies_array = np.array(entropies)
                round_statistics[agent_name][round_num] = {
                    "mean": np.mean(entropies_array),
                    "std": np.std(entropies_array),
                    "min": np.min(entropies_array),
                    "max": np.max(entropies_array),
                    "median": np.median(entropies_array),
                    "q25": np.percentile(entropies_array, 25),
                    "q75": np.percentile(entropies_array, 75),
                    "count": len(entropies_array),
                }

        return round_statistics

    def calculate_sample_statistics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate entropy statistics for each sample.

        Returns:
            Dictionary of statistics per sample
            Structure: {sample_id: {agent_name: {mean, std, min, max, median, count}}}
        """
        if not self.entropy_data:
            self.load_entropy_data()

        sample_statistics = {}

        samples = self.entropy_data.get("samples", {})

        for sample_id, sample_data in samples.items():
            sample_statistics[sample_id] = {}

            for agent_name, steps_data in sample_data.items():
                all_entropies = []
                for step, entropies in steps_data.items():
                    all_entropies.extend(entropies)

                if all_entropies:
                    entropies_array = np.array(all_entropies)
                    sample_statistics[sample_id][agent_name] = {
                        "mean": np.mean(entropies_array),
                        "std": np.std(entropies_array),
                        "min": np.min(entropies_array),
                        "max": np.max(entropies_array),
                        "median": np.median(entropies_array),
                        "q25": np.percentile(entropies_array, 25),
                        "q75": np.percentile(entropies_array, 75),
                        "count": len(entropies_array),
                    }

        return sample_statistics

    def visualize_step_entropy_curves(
        self, save_path: str, figsize: Tuple[int, int] = (14, 8)
    ) -> None:
        """
        Visualize entropy change curves for each agent at each agent sequence step.

        Args:
            save_path: Path to save the visualization
            figsize: Figure size (width, height)
        """
        if not self.entropy_data:
            self.load_entropy_data()

        step_statistics = self.calculate_step_statistics()
        if not step_statistics:
            logger.warning("No step statistics available for visualization")
            return

        fig, axes = plt.subplots(
            len(step_statistics), 1, figsize=figsize, squeeze=False
        )
        fig.suptitle("Entropy Change Curves by Agent and Agent Sequence", fontsize=16)

        for idx, (agent_name, sequences) in enumerate(sorted(step_statistics.items())):
            ax = axes[idx, 0]

            sorted_sequences = sorted(sequences.keys())
            means = [sequences[seq]["mean"] for seq in sorted_sequences]
            stds = [sequences[seq]["std"] for seq in sorted_sequences]

            ax.errorbar(
                sorted_sequences,
                means,
                yerr=stds,
                marker="o",
                capsize=5,
                capthick=2,
                linewidth=2,
            )
            ax.fill_between(
                sorted_sequences,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.2,
            )

            ax.set_xlabel("Agent Sequence")
            ax.set_ylabel("Entropy")
            ax.set_title(f"{agent_name} - Entropy Change Across Agent Sequences")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Step entropy curves visualization saved to: {save_path}")

    def visualize_round_entropy_curves(
        self,
        save_path: str,
        agents_per_round: int = 4,
        figsize: Tuple[int, int] = (14, 8),
    ) -> None:
        """
        Visualize entropy change curves for each agent at each round.

        Args:
            save_path: Path to save the visualization
            agents_per_round: Number of agents per round (default: 4)
            figsize: Figure size (width, height)
        """
        if not self.entropy_data:
            self.load_entropy_data()

        round_statistics = self.calculate_round_statistics(agents_per_round)
        if not round_statistics:
            logger.warning("No round statistics available for visualization")
            return

        fig, axes = plt.subplots(
            len(round_statistics), 1, figsize=figsize, squeeze=False
        )
        fig.suptitle("Entropy Change Curves by Agent and Round", fontsize=16)

        for idx, (agent_name, rounds) in enumerate(sorted(round_statistics.items())):
            ax = axes[idx, 0]

            sorted_rounds = sorted(rounds.keys())
            means = [rounds[round_num]["mean"] for round_num in sorted_rounds]
            stds = [rounds[round_num]["std"] for round_num in sorted_rounds]

            ax.errorbar(
                sorted_rounds,
                means,
                yerr=stds,
                marker="o",
                capsize=5,
                capthick=2,
                linewidth=2,
            )
            ax.fill_between(
                sorted_rounds,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.2,
            )

            ax.set_xlabel("Round")
            ax.set_ylabel("Entropy")
            ax.set_title(f"{agent_name} - Entropy Change Across Rounds")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Round entropy curves visualization saved to: {save_path}")

    def visualize_step_entropy_distribution(
        self, save_path: str, figsize: Tuple[int, int] = (16, 10)
    ) -> None:
        """
        Visualize entropy distribution for each agent at each agent sequence.

        Args:
            save_path: Path to save the visualization
            figsize: Figure size (width, height)
        """
        if not self.entropy_data:
            self.load_entropy_data()

        step_statistics = self.calculate_step_statistics()
        if not step_statistics:
            logger.warning("No step statistics available for visualization")
            return

        agents = sorted(step_statistics.keys())
        all_sequences = set()
        for sequences in step_statistics.values():
            all_sequences.update(sequences.keys())
        sorted_sequences = sorted(all_sequences)

        n_agents = len(agents)
        n_sequences = len(sorted_sequences)

        fig, axes = plt.subplots(n_agents, n_sequences, figsize=figsize, squeeze=False)
        fig.suptitle("Entropy Distribution by Agent and Agent Sequence", fontsize=16)

        for agent_idx, agent_name in enumerate(agents):
            for seq_idx, sequence in enumerate(sorted_sequences):
                ax = axes[agent_idx, seq_idx]

                if sequence in step_statistics[agent_name]:
                    stats_data = step_statistics[agent_name][sequence]

                    samples = self.entropy_data.get("samples", {})
                    sequence_entropies = []
                    for sample_data in samples.values():
                        if (
                            agent_name in sample_data
                            and sequence in sample_data[agent_name]
                        ):
                            sequence_entropies.extend(sample_data[agent_name][sequence])

                    if sequence_entropies:
                        ax.hist(
                            sequence_entropies, bins=20, alpha=0.7, edgecolor="black"
                        )
                        ax.axvline(
                            stats_data["mean"],
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label=f'Mean: {stats_data["mean"]:.4f}',
                        )
                        ax.set_xlabel("Entropy")
                        ax.set_ylabel("Frequency")
                        ax.legend(fontsize=8)
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No data",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

                if agent_idx == n_agents - 1:
                    ax.set_xlabel(f"Sequence {sequence}")
                if seq_idx == 0:
                    ax.set_ylabel(agent_name)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Step entropy distribution visualization saved to: {save_path}")

    def visualize_entropy_curves(
        self, save_path: str, figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Visualize entropy change curves for each agent.

        Args:
            save_path: Path to save the visualization
            figsize: Figure size (width, height)
        """
        if not self.entropy_data:
            self.load_entropy_data()

        samples = self.entropy_data.get("samples", {})
        if not samples:
            logger.warning("No sample data available for visualization")
            return

        plt.figure(figsize=figsize)

        for agent_name in sorted(self.agent_names):
            agent_entropies = []

            for sample_id, sample_data in samples.items():
                if agent_name in sample_data:
                    steps_data = sample_data[agent_name]
                    for step, entropies in sorted(steps_data.items()):
                        agent_entropies.extend(entropies)

            if agent_entropies:
                x_values = list(range(len(agent_entropies)))
                plt.plot(
                    x_values,
                    agent_entropies,
                    label=agent_name,
                    marker="o",
                    markersize=3,
                )

        plt.xlabel("Entropy Index")
        plt.ylabel("Entropy")
        plt.title("Entropy Change Curves by Agent")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Entropy curves visualization saved to: {save_path}")

    def visualize_entropy_distribution(
        self, save_path: str, figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Visualize entropy distribution for each agent.

        Args:
            save_path: Path to save the visualization
            figsize: Figure size (width, height)
        """
        if not self.entropy_data:
            self.load_entropy_data()

        agents_data = self.entropy_data.get("agents", {})
        if not agents_data:
            logger.warning("No agent data available for visualization")
            return

        plt.figure(figsize=figsize)

        all_entropies = []
        agent_labels = []

        for agent_name in sorted(agents_data.keys()):
            entropies = agents_data[agent_name]
            all_entropies.extend(entropies)
            agent_labels.extend([agent_name] * len(entropies))

        if all_entropies:
            df_dict = {"Entropy": all_entropies, "Agent": agent_labels}
            sns.boxplot(x="Agent", y="Entropy", data=df_dict)
            plt.title("Entropy Distribution by Agent")
            plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Entropy distribution visualization saved to: {save_path}")

    def analyze_entropy_accuracy_correlation(
        self, accuracy_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze correlation between entropy and accuracy.

        Args:
            accuracy_data: Dictionary containing accuracy results

        Returns:
            Dictionary containing correlation analysis results
        """
        if not self.entropy_data:
            self.load_entropy_data()

        samples = self.entropy_data.get("samples", {})
        if not samples:
            logger.warning("No sample data available for correlation analysis")
            return {}

        sample_entropies = []
        sample_accuracies = []

        sample_results = accuracy_data.get("sample_results", [])
        sample_id_to_accuracy = {}
        for idx, sample_result in enumerate(sample_results):
            sample_id = sample_result.get("sample_id", f"sample_{idx}")
            sample_id_to_accuracy[sample_id] = sample_result.get("is_correct", False)

        for sample_id, sample_data in samples.items():
            sample_entropy = []
            for agent_name, steps_data in sample_data.items():
                for step, entropies in steps_data.items():
                    sample_entropy.extend(entropies)

            if sample_entropy:
                avg_entropy = np.mean(sample_entropy)
                sample_entropies.append(avg_entropy)

                is_correct = sample_id_to_accuracy.get(sample_id, False)
                sample_accuracies.append(1.0 if is_correct else 0.0)

        if len(sample_entropies) < 2:
            logger.warning("Insufficient data for correlation analysis")
            return {}

        pearson_corr, pearson_p = pearsonr(sample_entropies, sample_accuracies)
        spearman_corr, spearman_p = spearmanr(sample_entropies, sample_accuracies)

        return {
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
            "sample_count": len(sample_entropies),
        }

    def visualize_entropy_accuracy_correlation(
        self,
        accuracy_data: Dict[str, Any],
        save_path: str,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """
        Visualize correlation between entropy and accuracy.

        Args:
            accuracy_data: Dictionary containing accuracy results
            save_path: Path to save the visualization
            figsize: Figure size (width, height)
        """
        if not self.entropy_data:
            self.load_entropy_data()

        samples = self.entropy_data.get("samples", {})
        if not samples:
            logger.warning("No sample data available for correlation visualization")
            return

        sample_entropies = []
        sample_accuracies = []

        sample_results = accuracy_data.get("sample_results", [])
        sample_id_to_accuracy = {}
        for idx, sample_result in enumerate(sample_results):
            sample_id = sample_result.get("sample_id", f"sample_{idx}")
            sample_id_to_accuracy[sample_id] = sample_result.get("is_correct", False)

        for sample_id, sample_data in samples.items():
            sample_entropy = []
            for agent_name, steps_data in sample_data.items():
                for step, entropies in steps_data.items():
                    sample_entropy.extend(entropies)

            if sample_entropy:
                avg_entropy = np.mean(sample_entropy)
                sample_entropies.append(avg_entropy)

                is_correct = sample_id_to_accuracy.get(sample_id, False)
                sample_accuracies.append(1.0 if is_correct else 0.0)

        if not sample_entropies:
            return

        plt.figure(figsize=figsize)

        correct_indices = [i for i, acc in enumerate(sample_accuracies) if acc == 1.0]
        incorrect_indices = [i for i, acc in enumerate(sample_accuracies) if acc == 0.0]

        correct_entropies = [sample_entropies[i] for i in correct_indices]
        incorrect_entropies = [sample_entropies[i] for i in incorrect_indices]

        plt.scatter(
            correct_entropies,
            [1.0] * len(correct_entropies),
            color="green",
            label="Correct",
            alpha=0.6,
        )
        plt.scatter(
            incorrect_entropies,
            [0.0] * len(incorrect_entropies),
            color="red",
            label="Incorrect",
            alpha=0.6,
        )

        plt.xlabel("Average Entropy")
        plt.ylabel("Accuracy (1=Correct, 0=Incorrect)")
        plt.title("Entropy vs Accuracy Correlation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Entropy-accuracy correlation visualization saved to: {save_path}")

    def compare_agents(self) -> Dict[str, Any]:
        """
        Compare entropy patterns between different agents.

        Returns:
            Dictionary containing comparison results
        """
        if not self.entropy_data:
            self.load_entropy_data()

        agents_data = self.entropy_data.get("agents", {})
        if len(agents_data) < 2:
            logger.warning("Need at least 2 agents for comparison")
            return {}

        statistics = self.calculate_statistics()
        comparison = {"statistics": statistics, "pairwise_comparisons": {}}

        agent_names = sorted(agents_data.keys())

        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                agent1 = agent_names[i]
                agent2 = agent_names[j]

                entropies1 = np.array(agents_data[agent1])
                entropies2 = np.array(agents_data[agent2])

                t_stat, t_p = stats.ttest_ind(entropies1, entropies2)
                ks_stat, ks_p = stats.ks_2samp(entropies1, entropies2)

                comparison["pairwise_comparisons"][f"{agent1}_vs_{agent2}"] = {
                    "t_statistic": t_stat,
                    "t_p_value": t_p,
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_p,
                }

        return comparison

    def visualize_agent_comparison(
        self, save_path: str, figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Visualize entropy comparison between agents.

        Args:
            save_path: Path to save the visualization
            figsize: Figure size (width, height)
        """
        if not self.entropy_data:
            self.load_entropy_data()

        agents_data = self.entropy_data.get("agents", {})
        if len(agents_data) < 2:
            logger.warning("Need at least 2 agents for comparison visualization")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Agent Entropy Comparison", fontsize=16)

        agent_names = sorted(agents_data.keys())

        for idx, agent_name in enumerate(agent_names):
            ax = axes[idx // 2, idx % 2]
            entropies = agents_data[agent_name]

            ax.hist(entropies, bins=20, alpha=0.7, edgecolor="black")
            ax.set_title(f"{agent_name} Entropy Distribution")
            ax.set_xlabel("Entropy")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        for idx in range(len(agent_names), 4):
            fig.delaxes(axes[idx // 2, idx % 2])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Agent comparison visualization saved to: {save_path}")

    def generate_entropy_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive entropy analysis report.

        Returns:
            Dictionary containing entropy analysis report
        """
        if not self.entropy_data:
            self.load_entropy_data()

        statistics = self.calculate_statistics()
        step_statistics = self.calculate_step_statistics()
        sample_statistics = self.calculate_sample_statistics()
        comparison = self.compare_agents()

        return {
            "experiment_dir": self.experiment_dir,
            "agent_names": sorted(self.agent_names),
            "overall_statistics": statistics,
            "step_statistics": step_statistics,
            "sample_statistics": sample_statistics,
            "comparison": comparison,
            "total_samples": len(self.entropy_data.get("samples", {})),
        }
