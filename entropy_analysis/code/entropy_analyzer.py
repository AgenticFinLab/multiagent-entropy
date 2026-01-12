"""Entropy analysis module for multi-agent system performance evaluation.

This module provides comprehensive analysis capabilities for examining entropy
characteristics in multi-agent systems. It includes statistical analysis,
correlation studies, evolution tracking, and machine learning-based insights
to understand how entropy relates to system performance across different
architectures and collaboration patterns.
"""

import warnings
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from constants import ARCHITECTURES, MULTI_AGENT_ARCHITECTURES, SINGLE_AGENT_ARCHITECTURES

warnings.filterwarnings("ignore")


class EntropyAnalyzer:
    """Analyzes entropy characteristics in multi-agent system performance data.

    This class provides methods to analyze entropy features from multiple
    perspectives: architectural differences, correlation with accuracy,
    evolution across rounds, collaboration patterns, and sample-architecture
    interactions. It also supports advanced analysis through PCA and clustering.

    Attributes:
        data: DataFrame containing the preprocessed experimental data.
        results: Dictionary storing analysis results.
        architectures: List of available architecture types.
    """

    def __init__(self, data: pd.DataFrame):
        """Initialize the EntropyAnalyzer with experimental data.

        Args:
            data: DataFrame containing preprocessed experimental data.
        """
        self.data = data
        self.results = {}
        self.architectures = ARCHITECTURES

    def analyze_architecture_differences(self) -> Dict[str, pd.DataFrame]:
        """Analyze differences in entropy features across different architectures.

        Performs statistical analysis including mean, standard deviation, and
        median calculations for each architecture, followed by ANOVA tests to
        identify significant differences.

        Returns:
            Dictionary containing statistics DataFrame and ANOVA results.
        """
        print("Analyzing entropy feature differences across architectures...")

        entropy_features = [
            col for col in self.data.columns if "entropy" in col.lower()
        ]

        results = {}
        for arch in self.architectures:
            arch_data = self.data[self.data["architecture"] == arch]
            results[arch] = {
                "mean": arch_data[entropy_features].mean(),
                "std": arch_data[entropy_features].std(),
                "median": arch_data[entropy_features].median(),
            }

        stats_df = pd.DataFrame(
            {arch: results[arch]["mean"] for arch in self.architectures}
        )

        anova_results = {}
        for feature in entropy_features:
            groups = [
                self.data[self.data["architecture"] == arch][feature].dropna()
                for arch in self.architectures
            ]
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                anova_results[feature] = {"F_statistic": f_stat, "p_value": p_value}
            except Exception:
                pass

        self.results["architecture_differences"] = {
            "statistics": stats_df,
            "anova": anova_results,
        }

        return self.results["architecture_differences"]

    def analyze_entropy_accuracy_correlation(self) -> Dict[str, pd.DataFrame]:
        """Analyze correlation between entropy features and accuracy.

        Calculates Pearson correlation coefficients between all entropy
        features and the experimental accuracy metric.

        Returns:
            Dictionary containing correlation DataFrame and significant features.
        """
        print("Analyzing correlation between entropy features and accuracy...")

        entropy_features = [
            col for col in self.data.columns if "entropy" in col.lower()
        ]

        correlations = {}
        for feature in entropy_features:
            corr = self.data[feature].corr(self.data["exp_accuracy"])
            correlations[feature] = corr

        corr_df = pd.DataFrame.from_dict(
            correlations, orient="index", columns=["correlation"]
        )
        corr_df = corr_df.sort_values("correlation", ascending=False)

        significant_features = corr_df[abs(corr_df["correlation"]) > 0.1]

        self.results["entropy_accuracy_correlation"] = {
            "correlations": corr_df,
            "significant_features": significant_features,
        }

        return self.results["entropy_accuracy_correlation"]

    def analyze_round_entropy_evolution(self) -> Dict[str, pd.DataFrame]:
        """Analyze entropy evolution across multiple processing rounds.

        Tracks how entropy values change throughout the multi-round processing
        and compares evolution patterns between correct and incorrect samples.

        Returns:
            Dictionary containing overall statistics and separate statistics
            for correct and incorrect samples.
        """
        print("Analyzing entropy evolution across processing rounds...")

        round_data = (
            self.data.groupby(["sample_id", "agent_round_number"])
            .agg(
                {
                    "round_total_entropy": "first",
                    "round_avg_entropy": "first",
                    "round_total_time": "first",
                    "round_total_token": "first",
                    "is_finally_correct": "first",
                }
            )
            .reset_index()
        )

        round_stats = round_data.groupby("agent_round_number").agg(
            {
                "round_total_entropy": ["mean", "std", "median"],
                "round_avg_entropy": ["mean", "std", "median"],
                "round_total_time": ["mean", "std", "median"],
                "round_total_token": ["mean", "std", "median"],
                "is_finally_correct": "mean",
            }
        )

        correct_round_data = round_data[round_data["is_finally_correct"] == True]
        incorrect_round_data = round_data[round_data["is_finally_correct"] == False]

        correct_stats = correct_round_data.groupby("agent_round_number").agg(
            {
                "round_total_entropy": "mean",
                "round_avg_entropy": "mean",
                "round_total_time": "mean",
                "round_total_token": "mean",
            }
        )

        incorrect_stats = incorrect_round_data.groupby("agent_round_number").agg(
            {
                "round_total_entropy": "mean",
                "round_avg_entropy": "mean",
                "round_total_time": "mean",
                "round_total_token": "mean",
            }
        )

        self.results["round_entropy_evolution"] = {
            "overall_stats": round_stats,
            "correct_samples": correct_stats,
            "incorrect_samples": incorrect_stats,
        }

        return self.results["round_entropy_evolution"]

    def analyze_collaboration_patterns(self) -> Dict[str, pd.DataFrame]:
        """Compare entropy characteristics between different collaboration patterns.

        Analyzes differences between multi-agent architectures and single-agent
        systems, as well as comparisons among different multi-agent architectures.

        Returns:
            Dictionary containing multi-agent vs single-agent comparison and
            architecture-specific comparison.
        """
        print("Comparing entropy characteristics across collaboration patterns...")

        multi_agent_data = self.data[self.data["architecture"].isin(MULTI_AGENT_ARCHITECTURES)]
        single_agent_data = self.data[self.data["architecture"].isin(SINGLE_AGENT_ARCHITECTURES)]

        entropy_features = [
            col for col in self.data.columns if "entropy" in col.lower()
        ]

        comparison = {}
        for feature in entropy_features:
            multi_mean = multi_agent_data[feature].mean()
            single_mean = single_agent_data[feature].mean()

            t_stat, p_value = stats.ttest_ind(
                multi_agent_data[feature].dropna(), single_agent_data[feature].dropna()
            )

            comparison[feature] = {
                "multi_agent_mean": multi_mean,
                "single_agent_mean": single_mean,
                "difference": multi_mean - single_mean,
                "t_statistic": t_stat,
                "p_value": p_value,
            }

        comparison_df = pd.DataFrame(comparison).T

        arch_comparison = {}
        for arch in MULTI_AGENT_ARCHITECTURES:
            arch_data = self.data[self.data["architecture"] == arch]
            arch_comparison[arch] = {
                "mean_entropy": arch_data["sample_mean_entropy"].mean(),
                "std_entropy": arch_data["sample_std_entropy"].mean(),
                "accuracy": arch_data["exp_accuracy"].mean(),
                "avg_tokens": arch_data["sample_all_agents_token_count"].mean(),
            }

        arch_comparison_df = pd.DataFrame(arch_comparison).T

        self.results["collaboration_patterns"] = {
            "multi_vs_single": comparison_df,
            "arch_comparison": arch_comparison_df,
        }

        return self.results["collaboration_patterns"]

    def analyze_sample_architecture_interaction(self) -> Dict[str, pd.DataFrame]:
        """Analyze interaction between sample characteristics and architecture.

        Examines how entropy-accuracy correlations vary across architectures
        and compares performance on high vs low entropy samples.

        Returns:
            Dictionary containing correlation by architecture and entropy
            level comparison.
        """
        print("Analyzing sample-architecture interaction effects...")

        sample_features = [
            col for col in self.data.columns if col.startswith("sample_")
        ]

        interaction_results = {}
        for arch in self.architectures:
            arch_data = self.data[self.data["architecture"] == arch]

            arch_results = {}
            for feature in sample_features:
                if "entropy" in feature:
                    corr = arch_data[feature].corr(
                        arch_data["is_finally_correct"].astype(int)
                    )
                    arch_results[feature] = corr

            interaction_results[arch] = arch_results

        interaction_df = pd.DataFrame(interaction_results)

        high_entropy_samples = self.data[
            self.data["sample_mean_entropy"]
            > self.data["sample_mean_entropy"].quantile(0.75)
        ]
        low_entropy_samples = self.data[
            self.data["sample_mean_entropy"]
            < self.data["sample_mean_entropy"].quantile(0.25)
        ]

        high_entropy_accuracy = high_entropy_samples.groupby("architecture")[
            "exp_accuracy"
        ].mean()
        low_entropy_accuracy = low_entropy_samples.groupby("architecture")[
            "exp_accuracy"
        ].mean()

        entropy_level_comparison = pd.DataFrame(
            {"high_entropy": high_entropy_accuracy, "low_entropy": low_entropy_accuracy}
        )

        self.results["sample_architecture_interaction"] = {
            "correlation_by_arch": interaction_df,
            "entropy_level_comparison": entropy_level_comparison,
        }

        return self.results["sample_architecture_interaction"]

    def perform_pca_analysis(self) -> Dict[str, object]:
        """Perform Principal Component Analysis on entropy features.

        Reduces dimensionality of entropy features to identify underlying
        patterns and relationships.

        Returns:
            Dictionary containing PCA loadings, explained variance, and
            transformed data.
        """
        print("Performing Principal Component Analysis...")

        entropy_features = [
            col for col in self.data.columns if "entropy" in col.lower()
        ]
        entropy_data = self.data[entropy_features].dropna()

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(entropy_data)

        pca = PCA(n_components=0.95)
        pca_result = pca.fit_transform(scaled_data)

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            index=entropy_features,
        )

        explained_variance = pd.DataFrame(
            {
                "PC": [f"PC{i+1}" for i in range(pca.n_components_)],
                "Explained_Variance_Ratio": pca.explained_variance_ratio_,
                "Cumulative_Variance": np.cumsum(pca.explained_variance_ratio_),
            }
        )

        self.results["pca_analysis"] = {
            "loadings": loadings,
            "explained_variance": explained_variance,
            "pca_result": pca_result,
        }

        return self.results["pca_analysis"]

    def perform_clustering_analysis(self, n_clusters: int = 3) -> Dict[str, object]:
        """Perform K-means clustering analysis on entropy features.

        Groups samples into clusters based on entropy characteristics and
        analyzes cluster properties.

        Args:
            n_clusters: Number of clusters to create.

        Returns:
            Dictionary containing cluster statistics, architecture distribution,
            and cluster assignments.
        """
        print("Performing clustering analysis...")

        entropy_features = [
            col for col in self.data.columns if "entropy" in col.lower()
        ]
        entropy_data = self.data[entropy_features].dropna()

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(entropy_data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        entropy_data["cluster"] = clusters
        entropy_data["architecture"] = self.data.loc[entropy_data.index, "architecture"]
        entropy_data["exp_accuracy"] = self.data.loc[entropy_data.index, "exp_accuracy"]

        cluster_stats = entropy_data.groupby("cluster").agg(
            {
                "exp_accuracy": "mean",
                "architecture": lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
            }
        )

        cluster_arch_dist = pd.crosstab(
            entropy_data["cluster"], entropy_data["architecture"]
        )

        self.results["clustering_analysis"] = {
            "cluster_stats": cluster_stats,
            "cluster_arch_distribution": cluster_arch_dist,
            "cluster_assignments": clusters,
        }

        return self.results["clustering_analysis"]

    def generate_comprehensive_report(self) -> Dict[str, object]:
        """Generate a comprehensive analysis report.

        Executes all analysis methods and combines results into a single report.

        Returns:
            Dictionary containing results from all analysis methods.
        """
        print("Generating comprehensive analysis report...")

        report = {
            "architecture_differences": self.analyze_architecture_differences(),
            "entropy_accuracy_correlation": self.analyze_entropy_accuracy_correlation(),
            "round_entropy_evolution": self.analyze_round_entropy_evolution(),
            "collaboration_patterns": self.analyze_collaboration_patterns(),
            "sample_architecture_interaction": (
                self.analyze_sample_architecture_interaction()
            ),
            "pca_analysis": self.perform_pca_analysis(),
            "clustering_analysis": self.perform_clustering_analysis(),
        }

        return report

    def save_results(self, output_dir: str) -> None:
        """Save analysis results to CSV files.

        Args:
            output_dir: Directory path where results should be saved.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for key, value in self.results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.DataFrame):
                        sub_value.to_csv(
                            output_dir / f"{key}_{sub_key}.csv", index=True
                        )
                    elif isinstance(sub_value, (pd.Series, dict)):
                        pd.DataFrame([sub_value]).to_csv(
                            output_dir / f"{key}_{sub_key}.csv", index=False
                        )
            elif isinstance(value, pd.DataFrame):
                value.to_csv(output_dir / f"{key}.csv", index=True)

        print(f"Analysis results saved to: {output_dir}")
