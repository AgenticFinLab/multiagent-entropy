"""Aggregation analysis script: Aggregates all experimental results in multiagent-entropy/data_mining/results/

Functionality:
1. Traverses all experiment folders under results/
2. Reads classification and shap results for each experiment
3. Aggregates feature importance and SHAP value information
4. Calculates statistical metrics for SHAP values (positive and negative impacts)
5. Outputs to the corresponding experiment folder in aggregated_results/
"""

import warnings
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class ExperimentAggregator:
    """Experiment result aggregator."""

    def __init__(self, results_dir: str, output_dir: str):
        """Initializes the aggregator.

        Args:
            results_dir: Root directory for experiment results
            output_dir: Output root directory for aggregated results
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)

    def aggregate_all_experiments(self):
        """Aggregates all experiments."""
        if not self.results_dir.exists():
            print(
                f"Error: Experiment results directory does not exist: {self.results_dir}"
            )
            return

        # Traverse all experiment folders
        exp_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]

        if not exp_dirs:
            print(f"Warning: No experiment folders found: {self.results_dir}")
            return

        print(f"Found {len(exp_dirs)} experiment folders")

        successful = 0
        failed = 0

        for exp_dir in sorted(exp_dirs):
            exp_name = exp_dir.name
            print(f"\n{'='*80}")
            print(f"Processing experiment: {exp_name}")
            print(f"{'='*80}")

            try:
                self.aggregate_experiment(exp_name)
                successful += 1
                print(f"Successfully aggregated experiment: {exp_name}")
            except Exception as e:
                failed += 1
                print(f"Failed: {exp_name}")
                print(f"   Error message: {str(e)}")

        print(f"\n{'='*80}")
        print(f"Aggregation completion statistics:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Total: {len(exp_dirs)}")
        print(f"{'='*80}\n")

    def aggregate_experiment(self, exp_name: str):
        """Aggregates a single experiment.

        Args:
            exp_name: Name of the experiment (folder name)
        """
        exp_path = self.results_dir / exp_name

        # Define file paths
        files = {
            "lightgbm_importance": exp_path
            / "classification"
            / "Feature_Importance_-_LightGBM_(Classification).csv",
            "xgboost_importance": exp_path
            / "classification"
            / "Feature_Importance_-_XGBoost_(Classification).csv",
            "lightgbm_shap_importance": exp_path
            / "shap"
            / "shap_feature_importance_LightGBM_classification.csv",
            "xgboost_shap_importance": exp_path
            / "shap"
            / "shap_feature_importance_XGBoost_classification.csv",
            "lightgbm_shap_values": exp_path
            / "shap"
            / "shap_values_LightGBM_classification.csv",
            "xgboost_shap_values": exp_path
            / "shap"
            / "shap_values_XGBoost_classification.csv",
        }

        # Check if files exist
        missing_files = [name for name, path in files.items() if not path.exists()]
        if missing_files:
            print(f"Skipping experiment {exp_name}: Missing files {missing_files}")
            return

        # Read data
        print("Reading data files...")
        data = {}
        for name, path in files.items():
            try:
                data[name] = pd.read_csv(path)
                print(f"   ✓ {name}: {len(data[name])} rows")
            except Exception as e:
                raise Exception(f"Failed to read file {path}: {str(e)}")

        # Aggregate feature importance
        print("Aggregating feature importance...")
        summary_df = self._aggregate_feature_importance(data)

        # Add SHAP value statistics
        print("Calculating SHAP value statistics...")
        summary_df = self._add_shap_statistics(summary_df, data)

        # Save results
        output_file = self.output_dir / f"{exp_name}.csv"

        summary_df.to_csv(output_file, index=False)
        print(f"Saving results to: {output_file}")
        print(f"   Total {len(summary_df)} features")

    def _aggregate_feature_importance(
        self, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Aggregates feature importance.

        Args:
            data: Dictionary containing all data

        Returns:
            Aggregated DataFrame
        """
        # Extract feature importance
        lgb_imp = data["lightgbm_importance"].rename(
            columns={"Feature": "feature", "Importance": "lightgbm_importance"}
        )
        xgb_imp = data["xgboost_importance"].rename(
            columns={"Feature": "feature", "Importance": "xgboost_importance"}
        )

        # Extract SHAP importance
        lgb_shap_imp = data["lightgbm_shap_importance"].rename(
            columns={"Feature": "feature", "Mean_Abs_SHAP": "lightgbm_mean_abs_shap"}
        )
        xgb_shap_imp = data["xgboost_shap_importance"].rename(
            columns={"Feature": "feature", "Mean_Abs_SHAP": "xgboost_mean_abs_shap"}
        )

        # Merge all data
        summary = lgb_imp.merge(xgb_imp, on="feature", how="outer")
        summary = summary.merge(lgb_shap_imp, on="feature", how="outer")
        summary = summary.merge(xgb_shap_imp, on="feature", how="outer")

        # Fill missing values
        summary = summary.fillna(0)

        # Calculate averages
        summary["mean_importance"] = (
            summary["lightgbm_importance"] + summary["xgboost_importance"]
        ) / 2

        # Normalize importance values (Min-Max normalization to 0-1 range)
        lgb_min, lgb_max = summary["lightgbm_importance"].min(), summary["lightgbm_importance"].max()
        xgb_min, xgb_max = summary["xgboost_importance"].min(), summary["xgboost_importance"].max()

        lgb_norm = (summary["lightgbm_importance"] - lgb_min) / (lgb_max - lgb_min) if lgb_max > lgb_min else summary["lightgbm_importance"] * 0
        xgb_norm = (summary["xgboost_importance"] - xgb_min) / (xgb_max - xgb_min) if xgb_max > xgb_min else summary["xgboost_importance"] * 0

        summary["mean_importance_normalized"] = (lgb_norm + xgb_norm) / 2

        summary["mean_mean_abs_shap"] = (
            summary["lightgbm_mean_abs_shap"] + summary["xgboost_mean_abs_shap"]
        ) / 2

        # Sort by average SHAP importance in descending order
        summary = summary.sort_values("mean_mean_abs_shap", ascending=False)

        return summary

    def _add_shap_statistics(
        self, summary_df: pd.DataFrame, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Adds SHAP value statistics.

        Args:
            summary_df: Summary DataFrame
            data: Dictionary containing SHAP values

        Returns:
            DataFrame with added statistics
        """
        lgb_shap_values = data["lightgbm_shap_values"]
        xgb_shap_values = data["xgboost_shap_values"]

        # Remove non-feature columns (e.g., sample_index)
        non_feature_cols = ["sample_index"]
        lgb_features = [
            col for col in lgb_shap_values.columns if col not in non_feature_cols
        ]
        xgb_features = [
            col for col in xgb_shap_values.columns if col not in non_feature_cols
        ]

        # Calculate statistics for each feature
        lgb_stats = self._calculate_shap_stats(
            lgb_shap_values[lgb_features], "lightgbm"
        )
        xgb_stats = self._calculate_shap_stats(xgb_shap_values[xgb_features], "xgboost")

        # Merge statistical data
        summary_df = summary_df.merge(lgb_stats, on="feature", how="left")
        summary_df = summary_df.merge(xgb_stats, on="feature", how="left")

        # Calculate combined metrics
        summary_df = self._calculate_combined_metrics(summary_df)

        return summary_df

    def _calculate_shap_stats(
        self, shap_df: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        """Calculates statistics for SHAP values.

        Statistical indicators explanation:
        - mean_shap: Mean of SHAP values (positive values indicate positive impact, negative values indicate negative impact)
        - positive_ratio: Proportion of samples with positive SHAP values
        - negative_ratio: Proportion of samples with negative SHAP values
        - mean_positive_shap: Average of positive SHAP values
        - mean_negative_shap: Average of negative SHAP values
        - std_shap: Standard deviation of SHAP values
        - impact_direction: Main impact direction (positive/negative/mixed)

        Args:
            shap_df: SHAP values DataFrame
            model_name: Model name (used for column name prefix)

        Returns:
            DataFrame containing statistics
        """
        stats = []

        for feature in shap_df.columns:
            values = shap_df[feature].values

            # Basic statistics
            mean_shap = np.mean(values)
            std_shap = np.std(values)

            # Positive and negative value statistics
            positive_mask = values > 0
            negative_mask = values < 0

            positive_ratio = np.mean(positive_mask)
            negative_ratio = np.mean(negative_mask)

            mean_positive = (
                np.mean(values[positive_mask]) if np.any(positive_mask) else 0
            )
            mean_negative = (
                np.mean(values[negative_mask]) if np.any(negative_mask) else 0
            )

            # Determine impact direction
            if positive_ratio > 0.6:
                direction = "positive"
            elif negative_ratio > 0.6:
                direction = "negative"
            else:
                direction = "mixed"

            stats.append(
                {
                    "feature": feature,
                    f"{model_name}_mean_shap": mean_shap,
                    f"{model_name}_std_shap": std_shap,
                    f"{model_name}_positive_ratio": positive_ratio,
                    f"{model_name}_negative_ratio": negative_ratio,
                    f"{model_name}_mean_positive_shap": mean_positive,
                    f"{model_name}_mean_negative_shap": mean_negative,
                    f"{model_name}_impact_direction": direction,
                }
            )

        return pd.DataFrame(stats)

    def _calculate_combined_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates combined metrics.

        Args:
            df: DataFrame containing statistics from various models

        Returns:
            DataFrame with added combined metrics
        """
        # Average of mean SHAP values
        df["mean_shap"] = (
            df.get("lightgbm_mean_shap", 0) + df.get("xgboost_mean_shap", 0)
        ) / 2

        # Average positive sample ratio
        df["mean_positive_ratio"] = (
            df.get("lightgbm_positive_ratio", 0) + df.get("xgboost_positive_ratio", 0)
        ) / 2

        # Average negative sample ratio
        df["mean_negative_ratio"] = (
            df.get("lightgbm_negative_ratio", 0) + df.get("xgboost_negative_ratio", 0)
        ) / 2

        # Overall impact direction judgment
        def determine_overall_direction(row):
            lgb_dir = row.get("lightgbm_impact_direction", "mixed")
            xgb_dir = row.get("xgboost_impact_direction", "mixed")

            if lgb_dir == xgb_dir:
                return lgb_dir
            elif lgb_dir == "mixed" or xgb_dir == "mixed":
                return "mixed"
            else:
                return "mixed"

        df["overall_impact_direction"] = df.apply(determine_overall_direction, axis=1)

        # Impact strength (considering both importance and direction consistency)
        # impact_strength = feature_importance * (1 - direction_mismatch)
        # This metric helps to find features with strong impact and consistent direction
        df["impact_strength"] = df["mean_mean_abs_shap"] * (
            1 - abs(df["mean_positive_ratio"] - df["mean_negative_ratio"])
        )

        # Reorder columns
        base_cols = [
            "feature",
            "lightgbm_importance",
            "xgboost_importance",
            "mean_importance",
            "mean_importance_normalized",
            "lightgbm_mean_abs_shap",
            "xgboost_mean_abs_shap",
            "mean_mean_abs_shap",
        ]

        shap_stat_cols = [
            "mean_shap",
            "mean_positive_ratio",
            "mean_negative_ratio",
            "overall_impact_direction",
            "impact_strength",
        ]

        detailed_cols = [
            col for col in df.columns if col not in base_cols + shap_stat_cols
        ]

        return df[base_cols + shap_stat_cols + detailed_cols]


def main():
    """Main function."""
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Set paths
    results_dir = project_root / "results"
    output_dir = project_root / "results_aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Experiment results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Create aggregator and run
    aggregator = ExperimentAggregator(str(results_dir), str(output_dir))
    aggregator.aggregate_all_experiments()


if __name__ == "__main__":
    main()
