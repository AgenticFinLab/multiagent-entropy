"""LLM-based analysis of visualization images.

This module provides functionality to analyze visualization images using LLMs,
specifically designed to extract insights from feature importance and SHAP value plots.
"""

import os
import json
import base64
import statistics
from pathlib import Path
from collections import Counter, defaultdict

from openai import OpenAI


class VisualizationSummarizer:
    """Class to handle LLM-based analysis of visualization images."""

    def __init__(self, input_dir: str, output_dir: str):
        """Initializes the summarizer.

        Args:
            input_dir: Directory containing visualization images to analyze
            output_dir: Directory to store LLM analysis results
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_visualizations_with_llm(self, n: int = 5):
        """Analyzes generated visualizations using ByteDance Ark OpenAI API.

        Args:
            n: Number of most important features to identify.
        """
        api_key = os.getenv("ARK_API_KEY")
        if not api_key:
            print(
                "\nError: ARK_API_KEY environment variable not set. Skipping LLM analysis."
            )
            print("Please set it with: export ARK_API_KEY='your-api-key'")
            return

        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
        )

        # Find all PNG files in input directory
        image_files = sorted(list(self.input_dir.glob("*.png")))
        if not image_files:
            print(f"\nWarning: No visualization images found in {self.input_dir}")
            print("Please run visualizations first by generating visualization images.")
            return

        print(f"\n{'='*80}")
        print(f"Starting LLM Analysis (using doubao-seed-1-8-251228)")
        print(f"Target: {len(image_files)} images, top {n} features each")
        print(f"{'='*80}")

        # Try to load existing results to continue from where we left off
        summary_path = self.output_dir / "summary.json"
        all_analysis_results = {}
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    all_analysis_results = json.load(f)
                print(
                    f"Loaded {len(all_analysis_results)} existing results from {summary_path}"
                )
            except Exception as e:
                print(
                    f"Could not load existing results file: {str(e)}. Starting fresh."
                )
                all_analysis_results = {}

        for image_path in image_files:
            exp_name = image_path.stem

            # Skip if this image has already been processed
            if exp_name in all_analysis_results:
                print(f"Skipping {exp_name} (already processed)")
                continue

            print(f"Processing: {exp_name}...")

            try:
                # Read and encode image to base64
                with open(image_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode("utf-8")

                # Format prompt as requested
                prompt = f"""
Analyze the visualization image containing four subplots:

1. Feature Importance Comparison (top-left): Shows normalized importance scores for features across LightGBM, XGBoost, and Mean Normalized metrics.
2. SHAP Importance - XGBoost (top-right): Displays SHAP feature importance for the XGBoost model using dot plots.
3. SHAP Value Impact (bottom-left): Shows mean SHAP values for features, with positive effects in green and negative effects in red.
4. SHAP Importance - LightGBM (bottom-right): Displays SHAP feature importance for the LightGBM model using dot plots.

Based on these visualizations, identify the top {n} most important features considering both the importance scores and SHAP values.

Return ONLY a JSON array with these fields for each feature:
- rank: (number) feature rank (1 to {n})
- feature_name: (string) name of the feature
- shap_correlation: (number) correlation between feature value and SHAP value based on SHAP importance (both XGBoost and LightGBM). Positive values indicate positive correlation, negative values indicate negative correlation, and the magnitude represents the strength.
- overall_direction: (number) overall effect direction based on SHAP value impact. Positive values indicate positive correlation, negative values indicate negative correlation, and the magnitude represents the strength.
- reason: (string) reason for the shap_correlation magnitude and overall_direction based on the visualizations

Example JSON output:
```json
[
  {{"rank": 1, "feature_name": "feature1", "shap_correlation": 0.5, "overall_direction": 0.8, "reason": ""}},
  {{"rank": 2, "feature_name": "feature2", "shap_correlation": -0.3, "overall_direction": -0.6, "reason": ""}}
]
```
"""

                # Call API using the specific format provided in the example
                response = client.responses.create(
                    model="doubao-seed-1-8-251228",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{base64_image}",
                                },
                                {"type": "input_text", "text": prompt},
                            ],
                        }
                    ],
                )
                # Extract content from response
                content_text = ""
                # Handle the Ark responses API structure
                try:
                    if hasattr(response, "output") and response.output:
                        # Look through each item in the output array
                        for output_item in response.output:
                            # Check if this is a message-type output with content
                            if hasattr(output_item, "content") and output_item.content:
                                # Process each content item in the message
                                for content_item in output_item.content:
                                    if (
                                        hasattr(content_item, "type")
                                        and content_item.type == "output_text"
                                        and hasattr(content_item, "text")
                                    ):
                                        content_text = content_item.text
                                        break
                            if content_text:  # Found content, exit outer loop
                                break
                    elif hasattr(response, "choices"):
                        # Handle standard OpenAI API response structure
                        if response.choices and len(response.choices) > 0:
                            first_choice = response.choices[0]
                            if hasattr(first_choice, "message") and hasattr(
                                first_choice.message, "content"
                            ):
                                content_text = first_choice.message.content
                except (AttributeError, IndexError, TypeError) as e:
                    print(
                        f"   Error extracting content from response for {exp_name}: {str(e)}"
                    )
                    continue

                if not content_text:
                    print(f"   Warning: No text content returned for {exp_name}")
                    continue

                # Parse JSON results
                json_str = content_text.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()

                try:
                    analysis_result = json.loads(json_str)
                    # Check if analysis_result is a list (as expected) and not None
                    if analysis_result is None:
                        print(f"   Warning: Parsed JSON is null for {exp_name}")
                        continue
                    if not isinstance(analysis_result, list):
                        print(
                            f"   Warning: Expected list but got {type(analysis_result)} for {exp_name}"
                        )
                        continue
                    all_analysis_results[exp_name] = analysis_result
                    print(f"   Successfully analyzed {exp_name}")

                    # Save to summary.json after each successful analysis
                    try:
                        with open(summary_path, "w", encoding="utf-8") as f:
                            json.dump(
                                all_analysis_results, f, indent=4, ensure_ascii=False
                            )
                        print(
                            f"   Results saved to: {summary_path} (current progress: {len(all_analysis_results)} items)"
                        )
                    except Exception as save_error:
                        print(
                            f"   Error saving intermediate results: {str(save_error)}"
                        )

                except json.JSONDecodeError as je:
                    print(f"   Error parsing JSON for {exp_name}: {str(je)}")
                    print(
                        f"   Content received: {content_text[:200]}..."
                    )  # Print first 200 chars of response
                    continue

            except Exception as e:
                print(f"   Error analyzing {exp_name}: {str(e)}")
                # Continue to next image
                continue

        print(f"\n{'='*80}")
        print(f"LLM analysis completed! Final results saved to: {summary_path}")
        print(f"Total analyzed: {len(all_analysis_results)} items")
        print(f"{'='*80}\n")

    def perform_hierarchical_statistical_analysis(self):
        """Performs hierarchical statistical analysis on the summary data and saves results to summary_sta.md in markdown format."""
        summary_path = self.output_dir / "summary.json"

        # Load the summary data
        if not summary_path.exists():
            print("No summary.json file found. Skipping statistical analysis.")
            return

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # Prepare output file
        output_path = self.output_dir / "summary_sta.md"

        with open(output_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("# Hierarchical Statistical Analysis of Summary Data\n\n")

            # 1. Top-level analysis (all datasets, models, exclude conditions)
            f.write("## 1. Top-Level Comprehensive Analysis\n\n")
            self._analyze_top_level(summary_data, f)

            # 2. Second-level analysis (exclude conditions)
            f.write("\n## 2. Analysis by Exclude Conditions\n\n")
            self._analyze_by_exclude_conditions(summary_data, f)

            # 3. Third-level analysis (datasets)
            f.write("\n## 3. Analysis by Datasets\n\n")
            self._analyze_by_datasets(summary_data, f)

            # 4. Bottom-level analysis (each dataset and model)
            f.write("\n## 4. Detailed Analysis by Dataset and Model\n\n")
            self._analyze_by_dataset_and_model(summary_data, f)

            f.write("\n\n## Analysis Summary Completed\n")

        print(
            f"Hierarchical statistical analysis completed! Results saved to: {output_path}"
        )

    def _analyze_top_level(self, summary_data, output_file):
        """Perform top-level analysis across all datasets, models, and exclude conditions."""
        output_file.write("### Overall Feature Importance Ranking Consistency\n")
        self._analyze_ranking_consistency(summary_data, output_file)

        output_file.write("\n### Overall Feature Impact Direction Distribution\n")
        self._analyze_direction_distribution(summary_data, output_file)

        output_file.write("\n### Overall Top 5 Features Overlap\n")
        self._analyze_overlap(summary_data, output_file)

        output_file.write("\n### Overall SHAP Value Distribution\n")
        self._analyze_shap_distribution(summary_data, output_file)

        output_file.write("\n### Overall Exclude Condition Differences\n")
        self._analyze_exclude_condition_differences(summary_data, output_file)

    def _analyze_by_exclude_conditions(self, summary_data, output_file):
        """Analyze data by exclude conditions (exclude_default vs exclude_base_model_metrics)."""
        # Group datasets by exclude condition
        exclude_default_data = {}
        exclude_base_data = {}

        for experiment_name, features in summary_data.items():
            # More precise identification of exclude condition based on exact match
            if experiment_name.endswith("exclude_default"):
                exclude_default_data[experiment_name] = features
            elif experiment_name.endswith("exclude_base_model_metrics"):
                exclude_base_data[experiment_name] = features
            # Also check for exclude patterns in middle of name
            elif "exclude_default" in experiment_name and not experiment_name.endswith(
                "exclude_default"
            ):
                exclude_default_data[experiment_name] = features
            elif (
                "exclude_base_model_metrics" in experiment_name
                and not experiment_name.endswith("exclude_base_model_metrics")
            ):
                exclude_base_data[experiment_name] = features
            # Handle exclude_{exclude feature group} format (all datasets, all models)
            elif experiment_name.startswith("exclude_"):
                # If it starts with exclude_, add to both groups or determine based on specific pattern
                if "default" in experiment_name:
                    exclude_default_data[experiment_name] = features
                elif "base_model_metrics" in experiment_name:
                    exclude_base_data[experiment_name] = features
                else:
                    # If unsure, add to exclude_default group
                    exclude_default_data[experiment_name] = features

        if exclude_default_data:
            output_file.write("### Analysis for exclude_default condition\n")
            self._analyze_ranking_consistency(exclude_default_data, output_file)
            self._analyze_direction_distribution(exclude_default_data, output_file)
            self._analyze_overlap(exclude_default_data, output_file)
            self._analyze_shap_distribution(exclude_default_data, output_file)

        if exclude_base_data:
            output_file.write(
                "\n### Analysis for exclude_base_model_metrics condition\n"
            )
            self._analyze_ranking_consistency(exclude_base_data, output_file)
            self._analyze_direction_distribution(exclude_base_data, output_file)
            self._analyze_overlap(exclude_base_data, output_file)
            self._analyze_shap_distribution(exclude_base_data, output_file)

        # Compare the two exclude conditions
        if exclude_default_data and exclude_base_data:
            output_file.write("\n### Comparison Between Exclude Conditions\n")
            self._compare_exclude_conditions(
                exclude_default_data, exclude_base_data, output_file
            )

    def _analyze_by_datasets(self, summary_data, output_file):
        """Analyze data by datasets (group by dataset type)."""
        # Group experiments by dataset type
        dataset_groups = defaultdict(dict)
        for experiment_name in summary_data.keys():
            # Extract dataset type from experiment name
            if "_exclude_" in experiment_name and "_dataset_" in experiment_name:
                # Format: model_{model}_dataset_{dataset}_exclude_{condition}
                dataset_start_idx = experiment_name.find("_dataset_") + len("_dataset_")
                exclude_idx = experiment_name.find("_exclude_", dataset_start_idx)
                dataset_type = experiment_name[dataset_start_idx:exclude_idx]
            elif (
                experiment_name.startswith("dataset_")
                and "_exclude_" in experiment_name
            ):
                # Format: dataset_{dataset}_exclude_{condition}
                dataset_start_idx = len("dataset_")
                exclude_idx = experiment_name.find("_exclude_", dataset_start_idx)
                dataset_type = experiment_name[dataset_start_idx:exclude_idx]
            elif experiment_name.startswith("dataset_"):
                # Format: dataset_{dataset}
                dataset_type = experiment_name[len("dataset_") :]
            elif experiment_name.startswith("exclude_"):
                # Format: exclude_{exclude feature group} (all datasets)
                dataset_type = "All_Datasets"  # All datasets
            else:
                # If no explicit dataset in name, group as 'All_Datasets'
                dataset_type = "All_Datasets"

            dataset_groups[dataset_type][experiment_name] = summary_data[
                experiment_name
            ]

        for dataset_type, data in dataset_groups.items():
            output_file.write(f"\n### Analysis for {dataset_type} dataset\n")
            self._analyze_ranking_consistency(data, output_file)
            self._analyze_direction_distribution(data, output_file)
            self._analyze_overlap(data, output_file)
            self._analyze_shap_distribution(data, output_file)

    def _analyze_by_dataset_and_model(self, summary_data, output_file):
        """Provide detailed analysis for each dataset and model combination."""
        for experiment_name, features in summary_data.items():
            output_file.write(f"\n### Detailed Analysis for {experiment_name}\n")

            # Extract model, dataset and exclude condition from experiment name
            model_type = "All Models"  # Default assumption
            dataset_type = "All Datasets"  # Default assumption
            exclude_condition = "unknown"

            # Parse the experiment name according to format:
            # model_{model_name}_dataset_{dataset_name}_exclude_{exclude feature group}
            # Or dataset_{dataset_name}_exclude_{exclude feature group}
            # Or model_{model_name}_exclude_{exclude feature group}
            # Or exclude_{exclude feature group}
            # Or just exclude_{exclude feature group} (all datasets, all models)
            if experiment_name.startswith("model_") and "_dataset_" in experiment_name:
                # Format: model_{model_name}_dataset_{dataset_name}_exclude_{condition}
                model_start_idx = 1  # after 'model'
                model_end_idx = experiment_name.find("_dataset_")
                model_type = experiment_name[model_start_idx:model_end_idx]

                dataset_start_idx = experiment_name.find("_dataset_") + len("_dataset_")
                exclude_idx = experiment_name.find("_exclude_", dataset_start_idx)
                if exclude_idx != -1:
                    dataset_type = experiment_name[dataset_start_idx:exclude_idx]
                    exclude_condition = experiment_name[
                        exclude_idx + len("_exclude_") :
                    ]
                else:
                    dataset_type = experiment_name[dataset_start_idx:]

            elif (
                experiment_name.startswith("dataset_")
                and "_exclude_" in experiment_name
            ):
                # Format: dataset_{dataset_name}_exclude_{condition}
                dataset_start_idx = len("dataset_")
                exclude_idx = experiment_name.find("_exclude_", dataset_start_idx)
                dataset_type = experiment_name[dataset_start_idx:exclude_idx]
                exclude_condition = experiment_name[exclude_idx + len("_exclude_") :]

            elif (
                experiment_name.startswith("model_") and "_exclude_" in experiment_name
            ):
                # Format: model_{model_name}_exclude_{condition}
                model_start_idx = len("model_")
                exclude_idx = experiment_name.find("_exclude_", model_start_idx)
                model_type = experiment_name[model_start_idx:exclude_idx]
                exclude_condition = experiment_name[exclude_idx + len("_exclude_") :]
                dataset_type = "All Datasets"
            
            elif experiment_name.startswith("exclude_"):
                # Format: exclude_{exclude feature group} (all datasets, all models)
                exclude_condition = experiment_name[len("exclude_") + 1:]  # Take everything after "exclude_" (skip the underscore)
                model_type = "All Models"
                dataset_type = "All Datasets"

            output_file.write(
                f"- Model: {model_type}, Dataset: {dataset_type}, Exclude Condition: {exclude_condition}\n\n"
            )

            # Single dataset analysis
            single_dataset_data = {experiment_name: features}
            self._analyze_ranking_consistency(single_dataset_data, output_file)
            self._analyze_direction_distribution(single_dataset_data, output_file)
            self._analyze_overlap(single_dataset_data, output_file)
            self._analyze_shap_distribution(single_dataset_data, output_file)

    def _compare_exclude_conditions(
        self, exclude_default_data, exclude_base_data, output_file
    ):
        """Compare the two exclude conditions."""
        output_file.write("#### Feature Ranking Comparison\n")

        # Get top features from each condition
        def get_top_features(data, n=5):
            feature_counts = Counter()
            for dataset, features in data.items():
                for feature_data in features[:n]:  # Only top n features
                    feature_counts[feature_data["feature_name"]] += 1
            return feature_counts

        top_exclude_default = get_top_features(exclude_default_data)
        top_exclude_base = get_top_features(exclude_base_data)

        output_file.write("**Top features in exclude_default condition:**\n")
        for feature, count in top_exclude_default.most_common(10):
            output_file.write(f"- {feature}: {count} datasets\n")

        output_file.write(
            "\n**Top features in exclude_base_model_metrics condition:**\n"
        )
        for feature, count in top_exclude_base.most_common(10):
            output_file.write(f"- {feature}: {count} datasets\n")

        # Calculate overlap
        common_features = set(top_exclude_default.keys()) & set(top_exclude_base.keys())
        output_file.write(
            f"\n**Common features in both conditions:** {len(common_features)}\n"
        )
        for feature in common_features:
            output_file.write(
                f"- {feature} (exclude_default: {top_exclude_default[feature]}, exclude_base: {top_exclude_base[feature]})\n"
            )

    def _analyze_ranking_consistency(self, summary_data, output_file):
        """Analyze consistency of feature rankings across datasets in markdown format."""
        # Count how many times each feature appears in top positions
        feature_rank_counts = defaultdict(lambda: [0, 0, 0, 0, 0])  # For ranks 1-5
        feature_appearances = Counter()
        # Track shap_correlation and overall_direction for each feature
        feature_shap_values = defaultdict(list)
        feature_direction_values = defaultdict(list)
        total_experiments = len(summary_data)

        for experiment_name, features in summary_data.items():
            for feature_data in features:
                rank = feature_data["rank"]
                feature_name = feature_data["feature_name"]
                if 1 <= rank <= 5:
                    feature_rank_counts[feature_name][rank - 1] += 1
                    feature_appearances[feature_name] += 1
                    # Store shap_correlation and overall_direction for each occurrence
                    feature_shap_values[feature_name].append(feature_data["shap_correlation"])
                    feature_direction_values[feature_name].append(feature_data["overall_direction"])

        # Sort features by total appearances in top 5
        sorted_features = sorted(
            feature_appearances.items(), key=lambda x: x[1], reverse=True
        )

        output_file.write(f"- Total experiments analyzed: {total_experiments}\n")
        output_file.write("- Features appearing in top 5 across experiments:\n")
        for feature, count in sorted_features[:10]:  # Top 10 features
            # Calculate average shap_correlation and representative direction
            avg_shap = sum(feature_shap_values[feature]) / len(feature_shap_values[feature])
            # Calculate average overall_direction value, converting strings to numbers if necessary
            converted_values = []
            for val in feature_direction_values[feature]:
                if isinstance(val, str):
                    # Convert string direction to numeric equivalent
                    if "positive" in val.lower():
                        converted_values.append(abs(float(val.replace("positive", "") or "1")))  # Default to 1 if no number specified
                    elif "negative" in val.lower():
                        converted_values.append(-abs(float(val.replace("negative", "") or "1")))  # Default to -1 if no number specified
                    else:
                        # Try to convert directly to float, default to 0 if fail
                        try:
                            converted_values.append(float(val))
                        except ValueError:
                            converted_values.append(0.0)
                else:
                    converted_values.append(float(val))
            
            avg_direction = sum(converted_values) / len(converted_values) if converted_values else 0.0
            # Determine direction sign for display
            direction_display = "positive" if avg_direction > 0 else "negative"
            # Use average direction value with sign indicator
            most_common_direction = f"{direction_display} ({avg_direction:.3f})"
            
            output_file.write(
                f"  - {feature}: {count}/{total_experiments} experiments ({count/total_experiments*100:.1f}%), avg shap_correlation: {avg_shap:.3f}, overall_direction: {most_common_direction}")

            # Show distribution across ranks
            rank_dist = feature_rank_counts[feature]
            rank_summary = []
            for i, rank_count in enumerate(rank_dist):
                if rank_count > 0:
                    rank_summary.append(f"R{i+1}:{rank_count}")
            if rank_summary:
                output_file.write(f" [Rank distribution: {', '.join(rank_summary)}]")
            output_file.write("\n")

        # Identify features that consistently appear in top positions
        output_file.write(
            f"\n- Features appearing in top 3 in at least 50% of experiments:\n"
        )
        consistent_top_features = [
            feat
            for feat, count in sorted_features
            if count >= total_experiments * 0.5
            and sum(feature_rank_counts[feat][:2]) > 0
        ]  # At least one appearance in top 2
        for feature in consistent_top_features:
            # Calculate average shap_correlation and representative direction
            avg_shap = sum(feature_shap_values[feature]) / len(feature_shap_values[feature])
            # Calculate average overall_direction value, converting strings to numbers if necessary
            converted_values = []
            for val in feature_direction_values[feature]:
                if isinstance(val, str):
                    # Convert string direction to numeric equivalent
                    if "positive" in val.lower():
                        converted_values.append(abs(float(val.replace("positive", "") or "1")))  # Default to 1 if no number specified
                    elif "negative" in val.lower():
                        converted_values.append(-abs(float(val.replace("negative", "") or "1")))  # Default to -1 if no number specified
                    else:
                        # Try to convert directly to float, default to 0 if fail
                        try:
                            converted_values.append(float(val))
                        except ValueError:
                            converted_values.append(0.0)
                else:
                    converted_values.append(float(val))
            
            avg_direction = sum(converted_values) / len(converted_values) if converted_values else 0.0
            # Determine direction sign for display
            direction_display = "positive" if avg_direction > 0 else "negative"
            # Use average direction value with sign indicator
            most_common_direction = f"{direction_display} ({avg_direction:.3f})"
            
            output_file.write(
                f"  - {feature}: {feature_appearances[feature]}/{total_experiments} experiments, avg shap_correlation: {avg_shap:.3f}, overall_direction: {most_common_direction}\n"
            )

    def _analyze_direction_distribution(self, summary_data, output_file):
        """Analyze the distribution of feature impact directions (positive/negative) in markdown format."""
        total_features = 0
        positive_count = 0
        negative_count = 0

        for _, features in summary_data.items():
            for feature_data in features:
                total_features += 1
                # Use numeric direction value for comparison, handling possible string values
                direction_value = feature_data["overall_direction"]
                if isinstance(direction_value, str):
                    # Convert string direction to numeric equivalent
                    if "positive" in direction_value.lower():
                        direction_value = 1
                    elif "negative" in direction_value.lower():
                        direction_value = -1
                    else:
                        try:
                            direction_value = float(direction_value)
                        except ValueError:
                            direction_value = 0
                else:
                    direction_value = float(direction_value)
                
                if direction_value > 0:
                    positive_count += 1
                elif direction_value < 0:
                    negative_count += 1

        output_file.write(f"- Total features analyzed: {total_features}\n")
        output_file.write(
            f"- Positive impacts: {positive_count} ({positive_count/total_features*100:.1f}%)\n"
        )
        output_file.write(
            f"- Negative impacts: {negative_count} ({negative_count/total_features*100:.1f}%)\n"
        )

        # Analyze direction consistency for features appearing multiple times
        feature_directions = defaultdict(list)
        for _, features in summary_data.items():
            for feature_data in features:
                feature_name = feature_data["feature_name"]
                # Use numeric direction value for consistency analysis, handling possible string values
                direction_value = feature_data["overall_direction"]
                if isinstance(direction_value, str):
                    # Convert string direction to numeric equivalent
                    if "positive" in direction_value.lower():
                        direction_value = 1
                    elif "negative" in direction_value.lower():
                        direction_value = -1
                    else:
                        try:
                            direction_value = float(direction_value)
                        except ValueError:
                            direction_value = 0
                else:
                    direction_value = float(direction_value)
                # Directly append the numeric value for consistency analysis
                feature_directions[feature_name].append(direction_value)

        inconsistent_features = []
        for feature, directions in feature_directions.items():
            # Convert directions to signs to check for inconsistencies
            direction_signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in directions]
            unique_signs = set(direction_signs)
            if len(unique_signs) > 1 or (len(unique_signs) == 1 and 0 not in unique_signs and len(set(directions)) > 1):
                inconsistent_features.append((feature, len(directions), directions))

        if inconsistent_features:
            output_file.write(
                f"\n- Features with inconsistent directions across datasets:\n"
            )
            for feature, count, directions in inconsistent_features[:10]:  # Top 10
                # Count positive, negative, and neutral directions
                pos_count = sum(1 for d in directions if d > 0)
                neg_count = sum(1 for d in directions if d < 0)
                neu_count = sum(1 for d in directions if d == 0)
                direction_summary = f"pos:{pos_count}, neg:{neg_count}, zero:{neu_count}"
                output_file.write(
                    f"  - {feature}: {direction_summary} (across {count} instances)\n"
                )
        else:
            output_file.write(
                f"\n- No features showed inconsistent directions across datasets.\n"
            )

    def _analyze_overlap(self, summary_data, output_file):
        """Analyze overlap of top 5 features across different datasets in markdown format."""
        experiment_features = {}
        all_features = set()

        for experiment_name, features in summary_data.items():
            top_features = [f["feature_name"] for f in features[:5]]  # Top 5 features
            experiment_features[experiment_name] = set(top_features)
            all_features.update(top_features)

        output_file.write(f"- Number of experiments: {len(experiment_features)}\n")

        # Calculate pairwise overlaps
        overlaps = []
        experiments_list = list(experiment_features.keys())
        for i in range(len(experiments_list)):
            for j in range(i + 1, len(experiments_list)):
                exp1, exp2 = experiments_list[i], experiments_list[j]
                common_features = experiment_features[exp1] & experiment_features[exp2]
                overlap_pct = (
                    len(common_features) / 5 * 100 if len(common_features) > 0 else 0
                )
                overlaps.append((exp1, exp2, len(common_features), overlap_pct))

        if overlaps:
            avg_overlap = statistics.mean([o[3] for o in overlaps])
            output_file.write(
                f"- Average overlap of top 5 features between experiments: {avg_overlap:.1f}%\n"
            )

            # Show highest and lowest overlaps
            max_overlap = max(overlaps, key=lambda x: x[3])
            min_overlap = min(overlaps, key=lambda x: x[3])
            output_file.write(
                f"- Highest overlap: {max_overlap[0]} vs {max_overlap[1]} - {max_overlap[2]}/5 features ({max_overlap[3]:.1f}%)\n"
            )
            output_file.write(
                f"- Lowest overlap: {min_overlap[0]} vs {min_overlap[1]} - {min_overlap[2]}/5 features ({min_overlap[3]:.1f}%)\n"
            )

        # Identify most common top features across all experiments
        feature_counts = Counter()
        for experiment_name, features in summary_data.items():
            for feature_data in features[:5]:  # Only top 5
                feature_counts[feature_data["feature_name"]] += 1

        output_file.write(
            f"\n- Most common features in top 5 across all experiments:\n"
        )
        for feature, count in feature_counts.most_common(10):
            pct = count / len(experiment_features) * 100
            output_file.write(
                f"  - {feature}: {count}/{len(experiment_features)} experiments ({pct:.1f}%)\n"
            )

    def _analyze_shap_distribution(self, summary_data, output_file):
        """Analyze the distribution of SHAP values in markdown format."""
        all_shap_correlations = []
        positive_shap_correlations = []
        negative_shap_correlations = []

        for dataset, features in summary_data.items():
            for feature_data in features:
                shap_corr = feature_data["shap_correlation"]
                all_shap_correlations.append(shap_corr)
                # Use numeric direction for grouping, handling possible string values
                direction_value = feature_data["overall_direction"]
                if isinstance(direction_value, str):
                    # Convert string direction to numeric equivalent
                    if "positive" in direction_value.lower():
                        direction_value = 1
                    elif "negative" in direction_value.lower():
                        direction_value = -1
                    else:
                        try:
                            direction_value = float(direction_value)
                        except ValueError:
                            direction_value = 0
                else:
                    direction_value = float(direction_value)
                
                if direction_value > 0:
                    positive_shap_correlations.append(shap_corr)
                else:
                    negative_shap_correlations.append(shap_corr)

        output_file.write(f"- SHAP Correlation Statistics:\n")
        if all_shap_correlations:
            output_file.write(f"  - Total values: {len(all_shap_correlations)}\n")
            output_file.write(f"  - Mean: {statistics.mean(all_shap_correlations):.3f}\n")
            output_file.write(f"  - Median: {statistics.median(all_shap_correlations):.3f}\n")
            output_file.write(f"  - Std Dev: {statistics.pstdev(all_shap_correlations):.3f}\n")
            output_file.write(f"  - Min: {min(all_shap_correlations):.3f}\n")
            output_file.write(f"  - Max: {max(all_shap_correlations):.3f}\n")

        output_file.write(
            f"\n- Positive SHAP Correlations (N={len(positive_shap_correlations)}):\n"
        )
        if positive_shap_correlations:
            output_file.write(f"  - Mean: {statistics.mean(positive_shap_correlations):.3f}\n")
            output_file.write(
                f"  - Median: {statistics.median(positive_shap_correlations):.3f}\n"
            )
            output_file.write(
                f"  - Std Dev: {statistics.pstdev(positive_shap_correlations):.3f}\n"
            )
            output_file.write(f"  - Min: {min(positive_shap_correlations):.3f}\n")
            output_file.write(f"  - Max: {max(positive_shap_correlations):.3f}\n")

        output_file.write(
            f"\n- Negative SHAP Correlations (N={len(negative_shap_correlations)}):\n"
        )
        if negative_shap_correlations:
            output_file.write(f"  - Mean: {statistics.mean(negative_shap_correlations):.3f}\n")
            output_file.write(
                f"  - Median: {statistics.median(negative_shap_correlations):.3f}\n"
            )
            output_file.write(
                f"  - Std Dev: {statistics.pstdev(negative_shap_correlations):.3f}\n"
            )
            output_file.write(f"  - Min: {min(negative_shap_correlations):.3f}\n")
            output_file.write(f"  - Max: {max(negative_shap_correlations):.3f}\n")

        # Identify most impactful features by absolute shap correlation
        sorted_by_impact = sorted(
            [(e, f) for e, fs in summary_data.items() for f in fs],
            key=lambda x: abs(x[1]["shap_correlation"]),
            reverse=True,
        )[:10]

        output_file.write(
            f"\n- Top 10 most impactful features (by absolute SHAP correlation):\n"
        )
        for experiment_name, feature in sorted_by_impact:
            # Use numerical overall_direction value, handling possible string values
            direction_value = feature['overall_direction']
            if isinstance(direction_value, str):
                # Convert string direction to numeric equivalent
                if "positive" in direction_value.lower():
                    direction_value = 1
                elif "negative" in direction_value.lower():
                    direction_value = -1
                else:
                    try:
                        direction_value = float(direction_value)
                    except ValueError:
                        direction_value = 0
            else:
                direction_value = float(direction_value)
            
            direction_text = 'positive' if direction_value > 0 else 'negative'
            
            output_file.write(
                f"  - {experiment_name}: {feature['feature_name']} ({feature['shap_correlation']}, {direction_text} {direction_value:.3f})\n"
            )

    def _analyze_exclude_condition_differences(self, summary_data, output_file):
        """Analyze differences in feature importance based on exclude conditions (e.g., exclude_default vs exclude_base_model_metrics) in markdown format."""
        # Group experiments by exclude condition based on naming patterns
        exclude_groups = defaultdict(list)
        for dataset_name in summary_data.keys():
            # Identify exclude condition from dataset name with more precision
            if dataset_name.endswith("_exclude_default"):
                exclude_condition = "exclude_default"
            elif dataset_name.endswith("_exclude_base_model_metrics"):
                exclude_condition = "exclude_base_model_metrics"
            elif "_exclude_default" in dataset_name and not dataset_name.endswith(
                "_exclude_default"
            ):
                exclude_condition = "exclude_default"
            elif (
                "_exclude_base_model_metrics" in dataset_name
                and not dataset_name.endswith("_exclude_base_model_metrics")
            ):
                exclude_condition = "exclude_base_model_metrics"
            # Handle exclude_{exclude feature group} format (all datasets, all models)
            elif dataset_name.startswith("exclude_"):
                if "default" in dataset_name:
                    exclude_condition = "exclude_default"
                elif "base_model_metrics" in dataset_name:
                    exclude_condition = "exclude_base_model_metrics"
                else:
                    exclude_condition = "other"
            else:
                exclude_condition = "other"

            exclude_groups[exclude_condition].append(dataset_name)

        output_file.write(f"- Experiment distribution by exclude condition:\n")
        for exclude_condition, datasets in exclude_groups.items():
            output_file.write(f"  - {exclude_condition}: {len(datasets)} experiments\n")

        # Compare feature importance across different exclude conditions
        if len(exclude_groups) > 1:
            output_file.write(
                f"\n- Feature ranking differences across exclude conditions:\n"
            )

            # Collect top features for each exclude condition
            exclude_top_features = defaultdict(Counter)
            for exclude_condition, datasets in exclude_groups.items():
                for dataset in datasets:
                    features = summary_data[dataset]
                    # Get top features for this dataset
                    for feature_data in features[:3]:  # Top 3 features per dataset
                        exclude_top_features[exclude_condition][
                            feature_data["feature_name"]
                        ] += 1

            # Compare top features across exclude conditions
            all_features = set()
            for counter in exclude_top_features.values():
                all_features.update(counter.keys())

            if all_features:
                output_file.write(
                    f"\n- Common features across exclude conditions (top 3 from each experiment):\n"
                )
                for feature in sorted(all_features):
                    feature_counts_by_condition = {}
                    for exclude_condition, counter in exclude_top_features.items():
                        feature_counts_by_condition[exclude_condition] = counter.get(
                            feature, 0
                        )

                    # Only show features that appear in multiple exclude conditions
                    active_conditions = {
                        k: v for k, v in feature_counts_by_condition.items() if v > 0
                    }
                    if len(active_conditions) > 1:
                        counts_str = ", ".join(
                            [f"{k}:{v}" for k, v in active_conditions.items()]
                        )
                        output_file.write(f"  - {feature} -> {counts_str}\n")

        # If there are multiple experiments for the same exclude condition, analyze consistency within exclude conditions
        output_file.write(f"\n- Consistency within exclude conditions:\n")
        for exclude_condition, datasets in exclude_groups.items():
            if len(datasets) > 1:
                # Calculate average overlap of top features within this exclude condition
                dataset_features = []
                for dataset in datasets:
                    top_features = {
                        f["feature_name"] for f in summary_data[dataset][:5]
                    }
                    dataset_features.append(top_features)

                # Calculate pairwise overlaps
                if len(dataset_features) > 1:
                    overlaps = []
                    for i in range(len(dataset_features)):
                        for j in range(i + 1, len(dataset_features)):
                            common = len(dataset_features[i] & dataset_features[j])
                            overlaps.append(common / 5 * 100)  # As percentage of top 5

                    if overlaps:
                        avg_within_condition_overlap = statistics.mean(overlaps)
                        output_file.write(
                            f"  - {exclude_condition}: Avg top-5 overlap within condition: {avg_within_condition_overlap:.1f}%\n"
                        )
                else:
                    output_file.write(
                        f"  - {exclude_condition}: Only one experiment, no overlap calculation\n"
                    )
            else:
                output_file.write(
                    f"  - {exclude_condition}: Only one experiment, no consistency analysis\n"
                )

def main():
    """Main function to demonstrate the summarizer functionality."""
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Set paths
    input_dir = project_root / "results_visualizations"
    output_dir = project_root / "results_summaries"

    # Configuration
    n_top_analysis = 5  # Number of top features to identify

    print("=" * 80)
    print("Visualization LLM Analysis Tool")
    print("=" * 80)
    print(f"Visualization directory: {output_dir}")
    print(f"Top features to identify: {n_top_analysis}")
    print("=" * 80)

    # Create summarizer and run
    summarizer = VisualizationSummarizer(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )
    # summarizer.analyze_visualizations_with_llm(n=n_top_analysis)
    # Perform hierarchical statistical analysis on the summary data
    summarizer.perform_hierarchical_statistical_analysis()

    print("\nLLM analysis completed!")


if __name__ == "__main__":
    main()
