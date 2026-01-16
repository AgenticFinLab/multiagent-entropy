"""LLM-based analysis of visualization images.

This module provides functionality to analyze visualization images using LLMs,
specifically designed to extract insights from feature importance and SHAP value plots.
"""

import os
import json
import base64
from pathlib import Path
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
- direction: (string) effect direction ("positive" or "negative") based on SHAP value impact
- magnitude: (number) effect magnitude based on importance scores and SHAP values
- reason: (string) reason for the effect magnitude based on the visualizations

Example JSON output:
```json
[
  {{"rank": 1, "feature_name": "feature1", "direction": "positive", "magnitude": 0.5, "reason": ""}},
  {{"rank": 2, "feature_name": "feature2", "direction": "negative", "magnitude": 0.3, "reason": ""}}
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
    summarizer.analyze_visualizations_with_llm(n=n_top_analysis)

    print("\nLLM analysis completed!")


if __name__ == "__main__":
    main()
