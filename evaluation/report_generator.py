"""
Report generator for evaluation results.

This module provides functionality to generate comprehensive evaluation reports
with tables and visualizations.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generator for evaluation reports."""

    def __init__(self, output_dir: str):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_text_report(
        self,
        accuracy_results: Dict[str, Any],
        entropy_results: Dict[str, Any],
        experiment_name: str,
    ) -> str:
        """
        Generate a text-based evaluation report.

        Args:
            accuracy_results: Dictionary containing accuracy evaluation results
            entropy_results: Dictionary containing entropy analysis results
            experiment_name: Name of the experiment

        Returns:
            Generated report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"EVALUATION REPORT: {experiment_name}")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("")

        report_lines.append("-" * 80)
        report_lines.append("1. ACCURACY EVALUATION")
        report_lines.append("-" * 80)
        report_lines.append("")

        if "accuracy" in accuracy_results:
            report_lines.append(f"Overall Accuracy: {accuracy_results['accuracy']:.4f}")
            report_lines.append(
                f"Correct Samples: {accuracy_results.get('correct_samples', 0)}"
            )
            report_lines.append(
                f"Total Samples: {accuracy_results.get('total_samples', 0)}"
            )
            report_lines.append("")

        if "statistics" in entropy_results:
            report_lines.append("-" * 80)
            report_lines.append("2. ENTROPY ANALYSIS")
            report_lines.append("-" * 80)
            report_lines.append("")

            statistics = entropy_results["statistics"]
            for agent_name, stats in statistics.items():
                report_lines.append(f"Agent: {agent_name}")
                report_lines.append(f"  Mean Entropy: {stats['mean']:.4f}")
                report_lines.append(f"  Std Entropy: {stats['std']:.4f}")
                report_lines.append(f"  Min Entropy: {stats['min']:.4f}")
                report_lines.append(f"  Max Entropy: {stats['max']:.4f}")
                report_lines.append(f"  Median Entropy: {stats['median']:.4f}")
                report_lines.append(f"  Sample Count: {stats['count']}")
                report_lines.append("")

        if "step_statistics" in entropy_results:
            report_lines.append("-" * 80)
            report_lines.append("3. STEP-SPECIFIC ENTROPY ANALYSIS")
            report_lines.append("-" * 80)
            report_lines.append("")

            step_statistics = entropy_results["step_statistics"]
            for agent_name, steps in step_statistics.items():
                report_lines.append(f"Agent: {agent_name}")
                report_lines.append(f"  Total Steps: {len(steps)}")
                for step in sorted(steps.keys()):
                    stats = steps[step]
                    report_lines.append(f"  Step {step}:")
                    report_lines.append(f"    Mean: {stats['mean']:.4f}")
                    report_lines.append(f"    Std: {stats['std']:.4f}")
                    report_lines.append(f"    Min: {stats['min']:.4f}")
                    report_lines.append(f"    Max: {stats['max']:.4f}")
                    report_lines.append(f"    Median: {stats['median']:.4f}")
                    report_lines.append(f"    Q25: {stats['q25']:.4f}")
                    report_lines.append(f"    Q75: {stats['q75']:.4f}")
                    report_lines.append(f"    Count: {stats['count']}")
                report_lines.append("")

        if "sample_statistics" in entropy_results:
            report_lines.append("-" * 80)
            report_lines.append("4. SAMPLE-SPECIFIC ENTROPY ANALYSIS")
            report_lines.append("-" * 80)
            report_lines.append("")

            sample_statistics = entropy_results["sample_statistics"]
            report_lines.append(f"Total Samples: {len(sample_statistics)}")
            report_lines.append("")

            for sample_id, agent_stats in list(sample_statistics.items())[:5]:
                report_lines.append(f"Sample: {sample_id}")
                for agent_name, stats in agent_stats.items():
                    report_lines.append(f"  Agent {agent_name}:")
                    report_lines.append(f"    Mean: {stats['mean']:.4f}")
                    report_lines.append(f"    Std: {stats['std']:.4f}")
                    report_lines.append(f"    Count: {stats['count']}")
                report_lines.append("")

            if len(sample_statistics) > 5:
                report_lines.append(
                    f"... and {len(sample_statistics) - 5} more samples"
                )
                report_lines.append("")

        if "correlation" in entropy_results:
            report_lines.append("-" * 80)
            report_lines.append("5. ENTROPY-ACCURACY CORRELATION")
            report_lines.append("-" * 80)
            report_lines.append("")

            correlation = entropy_results["correlation"]
            report_lines.append(
                f"Pearson Correlation: {correlation.get('pearson_correlation', 0):.4f} "
                f"(p-value: {correlation.get('pearson_p_value', 1):.4f})"
            )
            report_lines.append(
                f"Spearman Correlation: {correlation.get('spearman_correlation', 0):.4f} "
                f"(p-value: {correlation.get('spearman_p_value', 1):.4f})"
            )
            report_lines.append("")

        if (
            "comparison" in entropy_results
            and "pairwise_comparisons" in entropy_results["comparison"]
        ):
            report_lines.append("-" * 80)
            report_lines.append("6. AGENT COMPARISON")
            report_lines.append("-" * 80)
            report_lines.append("")

            pairwise = entropy_results["comparison"]["pairwise_comparisons"]
            for comparison_name, stats in pairwise.items():
                report_lines.append(f"Comparison: {comparison_name}")
                report_lines.append(
                    f"  T-test: t={stats['t_statistic']:.4f}, p={stats['t_p_value']:.4f}"
                )
                report_lines.append(
                    f"  KS-test: statistic={stats['ks_statistic']:.4f}, p={stats['ks_p_value']:.4f}"
                )
                report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def generate_html_report(
        self,
        accuracy_results: Dict[str, Any],
        entropy_results: Dict[str, Any],
        experiment_name: str,
        image_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate an HTML evaluation report.

        Args:
            accuracy_results: Dictionary containing accuracy evaluation results
            entropy_results: Dictionary containing entropy analysis results
            experiment_name: Name of the experiment
            image_paths: Dictionary mapping visualization names to image paths

        Returns:
            Generated HTML report
        """
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report: {experiment_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007bff;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #e9f5ff;
            border-radius: 5px;
            min-width: 200px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #007bff;
        }}
        .metric-value {{
            font-size: 24px;
            color: #333;
        }}
        .visualization {{
            margin: 30px 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            color: #777;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Report: {experiment_name}</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>1. Accuracy Evaluation</h2>
"""

        if "accuracy" in accuracy_results:
            html += f"""
        <div class="metric">
            <div class="metric-label">Overall Accuracy</div>
            <div class="metric-value">{accuracy_results['accuracy']:.4f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Correct Samples</div>
            <div class="metric-value">{accuracy_results.get('correct_samples', 0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Samples</div>
            <div class="metric-value">{accuracy_results.get('total_samples', 0)}</div>
        </div>
"""

        if "statistics" in entropy_results:
            html += """
        <h2>2. Entropy Analysis</h2>
        <table>
            <tr>
                <th>Agent</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
                <th>Median</th>
                <th>Count</th>
            </tr>
"""
            statistics = entropy_results["statistics"]
            for agent_name, stats in statistics.items():
                html += f"""
            <tr>
                <td>{agent_name}</td>
                <td>{stats['mean']:.4f}</td>
                <td>{stats['std']:.4f}</td>
                <td>{stats['min']:.4f}</td>
                <td>{stats['max']:.4f}</td>
                <td>{stats['median']:.4f}</td>
                <td>{stats['count']}</td>
            </tr>
"""
            html += """
        </table>
"""

        if "step_statistics" in entropy_results:
            html += """
        <h2>3. Step-Specific Entropy Analysis</h2>
"""
            step_statistics = entropy_results["step_statistics"]
            for agent_name, steps in step_statistics.items():
                html += f"""
        <h3>{agent_name}</h3>
        <table>
            <tr>
                <th>Step</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
                <th>Median</th>
                <th>Q25</th>
                <th>Q75</th>
                <th>Count</th>
            </tr>
"""
                for step in sorted(steps.keys()):
                    stats = steps[step]
                    html += f"""
            <tr>
                <td>{step}</td>
                <td>{stats['mean']:.4f}</td>
                <td>{stats['std']:.4f}</td>
                <td>{stats['min']:.4f}</td>
                <td>{stats['max']:.4f}</td>
                <td>{stats['median']:.4f}</td>
                <td>{stats['q25']:.4f}</td>
                <td>{stats['q75']:.4f}</td>
                <td>{stats['count']}</td>
            </tr>
"""
                html += """
        </table>
"""

        if "sample_statistics" in entropy_results:
            sample_statistics = entropy_results["sample_statistics"]
            html += f"""
        <h2>4. Sample-Specific Entropy Analysis</h2>
        <p>Total Samples: {len(sample_statistics)}</p>
        <table>
            <tr>
                <th>Sample ID</th>
                <th>Agent</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Count</th>
            </tr>
"""
            for sample_id, agent_stats in list(sample_statistics.items())[:10]:
                for agent_name, stats in agent_stats.items():
                    html += f"""
            <tr>
                <td>{sample_id}</td>
                <td>{agent_name}</td>
                <td>{stats['mean']:.4f}</td>
                <td>{stats['std']:.4f}</td>
                <td>{stats['count']}</td>
            </tr>
"""
            html += """
        </table>
"""
            if len(sample_statistics) > 10:
                html += f"<p>... and {len(sample_statistics) - 10} more samples</p>"

        if "correlation" in entropy_results:
            correlation = entropy_results["correlation"]
            html += """
        <h2>5. Entropy-Accuracy Correlation</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Correlation</th>
                <th>P-value</th>
            </tr>
            <tr>
                <td>Pearson</td>
                <td>{:.4f}</td>
                <td>{:.4f}</td>
            </tr>
            <tr>
                <td>Spearman</td>
                <td>{:.4f}</td>
                <td>{:.4f}</td>
            </tr>
        </table>
""".format(
                correlation.get("pearson_correlation", 0),
                correlation.get("pearson_p_value", 1),
                correlation.get("spearman_correlation", 0),
                correlation.get("spearman_p_value", 1),
            )

        if image_paths:
            html += """
        <h2>6. Visualizations</h2>
"""
            for viz_name, viz_path in image_paths.items():
                rel_path = os.path.relpath(viz_path, self.output_dir)
                html += f"""
        <div class="visualization">
            <h3>{viz_name}</h3>
            <img src="{rel_path}" alt="{viz_name}">
        </div>
"""

        html += """
        <div class="footer">
            <p>Generated by MultiAgent-Entropy Evaluation System</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def save_report(
        self,
        report_content: str,
        experiment_name: str,
        format: str = "txt",
    ) -> str:
        """
        Save report to file.

        Args:
            report_content: Content of the report
            experiment_name: Name of the experiment
            format: Format of the report (txt, html)

        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_report_{timestamp}.{format}"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Report saved to: {filepath}")
        return filepath

    def generate_summary_table(
        self, experiments: List[Dict[str, Any]], save_path: str
    ) -> None:
        """
        Generate a summary table for multiple experiments.

        Args:
            experiments: List of experiment dictionaries
            save_path: Path to save the table
        """
        data = []
        for exp in experiments:
            row = {
                "Experiment": exp.get("experiment_name", "Unknown"),
                "Task Type": exp.get("task_type", "Unknown"),
                "Accuracy": exp.get("accuracy", 0.0),
                "Mean Entropy": exp.get("mean_entropy", 0.0),
                "Std Entropy": exp.get("std_entropy", 0.0),
            }
            data.append(row)

        df = pd.DataFrame(data)

        if save_path.endswith(".csv"):
            df.to_csv(save_path, index=False)
        elif save_path.endswith(".xlsx"):
            df.to_excel(save_path, index=False)
        else:
            df.to_csv(save_path, index=False)

        logger.info(f"Summary table saved to: {save_path}")

    def generate_comparison_plot(
        self, experiments: List[Dict[str, Any]], save_path: str
    ) -> None:
        """
        Generate a comparison plot for multiple experiments.

        Args:
            experiments: List of experiment dictionaries
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        exp_names = [exp.get("experiment_name", "Unknown") for exp in experiments]
        accuracies = [exp.get("accuracy", 0.0) for exp in experiments]
        mean_entropies = [exp.get("mean_entropy", 0.0) for exp in experiments]

        axes[0].barh(exp_names, accuracies, color="skyblue")
        axes[0].set_xlabel("Accuracy")
        axes[0].set_title("Accuracy Comparison")
        axes[0].grid(axis="x", alpha=0.3)

        axes[1].barh(exp_names, mean_entropies, color="lightcoral")
        axes[1].set_xlabel("Mean Entropy")
        axes[1].set_title("Mean Entropy Comparison")
        axes[1].grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Comparison plot saved to: {save_path}")

    def generate_full_report(
        self,
        accuracy_results: Dict[str, Any],
        entropy_results: Dict[str, Any],
        experiment_name: str,
        entropy_analyzer=None,
    ) -> Dict[str, str]:
        """
        Generate a full evaluation report with all components.

        Args:
            accuracy_results: Dictionary containing accuracy evaluation results
            entropy_results: Dictionary containing entropy analysis results
            experiment_name: Name of the experiment
            entropy_analyzer: Optional EntropyAnalyzer instance for visualizations

        Returns:
            Dictionary containing paths to generated files
        """
        generated_files = {}

        viz_dir = os.path.join(self.output_dir, f"{experiment_name}_visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        image_paths = {}

        if entropy_analyzer:
            curves_path = os.path.join(viz_dir, "entropy_curves.png")
            entropy_analyzer.visualize_entropy_curves(curves_path)
            image_paths["Entropy Curves"] = curves_path

            distribution_path = os.path.join(viz_dir, "entropy_distribution.png")
            entropy_analyzer.visualize_entropy_distribution(distribution_path)
            image_paths["Entropy Distribution"] = distribution_path

            step_curves_path = os.path.join(viz_dir, "step_entropy_curves.png")
            entropy_analyzer.visualize_step_entropy_curves(step_curves_path)
            image_paths["Step Entropy Curves"] = step_curves_path

            step_distribution_path = os.path.join(
                viz_dir, "step_entropy_distribution.png"
            )
            entropy_analyzer.visualize_step_entropy_distribution(step_distribution_path)
            image_paths["Step Entropy Distribution"] = step_distribution_path

            if "correlation" in entropy_results:
                correlation_path = os.path.join(
                    viz_dir, "entropy_accuracy_correlation.png"
                )
                entropy_analyzer.visualize_entropy_accuracy_correlation(
                    accuracy_results, correlation_path
                )
                image_paths["Entropy-Accuracy Correlation"] = correlation_path

            comparison_path = os.path.join(viz_dir, "agent_comparison.png")
            entropy_analyzer.visualize_agent_comparison(comparison_path)
            image_paths["Agent Comparison"] = comparison_path

        text_report = self.generate_text_report(
            accuracy_results, entropy_results, experiment_name
        )
        text_path = self.save_report(text_report, experiment_name, format="txt")
        generated_files["text_report"] = text_path

        html_report = self.generate_html_report(
            accuracy_results, entropy_results, experiment_name, image_paths
        )
        html_path = self.save_report(html_report, experiment_name, format="html")
        generated_files["html_report"] = html_path

        generated_files["visualizations"] = image_paths

        return generated_files
