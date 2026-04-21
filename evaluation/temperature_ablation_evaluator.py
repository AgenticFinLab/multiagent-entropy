"""Temperature ablation evaluation script.

Evaluates experiments across temperature settings (0.4, 0.6, 0.8). Temperature
0.6 is computed by filtering existing standard results from
``evaluation/results_qwen``; 0.4 and 0.8 use ``TempDataLoader`` to read the
ablation tree at ``experiments/results_temp/raw``.

Shared CSV/summary generation lives in ``base.evaluator.BaseEvaluator``.
"""

import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

from .base.evaluator import BaseEvaluator
from .experiment_analyzer import ExperimentAnalyzer
from .entropy_statistic import EntropyStatistic
from .temperature_ablation_data_loader import TempDataLoader


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TempAblationEvaluator(BaseEvaluator):
    """Temperature ablation evaluator."""

    def __init__(self, base_path: str, args: argparse.Namespace):
        super().__init__(base_path, args)
        self.output_base = (
            Path(base_path) / "evaluation" / "results_temp" / args.dataset
        )

    # ----- per-temperature ----------------------------------------------

    def _process_temperature_other(
        self, dataset: str, model: str, temperature: float, output_dir: Path
    ) -> bool:
        """Process temperatures != 0.6 using ``TempDataLoader``."""
        logger.info(f"Processing temperature {temperature} for {model}/{dataset}")

        temp_data_loader = TempDataLoader(self.base_path)
        completed_experiments = temp_data_loader.get_completed_experiments(dataset)
        if model not in completed_experiments:
            logger.warning(f"No completed experiments found for model {model}")
            return False

        temp_prefix = f"t_{str(temperature).replace('.', '_')}_"
        temp_experiments = [
            exp
            for exp in completed_experiments.get(model, [])
            if exp.startswith(temp_prefix)
        ]
        if not temp_experiments:
            logger.warning(f"No experiments found for temperature {temperature}")
            return False

        logger.info(
            f"Found {len(temp_experiments)} experiments for temperature {temperature}"
        )

        analyzer = ExperimentAnalyzer(self.base_path, data_loader=temp_data_loader)
        entropy_statistic = EntropyStatistic(
            self.base_path, data_loader=temp_data_loader
        )

        from .base.constants import infer_task_type

        inferred_task_type = infer_task_type(dataset, self.args.task_type)

        all_metrics: Dict[str, Any] = {
            "dataset": dataset,
            "task_type": inferred_task_type,
            "models": {model: {"experiments": {}}},
        }
        for exp_name in temp_experiments:
            logger.info(f"  Analyzing experiment: {exp_name}")
            try:
                metrics = analyzer.analyze_experiment(
                    dataset, model, exp_name, self.args.task_type, self.args.timeout
                )
                all_metrics["models"][model]["experiments"][exp_name] = metrics
                logger.info(f"    Successfully analyzed {exp_name}")
            except Exception as e:
                logger.error(f"    Error analyzing {exp_name}: {e}")
                all_metrics["models"][model]["experiments"][exp_name] = {
                    "error": str(e)
                }

        all_entropy_results: Dict[str, Any] = {
            "dataset": dataset,
            "models": {model: {"experiments": {}}},
            "architectures": defaultdict(list),
        }
        for exp_name in temp_experiments:
            logger.info(f"  Analyzing entropy for experiment: {exp_name}")
            try:
                entropy_results = entropy_statistic.analyze_experiment_entropy(
                    dataset, model, exp_name
                )
                all_entropy_results["models"][model]["experiments"][
                    exp_name
                ] = entropy_results
                arch = entropy_results.get("agent_architecture", "unknown")
                all_entropy_results["architectures"][arch].append(f"{model}/{exp_name}")

                try:
                    trend_results = entropy_statistic.analyze_entropy_change_trends(
                        dataset, model, exp_name
                    )
                    all_entropy_results["models"][model]["experiments"][exp_name][
                        "trend_analysis"
                    ] = trend_results
                except Exception as e:
                    logger.warning(f"    Error analyzing trends for {exp_name}: {e}")
                    all_entropy_results["models"][model]["experiments"][exp_name][
                        "trend_analysis"
                    ] = {"error": str(e)}

                logger.info(f"    Successfully analyzed entropy for {exp_name}")
            except Exception as e:
                logger.error(f"    Error analyzing entropy for {exp_name}: {e}")
                all_entropy_results["models"][model]["experiments"][exp_name] = {
                    "error": str(e)
                }

        all_entropy_results["architectures"] = dict(
            all_entropy_results["architectures"]
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "all_metrics.json"
        analyzer.save_metrics_json(all_metrics, str(metrics_path))
        logger.info(f"Saved metrics to {metrics_path}")

        entropy_path = output_dir / "all_entropy_results.json"
        with open(entropy_path, "w", encoding="utf-8") as f:
            json.dump(all_entropy_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved entropy results to {entropy_path}")

        try:
            self.run_aggregator(metrics_path, entropy_path, output_dir)
        except Exception as e:
            logger.error(f"Error generating aggregated CSVs: {e}")

        try:
            self.run_summary(output_dir)
        except Exception as e:
            logger.error(f"Error generating summary CSV: {e}")

        return True

    def _process_temperature_06(
        self, dataset: str, models: List[str], output_dir: Path
    ) -> bool:
        """Filter existing ``results_qwen`` results down to ``models`` for temp 0.6."""
        logger.info(f"Processing temperature 0.6 for {models}/{dataset}")

        results_all_path = (
            Path(self.base_path) / "evaluation" / "results_qwen" / dataset
        )
        metrics_file = results_all_path / "all_metrics.json"
        entropy_file = results_all_path / "all_entropy_results.json"
        if not metrics_file.exists():
            logger.error(f"Metrics file not found: {metrics_file}")
            return False
        if not entropy_file.exists():
            logger.error(f"Entropy file not found: {entropy_file}")
            return False

        with open(metrics_file, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
        with open(entropy_file, "r", encoding="utf-8") as f:
            all_entropy_results = json.load(f)

        filtered_metrics = {
            "dataset": all_metrics.get("dataset", dataset),
            "task_type": all_metrics.get("task_type", "math"),
            "models": {},
        }
        for model in models:
            if model in all_metrics.get("models", {}):
                filtered_metrics["models"][model] = all_metrics["models"][model]
                logger.info(f"  Included model {model} in filtered metrics")
            else:
                logger.warning(f"  Model {model} not found in existing metrics")

        filtered_entropy = {
            "dataset": all_entropy_results.get("dataset", dataset),
            "models": {},
            "architectures": {},
        }
        for model in models:
            if model in all_entropy_results.get("models", {}):
                filtered_entropy["models"][model] = all_entropy_results["models"][model]
                logger.info(f"  Included model {model} in filtered entropy results")
            else:
                logger.warning(f"  Model {model} not found in existing entropy results")

        for arch, exp_list in all_entropy_results.get("architectures", {}).items():
            filtered_list = [
                p for p in exp_list if "/" in p and p.split("/")[0] in models
            ]
            if filtered_list:
                filtered_entropy["architectures"][arch] = filtered_list

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "all_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(filtered_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved filtered metrics to {metrics_path}")

        entropy_path = output_dir / "all_entropy_results.json"
        with open(entropy_path, "w", encoding="utf-8") as f:
            json.dump(filtered_entropy, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved filtered entropy results to {entropy_path}")

        try:
            self.run_aggregator(metrics_path, entropy_path, output_dir)
        except Exception as e:
            logger.error(f"Error generating aggregated CSVs: {e}")
        try:
            self.run_summary(output_dir)
        except Exception as e:
            logger.error(f"Error generating summary CSV: {e}")
        return True

    # ----- top-level ----------------------------------------------------

    def run(self) -> None:
        logger.info("Starting temperature ablation evaluation")
        logger.info(f"  Dataset: {self.args.dataset}")
        logger.info(f"  Models: {self.args.model}")
        logger.info(f"  Temperatures: {self.args.temperatures}")

        for temperature in self.args.temperatures:
            output_dir = self.output_base / f"t_{temperature}"
            logger.info("\n" + "=" * 60)
            logger.info(f"Processing temperature: {temperature}")
            logger.info(f"Output directory: {output_dir}")
            logger.info("=" * 60)

            if temperature == 0.6:
                self._process_temperature_06(
                    self.args.dataset, self.args.model, output_dir
                )
            else:
                for model in self.args.model:
                    success = self._process_temperature_other(
                        self.args.dataset, model, temperature, output_dir
                    )
                    if not success:
                        logger.warning(
                            f"Failed to process temperature {temperature} "
                            f"for model {model}"
                        )

        logger.info("\n" + "=" * 60)
        logger.info("Temperature ablation evaluation completed")
        logger.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-agent experiments across temperature settings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        help="Dataset to analyze (default: math500)",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=["qwen3_4b"],
        help="Model names to analyze (default: qwen3_4b)",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="*",
        default=[0.4, 0.6, 0.8],
        help="Temperatures to evaluate (default: 0.4 0.6 0.8)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["math", "code", "option", "auto"],
        default="auto",
        help="Task type (auto to infer from dataset)",
    )
    parser.add_argument(
        "--timeout", type=int, default=10, help="Maximum time in seconds for code tasks"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_path = str(Path(__file__).parent.parent)
    TempAblationEvaluator(base_path, args).run()


if __name__ == "__main__":
    main()
