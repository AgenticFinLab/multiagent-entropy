#!/usr/bin/env python3
"""
Regenerate gaia_evaluation_results.json from saved Batch_*_State traces.

Usage:
    # Single experiment
    python regenerate_gaia_evaluation.py <experiment_dir>

    # All experiments under results_gaia
    python regenerate_gaia_evaluation.py --all
    python regenerate_gaia_evaluation.py --all --results-root experiments/results_gaia
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaia_experiment.answer_extraction import extract_answer_from_result
from gaia_experiment.evaluation import evaluate_gaia_result, calculate_aggregate_metrics
from gaia_experiment.constants import GAIA_DATA_PATH


def load_all_samples(split: str = "validation") -> dict:
    data_file = os.path.join(GAIA_DATA_PATH, f"{split}-all-samples.json")
    with open(data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return {s["main_id"]: s for s in samples}


def load_store_blocks(traces_dir: str, prefix: str) -> dict:
    """Load all JSON block files for a given store prefix (e.g. 'Batch')."""
    info_path = os.path.join(traces_dir, f"{prefix}-store-information.json")
    if not os.path.exists(info_path):
        return {}
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    data = {}
    for block_name, meta in info.items():
        block_path = os.path.join(traces_dir, meta["path"].split("/traces/")[-1])
        if not os.path.exists(block_path):
            block_path = os.path.join(traces_dir, f"{block_name}.json")
        if os.path.exists(block_path):
            with open(block_path, "r", encoding="utf-8") as f:
                data.update(json.load(f))
    return data


def find_all_experiment_dirs(results_root: str) -> list:
    """Find all experiment directories (those containing a traces/ subdirectory)."""
    pattern = os.path.join(results_root, "**", "traces")
    return sorted(
        os.path.dirname(p)
        for p in glob.glob(pattern, recursive=True)
        if os.path.isdir(p)
    )


def regenerate(exp_dir: str, samples_by_id: dict, output_path: str = None) -> bool:
    traces_dir = os.path.join(exp_dir, "traces")
    output_path = output_path or os.path.join(exp_dir, "gaia_evaluation_results.json")

    if not os.path.isdir(traces_dir):
        print(f"  SKIP: no traces directory in {exp_dir}")
        return False

    batch_data = load_store_blocks(traces_dir, "Batch")
    if not batch_data:
        print(f"  SKIP: no Batch state data found in {traces_dir}")
        return False

    def batch_sort_key(k):
        try:
            return int(k.split("_")[1])
        except (IndexError, ValueError):
            return 0

    evaluation_results = []
    missing = []

    for key in sorted(batch_data.keys(), key=batch_sort_key):
        final_state = batch_data[key]
        batch_num = batch_sort_key(key)

        agent_results = final_state.get("agent_results", [final_state])
        if not isinstance(agent_results, list):
            agent_results = [agent_results]

        for i, batch_result in enumerate(agent_results):
            question_id = (
                batch_result.get("question_id")
                or batch_result.get("id")
                or batch_result.get("main_id")
            )
            sample_idx = batch_num - 1 + i
            sample = None
            if question_id and question_id in samples_by_id:
                sample = samples_by_id[question_id]
            else:
                all_samples = list(samples_by_id.values())
                if 0 <= sample_idx < len(all_samples):
                    sample = all_samples[sample_idx]

            if sample is None:
                missing.append(key)
                continue

            generated_answer = extract_answer_from_result({"agent_results": [batch_result]})
            eval_result = evaluate_gaia_result(sample, generated_answer)
            eval_result["batch_num"] = batch_num
            eval_result["sample_idx"] = sample_idx
            evaluation_results.append(eval_result)

    if missing:
        print(f"  WARNING: {len(missing)} batches could not be matched to samples")

    aggregate_metrics = calculate_aggregate_metrics(evaluation_results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_info": {
                    "experiment_dir": exp_dir,
                    "total_samples": len(evaluation_results),
                    "total_batches": len(batch_data),
                    "timestamp": datetime.now().isoformat(),
                    "regenerated": True,
                },
                "aggregate_metrics": aggregate_metrics,
                "individual_results": evaluation_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"  {len(evaluation_results)} samples | "
        f"correct {aggregate_metrics['correct_count']} | "
        f"accuracy {aggregate_metrics['accuracy']:.4f} | "
        f"-> {output_path}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Regenerate GAIA evaluation results from saved traces")
    parser.add_argument("experiment_dir", nargs="?", help="Path to a single experiment directory")
    parser.add_argument("--all", action="store_true", help="Process all experiments under --results-root")
    parser.add_argument("--results-root", default="experiments/results_gaia",
                        help="Root directory to search when using --all (default: experiments/results_gaia)")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", help="Output JSON path (single-experiment mode only)")
    args = parser.parse_args()

    if not args.all and not args.experiment_dir:
        parser.error("Provide an experiment_dir or use --all")

    print(f"Loading dataset samples ({args.split})...")
    samples_by_id = load_all_samples(args.split)
    print(f"  {len(samples_by_id)} samples loaded\n")

    if args.all:
        exp_dirs = find_all_experiment_dirs(args.results_root)
        if not exp_dirs:
            print(f"No experiment directories found under {args.results_root}")
            sys.exit(1)
        print(f"Found {len(exp_dirs)} experiment(s) under {args.results_root}\n")
        for exp_dir in exp_dirs:
            print(f"[{os.path.basename(exp_dir)}]")
            regenerate(exp_dir, samples_by_id)
            print()
    else:
        exp_dir = args.experiment_dir.rstrip("/\\")
        print(f"[{os.path.basename(exp_dir)}]")
        if not regenerate(exp_dir, samples_by_id, args.output):
            sys.exit(1)


if __name__ == "__main__":
    main()
