#!/usr/bin/env python3
"""
Regenerate gaia_evaluation_results.json from saved Result_block_*.json traces.

Each GAIA sample produces exactly one evaluation record, derived from the
architecture's final agent (the agent_type in ARCHITECTURE_FINAL_AGENT with
the largest execution_order). For the `single` architecture, the round-1
SingleSolver output is additionally evaluated as a base-model accuracy
proxy (`round1_*` fields).

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
import re
import sys
from collections import defaultdict
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from gaia_experiment.answer_extraction import extract_final_answer_by_identifier
from gaia_experiment.evaluation import evaluate_gaia_result, calculate_aggregate_metrics
from gaia_experiment.constants import GAIA_DATA_PATH

from evaluation.base.architecture import ARCHITECTURE_FINAL_AGENT


RESULT_ID_RE = re.compile(r"^Result_(.+)-([^-]+)-(\d+)_sample_(\d+)$")
ARCH_FROM_DIR_RE = re.compile(r"_gaia_([a-z_]+?)_agent_")


def load_all_samples(split: str = "validation") -> dict:
    data_file = os.path.join(GAIA_DATA_PATH, f"{split}-all-samples.json")
    with open(data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return {s["main_id"]: s for s in samples}


def detect_architecture(exp_dir: str) -> str:
    name = os.path.basename(exp_dir.rstrip("/\\"))
    m = ARCH_FROM_DIR_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse architecture from dir name: {name}")
    return m.group(1)


def load_all_result_blocks(traces_dir: str) -> dict:
    info_path = os.path.join(traces_dir, "Result-store-information.json")
    if not os.path.exists(info_path):
        return {}
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    data = {}
    for block_name in info.keys():
        # Keys may be either bare names ("Result_block_0") or filenames
        # ("Result_block_0.json"), depending on store-manager version.
        fname = block_name if block_name.endswith(".json") else f"{block_name}.json"
        block_path = os.path.join(traces_dir, fname)
        if os.path.exists(block_path):
            with open(block_path, "r", encoding="utf-8") as f:
                data.update(json.load(f))
    return data


def parse_result_id(result_id: str):
    m = RESULT_ID_RE.match(result_id)
    if not m:
        return None
    return {
        "main_id": m.group(1),
        "agent_type": m.group(2),
        "execution_order": int(m.group(3)),
    }


def find_all_experiment_dirs(results_root: str) -> list:
    pattern = os.path.join(results_root, "**", "traces")
    return sorted(
        os.path.dirname(p)
        for p in glob.glob(pattern, recursive=True)
        if os.path.isdir(p)
    )


def _pick_response(result_data: dict) -> str:
    for key in ("final_answer", "answer", "response", "output", "result"):
        if key in result_data and result_data[key] is not None:
            return str(result_data[key])
    return str(result_data)


def _extract_answer(result_data: dict) -> str:
    raw = _pick_response(result_data)
    extracted = extract_final_answer_by_identifier(raw)
    return extracted if extracted else raw


def regenerate(exp_dir: str, samples_by_id: dict, output_path: str = None) -> bool:
    traces_dir = os.path.join(exp_dir, "traces")
    output_path = output_path or os.path.join(exp_dir, "gaia_evaluation_results.json")

    if not os.path.isdir(traces_dir):
        print(f"  SKIP: no traces directory in {exp_dir}")
        return False

    try:
        architecture = detect_architecture(exp_dir)
    except ValueError as e:
        print(f"  SKIP: {e}")
        return False

    final_agent_type = ARCHITECTURE_FINAL_AGENT.get(architecture)
    if final_agent_type is None:
        print(
            f"  SKIP: architecture '{architecture}' has no registered final-agent type"
        )
        return False

    results = load_all_result_blocks(traces_dir)
    if not results:
        print(f"  SKIP: no Result block data found in {traces_dir}")
        return False

    by_sample: dict = defaultdict(list)
    for result_id, result_data in results.items():
        parsed = parse_result_id(result_id)
        if not parsed:
            continue
        parsed["data"] = result_data
        by_sample[parsed["main_id"]].append(parsed)

    evaluation_results = []
    missing = []

    for main_id, items in by_sample.items():
        sample = samples_by_id.get(main_id)
        if sample is None:
            missing.append(main_id)
            continue

        final_items = [it for it in items if it["agent_type"] == final_agent_type]
        if not final_items:
            missing.append(main_id)
            continue
        final_item = max(final_items, key=lambda it: it["execution_order"])

        generated_answer = _extract_answer(final_item["data"])
        eval_result = evaluate_gaia_result(sample, generated_answer)
        eval_result["architecture"] = architecture
        eval_result["final_agent_type"] = final_agent_type
        eval_result["final_execution_order"] = final_item["execution_order"]

        if architecture == "single":
            round1_items = [it for it in items if it["execution_order"] == 1]
            if round1_items:
                r1 = round1_items[0]
                r1_answer = _extract_answer(r1["data"])
                r1_eval = evaluate_gaia_result(sample, r1_answer)
                eval_result["round1_generated_answer"] = r1_answer
                eval_result["round1_evaluation_result"] = r1_eval["evaluation_result"]
                eval_result["round1_evaluation_score"] = r1_eval["evaluation_score"]

        evaluation_results.append(eval_result)

    if missing:
        print(f"  WARNING: {len(missing)} samples could not be evaluated")

    aggregate_metrics = calculate_aggregate_metrics(evaluation_results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_info": {
                    "experiment_dir": exp_dir,
                    "architecture": architecture,
                    "final_agent_type": final_agent_type,
                    "total_samples": len(evaluation_results),
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
        f"  arch={architecture} | "
        f"{len(evaluation_results)} samples | "
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
