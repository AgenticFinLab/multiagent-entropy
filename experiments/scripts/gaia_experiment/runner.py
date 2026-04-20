"""Core experiment runner for GAIA benchmark."""

import json
import logging
import os
import time
import traceback
import dotenv
from datetime import datetime
from typing import Any, Dict, List, Optional

from maep.language.debate import DebateMAS
from maep.language.single import SingleAgent
from maep.language.hybrid import OrchestratorHybrid
from maep.language.sequential import SequentialAgents
from maep.language.centralized import OrchestratorCentralized
from maep.language.decentralized import OrchestratorDecentralized
from maep.language.full_decentralized import OrchestratorFullDecentralized

from .constants import GAIA_TASK_TYPE, GAIA_DATA_PATH, GAIA_ATTACHMENTS_ROOT
from .evaluation import evaluate_gaia_result, calculate_aggregate_metrics
from .checkpoint import find_existing_experiment_dir, get_completed_batches
from .answer_extraction import extract_answer_from_result
from .tools import create_gaia_tools, get_all_tool_definitions
from .prompts import enhance_question_with_tools_context

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


def resolve_attachment_path(
    sample: Dict[str, Any], attachments_root: str = None
) -> str:
    """
    Return the local absolute path of the sample's attachment, or empty string
    if the sample has no attachment or the file does not exist locally.

    HuggingFace stores attachments at <repo_root>/<file_path>
    (e.g. 2023/validation/xxx.xlsx). After snapshot_download the layout is
    preserved under attachments_root, so the full local path is:
        <attachments_root>/<file_path>
    """
    root = attachments_root or GAIA_ATTACHMENTS_ROOT
    file_path = sample.get("sample_info", {}).get("file_path", "")
    if not file_path:
        return ""
    local_path = os.path.join(root, file_path)
    return local_path if os.path.exists(local_path) else ""


def _build_question(
    question: str, sample: Dict[str, Any], attachments_root: str
) -> str:
    """
    Build the full question text passed to the agent.

    Adds level and attachment metadata before the question text.
    The GAIA answer-format instructions are already injected via the system
    message (SINGLE_SYS["gaia"] in maep/prompts.py), so they are not repeated here.
    """
    level = sample.get("sample_info", {}).get("level", "")
    file_name = sample.get("sample_info", {}).get("file_name", "")
    local_file = resolve_attachment_path(sample, attachments_root)

    parts = []

    if level:
        parts.append(f"Difficulty Level: {level}")
    if file_name:
        parts.append(f"Attached File: {file_name}")
        if local_file:
            parts.append(f"Local Path: {local_file}")
        else:
            parts.append(
                "(Attachment not available locally — "
                "run download_gaia_attachments.py to fetch it)"
            )
    if parts:
        parts.append("")

    parts += ["Question: " + question, ""]
    return "\n".join(parts)


def _build_agent(agent_type: str, config: Dict[str, Any]):
    agents = {
        "single": SingleAgent,
        "sequential": SequentialAgents,
        "centralized": OrchestratorCentralized,
        "decentralized": OrchestratorDecentralized,
        "full_decentralized": OrchestratorFullDecentralized,
        "debate": DebateMAS,
        "hybrid": OrchestratorHybrid,
    }
    if agent_type not in agents:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    return agents[agent_type](run_config=config)


def run_gaia_experiment(
    config: Dict[str, Any],
    dry_run: bool = False,
    save_folder: str = "experiments/results",
    skip_evaluation: bool = False,
) -> Dict[str, Any]:
    """
    Run a single GAIA experiment with the given configuration.

    Tools provided to the agent:
      - google_web_search  (via SerpAPI / Serper API)
      - calculator         (safe arithmetic evaluator)

    Attachments are resolved from gaia_config.attachments_root (default:
    experiments/data/GAIA/attachments). Run download_gaia_attachments.py first
    to fetch them from HuggingFace.

    Args:
        config: Merged experiment configuration
        dry_run: If True, prepare but do not execute
        save_folder: Base path for checkpoint detection
        skip_evaluation: If True, skip exact-match evaluation

    Returns:
        Dict with experiment results/status
    """
    experiment_name = config.get("experiment_name", "unnamed_experiment")
    agent_type = config.get("agent_type", "single")
    logger.info(
        f"Starting GAIA experiment: {experiment_name} (Agent type: {agent_type})"
    )

    if dry_run:
        logger.info("Dry run mode - skipping experiment execution")
        return {
            "status": "dry_run",
            "experiment_name": experiment_name,
            "agent_type": agent_type,
            "config": config,
        }

    data_cfg = config["data"]
    dataset_name = data_cfg["data_name"].lower()
    model_name = (
        config["agents"][list(config["agents"].keys())[0]]["lm_name"]
        .split("/")[-1]
        .lower()
        .replace("-", "_")
        .replace(".", "_")
    )

    existing_dir = find_existing_experiment_dir(
        save_folder, dataset_name, model_name, experiment_name
    )
    completed_batches = 0
    resume_mode = False

    if existing_dir:
        completed_batches = get_completed_batches(existing_dir)
        if completed_batches > 0:
            batch_size = data_cfg["batch_size"]
            data_num = data_cfg.get("data_num", -1)
            if data_num != -1:
                expected_batches = (data_num + batch_size - 1) // batch_size
                if completed_batches >= expected_batches:
                    logger.info(
                        f"Experiment '{experiment_name}' already completed "
                        f"({completed_batches}/{expected_batches} batches). Skipping."
                    )
                    return {
                        "status": "skipped",
                        "experiment_name": experiment_name,
                        "agent_type": agent_type,
                        "reason": "already_completed",
                        "completed_batches": completed_batches,
                        "results_path": existing_dir,
                    }
            logger.info(f"Found existing experiment at: {existing_dir}")
            logger.info(f"Resuming from batch {completed_batches + 1}")
            resume_mode = True
            config["save_folder"] = existing_dir

    try:
        agent = _build_agent(agent_type, config)

        # Load dataset from pre-downloaded local JSON
        data_file = os.path.join(
            GAIA_DATA_PATH, f"{data_cfg['split']}-all-samples.json"
        )
        with open(data_file, "r", encoding="utf-8") as f:
            all_samples_list = json.load(f)

        gaia_config = config.get("gaia_config", {})

        # Apply optional level filter
        level_filter = gaia_config.get("level_filter")
        if level_filter:
            level_filter_str = [str(l) for l in level_filter]
            all_samples_list = [
                s
                for s in all_samples_list
                if str(s.get("sample_info", {}).get("level", "")) in level_filter_str
            ]
            logger.info(
                f"Level filter {level_filter} applied: {len(all_samples_list)} samples remaining"
            )

        dataset_len = len(all_samples_list)
        total_samples = (
            dataset_len
            if data_cfg["data_num"] == -1
            else min(data_cfg["data_num"], dataset_len)
        )
        batch_size = data_cfg["batch_size"]
        total_batches = (total_samples + batch_size - 1) // batch_size

        if resume_mode and existing_dir:
            completed_batches = get_completed_batches(existing_dir)
            if completed_batches >= total_batches:
                logger.info(
                    f"Experiment '{experiment_name}' already completed. Skipping."
                )
                return {
                    "status": "skipped",
                    "experiment_name": experiment_name,
                    "agent_type": agent_type,
                    "reason": "already_completed",
                    "completed_batches": completed_batches,
                    "results_path": existing_dir,
                }

        start_batch = completed_batches if resume_mode else 0
        start_sample_idx = start_batch * batch_size

        if resume_mode:
            logger.info(
                f"Resuming from batch {start_batch + 1}/{total_batches} (sample {start_sample_idx})"
            )

        # Attachment paths
        attachments_root = gaia_config.get("attachments_root", GAIA_ATTACHMENTS_ROOT)
        samples_subset = all_samples_list[:total_samples]
        n_attach_available = sum(
            1 for s in samples_subset if resolve_attachment_path(s, attachments_root)
        )
        n_attach_missing = sum(
            1
            for s in samples_subset
            if s.get("sample_info", {}).get("file_name")
            and not resolve_attachment_path(s, attachments_root)
        )

        # Initialize tools
        serpapi_key = gaia_config.get("serpapi_api_key") or os.getenv("SERPAPI_API_KEY")
        serper_key = gaia_config.get("serper_api_key") or os.getenv("SERPER_API_KEY")
        ark_api_key = gaia_config.get("ark_api_key") or os.getenv("ARK_API_KEY")
        doubao_model = gaia_config.get("doubao_model", "doubao-seed-2-0-lite-260215")
        gaia_tools = create_gaia_tools(
            serpapi_api_key=serpapi_key,
            serper_api_key=serper_key,
            ark_api_key=ark_api_key,
            doubao_model=doubao_model,
        )
        tool_definitions = get_all_tool_definitions(gaia_tools)

        web_tool = gaia_tools.get("google_web_search")
        api_status = web_tool.get_api_status() if web_tool else {}
        logger.info(
            f"Processing {total_samples} GAIA samples in batches of {batch_size}"
        )
        logger.info(
            f"Attachments: {n_attach_available} available locally, {n_attach_missing} missing"
        )
        logger.info(f"Tools: {list(gaia_tools.keys())}")
        logger.info(f"Web search provider: {api_status.get('active_provider', 'none')}")
        if not api_status.get("serpapi_configured") and not api_status.get(
            "serper_configured"
        ):
            logger.warning(
                "No web search API key set — google_web_search will return mock results"
            )
        if not ark_api_key:
            logger.warning(
                "ARK_API_KEY not set — multimodal_viewer will fail for image/audio/video files"
            )

        all_final_states = []
        all_evaluation_results = []
        start_time = time.time()

        for start_idx in range(start_sample_idx, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)

            enhanced_batch_samples = []
            for sample_idx in range(start_idx, end_idx):
                sample = all_samples_list[sample_idx]
                original_question = sample.get("question", "")
                built_question = _build_question(
                    original_question, sample, attachments_root
                )
                local_file = resolve_attachment_path(sample, attachments_root)

                enhanced_sample = sample.copy()
                enhanced_sample["question"] = enhance_question_with_tools_context(
                    built_question, sample, local_file
                )
                enhanced_sample["original_question"] = original_question
                # Expose resolved local path so downstream agent/tools can access the file
                enhanced_sample["local_file_path"] = local_file
                enhanced_batch_samples.append(enhanced_sample)

            batch_samples = {
                key: [s.get(key) for s in enhanced_batch_samples]
                for key in enhanced_batch_samples[0].keys()
            }

            batch_num = start_idx // batch_size + 1
            logger.info(
                f"Processing batch {batch_num}/{total_batches} (samples {start_idx}-{end_idx - 1})"
            )
            for i in range(start_idx, end_idx):
                s = all_samples_list[i]
                logger.debug(
                    f"Sample {i}: level={s['sample_info'].get('level')} "
                    f"file={s['sample_info'].get('file_name', 'none')}"
                )

            result = agent.run(
                batch_samples, tools=gaia_tools, tool_definitions=tool_definitions
            )
            final_state = result.final_state

            if "agent_results" in final_state:
                all_final_states.extend(final_state["agent_results"])
            elif "merged_results" in final_state:
                all_final_states.append(final_state["merged_results"])

            if not skip_evaluation:
                for i, sample_idx in enumerate(range(start_idx, end_idx)):
                    sample = all_samples_list[sample_idx]
                    if "agent_results" in final_state and i < len(
                        final_state["agent_results"]
                    ):
                        batch_result = final_state["agent_results"][i]
                    else:
                        batch_result = final_state
                    generated_answer = extract_answer_from_result(
                        {"agent_results": [batch_result]}
                    )
                    eval_result = evaluate_gaia_result(sample, generated_answer)
                    eval_result["batch_num"] = batch_num
                    eval_result["sample_idx"] = sample_idx
                    all_evaluation_results.append(eval_result)

            agent.store_manager.save(
                savename=f"Batch_{batch_num}_State", data=final_state
            )

        if all_final_states:
            agent.store_manager.save(
                savename="Combined_FinalState", data={"agent_results": all_final_states}
            )

        if not skip_evaluation and all_evaluation_results:
            aggregate_metrics = calculate_aggregate_metrics(all_evaluation_results)

            eval_save_path = os.path.join(
                config.get("save_folder", ""), "gaia_evaluation_results.json"
            )
            os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)
            with open(eval_save_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "experiment_info": {
                            "experiment_name": experiment_name,
                            "agent_type": agent_type,
                            "total_samples": total_samples,
                            "total_batches": total_batches,
                            "level_filter": level_filter,
                            "tools": list(gaia_tools.keys()),
                            "attachments_root": attachments_root,
                            "timestamp": datetime.now().isoformat(),
                        },
                        "aggregate_metrics": aggregate_metrics,
                        "individual_results": all_evaluation_results,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            logger.info("GAIA Evaluation Results:")
            logger.info(f"  - Total   : {aggregate_metrics['total_questions']}")
            logger.info(f"  - Correct : {aggregate_metrics['correct_count']}")
            logger.info(f"  - Accuracy: {aggregate_metrics['accuracy']:.4f}")
            for lvl, stats in sorted(aggregate_metrics["by_level"].items()):
                logger.info(
                    f"  - Level {lvl}: {stats['correct']}/{stats['count']} ({stats['accuracy']:.4f})"
                )
            logger.info(f"  - Saved to: {eval_save_path}")

        duration = time.time() - start_time
        batches_this_run = total_batches - start_batch
        logger.info(f"Experiment {experiment_name} completed in {duration:.2f}s")
        if resume_mode:
            logger.info(
                f"Resumed and processed {total_samples - start_sample_idx} samples "
                f"across {batches_this_run} batches"
            )

        result_dict = {
            "status": "completed",
            "experiment_name": experiment_name,
            "agent_type": agent_type,
            "samples_processed": total_samples,
            "batches": total_batches,
            "resumed": resume_mode,
            "batches_processed_this_run": (
                batches_this_run if resume_mode else total_batches
            ),
            "duration_seconds": duration,
            "results_path": config.get("save_folder", ""),
            "tools": list(gaia_tools.keys()),
            "attachments_available": n_attach_available,
            "attachments_missing": n_attach_missing,
            "web_search_provider": api_status.get("active_provider", "none"),
            "multimodal_enabled": bool(ark_api_key),
        }
        if not skip_evaluation and all_evaluation_results:
            result_dict["gaia_metrics"] = aggregate_metrics

        return result_dict

    except Exception as e:
        logger.error(f"Experiment {experiment_name} failed: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "failed",
            "experiment_name": experiment_name,
            "agent_type": agent_type,
            "error": str(e),
        }


def run_batch_gaia_experiments(
    batch_config_path: str,
    dry_run: bool = False,
    save_config_flag: bool = True,
    experiment_name: Optional[str] = None,
    save_folder: str = "experiments/results",
    skip_evaluation: bool = False,
) -> List[Dict[str, Any]]:
    """Run multiple GAIA experiments from a batch configuration file."""
    import yaml
    from config_loader import load_experiment_config, save_config

    logger.info(f"Loading batch configuration from: {batch_config_path}")
    with open(batch_config_path, "r", encoding="utf-8") as f:
        batch_config = yaml.safe_load(f)

    experiments = batch_config.get("experiments", [])

    if experiment_name:
        experiments = [exp for exp in experiments if exp.get("name") == experiment_name]
        if not experiments:
            logger.error(
                f"Experiment '{experiment_name}' not found in batch configuration"
            )
            return []
        logger.info(f"Running single experiment from batch: {experiment_name}")

    logger.info(f"Found {len(experiments)} GAIA experiments to run")
    results = []

    for exp_idx, exp in enumerate(experiments):
        logger.info(
            f"Processing experiment {exp_idx + 1}/{len(experiments)}: {exp.get('name', 'unnamed')}"
        )
        merged_config = load_experiment_config(
            base_config_path=exp.get(
                "base_config", "experiments/configs/base_config.yml"
            ),
            model_config_path=exp["model_config"],
            dataset_config_path=exp.get(
                "dataset_config", "experiments/configs/dataset_specific/gaia.yml"
            ),
            experiment_name=exp["name"],
            agent_type=exp.get("agent_type", "single"),
        )

        if save_config_flag:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            timestamp_ms = int(time.time() * 1000) % 1000
            pid = os.getpid()
            model_name = (
                merged_config["agents"][list(merged_config["agents"].keys())[0]][
                    "lm_name"
                ]
                .split("/")[-1]
                .lower()
                .replace("-", "_")
                .replace(".", "_")
            )
            config_save_path = (
                f"experiments/configs_exp/{merged_config['data']['data_name'].lower()}/"
                f"{model_name}/{exp['name']}_{timestamp}_{timestamp_ms}_{pid}.yml"
            )
            save_config(merged_config, config_save_path)
            logger.info(f"Saved merged configuration to: {config_save_path}")

        result = run_gaia_experiment(
            merged_config,
            dry_run=dry_run,
            save_folder=save_folder,
            skip_evaluation=skip_evaluation,
        )
        results.append(result)

    return results
