import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

from maep.language.debate import DebateMAS
from maep.language.single import SingleAgent
from maep.language.hybrid import OrchestratorHybrid
from lmbase.dataset import registry as data_registry
from maep.language.sequential import SequentialAgents
from maep.language.centralized import OrchestratorCentralized
from maep.language.decentralized import OrchestratorDecentralized
from maep.language.full_decentralized import OrchestratorFullDecentralized

from .constants import MAX_END_DATE, FINAGENT_TASK_TYPE
from .tools import create_financial_tools, get_all_tool_definitions
from .prompts import enhance_question_with_tools_context
from .evaluation import evaluate_finagent_result, calculate_aggregate_metrics
from .checkpoint import find_existing_experiment_dir, get_completed_batches
from .answer_extraction import extract_answer_from_result
from .tool_logger import get_tool_call_logger, reset_tool_call_logger

logger = logging.getLogger(__name__)


def run_finagent_experiment(
    config: Dict[str, Any],
    dry_run: bool = False,
    save_folder: str = "experiments/results",
    skip_evaluation: bool = False,
) -> Dict[str, Any]:
    experiment_name = config.get("experiment_name", "unnamed_experiment")
    agent_type = config.get("agent_type", "single")
    logger.info(f"Starting FinAgent experiment: {experiment_name} (Agent type: {agent_type})")

    if dry_run:
        logger.info("Dry run mode - skipping experiment execution")
        return {"status": "dry_run", "experiment_name": experiment_name, "agent_type": agent_type, "config": config}

    data_cfg = config["data"]
    dataset_name = data_cfg["data_name"].lower()
    model_name = (
        config["agents"][list(config["agents"].keys())[0]]["lm_name"]
        .split("/")[-1].lower().replace("-", "_").replace(".", "_")
    )

    existing_dir = find_existing_experiment_dir(save_folder, dataset_name, model_name, experiment_name)
    completed_batches = 0
    resume_mode = False

    if existing_dir:
        completed_batches = get_completed_batches(existing_dir)
        batch_size = data_cfg["batch_size"]
        data_num = data_cfg.get("data_num", -1)

        if data_num != -1:
            expected_batches = (data_num + batch_size - 1) // batch_size
            if completed_batches >= expected_batches:
                logger.info(f"Experiment '{experiment_name}' already completed ({completed_batches}/{expected_batches} batches). Skipping.")
                return {
                    "status": "skipped", "experiment_name": experiment_name, "agent_type": agent_type,
                    "reason": "already_completed", "completed_batches": completed_batches, "results_path": existing_dir,
                }
            elif completed_batches > 0:
                logger.info(f"Found existing experiment at: {existing_dir}")
                logger.info(f"Resuming from batch {completed_batches + 1} (completed: {completed_batches}/{expected_batches})")
                resume_mode = True
                config["save_folder"] = existing_dir

    try:
        if agent_type == "single":
            agent = SingleAgent(run_config=config)
        elif agent_type == "sequential":
            agent = SequentialAgents(run_config=config)
        elif agent_type == "centralized":
            agent = OrchestratorCentralized(run_config=config)
        elif agent_type == "decentralized":
            agent = OrchestratorDecentralized(run_config=config)
        elif agent_type == "full_decentralized":
            agent = OrchestratorFullDecentralized(run_config=config)
        elif agent_type == "debate":
            agent = DebateMAS(run_config=config)
        elif agent_type == "hybrid":
            agent = OrchestratorHybrid(run_config=config)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        dataset = data_registry.get(config=data_cfg, split=data_cfg["split"])

        data_save_dir = f"experiments/data/{data_cfg['data_name']}"
        os.makedirs(data_save_dir, exist_ok=True)
        dataset_save_path = os.path.join(data_save_dir, f"{data_cfg['split']}-all-samples.json")
        all_samples_list = [dataset[i] for i in range(len(dataset))]
        with open(dataset_save_path, "w", encoding="utf-8") as f:
            json.dump(all_samples_list, f, ensure_ascii=False, indent=2)

        if isinstance(dataset, dict):
            dataset_len = len(next(iter(dataset.values()))) if dataset else 0
        else:
            dataset_len = len(dataset)

        total_samples = dataset_len if data_cfg["data_num"] == -1 else min(data_cfg["data_num"], dataset_len)
        batch_size = data_cfg["batch_size"]
        total_batches = (total_samples + batch_size - 1) // batch_size

        if resume_mode and existing_dir:
            completed_batches = get_completed_batches(existing_dir)
            if completed_batches >= total_batches:
                logger.info(f"Experiment '{experiment_name}' already completed. Skipping.")
                return {
                    "status": "skipped", "experiment_name": experiment_name, "agent_type": agent_type,
                    "reason": "already_completed", "completed_batches": completed_batches, "results_path": existing_dir,
                }

        start_batch = completed_batches if resume_mode else 0
        start_sample_idx = start_batch * batch_size

        if resume_mode:
            logger.info(f"Resuming experiment from batch {start_batch + 1}/{total_batches} (sample {start_sample_idx})")

        logger.info(f"Processing {total_samples} FinAgent samples in batches of {batch_size}")

        finagent_config = config.get("finagent_config", {})
        serpapi_key = finagent_config.get("serpapi_api_key") or os.getenv("SERPAPI_API_KEY")
        serper_key = finagent_config.get("serper_api_key") or os.getenv("SERPER_API_KEY")
        sec_api_key = finagent_config.get("sec_api_key") or os.getenv("SEC_EDGAR_API_KEY")

        financial_tools = create_financial_tools(
            serpapi_api_key=serpapi_key, serper_api_key=serper_key, sec_api_key=sec_api_key,
        )
        logger.info(f"Initialized financial tools: {list(financial_tools.keys())}")

        web_search_tool = financial_tools.get("google_web_search")
        if web_search_tool:
            api_status = web_search_tool.get_api_status()
            if api_status["serpapi_configured"]:
                logger.info("SerpAPI key configured - Google web search enabled (primary)")
            if api_status["serper_configured"]:
                logger.info("Serper API key configured - Google web search enabled (fallback)")
            if not api_status["serpapi_configured"] and not api_status["serper_configured"]:
                logger.warning("No web search API key configured - web search will use mock results")
            logger.info(f"Web search active provider: {api_status['active_provider']}")

        if sec_api_key:
            logger.info("SEC API key configured - EDGAR search enabled")
        else:
            logger.warning("SEC_EDGAR_API_KEY not set - EDGAR search will use mock results")

        tool_definitions = get_all_tool_definitions(financial_tools)
        logger.debug(f"Tool definitions ready for function calling: {[t['function']['name'] for t in tool_definitions]}")

        use_tools_context = finagent_config.get("use_tools_context", True)
        max_tool_turns = finagent_config.get("max_tool_turns", 20)

        if use_tools_context:
            logger.info("FinAgent tools context enhancement enabled")

        all_final_states = []
        all_evaluation_results = []
        all_tool_metadata = []
        global_data_storage = {}
        start_time = time.time()

        for start_idx in range(start_sample_idx, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_data_storage = {}
            enhanced_batch_samples = []
            batch_tool_metadata = []

            for sample_idx in range(start_idx, end_idx):
                sample = all_samples_list[sample_idx]
                original_question = sample.get("question", "")
                if use_tools_context:
                    enhanced_question = enhance_question_with_tools_context(original_question, sample)
                else:
                    enhanced_question = original_question
                enhanced_sample = sample.copy()
                enhanced_sample["question"] = enhanced_question
                enhanced_sample["original_question"] = original_question
                enhanced_batch_samples.append(enhanced_sample)
                batch_tool_metadata.append({
                    "sample_idx": sample_idx,
                    "main_id": sample.get("main_id", ""),
                    "question_type": sample.get("sample_info", {}).get("question_type", ""),
                    "task": sample.get("sample_info", {}).get("task", ""),
                    "tools_context_added": use_tools_context,
                    "available_tools": list(financial_tools.keys()),
                })

            batch_samples = {
                key: [s.get(key) for s in enhanced_batch_samples]
                for key in enhanced_batch_samples[0].keys()
            }

            batch_num = start_idx // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} (samples {start_idx}-{end_idx-1})")

            for meta in batch_tool_metadata:
                logger.debug(f"Sample {meta['sample_idx']}: {meta['question_type']} - {meta['task']}")

            result = agent.run(batch_samples, tools=financial_tools, tool_definitions=tool_definitions)
            final_state = result.final_state

            if "agent_results" in final_state:
                all_final_states.extend(final_state["agent_results"])
            elif "merged_results" in final_state:
                all_final_states.append(final_state["merged_results"])

            all_tool_metadata.extend(batch_tool_metadata)

            if not skip_evaluation:
                for i, sample_idx in enumerate(range(start_idx, end_idx)):
                    sample = all_samples_list[sample_idx]
                    if "agent_results" in final_state and i < len(final_state["agent_results"]):
                        batch_result = final_state["agent_results"][i]
                    else:
                        batch_result = final_state
                    generated_answer = extract_answer_from_result({"agent_results": [batch_result]})
                    eval_result = evaluate_finagent_result(sample, generated_answer)
                    eval_result["batch_num"] = batch_num
                    eval_result["sample_idx"] = sample_idx
                    all_evaluation_results.append(eval_result)

            agent.store_manager.save(savename=f"Batch_{batch_num}_State", data=final_state)

        if all_final_states:
            agent.store_manager.save(savename="Combined_FinalState", data={"agent_results": all_final_states})

        if not skip_evaluation and all_evaluation_results:
            aggregate_metrics = calculate_aggregate_metrics(all_evaluation_results)
            eval_save_path = os.path.join(config.get("save_folder", ""), "finagent_evaluation_results.json")
            os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)
            with open(eval_save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_info": {
                        "experiment_name": experiment_name, "agent_type": agent_type,
                        "total_samples": total_samples, "total_batches": total_batches,
                        "max_end_date": MAX_END_DATE, "tools_available": list(financial_tools.keys()),
                        "tools_context_enabled": use_tools_context, "timestamp": datetime.now().isoformat(),
                    },
                    "aggregate_metrics": aggregate_metrics,
                    "individual_results": all_evaluation_results,
                    "tool_metadata": all_tool_metadata,
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"FinAgent Evaluation Results:")
            logger.info(f"  - Total Questions: {aggregate_metrics['total_questions']}")
            logger.info(f"  - Average Score: {aggregate_metrics['average_score']:.4f}")
            logger.info(f"  - Binary Accuracy: {aggregate_metrics['binary_accuracy']:.4f}")
            logger.info(f"  - Results saved to: {eval_save_path}")

        tool_logger = get_tool_call_logger()
        if tool_logger.get_statistics()["total_calls"] > 0:
            tool_logs_path = os.path.join(config.get("save_folder", ""), "tool_call_logs.json")
            tool_logger.export_to_file(tool_logs_path)
            stats = tool_logger.get_statistics()
            logger.info(f"Tool Call Statistics:")
            logger.info(f"  - Total tool calls: {stats['total_calls']}")
            for tool_name, tool_stats in stats.get("by_tool", {}).items():
                logger.info(
                    f"  - {tool_name}: {tool_stats['count']} calls, "
                    f"{tool_stats.get('success_rate', 0)*100:.1f}% success, "
                    f"{tool_stats.get('avg_time_ms', 0):.0f}ms avg"
                )
            logger.info(f"  - Tool logs saved to: {tool_logs_path}")

        reset_tool_call_logger()

        duration = time.time() - start_time
        batches_processed_this_run = total_batches - start_batch
        samples_processed_this_run = total_samples - start_sample_idx
        logger.info(f"Experiment {experiment_name} completed in {duration:.2f} seconds")
        if resume_mode:
            logger.info(f"Resumed and processed {samples_processed_this_run} samples across {batches_processed_this_run} batches")

        result_dict = {
            "status": "completed", "experiment_name": experiment_name, "agent_type": agent_type,
            "samples_processed": total_samples, "batches": total_batches, "resumed": resume_mode,
            "batches_processed_this_run": batches_processed_this_run if resume_mode else total_batches,
            "duration_seconds": duration, "results_path": config.get("save_folder", ""),
            "tools_info": {
                "tools_available": list(financial_tools.keys()),
                "tools_context_enabled": use_tools_context,
                "max_end_date": MAX_END_DATE,
                "web_search_provider": web_search_tool.get_api_status()["active_provider"] if web_search_tool else "none",
                "serpapi_configured": bool(serpapi_key),
                "serper_configured": bool(serper_key),
                "sec_api_configured": bool(sec_api_key),
            },
        }

        if not skip_evaluation and all_evaluation_results:
            result_dict["finagent_metrics"] = aggregate_metrics

        return result_dict

    except Exception as e:
        logger.error(f"Experiment {experiment_name} failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "failed", "experiment_name": experiment_name, "agent_type": agent_type, "error": str(e)}


def run_batch_finagent_experiments(
    batch_config_path: str,
    dry_run: bool = False,
    save_config_flag: bool = True,
    experiment_name: str = None,
    save_folder: str = "experiments/results",
    skip_evaluation: bool = False,
) -> List[Dict[str, Any]]:
    import yaml
    from config_loader import load_experiment_config, save_config

    logger.info(f"Loading batch configuration from: {batch_config_path}")
    with open(batch_config_path, "r", encoding="utf-8") as f:
        batch_config = yaml.safe_load(f)

    experiments = batch_config.get("experiments", [])

    if experiment_name:
        experiments = [exp for exp in experiments if exp.get("name") == experiment_name]
        if not experiments:
            logger.error(f"Experiment '{experiment_name}' not found in batch configuration")
            return []
        logger.info(f"Running single experiment from batch: {experiment_name}")

    logger.info(f"Found {len(experiments)} FinAgent experiments to run")
    results = []

    for exp_idx, exp in enumerate(experiments):
        logger.info(f"Processing experiment {exp_idx + 1}/{len(experiments)}: {exp.get('name', 'unnamed')}")
        merged_config = load_experiment_config(
            base_config_path=exp.get("base_config", "experiments/configs/base_config.yml"),
            model_config_path=exp["model_config"],
            dataset_config_path=exp.get("dataset_config", "experiments/configs/dataset_specific/finagent.yml"),
            experiment_name=exp["name"],
            agent_type=exp.get("agent_type", "single"),
        )

        if save_config_flag:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            timestamp_ms = int(time.time() * 1000) % 1000
            pid = os.getpid()
            model_name = (
                merged_config["agents"][list(merged_config["agents"].keys())[0]]["lm_name"]
                .split("/")[-1].lower().replace("-", "_").replace(".", "_")
            )
            config_save_path = (
                f"experiments/configs_exp/{merged_config['data']['data_name'].lower()}/"
                f"{model_name}/{exp['name']}_{timestamp}_{timestamp_ms}_{pid}.yml"
            )
            save_config(merged_config, config_save_path)
            logger.info(f"Saved merged configuration to: {config_save_path}")

        result = run_finagent_experiment(merged_config, dry_run=dry_run, save_folder=save_folder, skip_evaluation=skip_evaluation)
        results.append(result)

    return results
