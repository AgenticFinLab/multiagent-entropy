"""
Executable test script for Centralized MAS (two-layer sequential + aggregator) agent.

- Parses YAML config via PyYAML (yaml.safe_load)
- Command line option `-c/--config` points to the YAML config file
- Instantiates `OrchestratorAggAgents` (from centralized_mas.py)
- Loads questions from the dataset defined in config and runs inference
- Run: `python examples/uTEST/test_centralized_entropy_infer.py -c configs/uTEST/centralized_entropy_infer.yml`

Note:
- Update `lm_name` in the config file to point to your local model path
- Update `device` in the config file according to your hardware (cuda/cpu/mps)
- configs/uTEST/centralized_entropy_infer.yml
"""

import yaml
import argparse
from dotenv import load_dotenv
from maep.language.centralized_mas import OrchestratorCentralized
from lmbase.dataset import registry as data_registry


def main():
    """A demo to test the OrchestratorCentralized with a YAML config file."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run orchestrator coop agent test with YAML config."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/uTEST/orchestrator_coop_entropy_infer.yml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, "r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f)

    agent = OrchestratorCentralized(run_config=run_config)

    data_cfg = run_config["data"]
    dataset = data_registry.get(config=data_cfg, split="train")

    # Determine total samples to process
    total_samples = (
        len(dataset)
        if data_cfg["data_num"] == -1
        else min(data_cfg["data_num"], len(dataset))
    )
    batch_size = data_cfg["batch_size"]

    # Initialize list to store all batch results
    all_final_states = []

    # Process in batches
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_samples = dataset[start_idx:end_idx]

        print(
            f"\nProcessing batch {start_idx//batch_size + 1} (samples {start_idx}-{end_idx-1})"
        )

        # Run agent on current batch
        result = agent.run(batch_samples)
        final_state = result.final_state

        # Store batch results
        all_final_states.extend(final_state["agent_results"])

        # Save intermediate batch results
        agent.store_manager.save(
            savename=f"Batch_{start_idx//batch_size + 1}_State",
            data=final_state,
        )

        # Print current batch results
        # Note: agent_results structure is List[Dict[agent_name, responses]]
        # We might want to see the final output from the Orchestrator
        for i, agent_result_dict in enumerate(final_state["agent_results"]):
            # The last entry in agent_results usually corresponds to the last executed agent (Orchestrator)
            # but since it's a list of dicts appended sequentially, let's print the last one if available.
            pass

        # Print the final result from the Orchestrator for each sample in the batch
        # We need to find the Orchestrator's output in the state
        orchestrator_name = "OrchestratorAgent"
        orchestrator_outputs = []
        for res in final_state["agent_results"]:
            if orchestrator_name in res:
                orchestrator_outputs = res[orchestrator_name]
                break

        if orchestrator_outputs:
            for i, out in enumerate(orchestrator_outputs):
                print(f"Sample {start_idx + i} Orchestrator Answer:", out)
        else:
            print("No Orchestrator output found in this batch.")

    # Save combined final state
    combined_state = {"agent_results": all_final_states}
    agent.store_manager.save(
        savename="Combined_FinalState",
        data=combined_state,
    )

    print("\n=== Processing Complete ===")
    print(f"Total samples processed: {total_samples}")
    print(f"Total batches: {len(range(0, total_samples, batch_size))}")


if __name__ == "__main__":
    main()
