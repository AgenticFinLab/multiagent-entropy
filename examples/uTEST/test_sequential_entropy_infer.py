"""
Executable test script for SequentialAgent.

- Parses YAML config via PyYAML (yaml.safe_load)
- Command line option `-c/--config` points to the YAML config file
- Instantiates `SequentialAgent` (LangGraph + unified inference backend)
- Loads questions from the dataset defined in config and runs sequential inference
- Run: `python examples/uTEST/test_sequential_entropy_infer.py -c configs/uTEST/sequential_entropy_infer.yml`

Note:
- Update `lm_name` in the config file to point to your local model path
- Update `device` in the config file according to your hardware (cuda/cpu/mps)
- configs/uTEST/sequential_entropy_infer.yml
"""

import yaml
import argparse

from dotenv import load_dotenv
from maep.language.sequential import SequentialAgents
from lmbase.dataset import registry as data_registry


def main():
    """A demo to test the SingleAgent with a YAML config file."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run single-agent test with YAML config."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/uTEST/sequential_entropy_infer.yml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, "r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f)

    agent = SequentialAgents(run_config=run_config)

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
        for i, agent_result in enumerate(final_state["agent_results"]):
            print(f"Sample {start_idx + i} Answer:", list(agent_result.values())[0])

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
