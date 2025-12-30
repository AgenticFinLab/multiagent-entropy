"""
Executable test script for SingleAgent.

- Parses YAML config via PyYAML (yaml.safe_load)
- Command line option `-c/--config` points to the YAML config file
- Instantiates `SingleAgent` (LangGraph + unified inference backend)
- Loads a question from the dataset defined in config and prints the answer
- Run: `python examples/uTEST/test_single.py -c configs/uTEST/test_single.yml`
"""

import yaml
import argparse

from dotenv import load_dotenv
from maep.language.sequential import SequentialAgents
from maep.entropy_infer import HFEntropyInference
from lmbase.dataset import registry as data_registry
from lmbase.dataset.base import VisualTextSample


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
    sample: VisualTextSample = dataset[: data_cfg["data_num"]]
    result = agent.run(sample)
    final_state = result.final_state

    # Save the final state to the json
    agent.store_manager.save(
        savename="FinalState",
        data=final_state,
    )

    print("Final Answer:", list(final_state["agent_results"][-1].values())[0])


if __name__ == "__main__":
    main()
