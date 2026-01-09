"""
Basic single-agent built with LangGraph.

Topology:
Input -> Agent 1 -> Output

For multiple rounds:
Input -> Agent 1 -> Agent 1 -> ... -> Output

Memory:
Agent 1 will receive the input in the last round.

LLM calls: r (rounds)
"""

import time
from typing import List
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from lmbase.inference.base import InferInput, InferOutput

from maep.generic import AgentState, BaseAgents
from maep.entropy_infer import HFEntropyInference
from maep.prompts import get_identifier


class SingleAgent(BaseAgents):
    """Minimal single-agent built with LangGraph.
    - One node `solve`: render -> generate -> write answer
    - Inherits `run(batch)` for unified execution.
    """

    def __init__(self, run_config: dict):
        super().__init__(run_config)
        # Number of rounds for the single-agent pipeline
        self.round = run_config["round"]
        # History aggregation flag
        self.aggregate_history = run_config.get("aggregate_history", True)
        # History optimization settings
        self.max_history_chars = run_config.get("max_history_chars", 0)
        self.max_history_rounds = run_config.get("max_history_rounds", 0)

    def _get_agent_prompts(self):
        agent_system_msgs = {}
        agent_user_msgs = {}

        for name, config in self.agents_config.items():
            if "sys_message" in config:
                agent_system_msgs[name] = self._load_from_module(config["sys_message"])
            if "user_message" in config:
                agent_user_msgs[name] = self._load_from_module(config["user_message"])

        return agent_system_msgs, agent_user_msgs

    def define_agent_models(self):
        """Construct HFEntropyInference."""
        self.agents_lm = HFEntropyInference(
            lm_name=self.run_config["lm_name"],
            inference_config=self.run_config["inference_config"],
            entropy_config=self.run_config["entropy_config"],
            generation_config=self.run_config["generation_config"],
        )

    def format_round_history(self, state: AgentState, sample_idx: int) -> str:
        """
        Format the history of all previous rounds for a specific sample.

        Args:
            state: Current agent state containing execution history
            sample_idx: Index of the sample to format history for

        Returns:
            Formatted string containing all previous round interactions
        """
        if not self.aggregate_history:
            return ""

        history_parts = []

        # Apply max_history_rounds limit
        start_round = 0
        if self.max_history_rounds > 0:
            total_rounds = len(state["agent_results"])
            start_round = max(0, total_rounds - self.max_history_rounds)

        for round_idx, agent_result in enumerate(
            state["agent_results"][start_round:], start=start_round + 1
        ):
            agent_name = list(agent_result.keys())[0]
            response = agent_result[agent_name][sample_idx]

            history_parts.append(f"Round {round_idx} - {agent_name}:\n{response}\n")

        if not history_parts:
            return ""

        history_text = (
            "### Previous Rounds History ###\n"
            + "\n".join(history_parts)
            + "### End of History ###\n\n"
        )

        # Apply max_history_chars limit
        if self.max_history_chars > 0 and len(history_text) > self.max_history_chars:
            # Truncate from the beginning to keep most recent history
            history_text = (
                "### Previous Rounds History ###\n"
                + "...[earlier history truncated due to size limit]...\n\n"
                + history_text[-(self.max_history_chars - 100) :]
            )

        return history_text

    def build_prompt_with_history(
        self, base_prompt: str, question: str, state: AgentState, sample_idx: int
    ) -> str:
        """
        Build the prompt with optional history aggregation.

        Args:
            base_prompt: Base prompt template
            question: Current question
            state: Current agent state
            sample_idx: Index of the sample

        Returns:
            Complete prompt with history if aggregation is enabled
        """
        if not self.aggregate_history:
            return base_prompt.format(question=question)

        history = self.format_round_history(state, sample_idx)

        if history:
            identifier = get_identifier(self.task_type)
            return (
                f"{question}\n\n"
                f"{history}"
                f"Please consider the previous attempts above and place the final answer in {identifier}."
            )

        return base_prompt.format(question=question)

    def execute_agent(self, state: AgentState, agent_name: str) -> AgentState:
        """Main solve process of the agent."""
        # state["init_input"] is a dictionary with lists for each field
        samples = state["init_input"]
        # Get the number of samples from the length of the question list
        num_samples = len(samples["question"])
        execution_idx = len(state["agent_executed"]) + 1

        # Use the passed agent_name
        cur_name = agent_name
        # Obtain the time of the solve
        t0 = time.time()

        # Prepare inputs for all samples
        infer_inputs = []
        for i in range(num_samples):  # Iterate through each sample index
            question = samples["question"][i]
            main_id = samples["main_id"][i]

            # Obtain the prompts
            system_msg = state["agent_system_msgs"][cur_name]
            user_msg = state["agent_user_msgs"][cur_name]
            system_msg = system_msg.replace("{", "{{").replace("}", "}}")

            # Build prompt with optional history aggregation
            formatted_user_msg = self.build_prompt_with_history(
                user_msg, question, state, i
            )

            # Create InferInput for this sample
            infer_input = InferInput(system_msg=system_msg, user_msg=formatted_user_msg)
            infer_inputs.append(infer_input)

        # Forward the model to obtain the output for all samples
        out_list: List[InferOutput] = self.agents_lm.infer_batch(infer_inputs)

        # Process results for each sample
        responses = []
        # Iterate with main_id for saving
        for i, out in enumerate(out_list):
            main_id = samples["main_id"][i]
            # Save each sample's result individually
            savename = self.get_save_name(cur_name, execution_idx)

            self.store_manager.save(
                # Use the sample's specific main_id
                savename=f"Result_{main_id}-{savename}_sample_{i}",
                # Save individual result
                data=out.to_dict(),
            )
            responses.append(out.response)

        # Update the state with all responses
        state["agent_results"].append(
            {cur_name: responses}
        )  # Store list of all responses
        state["cost"].append({cur_name: {"latency": time.time() - t0}})
        state["agent_executed"].append(cur_name)

        return {
            "init_input": state["init_input"],
            "agent_results": state["agent_results"],
            "agent_executed": state["agent_executed"],
            "cost": state["cost"],
            "agent_system_msgs": state["agent_system_msgs"],
            "agent_user_msgs": state["agent_user_msgs"],
        }

    def check_round_limit(self, state: AgentState) -> str:
        """
        Check if the number of completed rounds has reached the limit.
        """
        # Count how many times the agent has executed
        agent_name = list(self.agents_config.keys())[0]
        agent_executions = [
            name for name in state["agent_executed"] if name == agent_name
        ]
        rounds_completed = len(agent_executions)

        if rounds_completed < self.round:
            return agent_name  # Continue to the same agent
        return "end"  # End the workflow

    def graph(self) -> CompiledStateGraph:
        """
        Build and compile the LangGraph for the single-agent.
        - One node `solve` as entry and terminal
        - Supports multiple rounds based on the round parameter
        """
        g = StateGraph(AgentState)
        agent_name = list(self.agents_config.keys())[0]

        g.add_node(agent_name, partial(self.execute_agent, agent_name=agent_name))
        g.set_entry_point(agent_name)

        if self.round > 1:
            # Add conditional edge to support multiple rounds
            g.add_conditional_edges(
                agent_name, self.check_round_limit, {agent_name: agent_name, "end": END}
            )
        else:
            # Single round, direct to end
            g.add_edge(agent_name, END)

        return g.compile()


class ConditionalSingleAgent(SingleAgent):
    """
    A single agent that runs in a loop until a condition is met.
    Default condition: checks for the presence of \\box{} or \\boxed{} in the output.
    """

    def check_condition(self, state: AgentState) -> str:
        """
        Determine if the agent should continue or stop.
        Checks if the last output contains \\box{} or \\boxed{}.
        """
        last_result = state["agent_results"][-1]
        content = list(last_result.values())[0]

        if "\\box{" in content or "\\boxed{" in content:
            return "end"
        return "continue"

    def graph(self) -> CompiledStateGraph:
        g = StateGraph(AgentState)
        # Only the single agent, thus get the first one directly
        agent_name = list(self.agents_config.keys())[0]

        g.add_node(agent_name, partial(self.execute_agent, agent_name=agent_name))
        g.set_entry_point(agent_name)

        g.add_conditional_edges(
            agent_name,
            self.check_condition,
            {
                "continue": agent_name,
                "end": END,
            },
        )
        return g.compile()
