"""
Basic single-agent built with LangGraph.

Topology:
Input -> Agent 1 -> Output

"""

import time
from typing import List
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from lmbase.inference.base import InferInput, InferOutput

from maep.generic import AgentState, BaseAgents
from maep.entropy_infer import HFEntropyInference


SINGLE_SYS = """You are a precise solver.
Solve the problem correctly and concisely."""

SINGLE_USER = """Question:
{question}
"""


class SingleAgent(BaseAgents):
    """Minimal single-agent built with LangGraph.
    - One node `solve`: render -> generate -> write answer
    - Inherits `run(batch)` for unified execution.
    """

    def _get_agent_prompts(self):
        agent_system_msgs = {}
        agent_user_msgs = {}

        agent_system_msgs["SingleSolver"] = SINGLE_SYS
        agent_user_msgs["SingleSolver"] = SINGLE_USER

        return agent_system_msgs, agent_user_msgs

    def define_agent_models(self):
        """Construct HFEntropyInference."""
        self.agents_lm = HFEntropyInference(
            lm_name=self.run_config["lm_name"],
            inference_config=self.run_config["inference_config"],
            entropy_config=self.run_config["entropy_config"],
            generation_config=self.run_config["generation_config"],
        )

    def execute_agent(self, state: AgentState, agent_name: str) -> AgentState:
        """Main solve process of the agent."""
        # state["input"] is a list containing one dictionary with lists for each field
        samples = state["input"]
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
            # Format with the current sample's question
            user_msg = user_msg.format(question=question)

            # Create InferInput for this sample
            infer_input = InferInput(system_msg=system_msg, user_msg=user_msg)
            infer_inputs.append(infer_input)

        # Forward the model to obtain the output for all samples
        out_list: List[InferOutput] = self.agents_lm.infer_batch(infer_inputs)

        # Process results for each sample
        responses = []
        # Iterate with main_id for saving
        for i, (out, main_id) in enumerate(zip(out_list, samples["main_id"])):
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
            "input": state["input"],  # Keep the original input structure
            "agent_results": state["agent_results"],
            "agent_executed": state["agent_executed"],
            "cost": state["cost"],
            "agent_system_msgs": state["agent_system_msgs"],
            "agent_user_msgs": state["agent_user_msgs"],
        }

    def graph(self) -> CompiledStateGraph:
        """
        Build and compile the LangGraph for the single-agent.
        - One node `solve` as entry and terminal
        """
        g = StateGraph(AgentState)
        agent_name = list(self.agents_config.keys())[0]

        g.add_node(agent_name, partial(self.execute_agent, agent_name=agent_name))
        g.set_entry_point(agent_name)
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
