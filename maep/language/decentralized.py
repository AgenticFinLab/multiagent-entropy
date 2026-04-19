"""
Orchestrator-Repeat (two-layer with loop) multi-agent structure with LangGraph.

Topology:
Loop P times:
  Agent 1 (Math) -> Agent 2 (Science) -> Agent 3 (Code)
    ^                                       |
    |_______________________________________| (if loop < P)
                                            | (if loop == P)
                                            v
                                        Orchestrator -> Output

Memory:
- Each layer 1 agent only receives the output from the immediately previous agent.
- At the end of each round, the Agent 3 will send the output to the Agent 1 in the next round (if it is not the last round).

LLM calls: r * N (rounds * number of agents) + 1 (orchestrator)

Corresponding to the MAS (Decentralized Multi-Agent System) [1] with two layers:
- Layer 1: N sequential agents running in a loop P times.
- Layer 2: Orchestrator that aggregates the outputs of all agents after each loop.

[1]. Towards a Science of Scaling Agent Systems.

"""

import time
from typing import List
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from lmbase.inference.base import InferInput, InferOutput
from maep.generic import AgentState
from maep.language.centralized import OrchestratorCentralized


class OrchestratorDecentralized(OrchestratorCentralized):
    """
    Decentralized Multi-Agent System (Decentralized MAS) with a repeat loop.

    Structure:
    - Layer 1 (Agents): N agents run sequentially in a loop for `repeat_count` (P) iterations.
      - Agent 1 -> Agent 2 -> ... -> Agent N -> (Loop back to Agent 1 if p < P)
    - Layer 2 (Orchestrator): After P iterations, the Orchestrator aggregates the results.

    Key Difference from Centralized MAS:
    - In Centralized MAS, agents run in parallel (logically) or sequence but interact with the Orchestrator every round.
    - In Decentralized MAS, agents interact with EACH OTHER in a loop for P times BEFORE the Orchestrator intervenes.
    """

    def __init__(self, run_config: dict):
        super().__init__(run_config)
        self.repeat_count = run_config["round"]

    def execute_sub_agent(self, state: AgentState, name: str) -> AgentState:
        """
        Execute a single layer1 agent, incorporating context from the previous agent executions.
        """
        samples = state["init_input"]
        num_samples = len(samples["question"])
        execution_idx = len(state["agent_executed"]) + 1

        t0 = time.time()
        # Prepare inputs for all samples
        infer_inputs = []
        for i in range(num_samples):
            question = samples["question"][i]

            # 1. Build context from previous agent's output if available
            prev_context = ""
            if state["agent_results"]:
                # Get the last execution result
                last_result = state["agent_results"][-1]
                # last_result is {agent_name: [responses]}
                last_agent_name = list(last_result.keys())[0]
                last_responses = last_result[last_agent_name]

                prev_context = (
                    f"\n\nContext from {last_agent_name}:\n{last_responses[i]}"
                )

            # 2. Format Prompt
            system_msg = (
                state["agent_system_msgs"][name].replace("{", "{{").replace("}", "}}")
            )
            base_user_prompt = state["agent_user_msgs"][name].format(question=question)

            # Append context
            user_prompt = f"{base_user_prompt}{prev_context}"

            infer_inputs.append(InferInput(system_msg=system_msg, user_msg=user_prompt))

        # Batch inference
        out_list: List[InferOutput] = self.react_infer_batch(infer_inputs)
        latency = time.time() - t0

        # Process results
        responses = []
        for i, out in enumerate(out_list):
            main_id = samples["main_id"][i]
            savename = self.get_save_name(name, execution_idx)
            self.store_manager.save(
                savename=f"Result_{main_id}-{savename}_sample_{i}",
                data=out.to_dict(),
            )
            responses.append(out.response)

        # Update state
        new_results = state["agent_results"] + [{name: responses}]
        new_cost = state["cost"] + [{name: {"latency": latency}}]
        new_executed = state["agent_executed"] + [name]

        return {
            "init_input": state["init_input"],
            "agent_results": new_results,
            "agent_executed": new_executed,
            "cost": new_cost,
            "agent_system_msgs": state["agent_system_msgs"],
            "agent_user_msgs": state["agent_user_msgs"],
        }

    def execute_orchestrator(self, state: AgentState) -> AgentState:
        """
        Execute the Orchestrator agent to aggregate results from the LAST loop of Layer 1.
        """
        samples = state["init_input"]
        num_samples = len(samples["question"])
        execution_idx = len(state["agent_executed"]) + 1
        name = self.orchestrator

        t0 = time.time()

        # Identify results from the last loop
        num_layer1 = len(self.layer1_agents)
        # We take the last N results
        relevant_results = state["agent_results"][-num_layer1:]

        # Prepare inputs
        infer_inputs = []
        for i in range(num_samples):
            question = samples["question"][i]

            parts = []
            for result_dict in relevant_results:
                # result_dict is {agent_name: [responses]}
                agent_name = list(result_dict.keys())[0]
                responses = result_dict[agent_name]
                parts.append(f"[{agent_name}]:\n{responses[i]}\n")

            block = "\n".join(parts)
            block = block.replace("\\", "\\\\").replace("{", "{{").replace("}", "}}")

            system_msg = (
                state["agent_system_msgs"][name].replace("{", "{{").replace("}", "}}")
            )
            user_prompt = state["agent_user_msgs"][name].format(
                question=question, block=block
            )

            infer_inputs.append(InferInput(system_msg=system_msg, user_msg=user_prompt))

        # Batch inference
        out_list: List[InferOutput] = self.react_infer_batch(infer_inputs)
        latency = time.time() - t0

        # Process results
        responses = []
        for i, out in enumerate(out_list):
            main_id = samples["main_id"][i]
            savename = self.get_save_name(name, execution_idx)
            self.store_manager.save(
                savename=f"Result_{main_id}-{savename}_sample_{i}",
                data=out.to_dict(),
            )
            responses.append(out.response)

        new_results = state["agent_results"] + [{name: responses}]
        new_cost = state["cost"] + [{name: {"latency": latency}}]
        new_executed = state["agent_executed"] + [name]

        return {
            "init_input": state["init_input"],
            "agent_results": new_results,
            "agent_executed": new_executed,
            "cost": new_cost,
            "agent_system_msgs": state["agent_system_msgs"],
            "agent_user_msgs": state["agent_user_msgs"],
        }

    def check_loop(self, state: AgentState) -> str:
        """
        Determine whether to loop back to the first agent or proceed to orchestrator.
        """
        num_agents = len(self.layer1_agents)
        total_executions = len(state["agent_executed"])

        # Calculate how many full loops have been completed
        # Note: This function is called after the last agent of a loop has executed.
        current_loops = total_executions // num_agents

        if current_loops < self.repeat_count:
            return "continue"
        return "finish"

    def graph(self) -> CompiledStateGraph:
        """
        Build and compile the Orchestrator-Repeat LangGraph.
        """
        g = StateGraph(AgentState)

        # 1. Add nodes for each Layer 1 agent
        previous_node = None
        for name in self.layer1_agents:
            node_func = partial(self.execute_sub_agent, name=name)
            g.add_node(name, node_func)

            if previous_node is None:
                g.set_entry_point(name)
            else:
                g.add_edge(previous_node, name)

            previous_node = name

        # 2. Add Orchestrator Node
        g.add_node("orchestrator", self.execute_orchestrator)

        # 3. Add Conditional Edge from Last Agent
        if previous_node:
            g.add_conditional_edges(
                previous_node,
                self.check_loop,
                {"continue": self.layer1_agents[0], "finish": "orchestrator"},
            )
        else:
            # Fallback if no layer 1 agents
            g.set_entry_point("orchestrator")

        g.add_edge("orchestrator", END)

        return g.compile()
