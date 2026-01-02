"""
Orchestrator-Repeat (two-layer with loop) multi-agent structure with LangGraph - Hybrid Version.

Topology:
Loop P times:
  Agent 1 -> Agent 2 -> ... -> Agent N -> Orchestrator
    ^                                    |
    |____________________________________| (if loop < P)
                                         | (if loop == P)
                                         v
                                        END

Memory:
- Each layer 1 agent will receive the outputs of all previous agents and orchestrator, including the output of the current round and all previous rounds.
- In each round, Orchestrator will receive the outputs of all agents in the current round and output the feedback.

LLM calls: r * N (rounds * number of agents) + r (rounds)

Corresponding to the MAS (Decentralized Multi-Agent System) [1] with two layers:
- Layer 1: N sequential agents running in a loop P times.
- Layer 2: Orchestrator that aggregates the outputs of all agents after EACH loop.

Key Difference from Decentralized MAS:
- In Decentralized MAS, each agent only receives the output from the immediately previous agent. In Hybrid MAS, each agent receives the outputs from ALL previous agents in the current loop.
  This means Agent N receives outputs from Agents 1, 2, ..., N-1 in the current loop, whereas in Decentralized MAS, Agent N only receives output from Agent N-1.
- In Decentralized MAS, Orchestrator only aggregated the outputs of all agents after P rounds. In Hybrid MAS, Orchestrator aggregated the outputs of all agents after EACH loop.

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


class OrchestratorHybrid(OrchestratorCentralized):
    """
    Hybrid Multi-Agent System (Hybrid MAS) with a repeat loop.

    Structure:
    - Layer 1 (Agents): N agents run sequentially in a loop for `repeat_count` (P) iterations.
      - Agent 1 -> Agent 2 -> ... -> Agent N -> (Loop back to Agent 1 if p < P)
    - Layer 2 (Orchestrator): After P iterations, the Orchestrator aggregates the results.

    Key Difference from Decentralized MAS:
    - In Decentralized MAS, each agent only receives the output from the immediately previous agent.
    - In Hybrid MAS, each agent receives the outputs from ALL previous agents in the current loop.
    """

    def __init__(self, run_config: dict):
        super().__init__(run_config)
        self.repeat_count = run_config["round"]

    def execute_sub_agent(self, state: AgentState, name: str) -> AgentState:
        """
        Execute a single layer1 agent, incorporating context from previous round's Orchestrator and all previous agents in current loop.
        """
        samples = state["init_input"]
        num_samples = len(samples["question"])
        execution_idx = len(state["agent_executed"]) + 1

        t0 = time.time()
        # Prepare inputs for all samples
        infer_inputs = []
        for i in range(num_samples):
            question = samples["question"][i]

            # 1. Build context from previous round's Orchestrator and agents
            prev_context = ""
            if state["agent_results"]:
                num_layer1 = len(self.layer1_agents)
                total_executions = len(state["agent_executed"])
                
                # Calculate current loop index based on orchestrator executions
                orch_executions = [name for name in state["agent_executed"] if name == self.orchestrator]
                current_loop_idx = len(orch_executions)
                current_agent_idx = self.layer1_agents.index(name)

                # First, add Orchestrator feedback from previous round (if exists)
                if current_loop_idx > 0:
                    # Find the most recent Orchestrator result
                    for result_dict in reversed(state["agent_results"]):
                        if self.orchestrator in result_dict:
                            orch_responses = result_dict[self.orchestrator]
                            prev_context = f"\n\nGuidance from {self.orchestrator} (Previous Round):\n{orch_responses[i]}"
                            break

                # Then, add previous round's agents' outputs
                if current_loop_idx > 0:
                    # Find the start index of previous round's layer1 agents
                    # Previous round's orchestrator is at position: current_loop_idx * (num_layer1 + 1) - 1
                    # Previous round's layer1 agents start at: (current_loop_idx - 1) * (num_layer1 + 1)
                    prev_loop_start = (current_loop_idx - 1) * (num_layer1 + 1)
                    prev_loop_results = state["agent_results"][prev_loop_start:prev_loop_start + num_layer1]

                    context_parts = []
                    for result_dict in prev_loop_results:
                        agent_name = list(result_dict.keys())[0]
                        responses = result_dict[agent_name]
                        context_parts.append(f"[{agent_name}]:\n{responses[i]}")

                    if context_parts:
                        prev_context += "\n\nPrevious Round Agent Outputs:\n" + "\n\n".join(context_parts)

                    # Also include all previous agents in current round
                    # Current round's layer1 agents start at: current_loop_idx * (num_layer1 + 1)
                    loop_start_idx = current_loop_idx * (num_layer1 + 1)
                    loop_results = state["agent_results"][loop_start_idx:loop_start_idx + num_layer1]
                    relevant_results = loop_results[:current_agent_idx]

                    if relevant_results:
                        current_round_parts = []
                        for result_dict in relevant_results:
                            agent_name = list(result_dict.keys())[0]
                            responses = result_dict[agent_name]
                            current_round_parts.append(f"[{agent_name}]:\n{responses[i]}")

                        if current_round_parts:
                            prev_context += "\n\nCurrent Round Previous Outputs:\n" + "\n\n".join(current_round_parts)
                else:
                    # First round: only include previous agents in current round
                    loop_start_idx = 0
                    loop_results = state["agent_results"][loop_start_idx:loop_start_idx + num_layer1]
                    relevant_results = loop_results[:current_agent_idx]

                    if relevant_results:
                        context_parts = []
                        for result_dict in relevant_results:
                            agent_name = list(result_dict.keys())[0]
                            responses = result_dict[agent_name]
                            context_parts.append(f"[{agent_name}]:\n{responses[i]}")

                        if context_parts:
                            prev_context += "\n\n" + "\n\n".join(context_parts)

            # 2. Format Prompt
            system_msg = (
                state["agent_system_msgs"][name].replace("{", "{{").replace("}", "}}")
            )
            base_user_prompt = state["agent_user_msgs"][name].format(question=question)

            # Append context from previous round and current round's previous agents
            user_prompt = f"{base_user_prompt}{prev_context}"

            infer_inputs.append(InferInput(system_msg=system_msg, user_msg=user_prompt))

        # Batch inference
        out_list: List[InferOutput] = self.agents_lm.infer_batch(infer_inputs)
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
        out_list: List[InferOutput] = self.agents_lm.infer_batch(infer_inputs)
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


    def graph(self) -> CompiledStateGraph:
        """
        Build and compile the Orchestrator-Repeat LangGraph.

        New Topology:
        Loop P times:
          Agent 1 -> Agent 2 -> ... -> Agent N -> Orchestrator
            ^                                    |
            |____________________________________| (if loop < P)
                                                 | (if loop == P)
                                                 v
                                                END
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

        # 3. Connect last Layer 1 agent to Orchestrator
        if previous_node:
            g.add_edge(previous_node, "orchestrator")
        else:
            # Fallback if no layer 1 agents
            g.set_entry_point("orchestrator")

        # 4. Add Conditional Edge from Orchestrator to control looping
        g.add_conditional_edges(
            "orchestrator",
            self.check_orchestrator_loop,
            {"continue": self.layer1_agents[0], "finish": END},
        )

        return g.compile()

    def check_orchestrator_loop(self, state: AgentState) -> str:
        """
        Determine whether to loop back to the first agent or finish after Orchestrator.
        """
        orch_executions = [
            name for name in state["agent_executed"] if name == self.orchestrator
        ]
        num_orch = len(orch_executions)

        if num_orch < self.repeat_count:
            return "continue"
        return "finish"
