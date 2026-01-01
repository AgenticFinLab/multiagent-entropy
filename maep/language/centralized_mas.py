"""
Centralized Multi-Agent System (MAS) structure with LangGraph.

Diagram
      Input
        │
        ▼
    ┌───┴───┐ <───────────────────────────┐
    ▼       ▼                             │
 Agent1   Agent2 ... (Layer 1: Agents)    │
    └─────┬─────┘                         │
          ▼                               │
     Orchestrator    (Layer 2)            │
          │                               │
          ├─────── (r < R: Feedback) ─────┘
          │
          ▼ (r == R)
        Output

Core Idea
- Two-layer architecture:
  1) Layer 1: Multiple domain-specific agents (Agent1...AgentN) execute in parallel (logically) or sequentially to generate initial solutions.
  2) Layer 2: A Centralized Orchestrator aggregates these solutions, resolves conflicts, and produces the final output.

Key Features:
- **Centralized Control**: The Orchestrator acts as the single source of truth and final decision maker.
- **Iterative Refinement (Optional)**: Supports multi-round execution where the Orchestrator's feedback is fed back to Layer 1 agents for refinement (controlled by `max_rounds`).

Corresponding to the MAS (Centralized Multi-Agent System) [1] where a central node coordinates distributed agents.

[1]. Towards a Science of Scaling Agent Systems.
"""

import time
from typing import List
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from lmbase.inference.base import InferInput, InferOutput

from maep.generic import AgentState, BaseAgents
from maep.entropy_infer import HFEntropyInference

# --- Prompts ---
# Removed hardcoded prompts. Now loaded from config.


class OrchestratorCentralized(BaseAgents):
    """
    Centralized Multi-Agent System (Centralized MAS).

    Architecture:
    - Layer 1 (Workers): Multiple domain-specific agents (e.g., Math, Code) run in parallel/sequence.
    - Layer 2 (Orchestrator): A central authority that aggregates Layer 1 outputs and provides feedback.

    Iteration Logic (R-rounds):
    - The process repeats for a maximum of `R` rounds (defined by `max_rounds`).
    - Round `r` (1 <= r <= R):
      1. Layer 1 agents execute, incorporating feedback from the Orchestrator (from round `r-1`).
      2. Orchestrator executes, aggregating all Layer 1 outputs from the current round.
      3. If `r < R`, the Orchestrator's output serves as feedback for the next round `r+1`.
      4. If `r == R`, the process terminates, and the Orchestrator's output is final.

    """

    def define_agent_models(self):
        """Construct HFEntropyInference."""
        self.agents_lm = HFEntropyInference(
            lm_name=self.run_config["lm_name"],
            inference_config=self.run_config["inference_config"],
            entropy_config=self.run_config["entropy_config"],
            generation_config=self.run_config["generation_config"],
        )

    def _get_agent_prompts(self):
        agent_system_msgs = {}
        agent_user_msgs = {}

        for name, config in self.agents_config.items():
            if "sys_message" in config:
                agent_system_msgs[name] = self._load_from_module(config["sys_message"])
            if "user_message" in config:
                agent_user_msgs[name] = self._load_from_module(config["user_message"])

        return agent_system_msgs, agent_user_msgs

    def __init__(self, run_config):
        super().__init__(run_config)
        self.max_rounds = run_config["max_rounds"]
        # Get layer-1 agents and layer-2 agent
        self.layer1_agents: List[str] = None
        self.orchestrator: str = None

        self.get_layer_agents()

    def get_layer_agents(self):
        """Get agents of the first layer and the orchestrator."""
        # Layer 1 agents are those configured to link to the orchestrator (or just not the orchestrator itself)
        # Assuming 'link_to' logic or simply all non-orchestrator agents are layer 1
        self.layer1_agents = [
            name for name in self.agents_config.keys() if name != "OrchestratorAgent"
        ]
        # Layer 2 is the Orchestrator
        self.orchestrator = "OrchestratorAgent"

        if self.orchestrator not in self.agents_config:
            raise ValueError("OrchestratorAgent must be defined in agents_config")

    def execute_sub_agent(self, state: AgentState, name: str) -> AgentState:
        """
        Execute a single layer1 agent. Used as a node in LangGraph.

        Args:
        - state: current agent state.
        - name: name of the agent to execute.

        Returns:
        - state: updated agent state.
        """
        samples = state["init_input"]
        # Get the number of samples from the length of the question list
        num_samples = len(samples["question"])
        execution_idx = len(state["agent_executed"]) + 1

        t0 = time.time()
        # Prepare inputs for all samples
        infer_inputs = []
        for i in range(num_samples):
            question = samples["question"][i]
            main_id = samples["main_id"][i]

            system_msg = (
                state["agent_system_msgs"][name].replace("{", "{{").replace("}", "}}")
            )
            base_user_prompt = state["agent_user_msgs"][name].format(question=question)

            # Build context from Orchestrator (if available from previous outer loop)
            orch_context = ""
            # Find the most recent Orchestrator result
            # We search backwards
            for result_dict in reversed(state["agent_results"]):
                if self.orchestrator in result_dict:
                    orch_responses = result_dict[self.orchestrator]
                    orch_context = (
                        f"\n\nGuidance from {self.orchestrator}:\n{orch_responses[i]}"
                    )
                    break

            # Append context
            user_prompt = f"{base_user_prompt}{orch_context}"

            infer_inputs.append(InferInput(system_msg=system_msg, user_msg=user_prompt))

        # Batch inference for all samples
        out_list: List[InferOutput] = self.agents_lm.infer_batch(infer_inputs)
        latency = time.time() - t0

        # Process results for each sample
        responses = []
        for i, out in enumerate(out_list):
            main_id = samples["main_id"][i]
            # Save each sample's result individually
            savename = self.get_save_name(name, execution_idx)
            self.store_manager.save(
                savename=f"Result_{main_id}-{savename}_sample_{i}",
                data=out.to_dict(),
            )
            responses.append(out.response)

        # Update state manually (since AgentState uses List without reducers, we append to the list)
        # Note: In LangGraph, we typically return the diff, but since we lack a reducer,
        # we ensure we are appending to the existing list from the state passed in.
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
        Execute the Orchestrator agent to aggregate results from Layer 1.

        Args:
        - state: current agent state containing layer 1 results.

        Returns:
        - state: updated agent state with the orchestrator's final result.
        """
        samples = state["init_input"]
        # Get the number of samples from the length of the question list
        num_samples = len(samples["question"])
        execution_idx = len(state["agent_executed"]) + 1
        name = self.orchestrator

        t0 = time.time()

        # Prepare inputs for all samples
        infer_inputs = []
        for i in range(num_samples):
            question = samples["question"][i]
            main_id = samples["main_id"][i]

            # Aggregate results from layer1 agents for this sample
            # state["agent_results"] is a list of dicts: [{"Agent1": [res_sample0, ...]}, ...]
            # We need to extract the i-th sample's response from each layer 1 agent
            # CRITICAL: We must get the LATEST result from the current round, so we search backwards.
            parts = []
            for agent_name in self.layer1_agents:
                # Find the result for this agent (most recent)
                for result_dict in reversed(state["agent_results"]):
                    if agent_name in result_dict:
                        responses = result_dict[agent_name]
                        parts.append(f"[{agent_name}]:\n{responses[i]}\n")
                        break

            block = "\n".join(parts)
            # Escape braces for format method
            block = block.replace("\\", "\\\\").replace("{", "{{").replace("}", "}}")

            system_msg = (
                state["agent_system_msgs"][name].replace("{", "{{").replace("}", "}}")
            )
            user_prompt = state["agent_user_msgs"][name].format(
                question=question, block=block
            )

            infer_inputs.append(InferInput(system_msg=system_msg, user_msg=user_prompt))

        # Batch inference for the orchestrator
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

        # Update state manually (append new results to lists)
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

    def execute_agent(self, state: AgentState, agent_name: str) -> AgentState:
        """
        Unified execution interface required by BaseAgents.
        Routes to specific execution methods based on the agent layer.
        """
        if agent_name == self.orchestrator:
            return self.execute_orchestrator(state)
        elif agent_name in self.layer1_agents:
            return self.execute_sub_agent(state, agent_name)
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    def check_outer_loop(self, state: AgentState) -> str:
        """
        Determine whether to start a new Outer Loop or End.
        Checks if Orchestrator execution count < max_rounds (R).
        """
        orch_executions = [
            name for name in state["agent_executed"] if name == self.orchestrator
        ]
        num_orch = len(orch_executions)

        if num_orch < self.max_rounds:
            return "continue"  # Go back to Layer 1
        return "finish"  # End

    def graph(self) -> CompiledStateGraph:
        """
        Build and compile the two-layer Orchestrator-Aggregator LangGraph.

        Topology
        - Entry: Agent 1 -> Agent 2 -> ... -> Agent N (Sequential Layer 1)
        - Edge: Agent N -> Orchestrator
        - Terminal: Orchestrator -> END
        """
        g = StateGraph(AgentState)

        # 1. Add nodes for each Layer 1 agent
        # We assume self.layer1_agents is ordered.
        previous_node = None
        for name in self.layer1_agents:
            # Use partial to bind the agent name to the function
            node_func = partial(self.execute_sub_agent, name=name)
            g.add_node(name, node_func)

            if previous_node is None:
                # First agent is the entry point
                g.set_entry_point(name)
            else:
                # Chain previous agent to current agent
                g.add_edge(previous_node, name)

            previous_node = name

        # 2. Add Orchestrator Node (Layer 2)
        g.add_node("orchestrator", self.execute_orchestrator)

        # 3. Connect last Layer 1 agent to Orchestrator
        if previous_node:
            g.add_edge(previous_node, "orchestrator")
        else:
            # Edge case: No layer 1 agents? Directly entry to orchestrator
            g.set_entry_point("orchestrator")

        # 4. Add Conditional Edge from Orchestrator (Outer Loop Control)
        g.add_conditional_edges(
            "orchestrator",
            self.check_outer_loop,
            {"continue": self.layer1_agents[0], "finish": END},
        )

        return g.compile()
