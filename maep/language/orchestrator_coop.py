"""
Orchestrator-Aggregator (two-layer) multi-agent structure with LangGraph.

Diagram
      Input
        │
    ┌─────┴─────┐
    ▼     ▼     ▼
Agent1 Agent2 Agent3   (Layer 1: parallel agents)
    └─────┬─────┘
        ▼
   Orchestrator        (Layer 2: aggregator)
        │
      Output

Core Idea
- Multiple nodes in the graph:
  1) `Agent1` ... `AgentN`: run first-layer agents sequentially (but logically parallel tasks).
  2) `orchestrator`: aggregates the outputs from layer 1 and produces a final result.
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

# Layer 1 Agents (example roles)
MATH_SYS = """You are the MathAgent. Solve the given question with clear steps."""
MATH_USER = """Question: {question}
Provide a concise mathematical solution, showing key steps."""

SCIENCE_SYS = """You are the ScienceAgent. Analyze and solve the given question with scientific reasoning."""
SCIENCE_USER = """Question: {question}
Explain your scientific reasoning and provide a final result."""

CODE_SYS = """You are the CodeAgent. Provide a self-contained Python function that solves the problem."""
CODE_USER = """Question: {question}
Write a single self-contained Python function in a markdown code block that solves the problem."""

# Layer 2 Orchestrator
ORCHESTRATOR_SYS = """You are the Orchestrator Agent. Your task is to aggregate the solutions provided by the first-layer agents and produce a final, comprehensive answer.
Analyze the provided solutions, resolve any conflicts, and synthesize a coherent final response."""

ORCHESTRATOR_USER = """Question: {question}

Here are the solutions from the expert agents:
=== Solutions ===
{block}
=== Solutions ===

Based on these inputs, provide the final answer."""


class OrchestratorAggAgents(BaseAgents):
    """
    A two-layer multi-agent structure where N agents run in parallel (Layer 1),
    and their outputs are aggregated by an Orchestrator (Layer 2).
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
        for name in self.agents_config.keys():
            if name == "MathAgent":
                agent_system_msgs[name] = MATH_SYS
                agent_user_msgs[name] = MATH_USER
            elif name == "ScienceAgent":
                agent_system_msgs[name] = SCIENCE_SYS
                agent_user_msgs[name] = SCIENCE_USER
            elif name == "CodeAgent":
                agent_system_msgs[name] = CODE_SYS
                agent_user_msgs[name] = CODE_USER
            elif name == "OrchestratorAgent":
                agent_system_msgs[name] = ORCHESTRATOR_SYS
                agent_user_msgs[name] = ORCHESTRATOR_USER
            else:
                raise ValueError(
                    f"Unknown agent name '{name}'. Extend _get_agent_prompts mapping."
                )
        return agent_system_msgs, agent_user_msgs

    def __init__(self, run_config):
        super().__init__(run_config)
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
        num_samples = len(samples)
        execution_idx = len(state["agent_executed"]) + 1

        t0 = time.time()
        # Prepare inputs for all samples
        infer_inputs = []
        for i in range(num_samples):
            question = samples[i]["question"]
            main_id = samples[i]["main_id"]

            system_msg = (
                state["agent_system_msgs"][name].replace("{", "{{").replace("}", "}}")
            )
            user_prompt = state["agent_user_msgs"][name].format(question=question)

            infer_inputs.append(InferInput(system_msg=system_msg, user_msg=user_prompt))

        # Batch inference for all samples
        out_list: List[InferOutput] = self.agents_lm.infer_batch(infer_inputs)
        latency = time.time() - t0

        # Process results for each sample
        responses = []
        for i, out in enumerate(out_list):
            main_id = samples[i]["main_id"]
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
        num_samples = len(samples)
        execution_idx = len(state["agent_executed"]) + 1
        name = self.orchestrator

        t0 = time.time()

        # Prepare inputs for all samples
        infer_inputs = []
        for i in range(num_samples):
            question = samples[i]["question"]
            main_id = samples[i]["main_id"]

            # Aggregate results from layer1 agents for this sample
            # state["agent_results"] is a list of dicts: [{"Agent1": [res_sample0, ...]}, ...]
            # We need to extract the i-th sample's response from each layer 1 agent
            parts = []
            for agent_name in self.layer1_agents:
                # Find the result for this agent
                for result_dict in state["agent_results"]:
                    if agent_name in result_dict:
                        responses = result_dict[agent_name]
                        if i < len(responses):
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
            main_id = samples[i]["main_id"]
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

        g.add_edge("orchestrator", END)

        return g.compile()
