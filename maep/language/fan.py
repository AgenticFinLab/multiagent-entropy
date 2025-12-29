"""
Fan-in (two-layer) multi-agent structure with LangGraph.

Diagram
      Input
        │
    ┌─────┴─────┐
    ▼     ▼     ▼
Agent1 Agent2 Agent3   (Layer 1: independent agents)
    └─────┬─────┘
        ▼
      Judger              (Layer 2: integrator)
        │
      Output

Core Idea
- Two nodes in the graph:
  1) `layer1_all`: run all first-layer agents over the same `question`.
     - Parallelized with `ThreadPoolExecutor` (max_workers = number of layer-1 agents).
     - Each agent renders system/user prompts from `AgentState`, calls the unified inference backend, and records per-agent latency.
  2) One node in layer 2 integrate the first-layer outputs.


"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from lmbase.inference.base import InferInput, InferOutput
from lmag.generic import AgentState, BaseAgents

MATH_SYS = """You are the MathAgent. Solve the given question with clear steps."""
MATH_USER = """Question: {question}
Provide a concise mathematical solution, showing key steps."""

SCIENCE_SYS = """You are the ScienceAgent. Analyze and solve the given question with scientific reasoning."""
SCIENCE_USER = """Question: {question}
Explain your scientific reasoning and provide a final result."""

CODE_SYS = """You are the CodeAgent. Provide a self-contained Python function that solves the problem."""
CODE_USER = """Question: {question}
Write a single self-contained Python function in a markdown code block that solves the problem."""

SUMMARIZER_SYS = """You are the SummarizerAgent. Aggregate the reasoning and produce a concise final answer."""
SUMMARIZER_USER = """Question: {question}
Based on the following series of reasoning:
{block}
Provide the final answer concisely."""


class FanAgentsTwoLayer(BaseAgents):
    """A two-layer fan-in multi-agent structure."""

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
            elif name == "SummarizerAgent":
                agent_system_msgs[name] = SUMMARIZER_SYS
                agent_user_msgs[name] = SUMMARIZER_USER
            else:
                raise ValueError(
                    f"Unknown agent name '{name}'. Extend _get_agent_prompts mapping."
                )
        return agent_system_msgs, agent_user_msgs

    def __init__(self, run_config):
        super().__init__(run_config)
        # Get layer-1 agents and layer-2 agent
        self.layer1_agents: List[str] = None
        self.layer2_agent: str = None

        self.get_layer_agents()

    def get_layer_agents(self):
        """Get agents of the first layer."""
        self.layer1_agents = [
            name
            for name, settings in self.agents_config.items()
            if "link_to" in settings
        ]
        self.layer2_agent = [
            name for name in self.agents_config.keys() if name not in self.layer1_agents
        ][0]

    def execute_layer1_all(self, state: AgentState) -> AgentState:
        """
        Execute all layer1 agents in parallel.

        Args:
        - state: current agent state.

        Returns:
        - state: updated agent state with results from all layer1 agents.
        """
        question = state["input"][0]["question"]

        def run_one(name: str):
            t0 = time.time()
            system_msg = (
                state["agent_system_msgs"][name].replace("{", "{{").replace("}", "}}")
            )
            user_prompt = state["agent_user_msgs"][name].format(question=question)
            out: InferOutput = self.agents_lm.run(
                InferInput(system_msg=system_msg, user_msg=user_prompt)
            )
            latency = time.time() - t0
            return name, out, latency

        results = []
        # Run all layer-1 agents concurrently via a thread pool; max_workers equals number of agents
        # Collect results as futures complete (submission order preserved when committing to state)
        with ThreadPoolExecutor(max_workers=len(self.layer1_agents) or 1) as ex:
            futures = [ex.submit(run_one, name) for name in self.layer1_agents]
            for f in futures:
                results.append(f.result())

        res_map = {n: (out, lat) for n, out, lat in results}
        for name in self.layer1_agents:
            out, lat = res_map[name]
            savename = self.get_save_name(name, len(state["agent_executed"]) + 1)
            self.store_manager.save(
                savename=f"Result_{state['input'][0]['main_id']}-{savename}",
                data=out.to_dict(),
            )
            state["agent_results"].append({name: out.response})
            state["cost"].append({name: {"latency": lat}})
            state["agent_executed"].append(name)

        return {
            "input": state["input"],
            "agent_results": state["agent_results"],
            "agent_executed": state["agent_executed"],
            "cost": state["cost"],
            "agent_system_msgs": state["agent_system_msgs"],
            "agent_user_msgs": state["agent_user_msgs"],
        }

    def execute_agent(self, state: AgentState, agent_name: str) -> AgentState:
        question = state["input"][0]["question"]
        name = agent_name
        t0 = time.time()
        system_msg = state["agent_system_msgs"][name]
        user_prompt = state["agent_user_msgs"][name]
        parts = []
        for n in self.layer1_agents:
            for kv in state["agent_results"]:
                if n in kv:
                    parts.append(f"[{n}]:\n{kv[n]}\n")
                    break
        block = "\n".join(parts)
        block = block.replace("\\", "\\\\").replace("{", "{{").replace("}", "}}")
        system_msg = system_msg.replace("{", "{{").replace("}", "}}")
        user_prompt = user_prompt.format(question=question, block=block)
        out: InferOutput = self.agents_lm.run(
            InferInput(system_msg=system_msg, user_msg=user_prompt)
        )
        savename = self.get_save_name(name, len(state["agent_executed"]) + 1)
        self.store_manager.save(
            savename=f"Result_{state['input'][0]['main_id']}-{savename}",
            data=out.to_dict(),
        )
        state["agent_results"].append({name: out.response})
        state["cost"].append({name: {"latency": time.time() - t0}})
        state["agent_executed"].append(name)
        return {
            "input": state["input"],
            "agent_results": state["agent_results"],
            "agent_executed": state["agent_executed"],
            "cost": state["cost"],
            "agent_system_msgs": state["agent_system_msgs"],
            "agent_user_msgs": state["agent_user_msgs"],
        }

    def graph(self) -> CompiledStateGraph:
        """
        Build and compile the two-layer fan-in LangGraph.

        Topology
        - Entry: `layer1_all` (parallel execution over layer-1 agents)
        - Fan-in: `layer2_agent` (aggregates and terminates)

        Notes
        - `layer2_agent` is required and always executed after `layer1_all`.
        - Node functions read and write `AgentState` only; no external side effects beyond trace persistence.
        """
        # Nodes: first-layer aggregator and the required second-layer integrator
        g = StateGraph(AgentState)
        g.add_node("layer1_all", self.execute_layer1_all)
        g.add_node(
            self.layer2_agent, partial(self.execute_agent, agent_name=self.layer2_agent)
        )
        # Entry: layer-1; Edge: fan-in to layer-2; Terminal: END
        g.set_entry_point("layer1_all")
        g.add_edge("layer1_all", self.layer2_agent)
        g.add_edge(self.layer2_agent, END)
        return g.compile()
