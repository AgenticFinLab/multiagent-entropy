"""
Sequential multi-agent framework built with LangGraph.

Topology:
Input -> Agent 1 --> Agent 2 --> Agent 3 --> ..... -> Output
"""

import time
from typing import List
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from lmbase.inference.base import InferInput, InferOutput

from maep.generic import AgentState, BaseAgents
from maep.entropy_infer import HFEntropyInference

PLANNER_SYS = """You are the planner agent. Generate plans that are the general instructions only.
Do not execute the plan, do not perform any calculations, and do not produce any answers or intermediate numerical results.
Output a structured, numbered plans."""
PLANNER_USER = """For the question: {question}
Please only generate plans that are guidances required for the subsequent reasoning for the problem-solving. Do not include any specific calculation or numerical results."""

SOLVER_SYS = """You are the solver agent. Solve strictly according to the provided plans. Execute each step precisely and produce the final result.
Output the final result into \\boxed{}."""
SOLVER_USER = """Question: {question}
### Plans ###
{block}
### Plans ###
Follow the plans to solve the question step by step."""

CRITIC_SYS = """You are the critic agent. Review the solver's solution in detail, re-derive independently, and correct any mistakes.
Keep the review terse."""
CRITIC_USER = """Review the solution for: {question}
### Solution ###
{block}
### Solution ###

If corrections are needed, output the mistaken steps and the analysis, otherwise output 'Correct'."""

JUDGER_SYS = """You are the final judge. Audit only the final candidate and ensure it is correct."""
JUDGER_USER = """Final check for: {question}
### Solution ###
{block}
### Solution ###

If correct, output only the final answer with no words, no labels, and no steps."""


class SequentialAgents(BaseAgents):
    """Sequential pipeline of agents using chat-style messages.

    Notes:
    - Uses a shared LLM backend defined in `run_config` (lm_name, inference_config, generation_config).
    - Prompts are initialized via `_get_agent_prompts` using built-in templates.
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
            if name == "planner":
                agent_system_msgs[name] = PLANNER_SYS
                agent_user_msgs[name] = PLANNER_USER
            elif name == "solver":
                agent_system_msgs[name] = SOLVER_SYS
                agent_user_msgs[name] = SOLVER_USER
            elif name == "critic":
                agent_system_msgs[name] = CRITIC_SYS
                agent_user_msgs[name] = CRITIC_USER
            elif name == "judger":
                agent_system_msgs[name] = JUDGER_SYS
                agent_user_msgs[name] = JUDGER_USER
            else:
                raise ValueError(
                    f"Unknown agent name '{name}'. Define templates or extend _get_agent_prompts mapping."
                )
        return agent_system_msgs, agent_user_msgs

    def execute_agent(self, state: AgentState, agent_name: str) -> AgentState:
        """
        Run one agent strictly using prompts stored in state.
        Requires `agent_system_msgs[name]` and `agent_user_msgs[name]` to be present.
        """
        t0 = time.time()
        # get all the samples
        samples = state["input"]
        # Get the number of samples from the length of the question list
        num_samples = len(samples["question"])
        num_executed = len(state["agent_executed"])
        execution_idx = num_executed + 1
        cur_name = agent_name

        # Prepare inputs for all samples
        infer_inputs = []
        for i in range(num_samples):
            question = samples["question"][i]
            system_msg = state["agent_system_msgs"][cur_name]
            user_prompt = state["agent_user_msgs"][cur_name]
            num_executed = len(state["agent_executed"])

            if num_executed > 0:
                prev_result = state["agent_results"][num_executed - 1]
                prev_result = list(prev_result.values())[0]

                for item in prev_result:
                    # as the '\' and '{}' may exist in the prev_result
                    item = (
                        item.replace("\\", "\\\\").replace("{", "{{").replace("}", "}}")
                    )
                user_prompt = user_prompt.format(
                    question=question,
                    block=item,
                )
            else:
                user_prompt = user_prompt.format(question=question)

            system_msg = system_msg.replace("{", "{{").replace("}", "}}")
            # Create InferInput for this sample
            infer_input = InferInput(system_msg=system_msg, user_msg=user_prompt)
            infer_inputs.append(infer_input)

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
        # Store list of all responses
        state["agent_results"].append({cur_name: responses})
        state["cost"].append({cur_name: {"latency": time.time() - t0}})
        state["agent_executed"].append(cur_name)

        return {
            "input": state["input"],
            "agent_results": state["agent_results"],
            "agent_executed": state["agent_executed"],
            "cost": state["cost"],
            "agent_system_msgs": state["agent_system_msgs"],
            "agent_user_msgs": state["agent_user_msgs"],
        }

    def graph(self) -> CompiledStateGraph:
        """Build and compile the sequential LangGraph: entry is the first agent, edges chain through to END."""
        agent_names = list(self.agents_config.keys())
        g = StateGraph(AgentState)
        for name in agent_names:
            g.add_node(name, partial(self.execute_agent, agent_name=name))

        if agent_names:
            g.set_entry_point(agent_names[0])
            for i in range(len(agent_names) - 1):
                g.add_edge(agent_names[i], agent_names[i + 1])
            g.add_edge(agent_names[-1], END)
        return g.compile()


class ConditionalSequentialAgents(SequentialAgents):
    """
    Sequential agents with a conditional loop.
    """

    def check_condition(self, state: AgentState) -> str:
        """
        Check if the last output contains \\box{} or \\boxed{}.
        Returns "end" if found, otherwise returns the name of the first agent to restart.
        """
        last_result = state["agent_results"][-1]
        content = list(last_result.values())[0]

        if "\\box{" in content or "\\boxed{" in content:
            return "end"

        # Restart from the first agent defined in config
        agent_names = list(self.agents_config.keys())
        return agent_names[0]

    def graph(self) -> CompiledStateGraph:
        agent_names = list(self.agents_config.keys())
        g = StateGraph(AgentState)

        # Add nodes with partial to bind agent_name
        for name in agent_names:
            g.add_node(name, partial(self.execute_agent, agent_name=name))

        if agent_names:
            g.set_entry_point(agent_names[0])
            # Linear edges
            for i in range(len(agent_names) - 1):
                g.add_edge(agent_names[i], agent_names[i + 1])

            # Conditional edge from the last agent
            path_map = {name: name for name in agent_names}
            path_map["end"] = END

            g.add_conditional_edges(agent_names[-1], self.check_condition, path_map)

        return g.compile()
