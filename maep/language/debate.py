"""
Debate Multi-Agent System (MAS) implementation with LangGraph.

Topology:
Loop P times:
  Agent 1 -> Agent 2 -> ... -> Agent N
    ^                        |
    |________________________| (if loop < P)
                             | (if loop == P)
                             v
                        Majority Voting -> Output

Memory:
- After the first round, each agent will receive the outputs of all previous agents in all previous rounds.
- In the final round, orchestrator will receive the outputs of all agents in the current rounds and output the final answer through majority voting.

LLM calls: r * N (rounds * number of agents, no orchestrator call)

Corresponding to the MAS (Decentralized Multi-Agent System) [1] with two layers:
- Layer 1: N sequential agents running in a loop P times.
- Layer 2: Majority Voting that extracts answers from \\boxed{} and selects the most frequent one after P rounds.

Key Difference from Decentralized MAS:
- In Decentralized MAS, the orchestrator uses LLM inference to aggregate all agent outputs and generate the final answer. In Debate MAS, the orchestrator does NOT use LLM inference. Instead, it performs majority voting:
  1. Extracts final answers wrapped in \\boxed{} from each agent's response
  2. Compares all extracted answers and counts their occurrences
  3. Selects the most frequent answer as the final result
- In Decentralized MAS, each agent only receives the output from the immediately previous agent. In Debate MAS, each agent receives the outputs from ALL previous agents in the current loop.
  This means Agent N receives outputs from Agents 1, 2, ..., N-1 in the current loop, whereas in Decentralized MAS, Agent N only receives output from Agent N-1.
- The agent execution logic (sequential execution with previous round context) is identical to Decentralized MAS.

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


class DebateMAS(BaseAgents):
    """
    Debate Multi-Agent System (MAS) implementation.

    In this architecture, each layer1 agent works sequentially in a loop P times.
    The orchestrator uses majority voting instead of LLM inference to determine the final answer.
    It extracts final answers wrapped in \\boxed{} from each agent's response and selects the most frequent one.
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

    def __init__(self, run_config: dict):
        super().__init__(run_config)
        self.repeat_count = run_config["round"]
        self.layer1_agents: List[str] = None
        self.orchestrator: str = None
        self.get_layer_agents()

    def get_layer_agents(self):
        """Get agents of the first layer and the orchestrator."""
        self.layer1_agents = list(self.agents_config.keys())
        self.orchestrator = "orchestrator"

    def execute_sub_agent(self, state: AgentState, name: str) -> AgentState:
        """
        Execute a single layer1 agent, incorporating context from the previous round's agents.
        """
        samples = state["init_input"]
        num_samples = len(samples["question"])
        execution_idx = len(state["agent_executed"]) + 1

        t0 = time.time()
        infer_inputs = []
        for i in range(num_samples):
            question = samples["question"][i]

            context = ""
            if state["agent_results"]:
                num_layer1 = len(self.layer1_agents)
                total_executions = len(state["agent_executed"])

                if total_executions >= num_layer1:
                    prev_loop_start = total_executions - num_layer1
                    prev_loop_results = state["agent_results"][
                        prev_loop_start:total_executions
                    ]

                    context_parts = []
                    for result_dict in prev_loop_results:
                        agent_name = list(result_dict.keys())[0]
                        responses = result_dict[agent_name]
                        context_parts.append(f"[{agent_name}]:\n{responses[i]}")

                    if context_parts:
                        context = "\n\nPrevious Round Outputs:\n" + "\n\n".join(
                            context_parts
                        )

            system_msg = (
                state["agent_system_msgs"][name].replace("{", "{{").replace("}", "}}")
            )
            base_user_prompt = state["agent_user_msgs"][name].format(question=question)

            user_prompt = f"{base_user_prompt}{context}"

            infer_inputs.append(InferInput(system_msg=system_msg, user_msg=user_prompt))

        out_list: List[InferOutput] = self.agents_lm.infer_batch(infer_inputs)
        latency = time.time() - t0

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

    def execute_orchestrator(self, state: AgentState) -> AgentState:
        """
        Use majority voting to determine the final answer.

        Extracts final answers wrapped in \\boxed{} from each agent's response
        and selects the most frequent one as the final answer.
        """
        samples = state["init_input"]
        num_samples = len(samples["question"])
        execution_idx = len(state["agent_executed"]) + 1
        name = self.orchestrator

        t0 = time.time()

        num_layer1 = len(self.layer1_agents)
        relevant_results = state["agent_results"][-num_layer1:]

        final_answers = []
        answer_votes_list = []

        for i in range(num_samples):
            all_answers = []

            for result_dict in relevant_results:
                agent_name = list(result_dict.keys())[0]
                responses = result_dict[agent_name]

                answer = self._extract_boxed_answer(responses[i])
                if answer:
                    all_answers.append(answer)

            if not all_answers:
                final_answers.append("No valid answers found in agent responses")
                answer_votes_list.append({})
            else:
                from collections import Counter

                answer_counts = Counter(all_answers)
                most_common_answer, count = answer_counts.most_common(1)[0]
                final_answers.append(most_common_answer)
                answer_votes_list.append(dict(answer_counts))

        latency = time.time() - t0

        for i in range(num_samples):
            main_id = samples["main_id"][i]
            savename = self.get_save_name(name, execution_idx)

            orchestrator_result = {
                "final_answer": final_answers[i],
                "answer_votes": answer_votes_list[i],
                "latency": latency,
                "all_agent_answers": [
                    result_dict[list(result_dict.keys())[0]][i]
                    for result_dict in relevant_results
                ],
            }

            self.store_manager.save(
                savename=f"Result_{main_id}-{savename}_sample_{i}",
                data=orchestrator_result,
            )

        new_results = state["agent_results"] + [{name: final_answers}]
        new_cost = state["cost"] + [
            {name: {"latency": latency, "answer_votes": answer_votes_list}}
        ]
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

    def _extract_boxed_answer(self, text: str):
        """
        Extract the final answer wrapped in \\boxed{} from the response text.

        Args:
            text: The agent's response text

        Returns:
            The extracted answer, or None if no \\boxed{} is found
        """
        import re

        pattern = r"\\boxed\{([^}]+)\}"
        matches = re.findall(pattern, text)

        if matches:
            return matches[-1].strip()

        return None

    def check_loop(self, state: AgentState) -> str:
        """
        Determine whether to loop back to the first agent or proceed to orchestrator.
        """
        num_agents = len(self.layer1_agents)
        total_executions = len(state["agent_executed"])

        current_loops = total_executions // num_agents

        if current_loops < self.repeat_count:
            return "continue"
        return "finish"

    def graph(self) -> CompiledStateGraph:
        """
        Build and compile the Debate LangGraph.
        """
        g = StateGraph(AgentState)

        previous_node = None
        for name in self.layer1_agents:
            node_func = partial(self.execute_sub_agent, name=name)
            g.add_node(name, node_func)

            if previous_node is None:
                g.set_entry_point(name)
            else:
                g.add_edge(previous_node, name)

            previous_node = name

        g.add_node("orchestrator", self.execute_orchestrator)

        if previous_node:
            g.add_conditional_edges(
                previous_node,
                self.check_loop,
                {"continue": self.layer1_agents[0], "finish": "orchestrator"},
            )
        else:
            g.set_entry_point("orchestrator")

        g.add_edge("orchestrator", END)

        return g.compile()
