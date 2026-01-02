"""
Sequential multi-agent framework built with LangGraph.

Topology:
Input -> Agent 1 --> Agent 2 --> Agent 3 --> ..... -> Output

For multiple rounds:
Input -> Agent 1 --> Agent 2 --> Agent 3 --> Agent 1 --> ..... -> Output
after each round, the history is aggregated and passed to the Agent 1 in next round for further planning.
LLM calls: r * N (rounds * number of agents)
"""

import time
from typing import List
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from lmbase.inference.base import InferInput, InferOutput

from maep.generic import AgentState, BaseAgents
from maep.entropy_infer import HFEntropyInference


class SequentialAgents(BaseAgents):
    """Sequential pipeline of agents using chat-style messages.

    Notes:
    - Uses a shared LLM backend defined in `run_config` (lm_name, inference_config, generation_config).
    - Prompts are initialized via `_get_agent_prompts` using built-in templates.
    """

    def __init__(self, run_config: dict):
        super().__init__(run_config)
        # Number of rounds for the sequential pipeline
        self.round = run_config["round"]
        # History aggregation flag
        self.aggregate_history = run_config.get("aggregate_history", True)
        # History optimization settings
        self.max_history_chars = run_config.get("max_history_chars", 10000)
        self.max_history_rounds = run_config.get("max_history_rounds", 0)

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
        
        agent_names = list(self.agents_config.keys())
        history_parts = []
        
        # Group executions by rounds
        rounds_completed = 0
        i = 0
        while i < len(state["agent_executed"]):
            # Check if the next sequence matches the agent_names (a complete round)
            match = True
            for j, agent_name in enumerate(agent_names):
                if (
                    i + j >= len(state["agent_executed"])
                    or state["agent_executed"][i + j] != agent_name
                ):
                    match = False
                    break
            
            if match:
                rounds_completed += 1
                # Add this round's history
                round_history = []
                for j, agent_name in enumerate(agent_names):
                    agent_result = state["agent_results"][i + j]
                    response = agent_result[agent_name][sample_idx]
                    round_history.append(
                        f"  - {agent_name}: {response}"
                    )
                
                history_parts.append(
                    f"Round {rounds_completed}:\n" + "\n".join(round_history)
                )
                i += len(agent_names)
            else:
                i += 1
        
        if not history_parts:
            return ""
        
        # Apply max_history_rounds limit
        if self.max_history_rounds > 0:
            history_parts = history_parts[-self.max_history_rounds:]
        
        history_text = (
            "### Previous Rounds History ###\n"
            + "\n\n".join(history_parts)
            + "\n### End of History ###\n\n"
        )
        
        # Apply max_history_chars limit
        if self.max_history_chars > 0 and len(history_text) > self.max_history_chars:
            # Truncate from the beginning to keep most recent history
            history_text = (
                "### Previous Rounds History ###\n"
                + "...[earlier history truncated due to size limit]...\n\n"
                + history_text[-(self.max_history_chars - 100):]
            )
        
        return history_text

    def build_prompt_with_history(
        self, 
        base_prompt: str, 
        question: str, 
        state: AgentState, 
        sample_idx: int,
        current_agent_name: str
    ) -> str:
        """
        Build the prompt with optional history aggregation.
        
        Args:
            base_prompt: Base prompt template
            question: Current question
            state: Current agent state
            sample_idx: Index of the sample
            current_agent_name: Name of the current agent being executed
            
        Returns:
            Complete prompt with history if aggregation is enabled
        """
        if not self.aggregate_history:
            return base_prompt.format(question=question)
        
        history = self.format_round_history(state, sample_idx)
        
        if history:
            return (
                f"{question}\n\n"
                f"{history}"
                f"Please consider the previous attempts above and provide your {current_agent_name} output."
            )
        
        return base_prompt.format(question=question)

    def _get_agent_prompts(self):
        agent_system_msgs = {}
        agent_user_msgs = {}
        
        for name, config in self.agents_config.items():
            if "sys_message" in config:
                agent_system_msgs[name] = self._load_from_module(config["sys_message"])
            if "user_message" in config:
                agent_user_msgs[name] = self._load_from_module(config["user_message"])
                
        return agent_system_msgs, agent_user_msgs

    def execute_agent(self, state: AgentState, agent_name: str) -> AgentState:
        """
        Run one agent strictly using prompts stored in state.
        Requires `agent_system_msgs[name]` and `agent_user_msgs[name]` to be present.
        """
        t0 = time.time()
        # get all the samples
        samples = state["init_input"]
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
                
                # Check if this is the first agent of a new round
                agent_names = list(self.agents_config.keys())
                is_first_agent_of_new_round = (
                    num_executed > 0 and 
                    num_executed % len(agent_names) == 0
                )
                
                if is_first_agent_of_new_round and self.aggregate_history:
                    # Build prompt with full history for new rounds
                    formatted_user_msg = self.build_prompt_with_history(
                        user_prompt, question, state, i, cur_name
                    )
                else:
                    # Use standard within-round context
                    user_prompt = user_prompt.format(
                        question=question,
                        block=item,
                    )
                    formatted_user_msg = user_prompt
            else:
                # First execution, no history
                formatted_user_msg = user_prompt.format(question=question)

            system_msg = system_msg.replace("{", "{{").replace("}", "}}")
            # Create InferInput for this sample
            infer_input = InferInput(system_msg=system_msg, user_msg=formatted_user_msg)
            infer_inputs.append(infer_input)

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
        # Store list of all responses
        state["agent_results"].append({cur_name: responses})
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
        agent_names = list(self.agents_config.keys())
        if not agent_names:
            return "end"

        # Count how many complete rounds have been executed
        # A round is complete when all agents in the sequence have executed
        rounds_completed = 0
        i = 0
        while i < len(state["agent_executed"]):
            # Check if the next sequence matches the agent_names
            match = True
            for j, agent_name in enumerate(agent_names):
                if (
                    i + j >= len(state["agent_executed"])
                    or state["agent_executed"][i + j] != agent_name
                ):
                    match = False
                    break
            if match:
                rounds_completed += 1
                i += len(agent_names)
            else:
                i += 1

        if rounds_completed < self.round:
            return agent_names[0]  # Start a new round
        return "end"  # End the workflow

    def graph(self) -> CompiledStateGraph:
        """Build and compile the sequential LangGraph: entry is the first agent, edges chain through to END.
        - Supports multiple rounds based on the round parameter
        """
        agent_names = list(self.agents_config.keys())
        g = StateGraph(AgentState)
        for name in agent_names:
            g.add_node(name, partial(self.execute_agent, agent_name=name))

        if agent_names:
            g.set_entry_point(agent_names[0])
            for i in range(len(agent_names) - 1):
                g.add_edge(agent_names[i], agent_names[i + 1])

            if self.round > 1:
                # Add conditional edge to support multiple rounds
                g.add_conditional_edges(
                    agent_names[-1],
                    self.check_round_limit,
                    {agent_names[0]: agent_names[0], "end": END},
                )
            else:
                # Single round, direct to end
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
