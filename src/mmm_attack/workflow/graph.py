"""Graph construction for the attack workflow"""

from langgraph.graph import StateGraph, END

from ..agents.state import AgentState
from .nodes import (
    strategy_agent_node,
    explainer_node,
    articulator_node,
    target_and_judge_node,
    summarizer_node,
    should_finish
)

def build_attack_graph(helper_llm, target_llm, judge_llm):
    """Build and compile the attack workflow graph"""
    
    # Create wrapper functions that include the LLM dependencies
    def strategy_node_wrapper(state: AgentState) -> AgentState:
        return strategy_agent_node(state, helper_llm)
    
    def explainer_node_wrapper(state: AgentState) -> AgentState:
        return explainer_node(state, helper_llm)
    
    def articulator_node_wrapper(state: AgentState) -> AgentState:
        return articulator_node(state, helper_llm)
    
    def target_judge_node_wrapper(state: AgentState) -> AgentState:
        return target_and_judge_node(state, target_llm, judge_llm)
    
    def summarizer_node_wrapper(state: AgentState) -> AgentState:
        return summarizer_node(state, helper_llm)
    
    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("StrategyAgent", strategy_node_wrapper)
    graph.add_node("ExplainerAgent", explainer_node_wrapper)
    graph.add_node("ArticulatorAgent", articulator_node_wrapper)
    graph.add_node("TargetJudge", target_judge_node_wrapper)
    graph.add_node("Summarizer", summarizer_node_wrapper)

    graph.set_entry_point("StrategyAgent")
    graph.add_edge("StrategyAgent", "ExplainerAgent")
    graph.add_edge("ExplainerAgent", "ArticulatorAgent")
    graph.add_edge("ArticulatorAgent", "TargetJudge")
    graph.add_conditional_edges("TargetJudge", should_finish, {
        "Summarizer": "Summarizer",
        "ExplainerAgent": "ExplainerAgent"
    })
    graph.set_finish_point("Summarizer")

    return graph.compile()