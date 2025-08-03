from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    malicious_goal: str
    model_name: str
    semantic_category: str
    context: str
    explainer_context: str
    target_response: Optional[str]
    strategy_agent_analysis: Optional[dict]
    refusal_analysis: Optional[dict]
    generated_question: Optional[str]
    judge_result: Optional[str]
    attack_log: list
    success: Optional[bool]
    iteration: int
    stalled_turns: int           # consecutive turns with Far/Very Far & no refusal
    goal_proximity: str          # "Very Far" | "Far" | "Moderate" | "Close" | "Very Close"
    potential_bridges_to_goal: list