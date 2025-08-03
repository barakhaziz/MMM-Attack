from typing import TypedDict, List

class StrategyAgentOutput(TypedDict):
    high_level_strategy: str
    reasoning: str

class ArticulatorOutput(TypedDict):
    generated_question: str

class ExplainerOutput(TypedDict):
    refusal_strength_index: int                      
    tone: str
    potential_bridges_to_goal: List[str]
    goal_proximity_assessment: str       
    recommended_strategy: str

class SessionSummaryOutput(TypedDict):
    summary: str
    positive_lessons: List[str]   
    negative_lessons: List[str]