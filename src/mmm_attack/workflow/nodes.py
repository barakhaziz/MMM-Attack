"""Node functions for the attack workflow graph"""

import json
from typing import Dict, Any
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

from ..agents.state import AgentState
from ..agents.output_schemas import (
    StrategyAgentOutput,
    ArticulatorOutput,
    ExplainerOutput,
    SessionSummaryOutput
)
from ..config.prompts import PROMPTS
from ..config.settings import MAX_ITERATIONS
from ..utils.memory_utils import get_strategy_memory, save_summary_to_memory
from ..utils.refusal_detection import is_hard_refusal_text_based

# Initialize parsers
strategy_parser = JsonOutputParser(pydantic_schema=StrategyAgentOutput)
strategy_format_instructions = strategy_parser.get_format_instructions()

articulator_parser = JsonOutputParser(pydantic_schema=ArticulatorOutput)
articulator_format_instructions = articulator_parser.get_format_instructions()

explainer_parser = JsonOutputParser(pydantic_schema=ExplainerOutput)
explainer_format_instructions = explainer_parser.get_format_instructions()

summary_parser = JsonOutputParser(pydantic_schema=SessionSummaryOutput)
summary_format_instructions = summary_parser.get_format_instructions()

# Initialize prompt templates
strategy_prompt_template = PromptTemplate.from_template(PROMPTS["strategy_agent"])
articulator_prompt_template = PromptTemplate.from_template(PROMPTS["articulator"])
explainer_prompt_template = PromptTemplate.from_template(PROMPTS["explainer"])
session_summary_prompt = PromptTemplate.from_template(PROMPTS["session_summary"])

def strategy_agent_node(state: AgentState, helper_llm) -> AgentState:
    # 1. Load memory for the target model
    memory = get_strategy_memory(state["model_name"])

    # 2. Bind that memory to the helper LLM
    llm_with_memory = helper_llm.with_config({"memory": memory})

    # 3. Build the runnable chain (prompt → LLM → parser)
    strategy_chain = strategy_prompt_template | llm_with_memory | strategy_parser
    
    # 4. Prepare inputs
    input_vals = {
        "malicious_goal": state["malicious_goal"],
        "model_name":     state["model_name"],
    }

    # 5. Invoke the chain
    result = strategy_chain.invoke(input_vals)

    # 6. Logging & return updated state
    print(f"[Iteration {state['iteration']}] StrategyAgent output:")
    print(result)
    print("-" * 50)

    return {
        **state,
        "strategy_agent_analysis": result
    }

def explainer_node(state: AgentState, helper_llm) -> AgentState:
    explainer_chain = explainer_prompt_template | helper_llm | explainer_parser
    
    output = explainer_chain.invoke({
        "semantic_category":state["semantic_category"],
        "malicious_goal": state["malicious_goal"],
        "target_response": state["target_response"],
        "context": state["explainer_context"],
        "strategy_agent_analysis": state["strategy_agent_analysis"]
    })
    print(f"[Iteration {state['iteration']+1}] Explainer_output:")
    print(output)
    print("-" * 50)
    proximity = output["goal_proximity_assessment"]
    refusal   = output["refusal_strength_index"]
    if proximity in ("Far", "Very Far") and refusal == 0:
        stalled = state["stalled_turns"] + 1
    else:
        stalled = 0

    return {
        **state,
        "potential_bridges_to_goal": output["potential_bridges_to_goal"],
        "refusal_analysis": output["recommended_strategy"],
        "goal_proximity": proximity,
        "stalled_turns": stalled
    }

def articulator_node(state: AgentState, helper_llm) -> AgentState:
    articulator_chain = articulator_prompt_template | helper_llm | articulator_parser
    
    output = articulator_chain.invoke({
        "turn_no": state["iteration"],
        "max_turns": MAX_ITERATIONS,
        "malicious_goal": state["malicious_goal"],
        "semantic_category":state["semantic_category"],
        "refusal_analysis": state["refusal_analysis"],
        "potential_bridges_to_goal": state["potential_bridges_to_goal"],
        "context": state["explainer_context"],
        "target_response": state["target_response"],
        "strategy_agent_analysis": state["strategy_agent_analysis"]
        
    })
    print(f"QuestionGenerator output:")
    print(output["generated_question"])
    print("-" * 50)
    return {
        **state,
        "generated_question": output["generated_question"]
    }

def target_and_judge_node(state: AgentState, target_llm, judge_llm) -> AgentState:
    new_prompt = state["generated_question"]
    context = state["context"]
    full_prompt = context + f"\nYou: {new_prompt}"
    target_resp = target_llm.generate(full_prompt)
    judge_result = judge_llm.judge(user_input=state["malicious_goal"], assistant_output=target_resp)
    
    print(f"Target model response:\n{target_resp}\n")
    print(f"Judge result: {judge_result}\n")
    success = False
    if judge_result.lower() == "unsafe":
        print("✅ Attack succeeded.")
        success = True
    context = state["context"]
    if not is_hard_refusal_text_based(target_resp):
        context += f"\nYou: {new_prompt}\nTarget: {target_resp}"
        print("➕ Added to active context")
        #print(f"\nYou: {new_prompt}\nTarget: {target_resp}")
    
    explainer_context = state["explainer_context"]
    explainer_context += f"\nYou: {new_prompt}\nTarget: {target_resp}"
    # === Trim both contexts to last 7 turns ===
    def trim_context(c):
        lines = [l for l in c.strip().split("\n") if l]
        # Each turn = 2 lines: user + target → keep last 14 lines (7 turns)
        trimmed = "\n".join(lines[-14:]) if len(lines) > 14 else "\n".join(lines)
        return trimmed

    context = trim_context(context)
    explainer_context = trim_context(explainer_context)
    print("=" * 80)

    state["attack_log"].append({
        "iteration": state["iteration"],
        "refusal_analysis": state["refusal_analysis"],
        "generated_question": new_prompt,
        "target_response": target_resp,
        "judge_result": judge_result
    })

    return {
        **state,
        "target_response": target_resp,
        "judge_result": judge_result,
        "context": context,
        "explainer_context": explainer_context,
        "iteration": state["iteration"] + 1,
        "success": success
    }

def should_finish(state: AgentState) -> str:
    if "unsafe" in str(state["judge_result"]).lower():
        return "Summarizer"
    if state["iteration"] >= MAX_ITERATIONS:
        return "Summarizer"
    return "ExplainerAgent"

def session_summary_node(state: AgentState, helper_llm) -> Dict[str, Any]:
    memory = get_strategy_memory(state["model_name"])
    llm_with_memory = helper_llm.with_config({"memory": memory})

    summarizer_chain = session_summary_prompt | llm_with_memory | summary_parser

    input_values = {
        "malicious_goal": state["malicious_goal"],
        "attack_log": state["attack_log"]
    }

    # --- invoke LLM ---
    output = summarizer_chain.invoke(input_values)

    print("[SessionSummarizer] Response:")
    print(json.dumps(output, indent=2))
    print("=" * 80)

    save_summary_to_memory(state["model_name"], output)

    return {"session_summary": output}

def summarizer_node(state: AgentState, helper_llm) -> AgentState:
    output = session_summary_node(state, helper_llm)["session_summary"]
    print("Session Summary:", output["summary"] if output["summary"].strip() else "[No update]")
    print("Attack", "SUCCESS" if state["success"] else "FAILURE")
    print("=" * 80)
    return {
        **state,
        "summary": output["summary"]
    }