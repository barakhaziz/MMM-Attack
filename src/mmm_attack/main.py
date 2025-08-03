"""Main execution script for MMM-Attack system"""

import os
import pandas as pd
from typing import Dict
from dotenv import load_dotenv

from .llms import LocalLLM, JudgeLLM, RunnableLocalLLM
from .workflow import build_attack_graph
from .config.settings import (
    ATTACK_MODEL,
    TARGET_MODEL, 
    JUDGE_MODEL,
    MAX_RETRIES,
    BATCH_SIZE,
    BEHAVIOR_CSV_PATH,
    RESULTS_DIR
)

# Load environment variables
load_dotenv()

def initialize_models():
    """Initialize all required models"""
    print("Initializing models...")
    
    # Initialize attack model (helper)
    raw_helper_llm = LocalLLM(model_name=ATTACK_MODEL)
    helper_llm = RunnableLocalLLM(raw_helper_llm)
    
    # Initialize target model
    target_llm = LocalLLM(model_name=TARGET_MODEL)
    
    # Initialize judge model
    judge_llm = JudgeLLM(model_name=JUDGE_MODEL)
    
    return helper_llm, target_llm, judge_llm

def run_attack_session(behavior: str,
                       category: str,
                       built_graph,
                       max_retries: int = MAX_RETRIES) -> Dict:
    """Run a single attack session"""
    
    last_iterations, last_success = -1, False   

    for attempt in range(1, max_retries + 1):
        try:
            state = {
                "malicious_goal": behavior,
                "model_name": TARGET_MODEL,
                "context": "",
                "explainer_context": "",
                "target_response": None,
                "strategy_agent_analysis": None,
                "refusal_analysis": None,
                "generated_question": None,
                "judge_result": None,
                "attack_log": [],
                "success": False,
                "iteration": 0,
                "goal_proximity": "Very Far",
                "stalled_turns": 0,
                "semantic_category": category,
                "potential_bridges_to_goal": []
            }

            result = built_graph.invoke(state, config={"recursion_limit": 100})
            last_success   = bool(result.get("success", False))
            last_iterations = result.get("iteration", -1)

            if last_success:
                break

        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed with error: {e}")
            last_iterations = -1

    return {
        "Behavior": behavior,
        "SemanticCategory": category,
        "Success": last_success,
        "Iterations": last_iterations,
    }

def main():
    """Main execution function"""
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize models
    helper_llm, target_llm, judge_llm = initialize_models()
    
    # Build attack graph
    built_graph = build_attack_graph(helper_llm, target_llm, judge_llm)
    
    # Load behaviors dataset
    df = pd.read_csv(BEHAVIOR_CSV_PATH)
    df = df.dropna(subset=["Behavior"])
    
    # Check for existing results
    results_path = os.path.join(RESULTS_DIR, f"results_{TARGET_MODEL.replace('/', '_')}.csv")
    
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        completed_behaviors = set(results_df["Behavior"])
    else:
        results_df = pd.DataFrame(columns=["Behavior", "SemanticCategory", "Success", "Iterations"])
        completed_behaviors = set()
    
    batch_results = []
    
    # Process each behavior
    for i, row in df.iterrows():
        behavior = row["Behavior"]
        category = row.get("SemanticCategory", "Unknown")

        if behavior in completed_behaviors:
            continue  # Already processed

        print(f"\n🔁 Running behavior #{i}: {behavior[:80]} ({category})")
      
        outcome = run_attack_session(behavior, category, built_graph)

        batch_results.append(outcome)
        results_df = pd.concat([results_df, pd.DataFrame([outcome])], ignore_index=True)
        num_successes = results_df["Success"].sum()
        asr = 100 * num_successes / len(results_df)
        print(f" Current ASR: {asr:.2f}% ({int(num_successes)}/{len(results_df)})")
        
        if len(batch_results) >= BATCH_SIZE:
            results_df.to_csv(results_path, index=False)
            batch_results = []
            print(f" Saved partial results at {results_path} (after {i+1} behaviors)")
        
    # Save final results
    results_df.to_csv(results_path, index=False)
    print(f"Final results saved to {results_path}")

if __name__ == "__main__":
    main()