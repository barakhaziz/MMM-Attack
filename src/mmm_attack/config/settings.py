"""Configuration settings for the MMM-Attack system"""

# Model configurations
ATTACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_MODEL = "meta-llama/Llama-Guard-3-8B"

# Attack parameters
MAX_ITERATIONS = 10
MAX_RETRIES = 1
BATCH_SIZE = 10

# File paths
BEHAVIOR_CSV_PATH = "harmbench_behaviors_text_all.csv"
RESULTS_DIR = "attack_results"