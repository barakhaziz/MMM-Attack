"""Configuration module for MMM-Attack system"""

from .prompts import PROMPTS
from .settings import *

__all__ = [
    "PROMPTS",
    "MAX_ITERATIONS",
    "ATTACK_MODEL", 
    "TARGET_MODEL",
    "JUDGE_MODEL",
    "BATCH_SIZE",
    "MAX_RETRIES"
]