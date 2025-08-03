from langchain_core.runnables import Runnable
from langchain_core.prompts.base import PromptValue
from typing import Union, Dict, Any

class RunnableLocalLLM(Runnable):
    def __init__(self, local_llm):
        self.local_llm = local_llm

    def invoke(self, input: Union[PromptValue, Dict[str, Any], str], config: Dict = None) -> str:
        if isinstance(input, PromptValue):
            prompt = input.to_string()
        elif isinstance(input, dict):
            prompt = input.get("input") or input.get("malicious_goal") or str(input)
        elif isinstance(input, str):
            prompt = input
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        return self.local_llm.generate(prompt)