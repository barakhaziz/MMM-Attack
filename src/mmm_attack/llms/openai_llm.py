import os, warnings
import openai
from typing import Optional

class OpenAIChatLLM:
    """
    Mimics LocalLLM.generate() so existing code continues to work.
    Only 'prompt' → returns raw assistant text.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        default_max_tokens: int = 512,
        default_temperature: float = 0.4,
    ):
        self.model_name         = model_name
        self.default_max_tokens = default_max_tokens
        self.default_temperature= default_temperature
        
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=openai_api_key)

    def generate(
        self,
        prompt: str,
        max_tokens:  int   = 512,
        temperature: float = 0.4,
        do_sample:   bool  = True,  
        clean_labels: bool = True,   
    ) -> str:
        try:
            resp = self.client.chat.completions.create(
                model       = self.model_name,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = max_tokens or self.default_max_tokens,
                temperature = temperature if temperature is not None else self.default_temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            warnings.warn(f"OpenAI call failed: {e}")
            return f"[Error generating response: {e}]"