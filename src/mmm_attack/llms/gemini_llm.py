import os, time, warnings
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, TooManyRequests 

class GeminiChatLLM:
    """
    Gemini-compatible class matching OpenAIChatLLM interface.
    Retries automatically on 429 / rate-limit up to 5 attempts (exponential backoff).
    """

    def __init__(
        self,
        model_name: str = "gemini-1.0-pro-vision-latest",      
        default_max_tokens: int = 1024,
        default_temperature: float = 0.4,
        max_retries: int = 5,               
        base_backoff: int = 2,              
    ):
        self.model_name          = model_name
        self.default_max_tokens  = default_max_tokens
        self.default_temperature = default_temperature
        self.max_retries         = max_retries
        self.base_backoff        = base_backoff

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)

    # ---------------------------------------------------------------------

    def _send_request(self, prompt: str, max_tokens: int, temperature: float):
        return self.model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )

    # ---------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        do_sample: bool = True,    
        clean_labels: bool = True, 
    ) -> str:
        max_tokens   = max_tokens   or self.default_max_tokens
        temperature  = temperature  if temperature is not None else self.default_temperature

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._send_request(prompt, max_tokens, temperature)
                return response.text.strip()
            except (ResourceExhausted, TooManyRequests) as e:
                if attempt == self.max_retries:
                    warnings.warn(f"Gemini rate-limit after {attempt} attempts: {e}")
                    return f"[Error: rate-limit – {e}]"

                backoff = self.base_backoff * attempt   # 2s, 4s, 6s, …
                time.sleep(backoff)
            except Exception as e:
                warnings.warn(f"Gemini call failed: {e}")
                return f"[Error generating response: {e}]"