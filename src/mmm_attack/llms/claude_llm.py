import os, warnings
import anthropic                        
import time

class ClaudeChatLLM:
    """
    Mimics LocalLLM.generate() interface (prompt → raw text) for Anthropic Claude.
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-latest",  
        default_max_tokens: int = 1024,
        default_temperature: float = 0.4,
    ):
        self.model_name          = model_name
        self.default_max_tokens  = default_max_tokens
        self.default_temperature = default_temperature
        ant_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not ant_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = anthropic.Anthropic(api_key=ant_api_key)



    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        do_sample: bool = True,          
        clean_labels: bool = True,       
    ) -> str:
        retries = 3
        for attempt in range(retries):
            try:
                message = self.client.messages.create(
                    model       = self.model_name,
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = max_tokens  or self.default_max_tokens,
                    temperature = temperature if temperature is not None else self.default_temperature,
                )
                return "".join(block.text for block in message.content).strip()

            except Exception as e:
                err_str = str(e).lower()
                if 'overloaded' in err_str or 'error code: 529' in err_str:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    warnings.warn(f"Anthropic overloaded (attempt {attempt+1}/{retries}), retrying in {wait_time} sec...")
                    time.sleep(wait_time)
                else:
                    warnings.warn(f"Anthropic call failed: {e}")
                    return f"[Error generating response: {e}]"

        # If we exhausted retries
        warnings.warn(f"Anthropic call failed after {retries} attempts due to overload.")
        return "[Error generating response: Model overloaded after retries]"