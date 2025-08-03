import torch, os, warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig, GPTQConfig

try:
    from auto_gptq import AutoGPTQForCausalLM
    HAS_AUTOGPTQ = True
except Exception:
    HAS_AUTOGPTQ = False

class LocalLLMQ:
    def __init__(
        self,
        model_name="hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
        device=None,
        trust_remote_code=True,
    ):
        cuda_idx   = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        self.device = device or (f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        print(f"Loading model {model_name} on {self.device}...")

        try:
         
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

            quant_cfg = AwqConfig(bits=4, fuse_max_seq_len=4096, do_fuse=True)

         
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,      # fp16 weights + int4 scales
                device_map="auto",
                quantization_config=quant_cfg,  # Main quantization config
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            print(f"✓ Successfully loaded {model_name}")

        except Exception as e:
            warnings.warn(f"Error loading model {model_name}: {e}")
            raise

    def generate(self, prompt, max_tokens=512, temperature=0.4, do_sample=True):
        """
        Generate text from a prompt using the loaded model.
        """
        try:
            # Use the tokenizer's chat template if available for better chat model performance
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                 messages = [{"role": "user", "content": prompt}] # Assuming a simple user message format
                 # add_generation_prompt=True is crucial for some models like Llama/Mistral
                 formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                 inputs = self.tokenizer(formatted_prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
                 #print(f"[DEBUG] Using chat template. Formatted prompt (partial): {formatted_prompt[:200]}...")
            else:
                 # Fallback to simple tokenization
                 inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
                 #print(f"[DEBUG] Not using chat template. Raw prompt (partial): {prompt[:200]}...")


            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # Add other generation parameters if needed, e.g., top_k, top_p
                )
            input_token_length = inputs['input_ids'].shape[1]
            if outputs[0].shape[0] > input_token_length:
                 # Model generated new tokens after the input sequence
                 output = self.tokenizer.decode(outputs[0][input_token_length:], skip_special_tokens=True)
            else:
                 # Model returned only the input tokens or less
                 output = ""
                 print("[WARNING] Model generated no new tokens or less than input tokens.")

            return output.strip()

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return f"[Error generating response: {str(e)}]" # Return specific error format


class FastGPTQ70B:
    def __init__(self, model_name="hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
                 device="cuda:0", trust_remote_code=True):
        self.model_name = model_name
        self.device = device
        print(f"Loading {model_name} on {device} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

       
        try:
            gptq_cfg = GPTQConfig(bits=4, exllama_config={"version": 2})
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": device},         
                quantization_config=gptq_cfg,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            print("Loaded with Transformers GPTQ + ExLlamaV2")
        except Exception as e:
            warnings.warn(f"Transformers+ExLlamaV2 load failed, fallback to AutoGPTQ: {e}")
            if not HAS_AUTOGPTQ:
                raise
            self.model = AutoGPTQForCausalLM.from_quantized(
                model_name,
                device=device,
                use_safetensors=True,
                trust_remote_code=trust_remote_code,
               
            )
            print("Loaded with AutoGPTQ backend")

        self.model.eval()


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def generate(self, prompt, max_tokens=256, temperature=0.0, do_sample=False):

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt", return_attention_mask=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        inp_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][inp_len:], skip_special_tokens=True).strip()


class LocalLLM:
    # Added trust_remote_code parameter with default True
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1", device=None, trust_remote_code=True):
        # Use os.environ.get for CUDA device index if needed
        cuda_device_index = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        self.device = device or (f"cuda:{cuda_device_index}" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        print(f"Loading model {model_name} on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code) # ADD trust_remote_code here too
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if (self.device).startswith("cuda") else torch.float32,
                device_map="auto",
                load_in_4bit=True, 
                trust_remote_code=trust_remote_code # ENSURE THIS IS True
            )
            self.model.eval() # Set model to evaluation mode
            print(f"✓ Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            # It's better to raise the exception if the model can't be loaded
            raise

    def generate(self, prompt, max_tokens=512, temperature=0.4, do_sample=True):
        """
        Generate text from a prompt using the loaded model.
        """
        try:
            # Use the tokenizer's chat template if available for better chat model performance
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                 messages = [{"role": "user", "content": prompt}] # Assuming a simple user message format
                 # add_generation_prompt=True is crucial for some models like Llama/Mistral
                 formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                 inputs = self.tokenizer(formatted_prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
                 #print(f"[DEBUG] Using chat template. Formatted prompt (partial): {formatted_prompt[:200]}...")
            else:
                 # Fallback to simple tokenization
                 inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
                 #print(f"[DEBUG] Not using chat template. Raw prompt (partial): {prompt[:200]}...")


            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    # Add other generation parameters if needed, e.g., top_k, top_p
                )
            input_token_length = inputs['input_ids'].shape[1]
            if outputs[0].shape[0] > input_token_length:
                 # Model generated new tokens after the input sequence
                 output = self.tokenizer.decode(outputs[0][input_token_length:], skip_special_tokens=True)
            else:
                 # Model returned only the input tokens or less
                 output = ""
                 print("[WARNING] Model generated no new tokens or less than input tokens.")

            return output.strip()

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return f"[Error generating response: {str(e)}]" # Return specific error format