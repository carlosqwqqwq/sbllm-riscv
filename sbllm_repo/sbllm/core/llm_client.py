import os
import sys
import time
import logging
import random
import openai
import concurrent.futures

# Lazy import for google.generativeai to avoid deprecation warning when not using Gemini
genai = None

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_keys, provider="openai"):
        self.api_keys = api_keys
        self.provider = provider
        if provider == "deepseek":
             self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        elif provider == "codellama":
             self.base_url = os.getenv("LLAMA_BASE_URL", "https://api.deepinfra.com/v1/openai")
        elif provider == "openai":
             self.base_url = os.getenv("OPENAI_BASE_URL", None)
        else:
             self.base_url = None

    def generate(self, model_name, messages, n=1, temperature=0.7):
        """
        Unified generation method.
        Returns a list of generated strings.
        """
        if not self.api_keys:
             logger.error(f"No API keys available for provider {self.provider}")
             return []

        # Select a key (simple round-robin or random could be done outside)
        # Here we just pick a random one or use index 0. 
        # Since the original code passed api_idx via cfg, we might want to handle that.
        # But for simplification, let's just use rotation here or assume key is passed.
        # Actually, let's stick to the calling convention: the caller manages keys?
        # No, the original code had api_idx logic inside the functions.
        
        # Let's implementation simple retry logic wrapper
        retries = 0
        max_retries = 5
        
        while retries < max_retries:
            key = random.choice(self.api_keys)
            try:
                if self.provider == "gemini":
                    return self._generate_gemini(key, model_name, messages, n, temperature)
                elif self.provider == "mock":
                    # Mock: return the code from the prompt (last user message)
                    content = messages[-1]['content']
                    return [content] * n
                else:
                    return self._generate_openai_compatible(key, model_name, messages, n, temperature)
            except Exception as e:
                logger.error(f"Standard generation error ({self.provider}): {e}")
                retries += 1
                time.sleep(2 * retries)
                if 'insufficient balance' in str(e):
                    sys.exit(-1)
        
        return []

    def _generate_gemini(self, api_key, model_name, messages, n, temperature):
        global genai
        if genai is None:
            import google.generativeai as genai_module
            genai = genai_module
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": temperature,
            "candidate_count": 1, # Gemini often restricts this to 1
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        
        # Mapping standard messages format to Gemini format
        # Standard: [{"role": "user", "content": "..."}]
        # Gemini: [{"role": "user", "parts": ["..."]}]
        gemini_messages = []
        for m in messages:
            # Map system to user, and assistant to model
            role = "user" if m["role"] in ["user", "system"] else "model"
            content = m.get("content") or m.get("parts")
            if isinstance(content, list):
                content = content[0] if content else ""
            gemini_messages.append({"role": role, "parts": [content]})
            
        model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config, safety_settings=safety_settings)
        
        responses = []
        # Gemini native n=1 often. Loop to get n.
        for i in range(n):
            try:
                 resp = model.generate_content(gemini_messages)
                 if resp.text:
                     responses.append(resp.text.strip())
                 else:
                     logger.warning(f"Gemini returned empty response for candidate {i}")
            except Exception as e:
                 logger.error(f"Gemini single generation error (candidate {i}): {e}")
                 # If it blocked due to safety, we might get an error here
                 if "finish_reason" in str(e) or "safety" in str(e).lower():
                     responses.append("Blocked by Gemini safety filters.")
                 
        return responses

    def _generate_openai_compatible(self, api_key, model_name, messages, n, temperature):
        # Set a longer timeout (default is often 60s, which is short for complex code tasks)
        client = openai.OpenAI(api_key=api_key, base_url=self.base_url, timeout=120.0)
        
        # DeepSeek specific: n=1 limit workaround
        # User confirmed API does NOT support n param naturally for multiple choices
        if self.provider == "deepseek" and n > 1:
            responses = []
            
            def _call_once(idx):
                inner_retries = 0
                max_inner_retries = 3
                last_error = None
                
                while inner_retries < max_inner_retries:
                    try:
                        # Add jitter delay based on candidate index and retry count
                        wait_time = random.uniform(0.1, 0.5) * (idx + 1) + (inner_retries * 2)
                        time.sleep(wait_time)
                        
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            n=1,
                            temperature=temperature,
                            stream=False, # Ensure no stream
                            timeout=120.0
                        )
                        return resp.choices[0].message.content.strip()
                    except Exception as e:
                        last_error = e
                        inner_retries += 1
                        logger.warning(f"DeepSeek candidate {idx} retry {inner_retries}/{max_inner_retries} due to: {e}")
                        time.sleep(2 ** inner_retries) # Exponential backoff
                
                logger.error(f"DeepSeek candidate {idx} failed after {max_inner_retries} retries: {last_error}")
                return None # Mark as failed

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 5)) as executor:
                futures = [executor.submit(_call_once, i) for i in range(n)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        res = future.result()
                        if res:
                            responses.append(res)
                    except Exception as e:
                        logger.error(f"DeepSeek unexpected worker error: {e}")
            return responses
        
        # Standard OpenAI or DeepSeek with n=1
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            n=n, # For standard support or n=1
            temperature=temperature
        )
        return [choice.message.content.strip() for choice in resp.choices]

def get_client(cfg, api_keys):
    model_name_lower = str(cfg.model_name).lower()
    if 'gemini' in model_name_lower:
         return LLMClient(api_keys, provider="gemini")
    elif 'deepseek' in model_name_lower:
         return LLMClient(api_keys, provider="deepseek")
    elif 'codellama' in model_name_lower or 'llama' in model_name_lower:
         return LLMClient(api_keys, provider="codellama")
    elif 'mock' in model_name_lower:
         return LLMClient(["mock_key"], provider="mock")
    else:
         return LLMClient(api_keys, provider="openai")
