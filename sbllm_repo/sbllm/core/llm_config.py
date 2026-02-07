
import os
import logging
from dotenv import load_dotenv

# Explicitly load .env from the project root in the container
env_path = os.path.join(os.getcwd(), '.env')
if not os.path.exists(env_path):
    env_path = '/work/sbllm_repo/.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

def load_keys(env_var):
    val = os.getenv(env_var, "")
    return [k.strip() for k in val.split(",") if k.strip()]

def get_api_keys(model_name):
    """
    Get the list of API keys for a specific model.
    """
    # Force reload from .env
    env_path = os.path.join(os.getcwd(), '.env')
    if not os.path.exists(env_path):
        env_path = '/work/sbllm_repo/.env'
    load_dotenv(dotenv_path=env_path, override=True)
    
    model_name_lower = str(model_name).lower()
    
    if 'mock' in model_name_lower:
        return ["mock_key"]
    
    if 'gemini' in model_name_lower:
        return load_keys("GEMINI_API_KEYS")
    if 'deepseek' in model_name_lower:
        keys = load_keys("DEEPSEEK_API_KEYS")
        if not keys:
            # Fallback to general DEEPSEEK_API_KEY
            val = os.getenv("DEEPSEEK_API_KEY", "")
            if val: keys = [val]
        return keys
    if 'codellama' in model_name_lower or 'llama' in model_name_lower:
        return load_keys("LLAMA_API_KEYS")
        
    return load_keys("OPENAI_API_KEYS")

def get_model_id(model_name):
    """
    Map friendly model name to actual model identifier.
    Prioritizes environment variables.
    """
    model_name_lower = str(model_name).lower()
    
    if 'gemini' in model_name_lower:
        return os.getenv("GEMINI_MODEL_NAME", "gemini-pro")
    if 'deepseek' in model_name_lower:
        return os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    if 'codellama' in model_name_lower or 'llama' in model_name_lower:
        return os.getenv("LLAMA_MODEL_NAME", "codellama/CodeLlama-34b-Instruct-hf")
    
    # Default fallback
    return "gpt-3.5-turbo-0613"
