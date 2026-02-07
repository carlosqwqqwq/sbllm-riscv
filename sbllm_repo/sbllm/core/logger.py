import logging
import os
import sys

def setup_app_logging(cfg):
    """
    Configure global logging to file and stdout.
    
    Args:
        cfg: Configuration object containing output_path and mode.
    """
    try:
        # Determine output directory
        # Priority: Env Var > Config > Fallback
        env_log_file = os.getenv("SBLLM_LOG_FILE")
        if env_log_file:
            log_file = env_log_file
            # Ensure dir exists
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        elif hasattr(cfg, 'output_path') and hasattr(cfg, 'mode'):
            output_dir = os.path.join(cfg.output_path, cfg.mode)
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "sbllm.log")
        else:
            # Fallback
            output_dir = os.getcwd()
            log_file = os.path.join(output_dir, "sbllm_fallback.log")

        # Determine log level (default to INFO, allow override)
        log_level_str = os.getenv("SBLLM_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True # Reset any existing config
        )
        
        # Silence some verbose loggers if needed
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        logging.getLogger(__name__).info(f"Logging initialized. Log file: {log_file}")
        
    except Exception as e:
        print(f"Failed to setup logging: {e}")
