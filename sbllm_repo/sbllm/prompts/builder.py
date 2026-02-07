import logging
from .templates import (
    format, 
    format_pattern, 
    format_ablation, 
    format_ablation_cpp, 
    format_cpp, 
    format_pattern_cpp, 
    _PROMPT_FOR_REVIEW, 
    _PROMPT_FOR_CANDIDATES, 
    _PROMPT_FOR_RISCV_EVOLUTION
)

logger = logging.getLogger(__name__)

def _build_performance_history(best_candidates, beam_number):
    """
    Constructs a history string from previous best candidates.
    """
    previous_performance = ''
    candidates_to_use = best_candidates[:beam_number] if best_candidates else []
    
    for idx, opt in enumerate(candidates_to_use):
        try:
            input_time = opt.get('input_time', 0)
            time_val = opt.get('time', 0)
            acc = opt.get('acc', 0)
            content = opt.get('content', '').strip()
            
            # OPT = 1 - new_time / old_time
            opt_ratio = 1 - time_val / input_time if input_time > 0 else 0
            
            if acc == 1 and opt_ratio > 0:
                previous_performance += 'A correct and optimized version {}:'.format(str(idx+1))
            elif acc == 1 and opt_ratio <= 0:
                previous_performance += 'A correct but unoptimized version {}:'.format(str(idx+1))
            elif acc < 1:
                previous_performance += 'An incorrect optimized version {}:'.format(str(idx+1))
            
            error_msg = opt.get('error', '')
            if acc < 1 and error_msg:
                 # Truncate error to avoid context explosion
                 previous_performance += f"\nError Feedback: {error_msg[:300]}...\n"
            
            previous_performance += '\n{}\nAccuracy: {} OPT: {}\n'.format(content, str(acc), str(round(opt_ratio, 2)))
        except Exception as e:
            logger.warning(f"Error building history for candidate {idx}: {e}")
            continue
            
    return previous_performance

def build_evolution_prompt(cfg, query, previous_result_obj):
    """
    Builds the chat messages list for the evolutionary iteration (iteration > 0).
    Args:
        cfg: Configuration object.
        query: Current query object (dict).
        previous_result_obj: Result object from the previous iteration associated with this query.
    Returns:
        List of message dicts (e.g. [{"role": "user", "content": ...}])
    """
    prompt_messages = []
    
    # Reconstruction of best_candidates from flattened exec_item structure if missing
    best_candidates = previous_result_obj.get('best_candidates', [])
    if not best_candidates:
        for i in range(5):
             code_key = f'model_generated_potentially_faster_code_col_{i}'
             if code_key in previous_result_obj:
                 cand = {
                     'content': previous_result_obj[code_key],
                     'acc': previous_result_obj.get(f'{code_key}_acc', 0),
                     'time': previous_result_obj.get(f'{code_key}_time_mean', 0),
                     'input_time': previous_result_obj.get('input_time_mean', 0),
                     'error': previous_result_obj.get(f'{code_key}_error', '')
                 }
                 best_candidates.append(cand)

    # Generate history string
    previous_performance = _build_performance_history(
        best_candidates, 
        cfg.beam_number
    )
    
    is_riscv_mode = hasattr(cfg, 'riscv_mode') and cfg.riscv_mode or cfg.lang == 'riscv' or (cfg.mode and 'riscv' in cfg.mode.lower())
    patterns = previous_result_obj.get('pattern', [])
    
    if is_riscv_mode and patterns and len(patterns) > 0:
        # RISC-V Mode with Patterns -> Genetic Evolution
        similar_pattern = patterns[0] if len(patterns) > 0 else ""
        different_pattern = patterns[1] if len(patterns) > 1 else ""
        optimization_mode = f"Similar Pattern:\n{similar_pattern}\n\nDifferent Pattern:\n{different_pattern}"
        
        # Get current code samples for the prompt
        current_code_samples = []
        # Use local best_candidates which was populated or verified
        for idx in range(min(len(best_candidates), 3)):
            current_code_samples.append(best_candidates[idx].get('content', ''))
        
        current_code_str = "\n\n".join([f"<Sample{i+1}>\n{sample}" for i, sample in enumerate(current_code_samples)])
        
        riscv_prompt = _PROMPT_FOR_RISCV_EVOLUTION.format(
            original_code=query['query'],
            current_code=current_code_str,
            optimization_mode=optimization_mode
        )
        
        prompt_messages = [{"role": "user", "content": riscv_prompt}]
        
    elif patterns and len(patterns) > 0:
        # Standard Mode with Patterns
        patterns_str = 'Pattern 1:\n{}\nPattern 2:\n{}\n'.format(patterns[0], patterns[1] if len(patterns)>1 else "")
        
        if cfg.lang == 'python':
            content = format_pattern + "The code you need to optimize:\n```python\n{}\n```\nSome existing versions with their performance:\n{}\nCode transformation pattern:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance, patterns_str)
        else:
            content = format_pattern_cpp + "The code you need to optimize:\n```cpp\n{}\n```\nSome existing versions with their performance:\n{}\nCode transformation pattern:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance, patterns_str)
            
        prompt_messages = [{"role": "user", "content": content}]
        
    else:
        # No Patterns (Standard or First Evolution Step without patterns)
        if is_riscv_mode:
            riscv_prompt = _PROMPT_FOR_CANDIDATES.format(
                original_code=query['query'],
                review_comments=previous_performance if previous_performance else "No prior optimization history.",
                n_candidates=cfg.generation_number if hasattr(cfg, 'generation_number') else 5,
                examples=query.get('examples', '')
            )
            prompt_messages = [{"role": "user", "content": riscv_prompt}]
        elif cfg.lang == 'python':
            content = format + "The code you need to optimize:\n```python\n{}\n```\nSome existing versions with their performance:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance)
            prompt_messages = [{"role": "user", "content": content}]
        else:
            content = format_cpp + "The code you need to optimize:\n```cpp\n{}\n```\nSome existing versions with their performance:\n{}\nPlease follow the above instructions and output format specification step-by-step to generate a better program.\n".format(query['query'], previous_performance)
            prompt_messages = [{"role": "user", "content": content}]
            
    return prompt_messages

def build_initial_generation_prompt(cfg, original_code, review_comments, examples=""):
    """
    Builds the prompt for the initial generation (Iteration 1/0) where we rely on review comments.
    """
    n_candidates = getattr(cfg, 'generation_number', 5)
    
    candidate_prompt = _PROMPT_FOR_CANDIDATES.format(
        original_code=original_code,
        review_comments=review_comments,
        n_candidates=n_candidates,
        examples=examples
    )
    
    if cfg.model_name == 'gemini':
         # Gemini typically expects "parts" in some raw API, but if using standard wrapper, content is usually safe.
         # Keeping consistent with original logic just in case the client handles it differently.
         # But LLMClient.generate usually takes standard list. 
         # Let's use standard content for now unless we know LLMClient treats them differently.
         # Original code had: {"role": "user", "parts": candidate_prompt} for gemini
         return [{"role": "user", "parts": candidate_prompt}]
    else:
         return [{"role": "user", "content": candidate_prompt}]

def build_review_prompt(cfg, code):
    """
    Builds the prompt for generating optimization suggestions.
    """
    prompt_text = _PROMPT_FOR_REVIEW.format(code=code)
    return [{"role": "user", "content": prompt_text}]
