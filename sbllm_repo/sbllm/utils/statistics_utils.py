import logging
import yaml
import os
from .codebleu_utils import get_codebleu_score, get_detailed_codebleu

logger = logging.getLogger(__name__)


def calculate_composite_score(candidate_time: float, baseline_time: float, 
                              candidate_size: int = 0, baseline_size: int = 0,
                              alpha: float = 0.7) -> float:
    """
    Calculate composite optimization score.
    Score = alpha * SpeedupRatio + (1-alpha) * SizeReductionRatio
    
    Args:
        candidate_time: Time taken by optimized code
        baseline_time: Time taken by original code
        candidate_size: Size of optimized binary/code
        baseline_size: Size of original binary/code
        alpha: Weight for speedup (default 0.7)
    
    Returns:
        float: Composite score
    """
    if baseline_time <= 0 or candidate_time >= 99999:
        return -float('inf')
        
    # OPT = 1 - candidate_time / baseline_time (Higher is better)
    # Note: Traditional speedup is baseline/candidate. Here we use reduction ratio 
    # consistent with original code: 1 - new/old.
    # If new > old, ratio is negative (regression).
    opt_ratio = 1.0 - (candidate_time / baseline_time)
    
    # Size reduction ratio
    size_reduction = 0.0
    if baseline_size > 0:
        size_reduction = (baseline_size - candidate_size) / baseline_size
        
    return alpha * opt_ratio + (1 - alpha) * size_reduction

def write_yaml(data, file_path):
    with open(file=file_path, mode='w', encoding='utf8') as f:
        yaml.dump(data, f)

def calculate_statistics(execution_data, cfg):
    """
    Calculate and print statistics based on execution data.
    """
    results = []
    references = []
    hypothesis = []
    ptr = 0
    correct = 0
    faster_count = 0
    unique_count = 0
    input_time_sum = 0
    generated_test_sum = 0
    unique_reference_time_sum = 0
    unique_generated_test_sum = 0
    codebleu_results = []

    cb_pairs = []
    processed_items = []
    
    for i in execution_data:
        acc = i.get('model_generated_potentially_faster_code_col_acc', 0)
        input_time = i.get('input_time_mean', 0)
        generated_time = i.get('model_generated_potentially_faster_code_col_time_mean', input_time)
        reference_time = i.get('reference_time_mean', input_time)
        
        if input_time is None or reference_time is None:
            continue
        
        if generated_time is None:
            generated_time = input_time
            
        results.append([generated_time, input_time, acc])
        processed_items.append(i)
        
        # Prepare for Batch CodeBLEU
        ref_code = i.get('code_v1_no_empty_lines') or i.get('code_v0_no_empty_lines', '')
        hypo_code = i.get('model_generated_potentially_faster_code_col', '') or ''
        cb_pairs.append((ref_code, hypo_code))
        
        if acc==1:
            correct+=1

    # Batch Calculate CodeBLEU in Docker (Guarantees valid tree-sitter environment)
    from .codebleu_utils import batch_get_codebleu_docker
    project_root = getattr(cfg, 'project_root', os.getcwd())
    cb_results = batch_get_codebleu_docker(cb_pairs, lang=cfg.lang, project_root=project_root)
    
    # Map results back to items
    for i, cb_res in zip(processed_items, cb_results):
        if cb_res:
            codebleu_results.append(cb_res)
            i['codebleu'] = cb_res['codebleu']
            i['codebleu_detail'] = cb_res
        
        if acc==1 and generated_time < input_time and generated_time > 0:
            if generated_time < reference_time:
                unique_count += 1
                unique_reference_time_sum += reference_time
                unique_generated_test_sum += generated_time
            
            if generated_time < input_time * 0.9:
                faster_count += 1
            
            input_time_sum += input_time
            generated_test_sum += generated_time
            ptr += input_time/generated_time - 1
        else:
            input_time_sum += input_time
            generated_test_sum += input_time
            ptr += 0
            
    print(cfg.mode)
    if len(results) > 0:
        print('OPT(%): ', round(100*faster_count/len(results), 2))
        print('SP: ', round(100*ptr/len(results), 2))
        if codebleu_results:
            avg_cb = sum(r['codebleu'] for r in codebleu_results) / len(codebleu_results)
            avg_ngram = sum(r['ngram'] for r in codebleu_results) / len(codebleu_results)
            avg_weighted = sum(r['weighted_ngram'] for r in codebleu_results) / len(codebleu_results)
            avg_syntax = sum(r['syntax'] for r in codebleu_results) / len(codebleu_results)
            avg_dataflow = sum(r['dataflow'] for r in codebleu_results) / len(codebleu_results)
            
            print('--- CodeBLEU Metrics ---')
            print('CodeBLEU (Total): ', round(avg_cb, 4))
            print('  - N-gram Match: ', round(avg_ngram, 4))
            print('  - Weighted N-gram: ', round(avg_weighted, 4))
            print('  - Syntax (AST): ', round(avg_syntax, 4))
            print('  - Semantic (DFG): ', round(avg_dataflow, 4))
            print('------------------------')
    else:
        print('No results to evaluate')
