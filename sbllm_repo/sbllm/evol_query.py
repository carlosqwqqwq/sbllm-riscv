import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
try:
    import fcntl
except ImportError:
    fcntl = None # Not available on Windows
import random
import logging
import jsonlines
import numpy as np
from tqdm import tqdm
from time import sleep
from copy import deepcopy
import multiprocessing
from dotenv import load_dotenv

from sbllm.core.llm_client import LLMClient, get_client
from sbllm.utils.code_utils import validate_c_syntax_heuristic, remove_comments_and_docstrings, extract_code_from_markdown
from sbllm.utils.text_utils import clean_text
from sbllm.utils.retrieval_utils import load_knowledge_base, retrieve_examples_difflib

# Global Knowledge Base (will be populated by retrieval_utils)
KNOWLEDGE_BASE = []

# New refactored modules
from sbllm.core.llm_config import get_api_keys, get_model_id
from sbllm.prompts.builder import build_evolution_prompt, build_initial_generation_prompt, build_review_prompt

load_dotenv()

logger = logging.getLogger(__name__)






def _process_single_review_comment(args):
    code, cfg = args
    return generate_review_comments(code, cfg)


def generate_review_comments(code: str, cfg, model_name: str = None) -> str:
    """
    Generate RISC-V optimization suggestions (review comments)
    """
    if model_name is None:
        model_name = cfg.model_name
    
    current_keys = get_api_keys(model_name)
    if not current_keys:
        logger.error(f"No API keys available for model {model_name}")
        return ""
    
    client = get_client(cfg, current_keys)
    messages = build_review_prompt(cfg, code)
    model_id = get_model_id(model_name)
    
    results = client.generate(model_id, messages, n=1, temperature=0.7)
    return results[0] if results else ""


def prompt_construction(cfg, queries):
    results = []
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, 'results.jsonl')) as f:
        for obj in f:
            results.append(obj)

    prompt = []
    ids = set()
    
    # Store results in a dict for robust ID matching
    results_map = {}
    for r in results:
        rid = r.get('query_idx')
        if rid is None:
            rid = r.get('idx')
        if rid is None:
            rid = r.get('id', 'unknown')
        results_map[str(rid)] = r

    for query in queries:
        # Use .get() to support both 'idx' and 'id'
        q_idx = str(query.get('idx') if query.get('idx') is not None else query.get('id', 'unknown'))
        if q_idx not in results_map:
            continue
            
        pperf = results_map[q_idx]
        if pperf.get('stop', 0) >= 1:
            continue

        ids.add(q_idx)
        
        # Delegate prompt building to the new module
        if hasattr(cfg, 'riscv_mode') and cfg.riscv_mode:
            query['examples'] = retrieve_examples_difflib(query['query'], KNOWLEDGE_BASE, k=3)
            
        query['prompt_chat'] = build_evolution_prompt(cfg, query, pperf)
        prompt.append(query)
    
    return ids, prompt


def generic_llm_worker(chunk_data):
    cfg = chunk_data['cfg']
    queries = chunk_data['queries']
    model_name = cfg.model_name
    
    current_keys = get_api_keys(model_name)
    
    if not current_keys:
        logger.error(f"No API keys available for model {model_name}")
        return
    
    client = get_client(cfg, current_keys)
    model_id = get_model_id(model_name)
    
    multi_fail_count = 0
    # Process queries sequentially within worker (quietly)
    for pos in range(len(queries)):
        query = queries[pos]
        query['prediction'] = []
        query['detailed_prediction'] = []
        
        # System prompt
        messages = [
            {"role": "system", "content": "You are a software developer and now you will help to improve code efficiency. Please follow the instructions and output format specification to generate a more efficient code. The improved code should be in code blocks (```{} ```).".format(cfg.lang)},
        ]
        messages.extend(query['prompt_chat'])
        
        # --- Enhanced Logging (Pre-Generation) ---
        log_file = os.environ.get('LLM_INTERACTION_LOG', '/app/llm_interaction.log')
        try:
            with open(log_file, 'a', encoding='utf-8') as lf:
                # Force Beijing Time (UTC+8)
                utc_now = datetime.now(timezone.utc)
                beijing_now = utc_now + timedelta(hours=8)
                timestamp = beijing_now.strftime("%Y-%m-%d %H:%M:%S")
                lf.write(f"\n{'='*80}\n")
                lf.write(f"[{timestamp}] Iteration: {cfg.iteration} | Worker: {cfg.api_idx} | Query Idx: {query.get('idx', 'Unknown')}\n")
                lf.write(f"{'='*80}\n")
                lf.write(">>> PROMPT MESSAGES:\n")
                for msg in messages:
                    lf.write(f"[{msg['role'].upper()}]:\n{msg['content']}\n{'-'*40}\n")
        except Exception as log_err:
            logger.debug(f"Pre-logging failed: {log_err}")

        try:
            n = cfg.generation_number if cfg.iteration > 0 else 5 # Default to 5 candidates
            results = client.generate(model_id, messages, n=n, temperature=0.7)
            
            # --- Enhanced Logging (Post-Generation) ---
            try:
                with open(log_file, 'a', encoding='utf-8') as lf:
                    lf.write("\n<<< GENERATED CANDIDATES:\n")
                    for i, res in enumerate(results):
                        lf.write(f"--- Candidate {i} ---\n{res}\n")
                    lf.write(f"{'='*80}\n\n")
            except Exception as log_err:
                logger.debug(f"Post-logging failed: {log_err}")
            
            query['prediction'].extend(results)
            query['detailed_prediction'].extend([{"message": {"content": r}} for r in results])
            
            if len(query['prediction']) >= 1:
                if 'prompt_chat' in query:
                    del query['prompt_chat']
                with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(cfg.api_idx),str(len(current_keys)), 'test')), "a") as f:
                    f.write_all([query])
                
                # Dynamic sleep based on provider
                if cfg.model_name == 'gemini':
                    sleep(random.uniform(1, 3))
                else:
                    sleep(0.02)
        except Exception as e:
            # Log API error to interaction log
            try:
                with open(log_file, 'a', encoding='utf-8') as lf:
                     lf.write(f"\n!!! API ERROR !!!\n{str(e)}\n{'='*80}\n\n")
            except: pass
            
            logger.error(f"Worker error at position {pos}: {e}")
            multi_fail_count += 1
            if multi_fail_count > 10:
                break
    
    return len(queries)


def read_file(cfg):
    queries = []
    exist_queries=set()
    with jsonlines.open(cfg.baseline_data_path) as reader:
        for obj in reader:
            queries.append(obj)
    queries = queries[int(len(queries)*(cfg.slice-1)/cfg.total):int(len(queries)*cfg.slice/cfg.total)]

    # --- Unified Data Adapter ---
    # Normalize data schema for ALL iterations to prevent KeyError in prompt_construction
    for query in queries:
        # 1. Normalize ID
        rid = query.get('idx')
        if rid is None:
            rid = query.get('id', 'unknown')
        query['idx'] = rid # Ensure 'idx' exists
        
        # 2. Normalize Query Content
        if not query.get('query'):
            # Fallback for baseline datasets using 'code_v0_no_empty_lines'
            query['query'] = query.get('code_v0_no_empty_lines', '')

    if cfg.iteration>0:
        ids, queries = prompt_construction(cfg, queries)
    else:
        # First Iteration: Review + Candidate Generation
        ids = [str(q.get('idx') if q.get('idx') is not None else q.get('id', 'unknown')) for q in queries]
        is_riscv_mode = hasattr(cfg, 'riscv_mode') and cfg.riscv_mode or cfg.lang == 'riscv' or (cfg.mode and 'riscv' in cfg.mode.lower())
        
        if is_riscv_mode:
            logger.info('Generating RISC-V optimization suggestions...')
            pool_args = []
            model_keys = get_api_keys(cfg.model_name)
            
            for query in queries:
                 task_cfg = deepcopy(cfg)
                 task_cfg.api_idx = (int(query.get('idx', 0)) % max(1, len(model_keys)))
                 original_code = query.get('query', '') or query.get('code_v0_no_empty_lines', '')
                 if not query.get('query'):
                     query['query'] = original_code
                 
                 pool_args.append((original_code, task_cfg))
            
            logger.info(f'Generating {len(queries)} reviews in parallel...')
            review_comments_list = [None] * len(queries)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=getattr(cfg, 'process_number', 8)) as executor:
                future_to_idx = {executor.submit(_process_single_review_comment, arg): i for i, arg in enumerate(pool_args)}
                for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc="Generating Reviews"):
                    idx = future_to_idx[future]
                    try:
                        review_comments_list[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to generate review for item {idx}: {e}")
                        review_comments_list[idx] = ""
            
            for query, review_comments in zip(queries, review_comments_list):
                 original_code = query.get('query', '') or query.get('code_v0_no_empty_lines', '')
                 if not review_comments or len(review_comments.strip()) == 0:
                     review_comments = "Please optimize the RISC-V code for better performance, focusing on instruction selection, loop optimization, and leveraging RISC-V-specific features."
                 
                 # VECINTRIN SPECIFIC CONSTRAINT: Force std::vector<T> signature preservation
                 if 'vecintrin' in cfg.baseline_data_path.lower() or 'cpp' in cfg.lang:
                     review_comments += """
CRITICAL TYPE CONSTRAINTS:
1. SIGNATURE: For functions with std::vector<T> parameters, you MUST preserve the EXACT type.
   - DO NOT change `vector<float*> data_vec` to `float** data_vec` or `float* data_vec[]`
   - The std::vector type is REQUIRED for ABI compatibility.

2. RVV INTRINSIC TYPES:
   - When using reduction (e.g., __riscv_vfredusum_vs_f32m8_f32m1), the return type matches the LAST suffix (e.g., f32m1 -> vfloat32m1_t).
   - ERROR PATTERN: `vfloat32m8_t res = __riscv_vfredusum_vs_f32m8_f32m1(...)` is INVALID. Result is m1, not m8.

Example - CORRECT:
void eltwise(vector<float*> data_vec, float* out_data, ...)
"""
                 
                 # Use prompt builder
                 examples = ""
                 if is_riscv_mode:
                     examples = retrieve_examples_difflib(original_code, KNOWLEDGE_BASE, k=3)
                 
                 query['prompt_chat'] = build_initial_generation_prompt(cfg, original_code, review_comments, examples)
        
    logger.info('Iteration {} processing {} queries'.format(cfg.iteration, len(queries)))

    # Consolidation logic for previous partial runs
    unsort_queries = []
    current_keys = get_api_keys(cfg.model_name)
    
    for num in range(max(1, len(current_keys))):
        pred_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(current_keys)),'test'))
        if os.path.exists(pred_path):
            with jsonlines.open(pred_path) as f:
                for obj in f:
                    unsort_queries.append(obj)
        if len(unsort_queries)>0:
            with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test')), 'w') as f:
                f.write_all(unsort_queries)

    if os.path.exists(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))):
        with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))) as f:
            for obj in f:
                exist_queries.add(str(obj.get('idx') if obj.get('idx') is not None else obj.get('id', 'unknown')))
        new_queries = []
        for query in queries:
            q_idx = str(query.get('idx') if query.get('idx') is not None else query.get('id', 'unknown'))
            if q_idx not in exist_queries:
                new_queries.append(query)
        queries = new_queries
        
    return ids, queries, len(exist_queries)


def post_process(cfg, ids):
    data = []
    prediction_file = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))
    
    if not os.path.exists(prediction_file):
        return

    with jsonlines.open(prediction_file) as f:
        for obj in f:
            data.append(obj)
    
    processed_data = []
    for obj in data:
        obj_idx = str(obj.get('idx') if obj.get('idx') is not None else obj.get('id', 'unknown'))
        if obj_idx not in ids:
            continue
        
        post_prediction = []
        detailed_prediction = []
        lang_hint = 'python' if cfg.lang == 'python' else 'c'
        
        for res in obj['prediction']:
            detailed_prediction.append(res)
            extracted = extract_code_from_markdown(res, lang=lang_hint)
            if extracted and extracted.strip():
                # Validate C/RISC-V code
                if lang_hint != 'python' and not validate_c_syntax_heuristic(extracted):
                    continue
                post_prediction.append(extracted)
            
        obj['prediction'] = post_prediction
        if 'detailed_prediction' not in obj:
            obj['detailed_prediction'] = detailed_prediction
        processed_data.append(obj)
    
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'tmp_queries_{}test.jsonl'.format(cfg.model_name)), 'w') as f:
        f.write_all(processed_data)
    
    print(f"Post-processed {len(processed_data)} queries")
    
    # Cleanup old files
    current_keys = get_api_keys(cfg.model_name)
    
    for num in range(max(1, len(current_keys))):
        pred_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(current_keys)),'test'))
        if os.path.exists(pred_path):
            os.remove(pred_path)

from sbllm.core.arg_parser import cfg_parsing

if __name__ == "__main__":
    cfg = cfg_parsing()
    
    # Initialize Knowledge Base
    if hasattr(cfg, 'riscv_mode') and cfg.riscv_mode:
        KNOWLEDGE_BASE = load_knowledge_base()

    if not os.path.exists(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration))):
        os.makedirs(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration)))
    fail_count = 0
    
    while True:
        ids, queries, previous_length = read_file(cfg)
        print(f"Previous results: {previous_length}, New queries: {len(queries)}")
        
        if len(queries) == 0:
            print("No new queries to process. Running post-process...")
            post_process(cfg, ids)
            # Ensure tmp_queries exists even if empty to prevent warnings in evaluate.py
            tmp_queries_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'tmp_queries_{}test.jsonl'.format(cfg.model_name))
            if not os.path.exists(tmp_queries_path):
                with jsonlines.open(tmp_queries_path, 'w') as f:
                    pass
            sys.exit(0)

        active_keys = get_api_keys(cfg.model_name)
        num_processes = getattr(cfg, 'process_number', len(active_keys))
        if num_processes <= 0:
            num_processes = max(1, len(active_keys))
        
        if len(queries) < num_processes:
            num_processes = max(1, len(queries))

        pool = multiprocessing.Pool(num_processes)
        chunks_query = np.array_split(queries, num_processes)
        chunks_data = []
        print('Batch Fail Count: {}, Queries: {}'.format(str(fail_count), str(len(queries))))
        for i in range(num_processes):
            tmp_data={}
            tmp_data['cfg'] = deepcopy(cfg)
            if len(active_keys) > 0:
                tmp_data['cfg'].api_idx = i % len(active_keys)
            else:
                tmp_data['cfg'].api_idx = 0
            
            tmp_data['queries'] = chunks_query[i]
            chunks_data.append(tmp_data)
        
        # Use imap_unordered to show a single global progress bar
        with tqdm(total=len(queries), desc="LLM Code Generation") as pbar:
            for count in pool.imap_unordered(generic_llm_worker, chunks_data):
                pbar.update(count)
        
        pool.close()
        pool.join()

        data = []
        if os.path.exists(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))):
            with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))) as f:
                for i in f:
                    data.append(i)
        
        if len(data) != len(queries)+previous_length and fail_count>5:
            print(f"Warning: Generation incomplete ({len(data)}/{len(queries)+previous_length}), but saving partial results...")
            post_process(cfg, ids)
            sys.exit(0)
        elif len(data) != len(queries)+previous_length:
            fail_count+=1
        else:
            post_process(cfg, ids)
            sys.exit(0)



