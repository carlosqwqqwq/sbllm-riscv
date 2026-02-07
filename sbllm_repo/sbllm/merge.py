import os
import re
import ast
import random
import argparse
import jsonlines
from tqdm import tqdm
import difflib
import numpy as np
import editdistance
import multiprocessing
from collections import Counter
from rank_bm25 import BM25Okapi
import builtins
from .utils.code_utils import remove_comments_and_docstrings, extract_cpp, extract_py

# New modules
from .utils.code_abstractor import abstract_py_code, abstract_cpp_code, tokenize_py_code, tokenize_cpp_code
from .utils.retrieval_utils import load_or_build_vector_index, perform_hybrid_search

# Lazy imports for optional dependencies
# import faiss
# from sentence_transformers import SentenceTransformer

# ====
# Global variables for multiprocessing
# ====
_GLOBAL_EMBEDDING_MODEL = None
_GLOBAL_VECTOR_INDEX = None
_GLOBAL_TRAIN_DATA = None
_GLOBAL_RETRIEVAL_METHOD = 'bm25'
_GLOBAL_HYBRID_ALPHA = 0.5

def init_global_resources(embedding_model, vector_index, train_data, retrieval_method, hybrid_alpha):
    """Initialize global resources for children processes"""
    global _GLOBAL_EMBEDDING_MODEL, _GLOBAL_VECTOR_INDEX, _GLOBAL_TRAIN_DATA
    global _GLOBAL_RETRIEVAL_METHOD, _GLOBAL_HYBRID_ALPHA
    _GLOBAL_EMBEDDING_MODEL = embedding_model
    _GLOBAL_VECTOR_INDEX = vector_index
    _GLOBAL_TRAIN_DATA = train_data
    _GLOBAL_RETRIEVAL_METHOD = retrieval_method
    _GLOBAL_HYBRID_ALPHA = hybrid_alpha

def _pool_initializer(embedding_model, vector_index, train_data, retrieval_method, hybrid_alpha):
    """Pool initializer"""
    init_global_resources(embedding_model, vector_index, train_data, retrieval_method, hybrid_alpha)

def better(obj1, obj2):
    if obj2['acc'] == 0 and obj2['time']==99999:
        return True
    elif obj1['acc'] == 1 and obj1['input_time']/obj1['time'] > obj2['input_time']/obj2['time']:
        return True
    else:
        return False

def extract_code(code, content, lang='python'):
    """Extract optimized code from LLM response for any language."""
    markers = [
        'provide a new optimized code snippet:**',
        'provide a new optimized code snippet:',
        'provide a new optimized code snippet'
    ]
    for marker in markers:
        if content.count(marker) == 1:
            return content.split(marker.rstrip(':*'))[-1].strip()
    
    # Fallback: wrap in appropriate code block
    lang_map = {'python': 'python', 'cpp': 'cpp', 'riscv': 'c', 'c': 'c'}
    block_lang = lang_map.get(lang, 'c')
    return f'```{block_lang}\n{code}\n```'

def selection(candidates, cfg):
    correct_candidates = []
    wrong_candidates = []
    for obj in candidates:
        if obj['acc'] is not None and obj['acc']>0:
            correct_candidates.append(obj)
        else:
            wrong_candidates.append(obj)
    correct_candidates = sorted(correct_candidates, key=lambda x: (x['time']/x['input_time'])/(0.01+x['acc'])) # avoid divide zero

    selected = []
    seen = []
    for i in correct_candidates:
        dup = False
        for j in seen:
            try:
                # Use language-aware abstraction
                if cfg.lang == 'python':
                    abs_i = abstract_py_code(i['code'])
                    abs_j = abstract_py_code(j)
                else:
                    abs_i = abstract_cpp_code(i['code'])
                    abs_j = abstract_cpp_code(j)
                
                if str(abs_i) == str(abs_j):
                    dup = True
                    break
            except SyntaxError:
                dup = False
        if not dup:
            selected.append(i)
            seen.append(i['code'])
    if len(selected)<cfg.beam_number:
        distances = []
        for i, obj1 in enumerate(wrong_candidates):
            code1 = obj1['code']
            distance_sum = 0
            for j, obj2 in enumerate(wrong_candidates):
                code2 = obj2['code']
                if i != j:
                    try:
                        if cfg.lang == 'python':
                            seq1 = abstract_py_code(code1)
                            seq2 = abstract_py_code(code2)
                        else:
                            seq1 = abstract_cpp_code(code1)
                            seq2 = abstract_cpp_code(code2)
                        distance_sum += editdistance.eval(seq1, seq2)
                    except SyntaxError:
                        distance_sum += 9999
            distances.append((i, distance_sum))
        sorted_distances = sorted(distances, key=lambda x: x[1])
        closest_segments = [wrong_candidates[index] for index, _ in sorted_distances]
        selected.extend(closest_segments)

        
    return selected

    
def get_diff_lines(lines1, lines2):
    differ = difflib.Differ()
    diff = list(differ.compare(lines1, lines2))
    changes1 = [line[2:] for line in diff if line.startswith("- ")]
    changes2 = [line[2:] for line in diff if line.startswith("+ ")]
    return changes1, changes2



def process(data_chunk):
    # Vector retrieval globals
    global _GLOBAL_EMBEDDING_MODEL, _GLOBAL_VECTOR_INDEX, _GLOBAL_TRAIN_DATA
    global _GLOBAL_RETRIEVAL_METHOD, _GLOBAL_HYBRID_ALPHA

    cfg = data_chunk['cfg']
    ids = data_chunk['ids']
    code2content = data_chunk['code2content']
    results_data = data_chunk['results_data']
    previous_perf = data_chunk['previous_perf']

    train = _GLOBAL_TRAIN_DATA
    code_bm25 = data_chunk.get('code_bm25', None)
    edit_opt_bm25 = data_chunk.get('edit_opt_bm25', None)
    edit_code_bm25 = data_chunk.get('edit_code_bm25', None)
    
    vector_index = _GLOBAL_VECTOR_INDEX
    embedding_model = _GLOBAL_EMBEDDING_MODEL
    retrieval_method = _GLOBAL_RETRIEVAL_METHOD
    hybrid_alpha = _GLOBAL_HYBRID_ALPHA


    results = []
    differ = difflib.Differ()
    out_count=0
    for obj in results_data:
        if obj['stop'] >= 1 or obj['idx'] not in ids:
            results.append(obj)
        elif obj['query'] not in previous_perf:
            # Try cleaning query as fallback
            cleaned_query = remove_comments_and_docstrings(obj['query'], cfg.lang)
            if cleaned_query in previous_perf:
                 perf = previous_perf[cleaned_query]
            else:
                # Debug info
                # print(f"Warning: Query not found in perf report: {obj['query'][:30]}...")
                obj['stop'] = 2
                results.append(obj)
                continue
        else:
            perf = previous_perf[obj['query']]

            # update candidates
            if perf['input_time_mean'] is None:
                perf['input_time_mean'] = 30
            obj['iteration_{}'.format(str(cfg.iteration))] = []
            for i in range(cfg.generation_number):
                if perf['model_generated_potentially_faster_code_col_{}_time_mean'.format(str(i))] is None:
                    perf['model_generated_potentially_faster_code_col_{}_time_mean'.format(str(i))] = 99999
                if perf['model_generated_potentially_faster_code_col_{}'.format(str(i))] not in code2content:
                    out_count+=1
                    code2content[perf['model_generated_potentially_faster_code_col_{}'.format(str(i))]] = perf['model_generated_potentially_faster_code_col_{}'.format(str(i))]
                obj['iteration_{}'.format(str(cfg.iteration))].append({'code':perf['model_generated_potentially_faster_code_col_{}'.format(str(i))], 'acc':perf['model_generated_potentially_faster_code_col_{}_acc'.format(str(i))], 'time':perf['model_generated_potentially_faster_code_col_{}_time_mean'.format(str(i))], 'input_time':perf['input_time_mean'], 'content':code2content[perf['model_generated_potentially_faster_code_col_{}'.format(str(i))]]})
            last_candidates = ''.join([i['code'] for i in obj['best_candidates'][:3]]) # cfg.beam_number
            obj['best_candidates'].extend(obj['iteration_{}'.format(str(cfg.iteration))])
            obj['best_candidates'] = selection(obj['best_candidates'], cfg)[:5]
            new_candidates = ''.join([i['code'] for i in obj['best_candidates'][:3]]) # cfg.beam_number


            # compare and select the best result
            if perf['model_generated_potentially_faster_code_col_time_mean'] is None:
                perf['model_generated_potentially_faster_code_col_time_mean'] = 99999
            obj['iteration_{}_results'.format(str(cfg.iteration))] = {'code':perf['model_generated_potentially_faster_code_col'], 'acc':perf['model_generated_potentially_faster_code_col_acc'], 
                                                                      'time':perf['model_generated_potentially_faster_code_col_time_mean'], 
                                                                      'input_time': perf['input_time_mean'], 'input_acc': perf['input_acc'],
                                                                      'reference_time': perf['reference_time_mean'], 'reference_acc': perf['reference_acc']}
            if better(obj['iteration_{}_results'.format(str(cfg.iteration))], obj['best_result']):
                obj['best_result'] = obj['iteration_{}_results'.format(str(cfg.iteration))]
            # Implement "Convergence" requirement:
            # Do NOT stop just because we found one good solution. Continue iterating until max iterations.
            # Only stop if we specifically want to implement a convergence check (e.g. no improvement for 2 rounds).
            # For now, to match user request, we disable the "stop immediately on success" logic.
            # if last_candidates == new_candidates and obj['best_result']['acc'] == 1:
            #     obj['stop'] = 1
            if last_candidates == new_candidates:
                results.append(obj)
                continue

            if cfg.iteration < 5:
                try:
                    if cfg.lang == 'python':
                        query_abs = tokenize_py_code(abstract_py_code(obj['query']))
                    else:
                        # Use C++ abstractor for C/RISC-V as well
                        query_abs = tokenize_cpp_code(abstract_cpp_code(obj['query'])) 
                except:
                    obj['pattern'] = []
                    results.append(obj)
                    continue
                
                # Perform Hybrid Search using the new utility
                top_k = 100
                code_scores = perform_hybrid_search(
                    query_abs, train, retrieval_method, hybrid_alpha, 
                    vector_index, embedding_model, code_bm25, top_k
                )
                
                sim_scores = np.copy(code_scores)
                dissim_scores = np.copy(code_scores)
                
                for candidate in obj['best_candidates'][:cfg.beam_number]:
                    try:
                        if cfg.lang=='python':
                            candidate_abs = tokenize_py_code(abstract_py_code(candidate['code'])) 
                        else:
                            candidate_abs = tokenize_cpp_code(abstract_cpp_code(candidate['code']))
                    except:
                        continue
                    edit_code_abs, edit_opt_abs = get_diff_lines(' '.join(query_abs).split('\n'), ' '.join(candidate_abs).split('\n'))
                    edit_code_abs = '\n'.join(edit_code_abs).split()
                    edit_opt_abs = '\n'.join(edit_opt_abs).split()
                    
                    #    ?edit_code_abs    ?edit_opt_abs                                              ?
                    if retrieval_method in ['bm25', 'hybrid'] and edit_code_bm25 is not None and edit_opt_bm25 is not None:
                        edit_bm25_scores = np.array(edit_code_bm25.get_scores(edit_code_abs)) + np.array(edit_opt_bm25.get_scores(edit_opt_abs))
                        if edit_bm25_scores.max() - edit_bm25_scores.min() > 0:
                            edit_bm25_scores = (edit_bm25_scores - edit_bm25_scores.min()) / (edit_bm25_scores.max() - edit_bm25_scores.min())
                        else:
                            edit_bm25_scores = np.zeros(len(train))
                    else:
                        edit_bm25_scores = np.zeros(len(train))
                    
                    #                           ?edit_code_abs
                    if retrieval_method in ['vector', 'hybrid'] and vector_index is not None and embedding_model is not None and vector_index.ntotal > 0:
                        edit_text = ' '.join(edit_code_abs) + ' ' + ' '.join(edit_opt_abs)
                        edit_embedding = embedding_model.encode([edit_text], convert_to_numpy=True)
                        k = min(top_k, vector_index.ntotal)
                        edit_vector_scores, edit_vector_indices = vector_index.search(edit_embedding, k)
                        edit_vector_scores = edit_vector_scores[0]
                        if len(edit_vector_scores) > 0:
                            score_range = edit_vector_scores.max() - edit_vector_scores.min()
                            if score_range > 1e-8:
                                edit_vector_scores_normalized = 1 - (edit_vector_scores - edit_vector_scores.min()) / score_range
                            else:
                                edit_vector_scores_normalized = np.ones_like(edit_vector_scores)
                        else:
                            edit_vector_scores_normalized = np.array([])
                        
                        edit_vector_final = np.zeros(len(train))
                        if len(edit_vector_indices) > 0 and len(edit_vector_scores_normalized) > 0:
                            for idx, score in zip(edit_vector_indices[0], edit_vector_scores_normalized):
                                if 0 <= idx < len(edit_vector_final):
                                    edit_vector_final[idx] = score
                    else:
                        edit_vector_final = np.zeros(len(train))
                    
                    #             ?edit             ?
                    if retrieval_method == 'hybrid':
                        edit_scores = hybrid_alpha * edit_bm25_scores + (1 - hybrid_alpha) * edit_vector_final
                    elif retrieval_method == 'vector':
                        edit_scores = edit_vector_final
                    else:
                        edit_scores = edit_bm25_scores
                    
                    if edit_scores.max() - edit_scores.min() > 0:
                        edit_scores = (edit_scores - edit_scores.min()) / (edit_scores.max() - edit_scores.min() + 1e-8) 
                        median_val = np.median(edit_scores)
                        sim_edit_scores = np.where(edit_scores < median_val, 0, edit_scores)
                        dissim_edit_scores = np.where(edit_scores > median_val, 0, np.max(edit_scores)-edit_scores)
                        sim_scores+=sim_edit_scores/cfg.beam_number
                        dissim_scores+=dissim_edit_scores/cfg.beam_number
                
                sim_top_indices  = np.argsort(-sim_scores)[0]
                # Use correct field names from train.jsonl
                sim_query = train[sim_top_indices].get('code_v0_no_empty_lines', train[sim_top_indices].get('query', ''))
                sim_ref = train[sim_top_indices].get('target', train[sim_top_indices].get('reference', ''))
                sim_diff = list(differ.compare(sim_query.split('\n'), sim_ref.split('\n')))
                sim_pattern = [line for line in sim_diff if line.startswith("- ") or line.startswith("+ ")]
                dissim_top_indices  = np.argsort(-dissim_scores)[0]
                dissim_query = train[dissim_top_indices].get('code_v0_no_empty_lines', train[dissim_top_indices].get('query', ''))
                dissim_ref = train[dissim_top_indices].get('target', train[dissim_top_indices].get('reference', ''))
                dissim_diff = list(differ.compare(dissim_query.split('\n'), dissim_ref.split('\n')))
                dissim_pattern = [line for line in dissim_diff if line.startswith("- ") or line.startswith("+ ")]
                obj['pattern'] = ['\n'.join(sim_pattern), '\n'.join(dissim_pattern)]
                
                # Save retrieval info
                obj['retrieval'] = {
                    "query": ' '.join(query_abs),
                    "selected_ids": [int(sim_top_indices), int(dissim_top_indices)],
                    "method": retrieval_method
                }
            
            # append
            results.append(obj)

    return results




def main(cfg):
    previous_perf = {}
    report_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'test_execution_{}test.report'.format(cfg.model_name))
    
    if not os.path.exists(report_path):
        print(f"Warning: Report path does not exist ({report_path}), skipping merge for this round.")
        # Preserve previous results
        results_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, 'results.jsonl')
        if os.path.exists(results_path):
            results = []
            with jsonlines.open(results_path) as f:
                for obj in f:
                    results.append(obj)
            with jsonlines.open(results_path, 'w') as f:
                f.write_all(results)
        return
    
    with jsonlines.open(report_path) as f:
        for obj in f:
            # Store by both raw code and cleaned code to ensure matching
            raw_code = obj['code_v0_no_empty_lines']
            previous_perf[raw_code] = obj
            
            cleaned_code = remove_comments_and_docstrings(raw_code, cfg.lang)
            if cleaned_code not in previous_perf: # prioritization to raw
                previous_perf[cleaned_code] = obj

    code2content = {}
    tmp_queries_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'tmp_queries_{}test.jsonl'.format(cfg.model_name))
    
    if not os.path.exists(tmp_queries_path):
        print(f"Warning: Query file does not exist ({tmp_queries_path}), skipping merge for this round.")
        # Just copy previous results.jsonl to maintain state
        results_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, 'results.jsonl')
        if os.path.exists(results_path):
            results = []
            with jsonlines.open(results_path) as f:
                for obj in f:
                    results.append(obj)
            with jsonlines.open(results_path, 'w') as f:
                f.write_all(results)
        return
    
    with jsonlines.open(tmp_queries_path) as f:
        for obj in f:
            for code, content in zip(obj['prediction'], obj['detailed_prediction']):
                if isinstance(content, dict) and 'message' in content:
                    # assert code in content['message']['content']
                    if cfg.iteration ==1:
                        code2content[code] = content['message']['content']
                    else:
                        if cfg.lang=='python':
                            code2content[code] = extract_py(code, content['message']['content'])
                        else:
                            code2content[code] = extract_cpp(code, content['message']['content'])
                else:
                    # assert code in content
                    if cfg.iteration ==1:
                        code2content[code] = content
                    else:
                        if cfg.lang=='python':
                            code2content[code] = extract_py(code, content)
                        else:
                            code2content[code] = extract_cpp(code, content)



    train = []
    code_corpus = []
    edit_code_corpus = []
    edit_opt_corpus = []
    with jsonlines.open(cfg.training_data_path) as f:
        for obj in f:
            train.append(obj)
            if len(obj['query_abs'])>0 and len(obj['edit_code_abs'])>0 and len(obj['edit_opt_abs'])>0:
                code_corpus.append(obj['query_abs'])
                edit_code_corpus.append(obj['edit_code_abs'])
                edit_opt_corpus.append(obj['edit_opt_abs'])
    
    # BM25             ?
    code_bm25 = BM25Okapi(code_corpus, b=0.4) if len(code_corpus) > 0 else None
    edit_code_bm25 = BM25Okapi(edit_code_corpus, b=0.4) if len(edit_code_corpus) > 0 else None
    edit_opt_bm25 = BM25Okapi(edit_opt_corpus, b=0.4) if len(edit_opt_corpus) > 0 else None
    
    # Vector Index Initialization
    vector_index = None
    embedding_model = None
    train_embeddings = None
    
    if cfg.retrieval_method in ['vector', 'hybrid']:
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(cfg.embedding_model)
            print(f"Loaded embedding model: {cfg.embedding_model}")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            if cfg.retrieval_method == 'vector':
                print("Falling back to BM25")
                cfg.retrieval_method = 'bm25'

        if embedding_model is not None:
             vector_index, train_embeddings = load_or_build_vector_index(cfg, train, embedding_model)


    ids = set()
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'tmp_queries_{}test.jsonl'.format(cfg.model_name))) as f:
        for obj in f:
            ids.add(obj['idx'])
            
    results_data = []
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, 'results.jsonl')) as f:
        for obj in f:
            results_data.append(obj)

    processes = int(getattr(cfg, 'process_number', 20))
    #                                                                                          
    init_global_resources(embedding_model, vector_index, train, cfg.retrieval_method, cfg.hybrid_alpha)
    pool = multiprocessing.Pool(processes)
    chunks_results_data = np.array_split(results_data, processes)
    chunks_data = []
    for i in range(processes):
        tmp_data={}
        tmp_data['cfg'] = cfg
        tmp_data['ids'] = ids
        tmp_data['code2content'] = code2content
        tmp_data['results_data'] = chunks_results_data[i]
        tmp_data['previous_perf'] = previous_perf

        tmp_data['code_bm25'] = code_bm25
        tmp_data['edit_opt_bm25'] = edit_opt_bm25
        tmp_data['edit_code_bm25'] = edit_code_bm25
        chunks_data.append(tmp_data)
    results = []
    # Use imap_unordered to track progress across chunks
    with tqdm(total=len(results_data), desc="Merging Results") as pbar:
        for chunk in pool.imap_unordered(process, chunks_data):
            results.extend(chunk)
            pbar.update(len(chunk))
    
    pool.close()
    pool.join()
    
    # Sort results back to original order if needed, or just assert counts match
    assert len(results) == len(results_data)

    
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, 'results.jsonl'), 'w') as f:
        f.write_all(results)

    sp = 0
    ptr = 0
    faster_count = 0
    unique_count = 0
    input_time_sum = 0
    generated_test_sum = 0
    faster_input_time_sum = 0
    faster_generated_test_sum = 0
    unique_reference_time_sum = 0
    unique_generated_test_sum = 0
    for obj in results:
        # calculate current best results
        acc = obj['best_result']['acc']
        input_time = obj['best_result']['input_time']
        generated_time = obj['best_result']['time']
        reference_time = obj['best_result']['reference_time']
        # if obj['best_result']['input_time'] <1 or obj['best_result']['reference_time'] <1:
        #     continue
        if generated_time is None:
            generated_time=input_time
        if acc==1 and generated_time<input_time:
            if generated_time<reference_time:
               unique_count+=1
               unique_reference_time_sum += reference_time
               unique_generated_test_sum += generated_time
            faster_count+=1
            faster_input_time_sum += input_time
            faster_generated_test_sum += generated_time
            input_time_sum += input_time
            generated_test_sum += generated_time
            #ptr += input_time/generated_time
        else:
            input_time_sum += input_time
            generated_test_sum += input_time
            #ptr += 1
    print("\n" + "=" * 60)
    print("           合并阶段评估报告 (Merge Phase Summary)")
    print("=" * 60)
    print(f"{'指标 (Metric)':<40} | {'数值 (Value)':<15}")
    print("-" * 60)
    print(f"{'成功优化数 (Optimized Queries)':<40} | {faster_count}/{len(results)}")
    print(f"{'优化覆盖率 (Coverage)':<40} | {round(100*faster_count/len(results), 2)}%")
    
    speedup = round(faster_input_time_sum/faster_generated_test_sum, 2) if faster_generated_test_sum > 0 else 0.0
    print(f"{'平均加速比 (Avg Speedup)':<40} | {speedup:.2f}x (vs v0)")
    
    ptr = round(input_time_sum/generated_test_sum, 2) if generated_test_sum > 0 else 0.0
    # print(f"{'全局收益 (Global Perf Gain)':<40} | {ptr:.2f}x") # Less useful for user
    
    if unique_count > 0:
        print("-" * 60)
        print(f"{'超越参考解 (Better than Ref)':<40} | {unique_count}")
        unique_speedup = round(unique_reference_time_sum/unique_generated_test_sum, 2) if unique_generated_test_sum > 0 else 0.0
        print(f"{'超越幅度 (Speedup vs Ref)':<40} | {unique_speedup:.2f}x")
    
    print("=" * 60 + "\n")

    # --- Convergence State Output ---
    # Write current iteration's metrics to a state file for the shell script to read
    import json
    convergence_state_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, 'convergence_state.json')
    
    # Load previous state if exists
    prev_state = {}
    if os.path.exists(convergence_state_path):
        try:
            with open(convergence_state_path, 'r') as f:
                prev_state = json.load(f)
        except:
            prev_state = {}
    
    prev_speedup = prev_state.get('best_speedup', 0.0)
    prev_no_improvement_count = prev_state.get('no_improvement_count', 0)
    
    # Determine if there's improvement
    current_speedup = speedup
    if current_speedup > prev_speedup + 0.01:  # 1% improvement threshold
        no_improvement_count = 0
        best_speedup = current_speedup
    else:
        no_improvement_count = prev_no_improvement_count + 1
        best_speedup = max(prev_speedup, current_speedup)
    
    # Write updated state
    new_state = {
        'iteration': cfg.iteration,
        'current_speedup': current_speedup,
        'best_speedup': best_speedup,
        'faster_count': faster_count,
        'total_count': len(results),
        'no_improvement_count': no_improvement_count
    }
    with open(convergence_state_path, 'w') as f:
        json.dump(new_state, f, indent=2)
    
    print(f"[Convergence] No improvement rounds: {no_improvement_count} | Best speedup: {best_speedup:.2f}x")

    # Write final results for consolidated evaluation
    if cfg.iteration >= 2: # Or generally the last iteration
        print("\nGenerating final results for evaluation...")
        final_dir = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'final')
        os.makedirs(final_dir, exist_ok=True)
        
        final_results = []
        for item in results:
             # Just like top1, we take the best single candidate (or best few if we want to test multiple)
             # But usually 'final' implies the single best choice per query.
             # However, to be robust, let's include the top 1 candidate as the 'prediction'.
             # If we want to evaluate all candidates, we'd need to change evaluation logic.
             # For now, let's output the best candidate (Top-1) as the final choice.
             
            if len(item['best_candidates']) == 0:
                 best_code = item.get('query', '') or item.get('code_v0_no_empty_lines', '')
            else:
                 best_code = item['best_candidates'][0]['code']

            final_results.append({
                'query': item['query'],
                'prediction': [best_code] # Evaluate expects a list
            })
            
        output_path = os.path.join(final_dir, f'tmp_queries_{cfg.model_name}test.jsonl')
        with jsonlines.open(output_path, 'w') as f:
            f.write_all(final_results)
        print(f"  Written: {output_path} ({len(final_results)} queries)")



from sbllm.core.arg_parser import cfg_parsing

if __name__ == "__main__":
    cfg = cfg_parsing()
    main(cfg)


