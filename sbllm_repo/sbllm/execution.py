import os
import re
import tokenize
from io import StringIO
import time
import jsonlines 
import subprocess
import logging
from sbllm.utils.qemu_evaluator import QEMURISCVEvaluator, StandaloneEvaluator
import multiprocessing
import shutil
import tempfile
import traceback
from tqdm import tqdm
from sbllm.utils.code_utils import remove_comments_and_docstrings
from sbllm.utils.statistics_utils import calculate_statistics, write_yaml, calculate_composite_score
from sbllm.utils.report_utils import generate_markdown_report
from sbllm.core.evaluator_interface import BaseEvaluator, EvalMetric

logger = logging.getLogger(__name__)

def init_worker(log_level, log_file):
    """Initialize logging for spawned worker processes."""
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] [%(processName)s] %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8')]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Worker process initialized in {os.name} environment.")

def generic_eval_worker(args_dict):
    """
    Worker function for parallel evaluation.
    Supports Docker orchestration.
    """
    try:
        item = args_dict['item']
        cfg = args_dict['cfg']
        reference_data_item = args_dict['ref_data_item']
        use_docker = args_dict.get('use_docker', False)
        project_root = args_dict.get('project_root', None)
        
        # 1. Instantiate Evaluator Strategy
        # Setup worker temp dir
        parent_temp_dir = os.path.join(cfg.output_path, cfg.mode, 'qemu_temp_run')
        os.makedirs(parent_temp_dir, exist_ok=True)
        worker_temp_dir = tempfile.mkdtemp(dir=parent_temp_dir)

        if getattr(cfg, 'dataset_type', '') in ['project_rvv', 'project_vecintrin']:
            evaluator = StandaloneEvaluator(
                qemu_path=cfg.qemu_path,
                riscv_gcc_toolchain_path=cfg.riscv_gcc_toolchain_path,
                temp_dir=worker_temp_dir,
                use_docker=use_docker,
                project_root=project_root
            )
        else:
            evaluator = QEMURISCVEvaluator(
                qemu_path=cfg.qemu_path,
                riscv_gcc_toolchain_path=cfg.riscv_gcc_toolchain_path,
                temp_dir=worker_temp_dir,
                use_docker=use_docker,
                project_root=project_root
            )

        # 2. Prepare Data
        original_code = item['slow_code_col']
        candidate_codes = item['model_generated_potentially_faster_code_col']
        if not isinstance(candidate_codes, list):
            candidate_codes = [candidate_codes]
        
        ref_code = reference_data_item.get('code', original_code)
        
        # 3. Evaluate Baseline
        baseline_metadata = reference_data_item.copy()
        baseline_metadata['input'] = reference_data_item.get('input', '')
        baseline_metadata['reference_output'] = None
        
        from sbllm.utils.code_sanitizer import CodeSanitizer
        benchmark_name = reference_data_item.get('benchmark_name', '')
        
        # ROUND 19.8: Sanitize baseline code to handle reserved names (e.g. memcpy)
        original_code_sanitized = CodeSanitizer.sanitize(original_code, benchmark_name)
        
        baseline_metric = evaluator.evaluate(original_code_sanitized, baseline_metadata, 
                                      num_runs=cfg.eval_num_runs, 
                                      timeout=cfg.eval_timeout,
                                      compile_timeout=cfg.compile_timeout)
        
        # Evaluate Reference
        # ROUND 19.8: Sanitize reference code as well
        ref_code_sanitized = CodeSanitizer.sanitize(ref_code, benchmark_name)
        ref_metric = evaluator.evaluate(ref_code_sanitized, baseline_metadata)
        standard_output = baseline_metric.output or ref_metric.output

        # 4. Evaluate Candidates
        best_candidate_idx = -1
        best_metric = baseline_metric
        best_score = -float('inf')
        candidate_results = []
        first_candidate_error = ""
        
        candidate_metadata = baseline_metadata.copy()
        candidate_metadata['reference_output'] = standard_output
        
        for idx, original_code_snippet in enumerate(candidate_codes[:10]):
            # ROUND 19: Apply Code Sanitizer (Auto-Correction)
            # This fixes RVV API versions and function names before compilation
            code = CodeSanitizer.sanitize(original_code_snippet, benchmark_name)
            
            metric = evaluator.evaluate(code, candidate_metadata,
                                           num_runs=cfg.eval_num_runs,
                                           timeout=cfg.eval_timeout,
                                           compile_timeout=cfg.compile_timeout)


        
            score = calculate_composite_score(
                metric.time, baseline_metric.time,
                metric.size, baseline_metric.size,
                alpha=cfg.score_alpha
            )
            
            if not metric.correct and not first_candidate_error and metric.error:
                 first_candidate_error = f"Candidate {idx} Error: {metric.error}"
            
            res_dict = {
                'idx': idx,
                'time': metric.time,
                'size': metric.size,
                'correct': metric.correct,
                'code': code,
                'output': metric.output or metric.error,
                'opt_ratio': 1.0 - (metric.time/baseline_metric.time) if baseline_metric.time > 0 and metric.time < 99999 else 0,
                'composite_score': score
            }
            candidate_results.append(res_dict)
            
            if metric.correct and score > best_score:
                best_score = score
                best_metric = metric
                best_candidate_idx = idx
        
        # 5. Construct Result Item
        exec_item = {
            'query_idx': reference_data_item.get('idx', reference_data_item.get('id', 'Unknown')),
            'query_desc': reference_data_item.get('description', reference_data_item.get('benchmark_name', '')),
            'code_v0_no_empty_lines': item['slow_code_col'],
            'input_time_mean': baseline_metric.time,
            'input_acc': 1 if baseline_metric.correct else 0,
            'baseline_size': baseline_metric.size, # Added baseline size for comparison
            
            'model_generated_potentially_faster_code_col': candidate_codes[best_candidate_idx] if best_candidate_idx >= 0 else original_code,
            'model_generated_potentially_faster_code_col_acc': 1 if best_metric.correct else 0,
            'model_generated_potentially_faster_code_col_time_mean': best_metric.time,
            'size': best_metric.size,
            'is_fallback': best_candidate_idx == -1, # Track if we fell back to original code
            'debug_output': best_metric.output or best_metric.error
        }

        
        if best_candidate_idx == -1 and first_candidate_error:
            exec_item['debug_output'] = first_candidate_error

        # Append individual candidate details
        for res in candidate_results:
            ridx = res['idx']
            exec_item[f'candidate_{ridx}_time'] = res['time']
            exec_item[f'candidate_{ridx}_correct'] = res['correct']
            exec_item[f'candidate_{ridx}_score'] = res['composite_score']
        
        # 6. Cleanup
        try:
            shutil.rmtree(worker_temp_dir)
        except:
            pass
            
        return exec_item

    except Exception as e:
        logger.error(f"Worker process failed: {e}\n{traceback.format_exc()}")
        return {'error': str(e), 'traceback': traceback.format_exc()}

def testing_and_reporting(cfg):
    # Check for RISC-V Mode
    is_riscv_mode = hasattr(cfg, 'riscv_mode') and cfg.riscv_mode or cfg.lang == 'riscv' or (cfg.mode and 'riscv' in cfg.mode.lower())
    
    if is_riscv_mode and hasattr(cfg, 'qemu_path') and hasattr(cfg, 'riscv_gcc_toolchain_path'):
        # Determine Evaluator Type
        dataset_type = getattr(cfg, 'dataset_type', 'standard')
        
        _testing_and_reporting_qemu(cfg)
    else:
        # Use Legacy Evaluator
        logger.info('Using Original Evaluator...')
        _testing_and_reporting_original(cfg)


def _testing_and_reporting_qemu(cfg):
    """Execution with QEMU/Standalone Evaluator"""
    logger.info('Mapping code and inputs...')
    reference_data = {}
    # Determine test data path
    if hasattr(cfg, 'baseline_data_path') and cfg.baseline_data_path:
        test_data_path = cfg.baseline_data_path
    elif hasattr(cfg, 'dataset_type') and cfg.dataset_type in ['project_rvv', 'project_vecintrin']:
        # Auto-detect path for new datasets
        if cfg.dataset_type == 'project_rvv':
            test_data_path = os.path.join(os.getcwd(), "processed_data", "rvv-bench", "test.jsonl")
        else:
            test_data_path = os.path.join(os.getcwd(), "processed_data", "VecIntrinBench", "test.jsonl")
    else:
        test_data_path = os.path.join(os.getcwd(), "processed_data", cfg.lang, "test.jsonl")
    
    if not os.path.exists(test_data_path):
        logger.error(f"Test data file not found: {test_data_path}")
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    
    # helper for cleaning key
    def clean_key(code):
        return remove_comments_and_docstrings(code, cfg.lang).strip()

    reference_data = {}
    reference_data_idx = {}
    with jsonlines.open(test_data_path) as f:
        for obj in f:
            # Map cleaned code to full object (input, code, output etc)
            key_code = clean_key(obj['code_v0_no_empty_lines'])
            reference_data[key_code] = obj
            
            # Map idx if available
            if 'idx' in obj:
                reference_data_idx[str(obj['idx'])] = obj

    print('Processing...')
    processed = []
    queries_path = os.path.join(cfg.output_path, cfg.mode, 'tmp_queries_{}.jsonl'.format(cfg.model_name+'test'))
    if not os.path.exists(queries_path):
        logger.warning(f"Warning: Queries file not found ({queries_path}), skipping evaluation.")
        # Create empty report
        report_path = os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.report'.format(cfg.model_name+'test'))
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with jsonlines.open(report_path, 'w') as f:
            pass  # Empty file
        return
    
    with jsonlines.open(queries_path) as f:
        for i in f:
            # Try to match by idx first (preferred)
            matched_ref = None
            query_idx = str(i.get('idx', ''))
            query_key = i.get('query', '')
            if not query_key and 'code_v0_no_empty_lines' in i:
                query_key = i['code_v0_no_empty_lines']
            
            if query_idx in reference_data_idx:
                matched_ref = reference_data_idx[query_idx]
            else:
                # Fallback to code string matching

                
                if query_key in reference_data:
                    matched_ref = reference_data[query_key]
                else:
                    # Try cleaning
                    cleaned_q = clean_key(query_key)
                    if cleaned_q in reference_data:
                        matched_ref = reference_data[cleaned_q]
            
            if matched_ref:
                processed.append({
                    'slow_code_col': matched_ref.get('code_v0_no_empty_lines', query_key),
                    'model_generated_potentially_faster_code_col': i.get('prediction', []),
                    'matched_ref': matched_ref
                })

    print(f'{len(processed)} queries matched reference data')
    
    if len(processed) == 0:
        print("No results to evaluate")
        # Create empty report
        report_path = os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.report'.format(cfg.model_name+'test'))
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with jsonlines.open(report_path, 'w') as f:
            pass
        return

    # Prepare args for parallel execution
    pool_args = []
    for item in processed:
        ref_data_item = item.get('matched_ref', {})
        pool_args.append((item, cfg, ref_data_item))

    print(f"Using {getattr(cfg, 'process_number', 8)} processes for parallel evaluation...")
    execution_data = []
    success_count = 0
    fail_count = 0
    
    # Detailed Failure Tracking
    failure_stats = {
        'Compilation Failed': 0,
        'Timeout': 0, 
        'Wrong Output': 0,
        'Runtime Error': 0,
        'Other': 0
    }

    start_time = time.time()
    
    # --- Pre-flight Check ---
    if os.name == 'nt' or getattr(cfg, 'use_docker', False):
        if getattr(cfg, 'use_docker', False):
            from sbllm.utils.docker_manager import docker_manager
            logger.info("Performing Docker Pre-flight check & Session Initialization...")
            docker_manager.initialize(os.getcwd())
            # Double check
            try:
                subprocess.run(['docker', 'info'], capture_output=True, check=True)
                logger.info("Docker daemon is running and accessible.")
            except Exception:
                logger.error("Docker daemon not found or not running. Please start Docker Desktop.")
                return
        else:
            logger.info("Performing WSL Pre-flight check...")
            try:
                wsl_check = subprocess.run(['wsl', 'uname', '-a'], capture_output=True, text=True, timeout=5)
                logger.info(f"WSL Connectivity confirmed: {wsl_check.stdout.strip()}")
            except Exception:
                logger.error("WSL check failed. Ensure WSL is installed and default distro is set.")
                return

    log_file = None
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = handler.baseFilename
            break
            
    # --- Worker Initialization ---
    if len(pool_args) > 0:
        with multiprocessing.Pool(
            processes=getattr(cfg, 'process_number', 8),
            initializer=init_worker,
            initargs=(logging.getLogger().getEffectiveLevel(), log_file)
        ) as pool:
            # Re-creating pool_args as list of dicts to allow modification
            new_pool_args = []
            for item, cfg_arg, ref_data_item in pool_args:
                arg_dict = {
                    'item': item,
                    'cfg': cfg_arg,
                    'ref_data_item': ref_data_item,
                    'use_docker': getattr(cfg, 'use_docker', False),
                    'project_root': os.getcwd()
                }
                new_pool_args.append(arg_dict)
            pool_args = new_pool_args 

            with tqdm(total=len(pool_args), desc="Evaluation Progress", mininterval=0.5) as pbar:
                for result in pool.imap_unordered(generic_eval_worker, pool_args):
                    if result:
                        if 'error' in result:
                            fail_count += 1
                            failure_stats['Other'] += 1
                            logger.error(f"Worker failure: {result['error']}")
                        else:
                            execution_data.append(result)
                            best_acc = result.get('model_generated_potentially_faster_code_col_acc', 0)
                            if best_acc == 1:
                                success_count += 1
                            else:
                                fail_count += 1
                    else:
                        fail_count += 1
                        failure_stats['Other'] += 1
                    
                    pbar.set_postfix(success=success_count, fail=fail_count)
                    pbar.update(1)

    # --- Cleanup ---
    if getattr(cfg, 'use_docker', False):
        from sbllm.utils.docker_manager import docker_manager
        docker_manager.cleanup()

    total_time = time.time() - start_time

    # --- Detailed Evaluation Summary ---
    print("\n" + "="*70)
    print("  详细评估结果 (Detailed Evaluation Results)")
    print("="*70)
    print("="*70)
    total_opt_count = 0
    # total_speedup_sum = 0.0 # No longer needed
    max_speedup_overall = 0.0
    for idx, result in enumerate(execution_data):
        # Extract query ID and description
        query_idx = result.get('query_idx', idx+1)
        query_desc = result.get('query_desc', '')
        # Fix: use '\n' for split, not '\\n'
        code_preview = result.get('code_v0_no_empty_lines', '').split('\n')[0][:40]
        original_time = result.get('input_time_mean', 99999)
        best_time = result.get('model_generated_potentially_faster_code_col_time_mean', 99999)
        best_acc = result.get('model_generated_potentially_faster_code_col_acc', 0)
        
        # Count successful candidates
        cand_success_count = 0
        for i in range(5):
            if result.get(f'model_generated_potentially_faster_code_col_{i}_acc', 0) == 1:
                cand_success_count += 1
        
        # Calculate best speedup for this query
        if original_time > 0 and 0 < best_time < 99999:
            speedup = original_time / best_time
        else:
            speedup = 0.0
        
        # Collect speedups for all candidates
        candidate_speedups = []
        for i in range(5):
             cand_time = result.get(f'model_generated_potentially_faster_code_col_{i}_time_mean', 99999)
             if result.get(f'model_generated_potentially_faster_code_col_{i}_acc', 0) == 1 and 0 < cand_time < 99999 and original_time > 0:
                 sp = original_time / cand_time
                 candidate_speedups.append(f"{sp:.2f}x")
             else:
                 candidate_speedups.append("-")

        if best_acc == 1 and speedup > 1.0:
            status = "✓ OPT"
            total_opt_count += 1
            if speedup > max_speedup_overall:
                max_speedup_overall = speedup
        elif best_acc == 1:
            status = "✓ OK (No Speedup)"
        else:
            status = "✗ Fail"
        
        # Display with query ID
        desc_display = f" ({query_desc})" if query_desc else ""
        print(f"  [ID:{query_idx}] {code_preview}...{desc_display}")
        print(f"      成功候选: {cand_success_count}/5 | 最佳加速比: {speedup:.2f}x (vs Original) | 状态: {status}")
        print(f"      候选详情: {', '.join(candidate_speedups)}")
        print("-"*70)
    
    # Summary statistics
    # avg_speedup = total_speedup_sum / max(total_opt_count, 1) # Removed as per request
    print(f"\n  汇总统计:")
    print(f"    - 总查询数: {len(pool_args)}")
    print(f"    - 成功执行: {len(execution_data)}")
    if sum(failure_stats.values()) > 0:
        print(f"    - 失败统计: {dict(failure_stats)}")
    print(f"    - 成功优化: {total_opt_count}/{len(execution_data)} ({100*total_opt_count/max(len(execution_data),1):.1f}%)")
    print(f"    - 最高加速比 (全局): {max_speedup_overall:.2f}x (vs Original)")
    print(f"    - 评估耗时: {total_time:.2f}s")
    print("="*70 + "\n")
    # --- End Detailed Summary ---

    # Calculate Statistics (Injects CodeBLEU and other metrics into execution_data)
    calculate_statistics(execution_data, cfg)

    # Save Report
    report_path = os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.report'.format(cfg.model_name+'test'))
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with jsonlines.open(report_path, 'w') as f:
        f.write_all(execution_data)
    
    # Generate Readable Markdown Report
    try:
        generate_markdown_report(execution_data, os.path.join(cfg.output_path, cfg.mode), cfg.model_name, failure_stats)
    except Exception as e:
        logger.error(f"Failed to generate markdown report: {e}")

    # Explicit Log Feedback
    log_file = None
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = handler.baseFilename
            break
    if log_file:
        print(f"\n[INFO] Full execution log available at: {log_file}")


def _testing_and_reporting_original(cfg):
    """Original Evaluation System"""
    print('Mapping...')
    input_code_map = {}
    test_data_path = os.path.join(os.getcwd(), "processed_data", cfg.lang, "test.jsonl")
    with jsonlines.open(test_data_path) as f:
        for obj in f:
            input_code_map[remove_comments_and_docstrings(obj['code_v0_no_empty_lines'], cfg.lang)] = obj.get('input', '')

    print('Processing...')
    processed = []
    with jsonlines.open(os.path.join(cfg.output_path, cfg.mode, 'tmp_queries_{}.jsonl'.format(cfg.model_name+'test'))) as f:
        for i in f:
            processed.append({'slow_code_col':input_code_map[i['query']], 'model_generated_potentially_faster_code_col':i['prediction']})
    with jsonlines.open(os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.jsonl'.format(cfg.model_name+'test')),'w') as f:
        f.write_all(processed)

    data = {}
    data['language'] = cfg.lang
    data['model_generated_outputs_path'] = os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.jsonl'.format(cfg.model_name+'test'))
    data['inputs_outputs_basepath'] = cfg.test_case_path
    data['reference_file_path'] = os.path.join(os.getcwd(), "processed_data", cfg.lang, "test.jsonl")
    data['output_report_file_path'] = os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.report'.format(cfg.model_name+'test'))
    data['preprocessed_output_file'] = os.path.join(os.getcwd(), "processed_data", cfg.lang, "processed_time.reports")
    data['num_problems_to_evaluate'] = -1
    data['num_trials'] = 8
    data['ignore_first_k'] = 1
    data['max_time_per_run'] = 10
    data['temp_dir'] = os.path.join(os.getcwd(), "processed_data", cfg.lang, "generated_tmp")
    data['model_generated_potentially_faster_code_col'] = "model_generated_potentially_faster_code_col"
    data['slow_code_col'] = "slow_code_col"
    data['reference_code_col'] = "target"
    data['reference_input_col'] = "input"
    data['is_prompt_based'] = False
    data['run_reference'] = True
    data['run_input'] = True
    data['cpu_number'] = 1
    data['process_number'] = int(cfg.process_number)
    write_yaml(data, os.path.join(cfg.output_path, cfg.mode, 'eval_config.yaml'))

    error_file = open(os.path.join(cfg.output_path, cfg.mode, "stderr.txt"), "wb")
    out_file = open(os.path.join(cfg.output_path, cfg.mode, "output.txt"),"wb")
    print("Testing...")
    
    if not os.path.exists(os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.report'.format(cfg.model_name+'test'))):
        # Find PIE directory
        pie_dir = "../pie"
        if os.path.exists("pie"):
            pie_dir = "pie"
        elif os.path.exists("sbllm_repo/pie"):
            pie_dir = "sbllm_repo/pie"
            
        cmd = 'cd {}; python src/codenet_eval/run_eval_feedback.py --eval_config {}'.format(pie_dir, os.path.join(cfg.output_path, cfg.mode, 'eval_config.yaml'))
        logger.info(f"Running PIE evaluation: {cmd}")
        child = subprocess.Popen(cmd, shell=True, stdout=out_file, stderr=error_file, bufsize=-1, start_new_session=True)
        while True:
            Flag = child.poll()
            if Flag == 0:
                error_file.close()
                out_file.close()
                break
            else:
                time.sleep(10)
    
    execution_data = []
    with jsonlines.open(os.path.join(cfg.output_path, cfg.mode, 'test_execution_{}.report'.format(cfg.model_name+'test'))) as f:
        for i in f:
            execution_data.append(i)
    
    calculate_statistics(execution_data, cfg)


from sbllm.core.arg_parser import cfg_parsing
from sbllm.core.logger import setup_app_logging

if __name__ == "__main__":
    cfg = cfg_parsing()
    setup_app_logging(cfg)
    testing_and_reporting(cfg)
