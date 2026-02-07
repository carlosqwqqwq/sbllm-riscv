import re
import tokenize
import argparse
from io import StringIO
import jsonlines
import os
from .utils.code_utils import remove_comments_and_docstrings


# Removed: remove_comments_and_docstrings now imported from code_utils


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='chatgpt', type=str, required=True)
parser.add_argument("--lang", default='python', type=str, required=True)
cfg = parser.parse_args()


data = []
if cfg.model_name == 'gpt4':
    with jsonlines.open('../processed_data/{}/processed_test_gpt4.jsonl'.format(cfg.lang)) as f:
        for obj in f:
            data.append(obj)
else:
    # Fallback to test.jsonl if processed_test.jsonl missing
    path = '../processed_data/{}/processed_test.jsonl'.format(cfg.lang)
    if not os.path.exists(path):
        path = '../processed_data/{}/test.jsonl'.format(cfg.lang)
    try:
        with jsonlines.open(path) as f:
            for obj in f:
                data.append(obj)
    except FileNotFoundError:
        print(f"Error: Could not find data file at {path}")
        exit(1)


previous_perf = {}
report_path = '../output/{}/cot/top5/test_execution_{}test.report'.format(cfg.lang, cfg.model_name)
if os.path.exists(report_path):
    with jsonlines.open(report_path) as f:
        for obj in f:
            previous_perf[remove_comments_and_docstrings(obj['code_v0_no_empty_lines'],cfg.lang)] = obj

processed_data = []
for obj in data:
    # Use code_v0_no_empty_lines as query if query field is empty or missing
    query_value = obj.get('query', '') or obj.get('code_v0_no_empty_lines', '')
    reference_value = obj.get('reference', '') or obj.get('code_v1_no_empty_lines', '') or obj.get('target', '')
    
    # Use mock values if perf data missing
    key = remove_comments_and_docstrings(query_value, cfg.lang) if query_value else ''
    if key in previous_perf:
        input_time = previous_perf[key]['input_time_mean']
        ref_time = previous_perf[key]['reference_time_mean']
    else:
        input_time = 1.0
        ref_time = 1.0
        
    best_result = {'code':'', 'acc':0, 'time':99999, 'input_time':input_time, 'reference_time':ref_time}
    processed_data.append({'idx':obj['idx'], 'query':query_value, 'reference':reference_value, 'stop':0, 'best_result':best_result, 'best_candidates':[], 'pattern':[]})
   
with jsonlines.open('../output/{}/initial_results_{}.jsonl'.format(cfg.lang, cfg.model_name), 'w') as f:
    f.write_all(processed_data)
