
import logging
from sbllm.prompts.builder import build_evolution_prompt

# Setup Mock Configuration
class MockConfig:
    def __init__(self):
        self.beam_number = 3
        self.riscv_mode = True
        self.lang = 'riscv'
        self.mode = 'valid'

cfg = MockConfig()

# Setup Mock Query
query = {
    'query': 'void foo() {}',
    'id': 'test_1',
    'idx': 10001
}

# Setup Mock Previous Result (Simulating execution.py output)
previous_result_obj = {
    'query_idx': 10001,
    'model_generated_potentially_faster_code_col_0': 'void foo_opt() { error }',
    'model_generated_potentially_faster_code_col_0_acc': 0,
    'model_generated_potentially_faster_code_col_0_time_mean': 99999,
    'model_generated_potentially_faster_code_col_0_error': 'gcc: error: unknown type name vector_t',
    
    'model_generated_potentially_faster_code_col_1': 'void foo_opt2() { works }',
    'model_generated_potentially_faster_code_col_1_acc': 1,
    'model_generated_potentially_faster_code_col_1_time_mean': 100,
    'input_time_mean': 200, # 2x speedup
}

# Run Builder
try:
    messages = build_evolution_prompt(cfg, query, previous_result_obj)
    prompt_content = messages[0]['content']
    
    print("--- Generated Prompt ---")
    print(prompt_content[:1000]) # Print first 1000 chars
    print("------------------------")
    
    if "Error Feedback: gcc: error: unknown type name vector_t" in prompt_content:
        print("SUCCESS: Error feedback found in prompt.")
    else:
        print("FAILURE: Error feedback NOT found in prompt.")

except Exception as e:
    print(f"FAILURE: Exception occurred: {e}")
