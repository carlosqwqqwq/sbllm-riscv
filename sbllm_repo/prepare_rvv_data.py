import os
import glob
import jsonlines

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def prepare_rvv_dataset():
    print("Preparing RVV Bench Dataset...")
    
    # Paths
    base_dir = "sbllm_repo/rvv-bench"
    bench_dir = os.path.join(base_dir, "bench")
    output_path = os.path.join("processed_data", "rvv-bench", "test_full.jsonl")
    
    # Headers
    config_h = os.path.join(bench_dir, "config.h")
    nolibc_h = os.path.join(base_dir, "nolibc.h")
    bench_h = os.path.join(bench_dir, "bench.h")
    
    if not os.path.exists(bench_dir):
        print(f"Error: {bench_dir} not found.")
        return

    # Read Headers
    try:
        config_content = read_file(config_h)
        nolibc_content = read_file(nolibc_h)
        bench_header_content = read_file(bench_h)
    except FileNotFoundError as e:
        print(f"Error reading headers: {e}")
        return

    # Inline headers into bench.h content
    # Replace #include "config.h"
    bench_header_content = bench_header_content.replace('#include "config.h"', config_content)
    # Replace #include "../nolibc.h"
    bench_header_content = bench_header_content.replace('#include "../nolibc.h"', nolibc_content)
    
    # Add generated_impl.h stub if needed, or just dummy it out
    # Some files use #include "generated_impl.h"
    generated_impl_stub = "void* memcpy_model_generated(void *dest, const void *src, size_t n) { return dest; }"
    
    data = []
    idx = 0
    
    # Process all .c files in bench/
    c_files = glob.glob(os.path.join(bench_dir, "*.c"))
    print(f"Found {len(c_files)} benchmarks in {bench_dir}")
    
    for c_file in c_files:
        filename = os.path.basename(c_file)
        bench_name = os.path.splitext(filename)[0]
        
        # Skip generated_impl.c if it exists? No, it's .h
        
        code = read_file(c_file)
        
        # Inline bench.h
        if '#include "bench.h"' in code:
            code = code.replace('#include "bench.h"', bench_header_content)
        
        # Inline generated_impl.h if present
        if '#include "generated_impl.h"' in code:
             code = code.replace('#include "generated_impl.h"', generated_impl_stub)
            
        item = {
            "idx": idx,
            "dataset": "rvv-bench",
            "filename": filename,
            "benchmark_name": bench_name,
            "code_v0_no_empty_lines": code,
            "code_v1_no_empty_lines": "", 
            "input": "",
            "lang": "c"
        }
        data.append(item)
        idx += 1
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, 'w') as f:
        f.write_all(data)
    
    print(f"Saved {len(data)} items to {output_path}")

if __name__ == "__main__":
    prepare_rvv_dataset()
