import os
import glob
import jsonlines
import re
import shutil

def extract_scalar_function(file_path, func_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple heuristic to find the start of the function
    # pattern = r"void\s+" + re.escape(func_name) + r"\s*\([^)]*\)\s*\{"
    # This might be too strict with newlines.
    
    # Let's try to just find the function name and assume it starts there.
    # But C parsing is hard.
    # Fallback: Return the whole file content if extraction is too risky.
    # Given the LLM's capability, providing the whole file is often BETTER as it gives context (includes).
    # The existing test.jsonl had just the function, but it had \n\nvoid...
    
    # Let's try to find the region.
    start_idx = content.find(func_name)
    if start_idx == -1:
        return content # Fallback
    
    # Backtrack to find return type "void" or beginning of line
    # Actually, let's just use the whole file. It's safer and gives imports.
    # But wait, we want to replace the scalar function. 
    # If we provide whole file, LLM might rewrite the whole file or just the function.
    # The system prompt usually asks to "optimize this code".
    # If we pass whole file, it might optimize everything.
    # The prompt generator uses "code_v0_no_empty_lines".
    
    # Let's stick to the previous pattern: Extract the scalar function.
    # We can use a simple brace counting if we find the start.
    
    # Find start of function signature
    # Most benign scalar functions in rvv-bench start with 'void \n <name>_scalar' or 'void <name>_scalar'
    match = re.search(r'void\s+' + re.escape(func_name) + r'\s*\([^)]*\)\s*\{', content, re.DOTALL)
    if not match:
        # Try simplified search
        match = re.search(r'void\s+' + re.escape(func_name), content)
        if not match:
             return content
    
    start_pos = match.start()
    # Find the opening brace
    brace_start = content.find('{', start_pos)
    if brace_start == -1:
        return content
        
    # Count braces
    count = 1
    end_pos = -1
    for i in range(brace_start + 1, len(content)):
        if content[i] == '{':
            count += 1
        elif content[i] == '}':
            count -= 1
        
        if count == 0:
            end_pos = i + 1
            break
            
    if end_pos != -1:
        # Include the signature (from start_pos)
        return content[match.start():end_pos]
    
    return content

def prepare_rvv_bench():
    print("Preparing RVV-Bench Full Dataset...")
    bench_dir = os.path.join("rvv-bench", "bench")
    output_path = os.path.join("processed_data", "rvv-bench", "test_full.jsonl")
    
    if not os.path.exists(bench_dir):
        print(f"Error: {bench_dir} not found.")
        return

    data = []
    idx = 0
    
    # Process all .c files
    c_files = glob.glob(os.path.join(bench_dir, "*.c"))
    print(f"Found {len(c_files)} benchmarks.")
    
    for c_file in c_files:
        filename = os.path.basename(c_file)
        bench_name = os.path.splitext(filename)[0]
        scalar_func_name = f"{bench_name}_scalar"
        
        code = extract_scalar_function(c_file, scalar_func_name)
        
        item = {
            "idx": idx,
            "dataset": "rvv-bench",
            "filename": filename,
            "benchmark_name": bench_name,
            "code_v0_no_empty_lines": code,
            "code_v1_no_empty_lines": "", # No reference needed for generation?
            "input": "",
            "lang": "c"
        }
        data.append(item)
        idx += 1
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, 'w') as f:
        f.write_all(data)
    
    print(f"Saved {len(data)} items to {output_path}")

def prepare_riscv_standard():
    print("Preparing Standard RISC-V Full Dataset...")
    src_path = os.path.join("processed_data", "riscv", "train.jsonl")
    dst_path = os.path.join("processed_data", "riscv", "test_full.jsonl")
    
    if not os.path.exists(src_path):
         print(f"Error: {src_path} not found.")
         return
         
    # Just copy it
    # But we might want to ensure 'idx' are unique or consistent? 
    # train.jsonl usually works fine.
    
    # Let's count it
    count = 0
    with jsonlines.open(src_path) as reader:
        for _ in reader:
            count += 1
            
    print(f"Source has {count} items.")
    
    shutil.copy2(src_path, dst_path)
    print(f"Copied to {dst_path}")

if __name__ == "__main__":
    prepare_rvv_bench()
    prepare_riscv_standard()
