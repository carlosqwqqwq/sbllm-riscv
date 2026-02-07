import os
import glob
import jsonlines

def prepare_vecintrin_dataset():
    print("Preparing VecIntrinBench Dataset...")
    
    # Paths
    base_dir = "VecIntrinBench"
    scalar_dir = os.path.join(base_dir, "src", "scalar")
    output_path = os.path.join("processed_data", "vecintrin", "test_full.jsonl")
    
    if not os.path.exists(scalar_dir):
        print(f"Error: {scalar_dir} not found.")
        return

    data = []
    idx = 0
    
    # Process all .cpp files in src/scalar
    cpp_files = glob.glob(os.path.join(scalar_dir, "*.cpp"))
    print(f"Found {len(cpp_files)} benchmarks in {scalar_dir}")
    
    for cpp_file in cpp_files:
        filename = os.path.basename(cpp_file)
        # Benchmark name is filename without extension? 
        # Actually Google Benchmark filters by function name, but for file replacement we need filename.
        bench_name = os.path.splitext(filename)[0]
        
        with open(cpp_file, 'r', encoding='utf-8') as f:
            code = f.read()
            
        item = {
            "idx": idx,
            "dataset": "vecintrin",
            "filename": filename,
            "benchmark_name": bench_name,
            "code_v0_no_empty_lines": code,
            "code_v1_no_empty_lines": "", # No reference needed
            "input": "",
            "lang": "cpp"
        }
        data.append(item)
        idx += 1
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, 'w') as f:
        f.write_all(data)
    
    print(f"Saved {len(data)} items to {output_path}")

if __name__ == "__main__":
    prepare_vecintrin_dataset()
