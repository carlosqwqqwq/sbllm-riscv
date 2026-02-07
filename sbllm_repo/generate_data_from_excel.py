import pandas as pd
import jsonlines
import os
import re

def extract_original_code(description):
    """
    Heuristic to extract 'original code' from description column.
    Common pattern: '优化前：[CODE] 优化后：'
    """
    if not isinstance(description, str):
        return ""
    
    # Try to find content between "优化前：" and "优化后："
    match = re.search(r'优化前：(.*?)(?=优化后：|$)', description, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""

def generate_train_data(excel_path='数据集收集.xlsx', output_path='processed_data/riscv/train.jsonl'):
    """
    Reads data from Excel and generates a JSONL file for training.
    Uses corrected mapping for '数据集收集.xlsx'.
    """
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        return

    try:
        # Load Excel with header (default)
        df = pd.read_excel(excel_path)
        
        print(f"Loaded {len(df)} rows from {excel_path}")
        print(f"Columns found: {df.columns.tolist()}")
        
        data = []
        for index, row in df.iterrows():
            # Correct Mapping:
            # 序号 -> id
            # 描述 -> optimization_description (and contains original code snippet)
            # 关键修改 -> optimized_code
            # 优化类型 -> optimization_type
            # 链接 -> source_url
            
            # Extract codes
            optimized_code = str(row.get('关键修改', '')).strip() if not pd.isna(row.get('关键修改')) else ""
            original_code = extract_original_code(row.get('描述', ''))
            
            # Fallback for original code if extraction fails
            if not original_code:
                 # If we have optimized code but no original, 
                 # it might be a pure addition or we just can't parse it.
                 # For training, having a base is better.
                 original_code = "// No original code extracted from description\n"
            
            item = {
                "id": row.get('序号', index),
                "source": "riscv-dataset-excel",
                "optimization_type": str(row.get('优化类型', 'General Optimization')),
                "optimization_description": str(row.get('描述', '')),
                "original_code": original_code,
                "optimized_code": optimized_code,
                "source_url": str(row.get('链接', '')),
                
                # Evaluation pipeline requirements (compat)
                "code_v0_no_empty_lines": original_code,
                "code_v1_no_empty_lines": optimized_code,
                "target": optimized_code,
                "input": "", # No stdin input for these snippets
                
                # Placeholders
                "query_abs": "", 
                "edit_code_abs": "",
                "edit_opt_abs": "",
                "text_representation": str(row.get('描述', ''))
            }
            
            # Minimal validation: must have some code
            if optimized_code == "nan" or not optimized_code:
                continue
                
            data.append(item)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with jsonlines.open(output_path, 'w') as writer:
            writer.write_all(data)
            
        print(f"Successfully wrote {len(data)} items to {output_path}")
        if data:
            print(f"Sample Item 0 ID: {data[0]['id']}")
            print(f"Sample Original Code Snippet: {data[0]['original_code'][:50]}...")
            print(f"Sample Optimized Code Snippet: {data[0]['optimized_code'][:50]}...")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    generate_train_data()
