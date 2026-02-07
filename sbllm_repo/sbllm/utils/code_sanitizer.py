import re
import logging

logger = logging.getLogger(__name__)

class CodeSanitizer:
    @staticmethod
    def sanitize(code: str, benchmark_name: str) -> str:
        """
        Sanitize and auto-correct LLM generated code.
        1. Replace old RVV API calls with new RVV 1.0 API calls.
        2. Force function renaming to match benchmark_name.
        """
        sanitized_code = code

        # 1. RVV API Auto-Correction
        # Mapping of common old/wrong APIs to valid RVV 1.0 APIs
        # 1. RVV API Auto-Correction
        # Mapping of common old/wrong APIs to valid RVV 1.0 APIs
        api_replacements = [
            (r'\bvundefined_f32m1\s*\(', '__riscv_vundefined_f32m1('),
            (r'\bvfmv_s_f_f32m1\s*\(', '__riscv_vfmv_s_f_f32m1('),
            (r'\bvfmv_v_f_f32m1\s*\(', '__riscv_vfmv_v_f_f32m1('),
            (r'\bvadd_vv_f32m1\s*\(', '__riscv_vadd_vv_f32m1('),
            (r'\bvsub_vv_f32m1\s*\(', '__riscv_vsub_vv_f32m1('),
            (r'\bvmul_vv_f32m1\s*\(', '__riscv_vmul_vv_f32m1('),
            (r'\bvmacc_vv_f32m1\s*\(', '__riscv_vmacc_vv_f32m1('),
        ]
        
        for old, new in api_replacements:
             if re.search(old, sanitized_code):
                 sanitized_code = re.sub(old, new, sanitized_code)
                 logger.debug(f"[Sanitizer] Replaced usage of old API pattern: {old} -> {new}")

        # 2. Function Name Enforcement (Robust Rewrite)
        # Goal: Rename any function like `benchmark_name_*` to `target_name`
        from sbllm.utils.harness_standalone import RESERVED_SYMBOL_MAP
        target_name = RESERVED_SYMBOL_MAP.get(benchmark_name, benchmark_name)
        
        # Stage 2a: Build list of candidate function names to look for
        # LLM often generates: memcpy_opt, memcpy_optimized, memcpy_rvv_impl, etc.
        names_to_find = set()
        
        # Pattern 1: Function DEFINITION with standard C/C++ signature
        # Matches: `<type> <name>(` or `<type>* <name>(` or `<type> *<name>(`
        # Capture group 1: the function name
        definition_patterns = [
            # Most common: `void memcpy_opt(` or `int memcpy_opt(`
            fr'\b\w+\s+({benchmark_name}(?:_\w*)?)\s*\(',
            # Pointer return: `void* memcpy_opt(` or `char * memcpy_opt(`
            fr'\b\w+\s*\*\s*({benchmark_name}(?:_\w*)?)\s*\(',
            # Tight: `void*memcpy_opt(` (no space before name)
            fr'\b\w+\*({benchmark_name}(?:_\w*)?)\s*\(',
        ]
        
        for pattern_str in definition_patterns:
            for match in re.finditer(pattern_str, sanitized_code, re.MULTILINE):
                found_name = match.group(1)
                if found_name and found_name != target_name:
                    names_to_find.add(found_name)
        
        # Stage 2b: Perform replacements for all found names
        for wrong_name in names_to_find:
            logger.warning(f"[Sanitizer] Found naming mismatch: '{wrong_name}'. Forcing rename to '{target_name}'.")
            # Use word boundary to replace all occurrences (definition AND calls)
            sanitized_code = re.sub(fr'\b{re.escape(wrong_name)}\b', target_name, sanitized_code)
        
        # Stage 2c: Nuclear Option for Reserved Symbols
        # If target_name differs from benchmark_name (e.g., memcpy -> memcpy_rvv),
        # ensure the original name is completely eradicated.
        if target_name != benchmark_name:
            if re.search(fr'\b{re.escape(benchmark_name)}\b', sanitized_code):
                logger.warning(f"[Sanitizer] Reserved symbol '{benchmark_name}' still detected. Performing global replacement to '{target_name}'.")
                sanitized_code = re.sub(fr'\b{re.escape(benchmark_name)}\b', target_name, sanitized_code)
        
        # Stage 2d: Post-Verification
        # Check if target_name now exists in the code. If not, log a critical warning.
        if not re.search(fr'\b{re.escape(target_name)}\s*\(', sanitized_code):
            logger.error(f"[Sanitizer] CRITICAL: After sanitization, target function '{target_name}' not found in code! Code may fail to link.")


        # 3. Signature Fixes (Specific Benchmarks)
        if benchmark_name == 'memcpy':
            # --- Fix 3a: Remove duplicate function definitions ---
            # LLM may generate multiple function variants (memcpy_optimized, memcpy_vector, etc.)
            # All get renamed to target_name, causing "redefinition" errors.
            # Strategy: Keep only the FIRST function definition, delete the rest.
            func_def_pattern = fr'(void\s+{re.escape(target_name)}\s*\([^)]*\)\s*\{{)'
            all_matches = list(re.finditer(func_def_pattern, sanitized_code))
            if len(all_matches) > 1:
                logger.warning(f"[Sanitizer] Detected {len(all_matches)} definitions of '{target_name}'. Removing duplicates.")
                # Find the position of the first definition's opening brace
                first_def_end = all_matches[0].end()
                # For each subsequent definition, find and remove the entire function body
                for match in reversed(all_matches[1:]):  # Reverse to avoid index shifting
                    start_pos = match.start()
                    # Find the matching closing brace by counting braces
                    brace_count = 1
                    end_pos = match.end()
                    while brace_count > 0 and end_pos < len(sanitized_code):
                        if sanitized_code[end_pos] == '{':
                            brace_count += 1
                        elif sanitized_code[end_pos] == '}':
                            brace_count -= 1
                        end_pos += 1
                    # Remove the entire function (including any preceding comments/whitespace)
                    # Find start of line containing the function
                    line_start = sanitized_code.rfind('\n', 0, start_pos)
                    if line_start == -1:
                        line_start = 0
                    else:
                        line_start += 1
                    logger.info(f"[Sanitizer] Removing duplicate function at position {line_start}-{end_pos}")
                    sanitized_code = sanitized_code[:line_start] + sanitized_code[end_pos:]
            
            # --- Fix 3b: RVV Signed/Unsigned Intrinsic Type Mismatch ---
            # LLM sometimes uses signed intrinsics (vint8m8_t, __riscv_vle8_v_i8m8) with uint8_t* pointers.
            # For memcpy, we should use unsigned types throughout.
            rvv_type_fixes = [
                (r'\bvint8m8_t\b', 'vuint8m8_t'),
                (r'__riscv_vle8_v_i8m8\b', '__riscv_vle8_v_u8m8'),
                (r'__riscv_vse8_v_i8m8\b', '__riscv_vse8_v_u8m8'),
            ]
            for pattern, repl in rvv_type_fixes:
                if re.search(pattern, sanitized_code):
                    logger.info(f"[Sanitizer] Fixing RVV signed/unsigned mismatch: {pattern} -> {repl}")
                    sanitized_code = re.sub(pattern, repl, sanitized_code)

            # --- Fix 3c: char* vs uint8_t* pointer type mismatch ---
            # LLM often uses char* while our harness needs uint8_t* for RVV
            # Also handle C++ strictness for implicit void* to char* conversion
            char_replacements = [
                (r'\bchar\s*\*\s*(\w+)\s*=\s*\((?:char\s*\*)\)dest', r'uint8_t* \1 = (uint8_t*)dest'),
                (r'\bconst\s+char\s*\*\s*(\w+)\s*=\s*\(const\s+char\s*\*\)src', r'const uint8_t* \1 = (const uint8_t*)src'),
                (r'\bchar\s*\*\s*(\w+)\s*=\s*dest\b', r'uint8_t* \1 = (uint8_t*)dest'),
                (r'\bconst\s+char\s*\*\s*(\w+)\s*=\s*src\b', r'const uint8_t* \1 = (const uint8_t*)src'),
                (r'\buint8_t\s*\*\s*(\w+)\s*=\s*dest\b', r'uint8_t* \1 = (uint8_t*)dest'),
                (r'\bconst\s+uint8_t\s*\*\s*(\w+)\s*=\s*src\b', r'const uint8_t* \1 = (const uint8_t*)src'),
            ]
            for pattern, repl in char_replacements:
                if re.search(pattern, sanitized_code):
                    logger.info(f"[Sanitizer] Fixing memcpy type mismatch: {pattern} -> {repl}")
                    sanitized_code = re.sub(pattern, repl, sanitized_code)

                    
        if benchmark_name == 'eltwise':
            # Fix mismatch between harness (vector<float*>) and typical LLM output (float**)
            # LLM: void eltwise(float** data_vec, ...) OR void eltwise(float* data_vec[], ...)
            # Harness: void eltwise(std::vector<float*>& data_vec, ...)
            
            # Pattern 1: float**
            if re.search(r'void\s+eltwise\s*\(\s*float\s*\*\*', sanitized_code):
                 logger.info(f"[Sanitizer] Fixing eltwise signature mismatch (float** -> vector<float*>&)")
                 sanitized_code = re.sub(r'void\s+eltwise\s*\(\s*float\s*\*\*', 
                                         r'void eltwise(std::vector<float*>& ', sanitized_code)
            
            # Pattern 2: float* var[]
            # CAUTION: Regex must be robust to spaces and variable names. 
            # We look for "void eltwise(float* <OPTIONAL_RESTRICT> <VAR>[]"
            # Simplified: match "void eltwise(float* ... []" up to the comma
            # Replaces with "void eltwise(std::vector<float*>& <VAR>"
            
            # Matches: void eltwise(float* __restrict data_vec[], 
            # Capture group 1: keywords/modifiers + variable name
            match = re.search(r'void\s+eltwise\s*\(\s*float\s*\*\s*([\w\s]+)\[\]', sanitized_code)
            if match:
                 var_part = match.group(1).strip()
                 # Remove __restrict if present to avoid syntax issues with vector reference? 
                 # std::vector<float*>& __restrict var is valid? Reference itself cannot be restrict.
                 # Safest is to remove modifiers.
                 var_name = var_part.split()[-1]
                 logger.info(f"[Sanitizer] Fixing eltwise signature mismatch (float* [] -> vector<float*>&)")
                 sanitized_code = re.sub(r'void\s+eltwise\s*\(\s*float\s*\*\s*([\w\s]+)\[\]', 
                                         f'void eltwise(std::vector<float*>& {var_name}', sanitized_code)

        return sanitized_code
