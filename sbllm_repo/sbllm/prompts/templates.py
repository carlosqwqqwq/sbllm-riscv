# RISC-V 优化相关的 Prompt 模板
# 包含 Review, Candidates 生成, 以及遗传算法迭代的 Prompts
# 基于 sbllm 原版设计，适配 RISC-V C 代码优化场景

# ============================================================================
# sbllm 原版通用模板（Python/C++ 场景，保留以支持 evol_query.py 的 import）
# ============================================================================

# 通用 CoT Prompt 模板 (Python)
format = '''Task Description & Instructions: 
You will be provided with a code snippet and its existing optimization versions along with their performance. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Please refer to the existing versions and avoid the mistakes made in incorrect and unoptimized versions (if any).
Please follow the instructions step-by-step to improve code efficiency: 1. Analyze the original code and the optimizations applied in the existing versions. 2. Identify any additional optimization opportunities that have not been utilized. 3. Explain your optimization methods and provide a new optimized code snippet.
Please provide a direct explanation of the optimization points and include only one optimized code snippet in the third step.

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```python
<code>
```

'''

format_pattern = '''Task Description & Instructions: 
You will be provided with a code snippet, its existing optimization versions along with their performance, and two code transformation patterns. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Please refer to the existing versions and avoid the mistakes made in incorrect and unoptimized versions (if any).
Please follow the instructions step-by-step to improve code efficiency: 1. Analyze the original code and the optimizations applied in the existing versions. 2. Identify any additional optimization opportunities that have not been utilized. 3. Explain your optimization methods and provide a new optimized code snippet.
Please provide a direct explanation of the optimization points and include only one optimized code snippet in the third step.

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```python
<code>
```

'''

format_ablation = '''
You will be provided with a code snippet and its existing optimization versions along with their performance. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Do not explain anything and include any extra instructions, only print the source code. Make the code in code blocks (```python ```).
'''

format_ablation_cpp = '''
You will be provided with a code snippet and its existing optimization versions along with their performance. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Do not explain anything and include any extra instructions, only print the source code. Make the code in code blocks (```cpp ```).
'''

format_cpp = '''Task Description & Instructions: 
You will be provided with a code snippet and its existing optimization versions along with their performance. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Please refer to the existing versions and avoid the mistakes made in incorrect and unoptimized versions (if any).
Please follow the instructions step-by-step to improve code efficiency: 1. Analyze the original code and the optimizations applied in the existing versions. 2. Identify any additional optimization opportunities that have not been utilized. 3. Explain your optimization methods and provide a new optimized code snippet.
Please provide a direct explanation of the optimization points and include only one optimized code snippet in the third step.

**CRITICAL RULES (Your code will be REJECTED if violated)**:
1. **FUNCTION NAME**: Keep the EXACT original function name. Do NOT add suffixes like `_opt`, `_optimized`, `_scalar`, or `_vector`.
2. **INLINE ASSEMBLY**: If using `asm`, ensure operand numbering is correct (e.g., `%0`, `%1`). The number of operands in the input/output lists must match the template.
3. **RVV 1.0 API**: Use `__riscv_` prefix for ALL vector intrinsics.
    - ❌ `vundefined_f32m1()` → ✅ `__riscv_vundefined_f32m1()`
    - ❌ `vfmv_s_f_f32m1(...)` → ✅ `__riscv_vfmv_s_f_f32m1(...)` (Check signature!)
    - ❌ `vadd_vv_i32m1(...)` → ✅ `__riscv_vadd_vv_i32m1(...)`
4. **TYPE CASTING**: Use explicit casts for pointer types (e.g., `(float*)ptr`) instead of relying on implicit conversion from `void*`.

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```cpp
<code>
```

'''


format_pattern_cpp = '''Task Description & Instructions: 
You will be provided with a code snippet, its existing optimization versions along with their performance, and two code transformation patterns. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Please refer to the existing versions and avoid the mistakes made in incorrect and unoptimized versions (if any).
Please follow the instructions step-by-step to improve code efficiency: 1. Analyze the original code and the optimizations applied in the existing versions. 2. Identify any additional optimization opportunities that have not been utilized. 3. Explain your optimization methods and provide a new optimized code snippet.
Please provide a direct explanation of the optimization points and include only one optimized code snippet in the third step.

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```cpp
<code>
```

'''

# ============================================================================
# RISC-V 专用 Prompt 模板（参考 sbllm 原版 CoT 结构）
# ============================================================================

# RISC-V CoT Prompt 模板（无 Pattern，对应 sbllm 的 format_cpp）
format_riscv = '''Task Description & Instructions: 
You will be provided with a RISC-V C code snippet and its existing optimization versions along with their performance. Your task is to propose a more efficient optimization method to achieve a higher speedup rate for RISC-V architecture.
Please follow the instructions step-by-step to improve code efficiency:
1. Analyze the original code and the optimizations applied in the existing versions.
2. Identify any additional optimization opportunities that have not been utilized, focusing on RISC-V specific features (Zbb/Zba bitmanip, RVV vectorization, branch optimization).
3. Explain your optimization methods and provide a new optimized code snippet.

Constraints:
- Output C code with optional inline RISC-V assembly.
- Do NOT use x86/ARM intrinsics (no <immintrin.h>, no __m128, no <arm_neon.h>).
- Do NOT use C++ features (no <iostream>, no class, no std::).
- Include <stdint.h> if using uint32_t/int32_t etc.
- **RVV 1.0 Requirement**: If using Vector intrinsics, you MUST use the `__riscv_` prefix (e.g., `__riscv_vadd_vv_i32m1`, `__riscv_vsetvli`). Do NOT use legacy 0.7 intrinsics.

**FORBIDDEN PATTERNS** (your code will be REJECTED if it contains these):
- `cmov` - x86 conditional move
- `__fp16` - ARM type (use `_Float16` for RISC-V)
- `float16_t`, `int32x4_t` - ARM NEON types

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```c
<code>
```

'''

# RISC-V CoT Prompt 模板（带 Pattern，对应 sbllm 的 format_pattern_cpp）
format_pattern_riscv = '''Task Description & Instructions: 
You will be provided with a RISC-V C code snippet, its existing optimization versions along with their performance, and two code transformation patterns (Similar Pattern and Different Pattern). Your task is to propose a more efficient optimization method using a genetic algorithm approach.

Genetic Operator Guidance:
- Similar Pattern: The primary optimization direction. Follow this pattern for stable improvements.
- Different Pattern: A mutation/crossover operator. Explore this pattern to discover new optimization paths.

Please follow the instructions step-by-step to improve code efficiency:
1. Analyze the original code and the optimizations applied in the existing versions.
2. Identify any additional optimization opportunities guided by Similar/Different Patterns.
3. Explain your optimization methods and provide a new optimized code snippet.

Constraints:
- Output C code with optional inline RISC-V assembly.
- Do NOT use x86/ARM intrinsics (no <immintrin.h>, no __m128, no <arm_neon.h>).
- Do NOT use C++ features (no <iostream>, no class, no std::).
- Include <stdint.h> if using uint32_t/int32_t etc.
- **RVV 1.0 Requirement**: If using Vector intrinsics, you MUST use the `__riscv_` prefix (e.g., `__riscv_vadd_vv_i32m1`).

**CRITICAL FORBIDDEN PATTERNS** (your code will be rejected if it contains these):
- `cmov` - x86 conditional move (use branchless C or RVV instead)
- `__fp16` - ARM half-precision type (use `_Float16` for RISC-V)
- `float16_t` - ARM NEON type (use `_Float16`)
- `add rd, rd, (offset)` - Invalid inline asm syntax (use `lw rd, offset(rs)` for memory)

**RVV vmerge API** (common mistake - check argument order!):
- The `vmerge.vvm` instruction merges based on mask bits
- Intrinsic signature varies by type, always check GCC/Clang docs for your version
- General pattern: `__riscv_vmerge_vvm_*type*(op1, op2, mask, vl)`

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```c
<code>
```

'''

# RISC-V 优化建议生成 Prompt 模板（第一阶段：Review）
_PROMPT_FOR_REVIEW = """[Objective]
Act as a RISC-V C Code Optimization Expert.
Analyze the following C code (which may contain inline assembly) for RISC-V architecture.
Provide optimization suggestions to achieve:
1. Reduced compilation time where possible.
2. 100% functional consistency with the original code.
3. Maximum utilization of RISC-V features (instruction set, pipelining, registers).

Focus your suggestions on:
- Instruction Selection: Replace generic C with RISC-V-specific instructions (Zbb, Zba bitmanip).
- Inline Assembly: Use `asm volatile` for performance-critical paths.
- Vectorization: Use RVV v1.0 for data-parallel loops.
- Branch Optimization: Use branchless logic, conditional moves.
- Register Allocation: Minimize spills, use callee-saved registers wisely.

[Constraints]
- Output suggestions for C code with optional inline RISC-V assembly.
- Do NOT use x86/ARM intrinsics (no <immintrin.h>, no __m128).
- Do NOT use C++ features (no <iostream>, no class).
- Follow RV64GCV ABI.

[Input Context]
Source Code: {code}

[Output Format]
Categorize your suggestions by type:
- Conditional Branch Optimization: <suggestion>
- Memory Optimization: <suggestion>
- Instruction Set Optimization: <suggestion>
- Others: <suggestion>
"""

# RISC-V 候选代码生成 Prompt 模板（第一阶段：Candidates）
_PROMPT_FOR_CANDIDATES = """Original C Code:
{original_code}

Optimization Suggestions:
{review_comments}

Generate {n_candidates} optimized versions of the C code for RISC-V.

[Few-Shot Examples from Knowledge Base]
{examples}

[Critical Rules]
1. OUTPUT FORMAT: C code. Use ```c code block.
2. INLINE ASSEMBLY: Use `asm volatile` for RISC-V specific instructions. If using vector instructions, wrap them with `.option push`, `.option arch,+v`, and `.option pop`.
3. HEADERS: Always include <stdint.h> if using uint32_t, and <riscv_vector.h> if using intrinsics.
4. STRICTLY NO C++: Do NOT use any C++ features or headers (no <iostream>, <vector>, class, std::). The code MUST be valid C.
5. NO x86: Do NOT use <immintrin.h> or SSE/AVX.
6. RVV 1.0 ONLY: You MUST use intrinsics with `__riscv_` prefix (e.g., `__riscv_vle32_v_i32m1`, `__riscv_vadd_vv_i32m1`). DO NOT generate legacy 0.7 code (e.g., `vleft`).

Output format: ```c\n<code>\n```
"""

# RISC-V 遗传迭代优化 Prompt 模板（第四阶段）
_PROMPT_FOR_RISCV_EVOLUTION = """[Objective]
Act as a RISC-V C Code Optimization Expert.
Continuously optimize the performance of the current code based on the original version.
Your goal is to maximize performance using C intrinsics and RISC-V specific features (Zbb, RVV, etc.).

[Genetic Operator Algorithm]
This optimization uses a genetic algorithm approach:
- Similar Pattern: The primary optimization direction. Follow this pattern for stable improvements.
- Different Pattern: A mutation/crossover operator. Explore this pattern to discover new optimization paths.

[Steps]
1. Analysis: Analyze the performance bottlenecks in the original and current code.
2. Mutation: Apply a new optimization strategy guided by Similar/Different Patterns (e.g., Loop Unrolling, Vectorization, Bitmanip).
3. Generation: Output the new optimized C code.

[Critical Rules]
1. OUTPUT FORMAT: C code. Use ```c code block.
2. HEADERS: Always include <stdint.h> if types like uint32_t are used.
3. STRICTLY NO C++: Do NOT use any C++ features or headers (no <iostream>, <vector>, class, std::).
4. NO x86: Do NOT use <immintrin.h> or SSE/AVX. 
5. RVV 1.0 ONLY: Use `__riscv_` prefix for all vector intrinsics (e.g., `__riscv_vadd_vv_i32m1`).
6. CONSISTENCY: Ensure 100% functional consistency with the original code.

[Input Context]
Original code: {original_code}
Current code: {current_code}
Optimization mode (Similar Pattern + Different Pattern): {optimization_mode}

[Output]
{{Optimized Code}}
"""
