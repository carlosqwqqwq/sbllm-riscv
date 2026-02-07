#!/usr/bin/env python3
"""
系统性根因分析：测试 Sanitizer 正则匹配行为

问题：前两个候选没有触发 Sanitizer 的重命名日志，但编译时出现 undefined reference 错误。
假设：正则未能正确匹配 LLM 生成的函数定义。
"""
import re

# 测试配置
benchmark_name = 'memcpy'
target_name = 'memcpy_rvv'  # From RESERVED_SYMBOL_MAP

# Sanitizer 正则模式（直接从 code_sanitizer.py 复制）
definition_patterns = [
    fr'\b\w+\s+({benchmark_name}(?:_\w*)?)\s*\(',
    fr'\b\w+\s*\*\s*({benchmark_name}(?:_\w*)?)\s*\(',
    fr'\b\w+\*({benchmark_name}(?:_\w*)?)\s*\(',
]

# 最新一次运行的 LLM 生成代码（从 tmp_queries_deepseektest.jsonl 提取）
test_codes = [
    # Candidate 0
    '''#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

void memcpy_opt(void *restrict dest, const void *restrict src, size_t n) {
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;
}''',
    # Candidate 1
    '''#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

void memcpy_opt(void *restrict dest, const void *restrict src, size_t n) {
    uint8_t *d = (uint8_t *)dest;
}''',
]

print("=" * 60)
print("系统性根因分析：Sanitizer 正则匹配测试")
print("=" * 60)
print(f"benchmark_name: {benchmark_name}")
print(f"target_name: {target_name}")
print()

for i, code in enumerate(test_codes):
    print(f"--- Candidate {i} ---")
    print(f"Code Preview: {code[:100]}...")
    print()
    
    names_to_find = set()
    
    for pattern_str in definition_patterns:
        for match in re.finditer(pattern_str, code, re.MULTILINE):
            found_name = match.group(1)
            print(f"  Pattern: {pattern_str}")
            print(f"  Match: {match.group(0)}")
            print(f"  Captured function name: {found_name}")
            if found_name and found_name != target_name:
                names_to_find.add(found_name)
                print(f"  -> Would add to names_to_find: {found_name}")
    
    if not names_to_find:
        print("  *** PROBLEM: No names_to_find! Sanitizer would NOT rename anything! ***")
    else:
        print(f"  names_to_find = {names_to_find}")
        print(f"  Sanitizer WOULD rename these to: {target_name}")
    
    print()

# 额外测试：Nuclear Option 检查
print("=" * 60)
print("Nuclear Option 测试：检查是否存在 'memcpy' 字符串")
print("=" * 60)
for i, code in enumerate(test_codes):
    if re.search(fr'\b{re.escape(benchmark_name)}\b', code):
        print(f"Candidate {i}: 包含 '{benchmark_name}' 关键字")
    else:
        print(f"Candidate {i}: 不包含 '{benchmark_name}' 关键字 (Nuclear Option 不会触发)")
