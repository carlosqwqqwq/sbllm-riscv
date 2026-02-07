# Evaluation Summary: deepseek

**Date**: N/A

## Executive Metrics
- **Total Processed**: 3
- **Successful Runs**: 3
- **Optimized**: 2
- **Max Speedup**: 1.86x


## Detailed Results
| Benchmark | Baseline | Best | Speedup | Size (Base/Opt) | CB (LLM) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| absval | 0.0001s | 0.0002s | 0.76x | 1210919B / 1210931B (+12B) | 0.5144 | ✅ Func |
| innerproduct | 0.0003s | 0.0002s | 1.86x | 1211107B / 1211279B (+172B) | 0.4745 | ✅ P-Opt |
| eltwise | 0.0028s | 0.0021s | 1.34x | 1211763B / 1211711B (-52B) | 0.6246 | ✅ P-Opt |