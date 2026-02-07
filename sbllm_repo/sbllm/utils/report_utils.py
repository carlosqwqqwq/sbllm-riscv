import os
import logging

logger = logging.getLogger(__name__)

def generate_markdown_report(execution_data, output_path, model_name, failure_stats=None):
    """
    Generates a human-readable Markdown summary of the evaluation results.
    """
    report_file = os.path.join(output_path, f"summary_{model_name}.md")
    
    # metrics calculation
    total_benchmarks = len(execution_data)
    successful_optimizations = 0
    max_speedup = 0.0
    
    # Process successful results
    rows = []
    for item in execution_data:
        desc = item.get('query_desc', 'Unknown')
        
        # Time Metrics
        baseline = item.get('input_time_mean', 99999.0)
        best = item.get('model_generated_potentially_faster_code_col_time_mean', 99999.0)
        
        # Calculate Speedup
        speedup = 1.0
        is_fallback = item.get('is_fallback', False)
        
        if baseline > 0 and 0 < best < 90000:
            speedup = baseline / best
            if speedup > max_speedup and not is_fallback:
                max_speedup = speedup
            # Optimization logic: must differ from baseline and have valid speedup
            if speedup > 1.05 and item.get('model_generated_potentially_faster_code_col_acc') == 1 and not is_fallback:
                successful_optimizations += 1

        # CodeBLEU (Capture LLM's intent even if it failed)
        codebleu_val = item.get('codebleu', 0.0)
        codebleu_score = f"{codebleu_val:.4f}" if codebleu_val > 0 else "-"
        
        # Code Size Comparison
        base_size = item.get('baseline_size', 0)
        opt_size = item.get('size', 0)
        if base_size > 0 and opt_size > 0:
            size_diff = opt_size - base_size
            diff_sign = "+" if size_diff > 0 else ""
            size_str = f"{base_size}B / {opt_size}B ({diff_sign}{size_diff}B)"
        else:
            size_str = f"{opt_size}B" if opt_size > 0 else "-"

        # Status & Labels
        acc = item.get('model_generated_potentially_faster_code_col_acc', 0)
        if is_fallback:
            status = "⏺️ Base" # Fell back to original
        elif acc == 1:
            if speedup > 1.05:
                status = "✅ P-Opt" # Performed Optimization
            else:
                status = "✅ Func" # Functional Pass (No significant speedup)
        else:
            status = "❌ Fail"

        # Format rows
        baseline_str = f"{baseline:.4f}s" if baseline < 90000 else "-"
        best_str = f"{best:.4f}s" if best < 90000 else "-"
        rows.append(f"| {desc} | {baseline_str} | {best_str} | {speedup:.2f}x | {size_str} | {codebleu_score} | {status} |")

    # Failure Analysis Section
    fail_section = ""
    if failure_stats:
        total_failures = sum(failure_stats.values())
        if total_failures > 0:
            fail_section = "## Failure Analysis\n"
            fail_section += f"- **Total Failures**: {total_failures}\n"
            for k, v in failure_stats.items():
                if v > 0:
                    fail_section += f"- **{k}**: {v}\n"
            fail_section += "\n> Note: Check `sbllm.log` for detailed error tracebacks.\n"

    # Executive Summary Construction
    total_attempts = total_benchmarks + (sum(failure_stats.values()) if failure_stats else 0)
    
    md_content = f"""# Evaluation Summary: {model_name}

**Date**: {os.environ.get('DATE', 'N/A')}

## Executive Metrics
- **Total Processed**: {total_attempts}
- **Successful Runs**: {total_benchmarks}
- **Optimized**: {successful_optimizations}
- **Max Speedup**: {max_speedup:.2f}x

{fail_section}
## Detailed Results
"""

    if not rows:
        md_content += "\n*No successful execution data available.*\n"
    else:
        md_content += "| Benchmark | Baseline | Best | Speedup | Size (Base/Opt) | CB (LLM) | Status |\n"
        md_content += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        md_content += "\n".join(rows)

    
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Markdown report generated: {report_file}")
    except Exception as e:
        logger.error(f"Failed to write markdown report: {e}")
