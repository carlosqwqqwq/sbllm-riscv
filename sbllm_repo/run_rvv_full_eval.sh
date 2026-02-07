#!/bin/bash
set -e

# =================================================================
# RVV-Bench Full Evaluation Pipeline
# =================================================================
# 1. Prepare Dataset (Inline headers into C code)
# 2. Generate Optimized Candidates (LLM)
# 3. Evaluate Candidates (QEMU + GCC)
# =================================================================

export PYTHONPATH=/work/sbllm_repo

# --- Configuration ---
BASE_DIR="/work"
OUT_DIR="$BASE_DIR/results"
MODEL_NAME="deepseek" 
DATASET_PATH="$BASE_DIR/processed_data/rvv-bench/test_full.jsonl"

# --- Setup Directories ---
echo "[*] Setting up output directories..."
mkdir -p "$OUT_DIR/rvv-bench/valid"
mkdir -p "$OUT_DIR/rvv_bench_evol"

# --- Step 1: Dataset Preparation ---
echo ""
echo "================================================================="
echo "[Step 1] Preparing RVV-Bench Dataset"
echo "================================================================="
python3 prepare_rvv_data.py

# --- Step 2: Generation ---
echo ""
echo "================================================================="
echo "[Step 2] Generating Optimized Candidates (LLM)"
echo "================================================================="
python3 -m sbllm.evol_query \
    --baseline_data_path "$DATASET_PATH" \
    --generation_path "$OUT_DIR/rvv_bench_evol/" \
    --lang c --mode valid \
    --generation_number 1 \
    --iteration 0 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 4

# --- Step 3: Evaluation ---
echo ""
echo "================================================================="
echo "[Step 3] Evaluating Candidates (Standard QEMU Evaluator)"
echo "================================================================="
# Copy tmp queries for evaluator
cp "$OUT_DIR/rvv_bench_evol/c/valid/0/tmp_queries_${MODEL_NAME}test.jsonl" "$OUT_DIR/rvv-bench/valid/tmp_queries_${MODEL_NAME}test.jsonl" || true

# Run Standard Evaluator (Implicitly uses QEMURISCVEvaluator via riscv_mode)
# We do NOT use --dataset_type project_rvv because that triggers StandaloneEvaluator
python3 -m sbllm.evaluate \
    --baseline_data_path "$DATASET_PATH" \
    --output_path "$OUT_DIR/rvv-bench/valid/" \
    --riscv_mode \
    --model_name "$MODEL_NAME" \
    --qemu_path "/usr/local/bin/qemu-riscv64" \
    --riscv_gcc_toolchain_path "/opt/riscv"

echo ""
echo "================================================================="
echo "RVV-Bench Optimization Loop Completed."
echo "Results available in: $OUT_DIR/rvv-bench/valid/"
echo "================================================================="
