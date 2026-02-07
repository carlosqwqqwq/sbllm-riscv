#!/bin/bash
set -e

# =================================================================
# VecIntrinBench Full Evaluation Pipeline
# =================================================================
# 1. Prepare Dataset (Extract scalar C++ code)
# 2. Generate Optimized Candidates (LLM)
# 3. Evaluate Candidates (CMake Build + Google Benchmark)
# =================================================================

export PYTHONPATH=/work/sbllm_repo

# --- Configuration ---
BASE_DIR="/work"
OUT_DIR="$BASE_DIR/results"
MODEL_NAME="deepseek" 
DATASET_PATH="$BASE_DIR/processed_data/vecintrin/test_full.jsonl"

# --- Setup Directories ---
echo "[*] Setting up output directories..."
mkdir -p "$OUT_DIR/vecintrin/valid"
mkdir -p "$OUT_DIR/vecintrin_evol"

# --- Step 1: Dataset Preparation ---
echo ""
echo "================================================================="
echo "[Step 1] Preparing VecIntrinBench Dataset"
echo "================================================================="
python3 prepare_vecintrin_data.py

# --- Step 2: Generation ---
echo ""
echo "================================================================="
echo "[Step 2] Generating Optimized Candidates (LLM)"
echo "================================================================="
python3 -m sbllm.evol_query \
    --baseline_data_path "$DATASET_PATH" \
    --generation_path "$OUT_DIR/vecintrin_evol/" \
    --lang cpp --mode valid \
    --generation_number 1 \
    --iteration 0 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 4

# --- Step 3: Evaluation ---
echo ""
echo "================================================================="
echo "[Step 3] Evaluating Candidates (Project-Based)"
echo "================================================================="
# Copy tmp queries for evaluator
cp "$OUT_DIR/vecintrin_evol/cpp/valid/0/tmp_queries_${MODEL_NAME}test.jsonl" "$OUT_DIR/vecintrin/valid/tmp_queries_${MODEL_NAME}test.jsonl" || true

# Run Project Evaluator
# Note: We use --dataset_type project_vecintrin to trigger the specific logic
python3 -m sbllm.evaluate \
    --baseline_data_path "$DATASET_PATH" \
    --output_path "$OUT_DIR/vecintrin/valid/" \
    --riscv_mode \
    --dataset_type project_vecintrin \
    --model_name "$MODEL_NAME" \
    --qemu_path "/usr/local/bin/qemu-riscv64" \
    --riscv_gcc_toolchain_path "/opt/riscv"

echo ""
echo "================================================================="
echo "VecIntrinBench Optimization Loop Completed."
echo "Results available in: $OUT_DIR/vecintrin/valid/"
echo "================================================================="
