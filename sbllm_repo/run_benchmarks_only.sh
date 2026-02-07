#!/bin/bash
set -e

# =================================================================
# SBLLM Benchmark Execution - Dedicated Script
# =================================================================
# This script runs ONLY the project-based benchmarks:
# 1. RVV-Bench (18 items)
# 2. VecIntrinBench (Google Benchmark suite)
#
# It SKIPS the 993-item Standard RISC-V dataset.
# =================================================================

export PYTHONPATH=/work/sbllm_repo

# --- Configuration ---
BASE_DIR="/work"
BASELINE_RVV="$BASE_DIR/processed_data/rvv-bench/test_full.jsonl"
OUT_DIR="$BASE_DIR/results"
MODEL_NAME="deepseek" 

# --- Setup Directories ---
echo "[*] Setting up benchmark output directories..."
mkdir -p "$OUT_DIR/full_rvv/valid"
mkdir -p "$OUT_DIR/full_rvv_evol"
mkdir -p "$OUT_DIR/vecintrin"

# --- Phase 1: RVV-Bench (18 Items) ---
echo ""
echo "================================================================="
echo "[Phase 1] RVV-Bench: Optimization & Evaluation"
echo "================================================================="
echo "[*] Generating Candidates..."
python3 -m sbllm.evol_query \
    --baseline_data_path "$BASELINE_RVV" \
    --generation_path "$OUT_DIR/full_rvv_evol/" \
    --lang riscv --mode valid \
    --generation_number 1 \
    --iteration 0 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 4

echo "[*] Post-Processing & Evaluating..."
cp "$OUT_DIR/full_rvv_evol/riscv/valid/0/tmp_queries_${MODEL_NAME}test.jsonl" "$OUT_DIR/full_rvv/valid/tmp_queries_${MODEL_NAME}test.jsonl" || true

python3 -m sbllm.evaluate \
    --baseline_data_path "$BASELINE_RVV" \
    --output_path "$OUT_DIR/full_rvv/valid/" \
    --riscv_mode \
    --dataset_type project_rvv \
    --model_name "$MODEL_NAME"

# --- Phase 2: VecIntrinBench (50 Items) ---
echo ""
echo "================================================================="
echo "[Phase 2] VecIntrinBench: Optimization & Evaluation"
echo "================================================================="
echo "[*] Data Preparation..."
python3 prepare_vecintrin_data.py

echo "[*] Generating Optimized Candidates..."
python3 -m sbllm.evol_query \
    --baseline_data_path "$BASE_DIR/processed_data/vecintrin/test_full.jsonl" \
    --generation_path "$OUT_DIR/vecintrin_evol/" \
    --lang cpp --mode valid \
    --generation_number 1 \
    --iteration 0 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 4

echo "[*] Post-Processing & Evaluating..."
mkdir -p "$OUT_DIR/vecintrin/valid"
cp "$OUT_DIR/vecintrin_evol/cpp/valid/0/tmp_queries_${MODEL_NAME}test.jsonl" "$OUT_DIR/vecintrin/valid/tmp_queries_${MODEL_NAME}test.jsonl" || true

python3 -m sbllm.evaluate \
    --baseline_data_path "$BASE_DIR/processed_data/vecintrin/test_full.jsonl" \
    --output_path "$OUT_DIR/vecintrin/valid/" \
    --riscv_mode \
    --dataset_type project_vecintrin \
    --model_name "$MODEL_NAME"

echo ""
echo "================================================================="
echo "All Benchmarks Completed Successfully."
echo "Results available in:"
echo "  - RVV: $OUT_DIR/full_rvv/valid/"
echo "  - VecIntrin: $OUT_DIR/vecintrin/valid/"
echo "================================================================="
