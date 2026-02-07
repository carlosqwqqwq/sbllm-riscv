#!/bin/bash
set -e

# =================================================================
# SBLLM Evaluation Pipeline - Unified Execution Script
# =================================================================
# This script automates the entire process of:
# 1. Setting up the environment
# 2. Generating optimized code candidates for RVV-Bench
# 3. Generating optimized code candidates for Standard RISC-V
# 4. Evaluating all candidates using QEMU
# 5. Generating final performance reports
# =================================================================

export PYTHONPATH=/work/sbllm_repo

# --- Configuration ---
BASE_DIR="/work"
BASELINE_RVV="$BASE_DIR/processed_data/rvv-bench/test_full.jsonl"
OUT_DIR="/work/results"
MODEL_NAME="deepseek"  # Change this if using a different model map in .env

# --- Setup Directories ---
echo "[*] Setting up output directories..."
mkdir -p "$OUT_DIR/full_rvv/valid"
mkdir -p "$OUT_DIR/full_rvv_evol"
mkdir -p "$OUT_DIR/vecintrin/valid"
mkdir -p "$OUT_DIR/vecintrin_evol"

# --- Phase 1: RVV-Bench ---
echo ""
echo "================================================================="
echo "[Phase 1] RVV-Bench: Generating Candidates & Reviews"
echo "================================================================="
python3 -m sbllm.evol_query \
    --baseline_data_path "$BASELINE_RVV" \
    --generation_path "$OUT_DIR/full_rvv_evol/" \
    --lang riscv --mode valid \
    --generation_number 1 \
    --iteration 0 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 4

echo "[*] Copying RVV candidates for evaluation..."
cp "$OUT_DIR/full_rvv_evol/riscv/valid/0/tmp_queries_${MODEL_NAME}test.jsonl" "$OUT_DIR/full_rvv/valid/tmp_queries_${MODEL_NAME}test.jsonl" || true

# --- Phase 2: VecIntrinBench ---
echo ""
echo "================================================================="
echo "[Phase 2] VecIntrinBench: Generating Candidates & Reviews"
echo "================================================================="
python3 prepare_vecintrin_data.py
python3 -m sbllm.evol_query \
    --baseline_data_path "$BASE_DIR/processed_data/vecintrin/test_full.jsonl" \
    --generation_path "$OUT_DIR/vecintrin_evol/" \
    --lang cpp --mode valid \
    --generation_number 1 \
    --iteration 0 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 4

echo "[*] Copying VecIntrin candidates for evaluation..."
cp "$OUT_DIR/vecintrin_evol/cpp/valid/0/tmp_queries_${MODEL_NAME}test.jsonl" "$OUT_DIR/vecintrin/valid/tmp_queries_${MODEL_NAME}test.jsonl" || true

# --- Phase 3: Final Evaluation ---
echo ""
echo "================================================================="
echo "[Phase 3] Final Evaluation (QEMU & Project Handlers)"
echo "================================================================="

echo ">>> Evaluating RVV-Bench..."
python3 -m sbllm.evaluate \
    --mode valid \
    --lang riscv \
    --baseline_data_path "$BASELINE_RVV" \
    --output_path "$OUT_DIR/full_rvv/" \
    --riscv_mode \
    --dataset_type project_rvv \
    --model_name "$MODEL_NAME"

echo ">>> Evaluating VecIntrinBench..."
python3 -m sbllm.evaluate \
    --mode valid \
    --lang cpp \
    --baseline_data_path "$BASE_DIR/processed_data/vecintrin/test_full.jsonl" \
    --output_path "$OUT_DIR/vecintrin/" \
    --riscv_mode \
    --dataset_type project_vecintrin \
    --model_name "$MODEL_NAME"

echo ""
echo "================================================================="
echo "Full Evaluation Suite Completed Successfully!"
echo "Reports are located in:"
echo "  - RVV: $OUT_DIR/full_rvv/valid/"
echo "  - VecIntrin: $OUT_DIR/vecintrin/valid/"
echo "================================================================="
