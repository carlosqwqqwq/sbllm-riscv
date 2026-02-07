#!/bin/bash
set -e
set -o pipefail

# =================================================================
# Full Benchmark Suite Runner (RVV + VecIntrin)
# =================================================================

# --- Environment Check ---
if [ ! -d "/work" ]; then
    echo "WARNING: Running outside /work (Docker). Ensure variables are set correctly."
    export PYTHONPATH=$PYTHONPATH:$(pwd)
else
    export PYTHONPATH=/work/sbllm_repo
    export SBLLM_LOG_LEVEL=INFO
fi

# --- Configuration ---
# Detect basedir
if [ -d "/work" ]; then
    BASE_DIR="/work"
else
    BASE_DIR=$(pwd)
fi

OUT_DIR="$BASE_DIR/results/full_benchmarks"
BASELINE_RVV="$BASE_DIR/processed_data/rvv-bench/test_full.jsonl"
BASELINE_VECINTRIN="$BASE_DIR/processed_data/vecintrin/test_full.jsonl"
MODEL_NAME="deepseek"

echo "--- Full Benchmark Suite Run Started ---"

# --- Directories ---
echo "[*] Setting up output directories..."
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/rvv/valid"
mkdir -p "$OUT_DIR/rvv_evol"
mkdir -p "$OUT_DIR/vecintrin/valid"
mkdir -p "$OUT_DIR/vecintrin_evol"

# --- Step 0: Data Preparation ---
echo ""
echo "================================================================="
echo "[Step 0] Data Preparation"
echo "================================================================="

# Prepare RVV
if [ -f "sbllm_repo/prepare_rvv_data.py" ]; then
    python3 sbllm_repo/prepare_rvv_data.py
elif [ -f "prepare_rvv_data.py" ]; then
    python3 prepare_rvv_data.py
else
    echo "Warning: prepare_rvv_data.py not found."
fi

# Prepare VecIntrin
if [ -f "sbllm_repo/prepare_vecintrin_data.py" ]; then
    python3 sbllm_repo/prepare_vecintrin_data.py
elif [ -f "prepare_vecintrin_data.py" ]; then
    python3 prepare_vecintrin_data.py
else
    echo "Warning: prepare_vecintrin_data.py not found."
fi

# --- Step 1: Generation ---
echo ""
echo "================================================================="
echo "[Step 1] Generation (Iter 0)"
echo "================================================================="

echo ">>> Generating RVV Candidates..."
python3 -m sbllm.evol_query \
    --baseline_data_path "$BASELINE_RVV" \
    --generation_path "$OUT_DIR/rvv_evol" \
    --lang riscv --mode valid \
    --generation_number 1 \
    --iteration 0 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 4

echo ">>> Generating VecIntrin Candidates..."
python3 -m sbllm.evol_query \
    --baseline_data_path "$BASELINE_VECINTRIN" \
    --generation_path "$OUT_DIR/vecintrin_evol" \
    --lang cpp --mode valid \
    --generation_number 1 \
    --iteration 0 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 4

# Copy Candidates
echo "Copying candidates..."
# Use eval to handle wildcards
cp $OUT_DIR/rvv_evol/riscv/valid/0/tmp_queries_*.jsonl "$OUT_DIR/rvv/valid/tmp_queries_${MODEL_NAME}test.jsonl" || true
cp $OUT_DIR/vecintrin_evol/cpp/valid/0/tmp_queries_*.jsonl "$OUT_DIR/vecintrin/valid/tmp_queries_${MODEL_NAME}test.jsonl" || true

# --- Step 2: Evaluation ---
echo ""
echo "================================================================="
echo "[Step 2] Evaluation"
echo "================================================================="

echo ">>> Evaluating RVV-Bench Full Dataset..."
python3 -m sbllm.evaluate \
    --mode valid --lang riscv \
    --baseline_data_path "$BASELINE_RVV" \
    --output_path "$OUT_DIR/rvv/" \
    --riscv_mode --dataset_type project_rvv \
    --model_name "$MODEL_NAME" --process_number 4

echo ">>> Evaluating VecIntrinBench Full Dataset..."
python3 -m sbllm.evaluate \
    --mode valid --lang cpp \
    --baseline_data_path "$BASELINE_VECINTRIN" \
    --output_path "$OUT_DIR/vecintrin/" \
    --riscv_mode --dataset_type project_vecintrin \
    --model_name "$MODEL_NAME" --process_number 4

echo ""
echo "================================================================="
echo "Full Benchmark Suite Completed."
echo "Results in: $OUT_DIR"
echo "================================================================="
