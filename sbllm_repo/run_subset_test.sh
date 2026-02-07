#!/bin/bash
set -e
set -o pipefail

# --- Environment Check ---
if [ ! -d "/work" ]; then
    echo "ERROR: This script must be run INSIDE the Docker container (riscv-opt-env)."
    echo "Please use the host-side bootstrap script instead:"
    echo "  PowerShell: .\run_subset_test.ps1"
    exit 1
fi

# --- Logging Setup ---
LOG_DIR="/work/sbllm_repo/results"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/pipeline_run.log"

# Clear previous log
echo "--- Pipeline Run Started at $(date) ---" > "$MAIN_LOG"

# Helper: Print header
print_header() {
    local title="$1"
    echo ""
    echo -e "\033[1;34m=================================================================\033[0m"
    echo -e "\033[1;32m [$(date +'%Y-%m-%d %H:%M:%S')] $title\033[0m"
    echo -e "\033[1;34m=================================================================\033[0m"
}

# Redirect all following output to log file while keeping it in terminal
exec > >(tee -a "$MAIN_LOG") 2>&1

export PYTHONPATH=/work/sbllm_repo
export SBLLM_LOG_LEVEL=DEBUG
# Force all python subprocesses to log to the same file
export SBLLM_LOG_FILE="$MAIN_LOG"

# --- Configuration ---
BASE_DIR="/work"
BASELINE_RVV="$BASE_DIR/processed_data/rvv-bench/test_subset.jsonl"
BASELINE_VECINTRIN="$BASE_DIR/processed_data/vecintrin/test_subset.jsonl"
OUT_DIR="/work/sbllm_repo/results/subset_test"
MODEL_NAME="deepseek" 
export LLM_INTERACTION_LOG="/work/llm_interaction.log"

# --- Setup Directories ---
echo "[*] Cleaning and preparing directories..."
# Clean output directory (Log is now safe in root)
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/rvv/valid"
mkdir -p "$OUT_DIR/rvv_evol"
mkdir -p "$OUT_DIR/vecintrin/valid"
mkdir -p "$OUT_DIR/vecintrin_evol"

# --- Phase 1: Initial Generation (Iteration 0) with RAG ---
print_header "Phase 1: Iteration 0 - Initial Generation (RAG Enabled)"
# RVV Iteration 0
python3 -m sbllm.evol_query \
    --baseline_data_path "$BASELINE_RVV" \
    --generation_path "$OUT_DIR/rvv_evol" \
    --lang riscv --mode valid \
    --generation_number 1 \
    --iteration 0 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 8

# VecIntrin Iteration 0
python3 -m sbllm.evol_query \
    --baseline_data_path "$BASELINE_VECINTRIN" \
    --generation_path "$OUT_DIR/vecintrin_evol" \
    --lang cpp --mode valid \
    --generation_number 1 \
    --iteration 0 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 8

echo "[*] Copying candidates for evaluation..."
# Use dynamic matching since evol_query might add suffixes or IDs
# Note: Remove quotes to allow shell expansion
cp $OUT_DIR/rvv_evol/riscv/valid/0/tmp_queries_*.jsonl $OUT_DIR/rvv/valid/tmp_queries_${MODEL_NAME}test.jsonl || echo "RVV copy failed"
cp $OUT_DIR/vecintrin_evol/cpp/valid/0/tmp_queries_*.jsonl $OUT_DIR/vecintrin/valid/tmp_queries_${MODEL_NAME}test.jsonl || echo "VecIntrin copy failed"

# --- Phase 2: Evaluation vs GCC -O3 ---
print_header "Phase 2: Evaluation - Capturing Errors & Speedup"
echo ">>> Evaluating RVV-Bench Subset..."
python3 -m sbllm.evaluate \
    --mode valid --lang riscv \
    --baseline_data_path "$BASELINE_RVV" \
    --output_path "$OUT_DIR/rvv/" \
    --riscv_mode --dataset_type project_rvv \
    --model_name "$MODEL_NAME" --process_number 8

echo ">>> Evaluating VecIntrinBench Subset..."
python3 -m sbllm.evaluate \
    --mode valid --lang cpp \
    --baseline_data_path "$BASELINE_VECINTRIN" \
    --output_path "$OUT_DIR/vecintrin/" \
    --riscv_mode --dataset_type project_vecintrin \
    --model_name "$MODEL_NAME" --process_number 8

# --- Phase 3: Feedback Loop (Iteration 1) ---
print_header "Phase 3: Feedback Loop (Self-Correction)"
# Link results for feedback
echo "[*] Linking Evaluation Reports to Evolution Path..."
mkdir -p "$OUT_DIR/rvv_evol/riscv/valid"
mkdir -p "$OUT_DIR/vecintrin_evol/cpp/valid"
cp "$OUT_DIR/rvv/valid/test_execution_${MODEL_NAME}test.report" "$OUT_DIR/rvv_evol/riscv/valid/results.jsonl" || true
cp "$OUT_DIR/vecintrin/valid/test_execution_${MODEL_NAME}test.report" "$OUT_DIR/vecintrin_evol/cpp/valid/results.jsonl" || true

# Run Iteration 1 for RVV (Verification of Feedback)
python3 -m sbllm.evol_query \
    --baseline_data_path "$BASELINE_RVV" \
    --generation_path "$OUT_DIR/rvv_evol/" \
    --lang riscv --mode valid \
    --generation_number 1 \
    --iteration 1 \
    --riscv_mode \
    --model_name "$MODEL_NAME" --process_number 8

print_header "Subset Test Suite Completed!"
echo "[*] Main Log: $MAIN_LOG"
echo "[*] Interaction Log: $LLM_INTERACTION_LOG"
