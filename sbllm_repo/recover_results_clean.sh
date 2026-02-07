#!/bin/bash
set -e
export PYTHONPATH=/work/sbllm_repo

# Absolute paths
BASE_DIR="/work"
BASELINE_RVV="$BASE_DIR/processed_data/rvv-bench/test.jsonl"
BASELINE_RISCV="$BASE_DIR/processed_data/riscv/test.jsonl"
OUT_DIR="/work/results"

echo "--- RVV-Bench: Recovering Candidates ---"
python3 -m sbllm.evol_query --baseline_data_path "$BASELINE_RVV" --generation_path "$OUT_DIR/full_rvv_evol/" --lang riscv --mode valid --generation_number 1 --iteration 0 --riscv_mode --model_name mocktest --process_number 4

cp "$OUT_DIR/full_rvv_evol/riscv/valid/0/tmp_queries_mocktest.jsonl" "$OUT_DIR/full_rvv/valid/"

echo "--- Standard RISC-V: Recovering Candidates ---"
python3 -m sbllm.evol_query --baseline_data_path "$BASELINE_RISCV" --generation_path "$OUT_DIR/full_riscv_evol/" --lang riscv --mode valid --generation_number 1 --iteration 0 --riscv_mode --model_name mocktest --process_number 4

cp "$OUT_DIR/full_riscv_evol/riscv/valid/0/tmp_queries_mocktest.jsonl" "$OUT_DIR/full_riscv/valid/"

echo "--- RVV-Bench: Final Evaluation ---"
python3 -m sbllm.evaluate --baseline_data_path "$BASELINE_RVV" --output_path "$OUT_DIR/full_rvv/valid/" --riscv_mode --dataset_type project_rvv --model_name mocktest

echo "--- Standard RISC-V: Final Evaluation ---"
python3 -m sbllm.evaluate --baseline_data_path "$BASELINE_RISCV" --output_path "$OUT_DIR/full_riscv/valid/" --riscv_mode --model_name mocktest

echo "Recovery Completed."
