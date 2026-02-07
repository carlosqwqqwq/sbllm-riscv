#!/bin/bash
set -e
export PYTHONPATH=/work/sbllm_repo

# Absolute paths
BASE_DIR="/work"
BASELINE_RVV="$BASE_DIR/processed_data/rvv-bench/test_full.jsonl"
BASELINE_RISCV="$BASE_DIR/processed_data/riscv/test_full.jsonl"
OUT_DIR="$BASE_DIR/results"

echo "[*] Workspace Synced Status: Results now targeting $OUT_DIR"
