#!/bin/bash
export PYTHONPATH=/work/sbllm_repo
mkdir -p /work/results/full_rvv/valid
mkdir -p /work/results/full_riscv/valid

echo "--- RVV-Bench: Generating Optimized Candidates (Iteration 0) ---"
python3 -m sbllm.evol_query --mode valid --lang riscv --model_name deepseek --generation_path results/full_rvv_evol --baseline_data_path processed_data/rvv-bench/test.jsonl --generation_number 1 --process_number 2 --riscv_mode --iteration 0

# Align paths for evaluation
mkdir -p results/full_rvv/valid
cp results/full_rvv_evol/riscv/valid/0/tmp_queries_deepseektest.jsonl results/full_rvv/valid/

echo "--- RVV-Bench: Evaluating Candidates ---"
python3 evaluate_entry.py --mode valid --lang riscv --dataset_type project_rvv --model_name deepseek --output_path results/full_rvv --qemu_path qemu-riscv64 --riscv_gcc_toolchain_path /opt/riscv --process_number 4 --baseline_data_path processed_data/rvv-bench/test.jsonl

echo "--- Standard RISC-V: Generating Optimized Candidates (Iteration 0) ---"
python3 -m sbllm.evol_query --mode valid --lang riscv --model_name deepseek --generation_path results/full_riscv_evol --baseline_data_path processed_data/riscv/test.jsonl --generation_number 1 --process_number 2 --riscv_mode --iteration 0

# Align paths for evaluation
mkdir -p results/full_riscv/valid
cp results/full_riscv_evol/riscv/valid/0/tmp_queries_deepseektest.jsonl results/full_riscv/valid/

echo "--- Standard RISC-V: Evaluating Candidates ---"
python3 evaluate_entry.py --mode valid --lang riscv --dataset_type standard --model_name deepseek --output_path results/full_riscv --qemu_path qemu-riscv64 --riscv_gcc_toolchain_path /opt/riscv --process_number 4 --baseline_data_path processed_data/riscv/test.jsonl

echo "Full Evaluation Completed."
