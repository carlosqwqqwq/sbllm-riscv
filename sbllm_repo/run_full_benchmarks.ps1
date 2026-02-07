
$ErrorActionPreference = "Stop"

# --- Configuration ---
$BASE_DIR = Get-Location
$PYTHONPATH = "$BASE_DIR;$BASE_DIR\sbllm_repo"
$env:PYTHONPATH = $PYTHONPATH
$env:SBLLM_LOG_LEVEL = "INFO"

# Paths
$BASELINE_RVV = Join-Path $BASE_DIR "processed_data\rvv-bench\test_full.jsonl"
$BASELINE_VECINTRIN = Join-Path $BASE_DIR "processed_data\vecintrin\test_full.jsonl"
$OUT_DIR = Join-Path $BASE_DIR "results\full_benchmarks"
$MODEL_NAME = "deepseek"

Write-Host "--- Full Benchmark Suite Run Started ---" -ForegroundColor Cyan

# --- Directories ---
if (Test-Path $OUT_DIR) { 
    Write-Host "Cleaning output directory..."
    Remove-Item -Recurse -Force $OUT_DIR 
}
New-Item -ItemType Directory -Force -Path "$OUT_DIR\rvv\valid" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT_DIR\rvv_evol" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT_DIR\vecintrin\valid" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT_DIR\vecintrin_evol" | Out-Null

# --- Step 0: Data Preparation ---
Write-Host "Step 0: Data Preparation" -ForegroundColor Green

# Verify/Run RVV Prep
if (Test-Path "sbllm_repo\prepare_rvv_data.py") {
    Write-Host "Running RVV Data Prep..."
    python sbllm_repo\prepare_rvv_data.py
} elseif (Test-Path "prepare_rvv_data.py") {
    python prepare_rvv_data.py
} else {
    Write-Warning "prepare_rvv_data.py not found! Assuming data exists."
}

# Verify/Run VecIntrin Prep
if (Test-Path "sbllm_repo\prepare_vecintrin_data.py") {
    Write-Host "Running VecIntrin Data Prep..."
    python sbllm_repo\prepare_vecintrin_data.py
} elseif (Test-Path "prepare_vecintrin_data.py") {
    python prepare_vecintrin_data.py
} else {
    Write-Warning "prepare_vecintrin_data.py not found! Assuming data exists."
}

# --- Step 1: Generation ---
Write-Host "Step 1: Generation (Iter 0)" -ForegroundColor Green

# RVV
Write-Host "Generating RVV Candidates..."
python -m sbllm.evol_query `
    --baseline_data_path $BASELINE_RVV `
    --generation_path "$OUT_DIR\rvv_evol" `
    --lang riscv --mode valid `
    --generation_number 1 `
    --iteration 0 `
    --riscv_mode `
    --model_name $MODEL_NAME --process_number 4

# VecIntrin
Write-Host "Generating VecIntrin Candidates..."
python -m sbllm.evol_query `
    --baseline_data_path $BASELINE_VECINTRIN `
    --generation_path "$OUT_DIR\vecintrin_evol" `
    --lang cpp --mode valid `
    --generation_number 1 `
    --iteration 0 `
    --riscv_mode `
    --model_name $MODEL_NAME --process_number 4

# Copy Results
Write-Host "Copying candidates..."
Copy-Item "$OUT_DIR\rvv_evol\riscv\valid\0\tmp_queries_*.jsonl" "$OUT_DIR\rvv\valid\tmp_queries_${MODEL_NAME}test.jsonl"
Copy-Item "$OUT_DIR\vecintrin_evol\cpp\valid\0\tmp_queries_*.jsonl" "$OUT_DIR\vecintrin\valid\tmp_queries_${MODEL_NAME}test.jsonl"

# --- Step 2: Evaluation ---
Write-Host "Step 2: Evaluation" -ForegroundColor Green

Write-Host ">>> Evaluating RVV-Bench Full Dataset..."
python -m sbllm.evaluate `
    --mode valid --lang riscv `
    --baseline_data_path $BASELINE_RVV `
    --output_path "$OUT_DIR\rvv" `
    --riscv_mode --dataset_type project_rvv `
    --use_docker `
    --model_name $MODEL_NAME --process_number 4

Write-Host ">>> Evaluating VecIntrinBench Full Dataset..."
python -m sbllm.evaluate `
    --mode valid --lang cpp `
    --baseline_data_path $BASELINE_VECINTRIN `
    --output_path "$OUT_DIR\vecintrin" `
    --riscv_mode --dataset_type project_vecintrin `
    --use_docker `
    --model_name $MODEL_NAME --process_number 4

Write-Host "FULL BENCHMARK SUITE COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "Results in: $OUT_DIR"
