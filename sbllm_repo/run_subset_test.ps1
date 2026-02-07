
$ErrorActionPreference = "Stop"

# --- Configuration ---
$BASE_DIR = Get-Location
$PYTHONPATH = $BASE_DIR
$env:PYTHONPATH = $PYTHONPATH
$env:SBLLM_LOG_LEVEL = "DEBUG"

$BASELINE_RVV = Join-Path $BASE_DIR "processed_data\rvv-bench\test_subset.jsonl"
$BASELINE_VECINTRIN = Join-Path $BASE_DIR "processed_data\vecintrin\test_subset.jsonl"
$OUT_DIR = Join-Path $BASE_DIR "results\subset_test"
$MODEL_NAME = "deepseek"

Write-Host "--- Pipeline Run Started ---" -ForegroundColor Cyan

# --- Directories ---
if (Test-Path $OUT_DIR) { Remove-Item -Recurse -Force $OUT_DIR }
New-Item -ItemType Directory -Force -Path "$OUT_DIR\rvv\valid" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT_DIR\rvv_evol" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT_DIR\vecintrin\valid" | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT_DIR\vecintrin_evol" | Out-Null

# --- Phase 1: Iteration 0 ---
Write-Host "Phase 1: Iteration 0 (Generation)" -ForegroundColor Green

# RVV
python -m sbllm.evol_query `
    --baseline_data_path $BASELINE_RVV `
    --generation_path "$OUT_DIR\rvv_evol" `
    --lang riscv --mode valid `
    --generation_number 1 `
    --iteration 0 `
    --riscv_mode `
    --model_name $MODEL_NAME --process_number 4

# VecIntrin
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

# --- Phase 2: Evaluation ---
Write-Host "Phase 2: Evaluation (Single-File Mode Verification)" -ForegroundColor Green

Write-Host ">>> Evaluating RVV-Bench Subset..."
python -m sbllm.evaluate `
    --mode valid --lang riscv `
    --baseline_data_path $BASELINE_RVV `
    --output_path "$OUT_DIR\rvv" `
    --riscv_mode --dataset_type project_rvv `
    --use_docker `
    --model_name $MODEL_NAME --process_number 4

Write-Host ">>> Evaluating VecIntrinBench Subset..."
python -m sbllm.evaluate `
    --mode valid --lang cpp `
    --baseline_data_path $BASELINE_VECINTRIN `
    --output_path "$OUT_DIR\vecintrin" `
    --riscv_mode --dataset_type project_vecintrin `
    --use_docker `
    --model_name $MODEL_NAME --process_number 4

Write-Host "Phase 2 Complete. Checking Reports..."

# Check if reports exist
if (-not (Test-Path "$OUT_DIR\rvv\valid\test_execution_${MODEL_NAME}test.report")) {
    Write-Error "RVV Report missing!"
}
if (-not (Test-Path "$OUT_DIR\vecintrin\valid\test_execution_${MODEL_NAME}test.report")) {
    Write-Error "VecIntrin Report missing!"
}

Write-Host "VERIFICATION SCRIPT COMPLETED SUCCESSFULLY" -ForegroundColor Green
