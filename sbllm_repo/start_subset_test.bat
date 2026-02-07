@echo off
echo ========================================================
echo   Running Subset Verification Test (Run 7)
echo   Target: Subset of RVV and VecIntrin Benchmarks
echo ========================================================

docker run --rm ^
  -v "%cd%:/work" ^
  riscv-opt-env:latest ^
  /bin/bash /work/run_subset_test.sh
