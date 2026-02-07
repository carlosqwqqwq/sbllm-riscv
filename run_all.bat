@echo off
setlocal
echo ========================================================
echo [Windows] RISC-V RVV Benchmark One-Click Runner
echo ========================================================

:: 1. Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] Docker is NOT running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)

:: 2. Check if image exists
echo [Info] Checking for Docker image 'riscv-opt-env'...
docker image inspect riscv-opt-env >nul 2>&1
if %errorlevel% neq 0 (
    echo [Info] Image not found. Building from sbllm/Dockerfile...
    echo [Info] This may take a while to download toolchains...
    docker build -t riscv-opt-env -f sbllm/Dockerfile sbllm/
    if %errorlevel% neq 0 (
        echo [Error] Docker build failed.
        pause
        exit /b 1
    )
    echo [Info] Build completed successfully.
) else (
    echo [Info] Image found. Skipping build.
)

:: 3. Prepare Results Directory
:: Use PowerShell for a robust, locale-independent timestamp (YYYYMMDD_HHmm)
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd_HHmm'"') do set "TIMESTAMP=%%i"

if "%TIMESTAMP%"=="" set "TIMESTAMP=latest"

set "RESULT_DIR=results\%TIMESTAMP%"
mkdir "%RESULT_DIR%" >nul 2>&1

echo [Info] Select benchmark to run:
echo [1] rvv-bench (Quick Check)
echo [2] rvv-bench (Full Suite)
echo [3] VecIntrinBench (Complete)

set "choice=%~1"
if "%choice%"=="" (
    set /p choice="Choice [1-3, default 1]: "
) else (
    echo [Automation] Using provided choice: %choice%
)

set "TARGET_DIR=rvv-bench"
set "TARGET_SCRIPT=bash ./run_rvv_bench.sh"
set "BENCH_MODE="

if "%choice%"=="2" (
    set "BENCH_MODE=--full"
)

if "%choice%"=="3" (
    set "TARGET_DIR=VecIntrinBench"
    set "TARGET_SCRIPT=bash ./run_vecintrin_bench.sh"
    set "BENCH_MODE="
)

echo [Info] Starting Benchmark Container...
echo [Info] Results will be saved to: %RESULT_DIR%
echo.

:: Run and capture ALL output via the container's internal tee logic
:: We pass the path as a simple argument. Ensure paths are quoted.
docker run --rm -it ^
    -v "%cd%":/work ^
    -v "%cd%\%RESULT_DIR%":/work/results ^
    -w /work/%TARGET_DIR% ^
    riscv-opt-env %TARGET_SCRIPT% --internal %BENCH_MODE% /work/results

if %errorlevel% neq 0 (
    echo.
    echo [Error] Benchmark execution failed.
    echo [Info] Check logs in %RESULT_DIR%
    pause
    exit /b 1
)

echo.
echo ========================================================
echo [Success] All benchmarks completed successfully!
echo [Info] All results and logs are stored in: %RESULT_DIR%
echo ========================================================
pause
