@echo off
setlocal

REM ========================================================
REM  SBLLM One-Click Evaluation Launcher
REM ========================================================

SET "PROJECT_ROOT=%~dp0"
REM Remove trailing backslash if present
if "%PROJECT_ROOT:~-1%"=="\" SET "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo ========================================================
echo  SBLLM Automated Evaluation Pipeline
echo  Project Root: %PROJECT_ROOT%
echo ========================================================

echo.
echo [1/5] Checking for Docker image (riscv-opt-env)...
docker image inspect riscv-opt-env >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker image 'riscv-opt-env' not found!
    echo Please build or load the image first.
    pause
    exit /b 1
)

echo [2/5] Cleaning up old container...
docker rm -f production-eval >nul 2>&1

echo [3/5] Starting production-eval container...
docker run -itd --name production-eval ^
  -v "%PROJECT_ROOT%\sbllm_repo:/work/sbllm_repo" ^
  -v "%PROJECT_ROOT%\processed_data:/work/processed_data" ^
  -v "%PROJECT_ROOT%\results:/work/results" ^
  riscv-opt-env bash

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start Docker container.
    pause
    exit /b 1
)

echo [4/5] Injecting configuration...
if exist "%PROJECT_ROOT%\.env" (
    docker cp "%PROJECT_ROOT%\.env" production-eval:/work/sbllm_repo/.env
) else (
    echo [WARNING] .env file not found in %PROJECT_ROOT%. Using container defaults if available.
)

echo [5/5] Executing standard execution pipeline...
echo        This may take a while depending on the model and dataset size.
echo        Check progress in another terminal via: docker logs -f production-eval
echo.

docker exec -w /work/sbllm_repo production-eval bash -c "chmod +x run_full_evaluation.sh && ./run_full_evaluation.sh"

echo.
echo ========================================================
echo  Evaluation Pipeline Finished.
echo ========================================================
echo.
echo  Results are available in:
echo  %PROJECT_ROOT%\results
echo.
pause
