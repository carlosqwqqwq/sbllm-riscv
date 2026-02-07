@echo off
REM =================================================================
REM SBLLM Subset Test Launcher (Windows/Docker)
REM =================================================================

REM Get the absolute path of the current directory (project root)
set "PROJECT_ROOT=%~dp0"
REM Remove trailing backslash
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo [*] Launching SBLLM Subset Test in Docker...
echo [*] Project Root: %PROJECT_ROOT%

REM Run the Docker container with volume mounts
docker run --rm -it ^
    -v "%PROJECT_ROOT%:/work" ^
    -v "%PROJECT_ROOT%/processed_data:/work/processed_data" ^
    -v "%PROJECT_ROOT%/results:/work/results" ^
    riscv-opt-env:latest ^
    /bin/bash /work/sbllm_repo/run_subset_test.sh


REM echo [*] Subset Test Completed.
REM exit /b %errorlevel%

