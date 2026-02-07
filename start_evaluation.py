import argparse
import subprocess
import sys
import os
import shutil
import time
import datetime

# Configuration
IMAGE_NAME = "riscv-opt-env"
CONTAINER_NAME = "production-eval"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SBLLM_REPO = os.path.join(PROJECT_ROOT, "sbllm_repo")
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "processed_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def run_cmd(cmd, cwd=None, capture=False, check=True):
    """Run a shell command."""
    print(f"[EXEC] {' '.join(cmd)}")
    try:
        if capture:
            return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        else:
            return subprocess.run(cmd, cwd=cwd, check=check)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        if capture:
            print(f"[STDERR] {e.stderr}")
        return None

def check_docker():
    """Check if Docker is running."""
    res = run_cmd(["docker", "info"], capture=True, check=False)
    if not res or res.returncode != 0:
        print("[ERROR] Docker is NOT running. Please start Docker Desktop.")
        sys.exit(1)

def check_image():
    """Check if image exists, build if not."""
    print("[INFO] Checking for Docker image...")
    res = run_cmd(["docker", "image", "inspect", IMAGE_NAME], capture=True, check=False)
    if not res or res.returncode != 0:
        print(f"[INFO] Image '{IMAGE_NAME}' not found. Building from sbllm/Dockerfile...")
        sbllm_dir = os.path.join(PROJECT_ROOT, "sbllm")
        if not os.path.exists(sbllm_dir):
             # Fallback if sbllm dir is actually the repo root relative?
             # Based on list_dir, 'sbllm' inside 'sbllm_repo' is the package, 
             # but there is also 'sbllm_repo' dir. 
             # The run_all.bat says `docker build -t riscv-opt-env -f sbllm/Dockerfile sbllm/`
             # This implies there is a 'sbllm' folder in PROJECT_ROOT that contains Dockerfile.
             pass
        
        # Check actual path
        dockerfile_path = os.path.join(PROJECT_ROOT, "sbllm", "Dockerfile")
        build_context = os.path.join(PROJECT_ROOT, "sbllm")
        
        # In case the directory structure is different, try to locate Dockerfile
        if not os.path.exists(dockerfile_path):
             # Try sbllm_repo/docker? Or existing knowledge?
             # run_all.bat says: `docker build ... -f sbllm/Dockerfile sbllm/`
             pass

        run_cmd(["docker", "build", "-t", IMAGE_NAME, "-f", "sbllm/Dockerfile", "sbllm/"], cwd=PROJECT_ROOT)

def mode_standard():
    """Run Standard SBLLM Pipeline (replaces start_evaluation.bat)."""
    print("=== Starting Standard Evaluation Pipeline ===")
    
    # Clean old container
    run_cmd(["docker", "rm", "-f", CONTAINER_NAME], check=False, capture=True)
    
    # Run container
    print("[INFO] Starting container...")
    vol_args = [
        "-v", f"{SBLLM_REPO}:/work/sbllm_repo",
        "-v", f"{PROCESSED_DATA}:/work/processed_data",
        "-v", f"{RESULTS_DIR}:/work/results"
    ]
    
    run_cmd(["docker", "run", "-itd", "--name", CONTAINER_NAME] + vol_args + [IMAGE_NAME, "bash"])
    
    # Inject config
    env_file = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_file):
        print(f"[INFO] Injecting .env configuration...")
        run_cmd(["docker", "cp", env_file, f"{CONTAINER_NAME}:/work/sbllm_repo/.env"])
    else:
        print("[WARN] .env file not found. Using defaults.")
        
    # Exec script
    print("[INFO] Executing pipeline script...")
    cmd = "chmod +x run_full_evaluation.sh && ./run_full_evaluation.sh"
    run_cmd(["docker", "exec", "-w", "/work/sbllm_repo", CONTAINER_NAME, "bash", "-c", cmd])
    
    print(f"\n[INFO] Pipeline finished. Results in: {RESULTS_DIR}")

def mode_benchmark(bench_type, full_suite=False):
    """Run Specific Benchmarks (replaces run_all.bat)."""
    print(f"=== Starting {bench_type} Benchmark ===")
    
    # Setup Result Dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    res_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(res_dir, exist_ok=True)
    print(f"[INFO] Results will be saved to: {res_dir}")
    
    target_dir = ""
    target_script = ""
    bench_args = ""
    
    if bench_type == "rvv-bench":
        target_dir = "rvv-bench"
        target_script = "./run_rvv_bench.sh"
        if full_suite:
            bench_args = "--full"
    elif bench_type == "vecintrin":
        target_dir = "VecIntrinBench"
        target_script = "./run_vecintrin_bench.sh"
    
    # Run Ephemeral Container
    # Note: run_all.bat mounts %cd% to /work.
    # We should match that to ensure paths align.
    
    print("[INFO] Running benchmark container...")
    
    docker_cmd = [
        "docker", "run", "--rm", "-it",
        "-v", f"{PROJECT_ROOT}:/work",
        "-v", f"{res_dir}:/work/results", # Map specific result dir to internal /results override?
        # run_all.bat maps "%cd%\%RESULT_DIR%":/work/results 
        # But wait, run_all.bat passes argument `/work/results` to the script.
        # So we should map host res_dir to container /work/results is redundant if /work is already mapped?
        # run_all.bat: -v "%cd%":/work AND -v "%cd%\%RESULT_DIR%":/work/results
        # This acts as an override or dedicated mount.
        "-w", f"/work/{target_dir}",
        IMAGE_NAME, "bash", target_script, "--internal", bench_args, "/work/results"
    ]
    
    # Clean up empty args
    docker_cmd = [x for x in docker_cmd if x]
    
    run_cmd(docker_cmd)
    print(f"\n[INFO] Benchmark finished. Check {res_dir}")

def main():
    parser = argparse.ArgumentParser(description="SBLLM Unified Evaluation Launcher")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Evaluation Mode')
    
    # Standard Mode
    parser_std = subparsers.add_parser('standard', help='Run standard SBLLM pipeline (like start_evaluation.bat)')
    
    # RVV Bench Mode
    parser_rvv = subparsers.add_parser('rvv-bench', help='Run rvv-bench (like run_all.bat option 1/2)')
    parser_rvv.add_argument('--full', action='store_true', help='Run full test suite (otherwise quick check)')
    
    # VecIntrin Mode
    parser_vec = subparsers.add_parser('vecintrin', help='Run VecIntrinBench (like run_all.bat option 3)')

    args = parser.parse_args()
    
    check_docker()
    check_image()
    
    if args.mode == 'standard':
        mode_standard()
    elif args.mode == 'rvv-bench':
        mode_benchmark('rvv-bench', full_suite=args.full)
    elif args.mode == 'vecintrin':
        mode_benchmark('vecintrin')

if __name__ == "__main__":
    main()
