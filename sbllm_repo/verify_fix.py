
import os
import shutil
import logging
import sys
from sbllm.utils.project_evaluator import RVVBenchEvaluator, WorkspacePool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY_FIX")

def verify_fix():
    project_root = "/work/rvv-bench"
    pool = WorkspacePool(project_root, pool_size=1)
    
    try:
        # 1. Initialize Workspace
        logger.info("Initializing workspace pool...")
        workspaces = pool.initialize()
        ws_path = workspaces[0]
        logger.info(f"Workspace created at: {ws_path}")
        
        # 2. Instantiate Evaluator
        evaluator = RVVBenchEvaluator(ws_path, qemu_path="/usr/local/bin/qemu-riscv64")
        
        # 3. Test Baseline Build (is_baseline=True)
        logger.info("Testing Baseline Build (memcpy)...")
        if not evaluator.build(target="memcpy", is_baseline=True):
            logger.error(f"Baseline build failed: {evaluator.last_build_error}")
            sys.exit(1)
        logger.info("Baseline build success!")
        
        # 4. Check config.h
        config_path = os.path.join(ws_path, "bench", "config.h")
        with open(config_path, 'r') as f:
            content = f.read()
            if "MAX_MEM (1024*64)" not in content:
                logger.error("config.h was NOT overwritten with QEMU params!")
                sys.exit(1)
        logger.info("config.h verification success!")

        # 5. Check if memset was built (should NOT be if target works)
        memset_bin = os.path.join(ws_path, "bench", "memset")
        if os.path.exists(memset_bin):
            logger.warning("WARNING: memset binary exists! Target selection might be loose.")
        else:
            logger.info("Target isolation verified: memset not built.")

        # 6. Test Execution (Timeout check)
        logger.info("Testing Execution (memcpy)...")
        res = evaluator.run_benchmark({'benchmark_name': 'memcpy'})
        logger.info(f"Execution Result: {res}")
        
        if res.get('returncode') != 0:
            logger.error(f"Execution failed: {res.get('error') or res.get('output')}")
            sys.exit(1)
            
        logger.info("VERIFICATION SUCCESSFUL: Fixes are working.")
        
    except Exception as e:
        logger.error(f"Verification Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        pool.cleanup()

if __name__ == "__main__":
    verify_fix()
