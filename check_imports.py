import sys
import os
import traceback

sys.path.append(os.path.join(os.getcwd(), 'sbllm_repo'))

modules_to_check = [
    "sbllm.core.evaluator_interface",
    "sbllm.utils.statistics_utils",
    "sbllm.utils.qemu_evaluator",
    "sbllm.utils.project_evaluator",
    "sbllm.execution"
]

print(f"Checking imports from {os.getcwd()}...")

for mod in modules_to_check:
    print(f"Testing {mod}...")
    try:
        __import__(mod)
        print(f"  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
