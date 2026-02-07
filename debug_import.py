import sys
import os
import traceback

print(f"CWD: {os.getcwd()}")
print("PYTHONPATH:", os.environ.get("PYTHONPATH", "Not Set"))
print("sys.path:", sys.path)

try:
    print("\nAttempting 'import sbllm'...")
    import sbllm
    print(f"sbllm location: {sbllm.__file__}")
    
    print("Attempting 'from sbllm import evaluate'...")
    from sbllm import evaluate
    print(f"evaluate location: {evaluate.__file__}")
    
    print("Import Successful!")
except Exception as e:
    print("\nImport Failed!")
    traceback.print_exc()
