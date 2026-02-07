
import sys
import os

# Ensure the root of the project is in path
sys.path.append(os.path.join(os.getcwd(), 'sbllm_repo'))

# Now we can import the module properly
from sbllm.evaluate import main

if __name__ == '__main__':
    main()
