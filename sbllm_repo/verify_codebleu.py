import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sbllm.utils.codebleu_utils import get_codebleu_score, get_detailed_codebleu

def test_codebleu():
    ref = """
    #include <stdio.h>
    int main() {
        printf("Hello World\\n");
        return 0;
    }
    """
    hypo = """
    #include <stdio.h>
    int main() {
        printf("Hello World\\n");
        return 0;
    }
    """
    
    print("Testing identical codes...")
    results = get_detailed_codebleu(ref, hypo, lang='cpp')
    print(f"Results: {results}")
    assert results['codebleu'] > 0.99
    
    print("\nTesting different codes...")
    hypo_diff = """
    #include <stdio.h>
    int main() {
        printf("Hi\\n");
        return 1;
    }
    """
    results_diff = get_detailed_codebleu(ref, hypo_diff, lang='cpp')
    print(f"Diff Results: {results_diff}")
    assert results_diff['codebleu'] < results['codebleu']
    
    print("\nTesting RVV intrinsics weights...")
    ref_rvv = "vint32m1_t v = vle32_v_i32m1(p, vl);"
    hypo_rvv = "vint32m1_t v = vle32_v_i32m1(p, vl);"
    details = get_detailed_codebleu(ref_rvv, hypo_rvv, lang='riscv')
    print(f"RVV Details: {details}")
    
    print("Verification successful!")

if __name__ == "__main__":
    try:
        test_codebleu()
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
