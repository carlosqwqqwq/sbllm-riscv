"""
Standalone Harness Templates for Single-File Evaluation.
Refactored in Round 16: Header/Footer structure with dynamic call generation.
"""

# ============================================================================
# BENCHMARK CONFIGURATIONS
# Each benchmark defines:
# - header: includes and data setup
# - footer: timing and output
# - setup_code: code to prepare test data (runs once)
# - call_template: template for calling the function with {func_name}
# ============================================================================

# ============================================================================
# RESERVED SYMBOL MAP
# Maps benchmark names to safe function names to avoid C standard library conflicts.
# e.g., 'memcpy' -> 'memcpy_rvv' to avoid conflict with <cstring>
# ============================================================================
RESERVED_SYMBOL_MAP = {
    'memcpy': 'memcpy_rvv',
    'memset': 'memset_rvv',
    'memcmp': 'memcmp_rvv',
}

BENCHMARKS = {
    'absval': {
        'header': r'''
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <riscv_vector.h>

using namespace std;

// ==========================================
// Candidate Code Injection
// ==========================================
''',
        'footer': r'''
// ==========================================
// Harness
// ==========================================
int main() {{
    const int channels = 16;
    const int size = 1024;
    const int total = channels * size;
    const int NUM_RUNS = 500;

    vector<float> data(total);
    for(int i=0; i<total; ++i) data[i] = (i % 100) - 50.0f;

    // Call the function
    {func_name}(data.data(), channels, size);

    // Verify (Checksum)
    double checksum = 0;
    for(int i=0; i<total; ++i) checksum += data[i];
    cout << "[VERIFY] Checksum: " << checksum << endl;

    // Benchmark
    const int WARMUP_RUNS = 5;
    for(int i=0; i<WARMUP_RUNS; ++i) {{
        {func_name}(data.data(), channels, size);
    }}

    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();
    for(int i=0; i<NUM_RUNS; ++i) {{
        {func_name}(data.data(), channels, size);
        asm volatile("" : : "r"(data.data()) : "memory");
    }}
    auto end = Clock::now();
    
    double duration_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << "[TIME] Average Execution Time: " << (duration_ms / NUM_RUNS) << " ms" << endl;
    return 0;
}}
''',
        'func_pattern': r'void\s+(\w+)\s*\(\s*float\s*\*',
        'default_func': 'absval_opt',
    },

    'eltwise': {
        'header': r'''
#include <iostream>
#include <vector>
#include <chrono>
#include <riscv_vector.h>

using namespace std;

// ==========================================
// Candidate Code Injection
// ==========================================
''',
        'footer': r'''
// ==========================================
// Harness
// ==========================================
int main() {{
    const int channels = 32;
    const int size = 56 * 56;
    const int num_inputs = 2;
    const int total = channels * size;
    const int NUM_RUNS = 500;

    vector<vector<float>> inputs(num_inputs, vector<float>(total));
    vector<float*> ptrs(num_inputs);
    vector<float> output(total);
    vector<float> alpha(num_inputs, 1.0f);

    for(int k=0; k<num_inputs; ++k) {{
        for(int i=0; i<total; ++i) inputs[k][i] = (i % 10) * 0.1f;
        ptrs[k] = inputs[k].data();
    }}

    // Call the function
    {func_name}(ptrs, output.data(), channels, size, 1, alpha.data());

    // Verify (Checksum)
    double checksum = 0;
    for(int i=0; i<total; ++i) checksum += output[i];
    cout << "[VERIFY] Checksum: " << checksum << endl;

    // Benchmark
    const int WARMUP_RUNS = 5;
    for(int i=0; i<WARMUP_RUNS; ++i) {{
        {func_name}(ptrs, output.data(), channels, size, 1, alpha.data());
    }}

    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();
    for(int i=0; i<NUM_RUNS; ++i) {{
        {func_name}(ptrs, output.data(), channels, size, 1, alpha.data());
        asm volatile("" : : "r"(output.data()) : "memory");
    }}
    auto end = Clock::now();
    
    double duration_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << "[TIME] Average Execution Time: " << (duration_ms / NUM_RUNS) << " ms" << endl;
    return 0;
}}
''',
        'func_pattern': r'void\s+(\w+)\s*\(\s*vector\s*<\s*float\s*\*\s*>',
        'default_func': 'eltwise_opt',
    },

    'innerproduct': {
        'header': r'''
#include <iostream>
#include <vector>
#include <chrono>
#include <riscv_vector.h>

using namespace std;

// ==========================================
// Candidate Code Injection
// ==========================================
''',
        'footer': r'''
// ==========================================
// Harness
// ==========================================
int main() {{
    const int num_output = 64;
    const int size = 128;
    const int NUM_RUNS = 1000;

    vector<float> input(size, 1.0f);
    vector<float> weight(num_output * size, 0.5f);
    vector<float> bias(num_output, 0.1f);
    vector<float> output(num_output);

    // Call the function
    {func_name}(input.data(), weight.data(), bias.data(), output.data(), num_output, size);

    // Verify (Checksum)
    double checksum = 0;
    for(int i=0; i<num_output; ++i) checksum += output[i];
    cout << "[VERIFY] Checksum: " << checksum << endl;

    // Benchmark
    const int WARMUP_RUNS = 5;
    for(int i=0; i<WARMUP_RUNS; ++i) {{
        {func_name}(input.data(), weight.data(), bias.data(), output.data(), num_output, size);
    }}

    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();
    for(int i=0; i<NUM_RUNS; ++i) {{
        {func_name}(input.data(), weight.data(), bias.data(), output.data(), num_output, size);
        asm volatile("" : : "r"(output.data()) : "memory");
    }}
    auto end = Clock::now();
    
    double duration_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << "[TIME] Average Execution Time: " << (duration_ms / NUM_RUNS) << " ms" << endl;
    return 0;
}}
''',
        'func_pattern': r'void\s+(\w+)\s*\(\s*float\s*\*\s*\w*\s*,\s*float\s*\*\s*\w*\s*,\s*float\s*\*',
        'default_func': 'innerproduct_opt',
    },

    'memcpy': {
        'header': r'''
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <riscv_vector.h>

using namespace std;

// ==========================================
// Candidate Code Injection
// ==========================================
''',
        'footer': r'''
// ==========================================
// Harness
// ==========================================
int main() {{
    const size_t N = 1024 * 64;
    const int NUM_RUNS = 100;

    vector<uint8_t> src(N, 0xAA), dest(N, 0);
    
    // Call the function
    {func_name}(dest.data(), src.data(), N);

    // Verify (Checksum)
    double checksum = 0;
    for(int i=0; i<N; ++i) checksum += dest[i];
    cout << "[VERIFY] Checksum: " << checksum << endl;

    // Verification (Original)
    if (memcmp(dest.data(), src.data(), N) != 0) {{
        cerr << "[VERIFY] VERIFICATION FAILED (memcmp)" << endl;
        return 1;
    }}

    // Benchmark
    const int WARMUP_RUNS = 5;
    for(int i=0; i<WARMUP_RUNS; ++i) {{
        {func_name}(dest.data(), src.data(), N);
    }}

    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();
    for(int i=0; i<NUM_RUNS; ++i) {{
        {func_name}(dest.data(), src.data(), N);
        asm volatile("" : : "r"(dest.data()) : "memory");
    }}
    auto end = Clock::now();
    
    double duration_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << "[TIME] Average Execution Time: " << (duration_ms / NUM_RUNS) << " ms" << endl;
    return 0;
}}
''',
        'func_pattern': r'void\s+(\w+)\s*\(\s*void\s*\*',
        'default_func': 'memcpy_opt',
    },

    'ascii_to_utf16': {
        'header': r'''
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <stdint.h>
#include <riscv_vector.h>

using namespace std;

// ==========================================
// Candidate Code Injection
// ==========================================
''',
        'footer': r'''
// ==========================================
// Harness
// ==========================================
int main() {{
    const size_t DATA_SIZE = 1024 * 16;
    const int NUM_RUNS = 20;

    vector<uint8_t> src(DATA_SIZE);
    vector<uint16_t> dest(DATA_SIZE);

    for(size_t i=0; i<DATA_SIZE; ++i) src[i] = i % 128;

    // Call the function
    {func_name}(dest.data(), src.data(), DATA_SIZE);

    // Verify (Checksum)
    double checksum = 0;
    for(int i=0; i<DATA_SIZE; ++i) checksum += dest[i];
    cout << "[VERIFY] Checksum: " << checksum << endl;

    // Benchmark
    const int WARMUP_RUNS = 5;
    for(int i=0; i<WARMUP_RUNS; ++i) {{
        {func_name}(dest.data(), src.data(), DATA_SIZE);
    }}

    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();
    for(int i=0; i<NUM_RUNS; ++i) {{
        {func_name}(dest.data(), src.data(), DATA_SIZE);
        asm volatile("" : : "r"(dest.data()) : "memory");
    }}
    auto end = Clock::now();
    
    double duration_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << "[TIME] Average Execution Time: " << (duration_ms / NUM_RUNS) << " ms" << endl;
    return 0;
}}
''',
        'func_pattern': r'void\s+(\w+)\s*\(\s*uint16_t\s*\*',
        'default_func': 'ascii_to_utf16_opt',
    },

    'atan': {
        'header': r'''
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <riscv_vector.h>

using namespace std;

// ==========================================
// Candidate Code Injection
// ==========================================
''',
        'footer': r'''
// ==========================================
// Harness
// ==========================================
int main() {{
    const int N = 1024 * 4;
    const int NUM_RUNS = 50;
    
    vector<float> y(N), x(N), dst(N);
    for(int i=0; i<N; ++i) {{
        y[i] = (i % 100) / 10.0f;
        x[i] = (i % 50) / 5.0f;
    }}

    // Call the function
    {func_name}(y.data(), x.data(), dst.data(), N, true);

    // Verify (Checksum)
    double checksum = 0;
    for(int i=0; i<N; ++i) checksum += dst[i];
    cout << "[VERIFY] Checksum: " << checksum << endl;

    // Benchmark
    const int WARMUP_RUNS = 5;
    for(int i=0; i<WARMUP_RUNS; ++i) {{
        {func_name}(y.data(), x.data(), dst.data(), N, true);
    }}

    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();
    for(int i=0; i<NUM_RUNS; ++i) {{
        {func_name}(y.data(), x.data(), dst.data(), N, true);
        asm volatile("" : : "r"(dst.data()) : "memory");
    }}
    auto end = Clock::now();
    
    double duration_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << "[TIME] Average Execution Time: " << (duration_ms / NUM_RUNS) << " ms" << endl;
    return 0;
}}
''',
        'func_pattern': r'void\s+(\w+)\s*\(\s*(?:const\s+)?float\s*\*\s*\w*\s*,\s*(?:const\s+)?float\s*\*\s*\w*\s*,\s*float\s*\*',
        'default_func': 'fastAtan32f_opt',
    },

    'batchnorm': {
        'header': r'''
#include <iostream>
#include <vector>
#include <chrono>
#include <riscv_vector.h>

using namespace std;

// ==========================================
// Candidate Code Injection
// ==========================================
''',
        'footer': r'''
// ==========================================
// Harness
// ==========================================
int main() {{
    const int N = 1024 * 64;
    const int NUM_RUNS = 20;

    vector<float> in(N, 1.0f), out(N), mean(N, 0.1f), var(N, 1.2f), gamma(N, 0.5f), beta(N, 0.2f);
    
    // Call the function
    {func_name}(out.data(), in.data(), mean.data(), var.data(), gamma.data(), beta.data(), 1e-5f, N);

    // Verify (Checksum)
    double checksum = 0;
    for(int i=0; i<N; ++i) checksum += out[i];
    cout << "[VERIFY] Checksum: " << checksum << endl;

    // Benchmark
    const int WARMUP_RUNS = 5;
    for(int i=0; i<WARMUP_RUNS; ++i) {{
        {func_name}(out.data(), in.data(), mean.data(), var.data(), gamma.data(), beta.data(), 1e-5f, N);
    }}

    using Clock = chrono::high_resolution_clock;
    auto start = Clock::now();
    for(int i=0; i<NUM_RUNS; ++i) {{
        {func_name}(out.data(), in.data(), mean.data(), var.data(), gamma.data(), beta.data(), 1e-5f, N);
        asm volatile("" : : "r"(out.data()) : "memory");
    }}
    auto end = Clock::now();
    
    double duration_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << "[TIME] Average Execution Time: " << (duration_ms / NUM_RUNS) << " ms" << endl;
    return 0;
}}
''',
        'func_pattern': r'void\s+(\w+)\s*\(\s*float\s*\*',
        'default_func': 'batchnorm_opt',
    },
}


def get_benchmark_config(benchmark_name: str) -> dict:
    """Get the configuration for a specific benchmark."""
    return BENCHMARKS.get(benchmark_name)


def generate_full_code(benchmark_name: str, candidate_code: str, func_name: str) -> str:
    """
    Generate the complete standalone code by combining:
    - Header (includes, setup)
    - Candidate code (LLM generated)
    - Footer (main with call using extracted func_name)
    """
    config = BENCHMARKS.get(benchmark_name)
    if not config:
        return None
    
    header = config['header']
    footer = config['footer'].format(func_name=func_name)
    
    return header + candidate_code + "\n" + footer
