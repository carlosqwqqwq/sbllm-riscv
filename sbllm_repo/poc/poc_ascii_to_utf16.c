#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// ==========================================
// 1. Baseline Implementation (From Source)
// ==========================================
void ascii_to_utf16_scalar(uint16_t * dest, uint8_t const * src, size_t len)
{
    while (len--) *dest++ = *src++;
}

// ==========================================
// 2. Simulated LLM-Generated Code (Optimized)
//    (In real scenario, this is #include "llm_code.c")
// ==========================================
void ascii_to_utf16_opt(uint16_t * dest, uint8_t const * src, size_t len)
{
    // Simple unrolling for PoC
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        dest[i] = src[i];
        dest[i+1] = src[i+1];
        dest[i+2] = src[i+2];
        dest[i+3] = src[i+3];
    }
    for (; i < len; i++) {
        dest[i] = src[i];
    }
}

// ==========================================
// 3. Standalone Verification & Timing Harness
// ==========================================
#define DATA_SIZE 1024 * 16
#define NUM_RUNS 100

double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main() {
    // 1. Setup Data
    uint8_t *src = (uint8_t*)malloc(DATA_SIZE);
    uint16_t *dest_base = (uint16_t*)malloc(DATA_SIZE * sizeof(uint16_t));
    uint16_t *dest_opt = (uint16_t*)malloc(DATA_SIZE * sizeof(uint16_t));
    
    // Fill src
    for(int i=0; i<DATA_SIZE; i++) src[i] = i & 0xFF;
    
    // 2. Verification
    memset(dest_base, 0, DATA_SIZE * sizeof(uint16_t));
    memset(dest_opt, 0, DATA_SIZE * sizeof(uint16_t));
    
    ascii_to_utf16_scalar(dest_base, src, DATA_SIZE);
    ascii_to_utf16_opt(dest_opt, src, DATA_SIZE);
    
    if (memcmp(dest_base, dest_opt, DATA_SIZE * sizeof(uint16_t)) != 0) {
        printf("FAILED: Output mismatch!\n");
        return 1;
    }
    printf("VERIFICATION: SUCCESS\n");
    
    // 3. Timing Benchmark
    double start, end;
    
    // Baseline Time
    start = get_time_ns();
    for(int i=0; i<NUM_RUNS; i++) {
        ascii_to_utf16_scalar(dest_base, src, DATA_SIZE);
        // Prevent optimization
        __asm__ volatile("" : : "r"(dest_base) : "memory");
    }
    end = get_time_ns();
    double avg_base = (end - start) / NUM_RUNS;
    
    // Optimized Time
    start = get_time_ns();
    for(int i=0; i<NUM_RUNS; i++) {
        ascii_to_utf16_opt(dest_opt, src, DATA_SIZE);
        __asm__ volatile("" : : "r"(dest_opt) : "memory");
    }
    end = get_time_ns();
    double avg_opt = (end - start) / NUM_RUNS;
    
    printf("BASELINE_NS: %.2f\n", avg_base);
    printf("OPTIMIZED_NS: %.2f\n", avg_opt);
    printf("SPEEDUP: %.2fx\n", avg_base / avg_opt);

    free(src);
    free(dest_base);
    free(dest_opt);
    return 0;
}
