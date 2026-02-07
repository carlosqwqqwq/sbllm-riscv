import jsonlines
import os

def generate_riscv_assembly_samples(output_file):
    """
    Generate 5 RISC-V assembly code samples with slow and optimized versions.
    These are real RISC-V RV64I assembly code that can be assembled with riscv64-unknown-linux-gnu-as.
    
    Optimization types:
    1. Loop Unrolling - Reduce loop overhead
    2. Instruction Scheduling - Reduce pipeline stalls
    3. Register Allocation - Minimize memory access
    4. Branch Optimization - Reduce branch penalties
    5. Memory Access Optimization - Better cache usage
    """
    samples = [
        # ========== 1. Loop Unrolling: Array Sum ==========
        {
            "idx": "1",
            "id": "1",
            "query": "",
            "code_v0_no_empty_lines": """# RISC-V Assembly: Array Sum (Unoptimized)
# Sums N integers from an array
# Input: a0 = array pointer, a1 = count
# Output: a0 = sum

.global array_sum_slow
.text

array_sum_slow:
    li      t0, 0           # sum = 0
    li      t1, 0           # i = 0
loop_slow:
    bge     t1, a1, done_slow   # if i >= count, exit
    slli    t2, t1, 2       # t2 = i * 4 (byte offset)
    add     t3, a0, t2      # t3 = &arr[i]
    lw      t4, 0(t3)       # t4 = arr[i]
    add     t0, t0, t4      # sum += arr[i]
    addi    t1, t1, 1       # i++
    j       loop_slow       # repeat
done_slow:
    mv      a0, t0          # return sum
    ret

# Main function for testing
.global main
main:
    # Create test array on stack
    addi    sp, sp, -48
    sd      ra, 40(sp)
    
    # Initialize array with values 1,2,3,4,5,6,7,8
    li      t0, 1
    sw      t0, 0(sp)
    li      t0, 2
    sw      t0, 4(sp)
    li      t0, 3
    sw      t0, 8(sp)
    li      t0, 4
    sw      t0, 12(sp)
    li      t0, 5
    sw      t0, 16(sp)
    li      t0, 6
    sw      t0, 20(sp)
    li      t0, 7
    sw      t0, 24(sp)
    li      t0, 8
    sw      t0, 28(sp)
    
    mv      a0, sp          # array pointer
    li      a1, 8           # count = 8
    call    array_sum_slow
    
    # Expected sum: 1+2+3+4+5+6+7+8 = 36
    li      t0, 36
    bne     a0, t0, fail
    li      a0, 0           # return 0 (success)
    j       exit
fail:
    li      a0, 1           # return 1 (failure)
exit:
    ld      ra, 40(sp)
    addi    sp, sp, 48
    ret
""",
            "code_v1_no_empty_lines": """# RISC-V Assembly: Array Sum (Optimized with Loop Unrolling)
# Sums N integers from an array - 4x unrolled
# Input: a0 = array pointer, a1 = count (must be multiple of 4)
# Output: a0 = sum

.global array_sum_fast
.text

array_sum_fast:
    li      t0, 0           # sum = 0
    li      t1, 0           # i = 0
    
loop_fast:
    bge     t1, a1, done_fast   # if i >= count, exit
    
    # Unrolled: process 4 elements at once
    slli    t2, t1, 2       # byte offset
    add     t3, a0, t2      # base address
    
    lw      t4, 0(t3)       # arr[i]
    lw      t5, 4(t3)       # arr[i+1]
    lw      t6, 8(t3)       # arr[i+2]
    lw      a2, 12(t3)      # arr[i+3]
    
    add     t0, t0, t4      # sum += arr[i]
    add     t0, t0, t5      # sum += arr[i+1]
    add     t0, t0, t6      # sum += arr[i+2]
    add     t0, t0, a2      # sum += arr[i+3]
    
    addi    t1, t1, 4       # i += 4
    j       loop_fast
    
done_fast:
    mv      a0, t0          # return sum
    ret

.global main
main:
    addi    sp, sp, -48
    sd      ra, 40(sp)
    
    li      t0, 1
    sw      t0, 0(sp)
    li      t0, 2
    sw      t0, 4(sp)
    li      t0, 3
    sw      t0, 8(sp)
    li      t0, 4
    sw      t0, 12(sp)
    li      t0, 5
    sw      t0, 16(sp)
    li      t0, 6
    sw      t0, 20(sp)
    li      t0, 7
    sw      t0, 24(sp)
    li      t0, 8
    sw      t0, 28(sp)
    
    mv      a0, sp
    li      a1, 8
    call    array_sum_fast
    
    li      t0, 36
    bne     a0, t0, fail
    li      a0, 0
    j       exit
fail:
    li      a0, 1
exit:
    ld      ra, 40(sp)
    addi    sp, sp, 48
    ret
""",
            "target": "", "reference": "", "input": "",
            "description": "Loop Unrolling Optimization: Process 4 array elements per iteration",
            "optimization_type": "Loop Unrolling"
        },

        # ========== 2. Instruction Scheduling: Dot Product ==========
        {
            "idx": "2",
            "id": "2",
            "query": "",
            "code_v0_no_empty_lines": """# RISC-V Assembly: Dot Product (Unoptimized)
# Calculates sum of a[i] * b[i]
# Input: a0 = array A, a1 = array B, a2 = count
# Output: a0 = dot product

.global dot_product_slow
.text

dot_product_slow:
    li      t0, 0           # result = 0
    li      t1, 0           # i = 0
    
loop_dot_slow:
    bge     t1, a2, done_dot_slow
    
    slli    t2, t1, 2       # offset
    add     t3, a0, t2      # &a[i]
    add     t4, a1, t2      # &b[i]
    
    lw      t5, 0(t3)       # load a[i]
    lw      t6, 0(t4)       # load b[i]
    mul     t5, t5, t6      # a[i] * b[i]  -- stall waiting for t6
    add     t0, t0, t5      # result += product -- stall waiting for mul
    
    addi    t1, t1, 1
    j       loop_dot_slow
    
done_dot_slow:
    mv      a0, t0
    ret

.global main
main:
    addi    sp, sp, -64
    sd      ra, 56(sp)
    
    # Array A: 1, 2, 3, 4
    li      t0, 1
    sw      t0, 0(sp)
    li      t0, 2
    sw      t0, 4(sp)
    li      t0, 3
    sw      t0, 8(sp)
    li      t0, 4
    sw      t0, 12(sp)
    
    # Array B: 5, 6, 7, 8
    li      t0, 5
    sw      t0, 16(sp)
    li      t0, 6
    sw      t0, 20(sp)
    li      t0, 7
    sw      t0, 24(sp)
    li      t0, 8
    sw      t0, 28(sp)
    
    mv      a0, sp          # A
    addi    a1, sp, 16      # B
    li      a2, 4           # count
    call    dot_product_slow
    
    # Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    li      t0, 70
    bne     a0, t0, fail
    li      a0, 0
    j       exit
fail:
    li      a0, 1
exit:
    ld      ra, 56(sp)
    addi    sp, sp, 64
    ret
""",
            "code_v1_no_empty_lines": """# RISC-V Assembly: Dot Product (Optimized - Better Scheduling)
# Pipeline-friendly instruction ordering to reduce stalls
# Input: a0 = array A, a1 = array B, a2 = count
# Output: a0 = dot product

.global dot_product_fast
.text

dot_product_fast:
    li      t0, 0           # result = 0
    li      t1, 0           # i = 0
    
loop_dot_fast:
    bge     t1, a2, done_dot_fast
    
    slli    t2, t1, 2       # offset for iteration i
    add     t3, a0, t2      # &a[i]
    add     t4, a1, t2      # &b[i]
    
    # Interleave loads to hide latency
    lw      t5, 0(t3)       # load a[i]
    addi    t1, t1, 1       # i++ (independent, can execute during load)
    lw      t6, 0(t4)       # load b[i]
    
    # Multiply while loads complete
    mul     a3, t5, t6      # product (allow time for loads)
    add     t0, t0, a3      # accumulate
    
    j       loop_dot_fast
    
done_dot_fast:
    mv      a0, t0
    ret

.global main
main:
    addi    sp, sp, -64
    sd      ra, 56(sp)
    
    li      t0, 1
    sw      t0, 0(sp)
    li      t0, 2
    sw      t0, 4(sp)
    li      t0, 3
    sw      t0, 8(sp)
    li      t0, 4
    sw      t0, 12(sp)
    
    li      t0, 5
    sw      t0, 16(sp)
    li      t0, 6
    sw      t0, 20(sp)
    li      t0, 7
    sw      t0, 24(sp)
    li      t0, 8
    sw      t0, 28(sp)
    
    mv      a0, sp
    addi    a1, sp, 16
    li      a2, 4
    call    dot_product_fast
    
    li      t0, 70
    bne     a0, t0, fail
    li      a0, 0
    j       exit
fail:
    li      a0, 1
exit:
    ld      ra, 56(sp)
    addi    sp, sp, 64
    ret
""",
            "target": "", "reference": "", "input": "",
            "description": "Instruction Scheduling: Interleave independent instructions to hide latency",
            "optimization_type": "Instruction Scheduling"
        },

        # ========== 3. Branch Optimization: Max of Two ==========
        {
            "idx": "3",
            "id": "3",
            "query": "",
            "code_v0_no_empty_lines": """# RISC-V Assembly: Find Maximum (Branch-heavy)
# Finds maximum of two numbers with conditional branch
# Input: a0 = x, a1 = y
# Output: a0 = max(x, y)

.global max_slow
.text

max_slow:
    blt     a0, a1, use_y   # if x < y, use y
    mv      a0, a0          # result = x (redundant but shows logic)
    j       done_max
use_y:
    mv      a0, a1          # result = y
done_max:
    ret

.global main
main:
    addi    sp, sp, -16
    sd      ra, 8(sp)
    
    # Test case 1: max(5, 3) = 5
    li      a0, 5
    li      a1, 3
    call    max_slow
    li      t0, 5
    bne     a0, t0, fail
    
    # Test case 2: max(2, 7) = 7
    li      a0, 2
    li      a1, 7
    call    max_slow
    li      t0, 7
    bne     a0, t0, fail
    
    li      a0, 0
    j       exit
fail:
    li      a0, 1
exit:
    ld      ra, 8(sp)
    addi    sp, sp, 16
    ret
""",
            "code_v1_no_empty_lines": """# RISC-V Assembly: Find Maximum (Branchless)
# Uses conditional move pattern to avoid branch penalty
# Input: a0 = x, a1 = y
# Output: a0 = max(x, y)

.global max_fast
.text

max_fast:
    # Branchless max using slt + conditional selection
    slt     t0, a0, a1      # t0 = (x < y) ? 1 : 0
    neg     t0, t0          # t0 = -1 if x < y, else 0
    
    # mask = 0xFFFF... if x < y, else 0
    and     t1, t0, a1      # t1 = y if x < y, else 0
    not     t0, t0          # t0 = 0 if x < y, else 0xFFFF...
    and     t2, t0, a0      # t2 = 0 if x < y, else x
    or      a0, t1, t2      # result = max(x, y)
    ret

.global main
main:
    addi    sp, sp, -16
    sd      ra, 8(sp)
    
    li      a0, 5
    li      a1, 3
    call    max_fast
    li      t0, 5
    bne     a0, t0, fail
    
    li      a0, 2
    li      a1, 7
    call    max_fast
    li      t0, 7
    bne     a0, t0, fail
    
    li      a0, 0
    j       exit
fail:
    li      a0, 1
exit:
    ld      ra, 8(sp)
    addi    sp, sp, 16
    ret
""",
            "target": "", "reference": "", "input": "",
            "description": "Branch Optimization: Replace conditional branch with branchless logic",
            "optimization_type": "Conditional Branch Optimization"
        },

        # ========== 4. Memory Optimization: String Copy ==========
        {
            "idx": "4",
            "id": "4",
            "query": "",
            "code_v0_no_empty_lines": """# RISC-V Assembly: String Copy (Byte-by-byte)
# Copies null-terminated string
# Input: a0 = dest, a1 = src
# Output: a0 = dest

.global strcpy_slow
.text

strcpy_slow:
    mv      t0, a0          # save dest
    
copy_loop_slow:
    lb      t1, 0(a1)       # load byte from src
    sb      t1, 0(a0)       # store to dest
    beqz    t1, copy_done   # if null terminator, done
    addi    a0, a0, 1       # dest++
    addi    a1, a1, 1       # src++
    j       copy_loop_slow
    
copy_done:
    mv      a0, t0          # return original dest
    ret

.global main
main:
    addi    sp, sp, -48
    sd      ra, 40(sp)
    
    # Source string "HELLO" at sp
    li      t0, 0x48        # 'H'
    sb      t0, 0(sp)
    li      t0, 0x45        # 'E'
    sb      t0, 1(sp)
    li      t0, 0x4C        # 'L'
    sb      t0, 2(sp)
    li      t0, 0x4C        # 'L'
    sb      t0, 3(sp)
    li      t0, 0x4F        # 'O'
    sb      t0, 4(sp)
    li      t0, 0           # null terminator
    sb      t0, 5(sp)
    
    # Dest at sp+16
    addi    a0, sp, 16      # dest
    mv      a1, sp          # src
    call    strcpy_slow
    
    # Verify: check first char is 'H'
    lb      t0, 16(sp)
    li      t1, 0x48
    bne     t0, t1, fail
    
    li      a0, 0
    j       exit
fail:
    li      a0, 1
exit:
    ld      ra, 40(sp)
    addi    sp, sp, 48
    ret
""",
            "code_v1_no_empty_lines": """# RISC-V Assembly: String Copy (Word-aligned, faster)
# Copies aligned words when possible, then remaining bytes
# Input: a0 = dest, a1 = src (assumed word-aligned)
# Output: a0 = dest

.global strcpy_fast
.text

strcpy_fast:
    mv      t0, a0          # save dest
    
copy_loop_fast:
    # Try to load a word (4 bytes) at a time
    lw      t1, 0(a1)       # load 4 bytes
    
    # Check if any byte is zero (null terminator detection)
    # Simple check: if low byte is zero, done
    andi    t2, t1, 0xFF
    beqz    t2, copy_byte_loop
    
    sw      t1, 0(a0)       # store 4 bytes
    addi    a0, a0, 4
    addi    a1, a1, 4
    j       copy_loop_fast

copy_byte_loop:
    # Fallback to byte-by-byte for remaining
    lb      t1, 0(a1)
    sb      t1, 0(a0)
    beqz    t1, copy_done_fast
    addi    a0, a0, 1
    addi    a1, a1, 1
    j       copy_byte_loop
    
copy_done_fast:
    mv      a0, t0
    ret

.global main
main:
    addi    sp, sp, -48
    sd      ra, 40(sp)
    
    li      t0, 0x48
    sb      t0, 0(sp)
    li      t0, 0x45
    sb      t0, 1(sp)
    li      t0, 0x4C
    sb      t0, 2(sp)
    li      t0, 0x4C
    sb      t0, 3(sp)
    li      t0, 0x4F
    sb      t0, 4(sp)
    li      t0, 0
    sb      t0, 5(sp)
    
    addi    a0, sp, 16
    mv      a1, sp
    call    strcpy_fast
    
    lb      t0, 16(sp)
    li      t1, 0x48
    bne     t0, t1, fail
    
    li      a0, 0
    j       exit
fail:
    li      a0, 1
exit:
    ld      ra, 40(sp)
    addi    sp, sp, 48
    ret
""",
            "target": "", "reference": "", "input": "",
            "description": "Memory Optimization: Use word-aligned loads/stores instead of byte-by-byte",
            "optimization_type": "Memory Access Optimization"
        },

        # ========== 5. Register Allocation: Polynomial Evaluation ==========
        {
            "idx": "5",
            "id": "5",
            "query": "",
            "code_v0_no_empty_lines": """# RISC-V Assembly: Polynomial Evaluation (Stack-heavy)
# Evaluates ax^2 + bx + c using repeated memory access
# Input: a0 = x, a1 = a, a2 = b, a3 = c
# Output: a0 = result

.global poly_slow
.text

poly_slow:
    addi    sp, sp, -32
    
    # Store all parameters to stack (inefficient)
    sw      a0, 0(sp)       # x
    sw      a1, 4(sp)       # a
    sw      a2, 8(sp)       # b
    sw      a3, 12(sp)      # c
    
    # Compute x^2
    lw      t0, 0(sp)       # load x
    mul     t0, t0, t0      # x^2
    sw      t0, 16(sp)      # store x^2
    
    # Compute a * x^2
    lw      t0, 4(sp)       # load a
    lw      t1, 16(sp)      # load x^2
    mul     t0, t0, t1      # a * x^2
    sw      t0, 20(sp)      # store
    
    # Compute b * x
    lw      t0, 8(sp)       # load b
    lw      t1, 0(sp)       # load x
    mul     t0, t0, t1      # b * x
    sw      t0, 24(sp)      # store
    
    # Sum all terms
    lw      t0, 20(sp)      # a * x^2
    lw      t1, 24(sp)      # b * x
    lw      t2, 12(sp)      # c
    add     t0, t0, t1
    add     a0, t0, t2      # result
    
    addi    sp, sp, 32
    ret

.global main
main:
    addi    sp, sp, -16
    sd      ra, 8(sp)
    
    # Evaluate 2*3^2 + 4*3 + 5 = 18 + 12 + 5 = 35
    li      a0, 3           # x = 3
    li      a1, 2           # a = 2
    li      a2, 4           # b = 4
    li      a3, 5           # c = 5
    call    poly_slow
    
    li      t0, 35
    bne     a0, t0, fail
    
    li      a0, 0
    j       exit
fail:
    li      a0, 1
exit:
    ld      ra, 8(sp)
    addi    sp, sp, 16
    ret
""",
            "code_v1_no_empty_lines": """# RISC-V Assembly: Polynomial Evaluation (Register-optimized)
# Uses registers instead of stack for intermediate values
# Input: a0 = x, a1 = a, a2 = b, a3 = c
# Output: a0 = result

.global poly_fast
.text

poly_fast:
    # No stack needed - keep everything in registers
    # Using Horner's method: ((a * x) + b) * x + c
    
    mul     t0, a1, a0      # t0 = a * x
    add     t0, t0, a2      # t0 = a * x + b
    mul     t0, t0, a0      # t0 = (a * x + b) * x
    add     a0, t0, a3      # result = (a * x + b) * x + c
    
    ret

.global main
main:
    addi    sp, sp, -16
    sd      ra, 8(sp)
    
    li      a0, 3
    li      a1, 2
    li      a2, 4
    li      a3, 5
    call    poly_fast
    
    li      t0, 35
    bne     a0, t0, fail
    
    li      a0, 0
    j       exit
fail:
    li      a0, 1
exit:
    ld      ra, 8(sp)
    addi    sp, sp, 16
    ret
""",
            "target": "", "reference": "", "input": "",
            "description": "Register Allocation: Use Horner's method and registers instead of stack",
            "optimization_type": "Register Optimization"
        }
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(samples)
    
    print("=" * 60)
    print("Generated 5 RISC-V Assembly Optimization Test Cases")
    print("=" * 60)
    print("\n1. Loop Unrolling (Array Sum)")
    print("   - Slow: 1 element per iteration")
    print("   - Fast: 4 elements per iteration")
    print("\n2. Instruction Scheduling (Dot Product)")
    print("   - Slow: Sequential dependencies")
    print("   - Fast: Interleaved to hide latency")
    print("\n3. Branch Optimization (Max of Two)")
    print("   - Slow: Conditional branch")
    print("   - Fast: Branchless with SLT")
    print("\n4. Memory Optimization (String Copy)")
    print("   - Slow: Byte-by-byte copy")
    print("   - Fast: Word-aligned copy")
    print("\n5. Register Allocation (Polynomial)")
    print("   - Slow: Stack-heavy, many loads/stores")
    print("   - Fast: Horner's method, registers only")
    print("\n" + "=" * 60)
    print(f"Output: {output_file}")

if __name__ == "__main__":
    output_path = "processed_data/riscv/test.jsonl"
    generate_riscv_assembly_samples(output_path)
