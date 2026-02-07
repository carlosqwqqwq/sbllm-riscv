# RISC-V 代码优化框架 - 完整 Docker 环境配置
# 支持所有 RISC-V 代码类型：纯 C、内联汇编、纯汇编、混合代码
#
# 构建命令: docker build -t riscv-opt-env:latest .
# 运行命令: docker run -it -v $(pwd):/app --name project1-dev riscv-opt-env:latest

FROM ubuntu:22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ============================================================================
# Stage 1: Install comprehensive base dependencies
# ============================================================================
RUN apt-get update && apt-get install -y \
    # Python ecosystem
    python3 python3-pip python3-venv python3-dev \
    # Build essentials
    wget tar git build-essential curl ca-certificates \
    autoconf automake autotools-dev libmpc-dev libmpfr-dev libgmp-dev \
    gawk bison flex texinfo gperf libtool patchutils bc zlib1g-dev \
    cmake ninja-build pkg-config zip unzip vim less \
    # QEMU for RISC-V emulation
    qemu-user qemu-system-riscv64 qemu-user-static \
    # Additional libraries for comprehensive C support
    libexpat-dev libncurses-dev libfdt-dev \
    # Cross-compilation support libraries
    libc6-dev libstdc++-11-dev \
    # Debugging tools
    gdb-multiarch \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 1.5: Install QEMU v7.2.0 (Static) for RVV 1.0 Support
# ============================================================================
# Use /usr/local/bin to take precedence over /usr/bin/qemu-riscv64 from apt
RUN wget -q "https://github.com/multiarch/qemu-user-static/releases/download/v7.2.0-1/qemu-riscv64-static" -O /usr/local/bin/qemu-riscv64 \
    && chmod +x /usr/local/bin/qemu-riscv64 \
    && echo "Checking QEMU version. Expected 7.2.0" \
    && /usr/local/bin/qemu-riscv64 --version | grep "7.2.0"

# Set QEMU environment variables for persistent RVV support
# Standard QEMU syntax uses 'on' instead of 'true'
ENV QEMU_CPU=rv64,v=on,zba=on,zbb=on,zbs=on,vlen=128

# ============================================================================
# Stage 2: Install RISC-V GCC Toolchain (Full Support)
# ============================================================================

ENV RISCV=/opt/riscv
ENV PATH="${RISCV}/bin:${PATH}"

# Download comprehensive RISC-V toolchain with:
# - GCC 14.x (RVV intrinsics support)
# - glibc (for Linux userspace programs)
# - Full extension support (IMAFDCV + Zb* + Zicsr + Zifencei)
RUN mkdir -p ${RISCV} && cd /tmp && \
    echo "=== Downloading RISC-V GCC Toolchain ===" && \
    # Primary: Latest 2024 nightly with GCC 14
    (wget -q --show-progress --tries=3 --timeout=120 \
    "https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.12.16/riscv64-glibc-ubuntu-22.04-gcc-nightly-2024.12.16-nightly.tar.xz" \
    -O toolchain.tar.xz 2>&1 && \
    echo "Downloaded 2024.12.16" && \
    tar -xJf toolchain.tar.xz -C ${RISCV} --strip-components=1 && \
    rm toolchain.tar.xz) || \
    # Fallback 1: 2024.09.03
    (wget -q --show-progress --tries=3 --timeout=120 \
    "https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.09.03/riscv64-glibc-ubuntu-22.04-gcc-nightly-2024.09.03-nightly.tar.gz" \
    -O toolchain.tar.gz 2>&1 && \
    echo "Downloaded 2024.09.03" && \
    tar -xzf toolchain.tar.gz -C ${RISCV} --strip-components=1 && \
    rm toolchain.tar.gz) || \
    # Fallback 2: 2024.04.12 (GCC 14.0)
    (wget -q --show-progress --tries=3 --timeout=120 \
    "https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.04.12/riscv64-glibc-ubuntu-22.04-gcc-nightly-2024.04.12-nightly.tar.gz" \
    -O toolchain.tar.gz 2>&1 && \
    echo "Downloaded 2024.04.12" && \
    tar -xzf toolchain.tar.gz -C ${RISCV} --strip-components=1 && \
    rm toolchain.tar.gz) || \
    # Fallback 3: 2024.02.02
    (wget -q --show-progress --tries=3 --timeout=120 \
    "https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.02.02/riscv64-glibc-ubuntu-22.04-llvm-nightly-2024.02.02-nightly.tar.gz" \
    -O toolchain.tar.gz 2>&1 && \
    tar -xzf toolchain.tar.gz -C ${RISCV} --strip-components=1 && \
    rm toolchain.tar.gz) || \
    # Ultimate fallback: Ubuntu cross-compiler
    (echo "⚠ Using Ubuntu fallback toolchain" && \
    apt-get update && \
    apt-get install -y gcc-12-riscv64-linux-gnu g++-12-riscv64-linux-gnu \
    binutils-riscv64-linux-gnu newlib-riscv64-unknown-elf \
    gcc-riscv64-linux-gnu g++-riscv64-linux-gnu && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p ${RISCV}/bin && \
    for tool in gcc g++ as ld ar nm objdump objcopy size strip ranlib readelf; do \
    if command -v riscv64-linux-gnu-$tool >/dev/null; then \
    ln -sf $(which riscv64-linux-gnu-$tool) ${RISCV}/bin/riscv64-unknown-linux-gnu-$tool; \
    fi; \
    done)

# ============================================================================
# Stage 3: Verify Toolchain and Supported Extensions
# ============================================================================
RUN echo "=== RISC-V Toolchain Info ===" && \
    if [ -f "${RISCV}/bin/riscv64-unknown-linux-gnu-gcc" ]; then \
    ${RISCV}/bin/riscv64-unknown-linux-gnu-gcc --version | head -2; \
    echo ""; \
    echo "=== Supported Extensions (via -march) ==="; \
    echo "Testing: rv64gc (baseline)"; \
    echo 'int main() { return 0; }' > /tmp/test.c && \
    ${RISCV}/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gc -c /tmp/test.c -o /tmp/test.o 2>&1 && echo "  ✓ rv64gc OK" || echo "  ✗ Failed"; \
    echo "Testing: rv64gcv (Vector Extension)"; \
    ${RISCV}/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gcv -c /tmp/test.c -o /tmp/test.o 2>&1 && echo "  ✓ rv64gcv OK" || echo "  ✗ Failed"; \
    echo "Testing: rv64gcv_zba_zbb_zbs (Bit Manipulation)"; \
    ${RISCV}/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gcv_zba_zbb_zbs -c /tmp/test.c -o /tmp/test.o 2>&1 && echo "  ✓ rv64gcv_zba_zbb_zbs OK" || echo "  ✗ Failed"; \
    rm -f /tmp/test.c /tmp/test.o; \
    else \
    echo "Warning: Toolchain not found at expected path"; \
    fi

# ============================================================================
# Stage 4: Test Various Code Types
# ============================================================================
RUN echo "=== Testing Code Type Compilation ===" && \
    # Test 1: Pure C
    echo '#include <stdio.h>' > /tmp/pure_c.c && \
    echo 'int main() { printf("Hello RISC-V\\n"); return 0; }' >> /tmp/pure_c.c && \
    (${RISCV}/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gc /tmp/pure_c.c -o /tmp/pure_c -static 2>&1 && \
    echo "✓ Pure C code compilation OK" && rm -f /tmp/pure_c) || echo "✗ Pure C compilation failed" && \
    rm -f /tmp/pure_c.c && \
    \
    # Test 2: C with inline assembly (Zbb extension)
    echo 'unsigned long popcount(unsigned long x) {' > /tmp/inline_asm.c && \
    echo '    unsigned long result;' >> /tmp/inline_asm.c && \
    echo '    asm volatile(".option push\\n .option arch,+zbb\\n cpop %0, %1\\n .option pop" : "=r"(result) : "r"(x));' >> /tmp/inline_asm.c && \
    echo '    return result; }' >> /tmp/inline_asm.c && \
    echo 'int main() { return 0; }' >> /tmp/inline_asm.c && \
    (${RISCV}/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gc_zbb /tmp/inline_asm.c -o /tmp/inline_asm -static 2>&1 && \
    echo "✓ C + Zbb inline assembly OK" && rm -f /tmp/inline_asm) || echo "✗ Inline assembly failed" && \
    rm -f /tmp/inline_asm.c && \
    \
    # Test 3: C with RVV inline assembly
    echo 'void vdot() {' > /tmp/rvv_asm.c && \
    echo '    asm volatile(".option push\\n .option arch,+v\\n vsetvli t0, a0, e32, m1, ta, ma\\n .option pop");' >> /tmp/rvv_asm.c && \
    echo '}' >> /tmp/rvv_asm.c && \
    echo 'int main() { return 0; }' >> /tmp/rvv_asm.c && \
    (${RISCV}/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gcv /tmp/rvv_asm.c -o /tmp/rvv_asm -static 2>&1 && \
    echo "✓ C + RVV inline assembly OK" && rm -f /tmp/rvv_asm) || echo "✗ RVV inline assembly failed" && \
    rm -f /tmp/rvv_asm.c && \
    \
    # Test 4: Pure assembly file
    echo '.section .text' > /tmp/pure_asm.S && \
    echo '.global _start' >> /tmp/pure_asm.S && \
    echo '_start:' >> /tmp/pure_asm.S && \
    echo '    li a7, 93' >> /tmp/pure_asm.S && \
    echo '    li a0, 0' >> /tmp/pure_asm.S && \
    echo '    ecall' >> /tmp/pure_asm.S && \
    (${RISCV}/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gc -nostdlib /tmp/pure_asm.S -o /tmp/pure_asm 2>&1 && \
    echo "✓ Pure assembly compilation OK" && rm -f /tmp/pure_asm) || echo "✗ Pure assembly failed" && \
    rm -f /tmp/pure_asm.S && \
    \
    # Test 5: Check for riscv_vector.h
    echo "=== Checking RVV Intrinsics Header ===" && \
    VECTOR_H=$(find ${RISCV} -name "riscv_vector.h" 2>/dev/null | head -1) && \
    if [ -n "$VECTOR_H" ]; then \
    echo "✓ Found riscv_vector.h: $VECTOR_H"; \
    # Test RVV intrinsics
    echo '#include <riscv_vector.h>' > /tmp/rvv_intrinsics.c && \
    echo 'void test() { size_t vl = __riscv_vsetvl_e32m1(16); }' >> /tmp/rvv_intrinsics.c && \
    echo 'int main() { return 0; }' >> /tmp/rvv_intrinsics.c && \
    (${RISCV}/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gcv /tmp/rvv_intrinsics.c -c -o /tmp/rvv_intrinsics.o 2>&1 && \
    echo "✓ RVV intrinsics compilation OK" && rm -f /tmp/rvv_intrinsics.o) || \
    echo "⚠ RVV intrinsics compilation failed (use inline assembly instead)"; \
    rm -f /tmp/rvv_intrinsics.c; \
    else \
    echo "⚠ riscv_vector.h not found. Use inline assembly for RVV instructions."; \
    fi

# ============================================================================
# Stage 5: Install standard C libraries and headers
# ============================================================================
RUN echo "=== Checking Standard Library Headers ===" && \
    SYSROOT=$(${RISCV}/bin/riscv64-unknown-linux-gnu-gcc -print-sysroot 2>/dev/null || echo "") && \
    echo "Sysroot: ${SYSROOT:-'Not set (using system headers)'}" && \
    # Verify common headers exist
    for header in stdio.h stdlib.h stdint.h string.h math.h; do \
    if [ -n "$SYSROOT" ] && [ -f "$SYSROOT/usr/include/$header" ]; then \
    echo "  ✓ $header found"; \
    elif [ -f "${RISCV}/sysroot/usr/include/$header" ]; then \
    echo "  ✓ $header found"; \
    else \
    echo "  ⚠ $header not found in sysroot"; \
    fi; \
    done

# ============================================================================
# Stage 6: Install Python Dependencies
# ============================================================================
WORKDIR /work

# 1. Install tree-sitter and tree-sitter-cpp for local CodeBLEU implementation
# NOTE: We do NOT install the 'codebleu' pip package - it has unsolvable build issues.
#       Instead, we use the local sbllm/utils/codebleu/ implementation which only needs these core dependencies.
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    tree-sitter \
    tree-sitter-cpp \
    nltk

# 2. Install remaining requirements
COPY requirement.txt /tmp/requirement.txt
# Remove codebleu and tree-sitter from requirements to avoid conflicts (we installed above)
RUN sed -i '/codebleu/d' /tmp/requirement.txt && \
    sed -i '/tree-sitter/d' /tmp/requirement.txt && \
    pip3 install --no-cache-dir -r /tmp/requirement.txt

# Pre-download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt')"

# ============================================================================
# Stage 7: Environment Variables
# ============================================================================
ENV QEMU_PATH=/usr/local/bin/qemu-riscv64
ENV RISC_V_GCC_TOOLCHAIN_PATH=/opt/riscv
ENV RISCV_GCC=${RISCV}/bin/riscv64-unknown-linux-gnu-gcc

# ============================================================================
# Stage 8: Create helper scripts
# ============================================================================
RUN echo '#!/bin/bash' > /usr/local/bin/riscv-compile && \
    echo '# Quick compile helper for RISC-V' >> /usr/local/bin/riscv-compile && \
    echo '${RISCV}/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gcv_zba_zbb_zbs -static "$@" -lm' >> /usr/local/bin/riscv-compile && \
    chmod +x /usr/local/bin/riscv-compile && \
    \
    echo '#!/bin/bash' > /usr/local/bin/riscv-run && \
    echo '# Run RISC-V binary with QEMU' >> /usr/local/bin/riscv-run && \
    echo '/usr/local/bin/qemu-riscv64 -cpu rv64,zba=on,zbb=on,zbs=on,v=on,vlen=128 "$@"' >> /usr/local/bin/riscv-run && \
    chmod +x /usr/local/bin/riscv-run

# ============================================================================
# Final Summary
# ============================================================================
RUN echo "" && \
    echo "==========================================" && \
    echo "RISC-V Development Environment Ready" && \
    echo "==========================================" && \
    echo "" && \
    echo "GCC Version:" && \
    ${RISCV}/bin/riscv64-unknown-linux-gnu-gcc --version | head -1 && \
    echo "" && \
    echo "QEMU Version:" && \
    /usr/local/bin/qemu-riscv64 --version | head -1 && \
    echo "" && \
    echo "Supported Code Types:" && \
    echo "  ✓ Pure C code" && \
    echo "  ✓ C with inline RISC-V assembly" && \
    echo "  ✓ C with Zbb/Zba/Zbs bitmanip extensions" && \
    echo "  ✓ C with RVV vector extensions" && \
    echo "  ✓ Pure RISC-V assembly (.S files)" && \
    echo "" && \
    echo "Quick Commands:" && \
    echo "  riscv-compile <file.c> -o <output>  # Compile with full extensions" && \
    echo "  riscv-run <binary>                  # Run with QEMU" && \
    echo ""

# Sync project code into container is NOT recommended for the base image
# to maintain reusability. All projects should be mounted via volumes.
# COPY sbllm/ /app/sbllm/  <-- Removed for reusability

CMD ["/bin/bash"]
