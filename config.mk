WARN=-Wall -Wextra -Wno-unused-function -Wno-unused-parameter
CC=riscv64-unknown-linux-gnu-gcc
CFLAGS=-march=rv64gcv -O3 -static ${WARN}
