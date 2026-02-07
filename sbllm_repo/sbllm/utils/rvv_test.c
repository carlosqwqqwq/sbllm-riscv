#include <riscv_vector.h>
#include <stdio.h>

int main() {
    size_t avl = 10;
    size_t vl = __riscv_vsetvl_e32m1(avl);
    printf("Vector length: %zu\n", vl);
    return 0;
}
