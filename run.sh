#!/bin/sh
# QEMU Execution Wrapper
qemu-riscv64 -cpu rv64,x-v=true,x-zba=true,x-zbb=true,x-zbs=true "$@"
