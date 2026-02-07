# 根因分析报告 (Code Analyze Report)

## 1. 根本原因 (Root Cause)

**导致问题的最深层次原因：配置同步的不完整性与项目原有硬编码逻辑的冲突。**

- **现象**：`qemu-riscv64-fixed: not found`。
- **底层逻辑**：
  - `rvv-bench/bench/Makefile` 在执行 `make run` 时，硬编码调用了父目录的 `../run.sh`。
  - 在先前的优化中，我们将 QEMU 移入了 Docker 镜像内部 (`/usr/bin/qemu-riscv64`)，但未同步更新项目目录下的 `run.sh` 文件。
  - `run.sh` 中残留的内容依然指向挂载路径 `/work/qemu-riscv64-fixed`，导致执行失败。
  - 尽管我们在 `config.mk` 中定义了 `RUN_WRAPPER`，但 `Makefile` 的 `run` 目标并未使用该变量，造成了配置项的“虚假更新”。

## 2. 系统影响 (System Impact)

此问题并非孤立，它暴露了**环境迁移中的软链接/辅助脚本脆弱性**：

- 当系统架构从“依赖宿主机二进制”转向“全容器化”时，如果只关注主构建工具（如 GCC），而忽略了周边的辅助脚本（如 `run.sh`），会导致系统在运行时崩溃。
- 该不良模式（硬编码辅助脚本路径）在许多小型 Benchmark 项目中普遍存在，必须有一种机制确保这些脚本在容器启动时被动态重置或覆盖。

## 3. 设计可持续的解决方案

### 推荐方案：动态重构运行代理 (Dynamic Runner Refactoring)

**方案核心**：由 `run_rvv_bench.sh` 承担“环境自愈”职责，在每次运行前强制刷新 `run.sh`。

- **根治程度**：彻底消除对外部挂载 QEMU 的依赖。
- **泛化能力**：该模式可以推广到任何需要 shim/wrapper 的环境，确保脚本始终与当前容器内的路径一致。
- **改动范围**：仅需微调 `rvv-bench/run_rvv_bench.sh`，使其在生成 `config.mk` 的同时生成 `run.sh`。

### 详细设计

1. **强制覆盖 `run.sh`**：在 `run_rvv_bench.sh` 中添加 `cat > run.sh` 逻辑。
2. **利用环境变量**：新 `run.sh` 将直接调用 `qemu-riscv64`，其旗标（Flag）由 Dockerfile 中的 `ENV QEMU_CPU` 提供，实现解耦。

## 4. 实施考量与验证

- **向后兼容**：不改变 `Makefile` 调用 `run.sh` 的习惯。
- **性能影响**：无，仅为路径转发。
- **验证策略**：
  - 运行 `run_all.bat` 后，检查 `rvv-bench/run.sh` 的内容。
  - 验证 `make -C bench run` 是否成功输出结果。
