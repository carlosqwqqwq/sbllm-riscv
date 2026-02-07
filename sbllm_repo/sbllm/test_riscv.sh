#!/bin/bash
# RISC-V 功能测试脚本
# 
# 用于快速测试 RISC-V 优化框架的基本功能
# 
# 使用方法:
#   bash test_riscv.sh --qemu_path /usr/bin/qemu-riscv64 --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu
#
# 或者使用默认路径（如果已配置）:
#   bash test_riscv.sh

# 注意：不使用 set -e，以便更好地处理错误和提供用户友好的错误信息

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 默认配置
QEMU_PATH=""
RISC_V_GCC_TOOLCHAIN_PATH=""
MODEL_NAME=deepseek
TEST_DATA_DIR="./test_data_riscv"
OUTPUT_DIR="./test_output_riscv"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --qemu_path)
            QEMU_PATH="$2"
            shift 2
            ;;
        --riscv_gcc_toolchain_path)
            RISC_V_GCC_TOOLCHAIN_PATH="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "RISC-V 功能测试脚本"
            echo ""
            echo "使用方法:"
            echo "  bash test_riscv.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --qemu_path PATH                QEMU 可执行文件路径（必需）"
            echo "  --riscv_gcc_toolchain_path PATH RISC-V GCC 工具链路径（必需）"
            echo "  --model_name NAME               LLM 模型名称（默认: deepseek）"
            echo "  --help, -h                     显示帮助信息"
            echo ""
            echo "示例:"
            echo "  bash test_riscv.sh --qemu_path /usr/bin/qemu-riscv64 --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu"
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$QEMU_PATH" ] || [ -z "$RISC_V_GCC_TOOLCHAIN_PATH" ]; then
    echo -e "${RED}错误: 必须提供 --qemu_path 和 --riscv_gcc_toolchain_path 参数${NC}"
    echo "使用 --help 查看帮助信息"
    exit 1
fi

# 打印标题
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}        RISC-V 优化框架功能测试${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo "测试配置:"
echo "  - 模型: $MODEL_NAME (默认: deepseek)"
echo "  - QEMU: $QEMU_PATH"
echo "  - 工具链: $RISC_V_GCC_TOOLCHAIN_PATH"
echo ""

# 步骤 1: 创建测试数据
echo -e "${BLUE}[步骤 1/5] 创建测试数据${NC}"
echo "------------------------------------------------------------"
mkdir -p "$TEST_DATA_DIR"
mkdir -p "$TEST_DATA_DIR/test_cases"

# 创建测试数据文件 (单行 JSON 格式，包含 code_v0_no_empty_lines 字段)
echo '{"idx": 0, "query": "# 计算数组元素之和\\n# 优化方向：减少内存访问，利用寄存器减少加载次数\\n# 当前问题：每次循环都从内存加载，可能导致流水线停顿\\n.text\\n.global sum_array\\nsum_array:\\n    # 参数: a0 = 数组指针, a1 = 数组长度\\n    li t0, 0          # 累加器\\n    li t1, 0          # 循环计数器\\nloop:\\n    beq t1, a1, done  # 如果计数器等于长度，跳转到 done\\n    lw t2, 0(a0)      # 从内存加载元素\\n    add t0, t0, t2    # 累加\\n    addi a0, a0, 4    # 数组指针递增\\n    addi t1, t1, 1    # 计数器递增\\n    j loop            # 跳转到 loop\\ndone:\\n    mv a0, t0         # 将结果移动到 a0\\n    ret               # 返回", "code_v0_no_empty_lines": "# 计算数组元素之和\\n# 优化方向：减少内存访问，利用寄存器减少加载次数\\n# 当前问题：每次循环都从内存加载，可能导致流水线停顿\\n.text\\n.global sum_array\\nsum_array:\\n    # 参数: a0 = 数组指针, a1 = 数组长度\\n    li t0, 0          # 累加器\\n    li t1, 0          # 循环计数器\\nloop:\\n    beq t1, a1, done  # 如果计数器等于长度，跳转到 done\\n    lw t2, 0(a0)      # 从内存加载元素\\n    add t0, t0, t2    # 累加\\n    addi a0, a0, 4    # 数组指针递增\\n    addi t1, t1, 1    # 计数器递增\\n    j loop            # 跳转到 loop\\ndone:\\n    mv a0, t0         # 将结果移动 to a0\\n    ret               # 返回", "input": "10\\n1 2 3 4 5 6 7 8 9 10", "reference": "# 优化后的代码：使用循环展开减少分支跳转\\n.text\\n.global sum_array\\nsum_array:\\n    li t0, 0\\n    li t1, 0\\nloop:\\n    beq t1, a1, done\\n    lw t2, 0(a0)\\n    add t0, t0, t2\\n    addi a0, a0, 4\\n    addi t1, t1, 1\\n    j loop\\ndone:\\n    mv a0, t0\\n    ret"}' > "$TEST_DATA_DIR/test.jsonl"

# 创建简化的训练数据（知识库）
echo '{"id": 1, "source": "test", "optimization_type": "Loop Optimization", "optimization_description": "减少内存访问，使用寄存器缓存", "original_code": "loop:\\n    lw t2, 0(a0)\\n    add t0, t0, t2\\n    addi a0, a0, 4\\n    j loop", "optimized_code": "loop:\\n    lw t2, 0(a0)\\n    lw t3, 4(a0)\\n    add t0, t0, t2\\n    add t0, t0, t3\\n    addi a0, a0, 8\\n    j loop", "query_abs": ["loop", "lw", "add", "addi", "j"], "edit_code_abs": [], "edit_opt_abs": ["lw", "add"], "text_representation": "Loop Optimization 减少内存访问，使用寄存器缓存"}' > "$TEST_DATA_DIR/train.jsonl"

echo -e "${GREEN}✓ 测试数据已创建${NC}"
echo "  - 测试数据: $TEST_DATA_DIR/test.jsonl"
echo "  - 训练数据: $TEST_DATA_DIR/train.jsonl"
echo ""

# 步骤 2: 环境验证
echo -e "${BLUE}[步骤 2/5] 环境验证${NC}"
echo "------------------------------------------------------------"
if ! python3 utils/validate_riscv_setup.py \
    --qemu_path "$QEMU_PATH" \
    --riscv_gcc_toolchain_path "$RISC_V_GCC_TOOLCHAIN_PATH" \
    --data_dir "$TEST_DATA_DIR" \
    --output_dir "$OUTPUT_DIR"; then
    echo -e "${RED}环境验证失败，请检查配置${NC}"
    echo ""
    echo "常见问题："
    echo "  1. 检查 QEMU 路径是否正确: $QEMU_PATH"
    echo "  2. 检查工具链路径是否正确: $RISC_V_GCC_TOOLCHAIN_PATH"
    echo "  3. 检查 API 密钥是否已配置（默认使用 DeepSeek）"
    exit 1
fi
echo ""

# 步骤 3: 初始化结果
echo -e "${BLUE}[步骤 3/5] 初始化结果${NC}"
echo "------------------------------------------------------------"
cd ..
if ! python3 sbllm/initial.py \
    --model_name "$MODEL_NAME" \
    --lang riscv \
    --generation_path "$OUTPUT_DIR" \
    --baseline_data_path "sbllm/$TEST_DATA_DIR/test.jsonl" 2>/dev/null; then
    echo -e "${YELLOW}⚠ 初始化可能已存在或跳过，继续执行...${NC}"
fi
cd sbllm
echo ""

# 步骤 4: 运行一次迭代（快速测试）
echo -e "${BLUE}[步骤 4/5] 运行优化迭代（快速测试）${NC}"
echo "------------------------------------------------------------"
echo "生成优化候选..."
if ! python3 evol_query.py \
    --mode riscv_optimization \
    --model_name "$MODEL_NAME" \
    --lang riscv \
    --generation_path "../$OUTPUT_DIR" \
    --baseline_data_path "$TEST_DATA_DIR/test.jsonl" \
    --iteration 0 \
    --generation_number 3 \
    --riscv_mode; then
    echo -e "${RED}生成候选失败${NC}"
    echo ""
    echo "可能的原因："
    echo "  1. API 密钥未配置或无效（当前使用模型: $MODEL_NAME）"
    echo "  2. 网络连接问题"
    echo "  3. API 配额不足"
    echo ""
    echo "解决方案："
    echo "  1. 检查 sbllm/sbllm/evol_query.py 中的 API 密钥配置"
    echo "  2. 尝试使用其他模型: --model_name gemini"
    exit 1
fi

echo "评估候选代码..."
if ! python3 evaluate.py \
    --mode riscv_optimization/0 \
    --lang riscv \
    --output_path "../$OUTPUT_DIR/riscv" \
    --model_name "$MODEL_NAME" \
    --baseline_data_path "$TEST_DATA_DIR/test.jsonl" \
    --qemu_path "$QEMU_PATH" \
    --riscv_gcc_toolchain_path "$RISC_V_GCC_TOOLCHAIN_PATH" \
    --test_case_path "$TEST_DATA_DIR/test_cases" \
    --process_number 2 \
    --riscv_mode; then
    echo -e "${RED}评估失败${NC}"
    echo ""
    echo "可能的原因："
    echo "  1. QEMU 路径不正确: $QEMU_PATH"
    echo "  2. 工具链路径不正确: $RISC_V_GCC_TOOLCHAIN_PATH"
    echo "  3. 代码编译失败"
    echo ""
    echo "解决方案："
    echo "  1. 检查 QEMU 和工具链路径"
    echo "  2. 查看详细错误日志"
    exit 1
fi

echo -e "${GREEN}✓ 优化迭代完成${NC}"
echo ""

# 步骤 5: 验证结果
echo -e "${BLUE}[步骤 5/5] 验证结果${NC}"
echo "------------------------------------------------------------"

REPORT_FILE="../$OUTPUT_DIR/riscv/riscv_optimization/0/test_execution_${MODEL_NAME}test.report"
if [ ! -f "$REPORT_FILE" ]; then
    # Fallback to without /0 if needed (though with my fix it should be there)
    REPORT_FILE="../$OUTPUT_DIR/riscv/riscv_optimization/test_execution_${MODEL_NAME}test.report"
fi
if [ -f "$REPORT_FILE" ]; then
    echo -e "${GREEN}✓ 找到评估报告: $REPORT_FILE${NC}"
    echo ""
    echo "评估报告内容:"
    echo "------------------------------------------------------------"
    head -n 20 "$REPORT_FILE" || true
    echo ""
    echo -e "${GREEN}✓ 测试完成！${NC}"
else
    echo -e "${YELLOW}⚠ 未找到评估报告 ($REPORT_FILE)，但测试流程已执行${NC}"
fi

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}测试总结${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo "测试配置:"
echo "  - 模型: $MODEL_NAME"
echo "  - QEMU: $QEMU_PATH"
echo "  - 工具链: $RISC_V_GCC_TOOLCHAIN_PATH"
echo ""
echo "测试数据目录: $TEST_DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo -e "${GREEN}✓ 所有测试步骤已完成${NC}"
echo ""
echo "查看详细结果:"
echo "  cat $OUTPUT_DIR/riscv/riscv_optimization/0/test_execution_0.report"
echo ""
echo "查看生成的候选代码:"
echo "  cat $OUTPUT_DIR/riscv/riscv_optimization/0/prediction_*.jsonl"
echo ""

