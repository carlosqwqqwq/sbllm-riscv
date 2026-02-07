#!/bin/bash
# RISC-V 代码优化框架运行脚本
# 
# 使用方法:
#   bash run_riscv.sh
#
# 或者指定参数:
#   bash run_riscv.sh --qemu_path /usr/bin/qemu-riscv64 --riscv_gcc_toolchain_path /opt/riscv

set -e  # 遇到错误立即退出

# 颜色定义 (仅当输出到终端时使用)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    GREEN=''
    RED=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# 默认配置
LANG=riscv
MODEL_NAME=deepseek
GENERATION_NUMBER=5
BEAM_NUMBER=3
BASE_MODE=riscv_optimization
ITERATIONS=2
API_IDX=0
RESTART_POS=0
PROCESS_NUMBER=5
NO_IMPROVEMENT_THRESHOLD=3  # Stop if no improvement for this many consecutive rounds

# 路径配置
GENERATION_PATH=../output
DATA_DIR=../processed_data/$LANG
OUTPUT_PATH=../output/$LANG
BASELINE_DATA_PATH=$DATA_DIR/test.jsonl
TRAINING_DATA_PATH=$DATA_DIR/train.jsonl
PUBLIC_TEST_CASE_PATH=$DATA_DIR/test_cases
PRIVATE_TEST_CASE_PATH=$DATA_DIR/test_cases

# RISC-V 特定配置（需要用户配置）
QEMU_PATH=""
RISC_V_GCC_TOOLCHAIN_PATH=""
RETRIEVAL_METHOD=bm25
HYBRID_ALPHA=0.5
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_INDEX_PATH=""

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
        --generation_number)
            GENERATION_NUMBER="$2"
            shift 2
            ;;
        --beam_number)
            BEAM_NUMBER="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_PATH="$2"
            GENERATION_PATH="$2"
            shift 2
            ;;
        --retrieval_method)
            RETRIEVAL_METHOD="$2"
            shift 2
            ;;
        --vector_index_path)
            VECTOR_INDEX_PATH="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用方法: bash run_riscv.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --qemu_path PATH                    QEMU 可执行文件路径（必需）"
            echo "  --riscv_gcc_toolchain_path PATH     RISC-V GCC 工具链路径（必需）"
            echo "  --model_name NAME                  模型名称 (默认: deepseek, 可选: chatgpt/gpt4/gemini/deepseek/codellama)"
            echo "  --generation_number N               每次生成的候选数量"
            echo "  --beam_number N                    beam 搜索数量"
            echo "  --iterations N                     迭代次数"
            echo "  --data_dir PATH                     数据目录路径"
            echo "  --output_dir PATH                   输出目录路径"
            echo "  --retrieval_method METHOD           检索方法 (bm25/vector/hybrid)"
            echo "  --vector_index_path PATH           向量索引文件路径"
            exit 1
            ;;
    esac
done

# 打印配置信息
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RISC-V 代码优化框架${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}配置信息:${NC}"
echo "  语言: $LANG"
echo "  模型: $MODEL_NAME (默认: deepseek)"
echo "  生成数量: $GENERATION_NUMBER"
echo "  Beam 数量: $BEAM_NUMBER"
echo "  迭代次数: $ITERATIONS"
echo "  数据目录: $DATA_DIR"
echo "  输出目录: $OUTPUT_PATH"
echo "  检索方法: $RETRIEVAL_METHOD"
if [ -n "$VECTOR_INDEX_PATH" ]; then
    echo "  向量索引: $VECTOR_INDEX_PATH"
fi
echo ""

# 验证必需参数
if [ -z "$QEMU_PATH" ]; then
    echo -e "${RED}错误: 必须指定 --qemu_path${NC}"
    exit 1
fi

if [ -z "$RISC_V_GCC_TOOLCHAIN_PATH" ]; then
    echo -e "${RED}错误: 必须指定 --riscv_gcc_toolchain_path${NC}"
    exit 1
fi

# 运行环境验证
echo -e "${YELLOW}运行环境验证...${NC}"
python3 utils/validate_riscv_setup.py \
    --qemu_path "$QEMU_PATH" \
    --riscv_gcc_toolchain_path "$RISC_V_GCC_TOOLCHAIN_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_PATH" \
    --evol_query_file "evol_query.py"

if [ $? -ne 0 ]; then
    echo -e "${RED}环境验证失败，请修复问题后重试${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}环境验证通过！${NC}"
echo ""

# 更新路径
BASELINE_DATA_PATH=$DATA_DIR/test.jsonl
TRAINING_DATA_PATH=$DATA_DIR/train.jsonl
PUBLIC_TEST_CASE_PATH=$DATA_DIR/test_cases
PRIVATE_TEST_CASE_PATH=$DATA_DIR/test_cases

# Clean old output if not restarting
if [ "$RESTART_POS" -eq 0 ]; then
    echo -e "${YELLOW}Cleaning output directory...${NC}"
    rm -rf "$OUTPUT_PATH/$BASE_MODE"
    rm -f "$OUTPUT_PATH/initial_results_$MODEL_NAME.jsonl"
    rm -rf "$OUTPUT_PATH/cot"
    rm -f "$OUTPUT_PATH/$BASE_MODE/convergence_state.json"  # Reset convergence state
    rm -f "$OUTPUT_PATH/llm_interaction.log"  # Clear LLM interaction log
fi

# Set LLM interaction log path
export LLM_INTERACTION_LOG="$OUTPUT_PATH/llm_interaction.log"

# 创建必要的目录
mkdir -p "$OUTPUT_PATH/$BASE_MODE"
mkdir -p "$OUTPUT_PATH/$BASE_MODE/qemu_temp"
mkdir -p "$PUBLIC_TEST_CASE_PATH"  # Ensure test_cases directory exists

# 步骤 1: 初始化（如果尚未完成）
if [ ! -f "$OUTPUT_PATH/initial_results_$MODEL_NAME.jsonl" ]; then
    echo -e "${YELLOW}步骤 1: 初始化结果...${NC}"
    python3 initial.py --model_name $MODEL_NAME --lang $LANG
    if [ $? -ne 0 ]; then
        echo -e "${RED}初始化失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}初始化完成${NC}"
    echo ""
fi

# 准备初始结果文件
if [ ! -d "$OUTPUT_PATH/$BASE_MODE/0" ]; then
    echo -e "${YELLOW}准备初始结果文件...${NC}"
    mkdir -p "$OUTPUT_PATH/$BASE_MODE/1"
    if [ -f "$OUTPUT_PATH/initial_results_$MODEL_NAME.jsonl" ]; then
        cp "$OUTPUT_PATH/initial_results_$MODEL_NAME.jsonl" "$OUTPUT_PATH/$BASE_MODE/results.jsonl"
    fi
    if [ -d "$OUTPUT_PATH/cot" ]; then
        cp -r "$OUTPUT_PATH/cot" "$OUTPUT_PATH/$BASE_MODE/"
        mv "$OUTPUT_PATH/$BASE_MODE/cot" "$OUTPUT_PATH/$BASE_MODE/0"
    fi
    echo -e "${GREEN}初始结果文件准备完成${NC}"
    echo ""
fi

# 主循环：迭代优化
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}开始迭代优化（共 $ITERATIONS 轮）${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for i in $(seq 0 $ITERATIONS); do
    echo -e "${YELLOW}----------------------------------------${NC}"
    echo -e "${YELLOW}迭代 $i / $ITERATIONS${NC}"
    echo -e "${YELLOW}----------------------------------------${NC}"
    echo ""
    
    # 步骤 2: 生成优化候选
    echo -e "${BLUE}[迭代 $i] 步骤 1: 生成优化候选...${NC}"
    python3 evol_query.py \
        --from_file \
        --mode $BASE_MODE \
        --generation_path $GENERATION_PATH \
        --model_name $MODEL_NAME \
        --baseline_data_path $BASELINE_DATA_PATH \
        --api_idx $API_IDX \
        --restart_pos $RESTART_POS \
        --generation_number $GENERATION_NUMBER \
        --iteration $i \
        --beam_number $BEAM_NUMBER \
        --lang $LANG \
        --process_number $PROCESS_NUMBER \
        --riscv_mode
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[迭代 $i] 生成候选失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}[迭代 $i] 候选生成完成${NC}"
    echo ""
    
    # 步骤 3: 评估候选代码（使用 QEMU）
    echo -e "${BLUE}[迭代 $i] 步骤 2: 评估候选代码（QEMU）...${NC}"
    python3 evaluate.py \
        --do_train \
        --mode "$i" \
        --lang $LANG \
        --process_number $PROCESS_NUMBER \
        --output_path "$OUTPUT_PATH/$BASE_MODE" \
        --model_name $MODEL_NAME \
        --slice 1 \
        --testing_number 0 \
        --test_case_path $PUBLIC_TEST_CASE_PATH \
        --qemu_path "$QEMU_PATH" \
        --riscv_gcc_toolchain_path "$RISC_V_GCC_TOOLCHAIN_PATH" \
        --riscv_mode
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[迭代 $i] 评估失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}[迭代 $i] 评估完成${NC}"
    echo ""
    
    # 步骤 4: 知识库检索和合并
    if [ $i -gt 0 ]; then
        echo -e "${BLUE}[迭代 $i] 步骤 3: 知识库检索和合并...${NC}"
        
        MERGE_CMD="python3 merge.py \
            --iteration $i \
            --generation_number $GENERATION_NUMBER \
            --lang $LANG \
            --model_name $MODEL_NAME \
            --mode $BASE_MODE \
            --training_data_path $TRAINING_DATA_PATH \
            --generation_path $GENERATION_PATH \
            --beam_number $BEAM_NUMBER \
            --retrieval_method $RETRIEVAL_METHOD \
            --hybrid_alpha $HYBRID_ALPHA \
            --embedding_model $EMBEDDING_MODEL"
        
        if [ -n "$VECTOR_INDEX_PATH" ]; then
            MERGE_CMD="$MERGE_CMD --vector_index_path $VECTOR_INDEX_PATH"
        fi
        
        eval $MERGE_CMD
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}[迭代 $i] 知识库检索失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}[迭代 $i] 知识库检索完成${NC}"
        echo ""
        
        # Check for convergence
        CONVERGENCE_FILE="$OUTPUT_PATH/$BASE_MODE/convergence_state.json"
        if [ -f "$CONVERGENCE_FILE" ]; then
            NO_IMPROVEMENT=$(python3 -c "import json; f=open('$CONVERGENCE_FILE'); d=json.load(f); print(d.get('no_improvement_count', 0))")
            BEST_SPEEDUP=$(python3 -c "import json; f=open('$CONVERGENCE_FILE'); d=json.load(f); print(d.get('best_speedup', 0))")
            
            if [ "$NO_IMPROVEMENT" -ge "$NO_IMPROVEMENT_THRESHOLD" ]; then
                echo -e "${YELLOW}========================================${NC}"
                echo -e "${YELLOW}检测到收敛：连续 $NO_IMPROVEMENT 轮无性能提升${NC}"
                echo -e "${YELLOW}最佳加速比: ${BEST_SPEEDUP}x${NC}"
                echo -e "${YELLOW}提前结束迭代${NC}"
                echo -e "${YELLOW}========================================${NC}"
                break
            fi
        fi
    fi
    
    echo -e "${GREEN}[迭代 $i] 完成${NC}"
    echo ""
done

# Final evaluation (top1, top3, top5)
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}最终评估${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Final optimized evaluation
python3 evaluate.py \
    --do_train \
    --mode final \
    --lang $LANG \
    --process_number $PROCESS_NUMBER \
    --output_path "$OUTPUT_PATH/$BASE_MODE/$ITERATIONS" \
    --model_name $MODEL_NAME \
    --slice 1 \
    --testing_number 0 \
    --test_case_path $PRIVATE_TEST_CASE_PATH \
    --qemu_path "$QEMU_PATH" \
    --riscv_gcc_toolchain_path "$RISC_V_GCC_TOOLCHAIN_PATH" \
    --riscv_mode

if [ $? -ne 0 ]; then
    echo -e "${RED}最终评估失败${NC}"
else
    echo -e "${GREEN}最终评估完成${NC}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有任务完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "结果保存在: $OUTPUT_PATH/$BASE_MODE/"

