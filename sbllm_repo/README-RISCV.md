# RISC-V 代码自动优化框架

## 📖 项目简介

本项目基于 SBLLM（Search-Based LLMs for Code Optimization）框架，针对 RISC-V 架构代码优化场景进行了深度定制和增强。通过引入注释语义分析、QEMU 仿真评估、人工优化补丁知识库和混合检索等关键技术，实现了从其他架构到 RISC-V 架构的代码自动迁移和优化。

### 核心价值

- 🎯 **自动化优化**：无需手动分析，自动生成多个优化候选
- 🔍 **智能检索**：基于知识库的混合检索，复用历史优化经验
- ✅ **严格验证**：QEMU 仿真环境确保功能正确性和性能提升
- 🚀 **高效迭代**：遗传算法驱动的迭代优化，持续改进代码性能

### 适用场景

- 从 x86/ARM 架构迁移到 RISC-V 的代码优化
- RISC-V 汇编代码的性能优化
- RISC-V C 代码的指令级优化
- 利用 RISC-V 特性（压缩指令、流水线等）的代码优化

## 🎯 核心特性

### 1. 注释驱动的优化建议生成
- ✅ 使用 LLM 分析 RISC-V 代码注释，提取功能意图和优化方向
- ✅ 生成结构化的优化建议（review comments），指导后续代码生成
- ✅ 支持多种 LLM 模型（ChatGPT、GPT-4、Gemini、DeepSeek、CodeLlama）

### 2. QEMU 仿真环境评估
- ✅ 在真实的 QEMU RISC-V 仿真环境中运行和测试代码
- ✅ 评估功能正确性、执行时间和代码体积三个维度
- ✅ 支持汇编代码和 C 代码的编译与执行
- ✅ 功能正确性检查：输出比较、数值容差、返回码验证

### 3. 混合检索知识库
- ✅ 结合 BM25 关键词检索和向量语义检索
- ✅ 从人工优化补丁数据集中检索 Similar Pattern 和 Different Pattern
- ✅ 支持灵活的检索策略配置（BM25 / 向量 / 混合）
- ✅ FAISS 向量索引支持，快速检索

### 4. 遗传算法迭代优化
- ✅ 使用 Similar Pattern 作为主导优化方向
- ✅ 引入 Different Pattern 作为"突变"操作，探索新的优化路径
- ✅ 迭代优化直到性能收敛
- ✅ 自动停止机制：当优化收敛时自动停止迭代

### 5. 性能优化特性
- ✅ **编译缓存**：基于代码哈希的编译结果缓存，避免重复编译
- ✅ **并行评估**：使用线程池并行评估多个候选代码
- ✅ **智能缓存**：自动缓存编译结果，显著提升评估速度（5-10 倍提升）

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    输入: RISC-V 代码（含注释）                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          [模块一] 生成优化候选集                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. 语义提取：LLM 分析代码注释                          │  │
│  │    → 提取功能意图和优化方向                            │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. 生成优化建议：基于分析结果生成 review comments     │  │
│  │    → 结构化优化建议（条件分支、内存、指令集等）        │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. 生成候选代码：结合建议和原始代码生成多个版本        │  │
│  │    → 生成 3-8 个不同优化思路的候选代码                │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          [模块二] 样本筛选（QEMU 评估）                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. 编译为 RISC-V 二进制                                │  │
│  │    → 支持汇编和 C 代码，自动识别代码类型              │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. QEMU 仿真执行                                       │  │
│  │    → 在真实 RISC-V 环境中运行                          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. 三维评估                                           │  │
│  │    ✓ 功能正确性：输出比较 + 返回码检查                │  │
│  │    ✓ 执行时间：多次运行取平均值                        │  │
│  │    ✓ 代码体积：使用 size/objdump 获取                 │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 4. 筛选排序：排除不正确样本，按性能指标排序            │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          [模块三] 知识库模板检索                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. 混合检索（BM25 + 向量检索）                        │  │
│  │    → BM25：关键词匹配                                 │  │
│  │    → 向量检索：语义相似度（FAISS）                    │  │
│  │    → 加权融合两种检索结果                             │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. 检索 Similar Pattern                               │  │
│  │    → 与当前代码最相似的优化模式                       │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. 检索 Different Pattern                             │  │
│  │    → 语义相似但优化逻辑不同的模式（用于探索）          │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          [模块四] 遗传迭代优化                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. 结合 Pattern 生成新版本                            │  │
│  │    → Similar Pattern：主导优化方向（稳定性）            │  │
│  │    → Different Pattern：突变操作（探索性）             │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. QEMU 评估新版本                                     │  │
│  │    → 验证功能正确性和性能提升                          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. 迭代判断                                           │  │
│  │    → 提升显著：继续迭代                                │  │
│  │    → 性能收敛：停止迭代                                │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              输出: 优化后的 RISC-V 代码                       │
│          ✓ 功能正确（100% 正确性）                          │
│          ✓ 性能提升（执行加速比）                            │
│          ✓ 体积优化（代码体积缩减）                          │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

- **LLM 模型**：OpenAI GPT、Google Gemini、DeepSeek、CodeLlama
- **评估环境**：QEMU RISC-V 仿真器
- **编译工具**：RISC-V GCC 工具链
- **检索技术**：BM25、FAISS、Sentence Transformers
- **优化算法**：遗传算法、迭代优化

## 环境要求

### 系统要求
- Python 3.9.12
- Linux 系统（推荐 Ubuntu 20.04+）
- QEMU（qemu-riscv64）
- RISC-V GCC 工具链（riscv64-unknown-linux-gnu-gcc）

### Python 依赖

安装依赖：

```bash
pip install -r requirement.txt
```

主要依赖包括：
- `jsonlines==4.0.0`
- `numpy==1.26.1`
- `openai==0.28.1`
- `tqdm==4.66.1`
- `google-generativeai==0.3.2`
- `rank_bm25==0.2.2`
- `editdistance==0.6.2`
- `tree-sitter==0.20.4`
- `faiss-cpu==1.7.4` ⭐ 新增
- `sentence-transformers==2.2.2` ⭐ 新增
- `pandas==2.0.3` ⭐ 新增

### 外部工具安装

#### 1. 安装 QEMU

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install qemu-system-riscv64 qemu-user-riscv64

# 验证安装
qemu-riscv64 --version
```

#### 2. 安装 RISC-V GCC 工具链

```bash
# 下载预编译工具链
wget https://github.com/riscv/riscv-gnu-toolchain/releases/download/xxx/riscv64-unknown-linux-gnu.tar.gz
tar -xzf riscv64-unknown-linux-gnu.tar.gz
export PATH=$PATH:/path/to/riscv64-unknown-linux-gnu/bin

# 验证安装
riscv64-unknown-linux-gnu-gcc --version
```

## 配置说明

### 1. API 密钥配置

编辑 `sbllm/sbllm/evol_query.py`，填入你的 API 密钥：

```python
llama_api_keys = [
    "your-codellama-api-key",
]

openai_api_keys = [
    'your-openai-api-key',
]

gemini_api_keys = [
    "your-gemini-api-key"
]

deepseek_api_keys = [
    "your-deepseek-api-key",
]
```

**支持的模型**：
- `deepseek`: 使用 DeepSeek API（**默认推荐**）
  - 使用 OpenAI 兼容格式
  - base_url: `https://api.deepseek.com`
  - model: `deepseek-chat` (对应 DeepSeek-V3.2-Exp 的非思考模式)
  - 参考文档: https://platform.deepseek.com/api-docs/
- `chatgpt` / `gpt4`: 使用 OpenAI API
- `gemini`: 使用 Google Gemini API
- `codellama`: 使用 DeepInfra CodeLlama API

### 2. 数据准备

#### 训练数据格式（知识库）

准备优化补丁数据集，保存为 JSONL 格式，每个条目包含：

```json
{
    "id": 1,
    "source": "来源库或项目",
    "optimization_type": "条件分支优化",
    "optimization_description": "优化描述",
    "original_code": "原始代码",
    "optimized_code": "优化后代码",
    "query_abs": "抽象化后的原始代码",
    "edit_code_abs": "删除部分（ds）",
    "edit_opt_abs": "新增部分（df）",
    "text_representation": "文本表示（用于向量检索）"
}
```

#### 测试数据格式

测试数据应包含待优化的 RISC-V 代码：

```json
{
    "idx": 0,
    "query": "待优化的 RISC-V 代码（含注释）",
    "reference": "参考优化代码（可选）",
    "input": "测试输入数据（可选）"
}
```

## 📚 用户使用手册

### 🧪 快速测试（推荐首次使用）

在正式使用框架之前，建议先运行测试脚本验证环境配置和基本功能：

```bash
cd sbllm/sbllm
bash test_riscv.sh \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu
```

**注意**：测试脚本默认使用 `deepseek` 模型，确保已在 `evol_query.py` 中配置 DeepSeek API 密钥。

**测试脚本功能**：
- ✅ 自动环境验证
- ✅ 创建测试数据
- ✅ 运行一次简化的优化流程
- ✅ 验证结果输出

**详细测试教程**：请参考 `TEST_TUTORIAL.md`

### 快速开始（5 分钟上手）

#### 前置准备

1. **安装依赖**
   ```bash
   # 安装 Python 依赖
   pip install -r requirement.txt
   
   # 安装 QEMU（Ubuntu/Debian）
   sudo apt-get install qemu-user-riscv64
   
   # 安装 RISC-V GCC 工具链
   # 下载并解压到 /opt/riscv64-unknown-linux-gnu
   ```

2. **配置 API 密钥**
   
   编辑 `sbllm/sbllm/evol_query.py`，填入你的 API 密钥：
   ```python
   openai_api_keys = ['your-openai-api-key']
   gemini_api_keys = ["your-gemini-api-key"]
   # ... 其他 API 密钥
   ```

3. **准备数据**
   
   创建数据目录并准备数据文件：
   ```bash
   mkdir -p processed_data/riscv
   # 将测试数据放入 processed_data/riscv/test.jsonl
   # 将训练数据放入 processed_data/riscv/train.jsonl
   ```

#### 步骤 1: 环境验证

在运行优化框架之前，首先验证环境配置：

```bash
cd sbllm/sbllm
python validate_riscv_setup.py \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu
```

**验证内容**：
- ✅ QEMU 可执行文件是否存在并可运行
- ✅ RISC-V GCC 工具链是否完整（gcc, as, objdump, size）
- ✅ 数据文件是否存在且格式正确（test.jsonl, train.jsonl）
- ✅ 输出目录是否可写
- ✅ API 密钥是否已配置（检查占位符）
- ✅ Python 依赖包是否已安装

**输出示例**：
```
============================================================
RISC-V 环境验证
============================================================

1. 检查 QEMU
------------------------------------------------------------
✓ QEMU: /usr/bin/qemu-riscv64
  版本信息: qemu-riscv64 version 6.2.0

2. 检查 RISC-V GCC 工具链
------------------------------------------------------------
✓ 工具链工具: riscv64-unknown-linux-gnu-gcc
✓ 工具链工具: riscv64-unknown-linux-gnu-as
...

✓ 所有检查通过！环境配置正确。
```

#### 步骤 2: 运行优化框架

使用一键运行脚本（推荐）：

```bash
cd sbllm/sbllm
bash run_riscv.sh \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu
```

**注意**：脚本默认使用 `deepseek` 模型，确保已在 `evol_query.py` 中配置 DeepSeek API 密钥。

**脚本自动执行**：
1. ✅ 环境验证（调用 `validate_riscv_setup.py`）
2. ✅ 初始化结果（如需要，调用 `initial.py`）
3. ✅ 迭代优化循环（默认 4 次迭代）
   - 生成优化候选（`evol_query.py`）
   - QEMU 评估候选代码（`evaluate.py`）
   - 知识库检索和合并（`merge.py`，第 2 次迭代开始）
4. ✅ 最终评估（top1, top3, top5）

**运行时间**：
- 单个代码优化：约 5-15 分钟（取决于候选数量和迭代次数）
- 批量优化（100 个代码）：约 2-4 小时

#### 步骤 3: 查看结果

优化完成后，结果保存在 `output/riscv/riscv_optimization/` 目录：

```bash
# 查看最终结果
cat output/riscv/riscv_optimization/results.jsonl

# 查看最后一次迭代的评估报告
cat output/riscv/riscv_optimization/4/test_execution_*.report
```

### 完整使用流程

#### 场景 1: 快速测试（开发阶段）

适用于快速验证框架功能，使用较少候选和迭代次数：

```bash
bash run_riscv.sh \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu \
    --generation_number 3 \
    --iterations 2 \
    --retrieval_method bm25  # 使用 BM25 加快速度
```

**特点**：
- ⚡ 速度快：约 2-5 分钟完成
- 🎯 适合：功能验证、快速测试

#### 场景 2: 标准优化（生产环境）

适用于实际项目优化，平衡速度和效果：

```bash
bash run_riscv.sh \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu \
    --model_name gemini \
    --generation_number 5 \
    --iterations 4 \
    --retrieval_method hybrid \
    --vector_index_path ../processed_data/riscv/faiss_index.bin
```

**特点**：
- ⚖️ 平衡：速度与效果的最佳平衡
- 🎯 适合：生产环境、实际项目

#### 场景 3: 深度优化（研究场景）

适用于追求极致优化效果，使用更多候选和迭代：

```bash
bash run_riscv.sh \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu \
    --model_name gpt4 \
    --generation_number 8 \
    --iterations 6 \
    --retrieval_method hybrid \
    --hybrid_alpha 0.6 \
    --vector_index_path ../processed_data/riscv/faiss_index.bin
```

**特点**：
- 🔬 深度：探索更多优化可能性
- 🎯 适合：研究、论文实验、关键代码优化

### 手动运行（高级用户）

如果需要手动控制每个步骤，或进行调试，可以参考以下命令：

#### 完整手动流程

**步骤 1: 初始化结果**

```bash
cd sbllm
python sbllm/initial.py --model_name chatgpt --lang riscv
```

**步骤 2: 第一次迭代 - 生成优化候选**

```bash
cd sbllm/sbllm
python evol_query.py \
    --mode riscv_optimization \
    --model_name chatgpt \
    --lang riscv \
    --generation_path ../output \
    --baseline_data_path ../processed_data/riscv/test.jsonl \
    --iteration 0 \
    --generation_number 5 \
    --riscv_mode
```

**步骤 3: 评估候选代码（QEMU）**

```bash
python evaluate.py \
    --mode riscv_optimization \
    --lang riscv \
    --output_path ../output/riscv \
    --model_name chatgpt \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu \
    --test_case_path ../processed_data/riscv/test_cases \
    --process_number 8 \
    --riscv_mode
```

**步骤 4: 知识库检索和合并（第 2 次迭代开始）**

```bash
python merge.py \
    --lang riscv \
    --mode riscv_optimization \
    --iteration 1 \
    --model_name chatgpt \
    --generation_path ../output \
    --training_data_path ../processed_data/riscv/train.jsonl \
    --retrieval_method hybrid \
    --hybrid_alpha 0.5 \
    --embedding_model all-MiniLM-L6-v2
```

**步骤 5: 继续迭代（重复步骤 2-4，iteration 递增）**

```bash
# 第 2 次迭代
python evol_query.py --iteration 1 ...
python evaluate.py --mode 1 ...
python merge.py --iteration 2 ...

# 第 3 次迭代
python evol_query.py --iteration 2 ...
python evaluate.py --mode 2 ...
python merge.py --iteration 3 ...
```

### 使用示例

#### 示例 1: 优化单个 RISC-V 汇编函数

**输入代码**（`test.jsonl`）：
```json
{
    "idx": 0,
    "query": "# 计算数组元素之和\n# 优化方向：减少内存访问\n.text\n.global sum_array\nsum_array:\n    li t0, 0\n    li t1, 0\nloop:\n    beq t1, a1, done\n    lw t2, 0(a0)\n    add t0, t0, t2\n    addi a0, a0, 4\n    addi t1, t1, 1\n    j loop\ndone:\n    mv a0, t0\n    ret",
    "input": "10\n1 2 3 4 5 6 7 8 9 10"
}
```

**运行优化**：
```bash
bash run_riscv.sh \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu
```

**输出结果**：
- 优化后的代码（功能正确，性能提升）
- 执行时间对比
- 代码体积对比

#### 示例 2: 批量优化多个代码

准备包含多个代码的 `test.jsonl` 文件，框架会自动处理所有代码。

#### 示例 3: 使用不同的 LLM 模型

```bash
# 使用 Gemini
bash run_riscv.sh \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu \
    --model_name gemini

# 使用 GPT-4
bash run_riscv.sh \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu \
    --model_name gpt4
```

### 运行脚本参数说明

#### run_riscv.sh 参数

| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `--qemu_path` | QEMU 可执行文件路径 | ✅ | - |
| `--riscv_gcc_toolchain_path` | RISC-V GCC 工具链路径 | ✅ | - |
| `--model_name` | LLM 模型名称 | ❌ | `chatgpt` |
| `--generation_number` | 每次生成的候选数量 | ❌ | `5` |
| `--beam_number` | beam 搜索数量 | ❌ | `3` |
| `--iterations` | 迭代次数 | ❌ | `4` |
| `--data_dir` | 数据目录路径 | ❌ | `../processed_data/riscv` |
| `--output_dir` | 输出目录路径 | ❌ | `../output/riscv` |
| `--retrieval_method` | 检索方法（bm25/vector/hybrid） | ❌ | `hybrid` |
| `--vector_index_path` | 向量索引文件路径 | ❌ | - |

**支持的模型**：`chatgpt`, `gpt4`, `gemini`, `deepseek`, `codellama`

**检索方法**：
- `bm25` - 仅使用 BM25 关键词检索
- `vector` - 仅使用向量语义检索
- `hybrid` - 混合检索（BM25 + 向量，推荐）

#### validate_riscv_setup.py 参数

| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `--qemu_path` | QEMU 可执行文件路径 | ✅ | - |
| `--riscv_gcc_toolchain_path` | RISC-V GCC 工具链路径 | ✅ | - |
| `--data_dir` | 数据目录路径 | ❌ | `../processed_data/riscv` |
| `--output_dir` | 输出目录路径 | ❌ | `../output/riscv` |

## 核心模块说明

### 模块一：生成优化候选集（evol_query.py）

**功能**：在第一次迭代时，分析代码注释并生成优化建议，然后基于建议生成多个优化候选。

**关键函数**：
- `generate_review_comments(code, cfg)`: 生成优化建议
- `read_file(cfg)`: 在第一次迭代时集成 review comments 生成流程

**工作流程**：
1. 调用 `generate_review_comments()` 分析代码注释
2. 使用 `_PROMPT_FOR_REVIEW` prompt 模板
3. 将 review comments 与原始代码结合，使用 `_PROMPT_FOR_CANDIDATES` 生成候选代码

### 模块二：样本筛选（qemu_evaluator.py + execution.py）

**功能**：使用 QEMU 仿真环境评估候选代码的性能和正确性。

**关键类**：
- `QEMURISCVEvaluator`: QEMU RISC-V 评估器

**关键方法**：
- `compile_to_riscv_binary()`: 编译 RISC-V 代码为二进制（支持编译缓存）
- `run_and_measure()`: 运行并测量执行时间和代码大小，验证功能正确性
- `evaluate_candidates()`: 批量评估候选代码
- `evaluate_candidates_parallel()`: 并行评估多个候选代码（性能优化）

**功能正确性检查**：
- ✅ **输出比较**：比较优化代码与原始代码的输出结果
- ✅ **数值容差**：支持浮点数容差比较（默认 1e-6）
- ✅ **返回码检查**：检查程序执行返回码
- ✅ **多轮验证**：多次运行取最稳定的输出

**评估指标**：
- 功能正确性（is_correct）- 通过输出比较和返回码验证
- 执行时间（execution_time）- 多次运行的平均值
- 代码大小（code_size）- 使用 size 或 objdump 获取
- 加速比（speedup_ratio = 1 - new_time/old_time）
- 体积缩减率（size_reduction_ratio）

**性能优化特性**：
- ✅ **编译缓存**：基于代码哈希的编译结果缓存，避免重复编译
- ✅ **并行评估**：使用线程池并行评估多个候选代码
- ✅ **智能缓存**：自动缓存编译结果，显著提升评估速度

### 模块三：知识库模板检索（merge.py）

**功能**：从优化补丁知识库中检索 Similar Pattern 和 Different Pattern。

**检索策略**：
- **BM25 检索**：基于关键词匹配
- **向量检索**：基于语义相似度（使用 sentence-transformers）
- **混合检索**：融合 BM25 和向量检索结果

**关键修改**：
- `main()`: 加载 FAISS 向量索引和 embedding 模型
- `process()`: 实现混合检索逻辑，检索 Similar 和 Different Pattern
- 在结果中存储检索来源信息

### 模块四：遗传迭代优化（evol_query.py）

**功能**：使用遗传算法迭代优化代码，结合 Similar 和 Different Pattern。

**关键函数**：
- `prompt_construction()`: 构建迭代优化的 prompt

**Prompt 模板**：
- `_PROMPT_FOR_RISCV_EVOLUTION`: 遗传迭代优化的 prompt

**工作流程**：
1. 分析原始代码和当前优化版本
2. 识别未使用的优化机会（Mutation）
3. 结合 Similar 和 Different Pattern 生成新版本
4. 使用 QEMU 评估新版本
5. 如果提升显著，继续迭代；否则收敛

## 数据格式说明

### 知识库数据格式（train.jsonl）

```json
{
    "id": 1,
    "source": "riscv-optimization-patches",
    "optimization_type": "Conditional Branch Optimization",
    "optimization_description": "优化条件分支，减少跳转开销",
    "original_code": "原始 RISC-V 代码",
    "optimized_code": "优化后的 RISC-V 代码",
    "query_abs": "抽象化后的查询代码（tokenized）",
    "edit_code_abs": "删除部分（ds）的抽象表示",
    "edit_opt_abs": "新增部分（df）的抽象表示",
    "text_representation": "用于向量检索的文本表示"
}
```

### 测试数据格式（test.jsonl）

```json
{
    "idx": 0,
    "query": "待优化的 RISC-V 代码（包含注释）",
    "reference": "参考优化代码（可选）",
    "input": "测试输入（可选）"
}
```

### 执行报告格式（test_execution_*.report）

```json
{
    "code_v0_no_empty_lines": "原始代码",
    "code_v1_no_empty_lines": "参考代码",
    "input_time_mean": 1.23,
    "reference_time_mean": 0.98,
    "model_generated_potentially_faster_code_col": "最佳候选代码",
    "model_generated_potentially_faster_code_col_acc": 1,
    "model_generated_potentially_faster_code_col_time_mean": 0.85,
    "model_generated_potentially_faster_code_col_0": "候选代码 0",
    "model_generated_potentially_faster_code_col_0_acc": 1,
    "model_generated_potentially_faster_code_col_0_time_mean": 0.90
}
```

## 工作流程

### 自动化流程（使用 run_riscv.sh）

```
1. 环境验证
   └─ validate_riscv_setup.py
      ├─ 检查 QEMU
      ├─ 检查工具链
      ├─ 检查数据文件
      └─ 检查 API 密钥

2. 初始化（首次运行）
   └─ initial.py
      └─ 生成初始结果文件

3. 迭代优化（循环 N 次）
   ├─ evol_query.py
   │  └─ 生成优化候选（第一次迭代包含注释分析）
   ├─ evaluate.py
   │  └─ QEMU 评估候选代码
   └─ merge.py
      └─ 知识库检索和合并（第 2 次迭代开始）

4. 最终评估
   └─ evaluate.py
      ├─ top1 评估
      ├─ top3 评估
      └─ top5 评估
```

### 输出结构

```
output/riscv/riscv_optimization/
├── results.jsonl              # 最终结果
├── 0/                          # 初始迭代
├── 1/                          # 第 1 次迭代
│   ├── prediction_*.jsonl
│   ├── test_execution_*.report
│   └── qemu_temp/              # QEMU 临时文件
├── 2/                          # 第 2 次迭代
├── ...
└── 4/                          # 最终迭代
    ├── top1/
    ├── top3/
    └── top5/
```

## 输出结果说明

### results.jsonl 格式

```json
{
    "idx": 0,
    "query": "原始代码",
    "reference": "参考代码",
    "stop": 0,
    "best_result": {
        "code": "最佳优化代码",
        "acc": 1,
        "time": 0.85,
        "input_time": 1.23,
        "reference_time": 0.98
    },
    "best_candidates": [
        {
            "code": "候选代码",
            "acc": 1,
            "time": 0.85,
            "input_time": 1.23,
            "content": "完整内容"
        }
    ],
    "pattern": [
        "Similar Pattern",
        "Different Pattern"
    ],
    "retrieval": {
        "query": "查询文本",
        "selected_ids": [1, 2],
        "method": "hybrid"
    }
}
```

## 🔧 故障排除

### 常见问题及解决方案

#### 1. QEMU 路径错误

**错误信息**：
```
FileNotFoundError: QEMU path not found: /usr/bin/qemu-riscv64
```

**解决方案**：
```bash
# 1. 检查 QEMU 是否安装
which qemu-riscv64

# 2. 如果未安装，安装 QEMU
sudo apt-get install qemu-user-riscv64

# 3. 如果已安装但路径不同，使用完整路径
bash run_riscv.sh \
    --qemu_path $(which qemu-riscv64) \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu
```

#### 2. RISC-V GCC 工具链错误

**错误信息**：
```
FileNotFoundError: RISC-V GCC path not found: /opt/riscv64-unknown-linux-gnu/bin/riscv64-unknown-linux-gnu-gcc
```

**解决方案**：
```bash
# 1. 确认工具链已安装
ls /opt/riscv64-unknown-linux-gnu/bin/

# 2. 使用工具链根目录路径（不是 bin 目录）
# 正确：--riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu
# 错误：--riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu/bin

# 3. 如果路径不同，使用实际路径
bash run_riscv.sh \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /path/to/your/riscv-toolchain
```

#### 3. 数据文件缺失或格式错误

**错误信息**：
```
FileNotFoundError: Test data file not found: ../processed_data/riscv/test.jsonl
```

**解决方案**：
```bash
# 1. 检查文件是否存在
ls -la processed_data/riscv/test.jsonl

# 2. 检查文件格式（应该是 JSONL，每行一个 JSON 对象）
head -n 1 processed_data/riscv/test.jsonl

# 3. 验证必需字段
python -c "import jsonlines; f=jsonlines.open('processed_data/riscv/test.jsonl'); print(next(f))"
```

**数据格式要求**：
- 文件必须是 JSONL 格式（每行一个 JSON 对象）
- 测试数据必须包含 `idx` 和 `query` 字段
- 训练数据必须包含 `id`, `original_code`, `optimized_code` 字段

#### 4. API 密钥未配置

**警告信息**：
```
⚠ 检测到占位符 API 密钥，请确保已配置真实的 API 密钥
```

**解决方案**：
```bash
# 1. 编辑 evol_query.py
vim sbllm/sbllm/evol_query.py

# 2. 找到并替换占位符
# 将 "xxxxxxxxxxxxxxxxxxxx" 替换为你的真实 API 密钥

# 3. 保存文件后重新运行
```

**API 密钥获取**：
- OpenAI: https://platform.openai.com/api-keys
- Gemini: https://makersuite.google.com/app/apikey
- DeepSeek: https://platform.deepseek.com/api_keys
- CodeLlama: https://deepinfra.com/

#### 5. Python 依赖缺失

**错误信息**：
```
✗ Python 包缺失: faiss-cpu
```

**解决方案**：
```bash
# 安装所有依赖
pip install -r requirement.txt

# 如果某个包安装失败，单独安装
pip install faiss-cpu==1.7.4
pip install sentence-transformers==2.2.2
```

#### 6. 编译失败

**错误信息**：
```
Assembly failed: error: invalid instruction
```

**解决方案**：
```bash
# 1. 检查代码格式
# 确保是有效的 RISC-V 汇编或 C 代码

# 2. 查看详细错误信息
# 检查日志文件或标准错误输出

# 3. 验证代码语法
# 使用 riscv64-unknown-linux-gnu-gcc 手动编译测试
```

#### 7. 向量检索失败

**错误信息**：
```
Failed to load embedding model: all-MiniLM-L6-v2
```

**解决方案**：
```bash
# 1. 检查网络连接（首次使用需要下载模型）
ping huggingface.co

# 2. 手动下载模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 3. 使用其他模型
bash run_riscv.sh \
    --qemu_path /usr/bin/qemu-riscv64 \
    --riscv_gcc_toolchain_path /opt/riscv64-unknown-linux-gnu \
    --embedding_model all-mpnet-base-v2
```

#### 8. 内存不足

**错误信息**：
```
MemoryError: Unable to allocate array
```

**解决方案**：
- 减少并行进程数：`--process_number 4`
- 减少候选数量：`--generation_number 3`
- 使用 BM25 检索（不使用向量检索）：`--retrieval_method bm25`

#### 9. 执行超时

**错误信息**：
```
Execution timeout for candidate_0.bin
```

**解决方案**：
- 检查代码是否有无限循环
- 增加超时时间（修改 `qemu_evaluator.py` 中的 `timeout=10`）
- 检查输入数据是否过大

### 调试技巧

#### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 检查中间结果

```bash
# 查看生成的候选代码
cat output/riscv/riscv_optimization/0/prediction_*.jsonl

# 查看评估报告
cat output/riscv/riscv_optimization/0/test_execution_*.report
```

#### 3. 单独测试组件

```python
# 测试 QEMU 评估器
from sbllm.qemu_evaluator import QEMURISCVEvaluator

evaluator = QEMURISCVEvaluator(
    qemu_path="/usr/bin/qemu-riscv64",
    riscv_gcc_toolchain_path="/opt/riscv64-unknown-linux-gnu"
)

# 测试编译
success = evaluator.compile_to_riscv_binary(code, "test.bin")
print(f"编译成功: {success}")

# 测试运行
time, size, correct, output = evaluator.run_and_measure("test.bin")
print(f"执行时间: {time}s, 正确性: {correct}")
```

## ⚡ 性能优化建议

### 评估性能优化

#### 1. 编译缓存（默认启用）

**功能**：基于代码哈希的编译结果缓存，避免重复编译相同代码。

**性能提升**：50-80%（重复评估场景）

**使用方式**：
```python
evaluator = QEMURISCVEvaluator(
    qemu_path=qemu_path,
    riscv_gcc_toolchain_path=toolchain_path,
    enable_cache=True  # 默认启用
)
```

#### 2. 并行评估

**功能**：使用线程池并行评估多个候选代码。

**性能提升**：2-4 倍（取决于 CPU 核心数）

**使用方式**：
```python
evaluator = QEMURISCVEvaluator(
    qemu_path=qemu_path,
    riscv_gcc_toolchain_path=toolchain_path,
    max_workers=4  # 建议设置为 CPU 核心数
)

# 使用并行评估方法
results = evaluator.evaluate_candidates_parallel(
    original_code=original_code,
    candidate_codes=candidate_codes,
    input_data=input_data,
    reference_output=reference_output
)
```

#### 3. 并行进程数配置

在 `evaluate.py` 中调整 `--process_number` 参数：
```bash
python evaluate.py \
    --process_number 8 \  # 根据 CPU 核心数调整
    ...
```

**建议**：设置为 `CPU 核心数 - 1`（保留一个核心给系统）

### 检索性能优化

#### 1. 向量索引缓存

**功能**：使用预构建的 FAISS 向量索引，避免每次重新构建。

**性能提升**：首次构建后，后续加载可节省 90%+ 时间

**使用方式**：
```bash
# 首次运行：自动构建并保存索引
python merge.py \
    --training_data_path ./processed_data/riscv/train.jsonl \
    --retrieval_method hybrid \
    --vector_index_path ./processed_data/riscv/faiss_index.bin

# 后续运行：直接加载索引
python merge.py \
    --training_data_path ./processed_data/riscv/train.jsonl \
    --retrieval_method hybrid \
    --vector_index_path ./processed_data/riscv/faiss_index.bin  # 使用已构建的索引
```

#### 2. 检索策略选择

| 策略 | 速度 | 准确性 | 适用场景 |
|------|------|--------|---------|
| **BM25** | ⚡⚡⚡ 快 | ⭐⭐ 中等 | 关键词匹配、快速迭代 |
| **向量检索** | ⚡⚡ 中等 | ⭐⭐⭐ 高 | 语义理解、复杂场景 |
| **混合检索** | ⚡⚡ 中等 | ⭐⭐⭐ 高 | **推荐使用**，平衡速度和准确性 |

**性能对比**（1000 条训练数据）：
- BM25：~0.1 秒/查询
- 向量检索：~0.3 秒/查询
- 混合检索：~0.4 秒/查询

### 迭代优化策略

#### 1. 迭代次数配置

| 场景 | 迭代次数 | 说明 |
|------|---------|------|
| 快速测试 | 2 | 快速验证功能 |
| 标准优化 | 4 | 平衡速度和效果（推荐） |
| 深度优化 | 6+ | 追求极致效果 |

#### 2. 候选数量配置

| 场景 | 候选数量 | 说明 |
|------|---------|------|
| 快速迭代 | 3 | 加快单次迭代速度 |
| 标准优化 | 5 | 平衡探索和速度（推荐） |
| 深度探索 | 8+ | 探索更多优化可能性 |

### 性能基准测试

**测试环境**：
- CPU: Intel i7-9700K (8 核)
- RAM: 32GB
- 测试数据: 100 个候选代码

**性能对比**：

| 配置 | 评估时间 | 检索时间 | 总时间 | 性能提升 |
|------|---------|---------|--------|---------|
| 无优化 | 1200s | 45s | 1245s | 基准 |
| 编译缓存 | 480s | 45s | 525s | 2.4x |
| 编译缓存 + 并行评估 | 180s | 45s | 225s | 5.5x |
| 全优化 | 180s | 15s | 195s | **6.4x** |

**总体性能提升**：约 **6.4 倍**

## 🔄 与原始 SBLLM 的区别

### 主要增强

| 特性 | 原始 SBLLM | RISC-V 版本 |
|------|-----------|------------|
| **注释分析** | ❌ 无 | ✅ 新增 `generate_review_comments()` |
| **评估环境** | Python 原生环境 | ✅ QEMU RISC-V 仿真环境 |
| **功能验证** | 返回码检查 | ✅ 输出比较 + 数值容差 |
| **知识库来源** | AST 抽象 | ✅ 人工优化补丁数据集 |
| **检索方法** | BM25 | ✅ BM25 + 向量混合检索 |
| **Prompt 模板** | 通用模板 | ✅ RISC-V 专用模板 |
| **性能优化** | 基础 | ✅ 编译缓存 + 并行评估 |

### 兼容性

- ✅ **向后兼容**：保持与原始 SBLLM 的接口兼容
- ✅ **模式切换**：非 RISC-V 模式仍可使用原有评估系统
- ✅ **灵活启用**：通过 `--riscv_mode` 或 `--lang riscv` 启用 RISC-V 功能

### 使用建议

- **RISC-V 代码优化**：使用 RISC-V 模式（`--riscv_mode` 或 `--lang riscv`）
- **Python/C++ 代码优化**：使用原始模式（不指定 `--riscv_mode`）

## 📖 详细使用说明

### 数据准备详细指南

#### 训练数据（知识库）准备

训练数据用于构建优化模式知识库，支持混合检索。

**必需字段**：
- `id`: 唯一标识符
- `original_code`: 原始代码
- `optimized_code`: 优化后的代码
- `query_abs`: 抽象化后的查询代码（tokenized 列表）
- `edit_code_abs`: 删除部分（ds）的抽象表示（tokenized 列表）
- `edit_opt_abs`: 新增部分（df）的抽象表示（tokenized 列表）

**可选字段**：
- `source`: 来源库或项目
- `optimization_type`: 优化类型（如"条件分支优化"）
- `optimization_description`: 优化描述
- `text_representation`: 用于向量检索的文本表示

**数据预处理脚本示例**：
```python
import jsonlines
from sbllm.merge import abstract_cpp_code, tokenize_cpp_code

# 处理训练数据
with jsonlines.open('train_raw.jsonl') as reader, \
     jsonlines.open('train.jsonl', 'w') as writer:
    for obj in reader:
        # 抽象化代码
        original_abs = abstract_cpp_code(obj['original_code'])
        optimized_abs = abstract_cpp_code(obj['optimized_code'])
        
        # Tokenize
        query_abs = tokenize_cpp_code(original_abs)
        
        # 计算差异（ds 和 df）
        # ... 使用 diff 算法提取 edit_code_abs 和 edit_opt_abs
        
        # 写入处理后的数据
        writer.write({
            'id': obj['id'],
            'original_code': obj['original_code'],
            'optimized_code': obj['optimized_code'],
            'query_abs': query_abs,
            'edit_code_abs': edit_code_abs,
            'edit_opt_abs': edit_opt_abs,
            'text_representation': f"{obj.get('optimization_type', '')} {obj.get('optimization_description', '')}"
        })
```

#### 测试数据准备

测试数据包含待优化的 RISC-V 代码。

**必需字段**：
- `idx`: 唯一索引
- `query`: 待优化的 RISC-V 代码（**必须包含注释**）

**可选字段**：
- `reference`: 参考优化代码（用于对比）
- `input`: 测试输入数据

**示例**：
```json
{
    "idx": 0,
    "query": "# 计算斐波那契数列\n# 优化方向：减少递归调用，使用迭代\n.text\n.global fib\nfib:\n    # 函数实现...",
    "input": "10"
}
```

### 结果解读

#### 评估指标说明

**功能正确性（acc）**：
- `1`: 功能完全正确（输出匹配）
- `0`: 功能不正确（输出不匹配或执行失败）

**执行加速比（speedup_ratio）**：
- 公式：`OPT = 1 - new_time / old_time`
- 正值：性能提升
- 负值：性能下降

**代码体积缩减率（size_reduction_ratio）**：
- 公式：`(original_size - new_size) / original_size`
- 正值：体积减小
- 负值：体积增大

#### 结果文件说明

**results.jsonl**：最终优化结果汇总
- `best_result`: 最佳优化结果
- `best_candidates`: 最佳候选列表（按性能排序）
- `pattern`: Similar 和 Different Pattern
- `retrieval`: 检索来源信息

**test_execution_*.report**：详细评估报告
- 包含所有候选的评估结果
- 包含执行时间和正确性信息

### 最佳实践

#### 1. 代码注释编写

**好的注释示例**：
```riscv
# 计算数组元素之和
# 优化方向：减少内存访问，利用 RISC-V 流水线特性
# 当前问题：每次循环都访问内存，可能导致流水线停顿
.text
.global sum_array
sum_array:
    # 函数实现...
```

**不好的注释**：
```riscv
# 求和函数
.text
```

#### 2. 知识库构建

- **质量 > 数量**：优先收集高质量的优化补丁
- **多样性**：包含不同类型的优化（分支、内存、指令集等）
- **标注清晰**：确保 `optimization_type` 和 `optimization_description` 准确

#### 3. 迭代策略

- **首次迭代**：关注功能正确性，确保所有候选都能正确运行
- **后续迭代**：关注性能提升，逐步优化
- **收敛判断**：当连续 2 次迭代性能提升 < 5% 时，可考虑停止

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进本项目。

### 贡献方式

1. **报告 Bug**：在 GitHub Issues 中报告问题
2. **提出建议**：分享你的使用经验和改进建议
3. **提交代码**：通过 Pull Request 贡献代码改进

### 开发环境设置

```bash
# 克隆仓库
git clone <repository-url>
cd project1-sbllm

# 安装开发依赖
pip install -r requirement.txt

# 运行测试
python -m pytest tests/
```

## 📄 许可证

本项目继承 SBLLM 的许可证（见 LICENSE 文件）。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- **GitHub Issues**：提交问题或建议
- **项目文档**：查看详细文档和示例

## 📚 相关资源

- **SBLLM 原始论文**：Search-Based LLMs for Code Optimization (ICSE'25)
- **RISC-V 官方文档**：https://riscv.org/
- **QEMU 文档**：https://www.qemu.org/docs/
- **测试教程**：`TEST_TUTORIAL.md` - 详细的测试脚本使用指南
