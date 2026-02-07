# RISC-V 代码自动优化框架需求文档

## 项目概述

将要实现的代码，功能需求形成文档。

**需求**：在 sbllm 基础上，完成一个能将其他架构迁移到 RISC-V 架构的相关代码，进行自动修复的框架。

## 功能模块

一共分为四个功能模块：

### 1. 生成优化候选集

在 RISC-V 代码优化的初始阶段，我们首先输入待优化的 RISC-V 代码（Original Code）。

#### 输入

输入待优化的 RISC-V 代码（包含代码注释）

#### 流程

1. **语义提取**
   - 调用 LLM 对 RISC-V 代码的注释进行分析，获得功能意图以及开发者暗示的优化方向

2. **生成优化建议**
   - 基于（1）的分析结果，生成优化建议（review comments），作为后续代码改写的指导，相关 prompt 参考图 2。

3. **生成候选优化代码集合**
   - 将 review comments 结合 Original Code，通过 LLM 进一步生成多个优化版本，从而形成多个优化代码候选组成的优化候选集，该候选集不仅包含不同优化思路下的代码实现。

#### 输出

优化候选集（具体生成几个优化版本待定，比如 3、4、5 个都可以试一试）

#### Prompt 模板

用 LLM 对于待优化代码中的注释进行语义提取、语义分析，并生成对于该代码的优化建议 review comments，具体的 prompt 如下：

**Prompt Template**

```
[Objective]
Provide RISC-V adaptation and optimization solutions for the source code of the following software packages. The solutions should focus on the following requirements:
1. Reduced compilation time.
2. Functional consistency reaches 100%.
3. Make full use of the characteristics of RISC-V (e.g., instruction set, pipelining, registers, etc.).
4. Do not generate optimization code directly; instead, output detailed multiple optimization suggestions with their identified types (such as Conditional Branch Optimization, Memory Optimization, Instruction Set Optimization, or Others).

[Input Context]
Source Code: {The source code that retains the original comments to be optimized};

[Output]
{Optimization suggestions};
```

**图 2 第一模块 prompt 内容**

---

### 2. 样本筛选

在生成优化候选集之后，需要对其进行严格筛选，以确保最终进入后续阶段的优化样本既具备正确性，又能在性能方面带来实质提升，保证优化的有效。本研究采用了 QEMU 模拟器构建真实的 RISC-V 运行环境，对候选优化代码进行动态测试与量化评估。

#### 输入

上模块生成的优化候选集

#### 流程

使用 QEMU 模拟器搭建 RISC-V 仿真环境，在相同运行条件下，对候选代码进行动态执行与测试。

具体而言，我们使用以下三个维度的指标来评估候选样本的有效性：

**a) 功能一致性（Functional Consistency, 100% Correct）**

使优化后的代码在逻辑功能上与原始代码保持完全一致。功能一致性是基础约束，若样本在该指标上未达标，则直接淘汰，避免"以牺牲正确性换取性能"的情况。

**b) 执行加速比（Execution Speedup Ratio）**

定义为：`OPT = 1 - new_time / old_time`

其中 `new_time` 和 `old_time` 分别表示优化后与原始代码在相同运行环境下的平均执行时间，该指标用于衡量优化代码在运行时性能上的改进幅度。

**c) 代码体积缩减率（Code Size Reduction Ratio）**

通过比较优化前后指令条数、汇编代码行数，评估优化对代码体积的压缩效果。体积缩减不仅意味着更低的存储开销，也通常带来更低的指令缓存压力和流水线压力，从而间接提升执行效率。

#### 筛选与排序

在评估过程中，我们首先排除所有功能不一致的样本，仅保留 100% Correct 的优化代码；然后基于执行加速比和代码体积缩减率对样本进行排序，最终选取综合表现最佳的代表性样本作为输入，进入下一阶段的知识库优化模板检索环节。

#### 输出

代表性优化样本（功能一致，并性能和体积指标上表现优异）

---

### 3. 在知识库中寻找优化模板

#### 输入

第二模块得到的代表性优化样本，以及将收集到的人工优化补丁构成的数据集得到的优化模板知识库

#### 流程

**知识库构建**

从数据集中收集 pattern，形成模板库，对优化差异进行标准，`ds` 为优化过程中删掉的部分以及 `df` 优化中新增的部分。

**模式识别与定位**

使用语义嵌入表示代码，结合 BM25 与向量近邻检索，对 `ds/df` 与知识库模板进行相似度计算。即便代码有差异，也能通过语义相似度捕捉相同优化机会。

**模板检索**

分为 Similar Pattern 以及 Differnt Pattern：

- 模板库中相似得分最高即最相似的模板为 **Similar Pattern**，该模板的内容以及优化做的改动都与代表性样本最相似；
- 其次寻找 **Differnt Pattern**，检索与当前代表性样本语义相似但优化逻辑不同的模板，同样抽象化、算相似度，并取反（使得优化改动越相似，score 越低），检索获得

> **注**：原始文档中为 "Differnt Pattern"（应为 "Different Pattern"），此处保留原始写法。

#### 输出

检索到的 Similar + Differnt Pattern（如图 4）

**示例：**

① **Current Optimized code:**

```
define b2 @optbranch_3d038 %Arg() {
:RV32-LABEL: optbranch_32:
-:RV32-NEXT: B.a1.-1
-:RV32-NEXT: beq a0.a1, LBBO_2
-:RV32-NEXT: addi a0.a0.1
-:RV32-NEXT: ret
:RV32-NEXT: B.a0.-1
:RV32-NEXT: ret
```

② **Differnt Pattern:**

```
__vintr_includeIdx.v0;
__beq(failedIdx, minuteOne, NoFailure); // Explicit comparison -1
+__bin(failedIdx, NoFailure); // Use <0 to represent -1
+__beq(failedIdx, FailmanMatch); // Directly handle the boundary case of <0
```

③ **Refined Code:**

```
define b2 @optbranch_3d038 %Arg() {
:RV32-LABEL: optbranch_32:
+:RV32-NEXT: addi a0.a0.1
+:RV32-NEXT: bmea a0, LBBO_2
:RV32-NEXT: B.a0.-1
:RV32-NEXT: ret
```

**图 4 different pattern 示例**

---

### 4. 遗传算子驱动的优化迭代与最终代码生成

#### 输入

- Similar + Differnt Pattern
- 代表性样本代码

#### 流程

**迭代优化——评估——选择**

**遗传算子驱动优化**

将 Similar Pattern 作为主导优化方向，确保稳定性；将 Differnt Pattern 引入为"突变"或"交叉"操作，探索潜在的新优化路径。结合两者生成新的优化版本，形成优化候选。

**性能评估与选择**

使用与第二模块相同的 QEMU 仿真环境，对优化代码进行动态运行和性能指标测试。若新版本提升显著（超过预设阈值，如 xx%，该阈值待确定），说明仍有优化空间 → 进入新一轮迭代。若多次迭代后优化程度收敛并趋于稳定，则认为已达到最优状态 → 结束迭代。

**Prompt 驱动的优化**

使用设计好的 Prompt（参考图 5），指导 LLM 在每次迭代中结合模板进行运行。

#### 输出

最终优化后的 RISC-V 代码，满足：

- 保持与原始功能逻辑一致；
- 在性能与体积指标上达到收敛状态；
- 已通过迭代验证，具有最优或近似最优的性能表现。

#### Prompt Template

```
[Objective]
Provide RISC-V adaptation and optimization solutions for the following source codes. The objective is to continuously optimize the performance of the code through the iterative process of the genetic operator algorithm. Please follow the steps below to operate:
1. Analysis: Analyze the original code and the existing optimized version.
2. Mutation: Identify unused optimization opportunities.
3. Generation: Generate new optimization code through optimization methods and provide optimization points.

[Input Context]
Original code: {}
Current code: {<Sample1> <Sample2> <Sample3>}
Optimization mode: {<Similar Pattern> <Differnt Pattern>}

[Output]
{Optimized Code}
```

**图 5 第四阶段 prompt 内容**

---

## 与 sbllm 工作的区别和联系

### 联系

本项目完全继承了 sbllm 的总体框架和核心思想，即利用大型语言模型的生成能力、遗传算法的迭代优化能力以及模式检索的知识复用能力，形成一个自动化的代码优化系统。

### 区别

本项目针对 RISC-V 迁移这一垂直领域进行了深度定制和增强。通过引入注释分析、QEMU 评估、人工知识库和混合检索等关键技术，解决了原框架在硬件相关优化场景下评估不准、知识匮乏、优化方向不明确等问题，使其优化过程更加精准、高效和可靠。

### 需要在 sbllm 的基础上修改

1. 增加对 RISC-V 代码注释的语义分析，使用 LLM 分析代码注释，提取功能意图和优化方向。
2. 改变评估环境，将性能评估环境从原生 Python 环境切换到 QEMU RISC-V 仿真环境。
3. 重构知识库，将模式库的来源从 AST 抽象改为基于人工优化补丁数据集，并采用混合检索（BM25 + 向量语义检索）策略。

---

## 实现细节

### 模块一：生成优化候选集（修改 evol_query.py）

**目标**：修改 sbllm 中直接生成候选代码的逻辑，加入"分析注释 -> 生成建议 -> 生成代码"的两步走流程。

#### 输入修改

保持输入为待优化的 RISC-V 代码（字符串）。

#### 流程修改

**新增函数 `generate_review_comments(code: str) -> str`**

此函数封装 Prompt，调用 LLM。

**Prompt 模板：**

```python
_PROMPT_FOR_REVIEW = """
You are an expert in RISC-V architecture and performance optimization. Please analyze the following RISC-V assembly code and its comments. Extract the functional intent and any optimization hints provided by the developer.
Code:
{code}
Based on your analysis, provide specific optimization suggestions (review comments) to guide subsequent code rewriting. Focus on aspects like instruction selection, loop unrolling, register allocation, and leveraging RISC-V-specific features (e.g., compressed instructions).
"""
```

**修改函数 `generate_optimized_candidates(original_code: str, ...) -> List[str]`**

原函数是直接生成候选。现在需要先调用 `generate_review_comments(original_code)` 得到优化建议 `review_comments`。然后，将 `review_comments` 和 `original_code` 结合，形成一个新的、更强大的 Prompt 来生成候选代码。

**新的候选生成 Prompt 模板：**

```python
_PROMPT_FOR_CANDIDATES = """
Original RISC-V Code:
{original_code}

Optimization Suggestions:
{review_comments}
Please generate multiple optimized versions of the code based on the suggestions above. Each version should explore a different optimization strategy or combination of strategies. Ensure the optimized code is functionally equivalent to the original.
Generate {n_candidates} different versions.
"""
```

#### 输出

保持不变，返回一个优化后代码的列表 `List[str]`。

---

### 模块二：样本筛选（创建一个 qemu_evaluator.py）

**目标**：完全重写 sbllm 中的评估模块 `sbllm/execution.py`。原版的代码在本地执行 Python 代码并测量时间，现在需要在 QEMU 环境中运行 RISC-V 二进制文件。

**新建文件 `qemu_evaluator.py`，可以有以下核心类：**

```python
class QEMURISC-VEvaluator:
    def __init__(self, qemu_path: str, riscv_gcc_toolchain_path: str):
        """配置 QEMU 和交叉编译工具链的路径。"""
        
    def compile_to_riscv_binary(self, code: str, output_bin: str) -> bool:
        """使用 riscv-gcc 将输入的代码字符串编译成 RISC-V 二进制文件。需要处理汇编器指令。"""
        
    def run_and_measure(self, binary_path: str) -> Tuple[float, int, bool]:
        """使用 qemu-riscv64 运行二进制文件，测量：
        - execution_time：执行时间（秒）
        - code_size：二进制文件的代码段大小（objdump 或 size 命令获取）
        - is_correct：功能是否正确（需要通过预设的测试用例或比较与原始代码的输出结果）"""
        
    def evaluate_candidates(self, original_code: str, candidate_codes: List[str]) -> pd.DataFrame:
        """对每个候选代码进行评估，返回一个包含 execution_time, code_size, is_correct, speedup_ratio, size_reduction_ratio 等字段的 DataFrame，用于排序和筛选。"""
```

> **注**：原始文档中类名为 `QEMURISC-VEvaluator`（带连字符），但 Python 中类名不能包含连字符，实际实现时需改为 `QEMURISCVEvaluator`。

---

### 模块三：知识库模板检索（修改 merge.py）

1. **加载向量检索索引**
   - 在 `main(cfg)` 里，目前只加载了 BM25（基于 `edit_code_abs`），要做混合检索，需要在这里再加载 embedding 向量库（可以用 faiss / hnswlib / sentence-transformers）。

2. **修改 `process()` 里的检索逻辑**
   - 目前 `process()` 是 BM25，根据 `edit_code_abs` → 取最相似和最不相似的若干样本。
   - 改成：先用 BM25 检索 top_k，再用向量检索（faiss）取 top_k，融合两个结果，按加权分数排序（比如 `score = α * bm25_score + β * cos_sim`）

3. **在结果结构里存储检索来源**
   - 为了后续分析，可以在 `best_result` 里加一个字段，记录候选是 BM25 / 向量 / 混合得分。
   
   ```python
   result["retrieval"] = {
       "query": edit_code_abs,
       "selected_ids": [c["id"] for c in final_candidates],
       "method": "hybrid"
   }
   ```

4. **在 `main(cfg)` 增加参数控制**
   - 为了灵活，你需要在 `cfg_parsing()` 里加一些新参数，例如：
   
   ```python
   parser.add_argument("--retrieval_method", default="bm25", type=str,
                       choices=["bm25", "vector", "hybrid"])
   parser.add_argument("--hybrid_alpha", default=0.5, type=float,
                       help="Weight for BM25 score in hybrid retrieval.")
   ```
   
   - 在 `process()` 里根据 `cfg.retrieval_method` 判断是走 BM25 / 向量 / 混合。

---

### 模块四：遗传迭代（修改 evol_query.py）

**提示词重构**

在 `evol_query.py` 中修改 `prompt_construction` 函数。

---

## 数据集处理

### 数据集结构

数据集包含以下字段：

- **No**: 编号
- **来源**: 补丁的来源库或项目
- **优化类型**: 如条件分支优化、内存优化、指令集优化等
- **具体优化内容**: 原始代码和优化后的代码对
- **优化描述**: 对优化方法的文本描述

数据处理的目标是从这些补丁中提取优化模式（删除部分 `ds` 和新增部分 `df`），构建知识库，支持混合检索（BM25 关键词检索和向量语义检索），用于框架的各个模块。

### 数据清洗

**输入**：原始数据集文件（CSV/JSON 格式）

**流程**：

- 确保每个条目包含完整的原始代码和优化后的代码。如果只有优化描述，需从来源库中提取实际代码变更（例如，通过版本控制工具获取 diff）。
- 统一代码格式：移除无关字符（如多余空格、无关注释），但保留关键注释和代码结构。
- 对代码进行抽象化处理：替换变量名、寄存器名和立即数为通用标签（如 VAR、REG、NUM），以聚焦于代码结构变化而非具体标识符。

**输出**：清洗后的数据集，每个条目包含原始代码、优化后代码、优化类型、优化描述。

### 差异提取

对于每个补丁，计算原始代码与优化后代码的差异（使用 diff 算法或工具）。提取删除部分（`ds`）和新增部分（`df`），并存储为纯文本。

**输出**：每个补丁的 `ds` 和 `df` 文本。

### 向量嵌入生成

**处理**：使用句子转换器模型（如 all-MiniLM-L6-v2 或代码专用模型）为每个补丁的文本表示生成向量嵌入。向量维度应保持一致（例如 384 维）。

**输出**：每个补丁的向量嵌入（浮点数列表）。

### 知识库存储与索引构建

将处理后的数据保存为 JSONL 格式（`riscv_knowledge_base.jsonl`），每个条目包含：

- `id`: 补丁编号
- `source`: 来源
- `optimization_type`: 优化类型
- `optimization_description`: 优化描述
- `original_code`: 原始代码
- `optimized_code`: 优化后代码
- `ds`: 删除部分
- `df`: 新增部分
- `text_representation`: 文本表示
- `embedding`: 向量嵌入

**构建 BM25 索引**：使用文本表示集合作为语料库。

**构建 FAISS 向量索引**：将所有向量嵌入组合成矩阵；使用 FAISS 构建索引（如 IndexFlatL2），并保存索引文件（`faiss_index.bin`）。
