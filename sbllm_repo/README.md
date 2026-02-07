# SBLLM

这是我们 ICSE'25 论文的官方仓库："Search-Based LLMs for Code Optimization"（基于搜索的 LLM 代码优化）。

## 环境依赖

- **Python**: == 3.9.12
- **C++**: C++17 标准
- **编译器**: GCC 9.4.0
- **操作系统**: Linux (推荐)

在项目根目录下运行以下命令安装 Python 依赖：

```sh
pip install -r requirements.txt
```

## 使用说明 (统一入口)

项目现在提供了一个统一的、跨平台的 Python 入口脚本 `start_evaluation.py`，它可以自动管理 Docker 环境并执行各项评估任务。

### 运行 SBLLM 评估方案

执行标准的 SBLLM 优化流水线：

```bash
python start_evaluation.py standard
```

### 运行基准测试 (RVV-Bench / VecIntrin)

执行特定的基准测试：

- **RVV-Bench**: `python start_evaluation.py rvv-bench [--full]`
- **VecIntrin**: `python start_evaluation.py vecintrin`

该脚本已完全替代了旧版的 `start_evaluation.bat` 和 `run_all.bat`。

---

## 历史用法 (已弃用)

项目默认设置 `ns=3` 且 `iteration=4`。此设置与论文中的实验配置保持一致。

## 数据准备

请按照 `processed_data/` 目录下的说明下载用于实验的数据集和测试用例。

## 基准方法 (Baselines)

其他基准方法的源代码位于 `baselines/` 目录下。

运行 **直接指令 (Direct Instruction)**:

```bash
cd baselines  
bash direct.sh
```

运行 **少样本学习 (In-Context Learning)**:

```bash
cd baselines  
bash icl.sh
```

运行 **检索增强生成 (RAG)**:

```bash
cd baselines  
bash rag.sh
```

运行 **思维链 (Chain-of-Thought)**:

```bash
cd baselines  
bash cot.sh
```
