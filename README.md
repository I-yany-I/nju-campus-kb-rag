# 基于 BERT / LoRA / Prompt / RAG 的 AG News 文本分类对比系统

> 在 **AG News 四分类**任务上，统一训练、评估与推理流程，对比 **全量微调（BERT）**、**参数高效微调（LoRA）**、**纯提示词分类（Prompt）** 与 **检索增强（Sentence-BERT + FAISS + 小模型 LLM，含检索投票与混合决策）**，并配套 Gradio 交互与可复现指标输出。

## 系统架构（四条路径）

```
                    AG News 文本输入
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   BERT / LoRA      Prompt（Qwen2）    RAG 路径
   HuggingFace      单条指令 + 标签约束   Sentence-BERT 编码
   Trainer 训练      直接生成 0–3        FAISS 相似检索（top-k）
         │                 │                 │
         └─────────────────┴─────────────────┘
                           │
              统一 predictors 接口 → 准确率 / 加权 F1
                           │
              Gradio：单条 / 批量预测 + 方法切换
```

## 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 数据集 | Hugging Face `datasets` | `ag_news`，4 类：World / Sports / Business / Sci-Tech |
| 全量微调 | `bert-base-uncased` + Trainer | 分类头微调，checkpoint 与最终权重分目录保存 |
| 高效微调 | PEFT LoRA + `bert-base-uncased` | 低秩适配，与 BERT 共用统一评估入口 |
| 生成式 LLM | `Qwen/Qwen2-1.5B-Instruct`（fp16，`device_map=auto`） | Prompt 与 RAG 的生成后端（`transformers.pipeline`） |
| RAG 检索 | Sentence-Transformers `all-MiniLM-L6-v2` + FAISS `IndexFlatIP` | 向量 L2 归一化后内积检索；索引与向量缓存于 `vector_store/` |
| RAG 决策 | 检索加权投票 + LLM 在「弱 margin」时覆盖 | `margin < 0.15` 时采纳 LLM 标签，否则用投票结果 |
| 评估 | scikit-learn `accuracy` / `weighted F1` | 可选测试子集规模，输出 JSON/CSV 与柱状图 |
| 演示 | Gradio | 单条预测、批量（按行）预测、方法下拉切换 |

## 快速开始

### 1. 环境

```bash
pip install -r requirements.txt
```

建议使用 **Python 3.10+**，并已安装 **PyTorch**（CUDA 可选；LLM 推理在 GPU 上更快）。

### 2. 训练 BERT / LoRA（可选）

首次运行会从 Hugging Face 拉取 `ag_news` 与 `bert-base-uncased`。训练产物写入 `artifacts/checkpoints/` 与 `artifacts/models/`。

```bash
# 训练 BERT 与 LoRA（各跑一遍）
python train.py --model all --epochs 1

# 仅训练其一
python train.py --model bert --epochs 1
python train.py --model lora --epochs 1
```

默认 `TrainingArguments`：`lr=2e-5`，`batch` 16/64（train/eval），`max_length=128`，按 epoch 保存 checkpoint。

### 3. 统一评估（四方法对比）

对测试集**前 N 条**（默认 `N=200`）逐条推理，汇总准确率、加权 F1，并生成对比图与表格：

```bash
python evaluate.py
python evaluate.py --sample-size 500
```

在 **AG News 测试集全量（7600 条）** 上评测（BERT/LoRA 较快，Prompt/RAG 因逐条调用 LLM 会非常慢）：

```bash
python evaluate.py --sample-size 7600
```

输出位置：

- `artifacts/predictions/evaluation_summary.json` / `.csv`
- `artifacts/plots/accuracy_comparison.png`、`f1_comparison.png`

**说明：** Prompt 路径依赖生成稳定性与小模型能力，子集与随机性会导致指标波动；RAG 首次会构建 FAISS 索引（或加载 `vector_store/` 下缓存）。

### 4. Gradio 演示

```bash
python app.py
```

浏览器访问终端提示的本地地址（一般为 `http://127.0.0.1:7860`）。支持 **BERT / LoRA / PROMPT / RAG** 切换，**单条**与**批量（每行一条）**预测；RAG 结果中可展示检索到的相似新闻片段。

## 评估结果示例

在 **测试集前 200 条** 上的一次运行结果（仅供参考，以本地 `evaluate.py` 输出为准）：

| 方法 | Accuracy | Weighted F1 |
|------|----------|-------------|
| BERT | ~0.96 | ~0.96 |
| LoRA | ~0.925 | ~0.926 |
| Prompt | 视生成稳定性而定 | 视生成稳定性而定 |
| RAG | ~0.92 | ~0.92 |

完整对比请查看 `artifacts/predictions/evaluation_summary.json`。

## 项目结构

```
llm-text-classification-system/
├── train.py                 # 训练入口 → src/training/train_models.py
├── evaluate.py              # 评估入口 → evaluation/evaluate_models.py
├── app.py                   # Gradio 演示
├── evaluation/
│   └── evaluate_models.py   # 四方法评测、绘图与指标落盘
├── src/
│   ├── paths.py             # artifacts / data / vector_store 路径约定
│   ├── models/              # BERT / LoRA 封装与加载
│   ├── training/            # Trainer 训练逻辑
│   ├── inference/
│   │   └── predictors.py    # 统一 METHODS 与 get_model()
│   ├── prompt/              # Prompt 分类提示词与推理
│   ├── rag/
│   │   └── rag_pipeline.py  # 建库、检索、加权投票、混合 RAG 分类
│   └── llm/
│       └── shared_pipeline.py  # Qwen2 生成 pipeline 单例
├── artifacts/
│   ├── models/              # 训练完成后的 bert / lora
│   ├── checkpoints/         # 训练过程 checkpoint
│   ├── plots/               # 评估柱状图
│   └── predictions/         # evaluation_summary.json / csv
├── vector_store/            # FAISS 索引与 embeddings 缓存（自动生成）
├── data/                    # 可选本地数据扩展
└── requirements.txt
```

## 核心技术要点

### BERT 与 LoRA

- 同一 tokenizer（`bert-base-uncased`）与同一数据管线，便于公平对比参数量与推理路径差异。
- LoRA 通过 PEFT 注入适配层，适合在资源有限时复现「大模型下游任务」的常见做法。

### Prompt 路径

- 固定类别说明与输出格式（仅输出 `0`–`3`），用正则从生成文本中提取首个合法标签。
- 不经过检索，直接考察 **小尺寸指令模型在封闭标签集上的指令遵循能力**。

### RAG 路径

- 用 **句向量** 在训练集新闻上建库，查询阶段取 top-k 相似样本，构造 **few-shot 风格上下文** 再交给 Qwen2 生成标签。
- **检索加权投票**与 **LLM 输出** 做 **混合决策**：当投票分差较小（实现中 `margin < 0.15`）时更信任 LLM，否则以检索投票为主，兼顾稳健性与可解释性（可展示相似新闻）。

## 硬件建议

- **GPU**：推荐 8GB+ 显存（BERT/LoRA 训练与 Qwen2-1.5B 推理）；CPU 可跑 BERT/LoRA 推理，但 LLM 较慢。
- **磁盘**：预留空间用于 Hugging Face 模型缓存、`artifacts/` 与 `vector_store/` 索引文件。

## 相关项目

- [多模态 RAG 图像问答系统（CLIP + FAISS + Qwen2-VL）](https://github.com/I-yany-I/multimodal-rag)

---

*技术栈：PyTorch · Transformers · PEFT · Sentence-Transformers · FAISS · Gradio · AG News*
