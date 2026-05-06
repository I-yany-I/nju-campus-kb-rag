# 南京大学校园办事指南 RAG 问答系统

> 面向 **南京大学学生的校园 IT / 教务办事指南**，构建一个可引用、可拒答、可消融的文本知识库问答系统。BERT、LoRA、Prompt、RAG 在一条产品链路中各司其职：文本召回、BERT 重排、Prompt 约束生成、LoRA 领域适配。

## 项目定位

本项目解决的是学生常见办事问题，例如：

- 「校园 VPN 怎么用？」
- 「统一身份认证密码忘了怎么办？」
- 「成绩单或在读证明去哪里办理？」
- 「选课、退课、重修类问题应该看哪个流程？」

系统只基于知识库片段作答，回答中返回引用来源；当知识库没有依据时明确拒答，避免把通用大模型的记忆当成学校政策。

> 知识库（`data/campus_kb/nju_official_kb.jsonl`）收录了信息化服务、教务服务、学生服务、财务、出国（境）等多类办事指南，共 **75 条**文档，覆盖 VPN、统一身份认证、信息门户、移动 App、网盘、视频会议、培养方案、选课、考试、推免、四六级、心理咨询、医保、就业三方、档案、离校、学费水电、签证证明等高频场景。正式部署时应以南京大学各主管部门最新公开通知为准。

## 系统架构

```
用户问题
  │
  ├─ 查询改写 Prompt（可选：让口语问题更适合检索）
  │
  ▼
BM25 关键词召回 + Sentence-Transformer 稠密召回
  │
  ▼
RRF 融合候选
  │
  ▼
BERT Cross-Encoder 重排（可消融对比）
  │
  ▼
Top-N 引用片段
  │
  ▼
Qwen2 文本生成 / LoRA 适配模型（可选）
  │
  ▼
带引用的中文回答 + 拒答判断
```

## 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 知识库 | JSONL 文档 + 段落分块 | 含 `source`、`source_type`、`updated_at`、`collected_at` 等元数据 |
| 稠密检索 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + FAISS | 适合中文问句与办事片段的语义召回 |
| 稀疏检索 | BM25（rank-bm25）+ RRF 融合 | 保留「VPN」「统一身份认证」「成绩单」等关键词优势 |
| BERT 重排 | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 对 `(问题, 片段)` 精排，提高引用命中率 |
| Prompt | 有据作答、引用格式、拒答策略 | 约束模型只依据检索证据回答 |
| LoRA | PEFT adapter 可选加载 | 用少量问答对适配校园办事口吻与输出格式 |
| 生成端 | `Qwen/Qwen2-1.5B-Instruct` 或抽取式 fallback | 默认抽取式回答，配置 `generation.backend: llm` 开启 LLM |
| 演示 | Gradio | 单轮问答、引用片段、来源与检索分数展示 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

建议使用 Python 3.10+。CPU 可运行检索和抽取式回答；开启 Qwen2 生成时建议使用 CUDA。

### 2. 构建校园知识库索引

```bash
python build_campus_kb_index.py
```

默认读取 `data/campus_kb/nju_official_kb.jsonl`，把 FAISS 索引与元数据写入 `vector_store/campus_kb/`。强制重建索引：

```bash
python build_campus_kb_index.py --force
```

扩充知识库只需在 JSONL 中追加条目，字段说明：

```json
{
  "id": "nju-it-vpn",
  "title": "校园 VPN 使用说明",
  "department": "信息化建设管理服务中心",
  "source": "https://istic.nju.edu.cn/vpn",
  "source_type": "official_page",
  "updated_at": "2026-05-06",
  "collected_at": "2026-05-06",
  "tags": ["VPN", "校外访问"],
  "text": "正文内容..."
}
```

### 3. 启动 Gradio 演示

```bash
python app.py
```

页面展示回答、引用片段、来源标题和检索分数。示例问题：

- `校园网外怎么访问校内资源？`
- `统一身份认证密码忘记了怎么办？`
- `成绩单和在读证明应该找哪个部门？`

### 4. 运行评估

```bash
python evaluate_campus_kb.py
```

从 `data/campus_kb/eval_questions.jsonl`（81 条）加载评估问题，输出整体指标与按类别（it / academic / student / finance / international / refusal）分组的指标：

| 指标 | 含义 |
|------|------|
| `citation_hit_rate` | 至少命中一个期望文档的问题占比 |
| `citation_recall_at_k` | 所有期望文档被检索到的比例 |
| `refusal_accuracy` | 知识库外问题被正确拒答的比例 |
| `false_refusal_rate` | 有答案但系统拒答的比例（越低越好） |

## 配置说明（`config/campus_kb.yaml`）

| 配置项 | 说明 |
|--------|------|
| `knowledge_base.path` | 知识库 JSONL 路径 |
| `retrieval.hybrid_enabled` | 是否启用 BM25 + 稠密召回融合 |
| `retrieval.cross_encoder.enabled` | 是否启用 BERT Cross-Encoder 重排 |
| `generation.backend` | `extractive`（默认）或 `llm` |
| `generation.lora_adapter_path` | LoRA adapter 路径；为空则使用基座模型 |
| `prompt.refusal_threshold` | 低相似度时触发拒答 |

## 项目结构

```
nju-campus-kb-rag/
├── app.py                         # Gradio 校园问答演示
├── build_campus_kb_index.py       # 构建 / 刷新 FAISS 索引
├── evaluate_campus_kb.py          # 引用命中与拒答评估
├── config/
│   └── campus_kb.yaml             # 知识库、检索、重排、生成配置
├── data/
│   └── campus_kb/
│       ├── nju_official_kb.jsonl  # 75 条办事指南知识库
│       └── eval_questions.jsonl   # 81 条评估问题集
├── evaluation/
│   └── metrics.py                 # citation_hit_rate / refusal_accuracy 等指标
├── src/
│   └── campus_kb_rag/
│       ├── config.py              # 配置加载与路径解析
│       ├── documents.py           # JSONL 读取与段落分块
│       ├── retriever.py           # BM25 + FAISS + Cross-Encoder
│       ├── generator.py           # Prompt、LLM/LoRA、抽取式 fallback
│       └── pipeline.py            # 端到端问答流程
├── vector_store/                  # 本地索引缓存（自动生成，不提交）
└── requirements.txt
```

## 面试讲法

这个项目的核心不是「BERT、LoRA、Prompt、RAG 谁更强」，而是一个合理的文本 RAG 系统分层：

- **RAG** 是主链路：把南京大学校园办事材料作为外部证据，避免模型凭记忆回答。
- **BERT Cross-Encoder** 是检索增强模块：对候选片段精排，提高引用准确率，可通过关闭 `cross_encoder.enabled` 消融验证其贡献。
- **Prompt** 是输出约束：要求模型给出引用、说明依据不足时拒答。
- **LoRA** 是生成端可选适配：用少量校园办事问答把语气和格式适配到真实服务场景。

## 与多模态 RAG 的区别

本项目处理的是**纯文本办事指南**，证据是文档段落；另一个多模态 RAG 项目处理的是**图像证据**，核心在 CLIP 图文空间与 Qwen2-VL。两个项目都叫 RAG，但输入模态、检索对象和评估重点不同。

## 硬件建议

- **CPU**：可运行索引构建、检索、抽取式回答与评估。
- **GPU**：推荐用于 Qwen2 生成、Cross-Encoder 大批量重排或 LoRA adapter 推理。
- **磁盘**：预留 Hugging Face 模型缓存与 `vector_store/` 索引空间。

## 相关项目

- [多模态 RAG 图像问答系统（CLIP + FAISS + Qwen2-VL）](https://github.com/I-yany-I/multimodal-rag)

---

*技术栈：PyTorch · Transformers · PEFT · Sentence-Transformers · FAISS · rank-bm25 · Gradio*
