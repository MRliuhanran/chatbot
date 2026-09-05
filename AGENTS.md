# AGENTS.md - 项目配置

本文件配置 opencode agents 如何与这个 RAG 知识库项目交互。

## 项目概述

这是一个**四大名著 RAG 知识库**，流程：处理文本 → 构建向量索引 → Streamlit 查询服务。

## 快速开始

```bash
# 第一步：处理文本数据
python app.py process

# 第二步：构建向量索引
python app.py index

# 第三步：启动查询服务
python app.py
```

## 项目结构

```
├── app.py                  # 统一入口（处理/索引/服务 + Streamlit UI）
├── rag_engine.py           # 检索引擎（分块/索引/向量/BM25/RRF/重排，与 UI 解耦）
├── books/                  # 源文本文件
├── models/                 # 预训练模型
│   ├── bge-small-zh-v1.5/  # 向量化模型
│   └── bge-reranker-base/  # 重排序模型
├── chroma_db_v2/           # ChromaDB 向量数据库
├── cache_v2/               # BM25 缓存 + chunks.json
└── check_health.py         # 一键健康检查
```

## 核心配置

```python
# 分块参数（rag_engine.py）
CHILD_MAX_TOKENS = 128    # 子块：精确匹配
PARENT_MAX_TOKENS = 512   # 父块：完整上下文
CHUNK_OVERLAP = 32        # 重叠token数

# 检索参数（rag_engine.py）
TOP_K = 5                 # 最终返回给LLM的结果数
RERANK_TOP_K = 30         # 初筛候选数 / 重排序输入数
RERANK_BATCH = 16         # reranker 单批条数（降低峰值显存）
RERANK_MAX_LENGTH = 256   # reranker 最大序列长度
RRF_K = 60                # RRF 融合常数

# 模型（embedding 用 fp32；reranker 用 fp16 半精度，适配 8GB 统一内存）
EMBED_MODEL_PATH = "./models/bge-small-zh-v1.5"
RERANK_MODEL_PATH = "./models/bge-reranker-base"
```

> 注：所有检索/索引/分块相关常量与函数均位于 `rag_engine.py`，`app.py` 通过 `import rag_engine as RE` 复用并做兼容再导出。

## 常用操作

### 添加新书
1. 把 `.txt` 文件放到 `books/` 目录
2. 运行 `python app.py process`
3. 运行 `python app.py index`

### 调整分块大小
1. 修改 `rag_engine.py` 中的 `CHILD_MAX_TOKENS` / `PARENT_MAX_TOKENS`
2. 运行 `python app.py process`
3. 运行 `python app.py index`

### 调试检索

检索逻辑已解耦到 `rag_engine.py`，可直接 import 复用：

```python
from rag_engine import RAGEngine
engine = RAGEngine()
results = engine.hybrid_search("孙悟空大闹天宫", top_k=5)
```

要离线验证检索链路（复用同一引擎，非复制粘贴），运行：

```bash
python test_retrieval_quality.py
```

它会调用 `RAGEngine.hybrid_search`（向量 + BM25 + RRF + Rerank），校验 ChromaDB/BM25 id 清单与数量一致性，报告写入 `test_retrieval_report.json`。

## Python 环境

- Python 3.10（系统安装路径 `/Library/Frameworks/Python.framework/Versions/3.10/bin/python3`）
- 依赖包：torch, chromadb, transformers, rank_bm25, jieba, streamlit, numpy（均已装在系统 Python site-packages）

## GitHub 配置

- 用户名：MRliuhanran
- 仓库：chatbot（私有）
- Token 已配置在 `~/.zshrc`（环境变量 `GITHUB_TOKEN`）
- API 操作：`curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/...`
- 推送代码：`git push -u origin main`
