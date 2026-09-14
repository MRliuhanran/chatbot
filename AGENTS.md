# AGENTS.md - 项目配置

本文件配置 opencode agents 如何与这个 RAG 知识库项目交互。

## 项目概述

这是一个**四大名著 RAG 知识库**，采用 Qdrant Docker 部署，支持稠密+稀疏混合搜索。

## 快速开始

```bash
# 1. 启动Qdrant服务
docker compose up -d

# 2. 构建索引
python app.py process
python app.py index

# 3. 健康检查
python check_health.py

# 4. 启动服务
python app.py
```

## 项目结构

```
├── app.py              # 统一入口：CLI 命令 + Streamlit UI
├── rag_engine.py       # RAG 核心引擎：分块/嵌入/索引/检索
├── check_health.py     # 系统健康检查
├── compare_chunks.py   # 分块 A/B 比较器（验证分块改动是否零影响）
├── docker-compose.yml  # Docker编排文件
├── qdrant/
│   └── config.yaml     # Qdrant配置文件
├── books/              # 源文本文件（四大名著 .txt）
├── models/             # 预训练模型
│   ├── bge-small-zh-v1.5/  # 向量化模型
│   └── bge-reranker-base/  # 重排序模型
├── qdrant_storage/     # Qdrant Docker数据存储
├── cache_v2/           # 分块缓存
├── .env                # 环境变量配置
└── requirements.txt    # 依赖清单
```

## 核心配置

```python
# Qdrant Docker配置
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "books_v2"

# 分块参数
child_max_tokens = 128    # 子块：精确匹配
parent_max_tokens = 512   # 父块：完整上下文
chunk_overlap = 32        # 重叠token数

# 分句（rag_engine._split_sentences）
#   sentencex 单层分句（它自身把 \n\n 当句边界、从不跨段，故不再预切段落）
#   → 超长句按 token 逐级兜底（标点→换行→分号→逗号→硬切）
#   sentencex 是硬依赖：缺库直接 ImportError，刻意不降级（降级会静默切错引号）
sentencex >= 1.0.30       # MIT、零依赖
NORMALIZE_QUOTES = "1"    # 直引号 " 归一为弯引号（红楼梦专治），置 0 关闭

# 检索参数
top_k = 5                 # 最终返回给LLM的结果数
rerank_top_k = 30         # Rerank候选数
rerank_batch = 16         # reranker 单批条数
rerank_max_length = 256   # reranker 最大序列长度

# 模型
embed_model_path = "./models/bge-small-zh-v1.5"
rerank_model_path = "./models/bge-reranker-base"
```

## 常用操作

### 启动Qdrant服务
```bash
docker compose up -d
```

### 添加新书
1. 把 `.txt` 文件放到 `books/` 目录
2. 运行 `python app.py process`
3. 运行 `python app.py index`

### 健康检查
```bash
python check_health.py
```

## 技术栈

- **向量数据库**: Qdrant (Docker部署)
- **搜索方式**: 稠密向量 + BM25稀疏向量 → RRF融合
- **分句**: sentencex 单层分句（引语不可切，且自身把 `\n\n` 当句边界）+ 超长句 token 兜底
- **嵌入模型**: BGE-small-zh-v1.5
- **重排序模型**: BGE-reranker-base
- **UI**: Streamlit
- **生成模型**: Ollama (qwen3.5:2b)
