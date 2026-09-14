# 四大名著 RAG 知识库 — 项目文档

## 1. 项目概述

基于四大名著（三国演义、水浒传、红楼梦、西游记）的 RAG（检索增强生成）知识库系统。支持本地文本处理、向量索引构建、语义检索和 LLM 生成回答。

## 2. 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    用户查询                          │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Streamlit UI (app.py)                              │
│  - 查询输入 / 对话历史 / 结果展示                    │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  RAG Engine (rag_engine.py)                         │
│  - Embedding: bge-small-zh-v1.5 (fp32)             │
│  - 检索: Qdrant稠密+稀疏混合 → RRF融合              │
│  - Rerank: bge-reranker-base (fp16)                │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Qdrant Docker (localhost:6333)                     │
│  - 稠密向量: BGE embeddings (512维, COSINE)        │
│  - 稀疏向量: BM25 (IDF)                             │
│  - 融合: Reciprocal Rank Fusion (RRF)              │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Ollama (MODEL 取自 .env，当前 qwen2.5:7b)           │
│  - 生成回答                                          │
└─────────────────────────────────────────────────────┘
```

## 3. 核心文件说明

| 文件 | 说明 | 行数 |
|------|------|------|
| `app.py` | 统一入口：CLI 命令 + Streamlit UI | ~253 |
| `rag_engine.py` | RAG 核心引擎：分块/嵌入/索引/检索 | ~630 |
| `config.py` | 配置管理（Pydantic Settings） | ~87 |
| `check_health.py` | 系统健康检查 | ~128 |
| `test_rag.py` | 自动化测试套件 | ~400 |

## 4. 数据处理流程

### 4.1 文本分块（build_chunks）

```
原始文本 → 引号归一 → sentencex 分句 → 语义定界 → 父块(512 tokens) → 子块(128 tokens, 重叠32)
                                      └→ 超长句 token 兜底：标点→换行→分号→逗号→硬切
```

- **分块策略**: 层次化分块，父块提供完整上下文，子块提供精确匹配
- **分句**: `sentencex` 单层分句（引语内部不切分，引号错位 24%~36% → 0%）。它自身把 `\n\n` 与 `\r\n\r\n` 都当句边界且从不跨段，因此**不再预切段落**——实测四本书 66079 句，"先按段落预切再逐段分句"与"整本书一次分句"输出序列逐元素完全相同
- **sentencex 是硬依赖**: 缺库直接 `ImportError`，刻意不降级（内置字符扫描分句会静默把闭合引号切给下一句）
- **超长句兜底**: 无标点文言段最长 673 token，不兜底会撑破 CHILD_MAX_TOKENS 并被 tokenizer 静默截断。全库 779 句超 128 token（1.18%），19 句超 512 token
- **语义定界**: bge-small 逐句编码 + 相邻句余弦相似度阈值 0.6 且当前块 ≥500 字才切分（详见 `_semantic_split` 注释）
- **contextual_text**: 当前恒等于 `child_text`（不再拼接上下文前缀）。旧前缀形如 `《书名》 > 第五回 > 涉及: 武松\n\n<正文>`，实测 77.6% 的块除《书名》外无任何信息、回目正则整体只命中 1.4%，故已移除生成逻辑；字段本身保留（稠密索引与 rerank 消费它，`check_health.py` 校验其存在）
- **改动分块后请用 `compare_chunks.py` 做 A/B**：`python app.py process` 后执行 `python compare_chunks.py`，验证是否零影响/仅影响指定字段

### 4.2 向量索引（build_index）

```
chunks.json → Embedding (bge-small) → Qdrant (COSINE)
```

- **模型**: BGE-small-zh-v1.5 (fp32, 512 维)
- **设备**: 支持 MPS (Apple Silicon GPU) / CPU
- **批处理**: batch_size=128

### 4.3 检索流程（hybrid_search）

```
Query → Embedding(query 侧加 instruction)
      → Qdrant prefetch[稠密 top-30 ‖ 稀疏 top-30] → 服务端 RRF 融合 30 条
      → Reranker top-5 → 返回结果
```

- **检索**: 稠密（bge-small）+ 稀疏（jieba 分词 + Qdrant 内置 BM25，`Modifier.IDF`），每通道取 `RERANK_TOP_K=30`，由 Qdrant 服务端 `FusionQuery(RRF)` 融合
- **重排**: bge-reranker-base, rerank_batch=16, 输入 30 条 → 输出 top_k=5

> `search_limit` 是旧版 `chatbot.py` 的参数（已归档到 `bak/`），现行代码中不存在。
> RRF 的 k 值也不可配置——融合在 Qdrant 服务端完成。

## 5. 配置参数

### 5.1 分块参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `child_max_tokens` | 128 | 子块最大 token 数 |
| `parent_max_tokens` | 512 | 父块最大 token 数 |
| `chunk_overlap` | 32 | 分块重叠 token 数 |

### 5.2 检索参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k` | 5 | 最终返回结果数 |
| `rerank_top_k` | 30 | 每通道召回数 / Reranker 候选数 |
| `rerank_batch` | 16 | Reranker 批处理大小 |
| `rerank_max_length` | 256 | Reranker 最大序列长度 |

> 以上均为 `rag_engine.py` 中的常量，当前没有环境变量覆盖（改需改代码）。
> `search_limit` 属于旧版实现，已不存在。

### 5.3 模型路径

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embed_model_path` | `./models/bge-small-zh-v1.5` | Embedding 模型 |
| `rerank_model_path` | `./models/bge-reranker-base` | Reranker 模型 |

## 6. 使用方法

### 6.1 快速开始

```bash
# 一键构建（推荐）
python app.py build

# 强制全量重建
python app.py build -f

# 仅处理分块
python app.py process

# 仅构建索引
python app.py index

# 启动查询服务
python app.py
```

### 6.2 健康检查

```bash
python check_health.py
```

### 6.3 运行测试

```bash
python test_rag.py
```

## 7. 测试结果

### 7.1 自动化测试（9/9 通过）

| 测试项 | 结果 | 详情 |
|--------|------|------|
| 配置管理 | ✅ 9/9 | 配置加载、参数验证 |
| 分块逻辑 | ✅ 6/6 | 空文本、短文本、长文本、token 限制 |
| 嵌入模型 | ✅ 5/5 | 加载、维度、归一化 |
| Reranker 模型 | ✅ 4/4 | 加载、重排、空列表 |
| 分块构建 | ✅ 6/6 | 构建、结构、前缀 |
| Qdrant 索引 | ✅ 3/3 | 记录数一致性 |
| 检索引擎 | ✅ 9/9 | 基本检索、跨书检索、空查询 |
| 端到端效果 | ✅ 3/3 | 准确率、命中率、响应时间 |
| 健康检查 | ✅ 4/4 | 模型、书籍文件 |

### 7.2 效果指标

| 指标 | 值 |
|------|-----|
| 书名匹配准确率 | **100%** (8/8) |
| 关键词命中率 | **87.5%** (7/8) |
| 平均检索时间 | **1.09s** |
| 索引总量 | 19023 条 |

### 7.3 跨书检索验证

| 查询 | 预期 | 实际 | 结果 |
|------|------|------|------|
| 孙悟空大闹天宫 | 西游记 | 西游记 | ✅ |
| 林黛玉葬花 | 红楼梦 | 红楼梦 | ✅ |
| 诸葛亮草船借箭 | 三国演义 | 三国演义 | ✅ |
| 武松打虎 | 水浒传 | 水浒传 | ✅ |

## 8. 代码审查与修复

### 8.1 已修复问题

| 问题 | 严重程度 | 修复内容 |
|------|---------|---------|
| app.py 未使用的导入 | 低 | 移除 `List`, `SearchResult` |
| app.py 流连接泄漏 | 中 | 添加 `finally: r.close()` |
| app.py Ollama 异常处理 | 中 | 区分 `ConnectionError` / `Timeout` |
| config.py 未使用的导入 | 低 | 移除 `os`, `Optional` |
| rag_engine.py Qdrant host 检查 | 中 | 使用 `strip()` 确保非空白 |
| rag_engine.py delete_collection 异常 | 低 | 添加 debug 日志 |
| rag_engine.py 模块级常量 | 中 | 改为惰性加载（`__getattr__`） |
| rag_engine.py build_index 强制 CPU | 中 | 改为使用 MPS GPU |

### 8.2 代码质量评分

**7.5/10** — 主项目架构清晰，修复后质量提升。

## 9. 已知限制

1. **MPS 降频 / GPU 争用**: Apple Silicon 长时间高负载或与其它 GPU 任务并发时，索引构建速度会大幅波动（实测同一份代码同一批数据：119 条/秒 与 23 条/秒）。测速异常时先怀疑机器负载，再怀疑代码。
2. **生成模型**: 由 `.env` 的 `MODEL` 决定（当前 `qwen2.5:7b`），代码默认值是 `qwen3.5:2b-q4_K_M`；缺 python-dotenv 时曾静默使用默认值，现已改为直接报错。
3. **上下文前缀已移除**：原先用 `NAME_PATTERNS` 硬编码的 21 个人名 + 回目正则拼接前缀写入 `contextual_text`。实测该前缀 77.6% 的块除《书名》外无任何信息，回目仅命中 1.4%，故已删除生成逻辑，`contextual_text` 现恒等于 `child_text`。若要做真正的语义前缀（contextual retrieval，用 LLM 为每块生成"这段讲什么"），应作为独立特性加入并配 A/B 评测。
4. **单机模式**: Qdrant 使用本地存储，不支持多人并发；引擎为进程内单例、无锁。

## 10. 项目结构

```
chatbot/
├── app.py              # 统一入口（CLI + Streamlit）
├── rag_engine.py       # RAG 核心引擎
├── config.py           # 配置管理
├── check_health.py     # 健康检查
├── compare_chunks.py   # 分块 A/B 比较器（验证分块改动是否零影响）
├── test_rag.py         # 自动化测试
├── requirements.txt    # 依赖
├── .env                # 环境变量
├── books/              # 源文本
│   ├── 三国演义.txt
│   ├── 水浒传.txt
│   ├── 红楼梦.txt
│   └── 西游记.txt
├── models/             # 预训练模型
│   ├── bge-small-zh-v1.5/
│   └── bge-reranker-base/
├── cache_v2/           # 分块产物
│   ├── chunks.json                    # 当前使用的分块（21159 条）
│   ├── chunks_before_simplify.json    # 分块简化前的基线，供 compare_chunks.py 对照
│   └── chunks_baseline_20260913.json  # 更早的分块，供检索级 A/B 对照
├── qdrant_storage/     # Qdrant Docker 数据（docker-compose 挂载此目录）
└── bak/                # 已废弃产物（ChromaDB 库、rank_bm25 缓存、旧基线）
```

## 11. 依赖

```
streamlit>=1.32.0
requests>=2.31.0
qdrant-client>=1.9.0
jieba>=0.42.1
torch>=2.2.0
transformers>=4.38.0
numpy>=1.26.0
pydantic-settings>=2.0.0
scikit-learn>=1.0.0
```
