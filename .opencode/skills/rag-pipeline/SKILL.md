---
name: rag-pipeline
description: "用于 RAG 知识库项目的全流程管理：文本处理、分层分块、向量索引、ChromaDB、BM25、混合检索、重排序、检索调试。触发词：rag、索引、embedding、向量、chromadb、bm25、分块、rerank、检索、搜索、知识库、process、index、hybrid_search。"
---

# RAG 管道技能

本技能覆盖 RAG 知识库项目的 **数据处理 → 索引构建 → 查询服务** 全流程。

## 项目结构

```
项目/
├── app.py                  # 统一入口（处理/索引/服务 + Streamlit UI）
├── rag_engine.py           # 检索引擎（分块/索引/向量/BM25/RRF/重排，与 UI 解耦）
├── books/                  # 源文本文件（.txt）
├── models/                 # 预训练模型（bge 系列）
├── chroma_db_v2/           # ChromaDB 向量数据库
├── cache_v2/               # BM25 缓存 + chunks.json
│   ├── chunks.json         # 处理后的分块数据
│   ├── bm25_index.pkl      # BM25 索引缓存
│   ├── bm25_tokens.pkl     # 分词后的语料缓存
│   ├── bm25_ids.pkl        # chunk id 对齐清单（BM25 排序→id 映射）
│   └── hash.txt            # 书籍哈希（用于缓存校验）
```

## 管道阶段

### 阶段一：处理（清洗 + 分块）
```bash
python app.py process
```
1. 读取 `books/` 下所有 `.txt`；2. 按句号/感叹号/问号分句；3. 分层分块（父块 512 token / 子块 128 token + 32 token 重叠）；4. 生成上下文前缀（书名、章节、人物名）；5. 保存到 `cache_v2/chunks.json`。

### 阶段二：索引（向量化入库）
```bash
python app.py index
```
1. 加载 chunks.json；2. bge-small-zh-v1.5 批量编码（**CLS pooling**，doc 侧不加 instruction）；3. 写入 ChromaDB books_v2；4. child_text 构建 BM25 + 保存 id 对齐清单；5. 保存 hash.txt。

### 阶段三：服务（启动查询）
```bash
python app.py        # 或 python app.py serve
```
混合检索：向量（query 侧加 BGE instruction）→ BM25 → RRF 融合(k=60) → bge-reranker-base 分批重排 → Top-5 喂给 Ollama。

## 核心配置（rag_engine.py）

```python
CHILD_MAX_TOKENS = 128     # 子块大小
PARENT_MAX_TOKENS = 512    # 父块大小
CHUNK_OVERLAP = 32         # 重叠 token 数
TOP_K = 5                  # 最终返回给 LLM 的结果数
RERANK_TOP_K = 30          # 重排序输入数 / 初筛候选数
RERANK_BATCH = 16          # reranker 单批条数（降低峰值显存）
RERANK_MAX_LENGTH = 256    # reranker 最大序列长度
RRF_K = 60                 # RRF 融合常数
```

> 模型加载策略：embedding 用 fp32（小模型，fp16 反而慢）；reranker 用 fp16（1GB→550MB，适配 8GB 统一内存）。
> 索引构建内置 0.5s/批 间歇冷却：MPS 持续满负荷会触发 GPU 降频（实测从 67 条/秒 降至 9 条/秒）。

## 调试方法

### 检查 chunks.json
```bash
python -c "
import json
data = json.load(open('cache_v2/chunks.json'))
print(f'总分块数: {len(data)}')
"
```

### 检查 ChromaDB
```bash
python -c "
import chromadb
col = chromadb.PersistentClient(path='./chroma_db_v2').get_collection('books_v2')
print(f'向量总数: {col.count()}')
"
```

### 测试检索（复用生产引擎）
```python
from rag_engine import RAGEngine
engine = RAGEngine()
results = engine.hybrid_search("孙悟空大闹天宫", top_k=3)
for r in results:
    print(f"[{r['book']}] 相关度: {r['rerank_score']:.4f}")
    print(f"  {r['child_text'][:80]}...")
```

或运行完整检索质量测试：`python test_retrieval_quality.py`（报告写入 test_retrieval_report.json）。

## 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| 找不到 chunks.json | 没有运行 process | python app.py process |
| ChromaDB 为空 | 没有运行 index | python app.py index |
| 索引很慢 | MPS GPU 降频 | 已内置间歇冷却；避开与其它 GPU 任务并行 |
| 检索质量差 | 分块大小不对 | 调整 CHILD_MAX_TOKENS / PARENT_MAX_TOKENS 后 process+index |
| BM25 缓存过期 | 书籍内容变了 | 重新运行 process + index |
