---
name: rag-pipeline
description: "用于 RAG 知识库项目的全流程管理：文本处理、分层分块、Qdrant 向量索引、稠密+稀疏混合检索、重排序、检索调试。触发词：rag、索引、embedding、向量、qdrant、bm25、稀疏向量、分块、分句、rerank、检索、搜索、知识库、process、index、hybrid_search。"
---

# RAG 管道技能

本技能覆盖 RAG 知识库项目的 **数据处理 → 索引构建 → 查询服务** 全流程。

> **唯一权威实现**：`app.py`（CLI/UI）+ `rag_engine.py`（引擎）。
> 检索逻辑只此一份，工具脚本一律通过 `import rag_engine` 复用，不要复制实现。

## 项目结构

```
项目/
├── app.py                  # 统一入口（process / index / serve + Streamlit UI）
├── rag_engine.py           # 检索引擎（分块/嵌入/Qdrant索引/混合检索/重排）
├── check_health.py         # 健康检查（含稠密/稀疏真检查）
├── compare_chunks.py       # 分块 A/B 比较器
├── compare_ab.py           # 检索级 A/B（两个集合）
├── verify_qdrant.py        # 稠密/稀疏通道校验
├── reindex_sparse.py       # 只重建稀疏向量
├── docker-compose.yml      # Qdrant 编排
├── pytest.ini              # 测试分层 marker 配置
├── tests/                  # 自动化测试（L0~L3，见"调试方法"）
├── books/                  # 源文本（.txt）
├── models/                 # bge-base-zh-v1.5（向量，768维）、bge-reranker-base（重排）
├── cache_v2/chunks.json    # 分块缓存（process 的产物）
└── qdrant_storage/         # Qdrant Docker 的数据卷（compose 挂载）
```

## 管道阶段

### 阶段一：处理（分块）
```bash
python app.py process
```
1. 读取 `books/` 下所有 `.txt`（坏字节替换为 U+FFFD；**不再做 CRLF 归一** —— sentencex 自身把 `\r\n\r\n` 当句边界）；
2. 引号归一（`NORMALIZE_QUOTES=1`）：混合型（红楼梦，文中已有 “）按位置判定；纯直引号型按出现次序成对交替；
3. `sentencex` 分句（**不预切段落** —— sentencex 自身把 `\n\n` 当句边界且从不跨段）；
4. 超长句按 token 逐级兜底（标点 → 换行 → 分号 → 逗号 → 硬切）；
5. 分层分块（父块 512 token / 子块 128 token + 32 token 重叠）；
6. 过滤短块（<=5 字）→ **先过滤再编号**（`chunk_index` 连续、`total_chunks` 等于实际条数）→ 保存到 `cache_v2/chunks.json`。

> `contextual_text` 恒等于 `child_text`：上下文前缀生成已移除（实测 77.6% 的块除《书名》外无信息量，回目正则仅命中 1.4%）。

### 阶段二：索引（向量化入库）
```bash
python app.py index
```
1. 加载 chunks.json；
2. **bge-base-zh-v1.5（768维）** 批量编码（**CLS pooling**，doc 侧不加 instruction）→ 稠密向量；
3. 稀疏向量：**jieba 分词后用空格连接，交给 Qdrant 内置 BM25 模型打分**（`Document(model="qdrant/bm25", options=...)`），IDF 由集合的 `Modifier.IDF` 在查询时施加；
4. 写入 Qdrant：命名向量 `dense` + `sparse`，point id 用**整数序号**（Qdrant 只接受无符号整数或 UUID），原始 id 存在 `payload.chunk_id`；
5. 每个 point 写入 `build_id`，供检索端识别索引换代。

### 阶段三：服务（启动查询）
```bash
python app.py        # 或 python app.py serve
```
混合检索：query 侧加 BGE instruction 生成稠密向量 → **prefetch 稠密+稀疏 → Qdrant 服务端 RRF 融合** → bge-reranker-base 分批重排 → Top-5 喂给 Ollama。

## 核心配置（rag_engine.py）

```python
CHILD_MAX_TOKENS = 128     # 子块大小
PARENT_MAX_TOKENS = 512    # 父块大小
CHUNK_OVERLAP = 32         # 重叠 token 数
TOP_K = 5                  # 最终返回给 LLM 的结果数
RERANK_TOP_K = 20          # 重排序输入数 / 初筛候选数（每通道召回数）
RERANK_BATCH = 16          # reranker 单批条数（降低峰值显存）
RERANK_MAX_LENGTH = 384    # reranker 最大序列长度（父块 512 token，256 会砍掉一半）
EMBED_MODEL_PATH = "./models/bge-base-zh-v1.5"   # 768 维
COLLECTION_NAME = "books_v3"
```

> **RRF 的 k 值不在本项目可配**：融合由 Qdrant 服务端 `FusionQuery(RRF)` 完成，其内部 k 固定。历史上那个 `RRF_K = 60` 是死配置，已删除。
>
> 模型加载策略：embedding 用 fp32（小模型，fp16 反而慢）；reranker 用 fp16（1GB→550MB，适配 8GB 统一内存）。
>
> 稠密/稀疏是同一字段的两种表示，**文档侧与查询侧必须成对同构**；稀疏侧的 jieba 分词 + options 必须两侧一致，否则召回自毁。两者都由 `rag_engine.sparse_encode()` 统一产出，不要绕开它。

## 调试方法

### 健康检查（首选）
```bash
python check_health.py --offline   # 跳过 Ollama 在线检查，最快
```

### 检查 chunks.json
```bash
python -c "
import json
data = json.load(open('cache_v2/chunks.json'))
print(f'总分块数: {len(data)}')
"
```

### 检查 Qdrant
```bash
curl -s localhost:6333/collections/books_v3 | python -m json.tool | head -20
```

### 自动化测试（首选 —— 改完代码先跑这个）
```bash
pytest -m unit          # L0 纯函数，<1s，无需模型/Qdrant/语料
pytest -m needs_chunks  # L1 分块不变量
pytest -m needs_qdrant  # L2 索引一致性
pytest -m slow          # L3 检索质量回归（与黄金基线比，需先录制）
python -m tests.record_baseline   # 录制/更新 L3 黄金基线
```

> 改分块参数 / 分句器 / 引号归一后，**必须**跑 `pytest -m unit + needs_chunks`。
> 改模型或检索参数后跑 `pytest -m slow`（零容差，任何下降都判为回归）。

### 混合检索验证（测稠密 vs 混合的真实差距）
```bash
python verify_qdrant.py        # 集合静态检查 + 通道检查 + 召回对比
python compare_ab.py           # 两个集合的检索级 A/B
python compare_chunks.py A.json B.json   # 两份分块产物逐条比对（暴露边界漂移）
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

### 只重建稀疏向量（换分词方案时用，不必重跑 embedding）
```bash
python reindex_sparse.py       # 20,739 条约 50s；对比全量 build_index 约 3 分钟
```

## 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| 找不到 chunks.json | 没有运行 process | `python app.py process` |
| Qdrant 集合为空 | 没有运行 index | `python app.py index` |
| 检索"无结果"但无报错 | 历史实现会吞异常 | 已修为抛出/告警；若复现请查 `_get_chunks` 的 scroll |
| 稀疏通道无贡献 | 稀疏向量没写进去（Qdrant 不会自动生成） | `python check_health.py` 的 [4] 会直接报出来 |
| 中文稀疏召回差 | 用了 Qdrant 默认 word 分词器 | 确认走 `sparse_encode()`（jieba 预分词） |
| 索引很慢 | MPS 与其它 GPU 任务并发争用（实测 119 条/秒 → 23 条/秒） | 避开并发；时序差异极大时先怀疑机器负载，而非代码 |
| 重建索引后服务仍在用旧数据 | 进程内分块缓存 | 已加 `build_id` 自动失效；若仍复现请查 `_current_build_id()` |
| 检索质量差 | 分块大小 / 分句不对 | 调 CHILD_MAX_TOKENS / PARENT_MAX_TOKENS，再 process + index；用 `pytest -m needs_chunks` 验证不变量 |
| 改了 `.env` 没反应 | 缺 python-dotenv | 已改为直接报错提示；`pip install -r requirements.txt` |
| 集合点数与 chunks.json 条数不一致 | 改了分块但没重建索引，服务在跑旧数据 | `pytest -m needs_qdrant` 会直接报出来；重跑 `python app.py index` |
| 报"集合不存在"却提示查 Docker | 旧版 `check_health.py` 把 404 吞成连接失败 | 已修：先 `collection_exists()` 判断，与 `RAGEngine.count()` 一致 |
| `chunk_index` 有空洞 / `total_chunks` 偏大 | 旧版 `build_chunks` 在过滤短块**前**编号 | 已修为先过滤再编号；`pytest -m needs_chunks` 守卫 |
