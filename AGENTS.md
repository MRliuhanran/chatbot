# 垂直领域 RAG 知识库（skill）

**触发** rag、索引、embedding、向量、qdrant、bm25、稀疏、分块、分句、rerank、检索、知识库、process、index、hybrid_search

## 形态
单文件 `chatbot.py`（约 3580 行，按依赖顺序分节），节顺序即流水线：配置 → 分块 → 索引 → 检索 → 生成 → 编排 → 入口（Streamlit/API）→ 离线工具 → 命令分发。入口命令：`process`、`index`、`serve`、`api`、`health`、`reindex-sparse`、`compare-chunks`、`lexicon`、`stages`；`streamlit run chatbot.py` 进界面。重型依赖一律惰性导入。
日志：`log()`/`logger` 同时写控制台与单一文件 `RAG_LOG_FILE`（默认 `rag.log`，`*.log` 已 gitignore）。

## 设备与最小重跑
设备自动选型（MPS 优先、初始化失败回退 CPU）：`INDEX_DEVICE`/`SEMANTIC_EMBED_DEVICE` 留空即自动；批量重建前先 `ollama stop` 释放显存避免 Metal 争抢；水印 1.0/0.9 是内存缓冲，不要改小。`api`/`serve` 启动后台预热（`RAG_PRELOAD`，默认开）：jieba 词典、嵌入/重排模型、生成模型常驻 Ollama —— 把首查询冷加载提前到启动期。`python chatbot.py stages` 给出四阶段指纹比对与最小重跑命令：A 分块（语料/切分变 → process）、B 稠密（换嵌入模型/新内容 → index，内容哈希缓存只算增量）、C 稀疏（词表/jieba/BM25/编码变 → reindex-sparse）、D 入库（集合缺失/点数不符 → index）。

**增量（增删改）**：`process` 在 `artifacts/docs/<doc>.json` 缓存**每个文档**的分块，内容 sha 与切分指纹都没变就整篇复用（不读文件、不跑句向量），文档删除会自动清理缓存 —— 所以加/改文档只重分块它自己；`index` 的稠密向量按 child_text 内容哈希复用（只嵌入新增/变化的块）。未变部分不重算、结果逐字一致（测试：不变重跑 chunks.json 与首次完全相同）。

**通用化设计**：领域名、数据目录、文档匹配、来源字段与展示、词表路径、集合/别名全部可配置；核心代码不假设语料是书、不假设来源字段叫 `book`、不写死文件名或集合名。关键开关：`RAG_DOMAIN_NAME`、`RAG_DOMAIN_DESCRIPTION`、`RAG_DATA_DIR`、`RAG_DOC_GLOB`、`RAG_SOURCE_FIELD`、`RAG_SOURCE_LABEL`、`RAG_HEALTH_PROBE`、`RAG_LEXICON_DIR`、`RAG_ALIASES_FILE`、`RAG_STOPWORDS_FILE`、`RAG_COLLECTION_BASE`、`RAG_COLLECTION_ALIAS`、`RAG_ARTIFACTS_DIR`、`RAG_PRELOAD`、`RAG_RERANK_BATCH`、`RAG_GEN_HISTORY_MAX_TOKENS`。

## 数据与契约
`RAG_DATA_DIR`（默认 `corpus/`）内匹配 `RAG_DOC_GLOB`（默认 `*.txt`）的文档 → `artifacts/chunks.json`（+.meta.json 携带指纹）→ Qdrant 集合，一律经别名（默认 `rag_current`）访问，先建新集合再原子切别名。来源元数据键名由 `RAG_SOURCE_FIELD`（默认 `source`）决定；旧产物的 `book` 键仍兼容读取（`_source_of`）。分块记录字段与 `_PAYLOAD_SPEC` 必须同步。
配置：默认值权威，同名环境变量可覆盖，非法值启动即报错；生效值用 `python chatbot.py health` 回读，它会点名"写进 .env 但代码从不读取"的变量。

## 领域部署
通用默认值面向任意垂直领域；部署时只改 `.env` 领域块（`RAG_DOMAIN_NAME` / `RAG_DOMAIN_DESCRIPTION` / `RAG_DATA_DIR` 等），核心代码与配置不含任何书名、文件名或领域词。`lexicon/aliases.txt`、`lexicon/stopwords.txt` 是按同一格式整表替换的领域词表。
换领域：改 `.env` 领域块（或同名环境变量）→ `python chatbot.py process` → `python chatbot.py index`；词表不同则先跑 `python chatbot.py lexicon` 自检。

## 链路
分块：归一 → 分句 → 语义定界 → 小块并入 → 父块打包(上限512) → 子块切分(128/重叠32)。
索引：校验指纹 → 向量化(内容哈希缓存) → 稠密+稀疏双写 → 建字段索引 → 切别名 → 删旧集合。
检索：空查询守护 → 稠密/稀疏双通道召回(查询侧与入库同精度，MPS 上 fp16) → 客户端加权RRF → 按命中 id 惰性拉 payload(不全量滚动) → 父块去重 → rerank(对全体候选统一打分，默认单批 32) → 分数概览(置信度，只统计不判定)。
生成：上下文装配(资料独立消息，纯 parent_text、无编号/来源标签/说明文字；本部署 system 亦为空) → 历史预算裁剪(从最旧一侧整条丢弃、最旧一条按预算截断，截断后首条历史恒为 user) → 流式。**无拒答**：检索为空也照常生成，代码不替模型做拒答决定。

## 不变量（违反即缺陷）
切分逐字无损；索引点数=产物记录数、稀疏须显式生成、缓存原子改名；空查询与纯标点必拦、零权重通道既不融合也不召回、去重后不补被折兄弟、重排对全体候选统一打分、payload 按命中 id 精确回取；资料不寄生 system、裁剪只丢整条消息且截断后首条历史恒为 user、资料与本轮提问永不丢、生成失败不入历史。

## 禁改
指纹区注释（改注释=改指纹=全量重建）；重排粒度与融合权重的默认值。指纹只覆盖切分侧（切分函数+分块参数+源文本+分句器版本），不含检索参数。
`read_book_text` 名字保留为 `_CHUNKING_CODE_UNITS` 成员（通用别名 `read_document_text`）；改名会改指纹。

## 变更代价
改切分/分块参数 → 重建分块与索引（CPU 约 73 分钟 / MPS 约 30 分钟）；改词表 → 重建稀疏（几分钟，稠密不受影响）；改模型 → 重建索引 + 重新度量；改检索参数 → 重新度量；改检索输出构造 → 需一次检索级复核；改领域配置（领域名/数据目录/来源字段）→ 重建分块与索引（分块指纹含源文本，来源字段名决定 payload 键名）。用 `python chatbot.py stages` 确认哪些阶段过期，只重跑过期阶段。

## 失效原型
错误不报错，只给看似正常的结果：**静默**（配置不生效、判据恒真、降级掩盖）、**多份**（同一知识多处实现）、**边界**（空/超长/混合形态）、**状态**（缓存与模型失步、别名切换窗口）、**自崩**（诊断路径先崩）。

## 边界与未决
不支持并发访问；未决：上下文压缩与多轮时延、平台化多数据域、跨域词表复用（无判据）、无拒答后"检索为空却照常作答"的质量如何度量。

## 维护现状
无自动守卫，回归只能靠评审与人工测量；agent 不做验证与测量，禁用"已验证"措辞。挂起项：写者唯一性未定、字段字面量/parent_text 去重两项未开工（按命中回 payload 已实现，`py_compile`+health 本会话跑过）、产物与代码指纹已对齐（见下）。

## 已知待办（2026-09 通用化后）
`artifacts/` 已于 2026-09-27 用修复后的代码整体重建（`ollama stop; python chatbot.py process && python chatbot.py index`）：切分指纹 `91b50c8672cb0ecb`、词表指纹 `379ba7983cb74394`，`chunks.json` 与 Qdrant 均 **22020** 条，`stages` 四阶段新鲜、`health --offline` 通过。修复指纹覆盖缺陷时把语义嵌入路径 `_encode_sentences/_embed_batch/_get_embed_model` 纳入切分指纹、把 BM25 模型名/标点集/解析归一源码纳入稀疏指纹、把 `_embed_batch/embed` 源码纳入稠密缓存键；旧 `artifacts/embeddings/d1938572e589_*.npz` 已删除。注意：强制全量重分块（指纹变）时，MPS 上语义阈值附近的浮点差异可能让边界极少数漂移（本次 22019→22020），未变指纹下按缓存复用仍是逐字一致。`bak/` 里是重构期备份（仍含旧域名词），确认无误后可整个删除。
