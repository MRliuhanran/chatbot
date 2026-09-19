# AGENTS.md —— 项目总文档（唯一一份）

> **本文件是全项目唯一的文档**。原先的 `PROJECT_DOC.md` / `ARCHITECTURE.md` /
> `DEFECT_REPORT.md` / `MULTITURN_PLAN.md` / `TODO.md` 与 `.opencode/skills/*/SKILL.md`
> 已合并进来。分成六七份时，同一件事（配置、命令、缺陷编号）会在多处各说一套，
> 而**互相矛盾的文档比没有文档更糟**。
>
> **硬预算：必须 ≤ 64KB**（65,536 字节）。超过这个尺寸，agent 的上下文只会加载到
> 前面一半，后半等于不存在。**往里加内容之前，先想清楚删掉什么。**
>
> | 你要做的事 | 看哪节 |
> |---|---|
> | 动手改代码之前 | 「工作方式」（硬约束）、「几个必须知道的坑」 |
> | 搞清楚代码怎么组织的 | 「项目结构」、「数据处理流程与检索链路」 |
> | 改配置 / 加开关 | 「核心配置」（全量清单：默认值、别名、非法取值行为） |
> | 跑 / 写测试 | 「测试与验证」、「效果指标与产物实测」 |
> | 查某个缺陷为什么这么写 | 「测试与验证 → 缺陷护栏 D1~D16」 |
> | 排障 / 日常操作 | 「操作与排障」 |
> | 下一步做什么 / 考古 | 「待办」、「设计存档 A/B/C」 |
>
> **代码里的注释才是实测结论的正式记录**（每个"为什么不能这样改"都写在对应位置）；
> 本文件是索引、约定与跨模块背景。两者冲突时以代码为准，并把本文件改回来。
>
> 文中 `见 4.x` / `见 7.x` 等是合并前的历史编号，一部分已失效 —— 按标题搜，不要按编号找。
## 1. 工作方式（对 agent 的硬要求，最高优先级）

**只开发和设计。不要主动自测 —— 主动性别这么强。**

- **不要自己跑测试**：`pytest`（任何 `-m` 分层）、`chatbot.py health`、
  `chatbot.py ab-retrieval`、`chatbot.py ab-multiturn`、`chatbot.py after-rebuild`、
  `chatbot.py compare-ab`、`chatbot.py compare-chunks`、`chatbot.py verify-qdrant`、`record_baseline`
  —— 一概不主动跑。
- **不要自己做端到端验证**：不主动起服务（`serve` / `api`）、不主动 `docker compose up`、
  不主动 `process` / `index` / `reindex-sparse`、不主动调 Ollama 或 Qdrant 发请求试探。
- **不要为"确认一下"写临时脚本**：包括一次性 `python -c` 探针、临时评测脚本、
  随手起的子进程。
- **测试、验证、测量一律由用户明确要求时才做**。用户没提就默认不做 ——
  哪怕"顺手跑一下就知道"，也不跑。
- 交付时**不许写"已验证 / 已实测 / 全绿"**，除非用户刚才明确让你跑过。
  结论若依赖实测，写成「未验证，需要你跑 X」，并把命令列出来交给用户。
- 需要新测试用例时**可以设计与编写**（写进 `tests/`），但**不要执行**；
  跑不跑由用户决定。
- 唯一例外：**用户本轮明确要求测试/验证**。此时才跑，且只跑被要求的那一层。
## 2. 项目概述

这是一个**四大名著 RAG 知识库**，Qdrant Docker 部署，稠密+稀疏混合检索 →
**客户端加权 RRF** → 按父块去重 → 重排 → Ollama 生成（生成侧已真正接入检索结果）。

两条与旧版不同的主线，改代码前必须先知道：

- **融合在客户端**（`rag_engine.weighted_rrf`），不再用 Qdrant 服务端 `FusionQuery`。
  权重与 k 都可配，且融合是可单测的纯函数。
- **分块按「回」分段**（`chapter_parse.parse_chapters`），每块自带
  `chapter_index/label/title` 与 `parent_id`；分块产物另有指纹元信息，
  `build_index` 开工前会强制校验它与当前配置一致。
## 3. 快速开始

```bash
docker compose up -d          # 1. 起 Qdrant
python chatbot.py process     # 2. 分块
python chatbot.py index       # 3. 向量化入库
python chatbot.py health      # 4. 健康检查
python chatbot.py serve       # 5. 起 UI :8501（或 api → :8000）
pytest -m unit                # 测试由**用户**决定何时跑，agent 不主动跑
```
## 4. 项目结构

**全部生产代码在一个文件里**：`chatbot.py`（约 8060 行，按依赖顺序分 9 节）。

```
├── chatbot.py                  # 唯一的生产代码（引擎 + 四个入口 + 全部工具）
│     ① 基础设施      .env 加载 / 环境变量解析 / 日志
│     ② Ollama 客户端 模型名、think、预算、超时、流式解析（**单源**）
│     ③ 章回体切分    按行首回目切回，纯标准库、逐字无损
│     ④ 查询改写      多轮指代消解，三档降级链
│     ⑤ RAG 引擎      分块/嵌入/索引/检索（检索逻辑唯一权威实现）
│     ⑥ 一轮对话      检索→装配→拒答→生成 的事件流
│     ⑦ 四个入口      Streamlit UI / HTTP API / 极简 UI / 通用机器人
│     ⑧ 工具          健康检查 / 通道校验 / A-B / 词表 / 重建
│     ⑨ 命令行分发
├── tests/                  # 自动化测试 L0~L3 + 探针 + 评测执行器
│   ├── probes.py               # 探针 + 判定函数的**唯一来源**
│   ├── eval_runner.py          # "跑探针→聚合"的唯一实现
│   ├── conftest.py / record_baseline.py
│   └── test_*.py               # 见下"测试分层"
├── data/
│   ├── aliases.txt             # 人物别名表（规范名 + 别名，制表符分隔）
│   └── stopwords_classical.txt # 古白话停用词表（禁收否定词/程度词）
├── pytest.ini              # 测试分层 marker 配置
├── docker-compose.yml / qdrant/config.yaml   # Qdrant 编排与配置
├── books/                  # 源文本（四大名著 .txt）
├── models/                 # bge-base-zh-v1.5（768 维）/ bge-reranker-base
├── qdrant_storage/         # Qdrant Docker 数据卷
├── cache_v2/               # 分块产物 + 向量缓存
└── .env / requirements*.txt

### 命令行（唯一的入口）

```bash
python chatbot.py process          # 分块（读 books/ → cache_v2/chunks.json）
python chatbot.py index            # 向量化入库（写新集合 + 原子切别名）
python chatbot.py serve            # Streamlit UI（:8501）
python chatbot.py api              # HTTP API（:8000，NDJSON 流）
python chatbot.py health           # 系统健康检查
python chatbot.py verify-qdrant    # 稠密/稀疏通道校验
python chatbot.py reindex-sparse   # 只重建稀疏向量（换词表时用）
python chatbot.py compare-chunks A B
python chatbot.py compare-ab A B
python chatbot.py lexicon          # 别名表 / 停用词表校验
python chatbot.py ab-retrieval --label x
python chatbot.py ab-multiturn --label x
python chatbot.py after-rebuild

streamlit run chatbot.py           # UI；RAG_UI=chat 或 RAG_UI=bot 切到另两个界面
```

每个子命令**保留自己的 argparse**（含各自的 `--help` 与退出码），分发器只把
`argv[0]` 摘掉再交给它 —— 因此 `python chatbot.py health --help` 与合并前
`python check_health.py --help` 的行为一致。

### 为什么是"分节的单文件"而不是多模块

合并前有 19 个源文件（15 个在根目录、4 个在 `tools/`）。合并**省的代码其实很少**
（模块 docstring / import / `__main__` 包装共约 4.4%，另加 632 处跨模块前缀 ~4.1%），
真正的收益是结构性的：

* **"只有一份实现"成为结构事实** —— 参考 D9/D12（两份 `.env` 引导）、D13（两份流式
  实现与两处配置）、D14/D15（三份入口序列、五份 schema），成因全是"多个模块各自
  做同一件事"。单文件里这些**不可能发生**，不再依赖人的自觉。
* 没有跨模块 import 顺序问题（`.env` 必须在任何常量求值之前加载）。

代价（已落在代码注释里）：两个 UI 靠 `RAG_UI` 切换；重型依赖（torch /
transformers / jieba / qdrant_client / streamlit）必须留在函数内惰性 import，
否则 `pytest -m unit` 就不再是"零依赖可跑"。
## 5. 核心配置

**检索侧常量全部可用同名环境变量覆盖**，代码里的默认值仍是唯一权威
（不设环境变量时行为不变）。取值非法**启动即报错**，不静默回退默认值。

```python
# Qdrant
QDRANT_HOST / QDRANT_PORT = localhost / 6333   # 也接受 RAG_QDRANT_HOST/PORT 别名
COLLECTION_NAME   = "books_v3"                 # 逻辑基名
COLLECTION_ALIAS  = "books_current"            # 检索端一律走别名
RAG_USE_COLLECTION_ALIAS = 1                   # 建索引写新集合 → 原子切别名

# 分块
RAG_CHILD_MAX_TOKENS  = 128    # 子块：精确匹配
RAG_PARENT_MAX_TOKENS = 512    # 父块：完整上下文
RAG_CHUNK_OVERLAP     = 32     # 子块重叠 token
RAG_PARENT_MIN_TOKENS = 192    # 父块下限：低于此并入相邻块
RAG_CHILD_MIN_TOKENS  = 32     # 子块下限
RAG_SEMANTIC_THRESHOLD = 0.6   # 相邻句余弦相似度低于此值视为语义边界
RAG_SEMANTIC_MIN_CHARS = 500   # 切分下限：未达此长度前不允许切分

# 检索
RAG_TOP_K          = 5         # 最终返回给 LLM 的结果数
RAG_RERANK_TOP_K   = 20        # reranker 候选数 / 单通道召回数
RAG_RERANK_BATCH   = 16        # reranker 单批条数
RAG_RERANK_MAX_LENGTH = 384    # 仅在 RAG_RERANK_ON=parent 时真正起作用
RAG_RERANK_ON      = "child"   # child=判子块（现状） / parent=判父块（已实测：方向更好
                               #   但 n=24 样本不足，故不改默认值，见下"必须知道的坑"）
RAG_RRF_K          = 60
RAG_RRF_DENSE_WEIGHT / RAG_RRF_SPARSE_WEIGHT = 1.0 / 1.0
RAG_RECALL_LIMIT   = 0         # 0 = 跟随 RERANK_TOP_K
RAG_DEDUP_BY_PARENT = 1        # 按父块去重（同源兄弟子块只留名次最好的）

# 拒答（相对口径，实测校准，不是拍脑袋阈值）
RAG_ABSTAIN            = 1
RAG_ABSTAIN_MEAN_HARD  = -1.5  # 平均分低于此值 → 拒答
RAG_ABSTAIN_MEAN_LOW   = 0.5   # 配合"跨 >= 2 本书"才拒答
RAG_ABSTAIN_MIN_BOOKS  = 2

# 生成侧 / 集成（四个入口读）
RAG_USE_CONTEXT    = 1         # 1=注入检索结果  0=对照模式（system 与资料消息都不发）
RAG_CONTEXT_ROLE   = "user"    # 承载检索正文的那条独立消息的 role：user/tool/system。
                               #   **2026-09-19 起正文不再拼进 system**：旧写法把正文塞在
                               #   CONTEXT_SYSTEM_PROMPT 的 {context} 里，system 一旦被置空/
                               #   被服务端截断，RAG 就静默退化成"凭记忆作答"（2026-09-18
                               #   真发生过）。现在 system 只放规则（SYSTEM_RULES_PROMPT），
                               #   正文由 build_context_message 单独发：system → 历史 →
                               #   资料消息 → 本轮提问。"资料到没到"成为可直接断言的事实。
                               #   取值非法启动即报错；A/B 时一个进程一个状态
RAG_QUERY_REWRITE_CONCAT = 1   # 降级链中间档：LLM 改写不可用时拼"上一轮用户问句+本轮"
                               #   （实测 topic@k 42.9%→95.2%，零延迟零失败面）
RAG_QUERY_REWRITE_CONCAT_CHARS = 60  # 拼接时上轮问句的截断长度
RAG_QUERY_FUSION = 0           # 多路召回：LLM 改写成功的轮次**额外**用拼接档
                               #   各召回一次，两路一起进 RRF。LLM 失败时自动退化为
                               #   拼接单路，不新增失败面。
                               #   ⚠️ 保持关闭：修正 topic@k 的别名口径后实测
                               #   3 改善 / 1 退化（kw@k 70%→80%，但一条 entity_switch
                               #   的 topic@k 掉了）—— 撑不起改默认值。旧记录里的
                               #   "改善 3 / 退化 0"是度量假象
RAG_RERANK_QUERY = "primary"   # 多路时 reranker 拿哪句打分：primary / join / per_route。
                               #   新口径下 primary 与 per_route 持平，故保持默认；
                               #   两种口径的数字差异见「效果指标 (d)」
RAG_QUERY_REWRITE              # 多轮指代消解的**开关**，取值不在这里写死：
                               #   以 .env 为唯一权威（原因见"坑"第 4 条；本行曾写 1 而
                               #   .env 是 0，同一类文档漂移）。**2026-09-19 起 .env 是 1**，
                               #   与"恢复 _SYSTEM_PROMPT 四行规则"配套：实测"prompt 置空 +
                               #   开关打开"是最差的一档（topic@k 66.7%），两者不可分开动。
                               #   查当前生效值：health 或 GET /health 的 query_rewrite。
RAG_THINK / RAG_NUM_CTX / RAG_NUM_PREDICT
# 生成侧装配（三个入口共用 plan_generation / build_generation_messages）
RAG_HISTORY_MAX_TOKENS = 1500  # 兜底历史预算；**正常不用它** —— 调用方一律用
                               #   现用 history_budget(num_ctx, num_predict, system) 算：实测
                               #   system 占 1620~2590 token、num_predict 再扣 4096，留给历史的
                               #   只有 1200~2200 token（4~12 轮）。超窗时服务端不报错、保留
                               #   system，但会整条丢消息（16000 字的消息把输入塌到 40 token）
RAG_INPUT_RESERVE = 256        # 从窗口里额外扣掉的余量（模板开销 + 估算误差）

# 词表
RAG_ALIASES / RAG_STOPWORDS = 1 / 1
RAG_LEXICON_DIR = "./data"

# 产物路径（多数据集前置改造，纯搬家；设计见「设计存档 A」）
RAG_ARTIFACTS_DIR = "./cache_v2"  # chunks.json / chunks.meta.json / embeddings/ 都由它派生。此前三个
                                  #   路径各自硬编码，把 CHUNKS_JSON 指向临时目录的调用方
                                  #   （测试、A/B）会把缓存与元信息写回真实目录

# 生成侧超时（**只有一处读**，四个入口都从它取，不可能分叉）
OLLAMA_TIMEOUT = 180  # 秒。冷启动首字节实测会超过 180s；调大它对**四个入口**都生效
RAG_API_MAX_BODY = 65536  # /search、/ask 的请求体上限（字节）
RAG_LOG_LEVEL = "INFO"    # bootstrap.get_logger 读，控制日志级别

# 设备
SEMANTIC_EMBED_DEVICE = "cpu"   # 分块阶段句子编码（默认 cpu：MPS 会因 GPU 争用阻塞）
                                #   ⚠️ 吞吐随机器负载剧烈波动，先量再估时间（见下）
INDEX_DEVICE = ""               # 建索引阶段，空=自动优先 MPS；设 cpu 可强制 CPU
                                #   （8GB 机器上 MPS 常因统一内存被挤压而初始化失败，
                                #    build_index 已内置回退 CPU，见其注释）
```
## 6. 几个必须知道的坑

1. **分块产物有指纹门禁**：`build_index` 开工前会校验
   `cache_v2/chunks.meta.json` 与当前配置/源文本一致，不一致直接报错要求重跑
   `process`。用旧分块建索引**不会报错、只会让检索悄悄变差**，所以这里刻意硬失败。
2. **集合名要解析，不要硬编码**：启用别名后具体集合名是
   `books_v3__<build_id前8位>`，`COLLECTION_NAME` 只是逻辑基名。工具脚本请用
   `RE.default_collection_name(client)`。
3. **吞吐随负载剧烈波动，且休眠是真正的瓶颈**：`.env` 里那个 43 句/秒是**空载**数字；
   高负载机器上实测只有 **8.9 / 10 句/秒**，线程数（8/6/4）无显著差异
   （10.1 / 10.0 / 11.5 句/秒）。更关键的是**机器休眠会伪装成"极慢"** ——
   实测某次构建墙钟 8 小时而 CPU 时间只有 48 分钟。长任务请用
   `caffeinate -i -s -w <python进程PID>` 防休眠。**先量当前吞吐再估时间**，
   不要拿历史总时长当承诺。
4. **别把"文档里写了的环境变量"当已生效**：本仓库踩过**三次**——`RAG_QDRANT_HOST`
   （名字写错）、`RAG_ABSTAIN_RATIO`、以及 `RAG_QUERY_REWRITE`（D9：**import 顺序**
   决定 .env 生不生效，`import rag_engine` 在前就整个失效）。所以：①改配置前先 grep
   代码确认它真被读取；②确认当前生效值用 `python chatbot.py health`（它现在会把
   多轮改写开关的实际状态打出来），而不是读文档。
5. **`RAG_RERANK_ON` 默认保持 `child`、融合权重保持等权 —— 都别动**：`parent` 三项
   判别性指标同向更好且零回退，但名次级胜负只有**改善 2 / 退化 0 / 持平 22**，差异由
   2 条探针驱动，**n=24 不足以改默认值**；融合权重等权 1:1 已是最优（等比缩放不改变
   RRF 排序，`sparse=0.5`/`dense=2.0`/`sparse=2.0` 都验证了这个）。数字见「数据处理
   流程」与「效果指标」；A/B 用 `python chatbot.py ab-retrieval`，重建后全套复核用
   `python chatbot.py after-rebuild`。
6. **别把"给模型的上下文"寄生在 system 上**：正文必须走**独立消息**
   （`build_context_message`，role 见 `RAG_CONTEXT_ROLE`），system 只放规则。旧写法把
   正文拼进 `CONTEXT_SYSTEM_PROMPT` 的 `{context}`，于是 system 一旦被置空或被截断，
   RAG 就**静默**退化成"凭记忆作答"，回答照样通顺（2026-09-18 真发生过）。判断
   "这轮到底是不是 RAG"：看 messages 里有没有那条资料消息 / 推理链的「上下文装配 →
   messages」，**不要**靠"system 里有没有某个子串"推断。
## 7. 测试与验证

按"需要多少外部资源"分层，用 pytest marker 控制。资源缺失时**跳过**而非失败，
避免"本机没起 Docker"被误读成"代码坏了"。

```bash
pytest -m unit          # L0 纯函数：无需模型/Qdrant/语料，~22s / 304 条（agent 不主动跑）
pytest -m needs_chunks  # L1 分块不变量：需 cache_v2/chunks.json（16 条）
pytest -m needs_qdrant  # L2 索引一致性：需 Qdrant 在跑（14 条）
pytest -m needs_models  # 需真 tokenizer（硬切无损 4 + build_chunks 5 = 9 条；
                        #   L3 的 12 条也带此 marker，但被 slow 默认过滤）
pytest                  # 以上四层 = 366 条，全绿
pytest -m slow          # L3 检索质量回归：需模型+索引，分钟级（12 条）；全套件 = 378 条
```

| 层 | 文件 | 守卫的不变量 |
|----|------|-------------|
| L0 | `test_chunking.py` | 分句不跨段、软换行不算段落、引号归一两种语料型、超长句兜底不超限/不自递归、子块不超 128 token、父子包含 |
| L0 | `test_chapters.py` | 中文数字解析（含"第一百十回"）、逐字无损拼接、回序号连续、offset→回 二分查找、四本书真实回数 23/64/120/100 |
| L0 | `test_probes.py` | 探针结构/接地：关键词真在对应书里、负样本主题真不在语料里、rel_gap 口径、多轮指代 kind 合法且每类样本够 |
| L0 | `test_retrieval_units.py` | `weighted_rrf` / `dedup_candidates_by_parent` / `confidence_signal` / 空查询守护 / 别名归一 / 平衡打包 / **上下文装配（system 只放规则、正文走独立消息、资料永不因裁剪而丢）** / 查询改写 / **生成侧历史预算** / 环境变量解析 |
| L0 | `test_defects_regression.py` | D1~D16 护栏（见下）+ 引号归一语料级验收、`hybrid_search` 调用点 AST 静态护栏、`.env` 到达性（子进程）、生成侧配置/流式**单源**（对象同一性 + AST 禁读）、`plan_generation` 三入口同口径 |
| L0 | `test_api.py` | `_ENGINE_LOCK` 必须**可重入**（曾因非重入锁让 `/health` 自锁挂死 —— curl 只报超时、日志一个字都没有）、端点真能回话 |
| — | `test_hard_split.py` | 硬切兜底逐字无损且仍守 token 上限（需真 tokenizer） |
| — | `test_build_chunks.py` | 真实 `build_chunks`：空白块被丢、短正文块被保留、编号无空洞 |
| L1 | `test_chunk_invariants.py` | 字段完整、`parent_text` 含 `child_text`、`chunk_index` 连续、`total_chunks` 正确、无逐字空格损伤、**真实 tokenizer** 下 128/512 上限 |
| L2 | `test_index_consistency.py` | 点数 == chunks.json、point id 连续、稠密/稀疏双写、向量归一化、build_id 唯一、稀疏单通道可召回 |
| L3 | `test_retrieval_quality.py` | 正向 24 条 book@1/book@k/kw@k 不低于基线（**零容差**）；另分三组：冻结 12 条短探针单独回归（唯一能与历史基线按位置对拍的一组）、多轮 `topic@k` 回归（按指代类型分组打印）、负样本只记录不设断言 |

L3 基线需先录制：`python -m tests.record_baseline` → `tests/golden/retrieval_baseline.json`。

**探针唯一来源**是 `tests/probes.py`：12 短 + 12 长问句（正向）、10 负样本、
**30 多轮**（每类 5 条，`test_probes.MIN_PER_KIND=5`；六类指代 pronoun / assistant_only /
entity_switch / ellipsis / temporal / recall_detail，见 `VALID_MULTITURN_KINDS`；
每类失手原因与修法不同，故 `aggregate_multiturn_by_kind` 分组报比率）。
`chatbot.py compare-ab` / `verify-qdrant` / L3 都从这里导入。三类指标**分开报告、
不合成总分**（`tests/probes.py::aggregate_all`）——"短查询 100%、多轮 0%" 合成
一个 50 分就看不出问题在哪。
### 缺陷护栏（`tests/test_defects_regression.py`）

每条对应一个**已复现**的功能缺陷，断言的是正确行为。缺陷未修时用
`xfail(strict=True)` 保持套件绿色；修好后 strict xfail 会立刻变成 XPASS 失败，
强制摘掉标记。**D1~D16 现已全部修复，文件里已无 xfail 标记**。

| # | 缺陷 | 修法 |
|---|------|------|
| D1 | 句内软换行被判成"分句器跨段"，`process` 整库崩溃 | 段落判据改为**空行** `\n\n`（`_PARAGRAPH_GAP_RE`），句内软换行归一成空格（`_SOFT_WRAP_RE`） |
| D2 | `fix_quotes` 的混合型判定"全有全无"，一个 `“` 就改坏整本 | 改为「硬信号 + 状态机」：只有**成功配对**的弯引号参与状态（`_paired_curly_positions`）。⚠️ **该修法本身有缺陷，已被 D10 取代**——见下 |
| D3 | `len(child_text) > 5` 过滤丢掉真正的原文 | 判据改为"有无正文"（`c.strip()`）；已重跑 process 生效 |
| D4 | `.env` 写 `RAG_QDRANT_HOST`，代码只读 `QDRANT_HOST`，配置静默无效 | 两个名字都接受：`QDRANT_HOST or RAG_QDRANT_HOST` |
| D5 | 空查询 / 纯标点返回 5 条"看似正常"的结果 | `_has_searchable_content()` 空查询守护，直接返回空结果 |
| D6 | `requirements.txt` 描述了一个不存在的降级函数 | 注释改为"硬依赖、无降级路径"，并说明删掉 sentencex 是 ImportError 而非降级 |
| D7 | `cmd_serve` 把"连不上 Qdrant"误报成"向量库为空" | 连接失败与空集合分开报，并给出 `docker compose up -d` 提示 |
| D8 | 生成失败被当成"模型答案"写进会话历史并回灌模型 | `_stream_ollama` 新增独立的 `"error"` kind，UI/API 分开记录，不写入助手历史 |
| D9 | .env 里的 `RAG_QUERY_REWRITE=0` 是否生效**取决于 import 顺序**：`query_rewrite` 在 import 时读常量，而 `rag_engine` 顶层就 import 它。UI/API 恰好对（先 `load_dotenv`），而健康检查把"已关闭"报成"开启"，`record_baseline` / `ab_retrieval` / `compare_ab` / `verify_qdrant` 里 .env 的 0 一律失效（"关掉改写做 A/B"实际没关） | `query_rewrite` 自己 `load_dotenv()`（`override=False`，命令行环境变量仍优先，A/B 用法不受影响），与 app.py / ollama_client.py 一致；D9 用**子进程**断言两种 import 顺序得到同一结论 |
| D10 | **`fix_quotes` 把闭引号改成了开引号**：D2 引入的"硬信号"假定"直引号左侧紧跟句读 ⇒ 必为开引号"，而该假定在本语料上是**反的**（红楼梦 `："` 0 次、`？"` 915 次、`！"` 402 次，直引号一律是闭引号）。后果：2426 个直引号里 2094 个被判成开引号，未配对 `“` 从 1600 涨到 **3362**，坏文本已烤进 `cache_v2/chunks.json`（可直接搜到 `意欲何往？“那僧笑道`） | 重写为**单遍二值状态机**：`“ ”` 与已判定的 `"` 共同维护"是否在引号内"，在内则闭、在外则开；删掉 `_OPEN_QUOTE_CTX` 与 `_paired_curly_positions`。复算：未配对 1600 → **12**。⚠️ 修好后**必须重跑 `process` + `index` 并重录基线**（旧基线建立在被改坏的正文上） |
| D11 | `chat.py` 把 `hybrid_search` 的**列表**返回值按二元组解包（`results, _ = …`，而 `return_steps` 默认 False）→ 提问即 `ValueError`；条数恰为 2 时更糟（`results` 变成 dict，后续对它调 `.get`） | 改为单值赋值；并加 **AST 静态护栏**：凡按元组解包该调用者，必须显式带 `return_steps=True` |
| D12 | `rag_engine` 的常量在 import 时求值，它自己却**不** `load_dotenv()`；此前靠"import 了 self-load 的 query_rewrite 且那次 import 恰好在配置段之前"侥幸成立，挪一下 import 就全仓库 `.env` 失效 | `.env` 加载收进单源（`override=False`）；子进程断言 .env 的显式非默认值真的到达模块 **且** 命令行环境变量仍然优先 |
| D13 | **生成侧有两份同逻辑实现**：`app._stream_ollama(payload)` 与 `ollama_client.stream_chat(messages)`，且 MODEL / THINK / NUM_CTX / NUM_PREDICT / OLLAMA_TIMEOUT 在两边各读一遍 —— 超时一处硬编码 180s 而另一处可配（用户调大只对一个入口生效），RAG_USE_CONTEXT 只在 app.py 读而别的入口靠转发（chat.py 漏过） | 删掉重复的 `_stream_ollama`，四个入口统一用同一个 `stream_chat`；`RAG_USE_CONTEXT` 收进引擎侧。护栏：`chatbot.py` 里 `def stream_chat(` **只能出现一次**，MODEL/THINK/NUM_CTX/NUM_PREDICT/OLLAMA_TIMEOUT 各只能赋值一次 |
| D14 | `chat.py` 漏传 `use_context`，`RAG_USE_CONTEXT=0` 在它那里静默失效（app/api 都传了） | 抽 `rag_engine.plan_generation()`：三个入口共用同一个装配函数（use_context / low_evidence / 硬拒答 / 历史预算全在里面），入口只剩渲染 |
| D15 | **同一份知识写在多处 → 必然分叉**（结构类，不是行为类）。实测三例：`compare_ab` 抄了一份召回管线（抄漏 `with_payload` 而崩）、`verify_qdrant` 抄了一份抽样检查（缺守卫、缺去重）、`chat.py` 抄了一份装配（漏 `use_context`）。同类还有：分块 schema 被声明 5 遍、`steps` 键名散落 8 个文件、`check_health` 与 `verify_qdrant` 各写一份抽样检查、`ab_retrieval` 与 `eval_runner` 各写一份跑探针的循环 | 全部收敛到单源：schema → `_PAYLOAD_SPEC`（写读两端派生）；steps 键 → `STEP_*` 常量且**写权收回 rag_engine**（`plan_generation` 产出上下文装配/历史裁剪，入口只 `steps.update(plan["trace"])`）；抽样检查 → `sample_vector_coverage`；跑探针 → `eval_runner.run_probes_with`；一轮序列 → `run_turn` 事件流，三个 UI 只渲染事件。护栏：`TestContractsHaveSingleSource` 用 AST 断言"只允许一个写者/一个实现" |
| D16 | **19 个源文件 → 单个 `chatbot.py`**（结构合并）。合并本身只省约 8.5% 的行，拿掉的是**模块边界**这个「防止同一件事写两处」的机制 —— 而 D9/D12/D13/D14/D15 的成因全是「多个模块各自做同一件事」。在单文件里这些成因结构上不存在 | 按依赖顺序分 9 节，每节保留原文件 docstring 作章节说明；重型依赖留在函数内惰性 import，保住 `pytest -m unit` 的零依赖可跑；两个 UI 用 `RAG_UI` 切换。合并工具做过**逐行对拍**（除 import 上移/改名/去前缀外一个字未动） |

### 其他已修缺陷（不在 D 表，同样是"会静默出错"的那类）

以下每条的成因都一样：**错误不报错，只是悄悄给出错误结果**。修法已落在代码注释里。

| 位置 | 缺陷 → 修法 |
|---|---|
| API `/ask` | 响应头未发时抛异常却仍裸写 `wfile` 的 NDJSON → 客户端收到**没有状态行/header 的"响应"**（同一故障在 `/search` 是干净的 500 JSON）。修：`headers_sent` 分流 —— 未发头 → 500 JSON，已发头 → error 事件 + 关连接 |
| API | 未读完请求体就拒绝（404 / 超限）→ HTTP/1.1 keep-alive **失步**，客户端下一个合法请求收到 400。修：拒绝前 `_drain_body()` 有界排空；所有错误响应带 `Connection: close` |
| API | `_REQUEST_LOCK` 覆盖**整段流式生成**（think 打开时实测 **~135s**），且 Handler 无 socket 超时 → 一个慢/不读的客户端锁死全部请求。修：锁只包检索；`Handler.timeout = 60` |
| API | `history` 只校验是 list 不校验元素 → `content=123` 让 `strip_current_turn` 抛 AttributeError，返回 500 而不是 400。修：逐项校验 role/content；`top_k` 显式拒掉 `bool` |
| API | 不剥"末尾即本轮提问"，与两个 UI 入口（都传 `[:-1]`）行为不一致。修：调用 `strip_current_turn`，响应回显 `history_stripped` |
| `hybrid_search` | 去重后用 `folded`（**同父**兄弟子块）回填条数 → 把同一段 `parent_text` 又塞回上下文，与去重的唯一目的直接矛盾。修：**删除回填**，如实少返回，`steps["检索汇总条数"]` 上报缺口 |
| `build_generation_messages` | `_truncate_turn` 从不截用户消息 → "16000 字用户消息"场景仍会超窗，服务端静默整条丢。修：用户消息单独超预算时也截尾（标记 `_USER_TRUNCATION_MARK`），由 `truncated_user` 上报，UI/API 必须显示 |
| `query_rewrite` ×4 | ①`rewrite_prompt_stats` 在 `try` 之外无条件调用，非 dict 历史元素让整条检索链崩；②模型前言（`好的，我来改写：`）被当成检索问句且 `applied=True`；③`build_retrieval_routes` 追加拼接路时漏判 `concat_enabled` → `CONCAT=0` 时仍多召回一路而 health 报"只剩 LLM→字面"；④`CONCAT_CHARS<=0` 被解释成"截成空串"。修：跳过非 dict + 整体兜异常退化为字面档；逐行扫描跳过元话语行；与降级链同口径判断 |
| `chapter_parse` ×3 | ①回目正则要求"回"后紧跟空白 → "第一回"单独成行不匹配，**漏末回被并进上一回 body 且不报错**（修：前瞻 `(?=[^\S\n]\|$)` + 新增"漏尾审计"，对四本书零误报）；②注释声称兼容全角阿拉伯数字但字符类只有 `0-9`（修：补 `\uff10-\uff19`）；③"万"被纳入"相邻单位递减"校验 → `一万` ✔ 而 `十万`/`二十万` ✘（修：`unit == 10000` 的节结算提到递减校验之前） |
| `compare-ab` | `with_payload=False` 却读 `p.payload`（恒为 None）→ 工具**第一个探针即崩**。修：`with_payload=True`，字典构造统一走 `chunk_from_payload` |
| `compare-chunks` | 字段表漏 `parent_id / chapter_*`，未登记字段被静默忽略 → 对"零影响"改动给**假绿**。修：补全字段表 + **模式漂移自检**（产物出现未登记字段直接报错退出） |
| `verify-qdrant` ×2 | ①混合路两通道**拼接未去重** → 命中数可超过分母上限（能打印出 `4/2`），两侧口径不对等（修：按 id 去重，并把"命中数 > 上限"变成断言 —— 工具先能发现自己坏了）；②诊断代码在它要诊断的故障下自己崩（首条缺稀疏 → KeyError；抽样为空 → IndexError）（修：`.get()` + 显式报告，稠密/稀疏对称） |
| `reindex-sparse` | 空集合时两处一致性检查恒真（`0==0`）→ 打印"完成"并 `return 0`。修：`total == 0` 直接失败退出；两遍写合并成一趟，避免留下"稀疏已换、指纹未换" |
| `health` | 外层 except 把所有异常报成"无法连接 Qdrant Docker服务" → 本地 `chunks.json` 损坏被**误诊成 Docker 没起**。修：本地文件读取自带 try/except，报出真实原因 |
| UI（原 `app.py`）×3 | ①`THINK=0` 时 `done_reason=length`（答案被截断）的告警整块不执行 → 截断判定移出 `if think_status is not None`；②端口 8501 被别的进程占用时误报"Streamlit 已在运行"并 `exit 0` → 端口占用但 `_streamlit_pids()` 为空则报"端口被占用"并 `exit 1`；③每轮把整份 trace（同批正文存两份，**20~40KB/轮**）写进 `session_state` → `_slim_trace()` 只留计数/名次/id，但**保留 `上下文装配 → messages`**（"资料进没进"的唯一证据） |
| 机器人入口（原 `bot.py`） | 四个入口里唯一不做历史预算/空回合过滤的 → 长会话超窗后服务端静默丢消息。修：复用 `history_to_messages` + `build_generation_messages` + `history_budget`；补 None 保护 |
## 8. 数据处理流程与检索链路

> 下文的 `rag_engine` / `query_rewrite` / `chapter_parse` / `app.py` / `api.py` /
> `chat.py` / `bot.py` 是**合并前**的模块名 —— 现在都在 `chatbot.py` 的对应分节里
> （见「项目结构」的 9 节划分）。函数名限定的引用（如 `rag_engine.foo()`）保留原样。

### 8.1 分块（`build_chunks`）

```
books/*.txt → 引号归一 → 按回分段(parse_chapters) → 逐回：sentencex 分句 → 语义定界(0.6/500 字) → 小块并入相邻 → 平衡打包父块(512)
 → 父块内包子块(128, 重叠 32) → 超长句兜底：标点→换行→分号→逗号→硬切
```

- 回数：水浒 **23**/红楼 **64**/三国 **120**/西游 **100**（`pytest -m unit` 断言；非 bug：水浒节本、红楼 64 回残抄本）。
- 水浒「楔子」5743 字符落在 preface；`preface + "".join(ch.body) == text`。
- 块不跨回、带 chapter 元数据 → `book`/`chapter` 过滤与溯源；旧法搜回目 **1.4%**，按偏移 **100%**；旧 `len(child_text) > 5` 丢 14 字（`宝玉又道：`/`莫念！`），现只丢纯空白块。
- 旧父块 <128 token **1786** 条（<64 的 **1188**、最小 **6 token**；**1293** 条前一块 ≥480）；「武松打虎」首位父块仅 **45 字**。修法：低于 `RAG_PARENT_MIN_TOKENS=192`/`RAG_CHILD_MIN_TOKENS=32` 并入相邻块；合并从不超 512。
- `contextual_text` ≡ `child_text`（**77.6%** 的块除《书名》外无信息量）；`parent_id` 一父一 id：**21137** 子块 → **5701** 父块。

### 8.2 分块产物与指纹门禁

`build_index` 开工前强制校验 `cache_v2/chunks.json` + `chunks.meta.json`（指纹、源文本哈希、回清单、父块数）；指纹只覆盖分句/分块参数、源文本、sentencex 版本，检索侧参数不入指纹；必须硬失败：旧分块建的索引不报错只让质量变差。元信息路径**由 `CHUNKS_JSON` 派生**（`_chunks_meta_path()`）而非独立常量，否则把 CHUNKS_JSON 指向临时目录的调用方会把元信息写回真实 `cache_v2/`。同理**不提供"书没变就跳过分块"的增量开关**：chunks.json 还依赖分句器与分块参数（换一次 sentencex，书一个字节没变而分块全变）。

### 8.3 向量索引（`build_index`）

```
chunks.json → Embedding(内容哈希缓存) → Qdrant 新集合(COSINE+BM25) → payload 索引 → 原子切别名 → 删旧集合
```

- `verify_chunks_freshness()` 开工前校验。缓存 `cache_v2/embeddings/`：按 `(模型, 数值精度)` 分桶，**键=文本内容哈希**（`RAG_EMBED_CACHE`，约 65MB），先写临时文件再原子改名（中途被杀不留截断的 `.npz`）；`build_id` 写进每个 point，检索端据此自动丢进程内旧缓存。
- 写新集合 `books_v3__<build_id前8位>`，全部写完再把别名 `books_current` 切过去（Delete + Create 放在**同一次** `update_collection_aliases` 调用里，服务端按序应用，中间没有"别名指向空集合"的窗口）；旧集合在别名生效**之后**才删。旧做法 delete+create 在重建的长时间窗口里服务完全不可用，且中途失败就无从回滚。别名不可用时回退全量重建，并**明确说出来**（静默退化会让人以为原子切换在生效）。
- payload 索引 `book`(KEYWORD)/`chapter_index`(INTEGER)/`parent_id`(KEYWORD)。实测旧集合的 `payload_schema` 是空的（`{}`），任何按书名过滤都退化为全表扫描；建索引失败只告警不中断。

### 8.4 检索流程（`hybrid_search`）

```
query ─(有 history)→ 查询改写 → 空查询守护 → 稠密 ‖ 稀疏(jieba+BM25) → 两次单通道召回(各 top-20，权威) → 客户端加权 RRF(k=60)
 → 按父块去重(不足 top_k 回填) → Rerank → 置信度信号
```

- 融合挪客户端，**查询次数 3 → 2**（可单测、可 A/B）。
- `weighted_rrf` 确定性：并列用首现顺序兜底；权重 0 = 该通道整体退出融合。
- `RAG_DEDUP_BY_PARENT` 默认开：自然语言问句约 **1/3** 命中，白占约 **20%** 上下文预算（`为什么宝玉不喜欢读书` 2606 字 → 2078 字）；24 条正向探针合计折叠 **71** 个同源兄弟，最极端（`空城计`/`曹操为什么要杀杨修`）单个去掉 7 个。
- 顺序：`2 × RERANK_TOP_K` 融合 → 去重 → 截断回 `RERANK_TOP_K`；只在 `RERANK_TOP_K=20` 池里去重只剩约 **13** 个不同父块；不足 `top_k` 用被折兄弟回填。
- 空查询守护：判据不能是 `text.strip()`：`hybrid_search("？？？")` 实测返回 5 条跨 3 本书、带 rerank 分数的"正常"结果，无从区分真召回与废话输入。
- 词表指纹不对称：只加别名时索引仍存"孔明"，查询"孔明"改成"诸葛亮"后再也匹配不到写"孔明"的正文，召回反更差。指纹写进 payload，不合时每进程一次大声报警、不抛错；修复 `chatbot.py reindex-sparse`。

#### rerank 粒度 A/B（`child` 判子块 vs `parent` 判父块）

在**旧索引 `books_v3`** 上实测：`parent` 三项判别性指标同向更好（kw@k 87.5%→**91.7%**、
MRR 0.8264→**0.8556**、hit@1_kw 79.2%→**83.3%**），但 rerank 输入 token 中位 100→**447**
（4.5×）、检索耗时 710ms→1299ms（约 2×）。

**结论：不改默认值。** 名次级胜负只有**改善 2 / 退化 0 / 持平 22**（`三打白骨精` 2→1、
`林冲为什么会被逼上梁山` 未命中→第 5）—— 差异全由 2 条探针驱动，**n=24 不足以改默认配置**，
故 `RAG_RERANK_ON` 仍默认 `child`。另外 `parent` 时输入 token **447 > `RERANK_MAX_LENGTH=384`**，
**真会发生截断**；把上限提到 512 实测并未更好（kw@k 反而 91.7%→87.5%），故 384 不变。
新索引上的完整对照表（含权重扫描）见 §9 (b)；工具 `chatbot.py ab-retrieval`。

### 8.5 上下文装配与生成

纯函数，`app.py`/`api.py` 共用：

| 函数 | 作用 |
|------|------|
| `format_context(results)` / `context_sources(results)` | 带出处上下文 `[n] 出处：《红楼梦》第三回 <回目>\n<父块正文>`；来源清单（书名/回次/回目/分数） |
| `build_system_prompt(...)` / `build_context_message(...)` | system 只放规则（`SYSTEM_RULES_PROMPT`）；正文装成独立消息（role 默认 `user`） |

```
messages = [system(规则，无正文) → 历史… → user(【检索资料】…) → user(本轮提问)]
```

- 资料消息与 system、本轮提问**同级：永不因裁剪而丢**，且它的 token 由 `history_budget(..., extra=资料)` 先从窗口里扣掉。
- 正文不再拼进 system：system 被置空或被截断时 RAG 会静默退化成"凭参数化记忆作答"（2026-09-18 三段提示词被整体置空即此后果）。
- `RAG_USE_CONTEXT=1`（默认）发规则+资料消息，只依据资料回答并标注 `[n]` 来源；`=0` 为对照模式，system 为空、也不发资料消息。检索无结果时三入口统一硬拒答（`rag_engine.NO_CONTEXT_ANSWER`）。

### 8.6 拒答（`confidence_signal`）

判据在 34 条探针校准。两条走不通：①首位分数绝对阈值不可用 —— 正例"三打白骨精"首位 **-0.111** 却是正确答案，负例"孙悟空和关羽谁更厉害"首位 **2.635**，区间重叠；②分数形状不可分 —— 正例 ratio 0.052~0.742、负例 0.405~1.914（"武松打虎"五条全在水浒、首位 4.484、spread 0.28）。只有"整体水平低"与"书目分散"同时成立才指向域外问题。

| 判据 | 条件 | 实测命中 |
|------|------|---------|
| A | 平均分 < -1.5 | 正例 0 条、负例 3 条 |
| B | 平均分 < 0.5 且 命中跨 ≥ 2 本书 | 正例 0 条、负例 6 条 |
| 合计 | | **误拒 0/24，漏拒 1/10**（漏"贾府最后被抄家了吗"，只命中 1 本书） |

- 误拒为 0 是第一原则。已知失效模式：判据 B 把"跨两本书"当可疑信号，合法的跨书对比（"比较林冲与武松"）会被误判 —— 这也是 `RAG_ABSTAIN` 可直接关掉的原因。
- `refuse` 不硬拦：只追加告诫（`LOW_EVIDENCE_CLAUSE`）+ 用户提示，裁定交给生成侧。换索引/换模型/换分块后阈值须重新校准。

### 8.7 多轮查询改写（`query_rewrite.py`）

```
Q: "他最后结局如何" → 召回 西游记 红楼梦 红楼梦 红楼梦 水浒传
→ rerank 分数 [0.834, 0.693, 0.656, 0.497, 0.157]（全线低分，即全部无关）
```

改写用本地 Ollama（`num_predict=64`、`think=false`），无新依赖。

- 失败必须退化为原查询：Ollama 没起/超时/返回垃圾/与原文相同/超 `MAX_REWRITE_CHARS` 一律退回原句（构造/清洗是纯函数，`_ollama_chat` 经 `call` 注入）。
- 历史默认不截断（`RAG_QUERY_REWRITE_MAX_TURNS=0`，2026-09-19 起）：上限由 `RAG_QUERY_REWRITE_NUM_CTX`（默认跟随 `RAG_NUM_CTX`）决定；旧"只取最近 4 轮"砍掉 LLM 档独有能力，切片按消息条数。`rewrite_prompt_stats` 进推理链。
- 三态可辨：`rewrite_query_detailed` 返回 `applied`/`reason` 落进 `steps["查询改写"]`，API `/search` 另有 `rewrite_detail`。原因码 `query_rewrite.REASON_*`。
- `history` 末尾误带本轮提问会被剥掉并告警（`rag_engine.strip_current_turn`）；跨轮切换实体会把"他"消解成更早出现的那一个（"张飞之死"→"关羽"）。

### 8.8 生成侧装配与上下文预算

`app.py`/`api.py`/`chat.py` 共用 `rag_engine.build_generation_messages()`：整轮裁历史、system 与本轮提问永不丢弃。实测（2026-09-18，top_k=5，`num_ctx=8192`/`num_predict=4096`）：

| 项 | 量级 |
|---|---|
| system 提示词（规则 + 5 个父块正文） | 1620~2590 token |
| num_predict（须先扣） | 4096 |
| **留给全部历史的** | **1200~2200 token ≈ 4~12 轮** |

模拟 12 轮（每轮约 200 字）：第 10 轮还装得下，第 11 轮裁（丢 2 轮），第 12 轮丢 5 轮。

超窗后（实测；第一版注释猜错）：不报错（18743 字输入照样 HTTP 200）；system 会保留；丢的是整条消息、连刚刚那一轮也丢（16000 字用户消息让 `prompt_eval_count` 从 18743 字塌到 **40 token**）。动机：丢弃可控、可观测、超长消息截尾保留。

观测量：`输入 token`（`prompt_eval_count` vs 本地估算 vs `num_ctx`）；`estimate_tokens()` 取上界（实测 qwen 约 0.77 字/token）。

### 8.9 检索用问句降级链（`build_retrieval_query`）

三档降级链，每档记进推理链「查询改写」（`source` = `llm`/`concat`/`literal`，`reason` 保留失败原因）：`LLM 改写 → 拼接上一轮用户句 → 字面原句`；中间档零延迟、不联网、不会失败。

实测（2026-09-18，21 条多轮探针，同一个 `run_multiturn` 评测器，只换检索用问句）：

| 检索用问句 | book@1 | book@k | kw@k | topic@k |
|---|---|---|---|---|
| 字面原句（旧降级落点） | 81.0% | 95.2% | 38.1% | **42.9%** |
| 拼接上轮用户句（现降级落点） | **100%** | **100%** | 71.4% | **95.2%** |
| 上轮用户+助手 + 本轮 | 100% | 100% | 57.1% | 100% |
| LLM 改写（开关打开） | 100% | 100% | 76.2% | 95.2% |

复核（`RAG_QUERY_REWRITE=0`，21 轮全走 `source=concat`）：book@1 100% / book@k 100% / kw@k 71.4% / topic@k 95.2% —— 关掉改写不再等于"字面检索"。

- 只拼上一轮用户句，不拼助手回答（助手措辞会把语义平均掉）：`kw@k` 71.4% → 57.1%（差 3 条探针）；`topic@k` 打平（95.2 vs 100 只差 1 条探针，n=21 不足以下结论）。
- 只在"本该改写却没改成"时拼（`disabled`/`call_error`/`empty_output`/`too_long` 白名单）；`identical` 与"本轮无可检索内容"不拼 —— 否则"？？？"会变成"拿上一轮话题去检索"。

⚠️ LLM 档数字不可复现，不要引用：该行只留聚合值、没逐条留痕（`entity_switch` 命中原 1/4 vs 记录 3/4）。结论：拼接档收益可复现，LLM 档未复现，需一次留痕完整的评测（逐条记录 rewrite 输出 + 指标）。机器当时还有另一进程在跑同一仓库、内存 free 仅 20%、71 万次 pageout、Ollama 改写频繁 20s 超时 → 依赖 Ollama 的测量此刻都不可靠。

故未改默认提示词：加"指代只取最近一轮提到的实体"规则（P3）后，4 条 `entity_switch` 上更差（0/4）。
## 9. 效果指标与产物实测

(a) 黄金基线 `tests/golden/retrieval_baseline.json`（2026-09-19 重录 `record_baseline`，21971 条，别名 books_current；多轮 21 → 30 条，改写/融合开）：

|组别|n|指标|
|---|---|---|
|正向（12 短 + 12 长问句）|24|book@1 95.8%、book@k 100%、kw@k 83.3%|
|├ 冻结的 12 条短探针|12|book@1 91.7%、book@k 100%、kw@k 100%|
|多轮（带 history，触发改写；融合关）|30|book@1 100%、book@k 100%、kw@k 70.0%、topic@k 100%|
|负样本（域外 / 语料缺失 / 原著不存在）|10|正确拒答 6/10，平均书目分散 2.5 本|

> 正向 24 条与旧基线（2026-09-17）逐条逐指标完全一致 = 没碰单轮链路；多轮组换了探针集（21 → 30）与配置，不与旧数字对拍。

- 12 条短探针是唯一能按位置对拍的：91.7%/100%/100%，与旧基线（2026-09-15，同 12 条）完全一致；故分 `aggregate` / `aggregate_short`。
- 多轮组 5 → 21 → 30 条（`MIN_PER_KIND=5`）。旧判据曾长期生效：L3 写成 `if not REWRITE_ENABLED: skip(...)`，默认 .env 恰为 0，整组失效；错在 `RAG_QUERY_REWRITE=0` 只关 LLM 档，拼接档仍替换检索问句。现判据：基线档 vs 当前档（`multiturn_degrade_state()` / `multiturn_fusion_state()`），不一致才 skip 并提示重录。
- 负样本 6/10：漏的 4 条属"语料部分相关"（如"宋江最后接受招安了吗"），分数判不出，交生成侧裁定。

(c) 多轮分组实测（2026-09-18 首录，集合 `books_v3__71371e82`，21 条全带 history，同一 `run_multiturn`）

> ⚠️ 右列"改写开 100% / 95.2%"实为"改写开 + prompt 完整"（2026-09-19 对照纠正）：`_SYSTEM_PROMPT` 未置空时测得，置空后塌到 book@1 85.7% / topic@k 66.7%，填回四行规则立刻回到 100% / 95.2%（见「设计存档 B」）。

|指代类型|n|改写关 book@1 / topic@k|改写开 book@1 / topic@k|
|---|---|---|---|
|`pronoun`（代词回指）|4|100% / 75.0%|100% / 100%|
|`assistant_only`（实体只在助手轮）|5|80.0% / 60.0%|100% / 100%|
|`entity_switch`（跨轮切换实体）|4|75.0% / 0%|100% / 75.0%|
|`ellipsis`（省略主语宾语）|3|66.7% / 33.3%|100% / 100%|
|`temporal`（"后来呢"）|3|66.7% / 33.3%|100% / 100%|
|`recall_detail`（指代一个内容词）|2|100% / 50.0%|100% / 100%|
|合计|21|81.0% / 42.9%|100% / 95.2%|

- `entity_switch` topic@k 仍 75%：唯一失手"张飞之死"（历史里先关羽后张飞，改写器把"他"消解成关羽：`'他最后是被谁杀的'` → `'关羽最后是被谁杀的'`）。
  ⚠️ 2026-09-19 更正：原写"修法方向在提示词"是错的；实测（30 条）该条恰是拼接档独家命中，正确修法是多路融合（见 (d)）。
- `recall_detail` 改写关时 topic@k 仅 50%：问句指代上一轮回答里的内容词，字面检索无从下手，是"旧轮原文不留"结构缺陷的直接证据。

(d) 多路融合实测（2026-09-19，30 条多轮探针 / 每类 5 条，集合 `books_current` = `books_v3__71371e82`，工具 `python chatbot.py ab-multiturn`）：

|配置|book@1|kw@k|topic@k|逐条变化|
|---|---|---|---|---|
|单路（LLM 改写，现状）|100%|70.0%|100%|—|
|融合 + `primary` 重排|100%|80.0%|96.7%|改善 3 / 退化 1|
|融合 + `join` 重排|100%|73.3%|96.7%|—|
|融合 + `per_route` 重排|100%|80.0%|96.7%|与 primary 持平|

- ⚠️ **融合不是"净收益"**：kw@k +10 个点（3 条改善：`他最后结局如何`、`后来他去哪里落草了`、`后来她还做过哪些贪财的事`）换 topic@k −3.3 个点（1 条退化：`他最后是被谁杀的` —— 两路候选池把正确段落挤出 top_k，而 rerank 仍按主路的错误问句打分）。按纪律「3 改善 / 1 退化撑不起改默认值」，`.env` 已改回 **`RAG_QUERY_FUSION=0`**；想开：置 1 后**必须重录基线**（口径变了）。
- `primary` 下 topic@k 没动的**根因不是融合无效**，而是 `rerank_indices(search_query, …)` **只用主路问句打分** —— 主路把"他"消解成了关羽，拼接路正确召回的张飞段落全被按错误问句打了低分。为此加了 `RAG_RERANK_QUERY`（primary / join / per_route）。
- `per_route` 与 `primary` 在新口径下**完全持平**（旧口径里"per_route 修好了那条"是假象 —— 那条探针本来就该判 Y）。重排口径 n=30 判不出来，按纪律保持 `primary`；延迟无可分辨差异，确定性开销 +1 稀疏编码 + 2 Qdrant 查询。

(b) A/B 与阈值扫描（新索引 21971 条、24 条正向探针，工具 `chatbot.py ab-retrieval` / `chatbot.py after-rebuild`）：

|配置|kw@k|MRR|hit@1_kw|rerank 输入 token 中位|检索耗时中位|
|---|---|---|---|---|---|
|`RERANK_ON=child`|83.3%|0.8021|79.2%|94|577ms|
|`RERANK_ON=parent`|87.5%|0.8472|83.3%|297|1137ms|
|权重扫描 `sparse=0.5` / `dense=2.0`（同值）|79.2%|0.7604|75.0%|—|—|
|权重扫描 `sparse=2.0`（同等权）|83.3%|0.8021|79.2%|—|—|

1. 等权 1:1 已是最优：`sparse=0.5` 与 `dense=2.0` 数字完全相同（只差均匀缩放）；`sparse=2.0` 与等权同值。
2. `parent` 方向一致更好但样本不足，默认保持 `child`：三项判别性指标同向（kw@k +4.2pp、MRR +0.045、hit@1_kw +4.2pp）；名次级胜负只有改善 2 / 退化 0 / 持平 22，n=24 不足以改默认值。`parent` 时输入中位 297 token，`RERANK_MAX_LENGTH=384` 未触及，提到 512 实测未更好。

(c) 确定性：两个独立进程各跑 24 正向 + 5 多轮探针，48 组结果的召回序列、rerank 分数、喂进 reranker 的 token 数逐位一致。

### 9.6 分块产物实测

新产物（2026-09-17，指纹 `7750036f5cc4c8c0`）与旧产物对照：

|项目|旧产物|新产物|
|---|---|---|
|总分块数|21137（三国 6727 / 水浒 1939 / 红楼 5010 / 西游 7461）|21971（三国 6988 / 水浒 1999 / 红楼 5279 / 西游 7705）|
|不同父块数|5701（平均 3.7 子块/父块）|5632|
|父块 token 中位|494|313|
|父块 token 最大 / <64 / <128|512 / 1188 / 1786|512 / **7** / **28**|
|其中属"贪心打包孤儿尾巴"（前一块 ≥480）|1293|**9**|
|子块 token 最大 / >128 越界|128 / 0|128 / 0|
|逐字空格损伤 / `chunk_index` 空洞|0 / 0|0 / 0|

父子相同的块（无额外上下文）：1790 / 21137 → **30 / 21971**。带回目元数据的块：0 → **21903 / 21971**；
剩下 68 块无标签（`chapter_index=0`）：水浒传「楔子」65 块，其余三本各 1 块。
## 10. 操作与排障

### 日常操作

```bash
docker compose up -d                      # 起 Qdrant

# 添加新书：放 books/*.txt → process → index
#   非章回体不会崩：parse_chapters 找不到回目时整本按单段处理（chapter_index=0）

python chatbot.py lexicon                 # 改词表后先校验（语料接地 + 停用词里不得有否定词）
python chatbot.py reindex-sparse          # 只重建稀疏向量（十几秒，不必重算 embedding）
#   不跑会报警"词表与索引不一致"—— 那是**对的**：查询侧现算的词空间与索引侧
#   对不上，稀疏通道会静默错配。

python chatbot.py health                  # 系统健康检查（含稠密/稀疏真检查）
python chatbot.py health --offline        # 跳过 Ollama 在线检查，最快的一遍

python -c "import json;print(len(json.load(open('cache_v2/chunks.json'))))"
curl -s localhost:6333/collections/books_current | python -m json.tool | head -20
python -m tests.record_baseline           # 重录 L3 黄金基线 → tests/golden/retrieval_baseline.json

python chatbot.py api                     # 默认 127.0.0.1:8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/search -d '{"query":"武松打虎","top_k":5}'
curl -s -X POST localhost:8000/ask    -d '{"query":"武松打虎"}'   # NDJSON 流

# 改多轮（改写 / 拼接 / 融合）—— 开关是 import 时常量，必须一个状态一个进程
python chatbot.py ab-multiturn --label a                       # 现状，~4 分钟
RAG_QUERY_FUSION=1 python chatbot.py ab-multiturn --label b
python chatbot.py ab-multiturn --compare /tmp/ab_multiturn_a.json /tmp/ab_multiturn_b.json
```

### 改什么跑哪层

| 改动 | 必须跑 |
|---|---|
| 分块参数 / 分句器 / 引号归一 | `pytest -m unit` + `-m needs_chunks`，然后 `process` + `index` |
| 换词表（别名/停用词） | `python chatbot.py lexicon` + `reindex-sparse` |
| 换模型 / 检索参数 | 重录基线 + `pytest -m slow`（零容差） |
| 任何改动 | `pytest -m unit`（含 D1~D16 全部护栏） |

分块类改动之后**必须重跑 `process`**（指纹门禁会拦住复用旧产物，但报的是"请重跑 process"，
不会告诉你哪里改了）。**所有命令都由人决定何时跑**，agent 不主动执行（见 §1）。

### 常见问题（症状 → 原因 → 处理）

| 症状 | 原因 | 处理 |
|------|------|------|
| 找不到 `chunks.json` | 没跑 `process` | `python chatbot.py process` |
| 报"分块产物与当前配置不一致" | 改了分句器 / 分块参数 / 引号归一后没重跑 | 先 `process` 再 `index`。**这是指纹门禁在按设计工作**，不是 bug |
| 集合为空 / health 报"向量库为空" | 没跑 `index` | `python chatbot.py index` |
| health 报"无法连接 Qdrant Docker 服务" | Docker 没起 | `docker compose up -d`。health 已把"连不上"与"集合为空"分开报 —— 报哪条就查哪条 |
| 稀疏通道无贡献 | 稀疏向量根本没写进去（Qdrant 不会自动生成） | `health` 的通道校验会直接报出来 |
| 中文稀疏召回差 | 用了 Qdrant 默认 word 分词器 | 确认走 `sparse_encode()`（jieba 预分词 + `Modifier.IDF`），不要绕开它 |
| 索引很慢 | MPS 与其它 GPU 任务并发争用（实测 **119 条/秒 → 23 条/秒**） | 避开并发；耗时差异极大时**先怀疑机器负载，不要先怀疑代码** |
| 重建索引后服务仍在用旧数据 | 进程内分块缓存 | 已由每 point 的 `build_id` 自动失效；若仍复现查 `_current_build_id()` |
| 检索质量差 | 分块大小 / 分句不对 | 调 `RAG_CHILD_MAX_TOKENS` / `RAG_PARENT_MAX_TOKENS`，再 `process` + `index`；用 `pytest -m needs_chunks` 验证不变量 |
| 改了 `.env` 没反应 | 缺 `python-dotenv`（现在会直接报错）；或名字写错（`QDRANT_HOST` 与 `RAG_QDRANT_HOST` 两个都接受） | `pip install -r requirements.txt`；再核对变量名与「核心配置」清单 |
| 集合点数与 `chunks.json` 条数不一致 | 改了分块但没重建索引，服务在跑旧数据 | `pytest -m needs_qdrant` 会直接报出来；重跑 `index` |
| 多轮提问答非所问 | 指代没消解，或改写降级到了字面档 | 看推理链「查询改写」的 `source`/`reason`（`llm`/`concat`/`literal`）—— 三态可分辨且都落进 steps（§8.9） |
| 检索"无结果"却不报错；`/ask` 返回 400 且关连接 | 空查询守护 / keep-alive 排空保护 | 都是**设计行为**（§8.4） |
## 11. 已知限制与遗留

1. **rerank 默认仍 `child`、RRF 保持等权** —— 两者都因**样本不足**未改默认值，数字与理由见 §9 (b) 与 §6 第 5 条。L3 基线的分组口径见 §7。
2. **跨进程确定性已证实**（零容差基线的前提）：三进程实测查询向量 SHA256、召回 point id 序列、rerank 分数（6 位小数）完全相同；"不同进程候选池不同"的旧观察已证伪。
3. **别名**：2026-09-17 重建写 `books_v3__71371e82` 并原子切 `books_current`；**首次启用别名时旧集合不自动删除**；未 `import rag_engine` 时 Qdrant 返回空 body 502。
4. **`confidence_signal` 阈值必须随索引/模型/分块重新校准**：判据 B 会误判合法的跨书对比问题（见 §8.6）。
5. **吞吐随负载剧烈波动，而休眠是真正的瓶颈**：43 句/秒是**空载**值；高负载实测 8.9 句/秒（240 句，load ~13.6）、~10 句/秒（192 句；线程 8/6/4 = 10.1/10.0/11.5）。休眠会**伪装成"极慢"**（墙钟 8 小时、CPU 仅 48 分钟），`caffeinate -i -s -w <PID>` 后瞬时 210% CPU。**"约 55 分钟"不是承诺** —— 先量当前吞吐再估。
6. **生成侧预算强耦合**：`RAG_THINK=1` 时思考约 2000 token，`num_predict` 是思考+答案的**总**预算，设小则答案为 0。
7. **`contextual_text` 仍是空壳**：恒等于 `child_text`；真做需独立特性并配 A/B。
8. **L3 四组口径**：正向 24 条零容差；12 条短探针单独一组；多轮比 `aggregate_all.multiturn` 四指标、`RAG_QUERY_REWRITE=0` 时跳过；负样本只记录不设断言。
9. **单机模式**：Qdrant 本地存储；HTTP API 全局锁串行化、Streamlit 单例，**不支持并发**。
10. **文档与代码一致性需主动维护**：文档/注释与代码多次不符（`RAG_QDRANT_HOST`、`RAG_ABSTAIN_RATIO` 等），特征是**静默无效** —— 改配置前先 grep 确认它真被读取。

## 12. 依赖

见 `requirements.txt` / `requirements-dev.txt`（pytest）。两条硬依赖没有降级路径，取舍见文件内注释：

- **`sentencex`**：缺库即 `ImportError`。曾经的"内置标点分句降级"会把闭合引号切给下一句，且是**静默**错的。
- **`python-dotenv`**：静默跳过会让 `.env` 的 `MODEL` / `OLLAMA_BASE_URL` / `RAG_*` 全失效，
  程序悄悄跑在代码默认值上 —— 表现就是"配置改了没反应"。

```bash
pip install -r requirements.txt -r requirements-dev.txt
```
## 13. 技术栈

- **向量数据库**: Qdrant (Docker部署)，写新集合 + 别名原子切换
- **搜索方式**: 稠密 + BM25 稀疏 → **客户端加权 RRF**（k 与权重可配）；稀疏侧 = jieba 分词 →
  别名归一 → 去古白话停用词 → Qdrant 内置 BM25 (`Modifier.IDF`)
- **分块**: 按「回」分段 → sentencex 单层分句（引语不可切）→ 语义定界 → 父块(512) / 子块(128, 重叠 32)
- **嵌入模型**: BGE-base-zh-v1.5 (768维)，CLS pooling，query 侧加 instruction
- **重排序模型**: BGE-reranker-base (fp16)
- **多轮**: 本地 Ollama 指代消解 + **三档降级链**（LLM 改写 → 拼接上一轮用户句 → 字面原句，
  每档记进推理链的 `source`/`reason`）；`RAG_QUERY_FUSION` 可把前两档各召回一次再融合
  （现状关闭，数字见 §9 (d)）。⚠️ "两路失手点互补"是在 **`topic@k` 还不认别名时**做的
  独家命中分析，**未在新口径下复验**
- **UI**: Streamlit（`RAG_UI=app|chat|bot`）；**HTTP API**: 标准库 `ThreadingHTTPServer`，零新依赖
- **生成模型**: Ollama（取自 `.env` 的 `MODEL`，当前 qwen3.5:4b-q4_K_M）
## 14. 待办

> 只记**尚未开工、且已确定要做**的条目。完成一项就移走（或标记完成 + 日期）。

### 14.1 上下文管理：压缩 + 多轮推理速度优化

**目标**：多轮在**不丢关键信息**的前提下，压低「历史 + 资料 + 本轮提问」的占用，并降低端到端延迟。
**压缩的入口就在历史那一段** —— system 与 `num_predict` 先吃掉大头（数字见 §8.8）。多轮的额外开销
有三处需分别定位：① 改写档的 LLM 指代消解；② `RAG_QUERY_FUSION=1` 时多一路召回再融合；③ 重排耗时。

- [ ] **先量后改**：拆出「改写 / 召回 / 重排 / 生成首 token」四段占比（**不要凭直觉优化重排**）。
- [ ] 压缩策略选型：LLM 摘要 vs 规则摘要、按轮裁 vs 按 token 预算裁；**不得破坏"资料消息永不因裁剪而丢"**。
- [ ] 压缩前后 A/B：用多轮探针（`tests/probes.py`，按 `kind` 分组）比 `topic@k`，**不能只看"回复读起来还行"**。
- [ ] 速度候选逐个量：改写结果缓存、失败快速退化、融合档按需开、流式首 token 与总墙钟分开看。
- [ ] 验收口径同时写清：**质量指标不下降**（哪几项、容差多少）+ **延迟下降多少**（什么机器/负载下）。

### 14.2 平台通用，垂直性只落「数据源向量化」与「提问推理」两段

**目标**：检索链路、融合、重排、装配、生成、接口都不为某个领域特化；垂直的只有离线
「数据源 → 向量化写入」与在线「提问 → 读哪个分区推理」。设计定稿见 §15 A（一个 `profile`(YAML)
= 一个数据集）。关键结论（**勿重新发明**）：**"分域"的正确实现是分 Qdrant collection**，
不能靠 payload/filter 区分。该设计**尚未落地任何代码**。

- [ ] 确认落点就是 §15 A 的执行顺序，还是先补一份更细的实施清单；**避免两个文档各说一套**。
- [ ] 先结清三条前置（prompt 备份是否还在、别名 `books_current` 是否真存在及其回退行为、离线清洗工序）。
- [ ] 纪律（唯一保留的一条）：`artifacts_dir` 与 `collection` 必须由 profile 派生、**跨 profile 绝不共享**。
- [ ] 第二垂直领域的选型判据：能否与当前词表/IDF 共享？不能则分 collection。
- [ ] 落地后补"通用层不许出现领域分支"的护栏（有 2 个以上 profile 时再加）。

## 15. 设计存档（A/B/C 合并摘要）

> 结论摘要。细节见本文件「数据处理流程」「测试与验证」「效果指标」「缺陷护栏」。

### A. 配置驱动的多数据集（原 ARCHITECTURE.md）

- 判据与设计 → **分数据集、不分域 = 分 Qdrant collection**；一个 YAML = 一个数据集（`source_dir` /
  `artifacts_dir` / `collection` / `alias` + 参数覆盖，三条命令共用）。**跨 profile 绝不共享** `artifacts_dir` 与 `collection`。
- 待改：`CHUNKS_JSON` / `EMBED_CACHE_DIR` / `BM25_CACHE_DIR` / `COLLECTION_NAME` + 新增 `load_profile()`
  （前三个最要命：44MB `chunks.json` 撞车 → 指纹门禁硬失败却报"重跑 process"）。砍掉 `DomainPack` 协议等过度设计。

### B. 多轮对话质效方案（原 MULTITURN_PLAN.md）

- 主轴 → 瓶颈是**改写单点下注**，改双路召回（`concat` + `llm_prompt`）+ 客户端 RRF 融合。
- 四态实测（2026-09-19，`books_current`=`books_v3__71371e82`，21971 点，多轮 21 → **30 条**，`MIN_PER_KIND=5`；book@1 / topic@k）：`literal` 81.0 / 42.9；`concat` 100 / 95.2；`llm`（**prompt 置空**）**85.7 / 66.7**；`llm_prompt`（填回四行规则）**100 / 95.2**。
- 结论 → **`_SYSTEM_PROMPT` 置空是崩塌根因**，填回四行规则立刻回 100 / 95.2；融合**非净收益**，已改回 `RAG_QUERY_FUSION=0`；**+1 次稀疏编码 + 2 次 Qdrant 查询**。
- 候选实体替换路 → **降级为备选**（`concat` 路 `entity_switch` 4/4 全对）；历史改确定性卡片。
- 明确不做 → 强化 `_SYSTEM_PROMPT`；提示词加实体消歧规则（已由 concat 路覆盖）；语义校验；Q-PRM。
- 门禁 → P0 校准（`test_retrieval_quality.py` 的判据 + `MIN_PER_KIND=5`）→ P1 融合 → P2 实体路 +
  历史卡片 → P3 延迟。纪律：**每个改动挂 env 开关、默认关、先 A/B 再开**。

### C. 首轮缺陷报告（原 DEFECT_REPORT.md）

- 编号 → 本节 D1~D8 **与「缺陷护栏」的 D1~D16 不通用**；"红楼 1722 个直引号"已被 D10 证伪。
  其中已失效的条目不要再引用。
- 明确验证为正常（防重复排查）：四层 61 条全绿；payload 与 `chunks.json` 21137/21137 零差异；稀疏单通道可召回；异常输入不崩；`fix_quotes` 对无直引号三本书是逐字节等价空操作。
