# AGENTS.md - 项目配置

本文件配置 opencode agents 如何与这个 RAG 知识库项目交互。

## 工作方式（对 agent 的硬要求，最高优先级）

**只开发和设计。不要主动自测 —— 主动性别这么强。**

- **不要自己跑测试**：`pytest`（任何 `-m` 分层）、`check_health.py`、
  `tools/ab_retrieval.py`、`tools/ab_multiturn.py`、`tools/verify_after_rebuild.py`、
  `compare_ab.py`、`compare_chunks.py`、`verify_qdrant.py`、`record_baseline`
  —— 一概不主动跑。
- **不要自己做端到端验证**：不主动起服务（`app.py` / `api.py`）、不主动
  `docker compose up`、不主动 `process` / `index` / `reindex_sparse`、
  不主动调 Ollama 或 Qdrant 发请求试探。
- **不要为"确认一下"写临时脚本**：包括一次性 `python -c` 探针、临时评测脚本、
  随手起的子进程。
- **测试、验证、测量一律由用户明确要求时才做**。用户没提就默认不做 ——
  哪怕"顺手跑一下就知道"，也不跑。
- 交付时**不许写"已验证 / 已实测 / 全绿"**，除非用户刚才明确让你跑过。
  结论若依赖实测，写成「未验证，需要你跑 X」，并把命令列出来交给用户。
- 需要新测试用例时**可以设计与编写**（写进 `tests/`），但**不要执行**；
  跑不跑由用户决定。
- 唯一例外：**用户本轮明确要求测试/验证**。此时才跑，且只跑被要求的那一层。

## 项目概述

这是一个**四大名著 RAG 知识库**，Qdrant Docker 部署，稠密+稀疏混合检索 →
**客户端加权 RRF** → 按父块去重 → 重排 → Ollama 生成（生成侧已真正接入检索结果）。

两条与旧版不同的主线，改代码前必须先知道：

- **融合在客户端**（`rag_engine.weighted_rrf`），不再用 Qdrant 服务端 `FusionQuery`。
  权重与 k 都可配，且融合是可单测的纯函数。
- **分块按「回」分段**（`chapter_parse.parse_chapters`），每块自带
  `chapter_index/label/title` 与 `parent_id`；分块产物另有指纹元信息，
  `build_index` 开工前会强制校验它与当前配置一致。

## 快速开始

```bash
# 1. 启动Qdrant服务
docker compose up -d

# 2. 构建索引
python app.py process
python app.py index

# 3. 健康检查
python check_health.py

# 4. 自动化测试（**由用户决定何时跑**；agent 不主动跑，见顶部"工作方式"）
pytest -m unit

# 5. 启动服务（二选一）
python app.py            # Streamlit UI :8501
python api.py            # HTTP API   :8000
```

## 项目结构

```
├── app.py                  # 统一入口：CLI 命令 + Streamlit UI
├── api.py                  # HTTP API（标准库实现，零新依赖）：/health /search /ask
├── rag_engine.py           # RAG 核心引擎：分块/嵌入/索引/检索（检索逻辑唯一权威实现）
├── bootstrap.py            # 进程级基础设施（**单源**）：.env 加载 / 环境变量解析 / 日志
├── rag_turn.py             # **一轮 RAG 的序列**（检索→装配→拒答→生成）事件流，四入口只渲染
├── ollama_client.py        # Ollama 客户端（**单源**）：模型名/think/预算/超时 + 流式解析
├── chapter_parse.py        # 章回体切回：按行首回目切分，纯标准库、逐字无损
├── query_rewrite.py        # 多轮查询改写（指代消解），失败退回原查询
├── check_health.py         # 系统健康检查（含词表指纹一致性）
├── compare_chunks.py       # 分块 A/B 比较器（按列表位置逐条比对，暴露边界漂移）
├── compare_ab.py           # 检索级 A/B（完整链路对比两个集合）
├── verify_qdrant.py        # 稠密/稀疏通道校验（含召回对比）
├── reindex_sparse.py       # 只重建稀疏向量（换词表/分词方案时免重跑 embedding）
├── tools/
│   ├── verify_lexicon.py   # 别名词典 / 古白话停用词表校验器（含语料接地检查）
│   ├── ab_retrieval.py     # 检索配置 A/B（只改环境变量，正向探针 + MRR）
│   └── ab_multiturn.py     # 多轮 A/B（降级档 / 融合 / 重排口径，逐类 + 逐条对拍）
├── data/
│   ├── aliases.txt             # 人物别名表（规范名 + 别名，制表符分隔）
│   └── stopwords_classical.txt # 古白话停用词表（禁收否定词/程度词）
├── tests/                  # 自动化测试 L0~L3（见下）
├── pytest.ini              # 测试分层 marker 配置
├── docker-compose.yml      # Docker编排文件
├── qdrant/
│   └── config.yaml         # Qdrant配置文件
├── books/                  # 源文本文件（四大名著 .txt）
├── models/                 # 预训练模型
│   ├── bge-base-zh-v1.5/   # 向量化模型（768维）
│   └── bge-reranker-base/  # 重排序模型
├── qdrant_storage/         # Qdrant Docker数据存储
├── cache_v2/               # 分块缓存
│   ├── chunks.json             # 分块产物（process 产出，index 输入）
│   ├── chunks.meta.json        # 分块配置指纹 + 源文本哈希 + 每回清单
│   └── embeddings/             # 向量缓存（首次建索引时才创建，按内容哈希命中）
├── .env                    # 环境变量配置
├── requirements.txt        # 运行时依赖
└── requirements-dev.txt    # 开发依赖（pytest）
```

## 测试分层

按"需要多少外部资源"分层，用 pytest marker 控制。资源缺失时**跳过**而非失败，
避免"本机没起 Docker"被误读成"代码坏了"。

```bash
pytest -m unit          # L0 纯函数：无需模型/Qdrant/语料，~22s / 304 条
                        #   （**agent 不主动跑**，由用户指定；见顶部"工作方式"）
                        #   （D9 的 3 条要起子进程，整层因此比纯函数用例慢几秒）
pytest -m needs_chunks  # L1 分块不变量：需 cache_v2/chunks.json（16 条）
pytest -m needs_qdrant  # L2 索引一致性：需 Qdrant 在跑（14 条）
pytest -m needs_models  # 需真 tokenizer（硬切无损 4 条 + build_chunks 5 条 = 9 条；
                        #   注意 L3 的 12 条也带此 marker，但被 slow 默认过滤）
pytest                  # 以上四层（L3 slow 默认跳过，共 366 条，全绿）
pytest -m slow          # L3 检索质量回归：需模型+索引，分钟级（12 条）
# 全部套件 = 378 条（366 默认 + 12 slow）
```

| 层 | 文件 | 守卫的不变量 |
|----|------|-------------|
| L0 | `tests/test_chunking.py` | 分句不跨段（含守护断言本身）、软换行不算段落、引号归一两种语料型、超长句兜底不超限/不自递归、子块不超 128 token、父子包含 |
| L0 | `tests/test_chapters.py` | 中文数字解析（含"第一百十回"这类省略写法）、逐字无损拼接、回序号连续、offset→回 二分查找、四本书真实回数（23/64/120/100） |
| L0 | `tests/test_probes.py` | 探针结构/接地：关键词真的在对应书里、负样本主题真的不在语料里、rel_gap 口径、**多轮指代类型（kind）合法且每类样本够** |
| L0 | `tests/test_retrieval_units.py` | `weighted_rrf` / `dedup_candidates_by_parent` / `confidence_signal` / 空查询守护 / 别名归一 / 平衡打包 / **上下文装配（system 只放规则、正文走独立消息、资料永不因裁剪而丢）** / 查询改写 / **生成侧历史预算（system 永不丢、整轮裁、装配后必落进窗口）** / 环境变量解析 |
| L0 | `tests/test_defects_regression.py` | D1~D15 缺陷护栏（见下）+ 引号归一的**语料级验收**、`hybrid_search` 调用点的 AST 静态护栏、`.env` 到达性（子进程）、生成侧配置/流式的**单源**（对象同一性 + AST 禁读）、`plan_generation` 三入口同口径 |
| L0 | `tests/test_api.py` | API 层：`_ENGINE_LOCK` 必须**可重入**（曾因非重入锁让 `/health` 自锁挂死 —— curl 只报超时、日志一个字都没有）、端点真能回话 |
| — | `tests/test_hard_split.py` | 硬切兜底逐字无损（不引入分词器空格、不丢字）且仍守 token 上限（需真 tokenizer） |
| — | `tests/test_build_chunks.py` | 真实 `build_chunks` 主函数：空白块被丢、短正文块被保留、编号无空洞 |
| L1 | `tests/test_chunk_invariants.py` | 字段完整、`parent_text` 含 `child_text`（忽略空白）、`chunk_index` 连续、`total_chunks` 正确、无逐字空格损伤、**真实 tokenizer** 下的 128/512 上限 |
| L2 | `tests/test_index_consistency.py` | 点数 == chunks.json、point id 连续、稠密/稀疏双写、向量归一化、build_id 唯一、稀疏单通道可召回 |
| L3 | `tests/test_retrieval_quality.py` | 正向 24 条的 book@1/book@k/kw@k 不低于基线（零容差）；**另分三组**：冻结 12 条短探针单独回归（唯一能与历史基线按位置对拍的一组）、多轮 `topic@k` 回归（按指代类型分组打印）、负样本只记录不设断言 |

L3 基线需先录制：`python -m tests.record_baseline` → `tests/golden/retrieval_baseline.json`。

**探针唯一来源**是 `tests/probes.py`：12 短 + 12 长问句（正向）、10 负样本、
**30 多轮**（每类 5 条，见 `test_probes.MIN_PER_KIND=5`；分六类指代：pronoun / assistant_only / entity_switch / ellipsis /
temporal / recall_detail，见 `VALID_MULTITURN_KINDS`；每类的失手原因与修法不同，
故 `aggregate_multiturn_by_kind` 分组报比率，不合成一个总分），
`compare_ab.py` / `verify_qdrant.py` / L3 都从这里导入。三类指标**分开报告、
不合成总分**（见 `tests/probes.py::aggregate_all`）——"短查询 100%、多轮 0%" 合成
一个 50 分就看不出问题在哪。

### 缺陷护栏（`tests/test_defects_regression.py`）

每条对应一个**已复现**的功能缺陷，断言的是正确行为。缺陷未修时用
`xfail(strict=True)` 保持套件绿色；修好后 strict xfail 会立刻变成 XPASS 失败，
强制摘掉标记。**D1~D15 现已全部修复，文件里已无 xfail 标记**。

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
| D9 | .env 里的 `RAG_QUERY_REWRITE=0` 是否生效**取决于 import 顺序**：`query_rewrite` 在 import 时读常量，而 `rag_engine` 顶层就 import 它。UI/API 恰好对（先 `load_dotenv`），而 `check_health.py` 把"已关闭"报成"开启"，`record_baseline` / `ab_retrieval` / `compare_ab` / `verify_qdrant` 里 .env 的 0 一律失效（"关掉改写做 A/B"实际没关） | `query_rewrite` 自己 `load_dotenv()`（`override=False`，命令行环境变量仍优先，A/B 用法不受影响），与 app.py / ollama_client.py 一致；D9 用**子进程**断言两种 import 顺序得到同一结论 |
| D10 | **`fix_quotes` 把闭引号改成了开引号**：D2 引入的"硬信号"假定"直引号左侧紧跟句读 ⇒ 必为开引号"，而该假定在本语料上是**反的**（红楼梦 `："` 0 次、`？"` 915 次、`！"` 402 次，直引号一律是闭引号）。后果：2426 个直引号里 2094 个被判成开引号，未配对 `“` 从 1600 涨到 **3362**，坏文本已烤进 `cache_v2/chunks.json`（可直接搜到 `意欲何往？“那僧笑道`） | 重写为**单遍二值状态机**：`“ ”` 与已判定的 `"` 共同维护"是否在引号内"，在内则闭、在外则开；删掉 `_OPEN_QUOTE_CTX` 与 `_paired_curly_positions`。复算：未配对 1600 → **12**。⚠️ 修好后**必须重跑 `process` + `index` 并重录基线**（旧基线建立在被改坏的正文上） |
| D11 | `chat.py` 把 `hybrid_search` 的**列表**返回值按二元组解包（`results, _ = …`，而 `return_steps` 默认 False）→ 提问即 `ValueError`；条数恰为 2 时更糟（`results` 变成 dict，后续对它调 `.get`） | 改为单值赋值；并加 **AST 静态护栏**：凡按元组解包该调用者，必须显式带 `return_steps=True` |
| D12 | `rag_engine` 的常量在 import 时求值，它自己却**不** `load_dotenv()`；此前靠"import 了 self-load 的 query_rewrite 且那次 import 恰好在配置段之前"侥幸成立，挪一下 import 就全仓库 `.env` 失效 | `.env` 加载收进 `bootstrap.py`（单源，`override=False`），任何读环境变量的模块先 `import bootstrap`；子进程断言两种 import 顺序同值 **且** .env 的显式非默认值真的到达模块 |
| D13 | **生成侧有两份同逻辑实现**：`app._stream_ollama(payload)` 与 `ollama_client.stream_chat(messages)`，且 MODEL / THINK / NUM_CTX / NUM_PREDICT / OLLAMA_TIMEOUT 在两边各读一遍 —— 超时一处硬编码 180s 而另一处可配（用户调大只对一个入口生效），RAG_USE_CONTEXT 只在 app.py 读而别的入口靠转发（chat.py 漏过） | 删掉 `app._stream_ollama`，四个入口统一 `from ollama_client import …`；`RAG_USE_CONTEXT` 移到 `rag_engine`。护栏：断言四个入口拿到的是**同一个** `stream_chat` 对象与同一份常量，并用 AST 禁止入口再读这些环境变量 |
| D14 | `chat.py` 漏传 `use_context`，`RAG_USE_CONTEXT=0` 在它那里静默失效（app/api 都传了） | 抽 `rag_engine.plan_generation()`：三个入口共用同一个装配函数（use_context / low_evidence / 硬拒答 / 历史预算全在里面），入口只剩渲染 |
| D15 | **同一份知识写在多处 → 必然分叉**（结构类，不是行为类）。实测三例：`compare_ab` 抄了一份召回管线（抄漏 `with_payload` 而崩）、`verify_qdrant` 抄了一份抽样检查（缺守卫、缺去重）、`chat.py` 抄了一份装配（漏 `use_context`）。同类还有：分块 schema 被声明 5 遍、`steps` 键名散落 8 个文件、`check_health` 与 `verify_qdrant` 各写一份抽样检查、`ab_retrieval` 与 `eval_runner` 各写一份跑探针的循环 | 全部收敛到单源：schema → `_PAYLOAD_SPEC`（写读两端派生）；steps 键 → `STEP_*` 常量且**写权收回 rag_engine**（`plan_generation` 产出上下文装配/历史裁剪，入口只 `steps.update(plan["trace"])`）；抽样检查 → `sample_vector_coverage`；跑探针 → `eval_runner.run_probes_with`；一轮序列 → **新模块 `rag_turn.run_turn`**，app/api/chat 只渲染事件。护栏：`TestContractsHaveSingleSource` 用 AST 断言"只允许一个写者/一个实现" |

### 其他已修缺陷（不在 D 表，但同样是"会静默出错"的那类）

| 位置 | 缺陷 | 修法 |
|---|---|---|
| `api.py` `/ask` | 响应头还没发出时抛异常，却仍往 `wfile` 裸写 NDJSON → 客户端收到没有状态行/header 的"响应"；同一故障在 `/search` 却是干净的 500 JSON | 加 `headers_sent` 分流：未发头 → 500 JSON；已发头 → error 事件 + 关连接 |
| `api.py` | 未读完请求体就拒绝（404 / 超限）→ HTTP/1.1 keep-alive **失步**，客户端下一个合法请求收到 400 | 拒绝前 `_drain_body()` 有界排空；所有错误响应带 `Connection: close` 并置 `close_connection` |
| `api.py` | `_REQUEST_LOCK` 覆盖**整段流式生成**（think 打开时实测 ~135s），且 Handler 无 socket 超时 → 一个慢/不读的客户端锁死全部请求 | 锁只包检索（生成不再碰 engine 状态）；`Handler.timeout = 60` |
| `api.py` | `history` 只校验是 list、不校验元素；`strip_current_turn` 对 `content=123` 抛 AttributeError → 500 而不是 400 | 逐项校验 role/content；`top_k` 显式拒掉 `bool` |
| `api.py` | 不剥"末尾即本轮提问"，与两个 UI 入口（都传 `[:-1]`）行为不一致 | 调用 `RE.strip_current_turn`，并在响应里回显 `history_stripped` |
| `rag_engine.hybrid_search` | 去重后用 `folded`（**同父**兄弟子块）回填条数 → 把同一段 `parent_text` 又塞回上下文，与去重的唯一目的直接矛盾 | **删除回填**，如实少返回；`steps["检索汇总条数"]` 上报缺口 |
| `rag_engine.build_generation_messages` | `_truncate_turn` 从不截用户消息 → 文档自己记录的"16000 字用户消息"场景仍会超窗，服务端静默整条丢消息 | 用户消息**单独超预算时**也截尾，用独立标记 `_USER_TRUNCATION_MARK`，并由 `truncated_user` 上报（UI/API 必须显示） |
| `query_rewrite` | `rewrite_prompt_stats` 在 `try` 之外被无条件调用（开关关掉也调），非 dict 历史元素让整条检索链崩 | `build_rewrite_prompt` 跳过非 dict；`build_retrieval_query` 整体兜异常退化为字面档；API 入口另做元素校验 |
| `query_rewrite` | 模型输出的前言（`好的，我来改写：`）被当成检索问句，且 `applied=True` / 推理链显示"改写成功" | 逐行扫描，跳过以冒号结尾或明显元话语开头的行；非 str 输出不再穿透"绝不抛异常"的契约 |
| `query_rewrite` | `build_retrieval_routes` 追加拼接路时漏判 `concat_enabled`（降级链判了）→ `CONCAT=0` 时融合档仍多召回一路，而 check_health 报"只剩 LLM→字面" | 与降级链同口径判断；`CONCAT_CHARS<=0` 也不再被解释成"截成空串" |
| `chapter_parse` | 回目正则要求"回"后紧跟空白 → "第一回"单独成行时不匹配；**漏末回会被并进上一回 body 且不报错** | 分隔符改前瞻 `(?=[^\S\n]\|$)`；新增"漏尾审计"：正文里出现编号 == 已识别回数+1 的宽松回目时硬失败（对当前四本书零误报：水浒/西游/三国 宽松命中数 == 严格命中数，红楼梦多出的两条编号为 4/38，均已在集合内） |
| `chapter_parse` | 注释声称兼容全角阿拉伯数字，字符类里只有 `0-9`（`chinese_to_int` 的全角分支永远不可达） | 字符类补 `\uff10-\uff19` |
| `chapter_parse.chinese_to_int` | "万"被纳入"相邻单位必须递减"的校验 → `一万` ✔ 而 `十万`/`二十万` ✘ | 把 `unit == 10000` 的节结算**提到递减校验之前** |
| `compare_ab.py` | `with_payload=False` 却读 `p.payload`（恒为 None）→ 工具**第一个探针即崩** | 改 `with_payload=True`，且字典构造统一走 `RE.chunk_from_payload`（`_get_chunks` 用的同一个函数） |
| `compare_chunks.py` | 字段表漏 `parent_id / chapter_*`，未登记字段被静默忽略 → 对"零影响"改动给**假绿** | 补全字段表；新增**模式漂移自检**：产物出现未登记字段直接报错退出 |
| `verify_qdrant.py` | 混合路是两通道**拼接未去重** → 命中数可超过分母上限（能打印出 `4/2`），而稠密侧无重复，两侧口径不对等 | 按 id 去重；并把 `命中数 > 上限` 变成断言（工具先能发现自己坏了） |
| `verify_qdrant.py` | 诊断代码在它要诊断的故障下自己崩（首条缺稀疏向量 → KeyError；抽样为空 → IndexError） | 空抽样与缺失向量都用 `.get()` + 显式报告，且稠密/稀疏**对称**报告 |
| `reindex_sparse.py` | 空集合时两处一致性检查恒真（`0==0`）→ 打印"完成"并 `return 0` | `total == 0` 直接失败退出；两遍写（改向量 / 写指纹）合并成一趟，避免中途中断留下"稀疏已换、指纹未换" |
| `check_health.py` | 外层 except 把所有异常报成"无法连接 Qdrant Docker服务" → 本地 `chunks.json` 损坏被误诊成 Docker 没起 | 本地文件读取自带 try/except，报出真实原因 |
| `app.py` | `THINK=0` 时 `done_reason=length`（答案被截断）的告警整块不执行 | 截断判定移出 `if think_status is not None`，改用 `st.warning` |
| `app.py` | 端口 8501 被别的进程占用时误报"Streamlit 服务已在运行"并 `exit 0` | 端口占用但 `_streamlit_pids()` 为空 → 报"端口被占用"并 `exit 1` |
| `app.py` | 每轮把整份 trace（同一批正文存两份，20~40KB/轮）写进 `session_state`，每轮 rerun 全量重建 | `_slim_trace()` 只留计数/名次/id；**保留 `上下文装配 → messages`**（"资料进没进"的唯一证据） |
| `bot.py` | 四个入口里唯一不做历史预算/空回合过滤的 → 长会话超窗后服务端静默丢消息 | 复用 `history_to_messages` + `build_generation_messages` + `history_budget`；`process_ph` 补 None 保护 |

## 核心配置

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

# 生成侧 / 集成（app.py、api.py 读）
RAG_USE_CONTEXT    = 1         # 1=注入检索结果  0=对照模式（system 与资料消息都不发）
RAG_CONTEXT_ROLE   = "user"    # 承载检索正文的那条独立消息的 role：user/tool/system。
                               #   **2026-09-19 起正文不再拼进 system** —— 旧写法把
                               #   正文塞在 CONTEXT_SYSTEM_PROMPT 的 {context} 里，
                               #   system 一旦被置空/被服务端截断，RAG 就静默退化成
                               #   "凭记忆作答"（2026-09-18 真发生过：正文一条没进）。
                               #   现在 system 只放规则（SYSTEM_RULES_PROMPT）、正文
                               #   由 build_context_message 单独发，装配顺序：
                               #   system → 历史 → 资料消息 → 本轮提问。
                               #   "资料到没到"因此成为可直接断言的事实。
                               #   取值非法启动即报错。切换 role 做 A/B 时记得
                               #   一个进程一个状态（import 时常量）
RAG_QUERY_REWRITE_CONCAT = 1   # 降级链中间档：LLM 改写不可用时拼"上一轮用户问句+本轮"
                               #   （实测 topic@k 42.9%→95.2%，零延迟零失败面）
RAG_QUERY_REWRITE_CONCAT_CHARS = 60  # 拼接时上轮问句的截断长度
RAG_QUERY_FUSION = 0           # 多路召回：LLM 改写成功的轮次**额外**用拼接档
                               #   各召回一次，两路一起进 RRF。LLM 失败时自动退化为
                               #   拼接单路，不新增失败面。
                               #   ⚠️ 保持关闭：修正 topic@k 的别名口径后实测
                               #   3 改善 / 1 退化（kw@k 70%→80%，但一条 entity_switch
                               #   的 topic@k 掉了）—— 撑不起改默认值。旧记录里的
                               #   "改善 3 / 退化 0"是度量假象，见 MULTITURN_PLAN.md §1.4
RAG_RERANK_QUERY = "primary"   # 多路时 reranker 拿哪句打分：primary / join / per_route。
                               #   新口径下 primary 与 per_route 持平，故保持默认；
                               #   两种口径的数字差异见 MULTITURN_PLAN.md §1.4
RAG_QUERY_REWRITE              # 多轮指代消解的**开关**，取值不在这里写死：
                               #   以 .env 为唯一权威，原因见"几个必须知道的坑"第 4 条
                               #   ——本行曾写 1 而 .env 是 0，属于同一类文档漂移。
                               #   **2026-09-19 起 .env 是 1**，与"恢复 _SYSTEM_PROMPT
                               #   四行规则"配套：实测"prompt 置空 + 开关打开"是最差的
                               #   一档（topic@k 66.7%），两者不可分开动。
                               #   想确认当前生效值：python check_health.py
                               #   或 GET /health 的 query_rewrite 字段。
RAG_THINK / RAG_NUM_CTX / RAG_NUM_PREDICT
# 生成侧装配（三个入口共用 rag_engine.build_generation_messages）
RAG_HISTORY_MAX_TOKENS = 1500  # 兜底历史预算；**正常不用它** —— 调用方一律用
                               #   history_budget(num_ctx, num_predict, system) 现算：
                               #   实测 system 占 1620~2590 token、num_predict 再扣 4096，
                               #   留给历史的只有 1200~2200 token（4~12 轮）。
                               #   超窗时服务端不报错、保留 system，但会整条丢消息
                               #   （实测一条 16000 字的消息把输入从 18743 字塌到 40 token）
RAG_INPUT_RESERVE = 256        # 从窗口里额外扣掉的余量（模板开销 + 估算误差）

# 词表
RAG_ALIASES / RAG_STOPWORDS = 1 / 1
RAG_LEXICON_DIR = "./data"

# 产物路径（ARCHITECTURE.md §4.1 的多数据集前置改造，纯搬家）
RAG_ARTIFACTS_DIR = "./cache_v2"  # chunks.json / chunks.meta.json / embeddings/ 都由它派生。
                                  #   此前三个路径是各自硬编码的常量，把 CHUNKS_JSON 指向临时
                                  #   目录的调用方（测试、A/B）会把向量缓存与元信息写回真实目录

# 生成侧超时（**只有 ollama_client 读**，四个入口都从它取，不可能分叉）
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

完整清单见 `PROJECT_DOC.md` 第 5 节。

## 常用操作

### 启动Qdrant服务
```bash
docker compose up -d
```

### 添加新书
1. 把 `.txt` 文件放到 `books/` 目录
2. 运行 `python app.py process`
3. 运行 `python app.py index`

非章回体文本不会崩：`parse_chapters` 找不到回目时整本按单段处理（`chapter_index=0`）。

### 改了词表（别名/停用词）
```bash
python tools/verify_lexicon.py   # 先校验：语料接地 + 停用词里不得有否定词
python reindex_sparse.py         # 只重建稀疏向量（十几秒，不必重算 embedding）
```
不跑的话检索端与 `check_health.py` 会报警"词表与索引不一致"——那是**对的**：
查询侧现算的词空间与索引侧对不上，稀疏通道会静默错配。

### 健康检查
```bash
python check_health.py
```

### HTTP API
```bash
python api.py                    # 默认 127.0.0.1:8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/search -d '{"query":"武松打虎","top_k":5}'
curl -s -X POST localhost:8000/ask    -d '{"query":"武松打虎"}'   # NDJSON 流
```

### 重录检索基线
```bash
python -m tests.record_baseline  # → tests/golden/retrieval_baseline.json
pytest -m slow
```

### 改多轮（改写 / 拼接 / 融合）
```bash
python tools/ab_multiturn.py --label a                       # 现状，~4 分钟
RAG_QUERY_FUSION=1 python tools/ab_multiturn.py --label b    # 一个状态一个进程！
python tools/ab_multiturn.py --compare /tmp/ab_multiturn_a.json /tmp/ab_multiturn_b.json
```
开关是 import 时常量，**同进程改环境变量不生效**（见"几个必须知道的坑"第 4 条）。

## 几个必须知道的坑

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
   代码确认它真被读取；②确认当前生效值用 `python check_health.py`（它现在会把
   多轮改写开关的实际状态打出来），而不是读文档。
5. **`RAG_RERANK_ON` 默认保持 `child`（已实测两轮，样本不足）**：24 条正向探针上
   `parent` 三项判别性指标同向更好（kw@k 83.3%→87.5%、MRR 0.8021→0.8472、
   hit@1_kw 79.2%→83.3%，检索耗时 577ms→1137ms），**在旧索引与新索引上结论一致、
   零回退**；但**名次级胜负只有"改善 2 / 退化 0 / 持平 22"** —— 差异基本由 2 条
   探针驱动，n=24 不足以据此改默认值。要下结论请先扩充探针集。
   **融合权重已扫描：等权 1:1 就是最优**（`sparse=0.5` 与 `dense=2.0` 都降到
   kw@k 79.2%、MRR 0.7604；`sparse=2.0` 与等权同值），无需调整。
   A/B 用 `python tools/ab_retrieval.py`（同一评测器跑两次，只改环境变量）；
   重建后的全套复核用 `python tools/verify_after_rebuild.py`
   （确定性双跑 diff + 拒答阈值扫描 + 分类指标）。
6. **别把"给模型的上下文"寄生在 system 上**（2026-09-19 拆开，原因见下）：
   旧写法把检索正文拼进 system（`CONTEXT_SYSTEM_PROMPT` 里的 `{context}`），于是
   system 一旦被置空、或被服务端截断，**RAG 就静默退化成"凭记忆作答"**，
   回答照样通顺（2026-09-18 置空事故就发生在这一条上，是第四次同类漂移）。
   现在：system 只放规则（`SYSTEM_RULES_PROMPT`），正文由
   `build_context_message` 装配成**独立消息**（role 见 `RAG_CONTEXT_ROLE`，默认
   `user`），顺序 `system → 历史 → 资料 → 本轮提问`。判断"这轮到底是不是 RAG"
   请直接看 messages 里有没有那条资料消息 / 推理链里的「上下文装配 → messages」，
   **不要**再靠"system 里有没有某个子串"来推断。

## 技术栈

- **向量数据库**: Qdrant (Docker部署)，写新集合 + 别名原子切换
- **搜索方式**: 稠密向量 + BM25稀疏向量 → **客户端加权 RRF**（k 与权重可配）
- **稀疏方案**: jieba 分词 → 别名归一 → 去古白话停用词 → Qdrant 内置 BM25 (Modifier.IDF)
- **分块**: 按「回」分段 → sentencex 单层分句（引语不可切）→ 语义定界 → 父块(512) / 子块(128, 重叠 32)
- **嵌入模型**: BGE-base-zh-v1.5 (768维)，CLS pooling，query 侧加 instruction
- **重排序模型**: BGE-reranker-base (fp16)
- **多轮**: `query_rewrite.py` 用本地 Ollama 做指代消解，**三档降级链**
  （LLM 改写 → 拼接上一轮用户问句 → 字面原句，每档都记进推理链的 `source`/`reason`）；
  `RAG_QUERY_FUSION` 可把 LLM 档与拼接档**各召回一次再融合**。
  ⚠️ "两路失手点互补（LLM 独家救 assistant_only、拼接档独家救 entity_switch）"
  这一结论是在 **`topic@k` 还不认别名时**做的独家命中分析，**未在新口径下复验**，
  引用前请先重跑 `tools/ab_multiturn.py`（见 MULTITURN_PLAN.md §1.1 的注记）
- **UI**: Streamlit；**HTTP API**: `api.py`（标准库，零新依赖）
- **生成模型**: Ollama（取自 `.env` 的 `MODEL`，当前 qwen3.5:4b-q4_K_M）
