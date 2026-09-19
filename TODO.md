# TODO.md —— 待办清单

> 本文件只记**待办**，不记已完成的事。完成一项就把该项移走（或标记完成并注明日期），
> 不要让它长期留在这里当"曾经想做的事"。
>
> 与三大文档的分工：`ARCHITECTURE.md` 是**多数据集配置化**的设计定稿（其 §"待办（下次开工）"
> 是那轮设计的遗留项），`AGENTS.md` 是 agent 协作硬约束，`PROJECT_DOC.md` 是现状说明。
> 本文件记**尚未开工、且已确定要做**的条目。

---

## 1. 上下文管理：压缩 + 多轮推理速度优化

**目标**：多轮对话在**不丢关键信息**的前提下，把「历史 + 检索资料 + 本轮提问」占用压下去，
并把端到端多轮延迟降下来。

**现状（读代码/文档即可确认，未实测）**

- 生成侧装配由 `rag_engine.build_generation_messages` 统一负责，
  顺序为 `system → 历史 → 资料消息 → 本轮提问`，正文**不寄生在 system 上**（见 `AGENTS.md` 第 6 条坑）。
- 历史预算由 `history_budget(num_ctx, num_predict, system)` 现算，`RAG_HISTORY_MAX_TOKENS` 只是兜底；
  system 与 `num_predict` 先吃掉大头，留给历史的只剩很小一块——**这就是压缩要解决的入口**。
- 多轮的额外开销来源有三处，需分别定位：① `query_rewrite` 的 LLM 指代消解调用；
  ② `RAG_QUERY_FUSION=1` 时多一路召回再融合；③ 重排耗时（不同 `RAG_RERANK_ON` 口径差异明显）。

**待定/要做的**

- [ ] 先量后改：拆出「改写耗时 / 召回耗时 / 重排耗时 / 生成首 token 耗」四段各自的占比，
      搞清楚瓶颈在哪一段（不要凭直觉优化重排）。
- [ ] 压缩策略选型：历史摘要（LLM 摘要 vs 规则摘要）、按轮裁剪、按 token 预算裁剪，
      以及**压缩后如何保证"资料消息永不因裁剪而丢"**（现有不变量，不能被压缩破坏）。
- [ ] 压缩与压缩前做 A/B：用多轮探针（`tests/probes.py`，按 `kind` 分组）比对 `topic@k`，
      不能只看"回复读起来还行"。
- [ ] 速度优化的候选手段逐个量：改写结果缓存、失败快速退化、融合档按需开、
      流式首 token 与整体墙钟分开看。
- [ ] 验收口径要同时写清：**质量指标不下降**（哪几项、容差多少）+ **延迟下降多少**（在什么机器/负载下）。

**相关文件**：`rag_engine.py`、`query_rewrite.py`、`app.py`、`api.py`、`AGENTS.md`（生成侧装配与预算一节）。

---

## 2. 架构通用，但「数据源向量化」与「提问推理」两段分垂直领域

**目标**：平台代码保持**通用**（检索链路、融合、重排、装配、生成、接口都不为某个领域特化），
而**垂直性只落在两处**：离线"数据源 → 向量化写入"、在线"提问 → 读哪个分区推理"。

**现状**

- 已有设计定稿：`ARCHITECTURE.md` —— 一个 `profile`（YAML）= 一个数据集，
  通通用、垂直的只有 `source_dir` / `artifacts_dir` / `collection` / `alias` / `lexicon_dir` / `prompts_dir` + 参数覆盖。
- 关键结论（勿重新发明）：**"分域"的正确实现是分 Qdrant collection**，不能靠 payload/filter 区分，
  理由见 `ARCHITECTURE.md` §2.1 五条（IDF 污染 / 维度一致 / 别名原子切换 / 指纹门禁 / 拒答阈值标定）。
- 该设计**尚未落地任何代码**（`ARCHITECTURE.md` 明确"本轮不修改任何代码，只交付文档"）。

**待定/要做的**

- [ ] 确认本条的落点就是 `ARCHITECTURE.md` 的执行顺序（§8 第 1~4 步），
      还是需要先补一份更细的实施清单；**避免两个文档各说一套**。
- [ ] `ARCHITECTURE.md`「待办（下次开工）」里的三条前置（prompt 备份是否还在、
      别名 `books_current` 实际是否存在及其回退行为、离线清洗工序）**先结清**，
      否则配置化做起来地基不稳。
- [ ] 纪律（唯一保留的一条）：`artifacts_dir` 与 `collection` 必须由 profile 派生、
      **跨 profile 绝不共享**。
- [ ] 第二垂直领域的选型与判据：能否与当前词表/IDF 共享？不能则分 collection。
- [ ] 落地后补齐"通用层不许出现领域分支"的护栏（有 2 个以上 profile 时再加，单个数据集时不必）。

**相关文件**：`ARCHITECTURE.md`、`rag_engine.py`（`resolve_collection_name` / `default_collection_name` /
分块指纹校验）、`check_health.py`、`data/`、`cache_v2/`。
