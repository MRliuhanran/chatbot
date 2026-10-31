# 03 · 在线问答链路：双通道召回、RRF、重排、预算装配与流式生成

> 代码基线：chatbot.py（5872 行）。全文引用格式为 chatbot.py:Lx-Ly；"默认值"指代码里的默认，
> 本仓库 .env 的实际生效值统一列在 §12。本文只做只读分析，不涉及任何代码改动。
>
> 覆盖：L593-L813（采样/惩罚与 stream_chat）、L2872-L3030（检索原语）、L3030-L3599（上下文装配）、
> L3599-L3681（适配器协议）、L3681-L4097（RAGEngine）、L4100-L4216（run_turn）、L1034-L1058（单轮台账）；
> 另用 L4687-L4868 说明失败与 Stop 在 UI 层的收口（本层唯一的现网消费方，逻辑强耦合）。

## 0. 一句话概括

在线链路是一条单向管线："**稠密/稀疏双通道召回 → 加权 RRF 融合 → 父块去重 →
全体候选重排（池上限已退场，2026-10-30）→ 父块文本装配成带属性的资料块 → token 预算裁剪 → Ollama 流式生成**"。
生成侧对外只有一个封闭事件词表（retrieval / thinking / delta / error / done），error 与 done 互斥；
失败轮不写会话历史、只写一行单轮台账。整条链路贯彻同一条工程原则：**任何降级都必须显式可见**
（写日志 + 写 steps 诊断），不接受"看起来正常、实际退化"的静默状态。

## 1. 在分层架构中的位置

文件头把全仓分为 L1-L6（chatbot.py:L1-L9），在线链路正好横跨全部层级：

| 层 | 在线链路中的组件 | 代码位置 | 依赖方向 |
|---|---|---|---|
| L1 配置/日志/运行态 | 所有 env 常量、_RuntimeState、_timed | L49-L569、L490-L527 | 不依赖业务 |
| L2 纯函数（domain） | weighted_rrf、dedup_candidates_by_parent、confidence_signal、build_generation_messages、_cut_to_tokens | L2892-L2911、L2971-L3002、L3405-L3522 | 只吃显式入参 |
| L3 适配器（I/O 与资源） | EmbeddingBackend / RerankBackend / OllamaChatBackend | L3599-L3678 | 独占 requests/torch/qdrant_client |
| L4 检索引擎 | RAGEngine.hybrid_search | L3681-L4097 | 只依赖 L3 协议 + L2 纯函数 |
| L5 单轮编排 | run_turn、first_retrieval（drain_events 已并入 `_stream_reply` 的 `_stream_chunks`，2026-10-30） | 依赖 L4 + L3 的 chat 可调用对象 |
| L6 UI | _stream_reply、answer_is_good、_run_assistant_turn、run_streamlit | L4676-L4970 | 唯一 import streamlit 的层 |

这套分层对在线链路最实际的两个好处：
1. **后端可替换而不动编排**：嵌入/重排各有 torch 与 HTTP 两套实现，torch/http 的差异被收在
   L3（L3699-L3704 只选一次），hybrid_search 里只有一条路径（L3989）。
2. **编排可被非 Streamlit 调用**：run_turn 通过 chat 参数注入生成函数（L4137），测试/API 可以
   传入自己的流，不需要 import streamlit。

## 2. 端到端时序图

```
用户提问
  │
  ▼
run_streamlit                          L4965-L4970
  │  ① messages.append({role:"user", content:prompt})
  │  ② history = messages[:-1]          ← 本轮提问不重复作为历史
  ▼
_run_assistant_turn                    L4796-L4868
  │  events = run_turn(engine, prompt, history, top_k=TOP_K)
  ▼
run_turn                               L4266-L4343
  │  results  = engine.hybrid_search(query, top_k, source)   L4280
  │  confidence = results_confidence(results)                 L4281
  │  plan = plan_generation(results, history, query, num_ctx, use_context=…) L4283
  │  yield RETRIEVAL {query, results, confidence, gen, elapsed} L4287
  │  for kind, delta in chat(plan["messages"], num_ctx=…, num_predict=…)  L4295
  │      yield thinking / delta / done / error
  │  record_turn_stats(...)                                   L4318-L4335
  ▼
_run_assistant_turn → first_retrieval / _stream_reply（st.write_stream 追加式渲染）
  │  累积 thinking/answer/error，done 单独保存
  │  流中 error → st.error + 失败提示；空答 → st.warning；done_reason=length → st.warning
  ▼
answer_is_good(gen_error, done, answer) L4994-L4999（唯一调用点 L5062）
  ├─ True  → st.session_state.messages.append(assistant)      L5071-L5074
  └─ False → rollback_orphan_user()（撤掉本轮 user）          L5062-L5070
```

几个"看起来像细节、其实是契约"的点：

- **retrieval 是第一个事件，且必须由消费方用 next() 取走**（first_retrieval，L4253-L4265）。
  `_stream_chunks`（原 drain_events）收到词表外事件直接抛错，不把"事件分叉"伪装成正常结尾；
- **检索为空也照常调用生成**（L4284 注释），没有"检索为空就拒答"的短路。confidence 只统计不判定
  （L2983-L2984 的 docstring 明说"不产生拒答"）。这是产品决策：空知识库退化为纯对话。
- **失败轮不写历史**：判据在 UI 层 answer_is_good（无 error + 有 done + 正文非空），
  而不是塞进 run_turn。注释解释了原因：将来新增 API 消费点应各自声明起闸与否（L4790-L4792）。

## 3. 生成侧：采样参数、请求载荷与流式状态机（L593-L813）

### 3.1 参数单一来源

Ollama options 只在 _chat_options 一处组装（L659-L686）。它只显式"钉死"应用要负责的项：
temperature、num_ctx、num_predict、top_k、top_p、min_p、repeat_last_n、repeat_penalty、
presence_penalty、frequency_penalty、seed，外加可选 num_gpu。未列出的交给模型自带默认。

| 参数 | 代码默认 | 本仓 .env | 定义行 | 说明 |
|---|---|---|---|---|
| MODEL | qwen3.5:2b-q4_K_M | minicpm5-2b | L572、.env:L76 | Ollama 模型名 |
| RAG_THINK | False | 1 | L574、.env:L190 | 思维链唯一开关：只影响展示，永不进生成上下文 |
| RAG_NUM_CTX | 8192 | 32768 | L577、.env | 上下文窗口（token）；2026-10-28 由 131072 下调（8G 档 KV 2.86G→0.75G/路） |
| RAG_NUM_PREDICT | 4096 | 0 | L578、.env:L205 | ≤0 映射为请求里的 -1（不限长） |
| OLLAMA_TIMEOUT | 180s | 900 | L579、.env:L253 | 只用于启动预热探针，不用于问答流 |
| RAG_STREAM_READ_TIMEOUT | 300s | 300 | L583、.env:L256 | 流式读空闲超时（无新数据即判超时） |
| RAG_TURN_TIMEOUT | 1800s | 1800 | L584、.env:L257 | 整轮硬上限（0=关闭） |
| RAG_TEMPERATURE | 0.0 | 0.0 | L586、.env:L221 | 0=贪心 |
| RAG_REPEAT_LAST_N | 64 | 128 | L589、.env:L236 | 三类惩罚共用的回看窗口 |
| RAG_GEN_TOP_K | 20 | 20 | L597、.env:L224 | 贪心下不参与 argmax，保留给温度>0 的 A/B |
| RAG_GEN_TOP_P | 0.9 | 0.95 | L598、.env:L225 | 同上 |
| RAG_GEN_MIN_P | 0.0 | 0 | L599、.env:L226 | 同上 |
| RAG_GEN_PRESENCE_PENALTY | 0.0 | 0.0 | L603、.env:L231 | 显式覆盖模型自带 1.5：否则资料里每个词都被压制 |
| RAG_GEN_FREQUENCY_PENALTY | 0.0 | 0.0 | L604、.env:L232 | 同上 |
| RAG_GEN_REPEAT_PENALTY | 1.1 | 1.1 | L605、.env:L233 | 贪心下依然生效 |
| RAG_GEN_SEED | -1 | -1 | L607、.env:L238 | -1=每次随机 |
| RAG_GEN_NUM_GPU | 空=不发 | auto | L941-L952、.env:L332 | 见 §3.3 |

设计意图（代码注释已写明，值得复述）：**惩罚项直接改 logits，贪心下也生效**，所以"模型自带
presence_penalty=1.5"这类隐式默认必须显式归零，否则人名、术语的正常复现会被压制（L600-L605）。
而截断类参数在 temperature=0 时对 argmax 完全无影响，保留只为做 A/B（L594-L596）。

### 3.2 请求载荷与连接

stream_chat（L727-L804）每次请求组装：

```
payload = { model, messages, stream: true, think: <THINK 或显式覆盖>, options: _chat_options(...) }
POST {OLLAMA_BASE_URL}/api/chat   timeout=(10.0 连接, STREAM_READ_TIMEOUT 读空闲)
```

- **不发请求级 keep_alive**（L748-L749）：生成模型驻留/卸载完全交给 Ollama 服务端策略，
  应用层不编排生成模型内存（与 L920-L928 的"2026-10-24 定稿"一致）。
- **HTTP 连接复用是线程本地 Session**（_ollama_session，L634-L646）：不用进程级单例，
  因为 requests.Session 不保证线程安全；线程本地后 keep-alive 收益不减、跨线程竞争消失。
- **payload 里 think 字段**：关闭时既是一个开关，也是"生产端唯一闸门"——关闭时事件流里不存在
  EVENT_THINKING（L781-L790）。若模型没听 think=false 仍返回思维链，则按开关丢弃并告警一次。

### 3.3 num_gpu=auto 的一次性回落（L689-L724）

```
                    RAG_GEN_NUM_GPU
                          │
        ┌─────────────────┼──────────────────┐
        │ 空               │ 整数 (0/999/-1)  │ "auto"
        ▼                 ▼                  ▼
   不发 num_gpu      原样放进 options     先发 999
   (llama.cpp 自动分层)                    │
                                            ├─ 流开始前 HTTP 非 200
                                            │   且错误文本含设备类关键字？
                                            │      ├─ 否 → 不回落，直接报错
                                            │      └─ 是 → 去掉 num_gpu 重发一次
                                            │            并置进程级 fallback=True
                                            │            （此后本进程不再尝试 GPU）
                                            └─ 200 → 正常流
```

关键取舍（L711-L718 注释）：旧判据是"任何非 200 就回落"，于是容器冷启动时模型还在后台拉取的
404、或一次 503，就能把 GPU 机器此后所有请求永久降级成自动分层，且日志归因成"GPU 不可用"。
现在只认设备类关键字（cuda / out of memory / oom / device / gpu / metal / mps / vram /
insufficient memory，L715），**回落是一次性的、发生在零 token 产出之前**，因此不构成"生成重试"，
不会丢弃任何已生成内容。回落状态没有复位路径（进程级），这是刻意选择：避免每轮白等一次失败。

### 3.4 三层时间闸

| 闸 | 位置 | 默认 | 作用 | 失败表现 |
|---|---|---|---|---|
| 连接超时 | L753 | 10s（硬编码） | TCP/握手不无限等 | EVENT_ERROR 连接类文案 |
| 读空闲超时 | L753 | 300s | 流完全静默 = 明确错误，而非前端永久空白 | EVENT_ERROR"生成超时/连接中断" |
| 整轮硬上限 | L768-L770 | 1800s | 长回答有进度时也总有个上界 | EVENT_ERROR"生成本轮超过 …s" |

注意二者的相互作用：整轮上限**只在收到新行时检查**，所以"完全没有数据"的场景先由读空闲超时
（更短）触发；整轮上限真正约束的是"一直有 chunk、但拖得极久"的生成。这是有意的分工，不是漏洞。

### 3.5 流式事件状态机与"error/done 互斥"

```
iter_lines() ──空行──▶ 跳过
     │
     ├─ 整轮超时？ ──是──▶ yield ERROR; return
     ├─ chunk["error"]？ ──是──▶ yield ERROR; return     ← 关键：return 而非 continue
     ├─ message.thinking？ ── 开关开 → yield THINKING
     │                        └ 开关关 → 丢弃并只告警一次
     ├─ message.content？ ──▶ yield DELTA
     └─ chunk.done？ ──▶ yield DONE {done_reason, eval_count, prompt_eval_count}
异常 ──▶ 超时/连接类 → 友好文案；其余 → "{异常类名}: {消息}"，统一 yield ERROR
finally ──▶ r.close()
```

**error 与 done 互斥**是整条链路最重要的生产侧承诺（L772-L779）。旧写法遇到 error 用 continue，
后面的 {"done":true} 照常产出 done，于是"被错误打断的残缺答案"同时满足"有 done + 非空"，被写进
会话历史——而页面只渲染了错误，那段答案用户一个字符都没见过。现在 error 分支直接 return，
消费侧（_run_assistant_turn 只在 done 时入历史）才有可靠依据。

另一条容易忽略的：**finally 里 r.close()**。这是 Stop 按钮能真正中断上游生成的基础（见 §8.4）。

## 4. 检索原语（L2872-L3030）

### 4.1 双通道召回

两条路各有入口：

- _qdrant_dense_hits（L2914-L2915）：query_points(using="dense")。
- _qdrant_sparse_hits（L2918-L2924）：query_points(using="sparse")，**必须持 _QDRANT_INFERENCE_LOCK**
  （L2435）。原因：qdrant-client 的本地推理器（ModelEmbedder）每客户端一份且内部无锁，带
  Document 的请求并发会 pop(0) 竞争（IndexError）或两份 Document 互换——稀疏查询拿另一路提问
  的文本去打分。写入口早就有锁，检索侧曾经裸跑，失败又被 tolerate 吞成"退化为纯稠密"，页面无提示。

dual_channel_recall（L2932-L2952）是"两条都发"的兜底编排：ThreadPoolExecutor(max_workers=2)
同时提交 dense/sparse，墙钟 = max 而非和；dense 失败会直接抛，sparse 失败在
tolerate_sparse_error=True 时转成一条 note。hybrid_search 在"稠密权重大于 0 且稀疏有内容"时
走更快的一条专用路径：先把稀疏召回丢到后台线程，主线程做嵌入+稠密召回，再取稀疏结果
（L3990-L4001）——墙钟从"嵌入 + 稠密 + 稀疏"降到 max(嵌入+稠密, 稀疏)。

**降级留痕**：_degraded（L2955-L2958）同时写日志和 steps["降级"]；_sparse_recall_note（L2927-L2929）
给出异常类型、消息与"分词 N 个 term"（N 是按空格切分的近似值，因为真实分词在 Qdrant 服务端，
Python 拿不到）。

### 4.2 加权 RRF（L2892-L2911）

```
score[d] = Σ_channel  w_channel / (k + rank_channel(d))
k = RRF_K（默认 60）；rank 从 1 开始
排序：分数降序；同分按 first_seen（通道顺序 dining→sparse，再按 channel 内名次）
资源标注：resources[d] = ["dense#3", "sparse#1", …]
```

参数在 L1089-L1097：RAG_RRF_K=60、RAG_RRF_DENSE_WEIGHT=1.0、RAG_RRF_SPARSE_WEIGHT=1.0；
两个权重同时为 0 会在启动时报错（L1092-L1093：两条召回通道都退出 = 检索恒为空）。
channel_rankings（L2961-L2962）只把权重非零的通道放进 rankings，所以权重为 0 的通道既不召回
也不参与融合。加权 RRF 不做分数归一化——两路原始分数量纲不同（余弦/点积 vs BM25），
按名次倒数融合正是为了绕开"原始分不可比"。

### 4.3 重排（L2873-L2889）

rerank_indices：tokenizer(query, doc) 成对编码，padding+truncation（max_length 默认 384），
按 RERANK_BATCH（默认 32）分批前向，取 logits.squeeze(-1) 当分数，全部候选统一打分后
argsort 降序，截 top_k。**统一打分**是一条硬不变量：HTTP 重排实现会强制校验服务端返回条数
等于候选数、index 不重复、不缺失（L3759-L3790），否则直接 RuntimeError——按子集排序会静默
破坏可比性。

候选文本由 rerank_text_for 决定（L2965-L2968）：
- RAG_RERANK_ON=child（默认）：contextual_text or child_text；
- RAG_RERANK_ON=parent：parent_text or child_text。
这个选择会同时影响 steps 里"text/tokens/被截断"三列，代码特意把 scored_field 一并写进诊断，
避免"展示 child_text、却按 parent_text 算 token"的自相矛盾（L4068-L4071）。

### 4.4 父块去重（L2971-L2980）

```
key = parent_id or parent_text or id        （三级回退）
首次出现 → kept；重复 → folded
```

目的不是"省条数"，而是**去重复正文**：上下文装配用的是整段 parent_text，同一父块下的兄弟子块
都进去会让同一段文字出现多次。TRACE_NOTES（L3576-L3579）进一步说明：返回条数少于 top_k 只可能
是"不同父块本身不够"，**不补同父子块**。去重开关 RAG_DEDUP_BY_PARENT 默认 True（L1098）。

### 4.5 置信度信号（L2983-L3002）

confidence_signal 只统计、不判定：top、median_rest（除 top 外的中位数）、spread=top-median_rest、
ratio=spread/(|top|+1)、mean、min、n、n_sources。空输入返回全 None + n=0。
results_confidence 用 rerank_score 与来源字段。**没有阈值、没有拒答**——这是明确的非目标：
拒答需要产品策略与标注，代码只负责把"检索质量如何"作为事实暴露给诊断。

### 4.6 集合别名（L3005-L3027）

USE_COLLECTION_ALIAS 默认 True（L1001）。别名解析成功才缓存（缓存的是别名字符串，查询时再路由到
当前集合，reindex 原子切别名后自动跟随）；失败则回退直连 COLLECTION_NAME，并把
state.alias_degraded 置 True，供 _qdrant_gate/cmd_stages 对外告警，避免"看起来正常、实际直读旧集合"。

## 5. RAGEngine.hybrid_search：一次检索的完整流程（L3959-L4097）

### 5.1 主流程

```
hybrid_search(query, top_k=TOP_K, return_steps=False, source=None, book=None)
 │
 ├─ _resolve_collection()
 ├─ _gate_check_collection_fingerprint   ← fail-closed 指纹闸门（§5.3）
 ├─ query_filter = _source_filter(source)（should: source/source/book 三键任一命中）
 ├─ sparse_text = normalize_aliases(…)（纯标点用零宽哨兵占位）
 ├─ _sparse_terms = sparse_terms_of(query)（只判"有没有内容"）
 │
 ├─ 分支 A：稠密权重>0 且 稀疏有内容
 │     ├─ 线程池：_qdrant_sparse_hits(limit=10, with_payload=True)
 │     └─ 主线程：embed_query → _qdrant_dense_hits(limit=10, with_payload=True) → 等稀疏
 │           稀疏异常 → note（不抛）；稠密异常 → 直接抛
 ├─ 分支 B：否则 dual_channel_recall(..., with_payload=True, tolerate_sparse_error=True)
 │
 ├─ rankings = channel_rankings(dense, sparse)          （权重非零才入选）
 ├─ fused = weighted_rrf(rankings, k=RRF_K, limit=RECALL_LIMIT×n_channels)
 ├─ chunks = 命中 payload 重建（chunk_from_payload）    ← payload 随召回一步交付
 │            （2026-10-30 起无独立按 id retrieve 往返；_fetch_chunks 已删除）
 ├─ candidates = 按 fused 顺序落地（命中必有 payload；"缺失"只剩 payload 缺字段兜默认）
 ├─ dedup_candidates_by_parent（可选）→ kept / folded
 ├─ 全体 kept 进重排（截断机制已整体退场，2026-10-30 两度拍板"不用截断"）
 ├─ reranked, rr_tok = rerank_backend.rerank(query, cand_texts, rr_k)
 ├─ ⏱️  检索分段 ← 三段耗时打一行日志（见下"逐段计时"）
 ├─ output = 按重排顺序组装父块结果（含 chunk_index/total_chunks）
 └─ confidence = results_confidence(output)；return output(+steps)
```

**逐段计时（2026-10-29 新增；2026-10-30 随 payload 合并调整口径）**：闸门之后、重排返回之后，
hybrid_search 把检索墙钟拆成三段，每次检索打一行
`⏱️  检索分段: 召回 Xms / 融合+落地 Yms / 重排 Zms（最大项: …）`。分段口径：

- **召回** = sparse_encode + 嵌入 + 双通道 + 随返 payload（天然并发，墙钟 = max(嵌入+稠密, 稀疏)）；
- **融合+落地** = RRF + 去重（默认不截断；payload 已并入召回段，无独立回取往返）；
- **重排** = ≤ 召回上界（RECALL_LIMIT×2 = 20）候选的一次批量前向（RerankBatch=32 内单批通过）。

**并发/拿数形态（2026-10-30 定型）**：并行全部是**阻塞线程**、共享同一个 QdrantClient——不用
asyncio：每轮只有 2-3 个长延迟 IO（嵌入 http + 双路 recall），线程是最短实现；Streamlit 单轮是
同步栈（台账/停止按钮都在线程里），改造整轮换不来墙钟收益。qdrant-client 能交给它的尽量
交给它：`query_points(with_payload=True)` 使 payload 随召回一步交付，按 id retrieve 整个退场。

**回取环节已并入召回（2026-10-30 实施）**：`with_payload=True` 让 payload 随 query_points 一次
交付；_fetch_chunks（按 id retrieve + 空回取告警）整体删除，"命中 id 回取缺失"（重建竞态）
这个状态点随之消失——失败矩阵第 8 行已改为"payload 缺字段兜默认"。旧线索"回取与重排间可
并发"也随之关闭：回取不再是独立往返。

每一步的输入输出：

| 步 | 输入 | 输出 | 失败/降级 |
|---|---|---|---|
| 指纹闸门 | collection | 通过（置 checked）或 RuntimeError | 读集合失败/网络抖动 → 暂缓；不一致 → 抛 |
| 稀疏查询编码 | query 原文 | Document(归一文本, bm25 options) | 纯标点 → 哨兵；无内容 → 不参与融合 |
| 双通道召回 | dense 向量、Document、RECALL_LIMIT=10、filter | Qdrant 命中列表（**payload 全字段随召回**） | 稀疏失败 → note+dense 继续；稠密失败 → 抛 |
| RRF 融合 | 两路 (id, payload) 序列 + 权重 | [(point_id, score, resources)]，上限 = RECALL_LIMIT×通道数 | 无（纯函数） |
| 命中重建 | 命中 payload | {point_id: chunk dict}（chunk_from_payload） | payload 缺字段 → 兜默认（同旧 retrieve 行为） |
| 父块去重 | candidates 顺序 | kept / folded | 关闭时 folded=[] |
| 汇总（截断步已删） | — | 去重后的全体候选直接进重排，无池收口 | — |
| 重排 | query + 候选文本 + rr_k | [(下标, 分数)], tokenizer | torch/HTTP 异常 → 抛出（无稠密兜底） |
| 汇总 | reranked | 结果列表 ≤ top_k | 少于 top_k → log（不补同父子块） |

### 5.2 选型漏斗（默认配置的数量关系）

```
每个通道召回 limit = RECALL_LIMIT（.env 显式 = 10）
   dense: 10 ─┐  （payload 全字段随召回交付 → 无独立回取往返）
              ├─ RRF(k=60, w=1.0/1.0) → 排序
   sparse: 10 ┘
         │ 融合上限 = RECALL_LIMIT × n_channels（双通道 → 20）
         ▼
    ≤ 20 条融合候选 → 命中 payload 重建（chunk_from_payload）
        │ 父块去重（同父只留 RRF 名次最好的一条）
        ▼
   kept ≤ 20 → **不截断，全体 kept 进重排**（2026-10-30 两度拍板"不用截断"，机制整体删除）
        │ 重排全体 kept（RERANK_ON=child → contextual_text）
        ▼
   rr_k = min(top_k=10, kept) → 按重排分数取前 10 条父块（kept<10 时按实返回）
```

**为什么融合上限 = RECALL_LIMIT × n_channels**：上限只做一件事——把"本轮召回的命中"整体
保留。超过它不可能再多融合出东西（每通道最多 RECALL_LIMIT 个 id）；旧值 pool×n_channels
附带的"为截断预留"职责，随截断退场（2026-10-30）消失。

**截断为什么退场**：旧候选池（=RERANK_TOP_K=9）把"父块去重后 RRF 名次 10+"的候选挡在重排
之外——它们永远拿不到重排分数，错杀的召回无法回补。重排算力已外置（infinity 同实例双模型），
20 条在 cuda ≈0.3s / mps ≈0.84s，截断省下的秒数买不回错杀的召回。2026-10-30 用户两度拍板
"不用截断"后，截断/预算收缩机制（含显式入口 RAG_RERANK_BUDGET_MS）**整体删除**，代码里
"去重之后 → 重排"之间不再有任何减法。torch 降档若嫌慢：调小 RAG_RECALL_LIMIT 或换回
http，而不是恢复截断。

### 5.3 两道指纹闸门（fail-closed）

- **_check_chunk_freshness（L3823-L3853）**：索引里记录的 chunk_fingerprint 必须等于当前
  "切分配置 + 源文本 sha"算出的指纹。索引没记指纹（旧索引）= 无从判断 ≠ 一致 → 直接抛。
  只查一次，**通过才置位**；失败每轮重试，直到重建对齐。
- **_check_lexicon_consistency（L3854-L3888）**：索引 lexicon_id 必须等于当前词表指纹；旧索引
  未记录 → 抛。网络类读取失败（ConnectionError/TimeoutError 或消息含 connect/timeout）→
  只告警、暂缓核对（状态未知 ≠ 已知不一致），下一轮再试；其余异常照抛。
- **_gate_check_collection_fingerprint（L3889-L3921）**：开跑前 scroll 一条 payload 取分块指纹；
  词表指纹必须**全量滚动**（sparse 原地更新会留混合态，首点恒属已更新前缀），发现多个指纹
  = 混合态 → 抛"sparse 中断残留，请重建"。

这三处的共同设计是：**fail-closed**。对不上或无从判断就显式报错，绝不"暂缓核对"地继续检索——
否则用户拿到的是无声的旧知识库，而 stages 虽然报过期却没人看。

### 5.4 steps 诊断字典

return_steps=True 时，hybrid_search 返回 (results, steps)。键名定义在 L3565-L3575：

| 键 | 内容 |
|---|---|
| 稀疏查询文本 (服务端分词) | 归一后的稀疏文本（L3978） |
| 单通道召回 | [{channel, rank, point_id}] 的平铺表（L4010） |
| RRF 融合 | [{resource, score, id, text}]（L4021） |
| 候选落地分块表 | [[point_id, id]]（L4036） |
| 父块去重 | {enabled, 去重前候选, kept, folded, folded_ids, 说明}（pool_trimmed/截断键已随机制删除） |
| Rerank 重排 | [{id, fused_rank, score, tokens, text, scored_field, 送入上限, 被截断}]（L4064-L4076） |
| 检索汇总 / 检索汇总条数 / 置信度 | 结果列表、数量对照、confidence_signal（L4092-L4094） |
| 降级 | _degraded 追加的降级说明列表 |

诊断的取向是"**只报实际发生的事**"：text 就是送进 rerank 的那段文本；"被截断"在无法计算时写
None 而不是 False（False 会被读成"没被截断"，是另一种静默，L4073-L4075）。

### 5.5 空结果出口

_empty_search_result（L3592-L3596）：只写置信度（空列表的 confidence），按 return_steps 形态
返回。可能到达它的路径有三条：融合后为空、回取后无候选、去重后无候选（空查询守卫已移除，
空/纯标点查询走同一出口）。它们都返回空
而不是抛错——检索为空是合法业务状态，交给上层决定"照常生成/纯对话"。

## 6. 上下文装配与 token 预算（L3030-L3599）

### 6.1 四种资料格式（表驱动）

_context_formatter（L3106-L3114）按配置选一个 formatter，优先级 meta > tags > labels > plain：

| 策略 | 触发 | 单条资料形态 | 代码 |
|---|---|---|---|
| meta（默认） | RAG_CONTEXT_META=1 | <document index="n" source="…" id="…" chunk="i/t" score="0.91">三 parent_text 三</document> | L3076-L3079 |
| tags | META=0、TAGS=1 | <document>三 [n]（来源：X）三 正文 三</document> | L3087-L3091 |
| labels | META=0、TAGS=0、LABELS=1 | [n]（来源：X）三 正文 | L3082-L3084 |
| plain | 全关 | 仅 parent_text | L3094-L3095 |

meta 档的元数据用 XML 属性给出（模型最易解析），属性值走 xml.sax.saxutils.quoteattr 转义，
缺字段省略对应属性；index 恒有（资料清单与引用回指靠它）。score 保留两位小数。本仓
.env 是 META=1、LABELS=0、TAGS=0（.env:L111/L119/L121），即生效 meta 档。

两个与格式正交但更重要的不变量：
1. **不发 system prompt**：plan_generation 里 system 恒为 ""（L3526），build 时也不会产出
   system 消息（L3494-L3495）。原来的回答规则与领域身份句已整体移除。
2. **资料恒以 user 角色承载**（L3491-L3498）：RAG_CONTEXT_ROLE 已删除。这消除了"排在历史之后
   的 system 消息被 chat template 静默丢弃"这条失效路径；同时 <document> 标签承担"这是数据
   不是指令"的防注入职责（L3035-L3038）。

### 6.2 token 计数的三级降级（L3162-L3373）

所有生成装配的预算/裁剪/硬上界**只经过 estimate_tokens 一个入口**（L3359-L3373），避免"预算
按 A 算、裁剪按 B 算"。计数来源按优先级：

```
① RAG_GEN_TOKENIZER_PATH 指定的 tokenizer.json / HF 目录
      ├─ 若同目录 .meta.json 记录 gguf_sha256 与当前 MODEL 的 GGUF blob 不符 → 不用（预算会错算）
      ├─ tokenizer_sha256 校验失败 → 不用
      └─ 加载后先编一个冒烟串，失败也降级
② 按 MODEL 的 Ollama manifest 定位 GGUF，用 llama-cpp-python(vocab_only=True).tokenize
③ 保守估算：CJK 字符(>=U+2E80) 1 字 = 1 token，其余按 (len+3)//4
```

第②级的 _LlamaCppTokenizer（L3194-L3209）直接用 llama.cpp 的 tokenizer，与生成端逐字节一致，
add_bos=False 与 tokenizers 路径的 add_special_tokens=False 对齐。第③级刻意偏高（不越界但少用
窗口）。懒加载带锁且**失败缓存不重试**（_get_gen_tokenizer，L3343-L3356）：一旦确认这台机器没有
可用 tokenizer，就不必每次生成都重试一遍加载。

### 6.3 预算公式（plan_generation，L3525-L3562）

```
I = num_ctx − GEN_INPUT_RESERVE                 输入上界（I ≤ 0 直接 ValueError）
B = I − T(system) − T(context) − T(query)       给历史的最大预算
若 GEN_HISTORY_MAX_TOKENS > 0：B = min(B, GEN_HISTORY_MAX_TOKENS)
over_window = T(system) + T(context) + T(query) > I
```

本仓 .env 的 RAG_INPUT_RESERVE=0，所以 I = num_ctx = 32768；GEN_HISTORY_MAX_TOKENS=0 表示不额外
限制历史（只受窗口余量约束）。注意"输入窗口全部给输入、不做输出预留"（L590-L592、L3532-L3534）：
输出超窗由服务端上下文移位从最前面按 token 遗忘，而不是在客户端预留 num_predict 的空间。

启动期与调用期各有一道交叉校验（对称设计）：
- 启动：NUM_CTX <= GEN_INPUT_RESERVE → 抛（L3151-L3155）；
- 调用：外部传入的 num_ctx 扣除 reserve 后 ≤0 → 抛（L3538-L3542）。
没有这两道校验的后果是"装配截出空提问、模型收到空输入、启动时却没人报错"。

### 6.4 裁剪优先级（build_generation_messages，L3405-L3504）

```
1) 历史整轮淘汰：
   一轮 = 一条 user + 其后的连续 assistant；从最旧的一轮开始整轮丢，绝不 token 级切半句
   （_truncate_history_to_budget，L3380-L3402）；队首恒为 user
2) 硬上界覆盖整条 prompt：
   历史先按 body_cap = I − T(system) − T(context) − T(query) 再 FIFO 一遍
3) 仍超：按 FIFO 依次截
   system 前缀截断 → 资料先整条从尾部删、再对拼接后的资料做 token 前缀截断 → 最后才截提问前缀
   （提问永不整条删除，保证最后一条 user 消息存在）
4) 装配：[system] + 历史 + [资料(user)] + [提问(user)]
5) 终检：T(整条) > I → RuntimeError（用 raise 而非 assert，-O 下不失效）
```

几处值得单独说明：

- **资料块整条丢而不是先截 token**：context_blocks 与模型实际看到的拼接单元一致，截掉哪几条，
  sources/context_docs_kept 就只报哪几条（L3117-L3126、L3455-L3457）。
- **判据必须用真实拼接计数**：estimate_tokens 不可加（BPE 会跨边界合并，估算档还逐块取整），
  旧写法用加性运行计数当判据再与实际拼接比对后 raise，只要有 ≥2 条资料且触发裁剪就必自崩
  （L3460-L3467）。现在每步重算一次 join+估算——候选 ≤ 20 条，成本可忽略。
- **重复提问剥干净**：调用方可能把本轮提问也塞进 history，build 里循环调用 strip_current_turn
  直到不再命中（L3408-L3413），否则模型会收到同一问题两份。
- **统计字段**：dropped_messages、used_tokens、context_tokens、budget、truncated/truncated_user
  （恒 False，历史残留）、input_cap、cap_trimmed、frame_truncated、context_truncated、
  context_partial、context_docs_kept、query_truncated（L3504）。plan_generation 再补上
  query_tokens、over_window、prompt_tokens_est（L3553-L3559）。

### 6.5 硬上界不变量

build 末尾的 message_tokens(out) > cap → RuntimeError（L3500-L3503）。它不是"防御性 assert"，
而是"装配 bug 的显式暴露"：估算口径与装配同源，正常路径恒成立，越界一定是代码错。
同理 _cut_to_tokens（L3507-L3522）用二分找最长前缀，max_tokens≤0 直接返回空串。

## 7. 适配器协议（L3599-L3680）

| Protocol | 实现 | 启用条件 | 行为/失败 |
|---|---|---|---|
| EmbeddingBackend.embed_query | TorchEmbeddingBackend（L3610-L3625） | RAG_EMBED_BACKEND=torch | 取模型 → 进推理锁 → 锁内复核引用 → embed(is_query=True)（内层拼 BGE 指令前缀，L1268-L1271） |
| 同上 | HttpEmbeddingBackend（L3628-L3635） | RAG_EMBED_BACKEND=http | POST /v1/embeddings（实现收口到模块级 `http_embeddings`，对拍探针 embedpairs 共用一份）；查询指令前缀客户端拼；**合同三查 fail-closed**：返回条数=输入、自报 model=配置名、维度=本地 config 声明（2026-10-30 起，第三查拦"同名换模型/换维度"） |
| RerankBackend.rerank | TorchRerankBackend（L3638-L3652） | RAG_RERANK_BACKEND=torch | 与稠密前向共用推理锁；@_timed("rerank", 5s) |
| 同上 | HttpRerankBackend（L3655-L3668） | RAG_RERANK_BACKEND=http | /rerank（同一 infinity 进程的另一模型），客户端排序截断；并行加载 tokenizer 隐藏时间；预热期打一发单候选真探针 |
| Chat | OllamaChatBackend（L3671-L3675） | 恒 | stream() 即 stream_chat |

设计意图：**设备/后端差异只在这里选一次**（L3698-L3704），hybrid_search 只有一条路径。torch 的
嵌入/重排共用一把 _model_infer_lock（MPS 单模型不做真并行），但加载用另一把 _model_init_lock——
加载慢、推理短，共用一把会拖慢并发查询（L3694-L3697）。

BGE 查询指令默认是空串（RAG_EMBED_QUERY_INSTRUCTION，L1100；本仓 .env:L123 也为空），
即两条嵌入路径当前都走裸查询嵌入——这是刻意的 prompt 观：不把模型训练期的指令模板混进用户查询。

HTTP 接口的契约比 torch 严：
- _http_embed（L3804-L3820）校验返回向量条数 == 输入条数，否则 RuntimeError；
- _http_rerank（L3759-L3790）请求 top_n=候选总数，并校验条数、index 齐全、不重复、不缺失。
  服务端少给一条就直接报错——按子集排序会静默破坏"对全体候选统一打分"。

## 8. run_turn：单轮编排（L4100-L4216）

### 8.1 事件契约

封闭词表（L651-L656）：retrieval → thinking / delta / error / done。
- 首事件恒为 retrieval（run_turn 在检索后立即 yield，L4156），消费方必须用 first_retrieval 取走；
- `_stream_chunks`（原 `drain_events`，2026-10-30 并入 `_stream_reply`）对词表外 kind 直接抛错；
- 只有 done 携带 done_reason/eval_count/prompt_eval_count，并在转发前补上 answer_chars/elapsed
  （L4167-L4175）；answer_chars 只累计 delta 字符数。

### 8.2 单轮步骤

| 步 | 行为 | 行 |
|---|---|---|
| 参数归一 | source/book 兼容；chat 缺省 _CHAT_BACKEND.stream；top_k/use_context/num_ctx 取全局；num_predict≤0 → -1 | L4136-L4144 |
| 检索 | engine.hybrid_search(query, top_k, source) | L4149 |
| 置信度 | results_confidence(results) | L4150 |
| 装配 | plan_generation(results, history, query, num_ctx, use_context) | L4152 |
| 首事件 | yield RETRIEVAL {query, results, confidence, gen, elapsed} | L4156 |
| 生成 | for kind, delta in chat(plan["messages"], num_ctx=…, num_predict=request_predict) | L4164-L4178 |
| 台账 | status = done / error / stopped；非 stopped 写一行 | L4181-L4215 |

注意几个"看似多余"的选择：

- **use_context=False 仍会检索**：对照模式只决定资料是否注入（plan_generation），不跳过检索。
  这保留了"同一轮对照打开资料/关闭资料"的诊断能力，代价是仍有一次检索成本。
- **检索发生后才 yield retrieval**：所以"检索慢"的等待发生在首事件之前，前端看到的是空白而不是
  部分结果；这是简化的取舍，诊断数据（elapsed）随首事件一起给出。
- **同参数双传**：num_ctx/num_predict 既进 plan_generation 算预算，也进 chat 请求（L4164）。
  注释点明，若只传一边会出现"预算按 A 算、请求按 B 发"的静默分叉（L730-L734）。

### 8.3 与 UI 层的分工

run_turn **不做**：不写会话历史、不判拒答、不重试、不设长度/时间闸（除 stream_chat 的两道）。
它只保证"事件如实转发 + 台账如实记录"。UI 层（_run_assistant_turn，L4796-L4868）负责：
- rollback_orphan_user：失败时撤掉本轮 user，否则下一轮会把孤儿提问当历史重发（同题答两遍）；
- close_events：Stop/异常时主动关闭事件流，沿生成器链把 GeneratorExit 打进 stream_chat，
  其 finally 的 r.close() 让 Ollama 端看到连接关闭（否则断开不确定，单槽 Ollama 会继续烧完）；
- answer_is_good：无 error、有 done、正文 trim 非空才入历史。**done_reason=length 仍算成功**
  （只要正文非空），只额外给一条"被 num_predict 截断"的 warning（L4777-L4784）——这是一个
  有意的产品取舍：截断答案是"不完整但可用"，与"错误/空答"不同级。

### 8.4 Stop 断流路径

```
用户点 Stop
   ▼
Streamlit 抛 ScriptControlException（BaseException 子类，不是 Exception）
   ▼
_run_assistant_turn 的 except BaseException     L4845-L4853
   ├─ close_events()      ← 立即断开上游生成
   ├─ rollback_orphan_user()
   └─ raise               ← 原样上抛，不吞掉"停止"信号
   ▼
run_turn 生成器被 close → 在 yield 处抛 GeneratorExit → 台账代码不执行
   ▼
stream_chat 的 finally → r.close()
```

这也解释了台账里 status="stopped" 分支：正常 Stop 根本走不到那里（生成器被 close，尾部代码
不执行），"stopped" 是给"流自然结束但没有 done 也没有 error"这类情况留的兜底。

## 9. 单轮台账（L1034-L1058、L4179-L4215）

默认关闭，RAG_TURN_STATS=1 时每轮追加一行 JSON（固定字段、追加式，写失败只告警不抛：
诊断路径不许崩，L1044-L1058）。本仓 .env 已开启（.env:L343）。

| 字段 | 来源 | 用途 |
|---|---|---|
| ts | time.time() | 行时间 |
| status | done / error / stopped | 轮次结局 |
| num_ctx / request_predict | 本轮请求参数 | 复现实验 |
| history_turns | len(history) | 历史规模 |
| retrieval_elapsed / gen_elapsed | 计时 | 阶段耗时 |
| prompt_tokens_est | plan_generation 算好的 stats | 装配侧估算 |
| history_tokens / context_tokens / query_tokens | stats | 预算分解 |
| over_window | stats | 是否超窗 |
| answer_chars | delta 累计 | 输出规模 |
| prompt_eval_count / eval_count / done_reason | Ollama done | 服务端事实 |
| error | 截断到 300 字符 | 失败原因 |

台账最关键的用法是 **prompt_tokens_est 与服务端 prompt_eval_count 的比值**：≈1 说明只有本轮新增
段被 prefill；≫1 说明整段重算（README §11.4 的"重载税"就靠这一对观测，L1037-L1039、L3556-L3559）。
两个数字分别由装配侧与服务端产出，互为对账，不是同一来源抄两遍。

## 10. 失败矩阵

下表按链路顺序列出可预期的失败点。"历史/台账"列说明该失败对本轮状态的影响。

| # | 失败点 | 触发条件 | 代码行为 | 用户可见 | 历史/台账 | 恢复 |
|---|---|---|---|---|---|---|
| 1 | 空/纯标点提问 | 归一后无字母数字（守卫已移除，直接透传） | 稀疏零项 → 退化纯稠密，写降级；稠密权重为 0 则完全无召回 | 可能答非所问 | 正常轮 | 换问法 |
| 2 | 稀疏召回异常 | Qdrant/推理器异常 | note 进 steps["降级"]，保留稠密结果 | 无提示（仅日志/诊断） | 正常轮 | 下一轮重试 |
| 3 | 稠密嵌入异常 | torch OOM / HTTP 5xx | 异常上抛 | UI 报"本轮执行异常"+调用栈 | 无历史、无台账 | 修环境 |
| 4 | 分块指纹不一致/旧索引 | 索引与当前切分不符 | RuntimeError（fail-closed） | UI 报错并提示 process+index | 无历史、无台账 | 重建后重试 |
| 5 | 词表词面归属冲突 | 同一词面归属多个规范名（别名表逻辑冲突） | 加载即 RuntimeError（fail-closed，与 `lexicon` 自检共用 `_alias_owner_conflicts` 分类）；文件缺失/编码坏仍是资源类"空词表降级" | UI 报错 | 无历史、无台账 | 修表（删冲突词面/合并组） |
| 6 | 词表指纹网络抖动 | Qdrant 瞬断/超时 | 告警 + 暂缓核对，本轮继续 | 无提示 | 正常轮 | 下轮再核 |
| 7 | 集合别名解析失败 | get_aliases 异常 | 直连 COLLECTION_NAME + alias_degraded | 无提示 | 正常轮 | 自动 |
| 8 | payload 缺字段 | 旧索引 schema 漂移/极端坏档 | chunk_from_payload 兜默认值（文本可能为空串）；payload 随召回返回后，"按 id retrieve 缺失"这个状态点已消失 | 可能差资料 | 正常轮 | 重建 |
| 9 | 重排异常 | torch OOM / HTTP 契约不符 | 异常上抛（无稠密兜底） | UI 报错 | 无历史、无台账 | 修环境/换后端 |
| 10 | 检索结果为空 | 语料空/零命中/被过滤 | 照常生成（纯对话） | 正常回答 | 正常轮 | — |
| 11 | 生成 HTTP 非 200 | 模型缺失/服务异常 | auto+设备类错 → 去 num_gpu 重发一次；否则 EVENT_ERROR | st.error + 下一步提示 | 不入历史、台账 error | 修服务/拉模型 |
| 12 | 流内 error 块 | Ollama 报错 | yield ERROR 后 return（不产出 done） | st.error | 不入历史、台账 error | 重试 |
| 13 | 读空闲超时 | 300s 无新数据 | EVENT_ERROR"生成超时/连接中断" | st.error + 提示 | 不入历史、台账 error | 调 RAG_STREAM_READ_TIMEOUT |
| 14 | 整轮超时 | 有 chunk 但超 1800s | EVENT_ERROR | st.error | 不入历史、台账 error | 调 RAG_TURN_TIMEOUT |
| 15 | 空答案（有 done） | 预算被思考吃光等 | done 照常产出，answer 为空 | st.markdown"⚠️" + st.warning"无正文" | 不入历史、台账 done（answer_chars=0） | 关 think/调预算 |
| 16 | done_reason=length | 触 num_predict | 正常 done | st.warning"被截断" | **入历史**（正文非空）、台账 done | 调 num_predict |
| 17 | Stop 按钮 | Streamlit 抛 BaseException | close 事件流 → 回滚 user → 上抛 | 停止（write_stream 已流出的部分保留） | 无历史、无台账 | — |
| 18 | 未预期异常 | 任意 bug | traceback 落盘 + 页面给 file:line | st.error | 无历史、无台账 | 修 bug |
| 19 | 台账写失败 | 磁盘/权限 | 只告警 | 无 | 正常轮 | 修磁盘 |
| 20 | tokenizer 不可用 | 三级全失败 | 退估算计数（偏高） | 无 | 正常轮 | 装 tokenizer |
| 21 | tokenizer 与模型不符 | meta sha 不符 | 不用旧 tokenizer，退 GGUF/估算 | 无 | 正常轮 | 重建 tokenizer |
| 22 | 装配越硬上界 | 装配 bug | RuntimeError | UI 报错 | 无历史、无台账 | 修 bug |

生成失败的"下一步提示"是表驱动的（L4687-L4728）：404 模型缺失、OOM、HTTP 500、连接失败
四类，按顺序匹配、具体判据在前。这张表本身也修过一次经典 bug：旧顺序把 "connection" 排在
"http 500" 之前，含 connection 字样的 500 会被误报成"连不上生成服务"；且整表漏了 404，
而 entrypoint 故意把模型拉取放后台，冷启动期 404 恰恰是最可能的一类错误。

## 11. 设计取舍与不变量

1. **反静默是最高优先级**。退化必须同时写日志与 steps；无法计算的诊断列写 None 而不是 False；
   fail-closed 闸门宁可报错也不放行过期索引（L2955-L2958、L3823-L3888、L4073-L4075）。
2. **error 与 done 互斥**（L786-L792）：这是"失败不入历史"能成立的前提；生产侧一旦违约，
   消费侧还有 answer_is_good 的第二道判据（L4994-L4999，唯一调用点 L5062）。
3. **无重试、无长度闸**（L608-L609、L736-L739）：窗口是唯一上限，超窗交给服务端上下文移位；
   只把"卡死"变成显式错误（读空闲 + 整轮两道上限）。重试会带来"已生成内容怎么办"的语义问题，
   而 GPU 回落被严格限定在"零 token 产出之前"，因此不算重试。
4. **质量一致性优先于延迟**：候选池截断机制整体退场（2026-10-30 两度拍板"不用截断"，重排
   吃下全体去重后候选，无静默也无显式的减法）。同参同果比"容器上偷偷变快"更重要。
5. **预算先算后装、单一计数入口**：plan_generation 先算 B，build_generation_messages 再按同一
   算式兜底硬上界，estimate_tokens 是唯一计数器（L3359-L3363）。
6. **历史按整轮淘汰、锚点按 token 截断**：半个提问/半个回答没有语义价值；资料按整条丢是为了
   sources 对齐；提问只截前缀、永不整条删（L3380-L3402、L3444-L3489）。
7. **资料不寄生 system**：system 恒空、资料恒 user（L3140-L3143、L3491-L3498），<document>
   标签是唯一的"数据非指令"防线。
8. **检索结果不缓存**：payload 随召回交付（with_payload=True），chunk_from_payload 就地重建，
   进程内不留分块表（2026-10-30 起，原 _fetch_chunks 按第二次按 retrieve 已删）——既不占
   常驻内存，也不存在"索引重建后旧缓存失步"的状态点。
9. **一轮失败不污染会话**：错误/空答回滚本轮 user、不写 assistant（L4800-L4864）；Stop 主动
   断开上游而不是等 GC。
10. **适配器隔离后端**：torch/http 的差异在 L3 收敛，L4 只有一条路径；HTTP 契约额外强制
    "全候选统一打分"（L3698-L3704、L3759-L3790）。

## 12. 本仓库 .env 的在线链路生效值（速查）

| 项 | 值 | 出处 |
|---|---|---|
| 生成模型 | minicpm5-2b | .env:L76 |
| think | 1（开） | .env:L190 |
| num_ctx | 32768 | .env |
| num_predict | 0 → 请求 -1 | .env:L205 |
| input_reserve | 0（I = num_ctx） | .env:L217 |
| temperature | 0.0 | .env:L221 |
| top_k / top_p / min_p | 20 / 0.95 / 0 | .env:L224-L226 |
| presence / frequency / repeat | 0.0 / 0.0 / 1.1 | .env:L231-L233 |
| repeat_last_n | 128 | .env:L236 |
| seed | -1 | .env:L238 |
| RAG_TOP_K | 未设 → 10（.env 显式 10） | 默认 L847 |
| RAG_RERANK_TOP_K | 20（键已从 .env 移除） | 候选池上限退场后仅作 RECALL_LIMIT 缺省跟随源 |
| RAG_RECALL_LIMIT | 10（显式；payload 全字段随召回） | .env |
| RRF k / 权重 | 60 / 1.0、1.0 | 默认 L1089-L1091 |
| RAG_DEDUP_BY_PARENT | 未设 → True | 默认 L1098 |
| RAG_RERANK_ON | 未设 → child | 默认 L894 |
| 嵌入/重排后端 | http / http（同一 infinity 双模型） | .env:L323、L326 |
| 上下文格式 | META=1、LABELS=0、TAGS=0 → meta 档 | .env:L111、L119、L121 |
| 查询指令前缀 | 空（裸查询嵌入） | .env:L123 |
| 历史硬上限 | 0（不设） | .env:L210 |
| 生成 tokenizer | artifacts/gen_tokenizer | .env:L249 |
| num_gpu | auto（GPU 优先、失败一次回落） | .env:L332 |
| 读空闲 / 整轮超时 | 300s / 1800s | .env:L256、L257 |
| 单轮台账 | 开（artifacts/turns.jsonl） | .env:L343 |

按这组值，一次默认问答的形状是：召回 9+9 → RRF 取 18 → 去重后截到 9 → 重排取 5 → 以 meta 档
资料块装配进约 13 万 token 的窗口 → 贪心采样、输出不限长、think 与正文共用预算。

## 附：本文未覆盖的部分

分块/词表/嵌入缓存/建索引（task-2）、配置/日志/硬件画像/驻留策略/UI 外壳与 CLI（task-4）、
业务定位与数据资产（task-1）。在线链路与它们的接口只有四个：词表指纹与分块指纹（读）、
Qdrant 集合与 payload 契约（读）、Ollama /api/chat（写请求、读流）、单轮台账文件（追加写）。
