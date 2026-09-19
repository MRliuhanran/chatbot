"""检索评测执行器 —— 供 L3 回归测试与基线录制脚本共用。

与测试数据（probes.py）分离：probes.py 保持纯数据 + 纯函数，
本模块负责"把探针跑过完整检索链路"这段有副作用的逻辑。

三类探针各有自己的判定与聚合，**不合成一个总分**：
  positive   单轮正向（短查询 + 长问句）→ 召回与排序质量
  multiturn  多轮指代问句              → 查询改写是否真的起作用
  negative   域外/库中不存在的问题      → 拒答信号（分层：检索给信号、生成做裁定）
分母不同的比率相加没有意义，混在一起还会掩盖"长问句更差"这类分层事实。
"""

import chatbot as RE
from chatbot import CONCAT_ENABLED, REWRITE_ENABLED
from tests.probes import (
    MULTITURN_PROBES,
    NEGATIVE_PROBES,
    PROBES,
    SHORT_PROBES,
    aggregate_all,
    evaluate_multiturn,
    evaluate_negative,
    evaluate_probe,
    score_profile,
)


def _record(results, verdict=None, extra=None):
    """把一次检索的原始事实与判定打包成一条可存档的记录。

    books/top1_score/score_profile 都留档，理由是：指标只能告诉你"退化了"，
    而这些原始字段能告诉你"退化成了什么"。零容差基线一旦报警，需要立刻
    分辨"排序变了"还是"整批召回都变了"，重新跑一遍是来不及的。
    """
    rec = {
        "books": [r.get("book", "") for r in results],
        "top1_score": (results[0].get("rerank_score") if results else None),
        "n_results": len(results),
        "n_books": len({r.get("book", "") for r in results}),
        "score_profile": score_profile(results),
    }
    if verdict:
        rec.update(verdict)
    if extra:
        rec.update(extra)
    return rec


def run_probes_with(searcher, top_k=None, probes=None, record_extra=None):
    """跑一组正向探针并把结果记账 —— **"跑探针"的唯一实现**。

    searcher(query, top_k) -> (results, steps)：
        评测方把"检索"这一步注入进来。L3 与 record_baseline 传
        `RAGEngine.hybrid_search`；chatbot.py 的 ab-retrieval 子命令传一个额外计时的
        包装（它要报检索毫秒数与 rerank token 数）。
    record_extra(probe, results, steps, record)：
        就地补充该工具特有的字段（名次、token 数……）。
        存在的理由：ab_retrieval 与 eval_runner 此前各写一份"遍历探针 →
        调 hybrid_search → evaluate_probe → 记账"的循环，两份循环对
        "什么叫一次评测"的回答可以不同（实际也各不相同）—— 而 A/B 与
        基线必须只在**配置**上不同，不能在度量方式上不同。
    """
    top_k = top_k or RE.TOP_K
    out = {}
    for probe in (probes if probes is not None else PROBES):
        results, steps = searcher(probe.query, top_k)
        rec = _record(results, evaluate_probe(probe, results))
        if record_extra is not None:
            record_extra(probe, results, steps, rec)
        out[probe.query] = rec
    return out


def run_probes(engine, top_k=None, probes=None):
    """对正向探针跑一遍完整检索链路（L3 / 基线录制的入口）。

    返回 {query: {book@1, book@k, kw@k, books, top1_score, …}}。

    用 RAGEngine.hybrid_search（而不是自己拼 Qdrant 查询），保证评测链路与
    线上完全一致 —— 否则测的是一个线上不存在的检索器。
    """
    def searcher(query, k):
        return engine.hybrid_search(query, top_k=k), None

    return run_probes_with(searcher, top_k=top_k, probes=probes)


def run_multiturn(engine, top_k=None):
    """对多轮探针跑一遍 —— **带 history**，因此会触发查询改写。

    这是与 run_probes 的唯一实质差别：多轮探针考的是"指代消解能否把
    '他最后结局如何' 变成可检索的独立问句"，不传 history 就考不出任何东西。
    改写未启用或 Ollama 不可达时，rewrite_query 会退回原查询（不抛异常），
    此时该组指标退化为"字面检索多轮问句"，快照里用 rewrite_enabled 标出来，
    免得把"改写没跑"误读成"改写没用"。
    """
    top_k = top_k or RE.TOP_K
    out = {}
    for probe in MULTITURN_PROBES:
        results = engine.hybrid_search(
            probe.query, top_k=top_k, history=list(probe.history)
        )
        out[probe.query] = _record(
            # normalize=RE.normalize_aliases：topic@k 必须认别名，否则"关羽/云长"
            # 这类同人异名会记成假失败（见 evaluate_multiturn 的说明）。
            results, evaluate_multiturn(probe, results, normalize=RE.normalize_aliases),
            # kind 必须留档：事后做分层分析（aggregate_multiturn_by_kind）时，
            # 快照里没有它就只能去翻源码里探针当下的定义，而探针会被改动。
            extra={"history_turns": len(probe.history), "kind": probe.kind},
        )
    return out


def run_negative(engine, top_k=None):
    """对负样本探针跑一遍，记录"会不会拒答"的现状。

    **判据取线上那一套**（`chatbot.confidence_signal`），而不是探针模块自带的
    rel_gap 口径：本仓库已经实测过 rel_gap 在这套栈上**零区分度**
    （正例与负例均值几乎相同），用它聚合出的 refuse_rate 恒为 0，
    会让人以为"完全没有拒答能力"，而线上其实有（实测 6/10）。
    评测必须度量线上真正在用的判据，否则测的是另一个系统。
    rel_gap 的结果仍以 `rel_gap_refuse` 留档，供对照与复盘。
    """
    top_k = top_k or RE.TOP_K
    out = {}
    for probe in NEGATIVE_PROBES:
        results, steps = engine.hybrid_search(probe.query, top_k=top_k,
                                              return_steps=True)
        conf = steps.get(RE.STEP_CONFIDENCE) or {}
        rec = _record(results, evaluate_negative(probe, results), extra={
            "rel_gap_refuse": (evaluate_negative(probe, results) or {}).get("refuse"),
            "engine_refuse": conf.get("refuse"),
            "engine_reason": conf.get("reason", ""),
            "engine_mean": conf.get("mean"),
            "engine_n_books": conf.get("n_books"),
        })
        # 以引擎信号为准覆盖 refuse —— aggregate_negative 读的就是这个键
        rec["refuse"] = bool(conf.get("refuse"))
        out[probe.query] = rec
    return out


def aggregate(per_probe):
    """把正向逐条结果折算成三个比率（向后兼容的旧入口）。"""
    return aggregate_all(positive=per_probe)["positive"]


def multiturn_degrade_state(rewrite=None, concat=None):
    """本次运行里多轮检索**实际会落到哪一档**（纯函数，可 L0 单测）。

    降级链是 LLM 改写 → 拼接上轮用户问句 → 字面原句（见
    build_retrieval_query）。这里给出的是**上限**：
    真实一轮里 LLM 可能调用失败而掉到下一档，但"配置允许的最优档"
    是确定的，而基线与当前运行的比较必须以配置为准 —— 否则同一次
    比对里两边可能跑在不同的链上。

    返回值：
        "llm"     LLM 改写可用（改写关掉时不存在这一档）
        "concat"  LLM 不可用，但拼接档开着 —— **改写开关为 0 时的常态**
        "literal" 两档都关，检索直接用字面问句（这才是真正的"没有多轮"）
    """
    rw = REWRITE_ENABLED if rewrite is None else rewrite
    cc = CONCAT_ENABLED if concat is None else concat
    if rw:
        return "llm"
    return "concat" if cc else "literal"


def multiturn_fusion_state(fusion=None, rerank_query=None):
    """多轮里"多路召回"这一层的配置标签（纯函数，可 L0 单测）。

    与 `multiturn_degrade_state` 分开，是因为两者正交：
      降级档  决定"拿哪几句去召回"   （llm / concat / literal）
      融合档  决定"召回几路、用哪句重排"（fusion × rerank_query）
    只比降级档的话，开着融合和关着融合的两组数字会被当同一口径比较 ——
    而那正是这条护栏上一轮失效的同一种错（判据没覆盖真正决定指标的配置）。
    """
    f = RE.QUERY_FUSION if fusion is None else fusion
    m = RE.RERANK_QUERY_MODE if rerank_query is None else rerank_query
    return f"{bool(f)}:{m}"


def snapshot(engine, top_k=None, include_multiturn=True):
    """完整评测快照（含元信息），用于录制基线。

    collection 记的是**逻辑集合名**（COLLECTION_NAME）而不是别名解析后的
    具体名：启用别名后具体名带 build_id 后缀，每次重建都变，记它会让
    "基线是不是对当前索引录的"这条校验永远失败。真正需要防的
    "分块变了但基线没重录"由 chunks 计数 + 探针指标本身兜住。
    """
    top_k = top_k or RE.TOP_K
    positive = run_probes(engine, top_k=top_k)
    negative = run_negative(engine, top_k=top_k)
    multiturn = run_multiturn(engine, top_k=top_k) if include_multiturn else None

    snap = {
        "collection": RE.COLLECTION_NAME,
        "resolved_collection": getattr(engine, "collection_name", RE.COLLECTION_NAME),
        "chunks": engine.count(),
        "embed_model": RE.EMBED_MODEL_PATH,
        "rerank_model": RE.RERANK_MODEL_PATH,
        "top_k": top_k,
        "rerank_top_k": RE.RERANK_TOP_K,
        "rerank_max_length": RE.RERANK_MAX_LENGTH,
        # 会影响质量的检索配置也要进快照：它们一变，指标就不可比
        "rerank_on": RE.RERANK_ON,
        "rrf_k": RE.RRF_K,
        "rrf_dense_weight": RE.RRF_DENSE_WEIGHT,
        "rrf_sparse_weight": RE.RRF_SPARSE_WEIGHT,
        "dedup_by_parent": RE.DEDUP_BY_PARENT,
        "recall_limit": RE.RECALL_LIMIT,
        "aliases_enabled": RE.ALIASES_ENABLED,
        "stopwords_enabled": RE.STOPWORDS_ENABLED,
        # 多轮指标取决于**降级链实际走到哪一档**，而不是单个改写开关：
        # RAG_QUERY_REWRITE=0 只关掉 LLM 那一档，拼接档（CONCAT_ENABLED）
        # 仍会改变检索问句，多轮指标因此照样在变、照样可比。
        # 只记 rewrite_enabled 会让"改写关=不可比"这个错误推论成立
        # （test_retrieval_quality 曾据此 skip 掉整组多轮断言，护栏长期失效）。
        # 两档一起记，判据才是"配置是否与基线一致"。
        "rewrite_enabled": REWRITE_ENABLED,
        "concat_enabled": CONCAT_ENABLED,
        "multiturn_degrade_state": multiturn_degrade_state(),
        "multiturn_fusion_state": multiturn_fusion_state(),
        # aggregate：**只含正向**的扁平结构，与历史基线（旧格式）逐字段兼容。
        # 不要把嵌套结构塞进 "aggregate" —— test_retrieval_quality 与
        # record_baseline 都按扁平键读它，改了会让 L3 直接 KeyError。
        # 三类一起看请用 aggregate_all。
        "aggregate": aggregate_all(positive=positive)["positive"],
        # 冻结的 12 条短探针单独留一份：它是唯一能与旧基线**按位置对拍**的一组，
        # 混进 12 条长问句后数字会天然变低，新旧不可比。
        "aggregate_short": short_only(positive),
        "aggregate_all": aggregate_all(positive=positive, negative=negative,
                                       multiturn=multiturn),
        "per_probe": positive,
        "per_probe_negative": negative,
    }
    if multiturn is not None:
        snap["per_probe_multiturn"] = multiturn
    return snap


def short_only(per_probe):
    """只取 12 条短探针的指标 —— 与历史基线可比的那部分。

    旧基线只有这 12 条，若把 12 条长问句混进聚合比率，新旧数字就不可比
    （长问句天然更难），于是"是不是退化了"这个判断会被稀释掉。
    """
    keys = {p.query for p in SHORT_PROBES}
    return aggregate_all(positive={q: v for q, v in per_probe.items() if q in keys})["positive"]
