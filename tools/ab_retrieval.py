#!/usr/bin/env python3
"""检索配置 A/B —— 用**同一套探针**度量不同配置，只改环境变量。

## 为什么这样设计

配置已经全部环境变量化（见 rag_engine 顶部的 _env_* 与 .env 里的清单），
所以 A/B 不需要任何"两份检索逻辑"：

    RAG_RERANK_ON=child  python tools/ab_retrieval.py --label child  --json /tmp/a.json
    RAG_RERANK_ON=parent python tools/ab_retrieval.py --label parent --json /tmp/b.json
    python tools/ab_retrieval.py --compare /tmp/a.json /tmp/b.json

三条命令共用同一个评测器，因此**不存在"两次评测测的不是一个检索器"**这种
问题 —— 本项目已经吃过一次亏：探针曾在 compare_ab.py 与 verify_qdrant.py
各写一份副本，导致两次评测不可比。

评测链路一律走 `RAGEngine.hybrid_search`（线上同一个入口），
不自己拼 Qdrant 查询。

## 指标

    book@1 / book@k / kw@k   与 L3 基线同一套判定，便于横向对齐
    mrr                      平均倒数名次 —— 只看 book@1 看不出"第 2 名该升到第 1"
    hit@1_kw                 首位就命中的比例
    rerank_ms                重排耗时（RERANK_ON=parent 的代价就在这里）
    search_ms                端到端检索耗时（不含模型首次加载）

耗时只作参考：本机负载波动很大（实测同一配置在不同负载下能差数倍），
**不要用绝对耗时下结论，只用于横向比较同一次运行内的两个配置**。
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_engine as RE            # noqa: E402
from tests import eval_runner as ER   # noqa: E402  ("跑探针"的唯一实现)
from tests.probes import PROBES, keyword_hit  # noqa: E402


def current_config():
    """当前生效的检索配置快照 —— 存进结果文件，免得事后不知道比的是什么。"""
    return {
        "rerank_on": RE.RERANK_ON,
        "rerank_top_k": RE.RERANK_TOP_K,
        "rerank_max_length": RE.RERANK_MAX_LENGTH,
        "recall_limit": RE.RECALL_LIMIT,
        "top_k": RE.TOP_K,
        "rrf_k": RE.RRF_K,
        "rrf_dense_weight": RE.RRF_DENSE_WEIGHT,
        "rrf_sparse_weight": RE.RRF_SPARSE_WEIGHT,
        "dedup_by_parent": RE.DEDUP_BY_PARENT,
        "aliases": RE.ALIASES_ENABLED,
        "stopwords": RE.STOPWORDS_ENABLED,
    }


def run(label, top_k=None, probes=None):
    """跑一遍探针并汇总。

    "遍历探针 → 检索 → evaluate_probe → 记账"这段循环由
    ER.run_probes_with 统一提供（L3 与基线录制走同一份），本函数只注入两样
    它特有的东西：计时，以及名次/token 这些额外字段。此前这段循环在本文件
    与 tests/eval_runner.py 各写一份 —— 而 A/B 与基线必须只在**配置**上不同，
    不能在"什么算一次评测"上不同。
    """
    top_k = top_k or RE.TOP_K
    engine = RE.RAGEngine()
    search_times, rerank_tokens = [], []

    def searcher(query, k):
        t0 = time.time()
        results, steps = engine.hybrid_search(query, top_k=k, return_steps=True)
        search_times.append((time.time() - t0) * 1000)
        return results, steps

    def extra(probe, results, steps, rec):
        rr = steps.get(RE.STEP_RERANK) or []
        rerank_tokens.extend(r.get("tokens", 0) for r in rr)
        rec["hit_rank"] = next(
            (i for i, r in enumerate(results, 1)
             if keyword_hit(probe, (r.get("child_text", "") or "")
                            + (r.get("parent_text", "") or ""))),
            None,
        )
        rec["rerank_tokens"] = [r.get("tokens", 0) for r in rr]
        rec["folded"] = (steps.get(RE.STEP_DEDUP) or {}).get("folded", 0)

    per = ER.run_probes_with(searcher, top_k=top_k, probes=probes,
                             record_extra=extra)

    n = len(per) or 1
    mrr_sum = sum((1.0 / v["hit_rank"]) if v["hit_rank"] else 0.0
                  for v in per.values())
    hit1_kw = sum(
        int(bool(v["books"]) and v["hit_rank"] == 1) for v in per.values())

    return {
        "label": label,
        "config": current_config(),
        "env": {k: v for k, v in os.environ.items() if k.startswith("RAG_")},
        "n_probes": len(per),
        "metrics": {
            "book@1": sum(1 for v in per.values() if v["book@1"]) / n,
            "book@k": sum(1 for v in per.values() if v["book@k"]) / n,
            "kw@k": sum(1 for v in per.values() if v["kw@k"]) / n,
            "mrr": mrr_sum / n,
            "hit@1_kw": hit1_kw / n,
            "search_ms_median": round(statistics.median(search_times), 1),
            "rerank_tokens_median": round(statistics.median(rerank_tokens), 1)
            if rerank_tokens else 0,
        },
        "per_probe": per,
    }


def compare(path_a, path_b):
    a = json.load(open(path_a, encoding="utf-8"))
    b = json.load(open(path_b, encoding="utf-8"))
    print(f"\n{'指标':<22}{a['label']:>14}{b['label']:>14}{'变化':>14}")
    print("-" * 66)
    for key in ("book@1", "book@k", "kw@k", "mrr", "hit@1_kw"):
        va, vb = a["metrics"][key], b["metrics"][key]
        delta = vb - va
        flag = "  " if abs(delta) < 1e-9 else ("↑" if delta > 0 else "↓")
        print(f"{key:<22}{va:>14.4f}{vb:>14.4f}{flag}{delta:>+13.4f}")
    for key in ("search_ms_median", "rerank_tokens_median"):
        print(f"{key:<22}{a['metrics'][key]:>14}{b['metrics'][key]:>14}")
    print(f"\n配置 {a['label']}: {a['config']}")
    print(f"配置 {b['label']}: {b['config']}")

    # 逐条看"谁变好了、谁变差了"：聚合指标会掩盖个别退化
    diffs = []
    for q, va in a["per_probe"].items():
        vb = b["per_probe"].get(q)
        if not vb:
            continue
        for key in ("book@1", "book@k", "kw@k"):
            if va[key] != vb[key]:
                diffs.append(f"  {q[:24]:<26} {key}: "
                             f"{'Y' if va[key] else 'N'} → {'Y' if vb[key] else 'N'}")
    print("\n逐条差异:" + ("\n" + "\n".join(diffs) if diffs else "  （无）"))

    # ---- 名次级胜负统计 ----
    # 布尔指标（book@1/kw@k）只在"跨过阈值"时才动，MRR 对名次更敏感。
    # 只报"MRR 涨了 0.03"是不够的：必须同时说明**有多少条探针真的变了**，
    # 否则读者无法判断这是普适改善还是单条噪声。样本量小的时候尤其重要 ——
    # 本项目的探针集只有 24 条正向，1~2 条探针的差异完全可能是巧合。
    win = loss = tie = 0
    for q, va in a["per_probe"].items():
        vb = b["per_probe"].get(q)
        if not vb:
            continue
        ra, rb = va.get("hit_rank"), vb.get("hit_rank")
        # 名次越小越好；None 表示整条都没命中，视为无穷大
        sa = 10**6 if ra is None else ra
        sb = 10**6 if rb is None else rb
        if sb < sa:
            win += 1
        elif sb > sa:
            loss += 1
        else:
            tie += 1
    n = win + loss + tie
    print(f"\n名次级胜负（首位关键词命中名次）："
          f"改善 {win} / 退化 {loss} / 持平 {tie}   （n={n}）")
    if win + loss <= 2:
        print("  ⚠️  发生变化的探针不超过 2 条 —— 聚合指标的差异基本由个别探针驱动，")
        print("     不足以据此改动默认配置。要下结论请先扩充探针集（见 PROJECT_DOC）。")
    return 0


def main():
    ap = argparse.ArgumentParser(description="检索配置 A/B")
    ap.add_argument("--label", default="current", help="本次运行的标签")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--long-only", action="store_true",
                    help="只跑长问句探针（12 条），用于快速看趋势")
    ap.add_argument("--json", default=None, help="把结果写入该文件")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="比较两个结果文件，不发查询")
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)

    probes = PROBES if not args.long_only else PROBES[12:]
    res = run(args.label, top_k=args.top_k, probes=probes)
    m = res["metrics"]
    print(f"\n[{args.label}] n={res['n_probes']}  "
          f"book@1={m['book@1']:.1%}  book@k={m['book@k']:.1%}  "
          f"kw@k={m['kw@k']:.1%}  MRR={m['mrr']:.4f}  "
          f"hit@1_kw={m['hit@1_kw']:.1%}")
    print(f"  检索耗时中位 {m['search_ms_median']}ms  "
          f"rerank 输入 token 中位 {m['rerank_tokens_median']}")
    print(f"  配置: {res['config']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"  已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
