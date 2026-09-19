#!/usr/bin/env python3
"""多轮检索 A/B —— 只改环境变量，跑同一套多轮探针，逐类 + 逐条对拍。

与 `ab_retrieval.py` 的分工：
    ab_retrieval.py   正向探针（12 短 + 12 长）的 A/B，看 MRR / kw@k
    本脚本            多轮探针（带 history）的 A/B，看 book@1 / kw@k / topic@k

为什么多轮要单独一支：多轮指标取决于**降级链实际走到哪一档**
（LLM 改写 / 拼接上轮用户问句 / 字面原句），而这三档由
`RAG_QUERY_REWRITE` / `RAG_QUERY_REWRITE_CONCAT` / `RAG_QUERY_FUSION`
三个开关共同决定 —— 用正向探针 A/B 是完全测不出来的（正向探针不传 history）。

⚠️ **一个状态一个进程**：`REWRITE_ENABLED` / `CONCAT_ENABLED` 是
`query_rewrite` 在 import 时读成的模块常量，`QUERY_FUSION` 同理。
同进程里改 `os.environ` 不会生效（见 query_rewrite.py 顶部 D9 的说明）。

用法::

    python tools/ab_multiturn.py --label single                 # 现状
    RAG_QUERY_FUSION=1 python tools/ab_multiturn.py --label fusion
    python tools/ab_multiturn.py --compare single fusion        # 对拍两份 json

    # 输出默认写到 /tmp/ab_multiturn_<label>.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_engine as RE  # noqa: E402
import query_rewrite as QR  # noqa: E402
from tests.eval_runner import multiturn_degrade_state  # noqa: E402
from tests.probes import (  # noqa: E402
    MULTITURN_PROBES,
    aggregate_multiturn,
    aggregate_multiturn_by_kind,
    evaluate_multiturn,
)

METRICS = ("book@1", "book@k", "kw@k", "topic@k")


def run(engine, top_k):
    """跑一遍全部多轮探针，返回 per_probe（含每轮实际用的检索问句与来源）。"""
    per_probe = {}
    for p in MULTITURN_PROBES:
        results, steps = engine.hybrid_search(
            p.query, top_k=top_k, history=list(p.history), return_steps=True)
        rw = steps.get(RE.STEP_REWRITE) or {}
        # 与 eval_runner 同口径：topic@k 认别名（关羽/云长同人）
        rec = evaluate_multiturn(p, results, normalize=RE.normalize_aliases)
        rec.update({
            "kind": p.kind,
            "source": rw.get("source", "(none)"),
            "reason": rw.get("reason", ""),
            "search_query": rw.get("query", p.query),
            # 多路召回时把每一路都留档：融合之后"这一分是谁的功劳"只能靠它回答
            "routes": [r.get("query") for r in (rw.get("routes") or [])],
            "books": [r.get("book", "") for r in results],
            "top1_score": results[0].get("rerank_score") if results else None,
        })
        per_probe[p.query] = rec
    return per_probe


def report(label, per_probe, elapsed):
    agg = aggregate_multiturn(per_probe)
    by_kind = aggregate_multiturn_by_kind(per_probe)
    src = {}
    for v in per_probe.values():
        src[v["source"]] = src.get(v["source"], 0) + 1

    print(f"\n=== {label}  （降级档 {multiturn_degrade_state()}，"
          f"rewrite={QR.REWRITE_ENABLED} concat={QR.CONCAT_ENABLED} "
          f"fusion={RE.QUERY_FUSION}）耗时 {elapsed:.0f}s")
    print(f"    合计 n={agg['n']}  " + "  ".join(f"{k}={agg[k]:.1%}" for k in METRICS))
    print(f"    source 分布: {src}")
    for kind, m in by_kind.items():
        print(f"      {kind:16s} n={m['n']}  " +
              "  ".join(f"{k}={m[k]:.0%}" for k in METRICS))
    return agg, by_kind, src


def compare(path_a, path_b):
    """对拍两份结果：逐类比率 + 逐条"谁独家命中"。"""
    A = json.load(open(path_a, encoding="utf-8"))
    B = json.load(open(path_b, encoding="utf-8"))
    pa, pb = A["per_probe"], B["per_probe"]
    keys = [q for q in pa if q in pb]

    print(f"\n{'指标':10s} {'A=' + A['label']:>16s} {'B=' + B['label']:>16s}   变化")
    for grp, sa, sb in (("合计", A["aggregate"], B["aggregate"]),
                        *[(k, A["by_kind"].get(k, {}), B["by_kind"].get(k, {}))
                          for k in A["by_kind"]]):
        for m in METRICS:
            if m not in sa or m not in sb:
                continue
            d = sb[m] - sa[m]
            if abs(d) < 1e-9:
                continue
            print(f"{grp:10s} {m:8s} {sa[m]:15.1%} {sb[m]:16.1%}   {d:+.1%}")

    print("\n逐条差异（只列有变化的）:")
    n_diff = 0
    for q in keys:
        for m in METRICS:
            if pa[q].get(m) != pb[q].get(m):
                n_diff += 1
                print(f"  {q[:22]:24s} {pa[q]['kind']:15s} {m:8s} "
                      f"{'Y' if pa[q][m] else 'N'} → {'Y' if pb[q][m] else 'N'}")
    if not n_diff:
        print("  （无）")

    print("\nB 相对 A 的独家命中（B 命中而 A 没命中的条目）:")
    uniq = [q for q in keys if any(pb[q][m] and not pa[q][m] for m in METRICS)]
    lost = [q for q in keys if any(pa[q][m] and not pb[q][m] for m in METRICS)]
    print(f"  改善 {len(uniq)} 条: {uniq}")
    print(f"  退化 {len(lost)} 条: {lost}")


def main():
    ap = argparse.ArgumentParser(description="多轮检索 A/B")
    ap.add_argument("--label", default=None, help="本次运行的标签（写入 json 文件名）")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--json", default=None, help="结果写入路径")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="对拍两份已有结果（不跑检索）")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0
    if not args.label:
        ap.error("需要 --label（或使用 --compare）")

    top_k = args.top_k or RE.TOP_K
    engine = RE.RAGEngine()
    if engine.count() == 0:
        sys.exit("集合为空，请先 python app.py index")

    t0 = time.time()
    per_probe = run(engine, top_k)
    elapsed = time.time() - t0
    agg, by_kind, src = report(args.label, per_probe, elapsed)

    path = args.json or f"/tmp/ab_multiturn_{args.label}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "label": args.label,
            "degrade_state": multiturn_degrade_state(),
            "rewrite_enabled": QR.REWRITE_ENABLED,
            "concat_enabled": QR.CONCAT_ENABLED,
            "fusion": RE.QUERY_FUSION,
            "collection": engine.collection_name,
            "top_k": top_k,
            "elapsed_s": elapsed,
            "aggregate": agg,
            "by_kind": by_kind,
            "source_dist": src,
            "per_probe": per_probe,
        }, f, ensure_ascii=False, indent=2)
    print(f"    已写入 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
