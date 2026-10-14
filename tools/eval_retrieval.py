#!/usr/bin/env python
"""检索基线：重建索引前后用**同一批查询**对比，回答"这次改动让检索变了吗、变坏了吗"。

为什么要它：这个项目的产物是"指纹 + 全量重建"，而重建之后 point id 全变，
"自命中"这类指标不可比。可跨重建比较的只有**来源文档**（换分块不影响它），
所以主指标是 source_hit（期望来源是否落进 top-k）与 top-1 来源是否正确。

用法：
    python tools/eval_retrieval.py capture artifacts/eval/before.json     # 重建前抓基线
    python tools/eval_retrieval.py compare artifacts/eval/before.json     # 重建后对比（会现抓一份 after）

查询怎么来：从当前 chunks.json 里按来源分层抽样父块，取中段一段文本当查询 ——
它是**源文档的子串**，所以重建后依然存在，只有"检索是否还能找回它"会变。
对比的不变量：来源命中率、top-1 命中率、top-5 来源集合的重合度、首条重排分。
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chatbot as C  # noqa: E402


def _sample(n=30, seed=20261014):
    recs = json.load(open(C.CHUNKS_JSON, encoding="utf-8"))
    random.seed(seed)
    seen, picked = set(), []
    for r in recs:
        s = C._source_of(r)
        pt = (r.get("parent_text") or "").strip()
        if s in seen or len(pt) < 200:
            continue
        # 中段取样：避开开头（可能与其它块重复）与结尾（可能是标头）
        span = pt[len(pt) // 3: len(pt) // 3 + 40].strip()
        if len(span) < 12:
            continue
        seen.add(s)
        picked.append({"query": span, "expect_source": s})
        if len(picked) >= n:
            break
    return picked


def _run(cases):
    eng = C.RAGEngine()
    collection = C.resolve_collection_name(eng._client)
    out, t0 = [], time.time()
    for c in cases:
        res, _ = eng.hybrid_search(c["query"], top_k=5, return_steps=True)
        hits = [{"source": C._source_of(h), "score": round(float(h.get("rerank_score") or 0), 4)} for h in res]
        out.append({**c, "hits": hits,
                    "source_hit": any(h["source"] == c["expect_source"] for h in hits),
                    "top1_hit": bool(hits) and hits[0]["source"] == c["expect_source"]})
    return {"created": time.strftime("%F %T"), "collection": collection, "n": len(out),
            "source_hit_rate": round(sum(x["source_hit"] for x in out) / max(1, len(out)), 3),
            "top1_hit_rate": round(sum(x["top1_hit"] for x in out) / max(1, len(out)), 3),
            "elapsed": round(time.time() - t0, 1), "cases": out}


def _report(tag, r):
    print(f"[{tag}] 集合={r['collection']} 查询={r['n']} 来源命中={r['source_hit_rate']:.0%} "
          f"top1命中={r['top1_hit_rate']:.0%} 耗时={r['elapsed']}s")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
    path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/eval/before.json"
    if mode == "capture":
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = _run(_sample())
        json.dump(r, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        _report("capture", r)
        print(f"已写入 {path}")
        return 0
    if mode == "compare":
        before = json.load(open(path, encoding="utf-8"))
        after = _run([{"query": c["query"], "expect_source": c["expect_source"]} for c in before["cases"]])
        out_path = path.replace(".json", "_after.json")
        json.dump(after, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        _report("before", before)
        _report("after ", after)
        lost = [c["query"][:28] for b, c in zip(before["cases"], after["cases"]) if b["source_hit"] and not c["source_hit"]]
        gain = [c["query"][:28] for b, c in zip(before["cases"], after["cases"]) if c["source_hit"] and not b["source_hit"]]
        overlap = [len({h["source"] for h in b["hits"]} & {h["source"] for h in c["hits"]}) / 5
                   for b, c in zip(before["cases"], after["cases"])]
        print(f"top-5 来源平均重合度: {sum(overlap)/len(overlap):.0%}")
        print(f"由命中变未命中 {len(lost)} 条: {lost}")
        print(f"由未命中变命中 {len(gain)} 条: {gain}")
        print(f"已写入 {out_path}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main());
