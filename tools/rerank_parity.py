#!/usr/bin/env python3
"""重排一致性比对：进程内 torch vs http 服务（infinity 的 Cohere 风格 /rerank）。

为什么需要它：把重排拆到独立进程之前，必须先证明两条路径给出**同一个排序**。
2026-10-12 记录过 infinity × XLM-R 的 upstream bug（token_type_ids 514 vs 1028），
而 infinity 的 README 又把 bge-reranker-v2-m3 列进 CI 测试清单 —— 只有实测能定论。

三种模式分开跑（torch 与 http 各自持一份 ~2.1G 权重，8G 机器上不要同时驻留）：
  python tools/rerank_parity.py torch   /tmp/rerank_torch.json [--n 20]
  python tools/rerank_parity.py http    /tmp/rerank_http.json  [--n 20]
  python tools/rerank_parity.py compare /tmp/rerank_torch.json /tmp/rerank_http.json

候选取自 artifacts/chunks.json 的 parent_text（**不碰 Qdrant、不碰索引**），
按固定步长抽样，三次运行取同一批，保证可比。
"""
import argparse
import json
import math
import os
import sys

# 从 tools/ 运行：把仓库根加进 sys.path，否则 import chatbot 找不到
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUERIES = [
    "中国革命的对象是谁",
    "为什么要反对本本主义",
]


def load_docs(n, artifacts_dir=None, doc_chars=0):
    import chatbot as C
    path = os.path.join(artifacts_dir or C._ARTIFACTS_DIR, "chunks.json")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        raise SystemExit(f"产物为空: {path}")
    step = max(1, len(records) // n)
    picked, seen = [], set()
    for i in range(0, len(records), step):
        text = records[i].get("parent_text") or records[i].get("child_text") or ""
        if not text or text in seen:
            continue
        if doc_chars:
            text = text[:doc_chars]
        seen.add(text)
        picked.append(text)
        if len(picked) >= n:
            break
    return picked


def run_torch(docs):
    import chatbot as C
    tokenizer, model, device = C.load_reranker_model()
    out = {}
    for q in QUERIES:
        ranked = C.rerank_indices(q, docs, tokenizer, model, device, top_k=len(docs))
        out[q] = [[int(i), float(s)] for i, s in ranked]
    return out


def run_embed_torch(docs):
    import chatbot as C
    tokenizer, model, device = C.load_embedding_model()
    vecs = C.embed(docs, tokenizer, model, device, is_query=False, autocast=C._float16_ok(device))
    return {q: [v.tolist() for v in vecs] for q in QUERIES}  # 嵌入与查询无关，两份相同


def run_embed_http(docs):
    import requests

    import chatbot as C

    url = C.EMBED_URL + "/embeddings"
    r = requests.post(url, json={"input": docs, "model": C.EMBED_MODEL_NAME}, timeout=C.MODEL_HTTP_TIMEOUT)
    if r.status_code != 200:
        raise SystemExit(f"{url} HTTP {r.status_code}: {r.text[:300]}")
    rows = r.json().get("data")
    if not isinstance(rows, list) or len(rows) != len(docs):
        got = len(rows) if isinstance(rows, list) else "非列表"
        raise SystemExit(f"{url} 返回 {got} 条 ≠ 输入 {len(docs)} 条")
    vecs = [row["embedding"] for row in sorted(rows, key=lambda x: x.get("index", 0))]
    return {q: vecs for q in QUERIES}


def compare_embed(path_a, path_b):
    a = json.load(open(path_a, encoding="utf-8"))
    b = json.load(open(path_b, encoding="utf-8"))
    bad = 0
    for q in sorted(set(a) | set(b)):
        if q not in a or q not in b or len(a[q]) != len(b[q]):
            print(f"❌ {q!r}: 两侧向量条数不同")
            bad = 1
            continue
        max_abs, cos_min = 0.0, 1.0
        for va, vb in zip(a[q], b[q]):
            if len(va) != len(vb):
                print(f"❌ 维度不同: {len(va)} vs {len(vb)}")
                bad = 1
                break
            max_abs = max(max_abs, max(abs(x - y) for x, y in zip(va, vb)))
            na = sum(x * x for x in va) ** 0.5
            nb = sum(x * x for x in vb) ** 0.5
            dot = sum(x * y for x, y in zip(va, vb))
            cos_min = min(cos_min, dot / (na * nb) if na and nb else 0.0)
        print(f"查询 {q!r}  向量 {len(a[q])} 条 / {len(a[q][0])} 维")
        print(f"  最大绝对差      : {max_abs:.3e}")
        print(f"  最小 cosine     : {cos_min:.8f}")
        if max_abs > 1e-3 or cos_min < 0.9999:
            bad = 1
    print("\n结论：" + ("❌ 两条路径向量不一致" if bad else "✅ 向量一致（差异为精度噪声）"))
    return bad


def run_http(docs):
    import requests

    import chatbot as C

    url = C.RERANK_URL + "/rerank"
    out = {}
    for q in QUERIES:
        r = requests.post(url, json={"model": C.RERANK_MODEL_NAME, "query": q, "documents": docs, "top_n": len(docs)}, timeout=C.MODEL_HTTP_TIMEOUT)
        if r.status_code != 200:
            raise SystemExit(f"{url} HTTP {r.status_code}: {r.text[:300]}")
        rows = r.json().get("results")
        if not isinstance(rows, list) or len(rows) != len(docs):
            got = len(rows) if isinstance(rows, list) else "非列表"
            raise SystemExit(f"{url} 返回 {got} 条 ≠ 候选 {len(docs)} 条（必须对全体候选统一打分）")
        out[q] = sorted([[int(x["index"]), float(x.get("relevance_score", x.get("score")))] for x in rows], key=lambda p: -p[1])
    return out


def _spearman(a, b):
    """两个排序（下标序列）的 Spearman 秩相关，不引 scipy。"""
    n = len(a)
    if n < 2:
        return 1.0
    ra = {v: i for i, v in enumerate(a)}
    rb = {v: i for i, v in enumerate(b)}
    common = [v for v in a if v in rb]
    if len(common) < 2:
        return 0.0
    d2 = sum((ra[v] - rb[v]) ** 2 for v in common)
    m = len(common)
    return 1.0 - (6.0 * d2) / (m * (m * m - 1))


def compare(path_a, path_b, top_k=5):
    a = json.load(open(path_a, encoding="utf-8"))
    b = json.load(open(path_b, encoding="utf-8"))
    bad = 0
    for q in sorted(set(a) | set(b)):
        if q not in a or q not in b:
            print(f"❌ {q!r}: 只有一侧有结果")
            bad = 1
            continue
        sa = dict((int(i), float(s)) for i, s in a[q])
        sb = dict((int(i), float(s)) for i, s in b[q])
        if set(sa) != set(sb):
            print(f"❌ {q!r}: 候选下标集合不同（torch {len(sa)} / http {len(sb)}）")
            bad = 1
            continue
        order_a = [i for i, _ in sorted(sa.items(), key=lambda kv: -kv[1])]
        order_b = [i for i, _ in sorted(sb.items(), key=lambda kv: -kv[1])]
        deltas = [abs(sa[i] - sb[i]) for i in sa]
        sig_deltas = [abs(sb[i] - 1.0 / (1.0 + math.exp(-sa[i]))) for i in sa]
        top_a, top_b = set(order_a[:top_k]), set(order_b[:top_k])
        rho = _spearman(order_a, order_b)
        same_top = top_a == top_b
        print(f"查询 {q!r}  候选 {len(sa)}")
        print(f"  Spearman 秩相关       : {rho:.6f}")
        print(f"  top{top_k} 集合一致        : {'✅ 是' if same_top else '❌ 否'}")
        print(f"  原始分数最大绝对差    : {max(deltas):.3e}")
        print(f"  与 sigmoid(torch) 之差: {max(sig_deltas):.3e}")
        # 通过判据：排序一致，且分数要么直接一致、要么是 sigmoid 关系 —— Cohere 风格服务返回的是
        # [0,1] 相关性分（= sigmoid(logit)），应用侧返回原始 logits，两者单调等价，不是"噪声"。
        if not same_top or rho < 0.999 or min(max(deltas), max(sig_deltas)) > 1e-2:
            bad = 1
    print("\n结论：" + ("❌ 两条路径不一致，不要切 http 重排" if bad else "✅ 排序一致；分数是同一 logits 的单调变换（sigmoid，Cohere 风格）"))
    return bad


def main():
    ap = argparse.ArgumentParser(description="重排一致性比对（进程内 torch vs http 服务）")
    ap.add_argument("mode", choices=["torch", "http", "compare"])
    ap.add_argument("out", help="torch/http：写出的 JSON 路径；compare：第一份 JSON")
    ap.add_argument("other", nargs="?", help="compare：第二份 JSON")
    ap.add_argument("--n", type=int, default=20, help="候选数（默认 20，与 RAG_RERANK_TOP_K 一致）")
    ap.add_argument("--kind", choices=["rerank", "embed"], default="rerank", help="比对重排分数还是嵌入向量")
    ap.add_argument("--doc-chars", type=int, default=0, help="把候选截断到 N 字符（用于隔离截断长度差异）")
    args = ap.parse_args()
    if args.mode == "compare":
        if not args.other:
            raise SystemExit("compare 需要两份 JSON")
        return (compare_embed if args.kind == "embed" else compare)(args.out, args.other)
    docs = load_docs(args.n, doc_chars=args.doc_chars)
    print(f"候选 {len(docs)} 条，查询 {len(QUERIES)} 个，比对对象 {args.kind}", file=sys.stderr)
    if args.kind == "embed":
        out = run_embed_torch(docs) if args.mode == "torch" else run_embed_http(docs)
    else:
        out = run_torch(docs) if args.mode == "torch" else run_http(docs)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"已写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
