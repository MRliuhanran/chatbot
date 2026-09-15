#!/usr/bin/env python3
"""检索级 A/B —— 在同一批探针查询上对比两个 Qdrant 集合，用于回答
"新方案是否真的更好"，例如决定旧集合能否删除。

与 compare_sparse.py 的分工：
  compare_sparse.py  固定稠密通道，只换稀疏分词方案
  compare_ab.py      完整检索链路（稠密 + 稀疏 → RRF → rerank），换整个集合

关键实现细节：
  * 稠密维度自动匹配嵌入模型（512→bge-small / 768→bge-base / 1024→bge-m3）。
    这是必须的：查询侧与文档侧必须用同一个模型编码，否则向量空间错配，
    检索结果没有意义（而且不会报错，只会悄悄变差）。
  * 两个集合用**同一个 reranker**、同一套参数，保证只有被考察的变量在变。
  * 探针带"期望书目"和"期望关键词"两个标注：
      书目命中 = 排序质量（能否把正确的书排上来）
      关键词命中 = 召回质量（正文里到底有没有那段内容）
    两者都看，因为书目命中高但关键词全错，说明只是书名那层在起作用。

用法:
  python compare_ab.py                          # 默认 books_v2 vs books_v3
  python compare_ab.py A B --top-k 5
退出码: 0（本工具只报告，不做通过/失败判定）
"""

import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import rag_engine as RE
from qdrant_client import QdrantClient, models

# 维度 → 模型目录。查询侧必须与建索引时用的模型一致，见文件头说明。
MODEL_BY_DIM = {
    512: "./models/bge-small-zh-v1.5",
    768: "./models/bge-base-zh-v1.5",
    1024: "./models/bge-m3",
}

# 探针：(查询, 期望书目, 期望关键词)。关键词取查询涉及的人名/事件字面量，
# 命中判定为"该串出现在任一召回结果的 child_text 或 parent_text 中"。
PROBES = [
    ("武松打虎", "水浒传", "景阳冈"),
    ("黛玉葬花", "红楼梦", "葬花"),
    ("桃园结义", "三国演义", "桃园"),
    ("火烧赤壁", "三国演义", "赤壁"),
    ("倒拔垂杨柳", "水浒传", "垂杨"),
    ("大闹天宫", "西游记", "天宫"),
    ("刘姥姥进大观园", "红楼梦", "刘姥姥"),
    ("空城计", "三国演义", "空城"),
    ("三打白骨精", "西游记", "白骨"),
    ("鲁智深拳打镇关西", "水浒传", "镇关西"),
    ("草船借箭", "三国演义", "草船"),
    ("宝玉挨打", "红楼梦", "贾政"),
]


def dense_dim(client, coll):
    info = client.get_collection(collection_name=coll)
    name = next(iter(info.config.params.vectors))
    return name, info.config.params.vectors[name].size


def load_embed_for(dim):
    path = MODEL_BY_DIM.get(dim)
    if path is None:
        sys.exit(f"❌ 未知的稠密维度 {dim}，请在 MODEL_BY_DIM 中登记对应模型")
    print(f"    加载嵌入模型 {path} (dim={dim}) ...")
    # load_embedding_model() 从模块全局读模型路径，故临时改全局再还原；
    # 集合维度不同就可能需要不同模型，这是唯一不侵入引擎的接法。
    saved = RE.EMBED_MODEL_PATH
    RE.EMBED_MODEL_PATH = path
    try:
        tok, model, dev = RE.load_embedding_model(device="cpu")
    finally:
        RE.EMBED_MODEL_PATH = saved
    return tok, model, dev


def run_collection(coll, top_k, rr):
    """对一个集合跑完整检索链路，返回每条探针的结果。"""
    client = QdrantClient(host=RE.QDRANT_HOST, port=RE.QDRANT_PORT, timeout=60)
    vec_name, dim = dense_dim(client, coll)
    print(f"\n{'=' * 78}\n集合 {coll}  稠密维度={dim}  点数={client.count(collection_name=coll).count}\n{'=' * 78}")

    e_tok, e_model, e_dev = load_embed_for(dim)
    rr_tok, rr_model, rr_dev = rr

    out = []
    for query, want_book, want_kw in PROBES:
        t0 = time.time()
        # 查询侧稠密向量（BGE 用法：query 侧加 instruction，doc 侧不加）
        qv = RE.embed([query], e_tok, e_model, e_dev, is_query=True)[0].tolist()
        # 双通道召回 → 服务端 RRF 融合
        res = client.query_points(
            collection_name=coll,
            prefetch=[
                models.Prefetch(query=qv, using=RE.DENSE_VECTOR_NAME, limit=RE.RERANK_TOP_K),
                models.Prefetch(query=RE.sparse_encode(query), using=RE.SPARSE_VECTOR_NAME,
                                limit=RE.RERANK_TOP_K),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=RE.RERANK_TOP_K,
            with_payload=True,
        ).points

        cands = [p.payload for p in res]
        # 与线上一致的 rerank 输入：contextual_text
        texts = [c.get("contextual_text", "") for c in cands]
        ranked = RE.rerank_indices(
            query, texts, rr_tok, rr_model, rr_dev,
            top_k=top_k, batch_size=RE.RERANK_BATCH, max_length=RE.RERANK_MAX_LENGTH,
        )
        hits = [cands[i] for i, _ in ranked]
        scores = [s for _, s in ranked]

        books = [h.get("book", "") for h in hits]
        body = " ".join((h.get("child_text", "") or "") + (h.get("parent_text", "") or "")
                        for h in hits)
        out.append({
            "query": query,
            "book@1": bool(books) and books[0] == want_book,
            "book@k": want_book in books,
            "kw@k": want_kw in body,
            "top1_score": scores[0] if scores else float("-inf"),
            "books": books,
            "secs": time.time() - t0,
        })
        flag = "✅" if out[-1]["book@1"] else "❌"
        print(f"  {flag} {query:12s} book@1={books[0] if books else '-':8s} "
              f"kw={'Y' if out[-1]['kw@k'] else 'N'} top1={out[-1]['top1_score']:7.3f} "
              f"({out[-1]['secs']:.1f}s)")

    del e_model, e_tok
    return out


def summarize(name, rows):
    n = len(rows)
    b1 = sum(r["book@1"] for r in rows) / n
    bk = sum(r["book@k"] for r in rows) / n
    kw = sum(r["kw@k"] for r in rows) / n
    sc = sum(r["top1_score"] for r in rows if r["top1_score"] != float("-inf")) / n
    t = sum(r["secs"] for r in rows) / n
    print(f"  {name:10s} book@1={b1:6.1%}  book@k={bk:6.1%}  关键词@k={kw:6.1%}  "
          f"平均top1分={sc:7.3f}  平均耗时={t:.2f}s")
    return b1, bk, kw, sc


def main():
    ap = argparse.ArgumentParser(description="检索级 A/B（完整链路）")
    ap.add_argument("a", nargs="?", default="books_v2")
    ap.add_argument("b", nargs="?", default="books_v3")
    ap.add_argument("--top-k", type=int, default=RE.TOP_K)
    args = ap.parse_args()

    print(f"探针数={len(PROBES)}  top_k={args.top_k}  rerank 输入={RE.RERANK_TOP_K}  "
          f"max_length={RE.RERANK_MAX_LENGTH}")
    print("⚠️  注意：若两个集合的分块不同，本对比同时差在『模型』与『分块』两个变量上，")
    print("    结论只能说明『整套方案谁更好』，不能归因到单一模型。")

    # 两个集合共用同一个 reranker，保证只有被考察的变量在变
    print("\n加载 reranker（两集合共用）...")
    try:
        rr = RE.load_reranker_model()
        print(f"    ok: {rr[2]}")
    except Exception as exc:
        print(f"    ⚠️  MPS 加载失败，回退 CPU: {type(exc).__name__}: {exc}")
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        rr_tok = AutoTokenizer.from_pretrained(RE.RERANK_MODEL_PATH)
        rr_model = AutoModelForSequenceClassification.from_pretrained(RE.RERANK_MODEL_PATH)
        rr_model.eval()
        rr = (rr_tok, rr_model, "cpu")

    ra = run_collection(args.a, args.top_k, rr)
    rb = run_collection(args.b, args.top_k, rr)

    print(f"\n{'=' * 78}\n汇总\n{'=' * 78}")
    sa = summarize(args.a, ra)
    sb = summarize(args.b, rb)

    print(f"\n差值（{args.b} − {args.a}）:")
    for label, i, pct in [("book@1", 0, True), ("book@k", 1, True),
                          ("关键词@k", 2, True), ("平均top1分", 3, False)]:
        d = sb[i] - sa[i]
        print(f"  {label:10s} {d:+.1%}" if pct else f"  {label:10s} {d:+.3f}")

    # 逐条列出分歧，便于人工核对到底是哪几条在拉开差距
    diff = [(x["query"], x["book@1"], y["book@1"], x["kw@k"], y["kw@k"])
            for x, y in zip(ra, rb) if x["book@1"] != y["book@1"] or x["kw@k"] != y["kw@k"]]
    if diff:
        print(f"\n分歧探针（{len(diff)} 条）: 查询 | {args.a}:book@1/kw | {args.b}:book@1/kw")
        for q, a1, b1, ak, bk in diff:
            print(f"  {q:14s} | {'Y' if a1 else 'N'}/{'Y' if ak else 'N'} | {'Y' if b1 else 'N'}/{'Y' if bk else 'N'}")
    else:
        print("\n无分歧：两集合在所有探针上的 book@1 与关键词命中完全一致。")


if __name__ == "__main__":
    main()
