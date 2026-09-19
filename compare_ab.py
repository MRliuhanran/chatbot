#!/usr/bin/env python3
"""检索级 A/B —— 在同一批探针查询上对比两个 Qdrant 集合，用于回答
"新方案是否真的更好"，例如决定旧集合能否删除。

跑的是**完整检索链路**（稠密 + 稀疏 → RRF → rerank），两个集合整体对照。
若要固定稠密通道、只换稀疏分词方案，属于另一类实验，需自行改造本脚本。

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
  python compare_ab.py A B --top-k 5            # A、B 为两个 Qdrant 集合名
退出码: 0（本工具只报告，不做通过/失败判定）
"""

import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import rag_engine as RE
from qdrant_client import QdrantClient

# 探针统一来自 tests/probes.py —— 此前这里与 verify_qdrant.py 各写了一份，
# 同一个"武松打虎"一处期望"景阳冈"、一处期望"武松"，两份副本会各自漂移，
# 使两次评测的结果不可比（而可比正是 A/B 的全部意义）。
from tests.probes import PROBES, keyword_hit

# 维度 → 模型目录。查询侧必须与建索引时用的模型一致，见文件头说明。
MODEL_BY_DIM = {
    512: "./models/bge-small-zh-v1.5",
    768: "./models/bge-base-zh-v1.5",
    1024: "./models/bge-m3",
}


def dense_dim(client, coll):
    """返回集合的稠密向量维度（多个稠密空间时取第一个）。"""
    info = client.get_collection(collection_name=coll)
    name = next(iter(info.config.params.vectors))
    return info.config.params.vectors[name].size


def load_embed_for(dim):
    path = MODEL_BY_DIM.get(dim)
    if path is None:
        sys.exit(f"❌ 未知的稠密维度 {dim}，请在 MODEL_BY_DIM 中登记对应模型")
    print(f"    加载嵌入模型 {path} (dim={dim}) ...")
    # 显式传 model_path，不再临时改写 RE.EMBED_MODEL_PATH 这个模块全局：
    # 改全局会让调用方之间产生顺序依赖，且中途抛异常会把全局留在被改过的状态。
    return RE.load_embedding_model(device="cpu", model_path=path)


def run_collection(coll, top_k, rr):
    """对一个集合跑完整检索链路，返回每条探针的结果。"""
    client = QdrantClient(host=RE.QDRANT_HOST, port=RE.QDRANT_PORT, timeout=60)
    dim = dense_dim(client, coll)
    print(f"\n{'=' * 78}\n集合 {coll}  稠密维度={dim}  点数={client.count(collection_name=coll).count}\n{'=' * 78}")

    e_tok, e_model, e_dev = load_embed_for(dim)
    rr_tok, rr_model, rr_dev = rr

    out = []
    for probe in PROBES:
        query, want_book = probe.query, probe.book
        t0 = time.time()
        # 查询侧稠密向量（BGE 用法：query 侧加 instruction，doc 侧不加）
        qv = RE.embed([query], e_tok, e_model, e_dev, is_query=True)[0].tolist()
        # 双通道召回 → 加权 RRF → 按父块去重 → rerank
        #
        # 召回走 RE.dual_channel_recall + RE.weighted_rrf（线上同一套实现）。
        # A/B 工具若自己拼一份，比的就是一个**线上不存在的检索器** —— 而且这个
        # 副本真的坏过：它曾用 with_payload=False 却去读 p.payload，第一个探针
        # 就抛 AttributeError。with_payload=True 是这里必须的（下面要把 point
        # 变成候选字典），而 with_payload=False 时 p.payload 恒为 None。
        dense_hits, sparse_hits = RE.dual_channel_recall(
            client, coll, qv, RE.sparse_encode(query), RE.RECALL_LIMIT,
            with_payload=True,
        )
        fused = RE.weighted_rrf(
            RE.channel_rankings(dense_hits, sparse_hits),
            k=RE.RRF_K, limit=RE.RERANK_TOP_K,
        )
        # 字典构造交给 RE.chunk_from_payload（线上 _get_chunks 用的同一个函数）：
        # payload 里的键叫 chunk_id、线上字典里的键叫 id，自己手抄必然再错一次。
        by_id = {p.id: RE.chunk_from_payload(p.payload, p.id)
                 for p in list(dense_hits) + list(sparse_hits)}
        cands = [by_id[i] for i, _, _ in fused if i in by_id]
        cands, _folded = RE.dedup_candidates_by_parent(cands) if RE.DEDUP_BY_PARENT else (cands, [])
        # 与线上一致的 rerank 输入：由 RE.rerank_text_for 决定（child 或 parent）
        texts = [RE.rerank_text_for(c) for c in cands]
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
            "kw@k": keyword_hit(probe, body),
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
    """打印并返回四项聚合指标（键名同时用作下面差值的标签）。"""
    n = len(rows)
    # 平均 top1 分只在**真正召回到候选**的探针上求平均：-inf 表示"零候选"，
    # 旧写法把它剔出分子却仍计入分母 n，于是某一个探针召不回东西就会系统性
    # 拉低这一项，而且两侧候选数不同时会制造出虚假的优劣。
    scored = [r["top1_score"] for r in rows if r["top1_score"] != float("-inf")]
    zero = n - len(scored)
    stats = {
        "book@1": sum(r["book@1"] for r in rows) / n,
        "book@k": sum(r["book@k"] for r in rows) / n,
        "关键词@k": sum(r["kw@k"] for r in rows) / n,
        "平均top1分": (sum(scored) / len(scored)) if scored else float("-inf"),
    }
    t = sum(r["secs"] for r in rows) / n
    print(f"  {name:10s} book@1={stats['book@1']:6.1%}  book@k={stats['book@k']:6.1%}  "
          f"关键词@k={stats['关键词@k']:6.1%}  平均top1分={stats['平均top1分']:7.3f}  "
          f"平均耗时={t:.2f}s"
          + (f"  ⚠️ 零候选 {zero} 条（已剔出平均分）" if zero else ""))
    stats["零候选探针数"] = zero
    return stats


def main():
    ap = argparse.ArgumentParser(description="检索级 A/B（完整链路）")
    ap.add_argument("a", help="基线集合名")
    ap.add_argument("b", help="对照集合名")
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
    except Exception as exc:
        print(f"    ⚠️  自动设备加载失败，回退 CPU: {type(exc).__name__}: {exc}")
        rr = RE.load_reranker_model(device="cpu")
    print(f"    ok: {rr[2]}")

    ra = run_collection(args.a, args.top_k, rr)
    rb = run_collection(args.b, args.top_k, rr)

    print(f"\n{'=' * 78}\n汇总\n{'=' * 78}")
    sa = summarize(args.a, ra)
    sb = summarize(args.b, rb)

    print(f"\n差值（{args.b} − {args.a}）:")
    for label, sa_v in sa.items():
        d = sb[label] - sa_v
        print(f"  {label:10s} {d:+.1%}" if label != "平均top1分" else f"  {label:10s} {d:+.3f}")

    # 逐条列出分歧，便于人工核对到底是哪几条在拉开差距
    diff = [(x["query"], x["book@1"], y["book@1"], x["kw@k"], y["kw@k"])
            for x, y in zip(ra, rb) if x["book@1"] != y["book@1"] or x["kw@k"] != y["kw@k"]]
    if diff:
        print(f"\n分歧探针（{len(diff)} 条）: 查询 | {args.a}:book@1/kw | {args.b}:book@1/kw")
        for q, a1, b1, ak, bk in diff:
            print(f"  {q:14s} | {'Y' if a1 else 'N'}/{'Y' if ak else 'N'} | {'Y' if b1 else 'N'}/{'Y' if bk else 'N'}")
    else:
        print("\n无分歧：两集合在所有探针上的 book@1 与关键词命中完全一致。")
    # 本工具只做报告，不做通过/失败判定，故恒返回 0
    # （显式 return 是为了能被 sys.exit 承接，将来要加门槛时有明确的挂载点）。
    return 0


if __name__ == "__main__":
    sys.exit(main())
