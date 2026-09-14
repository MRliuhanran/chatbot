"""对比三种稀疏分词方案在真实语料上的召回表现（打分方案统一用 Qdrant 内置 BM25）：
  1. jieba 分词 + Qdrant BM25   ← 当前实现
  2. Qdrant multilingual 分词器 + Qdrant BM25
  3. Qdrant 默认 word 分词器 + Qdrant BM25（对照组：中文等于不切词）
稠密侧全部取 RE.COLLECTION_NAME 的同一路稠密召回，融合用同一套 RRF，只换稀疏通道。
只建稀疏向量，不需要跑 embedding，约 1 分钟可完成。
"""
import json
import time

import rag_engine as RE
from qdrant_client import QdrantClient, models

PROBES = [("武松打虎", "武松"), ("黛玉葬花", "葬花"), ("桃园结义", "桃园"),
          ("火烧赤壁", "赤壁"), ("倒拔垂杨柳", "垂杨"), ("大闹天宫", "天宫"),
          ("刘姥姥进大观园", "刘姥姥"), ("空城计", "空城")]
TOP_K = 30
TMP = "books_bm25_probe"
MULTI_OPTS = {"tokenizer": "multilingual", "stemmer": {"type": "none"},
              "stopwords": {}, "ascii_folding": True}


def main():
    client = QdrantClient(host=RE.QDRANT_HOST, port=RE.QDRANT_PORT, timeout=120)
    chunks = json.load(open(RE.CHUNKS_JSON, encoding="utf-8"))
    texts = [c["child_text"] for c in chunks]

    for name in (TMP + "_multi", TMP + "_word"):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    # 建两个临时集合：multilingual vs 默认 word 分词器
    # （jieba 方案直接用目标集合里已写入的稀疏向量，不再单独建集合）
    JIEBA_OPTS = dict(RE.BM25_TEXT_OPTIONS)
    build = {}
    for suffix, opts in [("_multi", MULTI_OPTS), ("_word", None)]:
        cname = TMP + suffix
        client.create_collection(cname, sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)})
        t0 = time.time()
        for i in range(0, len(texts), 1000):
            batch = texts[i:i + 1000]
            client.upsert(cname, points=[
                models.PointStruct(id=i + j, vector={"sparse": models.Document(
                    text=t, model="qdrant/bm25", options=opts) if opts else
                    models.Document(text=t, model="qdrant/bm25")})
                for j, t in enumerate(batch)])
        build[suffix] = time.time() - t0
        n_terms = client.scroll(cname, limit=1, with_vectors=True)[0][0].vector["sparse"]
        print(f"  {cname}: {len(texts)} 条, 入库 {build[suffix]:.1f}s, "
              f"首条 {len(n_terms.indices)} terms")

    eng = RE.RAGEngine()
    store = eng._get_chunks()
    tokenizer, model, device = eng._get_embed()

    def recall(ctxs, get_hits):
        hit_sum = ceil_sum = 0
        rows = []
        for query, kw in PROBES:
            n = sum(1 for c in store.values() if kw in c["child_text"])
            if n == 0:
                continue
            ceiling = min(n, TOP_K)
            qv = RE.embed([query], tokenizer, model, device, is_query=True)[0].tolist()
            ids = get_hits(qv, query)
            got = sum(1 for i in ids if kw in store[i]["child_text"])
            hit_sum += min(got, ceiling)
            ceil_sum += ceiling
            rows.append((query, got, ceiling))
        return hit_sum, ceil_sum, rows

    def dense_only(qv, _q):
        return [p.id for p in client.query_points(
            RE.COLLECTION_NAME, query=qv, using=RE.DENSE_VECTOR_NAME, limit=TOP_K).points]

    def hybrid_with(cname, qdrant_opts, jieba=False, k=60):
        """直接对目标集合做稀疏查询，稠密侧固定取同一集合，客户端 RRF 融合
        （临时集合只有稀疏空间，跨集合无法用服务端 RRF）。"""
        def f(qv, q):
            dense = client.query_points(RE.COLLECTION_NAME, query=qv,
                                        using=RE.DENSE_VECTOR_NAME, limit=TOP_K).points
            if jieba:
                sp = RE.sparse_encode(q)                       # jieba 分词 + Qdrant BM25
            else:
                sp = models.Document(text=q, model="qdrant/bm25", options=qdrant_opts)
            sparse = client.query_points(cname, query=sp, using="sparse", limit=TOP_K).points
            scores = {}
            for rank, p in enumerate(dense):
                scores[p.id] = scores.get(p.id, 0.0) + 1.0 / (k + rank + 1)
            for rank, p in enumerate(sparse):
                scores[p.id] = scores.get(p.id, 0.0) + 1.0 / (k + rank + 1)
            return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])[:TOP_K]]
        return f

    print("\n  查询              上限   纯稠密   jieba    Qdrant-multilingual")
    d_sum, ceil, rows = recall(None, dense_only)
    j_sum, _, _ = recall(None, hybrid_with(RE.COLLECTION_NAME, None, jieba=True))
    qm_sum, _, _ = recall(None, hybrid_with(TMP + "_multi", MULTI_OPTS))
    qw_sum, _, _ = recall(None, hybrid_with(TMP + "_word", None))
    for (query, dk, ceiling) in rows:
        print(f"  {query:<16}{ceiling:>4}{dk:>8}")
    print(f"\n  合计召回 上限 {ceil}")
    print(f"    纯稠密              {d_sum}/{ceil}  ({d_sum/ceil*100:.1f}%)")
    print(f"    jieba + Qdrant BM25 {j_sum}/{ceil}  ({j_sum/ceil*100:.1f}%)  ← 当前")
    print(f"    Qdrant multilingual {qm_sum}/{ceil}  ({qm_sum/ceil*100:.1f}%)")
    print(f"    Qdrant word(默认)   {qw_sum}/{ceil}  ({qw_sum/ceil*100:.1f}%)")

    for suf in ("_multi", "_word"):
        client.delete_collection(TMP + suf)
    print("\n  临时集合已清理")


if __name__ == "__main__":
    main()
