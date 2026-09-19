"""Qdrant 混合检索验证：确认稠密+稀疏真的落库、真的参与召回。

做三件事：
  1. 静态检查：集合是否同时具备稠密/稀疏空间，点上两种向量是否都写了；
  2. 通道检查：稠密单通道、稀疏单通道分别能否召回；
  3. 对照检查：纯稠密 vs 稠密+稀疏(RRF) 在同一批查询上的关键词命中对比。

注意：第 3 项的"关键词命中率"只是粗略代理指标（看 top-k 正文里有没有出现
查询的显著词），不是标准 Recall 评测，仅用于判断混合通道有没有起作用。

用法: python verify_qdrant.py
"""
import sys

import rag_engine as RE
from qdrant_client import QdrantClient

# 探针统一来自 tests/probes.py —— 此前这里与 compare_ab.py 各写了一份副本。
from tests.probes import PROBES

TOP_K = 30


def main():
    client = QdrantClient(host=RE.QDRANT_HOST, port=RE.QDRANT_PORT, timeout=60)
    # 集合名运行时解析（别名优先）：启用别名后具体集合名是 books_v3__<build_id>，
    # 写死 COLL 会在第一次别名重建后集体报"集合不存在"。
    COLL = RE.default_collection_name(client)

    print("=" * 68)
    print("[1] 集合静态检查")
    # 集合不存在时直接 count()/get_collection() 会抛 404，报出来的却是
    # "连接失败"这类误导性信息。先判断存在性，与 RAGEngine.count() 一致。
    if not client.collection_exists(COLL):
        print(f"  ❌ 集合 {COLL} 不存在，请先运行: python app.py index")
        return 1
    info = client.get_collection(COLL)
    dense = list(info.config.params.vectors.keys())
    sparse = list(info.config.params.sparse_vectors.keys())
    count = client.count(COLL, exact=True).count
    print(f"  集合: {COLL}  点数: {count}")
    print(f"  稠密空间: {dense}  (size={info.config.params.vectors[dense[0]].size if dense else '缺失'})")
    print(f"  稀疏空间: {sparse}")

    # 与 check_health.py 同一实现（RE.sample_vector_coverage）。此前这里自己
    # 下标 `pts[0].vector[SPARSE].indices`，遇到链式首条缺稀疏向量时自己抛
    # KeyError —— 在它最该报出"稀疏没写"的场景下崩掉；check_health 那份多了守卫
    # 所以没事。同一检查两个实现，一个有 bug 一个没有。
    cov = RE.sample_vector_coverage(client, COLL, limit=100)
    miss_d, miss_s, n_terms = cov["miss_dense"], cov["miss_sparse"], cov["first_sparse_terms"]
    print(f"  抽样 {cov['n']} 条: 缺稠密 {miss_d}, 缺稀疏 {miss_s} (首条稀疏 {n_terms} term)")

    # 稠密/稀疏的缺失必须**对称**报告：只报稀疏、不报稠密，等于把"方向不对称"
    # 这个缺陷本身固化下来（稠密缺失同样让检索失效，只是失效方式不同）。
    fail = 0
    if not dense:
        print("  ❌ 集合没有稠密向量空间 —— 稠密通道名存实亡")
        fail = 1
    if not sparse:
        print("  ❌ 集合没有稀疏向量空间 —— 混合检索名存实亡")
        fail = 1
    if miss_d:
        print(f"  ❌ 抽样中有 {miss_d} 条缺稠密向量")
        fail = 1
    if miss_s:
        print(f"  ❌ 抽样中有 {miss_s} 条缺稀疏向量 —— 混合检索名存实亡")
        fail = 1
    if not cov["n"]:
        print("  ⚠️  集合为空，抽样检查无从进行")

    print("\n[2] 通道检查（同一查询）")
    q = "武松打虎"
    dv = RE.embed([q], *RE.load_embedding_model(), is_query=True)[0].tolist()
    dense_hits, sparse_hits = RE.dual_channel_recall(
        client, COLL, dv, RE.sparse_encode(q), 5)
    print(f"  稠密通道: {len(dense_hits)} 条, 最高 {dense_hits[0].score:.3f}" if dense_hits else "  稠密通道: 空")
    print(f"  稀疏通道: {len(sparse_hits)} 条, 最高 {sparse_hits[0].score:.3f}" if sparse_hits else "  稀疏通道: 空")

    print("\n[3] 纯稠密 vs 稠密+稀疏(RRF) 关键词召回对比")
    print("    召回率分母 = min(语料中含该词的块数, TOP_K)，即该查询可达的召回上限。")
    print("    分母为 0 说明此词在语料中不存在，属于无效探针，直接跳过。\n")
    eng = RE.RAGEngine()
    chunks = eng._get_chunks()
    tokenizer, model, device = eng._get_embed()

    def hybrid(qv, q_text):
        """稠密 + 稀疏 → 加权 RRF，**与本项目线上实现同一套**。

        召回走 RE.dual_channel_recall + RE.weighted_rrf：线上已改为客户端加权
        RRF，工具若自己拼一份（或用 Qdrant 服务端 FusionQuery），度量到的召回率
        就不是线上的召回率 —— 而"召回通道校验"这个工具存在的全部意义就是回答
        "线上到底召回了多少"。这个副本此前真的跑偏过：它把两通道结果**拼接而不
        去重**，于是混合侧命中数可以超过分母上限（打印出 4/2），而稠密侧天然无
        重复，两侧口径不对等。
        """
        dense_hits, sparse_hits = RE.dual_channel_recall(
            client, COLL, qv, RE.sparse_encode(q_text), TOP_K)
        order = {pid: i for i, (pid, _s, _r) in
                 enumerate(RE.weighted_rrf(RE.channel_rankings(dense_hits, sparse_hits),
                                           k=RE.RRF_K, limit=TOP_K))}
        best = {}
        for p in dense_hits + sparse_hits:
            best.setdefault(p.id, p)
        return sorted(best.values(), key=lambda p: order.get(p.id, 10**9))[:TOP_K]

    print(f"  {'查询':<16}{'语料块数':>8}{'上限':>6}{'纯稠密':>9}{'混合RRF':>9}   {'召回率变化':<12}")
    dense_hit_sum = hybrid_hit_sum = ceiling_sum = 0
    for probe in PROBES:
        query = probe.query

        def kw_in(text):
            return any(k in text for k in probe.keywords)

        corpus_n = sum(1 for c in chunks.values() if kw_in(c["child_text"]))
        if corpus_n == 0:
            print(f"  {query:<16}{corpus_n:>8}   —— 跳过（语料中无此词）")
            continue
        ceiling = min(corpus_n, TOP_K)

        qv = RE.embed([query], tokenizer, model, device, is_query=True)[0].tolist()

        d = client.query_points(COLL, query=qv,
                                using=RE.DENSE_VECTOR_NAME, limit=TOP_K).points
        h = hybrid(qv, query)

        def hit_count(res):
            return sum(1 for p in res
                       if (c := chunks.get(p.id)) and kw_in(c["child_text"]))

        dk, hk = hit_count(d), hit_count(h)
        # 命中数超过分母上限 = 去重/口径出错了，工具必须先发现自己坏了。
        # （这条曾经真的会触发：混合路是两通道拼接、未去重时同一个块被数两次。）
        if hk > ceiling or dk > ceiling:
            print(f"  ❌ {query}: 命中数超过上限（稠密 {dk} / 混合 {hk} > {ceiling}）"
                  f"—— 工具自身的计数口径有误，结果不可信")
            fail = 1
        dense_hit_sum += min(dk, ceiling)
        hybrid_hit_sum += min(hk, ceiling)
        ceiling_sum += ceiling
        if hk == ceiling and dk < ceiling:
            change = "→ 补全召回"
        elif hk > dk:
            change = "→ 提升"
        elif hk == dk:
            change = "→ 持平"
        else:
            change = "→ 下降"
        print(f"  {query:<16}{corpus_n:>8}{ceiling:>6}"
              f"{dk:>6}/{ceiling}{hk:>6}/{ceiling}   {change}")

    if ceiling_sum:
        print(f"\n  合计召回: 纯稠密 {dense_hit_sum}/{ceiling_sum}"
              f" ({dense_hit_sum/ceiling_sum*100:.1f}%)"
              f"  混合 {hybrid_hit_sum}/{ceiling_sum}"
              f" ({hybrid_hit_sum/ceiling_sum*100:.1f}%)")

    if fail:
        print("\n❌ 校验失败（见上）")
    else:
        print("\n✅ 校验通过")
    return fail


if __name__ == "__main__":
    sys.exit(main())
