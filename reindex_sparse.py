"""只重建稀疏向量，不重算稠密 embedding。

用途一：换了稀疏方案（分词器 / 词表 / 权重算法）时重算稀疏通道。
用途二（更常见）：**改了 data/aliases.txt 或 data/stopwords_classical.txt 之后**。
        词表改变的是稀疏通道的**词空间** —— 文档侧稀疏向量是建索引时算好的，
        而查询侧每次现算。两边用了不同版本的词表，BM25 就不再是同一空间里的
        匹配，而会静默错配（例如把"孔明"归一到"诸葛亮"之后，索引里存的仍是
        "孔明"，于是查询"孔明"反而再也匹配不到写"孔明"的正文）。
        检索端与 check_health.py 都会比对词表指纹并报警，本脚本负责修复它。

为什么不必重算稠密：稠密向量取自 child_text 原文，词表不参与，故完全没变。
对比：全量 build_index 要重跑所有 embedding（本机 30~70 分钟）；本脚本十几秒。

用法: python reindex_sparse.py
"""
import sys
import time

import rag_engine as RE
from qdrant_client import QdrantClient, models

BATCH = 500


def main():
    client = QdrantClient(host=RE.QDRANT_HOST, port=RE.QDRANT_PORT, timeout=120)
    # 集合名运行时解析（别名优先），见 rag_engine.default_collection_name
    COLL = RE.default_collection_name(client)
    info = client.get_collection(COLL)
    total = info.points_count
    print(f"集合 {COLL}: {total} 条")
    print(f"稀疏方案: Qdrant 内置 BM25 + jieba 分词，options={RE.BM25_TEXT_OPTIONS}")

    if not total:
        # 空集合必须**失败退出**：否则两处一致性检查（更新条数、指纹条数）
        # 都会 0==0 恒真，脚本最后打印"完成"并 return 0 —— 对"索引根本没建"
        # 这件事给出绿色结论。
        print(f"❌ 集合 {COLL} 为空（索引未构建或别名指向了空集合），无事可做")
        print("   请先运行: python app.py process && python app.py index")
        return 1

    t0 = time.time()
    done = 0
    lexicon_id = RE.lexicon_fingerprint()
    offset = None
    while True:
        pts, offset = client.scroll(
            COLL, limit=BATCH, offset=offset,
            with_payload=True, with_vectors=False)
        if not pts:
            break
        ids = [p.id for p in pts]
        points = [
            models.PointVectors(
                id=p.id,
                vector={RE.SPARSE_VECTOR_NAME: RE.sparse_encode(p.payload.get("child_text", ""))},
            )
            for p in pts
        ]
        client.update_vectors(COLL, points=points)
        # 词表指纹在**同一趟**里回写：分两趟写会在中途中断时留下
        # "稀疏已换、指纹未换"的错配状态，而那时的自检恰好是绿的。
        client.set_payload(collection_name=COLL,
                           payload={RE.LEXICON_ID_FIELD: lexicon_id},
                           points=ids)
        done += len(pts)
        print(f"  {done}/{total}  ({time.time() - t0:.1f}s)")
        if offset is None:
            break

    print(f"词表指纹已回写: {lexicon_id}（{done} 条）")
    if done != total:
        print(f"❌ 只更新了 {done}/{total} 条，请重跑")
        return 1

    # 抽样校验
    sample, _ = client.scroll(COLL, limit=5, with_vectors=True)
    for p in sample:
        sp = p.vector.get(RE.SPARSE_VECTOR_NAME)
        print(f"  校验 point {p.id}: 稠密 {len(p.vector.get(RE.DENSE_VECTOR_NAME, []))} 维, "
              f"稀疏 {len(sp.indices) if sp else 0} terms")

    print(f"完成: {done} 条，耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
