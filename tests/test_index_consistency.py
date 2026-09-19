"""L2：索引一致性 —— 需要 Qdrant 在运行。

这一层回答的是"索引和分块源是否已经脱节"。此前没有任何自动检查，
导致文档里长期写着错误的索引条数，却无人发现。

需要: docker compose up -d  +  python chatbot.py index
运行: pytest -m needs_qdrant
"""

import pytest

import chatbot as RE

pytestmark = pytest.mark.needs_qdrant

# 运行时解析出来的集合名（别名优先）。**不要直接用 COLL**：
# 启用别名后具体集合名是 books_v3__<build_id前8位>，旧名会被删除，
# 而这些断言会因此全部失败在一个"已经不存在的名字"上 —— 排查方向完全错误。
COLL = None


@pytest.fixture(scope="module", autouse=True)
def _bind_collection(collection_name):
    global COLL
    COLL = collection_name
    return collection_name


@pytest.fixture(scope="module")
def all_payloads(qdrant_client, collection_info):
    """全量拉取 payload（分页），供多条断言复用。"""
    out = []
    offset = None
    while True:
        pts, offset = qdrant_client.scroll(
            collection_name=COLL,
            limit=RE.SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        out.extend(pts)
        if offset is None:
            break
    return out


# ============================================================================
# 集合配置
# ============================================================================
class TestCollectionConfig:
    def test_collection_exists(self, qdrant_client):
        assert qdrant_client.collection_exists(COLL), (
            f"集合 {COLL} 不存在，请运行: python chatbot.py index"
        )

    def test_has_both_vector_spaces(self, collection_info):
        """稠密 + 稀疏两个空间都必须存在，否则不是混合检索。"""
        dense = list(collection_info.config.params.vectors.keys())
        sparse = list(collection_info.config.params.sparse_vectors.keys())
        assert RE.DENSE_VECTOR_NAME in dense, f"缺稠密空间 {RE.DENSE_VECTOR_NAME}: {dense}"
        assert RE.SPARSE_VECTOR_NAME in sparse, f"缺稀疏空间 {RE.SPARSE_VECTOR_NAME}: {sparse}"

    def test_dense_dim_matches_model(self, collection_info):
        """集合稠密维度必须与嵌入模型输出维度一致（不符则查询侧向量根本进不去）。"""
        vecs = collection_info.config.params.vectors
        size = vecs[RE.DENSE_VECTOR_NAME].size
        assert size == 768, (
            f"稠密维度 {size} 与 bge-base-zh-v1.5 的 768 不符"
            f"（换模型后必须新建集合，旧集合无法增量迁移）"
        )

    def test_sparse_uses_idf_modifier(self, collection_info):
        """稀疏必须启用 IDF 修饰符 —— Qdrant 内置 BM25 的打分方案依赖它。"""
        from qdrant_client.models import Modifier

        cfg = collection_info.config.params.sparse_vectors[RE.SPARSE_VECTOR_NAME]
        assert cfg.modifier == Modifier.IDF, (
            f"稀疏空间 modifier={cfg.modifier}，不是 IDF —— BM25 打分会失真"
        )


# ============================================================================
# 数量与编号
# ============================================================================
class TestCounts:
    def test_count_matches_chunks_json(self, qdrant_client, collection_info, chunks):
        """集合点数必须等于 chunks.json 条数。

        这是"索引与分块源是否脱节"的一行判定。不一致意味着：
        要么索引没重建（服务在跑旧数据），要么 process 后忘了 index。
        """
        count = qdrant_client.count(collection_name=COLL, exact=True).count
        assert count == len(chunks), (
            f"集合 {COLL} 有 {count} 条，但 chunks.json 有 {len(chunks)} 条 —— "
            f"索引与分块源已脱节，请重跑: python chatbot.py index"
        )

    def test_point_ids_contiguous(self, all_payloads, collection_info):
        """point id 必须是 0..n-1 连续整数。

        出处：build_index 用全局序号 i+j 作 point id（Qdrant 只接受无符号整数
        或 UUID），原始 id 存在 payload.chunk_id。若出现空洞，说明写入过程有丢失。
        """
        ids = sorted(p.id for p in all_payloads)
        assert ids == list(range(len(ids))), (
            f"point id 不连续: 共 {len(ids)} 条，范围 {ids[0]}..{ids[-1]}"
        )

    def test_chunk_ids_match_chunks_json(self, all_payloads, chunks):
        """payload.chunk_id 的集合必须与 chunks.json 的 id 集合完全一致。"""
        got = set(p.payload.get("chunk_id") for p in all_payloads)
        want = set(c["id"] for c in chunks)
        assert got == want, (
            f"chunk_id 集合不一致: 索引多 {len(got - want)} 个、少 {len(want - got)} 个；"
            f"样例 多={sorted(got - want)[:3]} 少={sorted(want - got)[:3]}"
        )


# ============================================================================
# 向量写入
# ============================================================================
class TestVectors:
    def test_all_points_have_both_vectors(self, qdrant_client):
        """抽样确认每条点都同时写了稠密与稀疏向量。

        Qdrant 不会自动生成稀疏向量：不显式写入就没有，**且不报错**。
        这曾让"混合检索"长期退化为纯稠密而无人发现 —— 所以这条必须真实抽查，
        而不是判断 hasattr(config)（旧版 check_health 就是这么做的，恒为真）。
        """
        pts, _ = qdrant_client.scroll(
            collection_name=COLL,
            limit=200,
            with_payload=False,
            with_vectors=True,
        )
        assert pts, "集合为空"
        miss_dense = [p.id for p in pts if not p.vector.get(RE.DENSE_VECTOR_NAME)]
        miss_sparse = [p.id for p in pts if not p.vector.get(RE.SPARSE_VECTOR_NAME)]
        assert not miss_dense, f"{len(miss_dense)} 条缺稠密向量: {miss_dense[:5]}"
        assert not miss_sparse, (
            f"{len(miss_sparse)} 条缺稀疏向量: {miss_sparse[:5]}"
            f"（稀疏没写 = 混合检索名存实亡）"
        )

    def test_dense_vectors_normalized(self, qdrant_client):
        """稠密向量必须是 L2 归一化的（BGE 用法；未归一化则余弦相似度失效）。"""
        import math

        pts, _ = qdrant_client.scroll(
            collection_name=COLL,
            limit=5,
            with_payload=False,
            with_vectors=True,
        )
        for p in pts:
            v = p.vector[RE.DENSE_VECTOR_NAME]
            norm = math.sqrt(sum(x * x for x in v))
            assert abs(norm - 1.0) < 1e-3, f"point {p.id} 稠密向量模长 {norm}，未归一化"

    def test_sparse_single_channel_recall(self, qdrant_client, all_payloads):
        """稀疏单通道必须能用原文召回它自己（证明稀疏索引真的生效）。"""
        probe = all_payloads[0].payload.get("child_text", "")[:80]
        assert probe, "样本 child_text 为空"
        res = qdrant_client.query_points(
            collection_name=COLL,
            query=RE.sparse_encode(probe),
            using=RE.SPARSE_VECTOR_NAME,
            limit=3,
        ).points
        assert res, "稀疏单通道检索返回空 —— 稀疏索引未生效"


# ============================================================================
# build_id
# ============================================================================
class TestBuildId:
    def test_all_points_share_one_build_id(self, all_payloads):
        """同一次建索引写入的所有点必须共享同一个 build_id。"""
        ids = set(p.payload.get(RE.INDEX_BUILD_ID_FIELD) for p in all_payloads)
        assert len(ids) == 1, (
            f"出现 {len(ids)} 种 build_id: {list(ids)[:5]} —— "
            f"说明索引被分多次写入或写入过程被中断"
        )
        assert None not in ids, (
            f"存在没有 {RE.INDEX_BUILD_ID_FIELD} 字段的点（旧数据？）"
            f"—— 检索端据此判断索引换代，缺失会导致缓存永不失效"
        )

    def test_build_id_readable_at_point_zero(self, qdrant_client):
        """_current_build_id() 读 point 0 判定换代，故 point 0 必须存在且带该字段。"""
        pts = qdrant_client.retrieve(
            collection_name=COLL, ids=[0], with_payload=True
        )
        assert pts, "point id=0 不存在 —— _current_build_id() 会取不到标识"
        assert (pts[0].payload or {}).get(RE.INDEX_BUILD_ID_FIELD) is not None


# ============================================================================
# 与引擎自身的一致性
# ============================================================================
class TestEngineAgreement:
    def test_engine_chunk_table_complete(self, qdrant_client):
        """RAGEngine._get_chunks() 必须能完整加载（内部有数量自校验）。

        回归：旧实现 scroll(limit=100000) 只读第一页且不校验，语料越过阈值后
        会静默返回残缺的分块表，hybrid_search 里 "if cid in chunks" 再把命中
        无声丢弃 —— 召回悄悄变差、无异常无日志。
        """
        engine = RE.RAGEngine()
        table = engine._get_chunks()
        expected = qdrant_client.count(collection_name=COLL, exact=True).count
        assert len(table) == expected, (
            f"_get_chunks 加载 {len(table)} 条，集合实际 {expected} 条"
        )
