"""pytest 共享 fixture。

设计原则：**分层可跳过**。
  - L0（unit）不碰任何外部资源，永远能跑 —— 这是唯一能上 PR 门禁的一层；
  - L1/L2/L3 各自声明所需资源，缺失时 skip 而不是 fail，
    这样"本机没起 Docker"不会被误读成"代码坏了"。

为什么 L0 能成立：chatbot 把 torch / transformers / jieba 全部放在函数内部
惰性导入，模块顶层只依赖 sentencex。因此纯函数测试无需加载任何模型。
本文件里的 FakeTokenizer 就是利用这一点，用字符数顶替真实 tokenizer。
"""

import os

import pytest

import chatbot as RE


# ============================================================================
# L0：tokenizer 替身
# ============================================================================
class FakeTokenizer:
    """字符级 tokenizer 替身：id 即字符码点，故 encode/decode 可逆。

    chatbot 的分块路径只以两种方式接触 tokenizer：
        len(tokenizer.encode(text, add_special_tokens=False))   # 计数
        tokenizer.decode(ids, skip_special_tokens=True)         # 硬切后还原
    故这里只需实现这两个方法。

    **关键：空白计 0 token。** 这不是随手定的，而是对齐实测事实 ——
    BGE 用的 BERT 式中文分词器会吸收 `" ".join(...)` 产生的分隔空格。
    用真实 bge-base 量过现行 21137 条分块产物：

        子块 最大 128 token（0 条超 128）
        父块 最大 512 token（0 条超 512）

    刚好卡在上限，说明分隔空格确实不产生 token。若替身把空格也算成 token，
    就会伪造出"父块 610 token 超限"这种现实中不存在的失败。

    注意 _split_by_tokens / _split_into_children 是"先累加 token 数、
    再用空格 join"，这个累加只在"tokenizer 不charge分隔空格"时成立。
    真实的守卫是 test_chunk_invariants.py 里用**真 tokenizer** 跑的那条，
    换模型（如 sentencepiece 系的 bge-m3）时必须靠它发现。
    """

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [ord(ch) for ch in text if not ch.isspace()]

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(chr(i) for i in ids)

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, **kwargs):  # noqa: ARG002
        """支持 _hard_split_by_tokens 的无损切片路径。

        该方法会优先用 offset_mapping 把 token 边界还原成原文切片
        （避免 BERT decode 在 token 间插空格）。替身必须提供同样接口，
        否则会静默走带异常的兜底分支，测试就覆盖不到真正要测的路径。
        """
        ids, offsets = [], []
        for i, ch in enumerate(text):
            if not ch.isspace():
                ids.append(ord(ch))
                offsets.append((i, i + 1))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


@pytest.fixture
def fake_tok():
    return FakeTokenizer()


# ============================================================================
# L1：分块产物
# ============================================================================
@pytest.fixture(scope="session")
def chunks():
    """加载 cache_v2/chunks.json；缺失则跳过（提示先跑 python chatbot.py process）。"""
    if not os.path.exists(RE.CHUNKS_JSON):
        pytest.skip(f"缺少 {RE.CHUNKS_JSON}，请先运行: python chatbot.py process")
    import json

    with open(RE.CHUNKS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        pytest.skip(f"{RE.CHUNKS_JSON} 为空")
    return data


@pytest.fixture(scope="session")
def real_tokenizer():
    """真实嵌入模型的 tokenizer（只读词表，不加载权重，约 1s）。

    L1 里"子块/父块不超 token 上限"的守卫必须用真 tokenizer：FakeTokenizer
    是行为替身，无法证明真实分词结果不超限。
    """
    if not os.path.isdir(RE.EMBED_MODEL_PATH):
        pytest.skip(f"嵌入模型目录缺失: {RE.EMBED_MODEL_PATH}")
    try:
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover
        pytest.skip("未安装 transformers")
    return AutoTokenizer.from_pretrained(RE.EMBED_MODEL_PATH)


# ============================================================================
# L2：Qdrant
# ============================================================================
@pytest.fixture(scope="session")
def qdrant_client():
    """连接 Qdrant；不可达则跳过。"""
    try:
        from qdrant_client import QdrantClient
    except ImportError:  # pragma: no cover
        pytest.skip("未安装 qdrant-client")
    client = QdrantClient(host=RE.QDRANT_HOST, port=RE.QDRANT_PORT, timeout=30)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant 不可达 ({RE.QDRANT_HOST}:{RE.QDRANT_PORT}): {exc}")
    return client


@pytest.fixture(scope="session")
def collection_name(qdrant_client):
    """当前该访问的集合名 —— **必须运行时解析**（别名优先）。

    直接用 RE.COLLECTION_NAME 会在启用别名后出错：那时具体集合名是
    `books_v3__<build_id前8位>`，而旧名 books_v3 已被删除，于是所有依赖集合的
    用例都会走 "if not collection_exists: pytest.skip" 这条**跳过**分支 ——
    L2/L3 会静默不跑却显示绿色。这比报错危险得多：一个"跳过"的测试套件
    看起来和"全通过"一模一样。
    """
    name = RE.default_collection_name(qdrant_client)
    if not qdrant_client.collection_exists(name):
        pytest.skip(f"集合 {name} 不存在，请先运行: python chatbot.py index")
    return name


@pytest.fixture(scope="session")
def collection_info(qdrant_client, collection_name):
    """确认目标集合存在，返回集合信息。"""
    return qdrant_client.get_collection(collection_name=collection_name)


# ============================================================================
# L3：全栈引擎
# ============================================================================
@pytest.fixture(scope="session")
def engine(qdrant_client):
    """完整 RAGEngine（会加载嵌入 + 重排模型，耗时）。"""
    eng = RE.RAGEngine()
    if eng.count() == 0:
        # count() 内部已做了别名解析；这里回报它实际用的集合名，
        # 免得"集合 books_v3 为空"这种提示把你引向一个已经不存在的名字
        pytest.skip(f"集合 {eng.collection_name} 为空，请先运行: python chatbot.py index")
    return eng
