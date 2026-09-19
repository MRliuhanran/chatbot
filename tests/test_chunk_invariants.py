"""L1：分块产物不变量 —— 只需 cache_v2/chunks.json（+ 真 tokenizer 读词表）。

这一层把"分块流水线应该满足的性质"写成可执行断言。此前这些性质只存在于
代码注释里（"实测 254 个子块超限"、"0% 父块被机械切分"），注释不会被运行，
所以回归发生时无人知晓。

需要: python rag.py process  (生成 chunks.json)
运行: pytest -m needs_chunks
"""

import re
from collections import Counter, defaultdict

import pytest

import rag as RE

pytestmark = pytest.mark.needs_chunks

# 逐字空格损伤特征：**连续三个**中文字各自被空格隔开。
#
# 为什么必须是"连续三个"而不是两个：语料里本来就存在合法的中文间空格 ——
# 回目/标题行形如「《三国演义》罗贯中 第一回 宴桃园豪杰三结义 斩黄巾英雄首立功」，
# 用两字模式会把 1846 条正常记录误判为损坏（旧新两版产物都是这个量级，
# 正说明它们与本次缺陷无关）。而硬切损伤会让**每个字**之间都插空格，
# 三字连续出现的模式在正常语料中为 0 条。
#
# 实测对照（同一套判定）：
#   修复前 chunks.json：9 条命中（全部出自西游记经文清单/难数清单）
#   修复后 chunks.json：0 条命中
_CHAR_SPACING = re.compile(r"[\u4e00-\u9fff] [\u4e00-\u9fff] [\u4e00-\u9fff]")


def _norm_ws(s):
    """去掉所有空白，用于"忽略空白"的内容比对。

    父子文本是各自 " ".join(...) 出来的，切分粒度不同会多出分隔空格，
    因此内容包含关系必须按去空白后判断（实测 2011 条严格失败中 2010 条属此类）。
    """
    return re.sub(r"\s+", "", s)


# chunks.json 里每条记录必须具备的字段
REQUIRED_FIELDS = [
    "child_text",
    "parent_text",
    "book",
    "chunk_index",
    "total_chunks",
    "contextual_text",
    "id",
    # 按回分段引入的结构化元数据（检索侧要用它做去重、过滤与引用溯源）
    "parent_id",
    "parent_chunk_count",
    "chapter_index",
    "chapter_label",
    "chapter_title",
]


@pytest.fixture(scope="module")
def by_book(chunks):
    d = defaultdict(list)
    for c in chunks:
        d[c["book"]].append(c)
    return d


# ============================================================================
# 结构
# ============================================================================
class TestStructure:
    def test_all_fields_present(self, chunks):
        for i, c in enumerate(chunks):
            missing = [f for f in REQUIRED_FIELDS if f not in c]
            assert not missing, (
                f"第 {i} 条缺字段 {missing}。\n"
                f"若缺的是 parent_id / chapter_*，通常是磁盘上的 chunks.json 仍为"
                f"引入按回分段之前的旧产物 —— 请重跑: python rag.py process"
            )

    def test_child_text_non_empty(self, chunks):
        for i, c in enumerate(chunks):
            assert c["child_text"].strip(), f"第 {i} 条 child_text 为空"

    def test_short_chunks_are_real_text(self, chunks):
        """短块必须**被保留**，且必须是真文字（不是空白）。

        这条此前断言的是相反的事：`len(child_text) <= 5` 的块不存在 ——
        而那正是旧 `len(child_text) > 5` 过滤判据的不变量，也就是把
        "丢正文"当成了正确行为。实测四本书因此共丢 14 字
        （红楼的 `宝玉又道：`、西游的 `莫念！`），而短块走的是
        "父子相同"分支，parent_text == child_text，丢的不是重复内容而是原文。

        新判据：只丢**纯空白**块（那里面没有任何文字，丢了无损）。
        """
        short = [c for c in chunks if len(c["child_text"]) <= 5]
        # 短块允许存在；但每一条都必须含真文字
        for c in short:
            assert c["child_text"].strip(), (
                f"{c['id']} 是纯空白块，不该出现在产物里（无正文可丢）"
            )
        # 全文里不得出现纯空白块
        blanks = [c["id"] for c in chunks if not c["child_text"].strip()]
        assert not blanks, f"存在纯空白块: {blanks[:5]}"

    def test_parent_contains_child(self, chunks):
        """父子结构的核心契约：子块内容必须完整出现在父块中。

        这条一旦破坏，返回给 LLM 的 parent_text 与命中的 child_text 就对不上，
        等于上下文张冠李戴。

        **判据忽略空白**，因为两侧是各自独立 join 出来的句子序列：
        父块按 PARENT_MAX_TOKENS 粒度切、子块再按 CHILD_MAX_TOKENS 粒度切，
        子块侧的切分点会多插入一个分隔空格。实测 21137 条中严格子串失败
        2011 条，而其中 **2010 条是纯空白差异**（占 100%）—— 只有空白归一后
        仍不包含的那 1 条才是真正的问题（见 test_no_character_spacing_damage）。
        """
        bad = [
            c["id"] for c in chunks
            if _norm_ws(c["child_text"]) not in _norm_ws(c["parent_text"])
        ]
        assert not bad, (
            f"{len(bad)} 条的子块（忽略空白后）仍不在父块中: {bad[:5]}"
        )

    def test_no_character_spacing_damage(self, chunks):
        """分块文本不得出现"逐字空格"损伤。

        回归：_hard_split_by_tokens 旧实现用 tokenizer.decode(ids[a:b]) 还原切片，
        BERT 式分词器的 decode 会在 token 之间插空格，中文一字一 token，
        于是无标点的"二尊者即开报"变成"二 尊 者 即 开 报"。
        实测损伤 9 个子块、6 个父块（全部来自西游记经文清单与难数清单）——
        这些块的稠密向量与 BM25 稀疏向量都建立在被改坏的文本上。
        """
        bad = [
            c["id"] for c in chunks
            if _CHAR_SPACING.search(c["child_text"]) or _CHAR_SPACING.search(c["parent_text"])
        ]
        assert not bad, (
            f"{len(bad)} 条的分块文本出现逐字空格损伤（硬切有损）: {bad[:5]}"
        )

    def test_books_are_expected(self, chunks):
        books = set(c["book"] for c in chunks)
        assert books, "没有任何书籍"
        # 书记录名来自文件名，不应带扩展名
        for b in books:
            assert not b.endswith(".txt"), f"book 字段残留扩展名: {b}"


# ============================================================================
# 编号一致性（回归 build_chunks 的 enumerate 位置）
# ============================================================================
class TestIndexing:
    def test_chunk_index_contiguous_per_book(self, by_book):
        """chunk_index 必须从 0 起连续，不留空洞。

        回归：旧实现用 enumerate 在**过滤前**编号，被丢掉的短块会留下空洞 ——
        实测红楼梦 0..5011 却只有 5010 条、西游记 0..7463 却只有 7461 条。
        """
        for book, rows in sorted(by_book.items()):
            idx = sorted(r["chunk_index"] for r in rows)
            expected = list(range(len(rows)))
            assert idx == expected, (
                f"{book} 的 chunk_index 不连续: 共 {len(rows)} 条，"
                f"范围 {idx[0]}..{idx[-1]}，缺失 {sorted(set(expected) - set(idx))[:5]}"
            )

    def test_total_chunks_matches_actual(self, by_book):
        """total_chunks 必须等于该书实际条数。

        回归：旧实现写的是过滤前的块数 —— 红楼梦写 5012（实际 5010）、
        西游记写 7464（实际 7461）。
        """
        for book, rows in sorted(by_book.items()):
            totals = set(r["total_chunks"] for r in rows)
            assert totals == {len(rows)}, (
                f"{book} 的 total_chunks={totals} 与实际条数 {len(rows)} 不符"
            )

    def test_id_derived_from_book_and_index(self, chunks):
        """id 必须等于 f"{book}_{chunk_index}"（下游按它回填正文）。"""
        bad = [c["id"] for c in chunks if c["id"] != f"{c['book']}_{c['chunk_index']}"]
        assert not bad, f"{len(bad)} 条 id 与 book/chunk_index 不符: {bad[:5]}"

    def test_ids_unique(self, chunks):
        dupes = [k for k, v in Counter(c["id"] for c in chunks).items() if v > 1]
        assert not dupes, f"id 重复: {dupes[:5]}"


# ============================================================================
# contextual_text 的刻意退化
# ============================================================================
class TestContextualText:
    def test_equals_child_text(self, chunks):
        """contextual_text 恒等于 child_text —— 这是刻意的（已关闭上下文前缀）。

        把它写成断言而不是注释：将来若有人重新引入前缀拼接，这条会立刻失败，
        迫使改动者明确这是行为变更（而不是悄悄改变 rerank/稠密索引的输入）。
        出处：rag.build_chunks 里 "contextual_text == child_text" 的注释。
        """
        bad = [c["id"] for c in chunks if c["contextual_text"] != c["child_text"]]
        assert not bad, f"{len(bad)} 条 contextual_text != child_text: {bad[:5]}"


# ============================================================================
# 尺寸不变量 —— 必须用真 tokenizer
# ============================================================================
class TestTokenLimits:
    """用真实 tokenizer 验证尺寸上限。

    FakeTokenizer 无法替代这一层：它只是行为替身，证明不了真实分词不超限。
    这里用的就是建索引时用的那个 embed 模型词表。
    """

    def _n(self, tok, text):
        return len(tok.encode(text, add_special_tokens=False))

    def test_child_within_limit(self, chunks, real_tokenizer):
        over = [
            (c["id"], self._n(real_tokenizer, c["child_text"]))
            for c in chunks
            if self._n(real_tokenizer, c["child_text"]) > RE.CHILD_MAX_TOKENS
        ]
        assert not over, (
            f"{len(over)} 个子块超 {RE.CHILD_MAX_TOKENS} token"
            f"（会被 embedding 静默截断，检索到的是半句话）: {over[:5]}"
        )

    def test_parent_within_limit(self, chunks, real_tokenizer):
        over = [
            (c["id"], self._n(real_tokenizer, c["parent_text"]))
            for c in chunks
            if self._n(real_tokenizer, c["parent_text"]) > RE.PARENT_MAX_TOKENS
        ]
        assert not over, (
            f"{len(over)} 个父块超 {RE.PARENT_MAX_TOKENS} token: {over[:5]}"
        )

    def test_parent_not_truncated_by_rerank_max_length(self, chunks, real_tokenizer):
        """RERANK_MAX_LENGTH 不得回退到"把父块砍掉一半"的值。

        出处：RERANK_MAX_LENGTH 注释 —— "384 而非 256：父块 PARENT_MAX_TOKENS=512，
        256 会把父块砍掉一半，等于抵消父子分块的意义"。

        注意这里**不**断言 RERANK_MAX_LENGTH >= 父块长度：384 < 512 是刻意接受的
        折中（受 XLM-R 上限 514 约束），断言 >=512 会把设计选择误判成缺陷。
        真正要守的是"别再退回 256 那种砍一半的取值"。
        """
        assert RE.RERANK_MAX_LENGTH >= 384, (
            f"RERANK_MAX_LENGTH={RE.RERANK_MAX_LENGTH} 已退回会砍掉大半父块的水平"
            f"（父块最大 {RE.PARENT_MAX_TOKENS} token，注释明确否决过 256）"
        )
        assert RE.RERANK_MAX_LENGTH <= 512, (
            f"RERANK_MAX_LENGTH={RE.RERANK_MAX_LENGTH} 超过 XLM-R 上限 514，会被硬截断"
        )

    def test_report_sizes(self, chunks, real_tokenizer, capsys):
        """非断言：把实际尺寸分布打出来，便于人读（-s 时可见）。"""
        ch = [self._n(real_tokenizer, c["child_text"]) for c in chunks]
        pa = [self._n(real_tokenizer, c["parent_text"]) for c in chunks]
        with capsys.disabled():
            print(
                f"\n  子块 token: 中位 {sorted(ch)[len(ch)//2]} 最大 {max(ch)}"
                f" (上限 {RE.CHILD_MAX_TOKENS})"
                f"\n  父块 token: 中位 {sorted(pa)[len(pa)//2]} 最大 {max(pa)}"
                f" (上限 {RE.PARENT_MAX_TOKENS})"
                f"\n  父子相同的块: {sum(1 for a, b in zip(ch, pa) if a == b)}"
                f" / {len(ch)}"
            )
