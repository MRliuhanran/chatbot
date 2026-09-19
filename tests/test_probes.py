"""L0：探针数据的结构与接地校验 —— 纯数据 + 纯函数，不碰模型/Qdrant。

为什么值得单独一层：探针是**评测的唯一来源**，它出错的后果比其他代码更隐蔽 ——
一条关键词写错（比如水浒传 23 回本里根本没有"潘金莲"），该探针就永远不可能命中，
指标被静默稀释成一个无意义的下界，而测试仍然全绿。所以这里把三类不变量前置：

  1. 结构：NamedTuple 字段非空、类别非空、向后兼容（元组解包 + 属性访问）；
  2. 接地：正向/多轮探针的关键词**真的出现在对应书籍里**（读 books/ 校验）；
  3. 反向接地：负样本的主题**真的不在**语料里（absent_terms / never_together）。

读语料的用例用 os.path.exists 守卫、语料缺失时 skip —— 与本项目"分层可跳过"
的约定一致（见 conftest.py 开头）：本机没放语料不等于代码坏了。

运行: pytest -m unit   （或 python -m pytest tests/test_probes.py -v）
"""

import os
from collections import Counter

import pytest

from tests.probes import (
    LONG_QUERY_PROBES,
    MULTITURN_PROBES,
    NEGATIVE_PROBES,
    MultiturnProbe,
    PROBES,
    SHORT_PROBES,
    VALID_MULTITURN_KINDS,
    Probe,
    aggregate_all,
    aggregate_multiturn,
    aggregate_negative,
    aggregate_positive,
    evaluate_multiturn,
    evaluate_negative,
    evaluate_probe,
    rel_gap,
)

pytestmark = pytest.mark.unit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOKS_DIR = os.path.join(ROOT, "books")

# 语料只有这四本；探针的 book 字段必须落在这里面。
FOUR_BOOKS = ("水浒传", "红楼梦", "三国演义", "西游记")

# 历史基线的 12 条（逐字抄写）。存在的意义：向后兼容是硬约束 ——
# 老调用方（eval_runner / compare_ab / verify_qdrant）遍历 PROBES，
# 追加新探针只能**往后加**，前面 12 条的顺序与内容一个字符都不能动，
# 否则历史基线按位置对拍时会全错，且没人看得见。
ORIGINAL_12 = [
    ("武松打虎", "水浒传", ("武松", "景阳冈")),
    ("黛玉葬花", "红楼梦", ("葬花", "黛玉")),
    ("桃园结义", "三国演义", ("桃园", "结义")),
    ("火烧赤壁", "三国演义", ("赤壁",)),
    ("倒拔垂杨柳", "水浒传", ("垂杨", "鲁智深")),
    ("大闹天宫", "西游记", ("天宫", "大圣")),
    ("刘姥姥进大观园", "红楼梦", ("刘姥姥",)),
    ("空城计", "三国演义", ("空城",)),
    ("三打白骨精", "西游记", ("白骨", "悟空")),
    ("鲁智深拳打镇关西", "水浒传", ("镇关西", "鲁智深")),
    ("草船借箭", "三国演义", ("草船", "孔明", "诸葛亮")),
    ("宝玉挨打", "红楼梦", ("贾政", "宝玉")),
]

VALID_NEGATIVE_KINDS = ("out_of_domain", "corpus_gap", "not_in_canon")

# 历史遗留的"未接地"关键词白名单：(探针 query, 关键词)。
#
# 实测：‘草船’在《三国演义》里出现 0 次 —— 全书写的是"用奇谋孔明借箭"
# （"借箭"1 次、"草人"1 次），从没写过"草船"二字。这条探针之所以还能命中，
# 靠的是另两个关键词（孔明/诸葛亮），kw@k 一直虚高。
#
# 为什么不在本次顺手改掉：原 12 条是**冻结**的（见 ORIGINAL_12 的说明），
# 改了就让历史基线按位置对拍时全部错位；而 kw@k 是"任一命中"语义，
# 该关键词写错并不会造成判定错误，只是少了一份证据。故显式登记在此，
# 由下面的用例保证白名单不会腐烂（一旦有人修正探针或换语料，用例会提醒删除）。
LEGACY_UNGROUNDED = {("草船借箭", "草船")}


# ============================================================================
# 语料读取（唯一会碰文件系统的地方，缺语料就 skip）
# ============================================================================
@pytest.fixture(scope="module")
def corpus():
    """{书名: 正文}。

    必须用 errors="replace"：books/水浒传.txt 与 books/红楼梦.txt 各含 2 个
    非法 UTF-8 字节，严格模式会直接抛 UnicodeDecodeError，把"语料有两个坏字节"
    误报成"探针测试挂了"。
    """
    if not os.path.isdir(BOOKS_DIR):
        pytest.skip(f"缺少语料目录 {BOOKS_DIR}")
    out = {}
    for name in FOUR_BOOKS:
        path = os.path.join(BOOKS_DIR, f"{name}.txt")
        if not os.path.exists(path):
            pytest.skip(f"缺少 {path}（本机没放语料，跳过接地校验）")
        with open(path, encoding="utf-8", errors="replace") as f:
            out[name] = f.read()
    return out


# ============================================================================
# 1. 结构 + 向后兼容
# ============================================================================
class TestStructure:
    def test_original_12_are_unchanged_and_first(self):
        """前 12 条必须逐字等于历史基线，且必须排在最前面。"""
        assert [
            (p.query, p.book, p.keywords) for p in PROBES[:12]
        ] == ORIGINAL_12, "原有 12 条探针被改动或顺序被调整 —— 历史基线会静默错位"
        assert SHORT_PROBES == PROBES[:12]

    def test_new_probes_are_appended_not_prepended(self):
        """新探针只能追加：PROBES 长度 = 短查询 + 长问句。"""
        assert len(PROBES) == len(SHORT_PROBES) + len(LONG_QUERY_PROBES)
        assert LONG_QUERY_PROBES, "没有新增长问句探针"

    def test_probe_is_namedtuple_with_both_access_styles(self):
        """项目注释明确要求：属性访问与元组解包都必须成立。"""
        p = PROBES[0]
        assert isinstance(p, Probe)
        assert p._fields == ("query", "book", "keywords")
        for query, book, kws in PROBES:  # 旧调用方的写法
            assert query == query and book and kws

    def test_all_probes_wellformed(self):
        for p in PROBES:
            assert p.query and p.query == p.query.strip(), f"查询为空或有首尾空白: {p!r}"
            assert p.book in FOUR_BOOKS, f"{p.query!r} 的 book 非法: {p.book!r}"
            assert isinstance(p.keywords, tuple) and p.keywords, p.query
            for k in p.keywords:
                assert isinstance(k, str) and k.strip(), f"{p.query!r} 关键词非法: {k!r}"

    def test_queries_are_unique(self):
        """重名会让"以 query 为键"的评测结果互相覆盖（eval_runner 正是这么建字典的）。"""
        queries = [p.query for p in PROBES]
        assert len(queries) == len(set(queries)), "PROBES 里有重复 query"

    def test_every_book_is_covered_by_long_queries(self):
        """四本书都要有长问句，否则"长问句的缺陷"只在部分书上被验证。"""
        assert {p.book for p in LONG_QUERY_PROBES} == set(FOUR_BOOKS)


# ============================================================================
# 2. 正向/多轮探针的接地：关键词必须真的在语料里
# ============================================================================
class TestGrounding:
    def test_positive_keywords_exist_in_their_book(self, corpus):
        """每个关键词都必须能在对应书里找到 —— 否则该探针永远不可能命中。

        断言的是"全部关键词都存在"而不是"任一存在"：任一存在时，一个
        写错的关键词会被另一个正确关键词掩盖，kw@k 看着正常，实际少了一半证据
        （'草船借箭'正是这种情形，见 LEGACY_UNGROUNDED）。
        原 12 条只能整体冻结，所以那一个已知例外走白名单；新探针一律不豁免。
        """
        missing, allowlisted = [], set()
        for p in PROBES:
            for k in p.keywords:
                if k in corpus[p.book]:
                    continue
                if (p.query, k) in LEGACY_UNGROUNDED:
                    allowlisted.add((p.query, k))
                    continue
                missing.append(f"{p.query!r}: 《{p.book}》中无 {k!r}")
        assert not missing, "以下关键词在语料中不存在，探针无法命中: " + "; ".join(missing)
        # 白名单防腐：多登记了（已修好却没删）说明名单在说谎，同样是失败。
        assert allowlisted == LEGACY_UNGROUNDED, (
            f"白名单过期：实际未接地的只有 {sorted(allowlisted)}，"
            f"请同步修正 LEGACY_UNGROUNDED"
        )

    def test_new_probes_have_no_ungrounded_keyword(self, corpus):
        """新探针（长问句）不接受任何例外：关键词必须逐条接地。"""
        bad = [
            f"{p.query!r}: {k!r}"
            for p in LONG_QUERY_PROBES
            for k in p.keywords
            if k not in corpus[p.book]
        ]
        assert not bad, "新增长问句探针存在未接地关键词: " + "; ".join(bad)

    def test_multiturn_keywords_exist_in_their_book(self, corpus):
        missing = []
        for p in MULTITURN_PROBES:
            for k in p.keywords:
                if k not in corpus[p.book]:
                    missing.append(f"{p.query!r}: 《{p.book}》中无 {k!r}")
        assert not missing, "多轮探针关键词未接地: " + "; ".join(missing)

    def test_multiturn_topic_terms_exist_in_their_book(self, corpus):
        for p in MULTITURN_PROBES:
            for t in p.topic_terms:
                assert t in corpus[p.book], f"{p.query!r} 的 topic 词 {t!r} 不在《{p.book}》"


# ============================================================================
# 3. 多轮探针：必须真的是"指代"
# ============================================================================
class TestMultiturnShape:
    def test_history_is_openai_style_and_nonempty(self):
        for p in MULTITURN_PROBES:
            assert p.history, f"{p.query!r} 没有历史"
            for msg in p.history:
                assert set(msg) == {"role", "content"}, p.query
                assert msg["role"] in ("user", "assistant"), p.query
                assert msg["content"] and msg["content"].strip(), p.query

    def test_current_query_carries_no_entity(self):
        """当前问句里不能出现实体名，否则它就不是指代性问句了。

        这是本类探针的**定义**：只有"查询本身不含实体"时，检索器才被迫
        使用历史。若哪天有人把 query 改成"孙悟空最后结局如何"，这条用例会
        立刻失败 —— 那种改动等于把多轮退化回单轮，指标会虚高。
        """
        for p in MULTITURN_PROBES:
            for t in p.topic_terms:
                assert t not in p.query, f"{p.query!r} 含实体 {t!r}，不是指代性问句"

    def test_resolved_query_adds_the_entity(self):
        """resolved 必须把指代消解开 —— 它是诊断用的"标准改写"。"""
        for p in MULTITURN_PROBES:
            assert any(t in p.resolved for t in p.topic_terms), (
                f"{p.query!r} 的 resolved={p.resolved!r} 没有补回实体，"
                f"无法用它区分「改写错了」与「检索错了」"
            )

    def test_history_mentions_the_entity(self):
        """指代对象必须真的在历史里出现过，否则无从消解。

        允许出现在任意一轮（含 assistant 轮）：只用用户消息做改写的实现
        会在这条上暴露 —— 那正是我们要度量的缺陷之一。
        """
        for p in MULTITURN_PROBES:
            joined = " ".join(m["content"] for m in p.history)
            assert any(t in joined for t in p.topic_terms), (
                f"{p.query!r} 的历史里找不到指代对象 {p.topic_terms}"
            )

    def test_at_least_one_probe_needs_assistant_turn(self):
        """至少有一条探针的实体**只**出现在 assistant 轮里 —— 否则本集合测不出
        "只看用户消息做改写"这个常见实现缺陷。"""
        only_assistant = 0
        for p in MULTITURN_PROBES:
            user_text = " ".join(m["content"] for m in p.history if m["role"] == "user")
            if not any(t in user_text for t in p.topic_terms):
                only_assistant += 1
        assert only_assistant >= 1, "所有多轮探针的实体都在用户轮出现，测不到 assistant 轮"


class TestMultiturnKinds:
    """指代**类型**的覆盖度。

    只报一个多轮总分是不够的：代词没消解、实体切错、省略没补全、只认人名的
    实现缺陷，修法各不相同，而它们的现象都是"多轮指标掉了几个点"。分类之后
    才看得出是哪一类的锅（见 probes.aggregate_multiturn_by_kind）。
    """

    #: 每一类至少要有几条。2 条只能看出"是不是系统性失手"，而实测证明
    #: 2–3 条的规模下，一条探针进出就能翻转整组的比率（`topic@k 95.2%→100%`
    #: 在 n=21 上只差一条）。2026-09-19 把每类补齐到 5 条后上调到 5：
    #: 这个下限本身就是护栏 —— 以后新增 kind 时必须一次带够样本，
    #: 而不是先加一条占位。见 MULTITURN_PLAN.md §6。
    MIN_PER_KIND = 5

    def test_every_probe_is_classified(self):
        """新增探针必须显式声明 kind：默认空串会让它静默地不进任何分组。"""
        for p in MULTITURN_PROBES:
            assert p.kind in VALID_MULTITURN_KINDS, (
                f"{p.query!r} 的 kind={p.kind!r} 非法或未声明；"
                f"合法取值: {VALID_MULTITURN_KINDS}"
            )

    def test_every_kind_has_enough_samples(self):
        counts = Counter(p.kind for p in MULTITURN_PROBES)
        missing = [k for k in VALID_MULTITURN_KINDS
                   if counts.get(k, 0) < self.MIN_PER_KIND]
        assert not missing, (
            f"以下指代类型样本不足（需 >= {self.MIN_PER_KIND} 条）: "
            f"{ {k: counts.get(k, 0) for k in missing} }"
        )

    def test_query_is_unique(self):
        """按 query 建索引（per_probe 就是 dict）—— 重名会让一条静默覆盖另一条。"""
        dupes = [q for q, c in Counter(p.query for p in MULTITURN_PROBES).items() if c > 1]
        assert not dupes, f"多轮探针 query 重复（会互相覆盖）: {dupes}"

    def test_probe_count_is_enough_to_conclude(self):
        """n=5 时"改一项指标动 20 个点"，任何 A/B 结论都被单条探针左右 ——
        这正是 RAG_RERANK_ON 那次改不动默认值的原因。多轮这一组至少要够看分层。
        """
        assert len(MULTITURN_PROBES) >= 20, (
            f"多轮探针只有 {len(MULTITURN_PROBES)} 条，不足以支撑 A/B 结论"
        )

    def test_history_carries_at_least_two_turns_for_switch(self):
        """entity_switch 必须真有"两个实体先后出现"，否则它只是一条普通代词探针。"""
        for p in MULTITURN_PROBES:
            if p.kind != "entity_switch":
                continue
            user_turns = [m["content"] for m in p.history if m["role"] == "user"]
            assert len(user_turns) >= 2, f"{p.query!r} 需要至少两轮用户消息才能考切换"


# ============================================================================
# 4. 负样本：主题必须真的不在语料里
# ============================================================================
class TestTopicMetricIsAliasAware:
    """topic@k 必须认别名 —— 否则判据会**假失败**。

    实测踩过：新探针"后来这件事是怎么了结的"最初只写 topic_terms=("关羽",)，
    而《三国演义》里"关羽"9 次、"云长"443 次、"关公"519 次 —— 三种配置下
    全部 topic@k=N，看起来像"检索系统性失手"，实际是度量假象。

    归一化函数由调用方注入（probes.py 刻意零依赖），这里用一个等价于
    `rag.normalize_aliases` 语义的小桩来钉住行为。
    """

    #: 只认"云长/关公 → 关羽"这一条，够验证注入路径与比对逻辑
    STUB = staticmethod(lambda t: t.replace("云长", "关羽").replace("关公", "关羽"))

    def _probe(self):
        return MULTITURN_PROBES[0]

    def _results(self, text):
        return [{"book": "三国演义", "child_text": text, "parent_text": ""}]

    def test_alias_only_body_counts_as_a_hit(self):
        p = MultiturnProbe(
            history=[{"role": "user", "content": "关羽是谁"}],
            query="他后来怎么了", book="三国演义", keywords=("云长",),
            resolved="关羽后来怎么了", topic_terms=("关羽",), kind="pronoun")
        # 正文只写了别名"云长"，字面匹配会判失败
        assert evaluate_multiturn(p, self._results("云长提刀上马"))["topic@k"] is False
        assert evaluate_multiturn(p, self._results("云长提刀上马"),
                                  normalize=self.STUB)["topic@k"] is True

    def test_normalization_does_not_change_unrelated_text(self):
        p = MultiturnProbe(
            history=[{"role": "user", "content": "关羽是谁"}],
            query="他后来怎么了", book="三国演义", keywords=("张飞",),
            resolved="关羽后来怎么了", topic_terms=("关羽",), kind="pronoun")
        assert evaluate_multiturn(p, self._results("张飞大喝一声"),
                                  normalize=self.STUB)["topic@k"] is False

    def test_no_normalize_keeps_legacy_literal_behaviour(self):
        """不注入时不改变旧口径（旧基线与历史数字才可比）。"""
        p = self._probe()
        results = self._results("云长提刀上马")
        assert (evaluate_multiturn(p, results)["topic@k"]
                == any(t in "云长提刀上马" for t in p.topic_terms))


class TestNegativeGrounding:
    def test_negative_probes_wellformed(self):
        assert NEGATIVE_PROBES, "没有负样本，拒答能力无从度量"
        for p in NEGATIVE_PROBES:
            assert p.query and p.query.strip(), p
            assert p.reason and p.reason.strip(), f"{p.query!r} 缺少拒绝理由"
            assert p.kind in VALID_NEGATIVE_KINDS, f"{p.query!r} 的 kind 非法: {p.kind}"
            assert p.absent_terms or p.never_together, (
                f"{p.query!r} 没有任何接地证据（absent_terms / never_together）—— "
                f"无法证明它真的应该被拒答"
            )
            if p.kind == "corpus_gap":
                assert p.gap_book in FOUR_BOOKS, f"{p.query!r} 未指明缺内容属于哪本书"

    def test_all_three_kinds_present(self):
        """三类缺陷的成因不同，必须都有样本，否则某一类修好了也看不见。"""
        assert {p.kind for p in NEGATIVE_PROBES} == set(VALID_NEGATIVE_KINDS)

    def test_absent_terms_really_absent(self, corpus):
        """absent_terms 必须 0 次出现。

        corpus_gap 类只在 gap_book 那一本里查：该词在别的书里存在恰恰是
        危险来源（"招安"在三国/西游里有 22/11 次，检索器会拿别书的同名词
        来充数），此时要求它全语料缺席就跑偏了。
        """
        bad = []
        for p in NEGATIVE_PROBES:
            for term in p.absent_terms:
                if p.kind == "corpus_gap":
                    scope = {p.gap_book: corpus[p.gap_book]}
                else:
                    scope = corpus
                for book, text in scope.items():
                    if term in text:
                        bad.append(f"{p.query!r}: {term!r} 在《{book}》中出现 {text.count(term)} 次")
        assert not bad, "负样本的关键词其实存在于语料中，拒答理由不成立: " + "; ".join(bad)

    def test_never_together_pairs_are_disjoint_books(self, corpus):
        """never_together 的两个人/物必须各自存在于语料，但从不共处一本书。

        两个条件都要：若某个词整个语料都没有，"跨书"就成了废话（真正的原因
        是"词根本不存在"，属于 absent_terms 的范畴，标注错了）。
        """
        for p in NEGATIVE_PROBES:
            if not p.never_together:
                continue
            a, b = p.never_together
            books_a = {bk for bk, t in corpus.items() if a in t}
            books_b = {bk for bk, t in corpus.items() if b in t}
            assert books_a, f"{p.query!r}: {a!r} 在语料里一次都没出现，标注方式不对"
            assert books_b, f"{p.query!r}: {b!r} 在语料里一次都没出现，标注方式不对"
            assert not (books_a & books_b), (
                f"{p.query!r}: {a!r} 与 {b!r} 同时出现在 {sorted(books_a & books_b)}，"
                f"它们并非跨书关系"
            )

    def test_negative_queries_do_not_collide_with_positive(self):
        """负样本查询不能与正向探针重复，否则同一 query 的判定会自相矛盾。"""
        assert not ({p.query for p in NEGATIVE_PROBES} & {p.query for p in PROBES})
        assert not ({p.query for p in MULTITURN_PROBES} & {p.query for p in PROBES})


# ============================================================================
# 5. 拒答判定（纯函数）
# ============================================================================
def _res(*scores):
    """构造最小可用的 results：只需要 rerank_score。"""
    return [{"book": "西游记", "child_text": "x", "parent_text": "y",
             "rerank_score": s} for s in scores]


class TestRelGap:
    def test_absolute_scale_is_irrelevant(self):
        """**核心不变量**：整体平移分数不改变判定结果。

        bge-reranker-base 输出未过 sigmoid 的 logit，实测"三打白骨精"首位 -0.11
        是正确答案、"刘姥姥进大观园"首位 5.34 也是正确答案 —— 跨查询差了 5.4 分
        且跨零。所以判定只能依赖同一查询内部的形态：把一组分数整体 +100，
        结论必须一模一样。任何引入绝对阈值的实现都会在这条上失败。
        """
        base = [0.1, -0.4, -0.9, -1.2, -1.5]
        shifted = [s + 100.0 for s in base]
        assert rel_gap(_res(*base)) == pytest.approx(rel_gap(_res(*shifted)))
        for s in base:
            assert evaluate_negative(NEGATIVE_PROBES[0], _res(*base))["refuse"] == \
                evaluate_negative(NEGATIVE_PROBES[0], _res(*(x + 100.0 for x in base)))["refuse"]

    def test_flat_scores_mean_no_evidence(self):
        """全体分数重合 = 没有任何候选突出 = 无依据，应当拒答。"""
        assert rel_gap(_res(1.0, 1.0, 1.0, 1.0)) == 0.0
        assert evaluate_negative(NEGATIVE_PROBES[0], _res(1.0, 1.0, 1.0))["refuse"] is True

    def test_dominant_top1_is_not_refused(self):
        assert rel_gap(_res(8.0, 0.0, -1.0, -2.0)) > 0.5
        assert evaluate_negative(NEGATIVE_PROBES[0], _res(8.0, 0.0, -1.0, -2.0))["refuse"] is False

    def test_range_and_monotonicity(self):
        """rel_gap 必须落在 [0,1] 且随首位优势单调不减。"""
        assert rel_gap(_res(1.0, 0.0)) == pytest.approx(1.0)
        assert rel_gap(_res(1.0, 0.9, 0.8, 0.5)) < rel_gap(_res(1.0, 0.4, 0.3, 0.2))
        for scores in ([1.0, 0.0], [0.5, 0.4, 0.3], [-0.1, -0.5, -3.0], [5.0, 4.9, 4.8]):
            assert 0.0 <= rel_gap(_res(*scores)) <= 1.0

    def test_degenerate_inputs(self):
        """结果为空/单条/分数缺失都不能抛异常 —— 评测跑在线上链路上，崩了会中断整轮。"""
        assert rel_gap([]) == 0.0
        assert rel_gap(_res(3.0)) == 0.0
        assert rel_gap(_res(None, None)) == 0.0
        assert rel_gap([{"rerank_score": 2.0}, {"rerank_score": None}]) == 0.0

    def test_empty_results_are_refused_but_flagged(self):
        v = evaluate_negative(NEGATIVE_PROBES[0], [])
        assert v["refuse"] is True and v["empty"] is True and v["n"] == 0

    def test_evaluate_negative_reports_head_for_diagnostics(self):
        v = evaluate_negative(NEGATIVE_PROBES[0], _res(2.0, -1.0, -3.0))
        assert v["head"] == 2.0 and v["n"] == 3 and v["empty"] is False

    def test_negative_reports_book_scatter(self):
        """书目分散度是唯一有实测区分度的相对信号（正向 1.08 本 vs 负样本 2.14 本），
        作为诊断字段暴露，不能丢。"""
        results = [
            {"book": "水浒传", "child_text": "", "parent_text": "", "rerank_score": 2.0},
            {"book": "西游记", "child_text": "", "parent_text": "", "rerank_score": 1.0},
            {"book": "水浒传", "child_text": "", "parent_text": "", "rerank_score": 0.0},
        ]
        assert evaluate_negative(NEGATIVE_PROBES[0], results)["n_books"] == 2
        assert evaluate_negative(NEGATIVE_PROBES[0], [])["n_books"] == 0

    def test_book_scatter_alone_does_not_flip_refusal(self):
        """书目分散度**不得**并入 refuse。

        实测它虽有区分度（负样本 6/7 跨书），却是"领域归属"信号而非"相关度"
        信号：跨书提问恰好全落在一本书里时会漏判。若让它决定 refuse，
        引擎明明没有拒答通路，refuse_rate 却会显示得很高 —— 那是假阳性，
        比指标不灵更危险。
        """
        scattered = [
            {"book": b, "child_text": "", "parent_text": "", "rerank_score": s}
            for b, s in zip(("水浒传", "西游记", "红楼梦"), (8.0, 0.0, -1.0))
        ]
        v = evaluate_negative(NEGATIVE_PROBES[0], scattered)
        assert v["n_books"] == 3
        assert v["refuse"] is False, "书分散但首位极度突出时不应判为拒答"


class TestEvaluateMultiturnPure:
    def test_returns_single_turn_keys_plus_topic(self):
        """多轮指标必须与单轮同口径（三个同名键），否则两者无法横向比较。"""
        p = MULTITURN_PROBES[0]
        results = [{"book": p.book, "child_text": "悟空成佛", "parent_text": "", "rerank_score": 1.0}]
        v = evaluate_multiturn(p, results)
        assert set(v) == {"book@1", "book@k", "kw@k", "topic@k"}
        assert v["book@1"] and v["kw@k"] and v["topic@k"]

    def test_topic_misses_when_book_is_right_but_content_is_not(self):
        """这正是多轮缺陷的典型形态：书对了，内容与指代对象无关。

        book@k=True 会把这种情况判成通过，topic@k 不会 —— 没有这个指标就
        发现不了"召回到西游记里一段跟孙悟空无关的经文"。
        """
        p = MULTITURN_PROBES[0]
        results = [{"book": "西游记", "child_text": "如来讲经", "parent_text": "",
                    "rerank_score": 3.0}]
        v = evaluate_multiturn(p, results)
        assert v["book@k"] is True and v["topic@k"] is False


# ============================================================================
# 6. 聚合
# ============================================================================
class TestAggregate:
    def test_positive_rates(self):
        per = {
            "a": {"book@1": True, "book@k": True, "kw@k": True},
            "b": {"book@1": False, "book@k": True, "kw@k": False},
        }
        agg = aggregate_positive(per)
        assert agg["n"] == 2
        assert agg["book@1"] == 0.5 and agg["book@k"] == 1.0 and agg["kw@k"] == 0.5

    def test_multiturn_has_topic_metric(self):
        per = {"a": {"book@1": True, "book@k": True, "kw@k": False, "topic@k": True}}
        assert aggregate_multiturn(per)["topic@k"] == 1.0

    def test_negative_rates(self):
        per = {
            "n1": {"refuse": True, "rel_gap": 0.1, "empty": False, "n_books": 1},
            "n2": {"refuse": False, "rel_gap": 0.9, "empty": False, "n_books": 3},
            "n3": {"refuse": True, "rel_gap": 0.0, "empty": True, "n_books": 0},
            "n4": {"refuse": False, "rel_gap": 1.0, "empty": False, "n_books": 2},
        }
        agg = aggregate_negative(per)
        assert agg["n"] == 4
        assert agg["refuse_rate"] == 0.5 and agg["answer_rate"] == 0.5
        assert agg["empty_rate"] == 0.25
        assert agg["mean_rel_gap"] == pytest.approx(0.5)
        assert agg["mean_n_books"] == pytest.approx(1.5)

    def test_aggregate_all_only_includes_given_categories(self):
        """没传的类别不能出现 —— 给 0 会被误读成"该类全错"。"""
        agg = aggregate_all(positive={"a": {"book@1": True, "book@k": True, "kw@k": True}})
        assert set(agg) == {"positive", "n_total"}
        assert agg["n_total"] == 1

        full = aggregate_all(
            positive={"a": {"book@1": True, "book@k": True, "kw@k": True}},
            negative={"n": {"refuse": True, "rel_gap": 0.1, "empty": False}},
            multiturn={"m": {"book@1": True, "book@k": True, "kw@k": True, "topic@k": True}},
        )
        assert set(full) == {"positive", "negative", "multiturn", "n_total"}
        assert full["n_total"] == 3

    def test_aggregate_tolerates_empty_input(self):
        """空字典不能除零 —— 语料/索引缺失时调用方会传空集合进来。"""
        for agg in (aggregate_positive({}), aggregate_multiturn({}), aggregate_negative({})):
            assert agg["n"] == 0
            assert all(
                v == 0.0 for k, v in agg.items() if k != "n"
            ), f"空输入的比率应为 0 而不是 NaN/异常: {agg}"
        assert aggregate_all() == {"n_total": 0}

    def test_evaluate_probe_contract_unchanged(self):
        """向后兼容的硬约束：键名与判定语义都不许变。"""
        results = [{"book": "水浒传", "child_text": "武松", "parent_text": "", "rerank_score": 1.0}]
        v = evaluate_probe(PROBES[0], results)
        assert set(v) == {"book@1", "book@k", "kw@k"}
        assert v == {"book@1": True, "book@k": True, "kw@k": True}
        assert evaluate_probe(PROBES[0], []) == {"book@1": False, "book@k": False, "kw@k": False}
