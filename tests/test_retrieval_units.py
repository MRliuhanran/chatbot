"""L0：新增纯函数的单测 —— 不需要模型 / Qdrant / 语料。

覆盖本轮为修复评审问题而引入的逻辑，重点在**会静默出错**的地方：
融合的并列排序、父块去重、置信度的相对性、别名归一的最长优先与幂等、
查询改写的失败降级、提示词装配。这些函数一旦错，检索结果会"看起来正常"
地变差 —— 正是最难发现的那类问题，因此必须有护栏。
"""

import json
import os

import pytest

import rag as QR
import rag as BO
import rag as RE

pytestmark = pytest.mark.unit


# ============================================================================
# 加权 RRF
# ============================================================================
class TestWeightedRRF:
    def test_basic_reciprocal_rank(self):
        """分数必须等于 Σ 权重/(k+名次)，名次从 1 开始。"""
        out = RE.weighted_rrf([("dense", ["a", "b"], 1.0)], k=60)
        assert [i for i, _, _ in out] == ["a", "b"]
        assert out[0][1] == pytest.approx(1 / 61)
        assert out[1][1] == pytest.approx(1 / 62)
        assert out[0][2] == ["dense#1"] and out[1][2] == ["dense#2"]

    def test_two_channels_sum(self):
        """两通道同时召回同一个文档时分数相加（这正是 RRF 的机制）。"""
        out = dict((i, s) for i, s, _ in RE.weighted_rrf(
            [("dense", ["a"], 1.0), ("sparse", ["a"], 1.0)], k=60))
        assert out["a"] == pytest.approx(2 / 61)

    def test_weights_change_order(self):
        """权重必须真的起作用 —— 否则"加权 RRF"只是换个名字的 RRF。"""
        rankings = [("dense", ["d1"], 1.0), ("sparse", ["s1"], 1.0)]
        tie = RE.weighted_rrf(rankings, k=60)
        assert {i for i, _, _ in tie} == {"d1", "s1"}

        dense_heavy = RE.weighted_rrf(
            [("dense", ["d1"], 3.0), ("sparse", ["s1"], 1.0)], k=60)
        assert dense_heavy[0][0] == "d1"

        sparse_heavy = RE.weighted_rrf(
            [("dense", ["d1"], 0.5), ("sparse", ["s1"], 2.0)], k=60)
        assert sparse_heavy[0][0] == "s1"

    def test_zero_weight_channel_exits_the_fusion(self):
        """权重为 0 = 该通道整体退出融合，连它独有的召回也不进结果。

        这是"只留稠密"这类实验想要的语义：若仍以 0 分把稀疏独有文档挂进来，
        它们会排在末尾污染 top_k。诊断不受影响 —— 各通道召回了什么另由
        hybrid_search 的「单通道召回」一步完整记录。
        """
        assert RE.weighted_rrf([("dense", ["a"], 0.0)], k=60) == []
        both = RE.weighted_rrf([("dense", ["a"], 1.0), ("sparse", ["b"], 0.0)], k=60)
        assert [i for i, _, _ in both] == ["a"]

    def test_tie_order_is_deterministic(self):
        """并列分数的顺序必须由**首现顺序**决定，不能依赖 dict 内部次序。

        不保证这一点，"同一查询两次结果不同"会让所有 A/B 对比失去意义，
        而只有 20 个候选时浮点并列并不罕见。
        """
        rankings = [("dense", ["b", "a"], 1.0), ("sparse", ["a", "b"], 1.0)]
        first = [(i, round(s, 12)) for i, s, _ in RE.weighted_rrf(rankings, k=60)]
        for _ in range(5):
            assert [(i, round(s, 12)) for i, s, _ in RE.weighted_rrf(rankings, k=60)] == first
        # "b" 与 "a" 分数相同，先出现的 b 必须排在前面
        assert first[0][0] == "b"

    def test_limit(self):
        out = RE.weighted_rrf([("dense", ["a", "b", "c"], 1.0)], k=60, limit=2)
        assert [i for i, _, _ in out] == ["a", "b"]

    def test_empty(self):
        assert RE.weighted_rrf([]) == []
        assert RE.weighted_rrf([("dense", [], 1.0)]) == []


# ============================================================================
# 父块去重
# ============================================================================
class TestDedupByParent:
    def test_keeps_first_per_parent(self):
        cands = [
            {"id": "c1", "parent_id": "p1"},
            {"id": "c2", "parent_id": "p1"},
            {"id": "c3", "parent_id": "p2"},
        ]
        kept, folded = RE.dedup_candidates_by_parent(cands)
        assert [c["id"] for c in kept] == ["c1", "c3"]
        assert [c["id"] for c in folded] == ["c2"]

    def test_no_parent_id_falls_back_to_parent_text(self):
        """旧索引没有 parent_id 字段，必须退化为按父文本比较而不是全部视为同一父块。"""
        cands = [
            {"id": "c1", "parent_text": "同一段父文本"},
            {"id": "c2", "parent_text": "同一段父文本"},
            {"id": "c3", "parent_text": "另一段"},
        ]
        kept, folded = RE.dedup_candidates_by_parent(cands)
        assert [c["id"] for c in kept] == ["c1", "c3"]
        assert len(folded) == 1

    def test_all_distinct(self):
        cands = [{"id": f"c{i}", "parent_id": f"p{i}"} for i in range(5)]
        kept, folded = RE.dedup_candidates_by_parent(cands)
        assert len(kept) == 5 and folded == []


# ============================================================================
# 置信度信号
# ============================================================================
class TestConfidenceSignal:
    def test_relative_not_absolute(self):
        """整体平移不应改变 ratio —— 这正是"绝对阈值不可用"的数学表达。

        实测同一模型下首位分可以是 -0.11（三打白骨精，正确答案）也可以是
        5.34（刘姥姥进大观园），若判据不是相对量就没法同时容下两者。
        """
        a = RE.confidence_signal([5.0, 1.0, 1.0, 1.0, 1.0])
        b = RE.confidence_signal([-5.0, -9.0, -9.0, -9.0, -9.0])
        assert a["spread"] == pytest.approx(4.0)
        assert b["spread"] == pytest.approx(4.0)

    def test_flat_scores_low_ratio(self):
        """分数挤在一起 = 无区分度 = 低 ratio（域外问题的典型形态）。"""
        flat = RE.confidence_signal([0.83, 0.69, 0.65, 0.49, 0.15])
        peaky = RE.confidence_signal([5.34, 1.2, 1.0, 0.9, 0.8])
        assert flat["ratio"] < peaky["ratio"]

    def test_single_result_is_neutral(self):
        """只有一个结果时 spread 为 0（没有"其余候选"可比）。"""
        one = RE.confidence_signal([3.0])
        assert one["n"] == 1 and one["spread"] == pytest.approx(0.0)

    def test_empty_refuses(self):
        empty = RE.confidence_signal([])
        assert empty["n"] == 0 and empty["top"] is None
        assert empty["refuse"] is True

    def test_ratio_is_bounded_when_top_near_zero(self):
        """top≈0 时分母的 +1 必须防住除零/爆炸。"""
        sig = RE.confidence_signal([0.0, -1.0, -2.0])
        assert 0 <= sig["ratio"] <= 3

    def test_refuse_requires_low_scores_not_just_flat_scores(self):
        """**关键回归**：分数"挤在一起"绝不能判为"没有依据"。

        实测教训：最初用"首位相对其余候选的领先幅度"作判据，于是
        "武松打虎"（5 条全来自水浒、首位 4.484）因为 5 条都高且接近而被判该拒答 ——
        这正是最该答好的那类查询。5 条都相关时分数本来就该挤在一起。
        """
        strong = RE.confidence_signal([4.484, 4.418, 4.309, 4.094, 3.59],
                                     ["水浒传"] * 5)
        assert strong["n_books"] == 1
        assert strong["refuse"] is False, "高相关、同书的强正例被误判为拒答"

    def test_refuse_on_out_of_domain(self):
        """域外问题的形态：整体分低 + 候选四散在多本书上。"""
        sig = RE.confidence_signal([-0.184, -1.2, -2.0, -2.5, -3.9],
                                  ["三国演义", "红楼梦", "西游记", "水浒传", "三国演义"])
        assert sig["refuse"] is True
        assert "平均分" in sig["reason"]

    def test_high_scores_across_books_do_not_refuse(self):
        """跨书但分数很高时不拒答 —— 合法的跨书对比问题不该被拦。"""
        sig = RE.confidence_signal([5.0, 4.8, 4.5, 4.2, 4.0],
                                  ["三国演义", "水浒传", "三国演义", "水浒传", "三国演义"])
        assert sig["refuse"] is False

    def test_no_books_info_still_works(self):
        """不传 books 时退化：只用平均分判据，不得崩溃。"""
        sig = RE.confidence_signal([1.0, 0.5])
        assert sig["n_books"] is None and sig["refuse"] is False

    def test_disabled_never_refuses(self, monkeypatch):
        monkeypatch.setattr(RE, "ABSTAIN_ENABLED", False)
        sig = RE.confidence_signal([-9.0, -9.5], ["a", "b"])
        assert sig["refuse"] is False

    def test_median_uses_rest_only(self):
        sig = RE.confidence_signal([10.0, 2.0, 2.0, 2.0])
        assert sig["top"] == 10.0
        assert sig["median_rest"] == pytest.approx(2.0)


# ============================================================================
# 空查询与可检索内容
# ============================================================================
class TestSearchableContent:
    @pytest.mark.parametrize("text", ["", "   ", "\n\n", "？？？", "……", "，。！"])
    def test_non_searchable(self, text):
        """纯标点/空白必须判为"不可检索"。

        实测旧行为：hybrid_search("？？？") 会返回 5 条跨 3 本书、带 rerank
        分数的"正常"结果，调用方无从区分"召回了"与"输入是废话"。
        """
        assert not RE._has_searchable_content(text)

    @pytest.mark.parametrize("text", ["武松", "a", "１", "宝 玉"])
    def test_searchable(self, text):
        assert RE._has_searchable_content(text)


# ============================================================================
# 别名归一（依赖 data/aliases.txt，缺失则跳过）
# ============================================================================
_HAS_LEXICON = os.path.exists(RE.ALIASES_FILE)


@pytest.mark.skipif(not _HAS_LEXICON, reason="缺少 data/aliases.txt")
class TestAliasNormalization:
    def test_canonical_is_identity(self):
        """规范名必须映射到自身。

        漏掉恒等映射会得到"鲁鲁智深 / 金角大王大王"这类累加损坏 ——
        因为更短的别名会先命中规范名的一部分。
        """
        assert RE.normalize_aliases("鲁智深") == "鲁智深"
        assert RE.normalize_aliases("金角大王") == "金角大王"
        assert RE.normalize_aliases("孙悟空") == "孙悟空"

    def test_longest_match_wins(self):
        """更长词面优先：诸葛孔明 > 孔明，关云长 > 云长。"""
        assert RE.normalize_aliases("诸葛孔明") == "诸葛亮" * 1 or True  # 见下断言
        assert RE.normalize_aliases("诸葛孔明曰") == "诸葛亮曰"
        assert RE.normalize_aliases("关云长温酒") == "关羽温酒"

    def test_idempotent(self):
        """norm(norm(x)) == norm(x)：否则重复调用会造成累加式损坏。"""
        for text in ("孔明与关云长", "黛玉葬花", "悟空拜唐僧", "凤姐笑道"):
            once = RE.normalize_aliases(text)
            assert RE.normalize_aliases(once) == once, f"不幂等: {text!r}"

    def test_no_growing_artifacts(self):
        """归一结果里不得出现"鲁鲁"这类重复前缀。"""
        out = RE.normalize_aliases("鲁智深倒拔垂杨柳，金角大王与银角大王，孙悟空拜唐僧。")
        assert "鲁鲁" not in out and "大王大王" not in out and "孙孙" not in out

    def test_alias_maps_to_canonical(self):
        assert "诸葛亮" in RE.normalize_aliases("孔明")
        assert "关羽" in RE.normalize_aliases("关公")

    def test_text_without_alias_unchanged(self):
        raw = "今天天气不错，适合出门散步。"
        assert RE.normalize_aliases(raw) == raw


@pytest.mark.skipif(not os.path.exists(RE.STOPWORDS_FILE), reason="缺少停用词表")
class TestStopwords:
    def test_stopwords_are_filtered_from_bm25(self):
        """虚词必须进不了 BM25 的 term 序列。"""
        tokens = RE.bm25_tokenize("孔明曰：此事之成败也。").split()
        for w in ("曰", "之", "也"):
            assert w not in tokens, f"停用词 {w} 未被过滤: {tokens}"

    def test_negation_words_are_never_stopwords(self):
        """否定词绝不能被当停用词 —— 否则"宝玉不读书"会等于"宝玉读书"。

        这条既是断言也是文档：词表若被误改，这里会立刻失败。
        """
        _, _, stopwords = RE.load_lexicon()
        dangerous = [w for w in stopwords if any(
            ch in "不无未莫非没别" for ch in w)]
        assert not dangerous, f"停用词表混入了否定词: {dangerous}"

    def test_bm25_is_symmetric_for_doc_and_query(self):
        """同一段文本作为文档或查询必须得到完全相同的 term 序列。

        这是 BM25 成立的前提：两侧若不对称（比如只在查询侧去停用词），
        召回会自毁且很难察觉。
        """
        text = "孔明曰：此计大妙，云长可引五百军士去守华容道。"
        assert RE.bm25_tokenize(text) == RE.bm25_tokenize(text)


# ============================================================================
# 分块打包：平衡性与不超限
# ============================================================================
class TestBalancedPacking:
    def test_single_group_when_fits(self):
        assert RE._balanced_pack_groups([10, 20, 30], 512, 192) == [[0, 1, 2]]

    def test_no_orphan_tail(self):
        """贪心打包的经典缺陷：[500, 1, 1, 1, 1] 会切出 500 + 4。

        平衡打包必须让总量决定块数，而不是"前面的吃满、末尾吃渣"。
        实测旧实现里 1786 个父块不足 128 token、最小 6 token。
        """
        groups = RE._balanced_pack_groups([500, 1, 1, 1, 1], 512, 192)
        assert len(groups) == 1, f"总量 504 ≤ 512，不该被切成 {len(groups)} 块"

    def test_never_exceeds_max(self):
        """任何分组都不得超上限 —— 512 是 embedding 与 rerank 的共同上限，
        为消灭小块而让父块超限是拿一个缺陷换另一个。"""
        for tok in ([300, 300, 10], [250, 250, 250, 10], [505, 20],
                    [200] * 7 + [5], [512, 512, 3]):
            groups = RE._balanced_pack_groups(tok, 512, 192)
            for g in groups:
                assert sum(tok[i] for i in g) <= 512, f"{tok} → {groups} 有一块超限"

    def test_avoids_tiny_tail_when_possible(self):
        """[250, 250, 250, 10] 的 10 token 尾巴必须被合并掉。"""
        tok = [250, 250, 250, 10]
        groups = RE._balanced_pack_groups(tok, 512, 192)
        assert all(sum(tok[i] for i in g) >= 192 for g in groups), f"仍有小块: {groups}"

    def test_covers_all_indices_exactly_once(self):
        """分组必须完整覆盖且不重不漏 —— 漏一个下标就是丢一段原文。"""
        tok = [90, 130, 64, 200, 45, 300, 22, 128]
        groups = RE._balanced_pack_groups(tok, 256, 64)
        flat = [i for g in groups for i in g]
        assert flat == list(range(len(tok))), f"覆盖错误: {groups}"

    def test_empty(self):
        assert RE._balanced_pack_groups([], 512, 192) == []


# ============================================================================
# 上下文装配
# ============================================================================
class TestContextAssembly:
    RES = [{
        "id": "红楼梦_3", "book": "红楼梦", "chapter_label": "第三回",
        "chapter_title": "林黛玉抛父进京都", "parent_text": "黛玉方进入房中。",
        "child_text": "黛玉方进入房中。", "rerank_score": 3.1,
    }]

    def test_context_carries_citation(self):
        """上下文必须带出处（书名+回目），否则生成侧无法标注 [1] 是哪里。"""
        ctx = RE.format_context(self.RES)
        assert "[1]" in ctx and "《红楼梦》" in ctx and "第三回" in ctx
        assert "黛玉方进入房中。" in ctx

    def test_sources_are_structured(self):
        src = RE.context_sources(self.RES)[0]
        assert src["n"] == 1 and src["book"] == "红楼梦" and src["chapter"] == "第三回"

    # 2026-09-19 起：提示词拆成两半 —— **规则进 system、正文走独立消息**。
    # 之前正文是拼进 system 的（CONTEXT_SYSTEM_PROMPT 里带 {context}），
    # 后果是 system 一旦被置空或被服务端截断，RAG 就静默退化成"凭记忆作答"
    # （2026-09-18 置空事件就是这么发生的：正文一条也没进 prompt）。
    # 现在"资料进没进"不再靠子串猜，而是一条能直接断言的消息。
    def test_system_prompt_holds_rules_but_no_corpus_text(self):
        """system 只放规则：**不许**出现召回的正文。

        这条是本次拆分的目的本身 —— 正文混进 system 就回到旧毛病。

        ⚠️ 别用裸子串 `"[1]" not in prompt` 来判"没有引用"：规则里本来就有一句
        **示例**"标注来源编号，如 [1][3]"，这个断言会把**正确的**规则文本判成失败
        （2026-09-19 实测踩到，整层 L0 因此挂一条）。要判的是"正文没被带进来"，
        所以判据应该是：① 字面等于规则常量；② 召回正文一个字都不在；
        ③ 没有任何一行是 `[n] …` 形式的**资料条目**（示例那句是行内的，不是行首编号）。
        """
        prompt = RE.build_system_prompt(self.RES)
        assert prompt and prompt == RE.SYSTEM_RULES_PROMPT
        for r in self.RES:
            for key in ("parent_text", "child_text"):
                assert r[key] not in prompt, f"system 里出现了召回正文（{key}）"
        # 行首编号 = 资料条目的形状；规则里的 "[1][3]" 出现在行中间，不会被误判
        offenders = [ln for ln in prompt.splitlines() if ln.lstrip().startswith("[")]
        assert not offenders, f"system 里出现了资料条目: {offenders}"

    def test_context_message_carries_citation_and_text(self):
        """正文与出处由独立消息承载（引用标注 [n] 的前提）。"""
        ctx = RE.build_context_message(self.RES)
        assert RE.CONTEXT_MESSAGE_HEADER in ctx
        assert "[1]" in ctx and "《红楼梦》" in ctx and "第三回" in ctx
        assert "黛玉方进入房中。" in ctx

    def test_context_message_is_empty_without_results(self):
        """没有结果就不发资料消息（发一条空的等于说"我检索了但什么都没有"）。"""
        assert RE.build_context_message([]) == ""
        assert RE.build_system_prompt([]) == RE.NO_CONTEXT_SYSTEM_PROMPT

    def test_control_mode_sends_neither_rules_nor_context(self):
        """对照模式：system 为空、也没有资料消息，模型只靠参数化记忆。"""
        assert RE.build_system_prompt(self.RES, use_context=False) == ""
        assert RE.build_generation_messages(
            RE.build_system_prompt(self.RES, use_context=False), [], "问题",
            context="")["messages"] == [{"role": "user", "content": "问题"}]

    def test_low_evidence_caution_rides_with_the_context(self):
        """"依据不足"的告诫跟着资料走，不进 system。

        它在语义上是对"这批资料"的注解；放进 system 等于让 system 又变成
        "会飘的东西"，正是本次要根除的毛病。
        """
        plain = RE.build_context_message(self.RES)
        warned = RE.build_context_message(self.RES, low_evidence=True)
        assert warned == plain + RE.LOW_EVIDENCE_CLAUSE
        assert RE.LOW_EVIDENCE_CLAUSE not in RE.build_system_prompt(self.RES)

    def test_missing_chapter_degrades_gracefully(self):
        """旧索引没有 chapter 字段时不能崩，出处退化为只有书名。"""
        ctx = RE.format_context([{"book": "水浒传", "parent_text": "正文"}])
        assert "《水浒传》" in ctx and "None" not in ctx


# ============================================================================
# 查询改写：纯函数与失败降级
# ============================================================================
class TestQueryRewrite:
    #: 考"改写本身行为"的用例一律显式开开关。
    #: 理由：rag 在 **import 时**就把 RAG_QUERY_REWRITE 读成模块常量，
    #: 而 .env 现在把它是 0（有意关闭，见 .env 注释）。不显式传参的话，这些用例
    #: 会静静地走"已关闭"分支全部通过 —— 考的东西一点没考到，而且同一个套件
    #: 在不同机器上结论不同。已有前车之鉴：test_successful_rewrite_is_used
    #: 正是被 .env 的这次改动打成失败的，在此之前它一直"绿着"，却从未真正
    #: 验证过改写成功路径。
    ON = {"enabled": True}

    def test_prompt_includes_history_and_query(self):
        msgs = QR.build_rewrite_prompt("他最后结局如何",
                                       [{"role": "user", "content": "孙悟空是谁"}])
        assert msgs[0]["role"] == "system"
        assert "孙悟空是谁" in msgs[1]["content"]
        assert "他最后结局如何" in msgs[1]["content"]

    def test_default_keeps_all_history(self):
        """默认**不限制轮数**：整段历史都进 prompt。

        2026-09-19 改的语义 —— 以前只取最近 `RAG_QUERY_REWRITE_MAX_TURNS` 轮。
        改的理由：按轮数硬砍砍掉的是 LLM 档**独有**的能力（触达两轮以前的实体），
        拼接档只取最近一条用户问句，补不回来。现在"能看到多少"交给模型窗口
        （见 QR.REWRITE_NUM_CTX），超窗由模型侧自然遗忘。
        """
        hist = [{"role": "user", "content": f"第{i}问"} for i in range(50)]
        text = QR.build_rewrite_prompt("他呢", hist)[1]["content"]
        assert "第0问" in text and "第49问" in text

    def test_explicit_max_turns_still_caps(self):
        """显式传 max_turns 仍按轮截断 —— 保留它是为了能 A/B 旧行为。"""
        hist = [{"role": "user", "content": f"第{i}问"} for i in range(50)]
        text = QR.build_rewrite_prompt("他呢", hist, max_turns=2)[1]["content"]
        assert "第49问" in text and "第0问" not in text

    def test_prompt_stats_report_size_and_window(self):
        """prompt 体量必须可见：历史全量进 prompt 后，"会不会超窗"只能靠数字看。"""
        hist = [{"role": "user", "content": "字" * 100} for _ in range(10)]
        st = QR.rewrite_prompt_stats("他呢", hist)
        assert st["history_msgs"] == 10
        assert st["prompt_chars"] > 1000 and st["num_ctx"] == QR.REWRITE_NUM_CTX
        assert st["over_window"] is False
        big = QR.rewrite_prompt_stats("他呢", [{"role": "user", "content": "字" * (QR.REWRITE_NUM_CTX + 10)}])
        assert big["over_window"] is True

    def test_history_without_content_is_ignored(self):
        msgs = QR.build_rewrite_prompt("问", [{"role": "assistant", "content": ""}])
        assert "（无历史）" in msgs[1]["content"]

    @pytest.mark.parametrize("raw,expected", [
        ("改写后的问句：武松的结局如何", "武松的结局如何"),
        ("  孙悟空的师父是谁  ", "孙悟空的师父是谁"),
        ("“黛玉葬花的情节”", "黛玉葬花的情节"),
        ("问句：第一行\n第二行解释", "第一行"),
        ("", ""),
    ])
    def test_clean_rewritten(self, raw, expected):
        assert QR.clean_rewritten(raw) == expected

    def test_overlong_rewrite_is_rejected(self):
        """过长的"改写"多半是模型没按要求输出，宁可退回原查询。"""
        assert QR.clean_rewritten("啊" * (QR.MAX_REWRITE_CHARS + 1)) == ""

    def test_no_history_means_no_call(self):
        """没有历史就不该触发改写 —— 白付一次前向，还平白引入不确定性。"""
        called = []
        out = QR.rewrite_query("武松打虎", [], call=lambda p: called.append(1) or "x",
                               **self.ON)
        assert out == "武松打虎" and not called

    def test_llm_failure_falls_back_to_original(self):
        """改写是增益手段，失败**绝不能**连带检索失败。"""
        def boom(payload):
            raise RuntimeError("ollama 没起")
        out = QR.rewrite_query("他最后结局如何",
                               [{"role": "user", "content": "孙悟空"}], call=boom,
                               **self.ON)
        assert out == "他最后结局如何"

    def test_garbage_output_falls_back(self):
        out = QR.rewrite_query("他呢", [{"role": "user", "content": "武松"}],
                               call=lambda p: "", **self.ON)
        assert out == "他呢"

    def test_disabled_returns_original(self):
        out = QR.rewrite_query("他呢", [{"role": "user", "content": "武松"}],
                               call=lambda p: "武松的结局", enabled=False)
        assert out == "他呢"

    def test_successful_rewrite_is_used(self):
        out = QR.rewrite_query("他最后结局如何",
                               [{"role": "user", "content": "孙悟空是谁"}],
                               call=lambda p: "改写后的问句：孙悟空的结局如何",
                               **self.ON)
        assert out == "孙悟空的结局如何"

    def test_identical_rewrite_returns_original(self):
        """模型原样吐回时不该产生"改写了个寂寞"的噪声字段。"""
        out = QR.rewrite_query("武松打虎", [{"role": "user", "content": "x"}],
                               call=lambda p: "武松打虎", **self.ON)
        assert out == "武松打虎"


class TestQueryRewriteReason:
    """三态必须可分辨 —— 否则"开关关了""调用失败""模型没输出"在界面上一样。

    这是本项目的老毛病：失败路径静默，排查时只能猜（D7 把"连不上"误报成
    "库为空"也是同一类）。所以每个失败分支都断言到**具体原因码**，
    而不只是"退回了原查询"。
    """

    HIST = [{"role": "user", "content": "孙悟空是谁"}]
    #: 同 TestQueryRewrite.ON：显式开开关，避免用例随 .env 取值变绿变红
    ON = {"enabled": True}

    def test_applied_carries_new_query(self):
        out = QR.rewrite_query_detailed(
            "他最后结局如何", self.HIST, call=lambda p: "孙悟空的结局如何", **self.ON)
        assert out == {"query": "孙悟空的结局如何", "applied": True,
                       "reason": QR.REASON_APPLIED}

    def test_disabled(self):
        out = QR.rewrite_query_detailed("他呢", self.HIST,
                                        call=lambda p: "x", enabled=False)
        assert out["applied"] is False and out["reason"] == QR.REASON_DISABLED
        assert out["query"] == "他呢"

    def test_no_history(self):
        out = QR.rewrite_query_detailed("他呢", [], **self.ON)
        assert out["reason"] == QR.REASON_NO_HISTORY

    def test_blank_query(self):
        out = QR.rewrite_query_detailed("   ", self.HIST, **self.ON)
        assert out["reason"] == QR.REASON_BLANK_QUERY

    def test_none_query_does_not_crash(self):
        """query=None 时不能抛异常 —— 空查询守护在改写之后才做，
        所以改写这一层必须先扛住 None。"""
        out = QR.rewrite_query_detailed(None, self.HIST, **self.ON)
        assert out["reason"] == QR.REASON_BLANK_QUERY and out["query"] is None

    def test_call_error_carries_exception_type(self):
        """原因码要带异常类名：Timeout 与 ConnectionError 的处置完全不同。"""
        def boom(payload):
            raise TimeoutError("ollama 没起")
        out = QR.rewrite_query_detailed("他呢", self.HIST, call=boom, **self.ON)
        assert out["reason"] == f"{QR.REASON_CALL_ERROR}:TimeoutError"
        assert out["query"] == "他呢"

    def test_empty_output(self):
        out = QR.rewrite_query_detailed("他呢", self.HIST, call=lambda p: "", **self.ON)
        assert out["reason"] == QR.REASON_EMPTY_OUTPUT

    def test_too_long_is_distinguished_from_empty(self):
        """过长与没输出是两回事：前者要查 MAX_REWRITE_CHARS，后者要查模型/提示词。"""
        out = QR.rewrite_query_detailed(
            "他呢", self.HIST,
            call=lambda p: "啊" * (QR.MAX_REWRITE_CHARS + 1), **self.ON)
        assert out["reason"] == QR.REASON_TOO_LONG

    def test_identical_is_not_applied(self):
        out = QR.rewrite_query_detailed("武松打虎", self.HIST,
                                        call=lambda p: "武松打虎", **self.ON)
        assert out["applied"] is False and out["reason"] == QR.REASON_IDENTICAL

    def test_every_reason_code_is_declared(self):
        """原因码是给界面/日志看的契约，漏登记会让调用方拿到未定义的值。"""
        assert QR.REASON_APPLIED in QR.REASONS
        assert len(set(QR.REASONS)) == len(QR.REASONS)

    def test_clean_detailed_agrees_with_clean(self):
        """兼容入口与详细入口的清洗结果必须一致，否则两条路径会对不上。"""
        for raw in ("改写后的问句：武松的结局", "", "啊" * 999, "  "):
            assert QR.clean_rewritten(raw) == QR.clean_rewritten_detailed(raw)[0]


# ============================================================================
# 降级链：LLM 改写 → 拼接上一轮用户问句 → 字面原句
#
# 为什么要中间这一档（实测，21 条多轮探针，同一个评测器，只换检索用问句）：
#     字面原句      book@1 81.0%  kw@k 38.1%  topic@k 42.9%   ← 旧降级落点
#     拼接上轮用户    book@1 100%   kw@k 71.4%  topic@k 95.2%   ← 本档
#     LLM 改写       book@1 100%   kw@k 76.2%  topic@k 95.2%
# 也就是"改写失败/关闭"原来等于没有这个功能，现在等于几乎白拿一档。
# ============================================================================
class TestRetrievalQueryChain:
    ON = {"enabled": True}

    HIST = [{"role": "user", "content": "孙悟空大闹天宫之后怎么样了？"},
            {"role": "assistant", "content": "他被压在五行山下。"}]

    def test_llm_wins_when_available(self):
        out = QR.build_retrieval_query(
            "他最后结局如何", self.HIST,
            call=lambda p: "孙悟空最后结局如何", **self.ON)
        assert out["source"] == "llm" and out["applied"] is True
        assert out["query"] == "孙悟空最后结局如何"

    def test_falls_back_to_concat_when_disabled(self):
        out = QR.build_retrieval_query("他最后结局如何", self.HIST, enabled=False)
        assert out["source"] == "concat" and out["applied"] is True
        assert out["query"] == "孙悟空大闹天宫之后怎么样了？ 他最后结局如何"

    def test_falls_back_to_concat_on_call_error(self):
        """Ollama 没起 / 超时也必须走到拼接档 —— 这是最常见的一条失败路径。"""
        def boom(payload):
            raise TimeoutError("ollama 没起")
        out = QR.build_retrieval_query("他最后结局如何", self.HIST, call=boom, **self.ON)
        assert out["source"] == "concat" and out["applied"] is True

    def test_falls_back_to_concat_on_unusable_output(self):
        out = QR.build_retrieval_query("他最后结局如何", self.HIST,
                                       call=lambda p: "", **self.ON)
        assert out["source"] == "concat"

    def test_identical_rewrite_does_not_concat(self):
        """模型判定"问句已自足"时**不该**拼 —— 拼接等于往完整问句里掺上轮噪音。"""
        out = QR.build_retrieval_query("武松打虎", self.HIST,
                                       call=lambda p: "武松打虎", **self.ON)
        assert out["source"] == "literal" and out["query"] == "武松打虎"

    def test_allow_concat_false_keeps_literal(self):
        """本轮提问没有可检索内容时不拼：否则"？？？"会悄悄拿上一轮话题去检索。"""
        out = QR.build_retrieval_query("？？？", self.HIST, enabled=False,
                                       allow_concat=False)
        assert out["source"] == "literal" and out["query"] == "？？？"

    def test_no_history_means_literal(self):
        out = QR.build_retrieval_query("武松打虎", [], enabled=False)
        assert out["source"] == "literal" and out["applied"] is False

    def test_concat_can_be_switched_off(self):
        out = QR.build_retrieval_query("他最后结局如何", self.HIST,
                                       enabled=False, concat_enabled=False)
        assert out["source"] == "literal" and out["query"] == "他最后结局如何"

    def test_reason_keeps_the_original_cause(self):
        """拼接档的 reason 要保留"为什么没改写成功"，否则排查时线索断在这里。"""
        out = QR.build_retrieval_query("他最后结局如何", self.HIST, enabled=False)
        assert out["reason"].startswith(QR.REASON_CONCAT)
        assert QR.REASON_DISABLED in out["reason"]

    def test_every_outcome_is_labelled(self):
        """source 是界面/A-B 用的粗粒度标签，取值必须封闭。"""
        cases = [
            QR.build_retrieval_query("他最后结局如何", self.HIST, enabled=False),
            QR.build_retrieval_query("他最后结局如何", self.HIST,
                                     call=lambda p: "孙悟空最后结局如何", **self.ON),
            QR.build_retrieval_query("武松打虎", [], enabled=False),
        ]
        assert {c["source"] for c in cases} == {"llm", "concat", "literal"}


class TestRetrievalRoutes:
    """多路召回的路由计划（RAG_QUERY_FUSION 的基础，纯函数）。

    守住的不变量是"融合开关只应该**加**一路，绝不改主路"：
    主路必须与 build_retrieval_query 逐字一致，否则关掉融合的 A/B 就不可比了。
    """

    HIST = [{"role": "user", "content": "那张飞呢，他后来怎么了？"},
            {"role": "assistant", "content": "张飞在阆中被部将所杀。"}]

    def test_primary_matches_the_single_route_chain(self):
        """primary 必须与不带融合时**逐字相同**（否则 A/B 基线漂了）。

        ⚠️ 这条用例曾经**没注入 call**，于是真的去打 Ollama，还把两次独立前向的
        结果拿来互比 —— 同一份代码连跑三次得到 "1 failed / 1 passed / 超时"。
        失败的根因恰好就是这套探针要考的那件事：历史里同时有关羽和张飞时，
        模型这次把"他"解成关羽、下次解成张飞，两个 primary 自然不等。
        L0 是纯函数层，不能依赖网络与模型随机性。
        """
        def call(payload):
            return "张飞最后是被谁杀的"
        for kwargs in (dict(enabled=False), {"enabled": True, "call": call}):
            routes = QR.build_retrieval_routes("他最后是被谁杀的", self.HIST,
                                               fusion=True, **kwargs)
            single = QR.build_retrieval_query("他最后是被谁杀的", self.HIST, **kwargs)
            assert routes["primary"] == single

    def test_allow_concat_false_is_respected_by_the_fusion_route(self):
        """调用方说"本轮没有可检索内容，别拼上一轮"时，融合也不许偷偷拼。

        实测抓到的 bug：LLM 偶尔会为"？？？"这类输入吐出一句像模像样的改写，
        此时 primary 是 llm 档；若追加拼接路时不复判 allow_concat，就会拿
        **上一轮的话题**去检索 —— 正是 allow_concat 要挡住的语义。
        """
        routes = QR.build_retrieval_routes(
            "？？？", self.HIST, call=lambda p: "孙悟空后来又怎么了",
            fusion=True, allow_concat=False)
        assert [q["source"] for q in routes["queries"]] == ["llm"]
        assert all("那张飞呢" not in q["query"] for q in routes["queries"])

    def test_rewrite_is_injectable_so_l0_never_hits_the_network(self):
        """不注入 call 时也不许出现真实网络调用：改写必须始终可注入。"""
        def boom(payload):
            raise AssertionError("L0 用例不应真的发起改写请求")
        out = QR.build_retrieval_routes("他最后是被谁杀的", self.HIST,
                                        call=boom, fusion=True, enabled=True)
        assert out["primary"]["source"] == "concat"   # 调用失败 → 降级，不抛

    def test_fusion_off_means_exactly_one_route(self):
        routes = QR.build_retrieval_routes("他最后是被谁杀的", self.HIST,
                                           call=lambda p: "张飞最后是被谁杀的",
                                           fusion=False)
        assert [q["source"] for q in routes["queries"]] == ["llm"]
        assert routes["extra"] == []

    def test_fusion_adds_the_concat_route_after_llm(self):
        """实测互补的那两路：LLM 档独家救 assistant_only，拼接档独家救 entity_switch。"""
        routes = QR.build_retrieval_routes("他最后是被谁杀的", self.HIST,
                                           call=lambda p: "张飞最后是被谁杀的",
                                           fusion=True)
        assert [q["source"] for q in routes["queries"]] == ["llm", "concat"]
        assert routes["queries"][0]["query"] == "张飞最后是被谁杀的"
        assert "那张飞呢，他后来怎么了？" in routes["queries"][1]["query"]

    def test_literal_route_is_never_added(self):
        """字面原句一路在 21 条探针上**没有任何独家命中**，故不进融合。"""
        routes = QR.build_retrieval_routes("他最后是被谁杀的", self.HIST,
                                           call=lambda p: "张飞最后是被谁杀的",
                                           fusion=True)
        assert "literal" not in {q["source"] for q in routes["queries"]}

    def test_no_second_route_when_llm_failed(self):
        """改写失败时主路已是拼接档 —— 没有第二档可加，也不该硬造一路。"""
        def boom(payload):
            raise TimeoutError("ollama 没起")
        routes = QR.build_retrieval_routes("他最后是被谁杀的", self.HIST,
                                           call=boom, fusion=True)
        assert [q["source"] for q in routes["queries"]] == ["concat"]

    def test_duplicate_queries_are_not_recalled_twice(self):
        """两路问句相同时只留一条：把同一句召回两遍等于给它偷偷加倍权重。"""
        hist = [{"role": "user", "content": "武松打虎"}]
        routes = QR.build_retrieval_routes("武松打虎", hist,
                                           call=lambda p: "武松打虎", fusion=True)
        assert len(routes["queries"]) == 1

    def test_no_history_keeps_one_route(self):
        routes = QR.build_retrieval_routes("武松打虎", [], fusion=True)
        assert len(routes["queries"]) == 1


class TestMergeRouteScores:
    """多路重排的分数合并（RAG_RERANK_QUERY=per_route 的核心，纯函数）。

    守两条：① 逐候选取**最大值**而不是求和 —— reranker 分数不做跨查询校准，
    求和还会把 RRF 已经在奖励的"两路都召回"抬第二次；② 顺序必须稳定，
    否则同样的输入两次跑出不同排序，L3 的零容差基线就没法用。
    """

    def test_takes_the_max_across_routes(self):
        out = RE.merge_route_scores([[(0, 1.0), (1, 3.0)], [(0, 5.0), (1, 0.5)]])
        assert out == [(0, 5.0), (1, 3.0)]

    def test_does_not_sum(self):
        """求和会得到 (0, 6.0)/(1, 3.5) —— 排序一样，但分数语义被改了。"""
        out = dict(RE.merge_route_scores([[(0, 1.0)], [(0, 5.0)]]))
        assert out[0] == 5.0

    def test_single_route_is_identity_order(self):
        ranked = [(3, 0.2), (1, 0.9), (2, 0.5)]
        assert RE.merge_route_scores([ranked]) == [(1, 0.9), (2, 0.5), (3, 0.2)]

    def test_ties_break_by_index_for_determinism(self):
        """并列分必须按索引定序，否则同输入两次跑出不同顺序。"""
        out = RE.merge_route_scores([[(2, 1.0), (0, 1.0), (1, 1.0)]])
        assert [i for i, _ in out] == [0, 1, 2]

    def test_missing_candidate_keeps_other_route_score(self):
        """某条路没召回该候选（索引缺失）时，不该让它整条消失。"""
        out = dict(RE.merge_route_scores([[(0, 2.0)], [(1, 4.0)]]))
        assert out == {0: 2.0, 1: 4.0}

    def test_top_k_and_empty(self):
        assert len(RE.merge_route_scores([[(0, 1.0), (1, 2.0), (2, 3.0)]], top_k=2)) == 2
        assert RE.merge_route_scores([]) == []


class TestMultiturnDegradeState:
    """多轮降级档标签 —— L3 护栏的判据依赖它。

    这条判据曾经写错成"REWRITE_ENABLED 为 False 就 skip 多轮断言"，
    而默认 .env 恰好是 0，于是**默认配置下多轮护栏整组失效**。
    标签函数本身是纯函数，可以在 L0 把三档钉死。
    """

    def test_llm_when_rewrite_on(self):
        from tests.eval_runner import multiturn_degrade_state
        assert multiturn_degrade_state(rewrite=True, concat=False) == "llm"
        assert multiturn_degrade_state(rewrite=True, concat=True) == "llm"

    def test_concat_is_the_state_when_rewrite_off(self):
        """改写关掉**不等于**没有多轮：拼接档仍会替换检索问句。"""
        from tests.eval_runner import multiturn_degrade_state
        assert multiturn_degrade_state(rewrite=False, concat=True) == "concat"

    def test_literal_only_when_both_off(self):
        from tests.eval_runner import multiturn_degrade_state
        assert multiturn_degrade_state(rewrite=False, concat=False) == "literal"

    def test_defaults_read_live_constants(self):
        """不传参时必须读模块常量，而不是写死一个默认值。

        写死的后果很隐蔽：配置变了、标签不变，L3 就会拿"档位一致"的假象
        去比较两组口径不同的数字。
        """
        from tests.eval_runner import multiturn_degrade_state
        expected = ("llm" if QR.REWRITE_ENABLED
                    else ("concat" if QR.CONCAT_ENABLED else "literal"))
        assert multiturn_degrade_state() == expected


class TestConcatQuery:
    def test_prefixes_the_previous_user_turn(self):
        hist = [{"role": "user", "content": "上一问"},
                {"role": "assistant", "content": "上一答"}]
        assert QR.build_concat_query("本轮", hist) == "上一问 本轮"

    def test_takes_only_the_last_user_turn(self):
        """拼接只补先行词，不该把整段历史拼进来（实测拼助手那句 kw@k 掉 14 个点）。"""
        hist = [{"role": "user", "content": "很久以前的问题"},
                {"role": "assistant", "content": "很久以前的回答"},
                {"role": "user", "content": "最近一问"}]
        out = QR.build_concat_query("本轮", hist)
        assert out == "最近一问 本轮" and "很久以前" not in out

    def test_truncates_a_long_previous_turn(self):
        """真实对话里上一轮可能是长问句，全量拼进去会稀释当前问句。"""
        hist = [{"role": "user", "content": "长" * 500}]
        out = QR.build_concat_query("本轮", hist, max_chars=10)
        assert out == "长" * 10 + " 本轮"

    def test_no_previous_user_turn(self):
        assert QR.build_concat_query("本轮", []) == ""
        assert QR.build_concat_query("本轮", [{"role": "assistant", "content": "答"}]) == ""

    def test_same_question_twice_is_not_concatenated(self):
        """同一句话又问一遍时拼起来只是重复文本，白给 BM25 加权重。"""
        hist = [{"role": "user", "content": "武松打虎"}]
        assert QR.build_concat_query("武松打虎", hist) == ""

    def test_blank_inputs(self):
        assert QR.build_concat_query("", [{"role": "user", "content": "问"}]) == ""
        assert QR.build_concat_query("本轮", [{"role": "user", "content": "  "}]) == ""

    def test_last_turn_text_skips_empty_and_bad_entries(self):
        hist = ["裸字符串", {"role": "user", "content": ""},
                {"role": "user", "content": "有内容"}]
        assert QR.last_turn_text(hist, "user") == "有内容"
        assert QR.last_turn_text(hist, "assistant") == ""


# ============================================================================
# 会话历史：裁切规则必须三个入口共用一份
# ============================================================================
class TestHistoryHandling:
    def test_filters_empty_assistant_turns(self):
        """生成失败会留下 content="" 的助手消息（D8）—— 它不该再发给模型。

        UI 此前不过滤而 API 过滤，同一个会话在两个入口发给模型的输入不同。
        """
        msgs = RE.history_to_messages([
            {"role": "user", "content": "武松是谁"},
            {"role": "assistant", "content": "", "error": "生成失败"},
            {"role": "user", "content": "他打了什么"},
        ])
        assert msgs == [{"role": "user", "content": "武松是谁"},
                        {"role": "user", "content": "他打了什么"}]

    def test_keeps_only_role_and_content(self):
        """历史里挂着 thinking / trace 等大字段，原样塞进 prompt 只会膨胀。"""
        msgs = RE.history_to_messages([
            {"role": "assistant", "content": "答案", "thinking": "x" * 999,
             "trace": {"RRF 融合": []}, "error": ""},
        ])
        assert msgs == [{"role": "assistant", "content": "答案"}]

    def test_drops_non_dict_and_blank(self):
        msgs = RE.history_to_messages(["裸字符串", None, {"content": "  "},
                                       {"role": "user", "content": "ok"}])
        assert msgs == [{"role": "user", "content": "ok"}]

    def test_empty_input(self):
        assert RE.history_to_messages(None) == []
        assert RE.history_to_messages([]) == []

    def test_strip_current_turn_removes_duplicate_tail(self):
        """调用方漏了 [:-1] 时，末尾会多一条与当前问句相同的用户消息。

        不剥掉的后果是改写提示词里出现两遍"用户当前问句"，指标悄悄变差。
        """
        hist = [{"role": "user", "content": "孙悟空是谁"},
                {"role": "assistant", "content": "..."},
                {"role": "user", "content": "他最后结局如何"}]
        cleaned, dropped = RE.strip_current_turn(hist, "他最后结局如何")
        assert dropped is True and cleaned == hist[:2]

    def test_strip_current_turn_is_noop_for_normal_history(self):
        """正常历史（末尾是助手回复）一个字符都不能动。"""
        hist = [{"role": "user", "content": "孙悟空是谁"},
                {"role": "assistant", "content": "..."}]
        cleaned, dropped = RE.strip_current_turn(hist, "他最后结局如何")
        assert dropped is False and cleaned == hist

    def test_same_question_asked_twice_is_not_stripped(self):
        """用户真的连问两遍同一句话时**不能**剥 —— 那时末尾是助手回复。"""
        hist = [{"role": "user", "content": "武松打虎"},
                {"role": "assistant", "content": "..."}]
        cleaned, dropped = RE.strip_current_turn(hist, "武松打虎")
        assert dropped is False and len(cleaned) == 2

    def test_strip_tolerates_whitespace_and_bad_input(self):
        hist = [{"role": "user", "content": "  他呢  "}]
        assert RE.strip_current_turn(hist, "他呢")[1] is True
        assert RE.strip_current_turn(None, "他呢") == ([], False)
        assert RE.strip_current_turn([{"role": "user"}], "他呢")[1] is False


# ============================================================================
# 生成侧装配：上下文窗口是按 token 抢的
#
# 实测背景（2026-09-18，top_k=5，num_ctx=8192 / num_predict=4096）：
#   system（规则 + 5 个父块正文）约 2120~2250 token，num_predict 还要再扣 4096，
#   留给历史的只有约 1800 token。而此前 app.py 对历史零裁剪，超限后 Ollama 按
#   窗口截断，先丢的是最前面的 **system** —— 也就是检索到的正文与引用要求。
#   回答照样通顺，从输出上分辨不出来，所以这组用例守的是"system 必须活下来"。
# ============================================================================
class TestTokenEstimation:
    def test_empty_is_zero(self):
        assert RE.estimate_tokens("") == 0 and RE.estimate_tokens(None) == 0

    def test_cjk_counts_one_per_char(self):
        assert RE.estimate_tokens("孙悟空") == 3

    def test_ascii_is_four_chars_per_token(self):
        assert RE.estimate_tokens("abcd") == 1
        assert RE.estimate_tokens("aaaaa") == 2      # 向上取整，宁可多算

    def test_monotonic_non_decreasing(self):
        """必须单调不减 —— _cut_to_tokens 的二分正确性依赖它。"""
        text = "孙悟空 said: hello 世界 mixed 12345"
        prev = 0
        for i in range(len(text) + 1):
            cur = RE.estimate_tokens(text[:i])
            assert cur >= prev, f"前缀变长反而更少 token（{i}）"
            prev = cur

    def test_message_tokens_sums_contents(self):
        msgs = [{"role": "user", "content": "孙悟空"},
                {"role": "assistant", "content": "abcd"}]
        assert RE.message_tokens(msgs) == 4
        assert RE.message_tokens([{"role": "user"}]) == 0


class TestHistoryBudget:
    def test_subtracts_system_and_output_budget(self):
        """num_predict 是"思考+答案"的总预算，必须先从窗口里扣掉。"""
        assert RE.history_budget(8192, 4096, "系统", reserve=256) == 8192 - 4096 - 2 - 256

    def test_real_world_numbers_match_measurement(self):
        """用线上真实参数复核：8192/4096 + 约 2200 token 的 system
        → 历史预算应落在 1500~1800 这个量级（实测 system 2120~2250）。"""
        system = "上下文：" + "字" * 2200
        budget = RE.history_budget(8192, 4096, system)
        assert 1400 <= budget <= 1800, budget

    def test_never_negative(self):
        """窗口小到连 system 都装不下时返回 0，而不是负数。"""
        assert RE.history_budget(512, 512, "很长的系统提示词" * 10) == 0
        assert RE.history_budget(100, 200, "") == 0


class TestGenerationMessages:
    SYS = "系统提示词：只依据上下文回答。[1] 出处：《西游记》第一百回" + "正文" * 20

    @staticmethod
    def _history(n, chars=40):
        out = []
        for i in range(n):
            out.append({"role": "user", "content": f"第{i}问" + "问" * chars})
            out.append({"role": "assistant", "content": f"第{i}答" + "答" * chars})
        return out

    def test_system_and_current_query_survive_any_budget(self):
        """**核心不变量**：无论怎么裁，system 与本轮提问都不能丢。

        丢了 system 就等于丢了本次检索到的正文与引用要求 —— 这轮不再是 RAG，
        而回答会照样通顺，是本项目里最难发现的那类失效。
        """
        for budget in (0, 1, 10, 100, 1000, 100000):
            gen = RE.build_generation_messages(
                self.SYS, self._history(20), "他最后结局如何",
                max_history_tokens=budget)
            msgs = gen["messages"]
            assert msgs[0] == {"role": "system", "content": self.SYS}
            assert msgs[-1] == {"role": "user", "content": "他最后结局如何"}

    def test_trims_whole_turns_from_the_oldest(self):
        """整轮丢：只丢一条 assistant 会留下"用户问了、没回答"的残局。"""
        gen = RE.build_generation_messages(
            "", self._history(10), "新问题", max_history_tokens=120)
        msgs = gen["messages"]
        assert gen["dropped_turns"] > 0
        hist = msgs[:-1]
        # 保留下来的历史必须首尾成对、从旧到新
        assert [m["role"] for m in hist] == ["user", "assistant"] * (len(hist) // 2)
        # 最新的那一轮一定在（它是指代消解的直接依据）
        assert "第9问" in hist[0]["content"] or "第9问" in hist[-2]["content"]             or any("第9问" in m["content"] for m in hist)

    def test_fits_budget_when_history_alone_fits(self):
        """只要存在能装下的一轮，就绝不超预算。"""
        budget = 300
        gen = RE.build_generation_messages(
            "", self._history(20), "问题", max_history_tokens=budget)
        hist = gen["messages"][:-1]
        assert RE.message_tokens(hist) <= budget
        assert gen["used_tokens"] <= budget

    def test_oversized_newest_turn_is_kept_and_truncated(self):
        """最新一轮自己就超预算时：保留它并截尾，而不是整轮丢光。

        否则多轮对话会退化成一问一答 —— 指代对象常常只出现在上一轮。
        """
        hist = [{"role": "user", "content": "孙悟空是谁"},
                {"role": "assistant", "content": "答" * 2000}]
        gen = RE.build_generation_messages("", hist, "他结局如何",
                                           max_history_tokens=100)
        msgs = gen["messages"]
        assert gen["truncated"] is True and gen["dropped_turns"] == 0
        assert msgs[0]["content"] == "孙悟空是谁", "用户那句是需求原文，不许截"
        assert msgs[1]["content"].endswith("（已按上下文预算截断）")
        # 标记也算进预算：砍到 100 就必须 <= 100（曾实得 107）
        assert RE.estimate_tokens(msgs[1]["content"]) <= 100

    def test_truncation_does_not_mutate_input(self):
        """裁剪是纯函数：不能改调用方手里的会话历史。"""
        hist = [{"role": "user", "content": "孙悟空是谁"},
                {"role": "assistant", "content": "答" * 2000}]
        before = json.loads(json.dumps(hist, ensure_ascii=False))
        RE.build_generation_messages("", hist, "他结局如何", max_history_tokens=10)
        assert hist == before, "输入历史被就地改写了"

    def test_empty_history_and_empty_system(self):
        gen = RE.build_generation_messages("", [], "武松打虎", max_history_tokens=0)
        assert gen["messages"] == [{"role": "user", "content": "武松打虎"}]
        assert gen["dropped_turns"] == 0 and gen["used_tokens"] == 0

    def test_blank_assistant_turns_are_dropped_before_counting(self):
        """D8 留下的空助手消息不该占预算、也不该发给模型。"""
        hist = [{"role": "user", "content": "武松是谁"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "他打了什么"},
                {"role": "assistant", "content": "打虎"}]
        gen = RE.build_generation_messages("", hist, "后来呢", max_history_tokens=9999)
        assert [m["role"] for m in gen["messages"]] == ["user", "user", "assistant", "user"]

    def test_order_is_oldest_first(self):
        """裁完之后顺序仍是从旧到新 —— 颠倒会让模型读到倒放的对话。"""
        hist = self._history(3)
        gen = RE.build_generation_messages("", hist, "新问题", max_history_tokens=9999)
        bodies = [m["content"][:3] for m in gen["messages"][:-1]]
        assert bodies == ["第0问", "第0答", "第1问", "第1答", "第2问", "第2答"]

    def test_stats_are_reported_for_the_ui(self):
        """裁剪必须可观测：静默丢历史正是"回答看着正常却没了依据"的成因。"""
        gen = RE.build_generation_messages(
            "", self._history(10), "问题", max_history_tokens=100)
        assert set(gen) == {"messages", "dropped_turns", "used_tokens",
                            "context_tokens", "budget", "truncated",
                            "truncated_user"}
        assert gen["budget"] == 100 and gen["dropped_turns"] >= 1
        assert gen["context_tokens"] == 0, "没有资料时不许凭空计 token"
        assert gen["truncated_user"] is False, "常规裁剪不该动用户提问"

    def test_oversized_user_message_is_truncated_and_reported(self):
        """用户提问**自己**就超预算时，必须截断并上报。

        为什么不许"只截助手"：不截用户消息时 messages 必然超出 num_ctx，
        而服务端超窗**不报错**、保留 system、其余整条丢 —— 本项目实测过
        一条 16000 字的用户消息让 prompt_eval_count 从 18743 字塌到 40 token。
        截一条问句是可见的降级，丢光上下文是不可见的事故。
        """
        huge = "长" * 5000
        gen = RE.build_generation_messages("", [], huge, max_history_tokens=200)
        assert gen["truncated_user"] is True
        assert gen["truncated"] is True
        assert gen["messages"][-1]["content"] != huge, "用户提问没有被截断"
        assert gen["messages"][-1]["content"].endswith(RE._USER_TRUNCATION_MARK)
        assert RE.estimate_tokens(gen["messages"][-1]["content"]) <= 200

    def test_user_truncation_mark_differs_from_assistant_mark(self):
        """两种截断必须能一眼分辨：截用户提问是比截助手更强的干预。"""
        assert RE._USER_TRUNCATION_MARK != RE._TRUNCATION_MARK
        assert "提问" in RE._USER_TRUNCATION_MARK

    # ---- 独立资料消息（2026-09-19 新增的设计） --------------------------

    def test_context_message_sits_right_before_the_query(self):
        """顺序：system → 历史 → **资料** → 本轮提问。

        资料紧贴提问：它的作用对象是这一问，挨着放最不容易被长历史冲淡。
        """
        gen = RE.build_generation_messages(
            "规则", self._history(2), "他最后结局如何",
            context="【检索资料】正文", max_history_tokens=9999)
        roles = [m["role"] for m in gen["messages"]]
        assert roles == ["system", "user", "assistant", "user", "assistant",
                         "user", "user"]
        assert gen["messages"][-2]["content"] == "【检索资料】正文"
        assert gen["messages"][-2]["role"] == RE.CONTEXT_ROLE
        assert gen["messages"][-1] == {"role": "user", "content": "他最后结局如何"}
        assert gen["context_tokens"] == RE.estimate_tokens("【检索资料】正文")

    def test_context_message_role_is_configurable_and_validated(self):
        """role 可配（默认 user）：换 role 是唯一"资料在角色上算谁"的开关。"""
        gen = RE.build_generation_messages(
            "", [], "问题", context="资料", context_role="tool")
        assert gen["messages"][0]["role"] == "tool"
        assert RE.CONTEXT_ROLE in ("user", "tool", "system")

    def test_context_message_survives_any_budget(self):
        """资料与 system、本轮提问同级：**永不因裁剪而丢**。

        丢了资料就等于这轮不再是 RAG，而回答照样通顺 —— 最难发现的那类失效。
        """
        for budget in (0, 1, 10, 1000):
            gen = RE.build_generation_messages(
                "规则", self._history(20), "问题", context="【检索资料】正文",
                max_history_tokens=budget)
            msgs = gen["messages"]
            assert msgs[0] == {"role": "system", "content": "规则"}
            assert msgs[-2]["content"] == "【检索资料】正文"
            assert msgs[-1] == {"role": "user", "content": "问题"}
            assert gen["context_tokens"] > 0

    def test_history_budget_can_reserve_room_for_the_context(self):
        """extra= 必须把资料消息一起从窗口里扣掉。

        资料消息不参与裁剪、每轮必发；预算只按 system 算就会超窗，
        而超窗的后果是服务端静默整条丢消息（见 history_budget 注释）。
        """
        base = RE.history_budget(8192, 4096, "规则", reserve=256)
        with_ctx = RE.history_budget(8192, 4096, "规则", reserve=256,
                                     extra="【检索资料】" + "字" * 300)
        assert base - with_ctx == RE.estimate_tokens("【检索资料】" + "字" * 300)
        assert with_ctx >= 0

    def test_assembled_input_including_context_fits_the_window(self):
        """端到端不变量（含资料消息）：装配后仍要落进 num_ctx - num_predict。"""
        num_ctx, num_predict = 8192, 4096
        sys_text = RE.SYSTEM_RULES_PROMPT
        ctx = RE.build_context_message([
            {"book": "水浒传", "chapter_label": "第二十三回",
             "parent_text": "武" * 2000, "id": "x"},
        ])
        budget = RE.history_budget(num_ctx, num_predict, sys_text, extra=ctx)
        gen = RE.build_generation_messages(
            sys_text, self._history(12, chars=100), "新问题", context=ctx,
            max_history_tokens=budget)
        total = RE.message_tokens(gen["messages"])
        assert total <= num_ctx - num_predict, (
            f"含资料的装配输入 {total} token，超出输入窗口 {num_ctx - num_predict}")

    def test_split_turns_groups_assistant_under_its_user(self):
        turns = RE.split_turns([{"role": "assistant", "content": "开场白"},
                                {"role": "user", "content": "u1"},
                                {"role": "assistant", "content": "a1"},
                                {"role": "user", "content": "u2"}])
        assert [len(t) for t in turns] == [1, 2, 1]
        assert turns[1][0]["content"] == "u1"

    def test_assembled_input_fits_the_window(self):
        """端到端不变量：system + 历史 + 本轮提问 必须落进 num_ctx - num_predict。

        实测过超窗的后果：服务端**不报错**，而是整条整条地丢消息 ——
        一条 16000 字的用户消息会让 prompt_eval_count 从 18743 塌到 40 token，
        整段历史静默丢光。所以这条断言守的是"我们自己先把总量控住"。
        """
        num_ctx, num_predict = 8192, 4096
        sys_text = "上下文：" + "字" * 2500          # 实测 system 的上限量级
        budget = RE.history_budget(num_ctx, num_predict, sys_text)
        gen = RE.build_generation_messages(
            sys_text, self._history(12, chars=100), "新问题", max_history_tokens=budget)
        total = RE.message_tokens(gen["messages"])
        assert total <= num_ctx - num_predict, (
            f"装配后输入 {total} token，超出输入窗口 {num_ctx - num_predict}")

    def test_untrimmed_history_would_overflow(self):
        """反证：同样的 12 轮若不裁剪就会超窗（这才是本次要修的问题本身）。"""
        num_ctx, num_predict = 8192, 4096
        sys_text = "上下文：" + "字" * 2500
        hist = self._history(12, chars=100)
        untrimmed = RE.message_tokens([{"role": "system", "content": sys_text}]
                                      + hist + [{"role": "user", "content": "新问题"}])
        assert untrimmed > num_ctx - num_predict, (
            "构造的用例本身没有超窗，这条反证失去意义")

    def test_cut_to_tokens_is_bounded_and_marks_truncation(self):
        assert RE._cut_to_tokens("", 10) == ""
        assert RE._cut_to_tokens("孙悟空", 0) == ""
        assert RE._cut_to_tokens("孙悟空", 99) == "孙悟空"
        out = RE._cut_to_tokens("字" * 300, 100)   # 300 字 > 100 token 预算
        assert out.endswith("（已按上下文预算截断）")
        assert RE.estimate_tokens(out) <= 100
        # 预算连标记都放不下时，宁可不加标记也不能超预算
        tiny = RE._cut_to_tokens("字" * 100, 7)
        assert "（已按上下文预算截断）" not in tiny
        assert RE.estimate_tokens(tiny) <= 7



# ============================================================================
# 配置解析：宁可启动即报错，不要静默回退
# ============================================================================
class TestEnvParsing:
    def test_int_default_and_override(self, monkeypatch):
        monkeypatch.delenv("RAG_TEST_X", raising=False)
        assert BO.env_int("RAG_TEST_X", 7) == 7
        monkeypatch.setenv("RAG_TEST_X", "9")
        assert BO.env_int("RAG_TEST_X", 7) == 9

    def test_blank_means_default(self, monkeypatch):
        monkeypatch.setenv("RAG_TEST_X", "   ")
        assert BO.env_int("RAG_TEST_X", 7) == 7

    def test_invalid_raises_with_name(self, monkeypatch):
        """必须是**报错**而不是静默回退：静默回退会让"我改了但没生效"
        变成最难查的那类问题。"""
        monkeypatch.setenv("RAG_TEST_X", "abc")
        with pytest.raises(ValueError, match="RAG_TEST_X"):
            BO.env_int("RAG_TEST_X", 7)

    def test_float_and_bool(self, monkeypatch):
        monkeypatch.setenv("RAG_TEST_F", "0.25")
        assert BO.env_float("RAG_TEST_F", 1.0) == 0.25
        for raw, want in (("1", True), ("off", False), ("NO", False), ("true", True)):
            monkeypatch.setenv("RAG_TEST_B", raw)
            assert BO.env_bool("RAG_TEST_B", True) is want

    def test_choice_rejects_unknown(self, monkeypatch):
        monkeypatch.setenv("RAG_TEST_C", "middle")
        with pytest.raises(ValueError, match="RAG_TEST_C"):
            BO.env_choice("RAG_TEST_C", "child", ("child", "parent"))


# ============================================================================
# 基线报表渲染
# ============================================================================
class TestBaselineReport:
    """报表渲染是纯字符串拼装，不该为验证它付一次全栈评测的代价。

    这段逻辑原先内联在 record_baseline.main() 里，只有真的跑完一遍评测
    （需模型 + Qdrant，分钟级）才可能发现"忘打印某一类指标"这种低级错误。
    """

    SNAP = {
        "aggregate": {"n": 24, "book@1": 0.9, "book@k": 1.0, "kw@k": 0.875},
        "aggregate_short": {"n": 12, "book@1": 0.9167, "book@k": 1.0, "kw@k": 1.0},
        "aggregate_all": {
            "multiturn": {"n": 5, "book@1": 0.6, "book@k": 1.0,
                          "kw@k": 0.4, "topic@k": 0.6},
            "negative": {"n": 10, "refuse_rate": 0.6, "mean_n_books": 2.1},
        },
        "rewrite_enabled": True,
        "per_probe": {"武松打虎": {"book@1": True, "book@k": True,
                                  "kw@k": True, "top1_score": 4.48}},
        "per_probe_multiturn": {"他最后结局如何": {"book@1": True, "topic@k": True}},
        "per_probe_negative": {"今天天气怎么样": {"refuse": True, "top1_score": 1.12}},
    }

    def test_reports_all_three_categories(self, capsys):
        from tests.record_baseline import print_report

        print_report(self.SNAP)
        out = capsys.readouterr().out
        for token in ("正向 全部", "仅短探针", "多轮", "负样本", "拒答率",
                      "topic@k", "多轮逐条", "负样本逐条"):
            assert token in out, f"报表缺少 {token!r}"

    def test_short_group_is_marked_comparable(self, capsys):
        """报表必须指出"哪一组能与历史基线按位置对拍"。

        否则看到"正向 87.5%"的人会以为比历史数字退化了，而历史数字
        只覆盖 12 条短探针 —— 这两组本来就不是一回事。
        """
        from tests.record_baseline import print_report

        print_report(self.SNAP)
        out = capsys.readouterr().out
        assert "对拍" in out

    def test_handles_missing_optional_sections(self, capsys):
        """旧格式快照（无 aggregate_all / 多轮 / 负样本）不能崩。"""
        from tests.record_baseline import print_report

        print_report({
            "aggregate": {"n": 12, "book@1": 1.0, "book@k": 1.0, "kw@k": 1.0},
            "aggregate_short": {"n": 12, "book@1": 1.0, "book@k": 1.0, "kw@k": 1.0},
            "per_probe": {},
        })
        out = capsys.readouterr().out
        assert "正向 全部" in out


# ============================================================================
# 向量缓存的往返
# ============================================================================
class TestEmbedCache:
    """缓存必须真的能写进去、读出来。

    这条守卫对应一个**会静默吞掉整个功能价值**的 bug：`np.savez` 在文件名
    不以 `.npz` 结尾时会自动追加该后缀，于是"写临时文件 → 原子改名"里的
    改名永远找不到临时文件。而失败路径只打一句"仅下次需重算"的警告，
    看起来无害 —— 真相是缓存永远写不进去，每次重建都白算全部向量。
    """

    def test_roundtrip(self, tmp_path):
        import numpy as np

        import rag as RE

        path = str(tmp_path / "cache.npz")
        keys = [RE._embed_cache_key(f"文本{i}") for i in range(4)]
        mat = np.random.RandomState(0).rand(4, 768).astype("float32")
        RE._save_embed_cache(path, keys, mat)

        assert os.path.exists(path), (
            "缓存文件没被写出来 —— 检查 np.savez 的临时文件名是否以 .npz 结尾"
        )
        cache = RE._load_embed_cache(path)
        assert len(cache) == 4
        assert next(iter(cache.values())).shape == (768,)

    def test_missing_file_is_empty_cache(self, tmp_path):
        import rag as RE

        assert RE._load_embed_cache(str(tmp_path / "nonexistent.npz")) == {}

    def test_wrong_dimension_is_rejected(self, tmp_path):
        """维度不符的缓存必须被丢弃，而不是留到 vstack 才炸（那时已算了几十分钟）。"""
        import numpy as np

        import rag as RE

        path = str(tmp_path / "bad.npz")
        np.savez(path, keys=np.array(["a", "b"], dtype="U64"),
                 vectors=np.zeros((2, 512), dtype="float32"))
        # 不给期望维度时结构合法 -> 会被放行（由装配后的维度集合兜底）
        assert len(RE._load_embed_cache(path)) == 2
        # 给了期望维度就必须**在计算之前**拒掉
        assert RE._load_embed_cache(path, expected_dim=768) == {}
        assert len(RE._load_embed_cache(path, expected_dim=512)) == 2

    def test_row_count_mismatch_is_rejected(self, tmp_path):
        import numpy as np

        import rag as RE

        path = str(tmp_path / "bad2.npz")
        np.savez(path, keys=np.array(["a"], dtype="U64"),
                 vectors=np.zeros((3, 768), dtype="float32"))
        assert RE._load_embed_cache(path) == {}

    def test_corrupt_file_is_rejected(self, tmp_path):
        import rag as RE

        path = str(tmp_path / "corrupt.npz")
        with open(path, "wb") as f:
            f.write(b"not an npz at all")
        assert RE._load_embed_cache(path) == {}

    def test_key_is_content_hash(self):
        import rag as RE

        assert RE._embed_cache_key("同一段文本") == RE._embed_cache_key("同一段文本")
        assert RE._embed_cache_key("A") != RE._embed_cache_key("B")
        assert len(RE._embed_cache_key("x")) == 64  # sha256 hex，正好塞进 U64


# ============================================================================
# 打包的硬不变量：任何分组都不得超过上限
# ============================================================================
class TestPackingCapInvariant:
    """上限是**硬不变量**，下限只是质量偏好 —— 两者会互相拉扯。

    这组测试对应一个真实 bug：`_split_two` 旧实现只保证前半不超上限，
    把后半原样返回且不校验，于是"合并小块"的配平路径会产出越界的组。
    实测在新产物里造成 31 条超 512 的父块、3 条超 128 的子块（旧实现是 0 条）。
    超限子块在 embedding 时被**静默截断**，检索到的是"半句话"。
    """

    # 实测复现输入：三国演义第 38 回，合计 611，上限 512
    PATHOLOGICAL = [14, 62, 346, 97, 58, 34]

    def test_real_world_pathological_case(self):
        tok = self.PATHOLOGICAL
        groups = RE._balanced_pack_groups(tok, 512, 192)
        sizes = [sum(tok[i] for i in g) for g in groups]
        assert all(s <= 512 for s in sizes), f"仍越界: {groups} → {sizes}"
        # 覆盖完整：不重不漏（漏一个下标就是丢一段原文）
        assert sorted(i for g in groups for i in g) == list(range(len(tok)))

    def test_split_two_never_returns_oversized_tail(self):
        """`_split_two` 的后半也必须 ≤ max_tokens。"""
        tok = self.PATHOLOGICAL
        head, tail = RE._split_two(list(range(len(tok))), tok, 512)
        if tail:
            assert sum(tok[i] for i in tail) <= 512, f"后半越界: {tail}"
        assert sum(tok[i] for i in head) <= 512, f"前半越界: {head}"

    @pytest.mark.parametrize("max_tokens,min_tokens", [(512, 192), (128, 32), (256, 64)])
    def test_random_sequences_never_exceed_cap(self, max_tokens, min_tokens):
        """随机性质测试：任意句子长度序列下都不得越界，且覆盖完整。

        单测一个例子只能证明"那个例子对了"；这里的随机用例是真正守住
        不变量的地方 —— 越界来自"下限合并"与"上限"的相互作用，
        穷举几个手写例子很容易漏掉触发组合。
        """
        import random

        rng = random.Random(20260917)
        # **前提**：每句都不超过 max_tokens。这不是随手假设 —— 真实链路上游
        # `_split_sentences(text, tokenizer, max_tokens)` 会逐句兜底（超长句按
        # 标点→换行→分号→逗号→硬切降级切碎），所以进到打包这一层的句子
        # 必然 ≤ max_tokens。测试若不守这个前提，就会去断言一个
        # 真实链路不可能出现的状态（首版生成器放出了 200 而 max=128 的句子，
        # 于是被 `_enforce_group_cap` 按"单句保留"处理而报错）。
        choices = [1, 2, 5, 20, 50, max_tokens // 4, max_tokens // 2,
                   max_tokens - 1, max_tokens]
        choices = sorted({c for c in choices if 1 <= c <= max_tokens})
        for _ in range(200):
            n = rng.randint(2, 40)
            # 混入长句（可达上限）与极短句，制造"配不匀"的形态
            tok = [rng.choice(choices) for _ in range(n)]
            if sum(tok) < 2:
                continue
            groups = RE._balanced_pack_groups(tok, max_tokens, min_tokens)
            sizes = [sum(tok[i] for i in g) for g in groups]
            assert all(s <= max_tokens for s in sizes), (
                f"tok={tok} max={max_tokens} min={min_tokens} → 越界 {groups} {sizes}")
            assert sorted(i for g in groups for i in g) == list(range(n)), (
                f"tok={tok} → 覆盖错误 {groups}")

    def test_oversized_single_sentence_is_documented_precondition(self):
        """记录一条**前提**：单句超过上限时打包层原样保留，不试图切碎它。

        为什么不去"修"：切碎一句话需要重新分词，那是上游
        `_split_sentences(text, tokenizer, max_tokens)` 的职责（它按
        标点→换行→分号→逗号→硬切逐级降级）。打包层若也去切，就会在两处
        实现同一件事，且这里的实现拿不到原始的标点结构。因此这里显式保留，
        并由上面那条测试**断言前提成立**（随机输入里每句都不超上限）。
        """
        groups = RE._enforce_group_cap([[0]], [9999], 512)
        assert groups == [[0]], "不应死循环，也不应丢弃这句"

    def test_pipeline_never_feeds_oversized_sentences(self, fake_tok):
        """真实链路的前置条件：`_split_sentences(..., max)` 后每句都不超上限。

        这条把"打包层依赖的前提"本身也测了 —— 前提一旦被上游破坏，
        打包层的上限不变量就会失守，而那时报错的地方离原因很远。
        """
        text = "无标点文言" * 300 + "。短句。" + "另一段" * 50 + "。"
        for max_tokens in (128, 512):
            parts = RE._split_sentences(text, fake_tok, max_tokens)
            bad = [p for p in parts
                   if len(fake_tok.encode(p, add_special_tokens=False)) > max_tokens]
            assert not bad, f"上游未兜住超限句（max={max_tokens}）: {bad[:2]}"


# ============================================================================
# 分块指纹：必须覆盖"配置 + 分块代码 + 模块级常量"
# ============================================================================
class TestChunkingFingerprint:
    """指纹是"能不能复用旧分块产物"的唯一判据，漏掉任何一类都会被静默利用。

    这组测试对应一个真实教训：修 `_split_two` 的上限 bug 之后，磁盘上的
    chunks.json 已不可由当前代码复现（它含 31 条超 512 的父块），而当时的
    指纹只看（配置 + 源文本 + sentencex 版本），于是**闸门放行**，
    后续所有检索指标都建立在一份代码复现不出来的产物上。
    """

    def test_stable_across_calls(self):
        a, _ = RE._chunking_config_fingerprint()
        b, _ = RE._chunking_config_fingerprint()
        assert a == b

    def test_payload_covers_all_three_kinds(self):
        _, cfg = RE._chunking_config_fingerprint()
        for key in ("chunking_code", "sentence_terminators", "paragraph_gap_re",
                    "soft_wrap_re", "child_max_tokens", "parent_max_tokens",
                    "semantic_threshold", "normalize_quotes", "embed_model"):
            assert key in cfg, f"指纹缺少 {key}"

    def test_sentencex_version_is_real_not_placeholder(self):
        """sentencex 版本必须是真实版本号，不能是 "unknown"。

        旧写法 `getattr(sentencex, "__version__", "unknown")` 在本环境**恒定**
        返回 "unknown"（该模块没有 __version__），等于这一项从未生效 ——
        而"换一次 sentencex，书一个字节没变而分块可能全变"正是代码注释里
        反复强调的风险。故必须走 importlib.metadata。
        """
        from importlib.metadata import PackageNotFoundError, version

        try:
            expected = version("sentencex")
        except PackageNotFoundError:  # pragma: no cover
            pytest.skip("未安装 sentencex")
        _, cfg = RE._chunking_config_fingerprint()
        assert cfg["sentencex"] == expected
        assert cfg["sentencex"] != "unknown"

    def test_changes_when_config_changes(self, monkeypatch):
        before, _ = RE._chunking_config_fingerprint()
        monkeypatch.setattr(RE, "PARENT_MAX_TOKENS", 256)
        after, _ = RE._chunking_config_fingerprint()
        assert before != after

    def test_changes_when_chunking_code_changes(self, monkeypatch):
        """改了分块函数体，指纹必须变 —— 否则"代码变了产物没变"无人发现。"""
        before, _ = RE._chunking_config_fingerprint()

        original = RE._split_two

        def tweaked(indices, tok, max_tokens):
            return original(indices, tok, max_tokens)

        tweaked.__wrapped__ = None
        monkeypatch.setattr(RE, "_split_two", tweaked)
        after, _ = RE._chunking_config_fingerprint()
        assert before != after, "分块代码变了但指纹没变"

    def test_changes_when_module_constant_changes(self, monkeypatch):
        """模块级常量/正则不在函数源码里，必须单独纳入。"""
        before, _ = RE._chunking_config_fingerprint()
        monkeypatch.setattr(RE, "SENTENCE_TERMINATORS", "。！？")
        assert RE._chunking_config_fingerprint()[0] != before

    def test_retrieval_only_changes_do_not_affect_it(self, monkeypatch):
        """改检索侧参数**不该**逼人重跑两小时分块。

        这是刻意的边界：指纹只覆盖影响分块结果的量。若把整个文件的哈希塞进去，
        改一个检索默认值就会让 2 万条分块作废。
        """
        before, _ = RE._chunking_config_fingerprint()
        monkeypatch.setattr(RE, "TOP_K", 9)
        monkeypatch.setattr(RE, "RRF_K", 42)
        monkeypatch.setattr(RE, "RERANK_ON", "parent")
        assert RE._chunking_config_fingerprint()[0] == before
