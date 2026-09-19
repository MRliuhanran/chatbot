"""L0：纯函数单测 —— 无需模型 / Qdrant / 语料，秒级完成。

这一层存在的意义：项目此前**所有**验证手段都要求全栈就位（真实 books +
2 个模型 + Docker + 2 万条索引），导致实际执行频率极低，而恰恰是"改了分块
参数却没人跑回归"最容易出事。

之所以能这么便宜：rag_engine 把 torch / transformers / jieba 放在函数内部
惰性导入，模块顶层只依赖 sentencex。分块逻辑因此可以直接测。

每条测试都对应一个**真实发生过的缺陷**（见 test docstring 里的出处），
不是为覆盖率凑数。
"""

import pytest

import rag_engine as RE

# 本模块整体属于 L0：不碰模型 / Qdrant / 语料
pytestmark = pytest.mark.unit


# ============================================================================
# 分句：段落硬边界的守护断言
# ============================================================================
class TestSplitSentences:
    def test_no_cross_paragraph(self):
        """sentencex 把 \\n\\n 当句边界且从不跨段 —— 这是分块正确性的地基。"""
        parts = RE._split_sentences("第一句。第二句！\n\n第三段。")
        assert parts == ["第一句。", "第二句！", "第三段。"]
        assert not any("\n" in p or "\r" in p for p in parts)

    def test_cross_paragraph_raises(self, monkeypatch):
        """守护断言必须真的会响。

        _split_sentences 里的 ValueError 是在守护"sentencex 不跨段"这条
        由**第三方库**提供的性质。换版本 / 换库都可能让它失效，届时必须
        报错而不是静默继续分块。这里注入一个跨段的分句器来验证守卫有效。

        注意判据是**空行**（段落分隔），不是"含任意 \\n"：句内单个 \\n 只是
        按列硬折行的排版（网上 txt 最常见的形式），把它当成"分句器失效"会让
        新增这类书时 `python app.py process` 直接崩掉。见下面的
        test_soft_wrap_is_not_a_paragraph_break。
        """
        monkeypatch.setattr(RE, "segment",
                            lambda lang, s: ["跨段了\n\n第二段。", "尾句。"])
        with pytest.raises(ValueError, match="跨段"):
            RE._split_sentences("任意文本")

    def test_soft_wrap_is_not_a_paragraph_break(self, monkeypatch):
        """句内单个 \\n 是合法排版，不得报错，且应归一成空格。

        回归护栏：旧判据 `"\\n" in p` 会把每一本按 40/76 列硬折行的 txt 判成
        "分句器跨段"，直接抛 ValueError。现有四本典籍恰好没有句内换行
        （实测残留量 0），所以这个缺陷长期被掩盖。
        """
        monkeypatch.setattr(RE, "segment",
                            lambda lang, s: ["话说天下大势，\n分久必合。", "尾句。"])
        parts = RE._split_sentences("任意文本")
        assert parts == ["话说天下大势， 分久必合。", "尾句。"]
        assert not any("\n" in p or "\r" in p for p in parts)

    def test_empty_input(self):
        assert RE._split_sentences("") == []
        assert RE._split_sentences("   \n  ") == []

    def test_quotes_are_atomic(self):
        """引语内部不切分：闭合引号不能被切给下一句（全库错位率 24%~36% 的根因）。"""
        parts = RE._split_sentences("他说：“必获恶报。”角拜问姓名。")
        assert parts[0].endswith("”"), f"闭合引号被切到了下一句: {parts}"
        assert not any(p.startswith(("”", "’", "」", "』")) for p in parts)

    def test_token_limit_applied(self, fake_tok):
        """给定 tokenizer/max_tokens 时，每个片段都必须不超限。"""
        text = "无标点文言" * 100 + "。"
        parts = RE._split_sentences(text, fake_tok, 30)
        assert parts
        for p in parts:
            assert len(fake_tok.encode(p, add_special_tokens=False)) <= 30


# ============================================================================
# fix_quotes：两种语料类型的归一
# ============================================================================
class TestFixQuotes:
    def test_mixed_type_hongloumeng(self):
        """混合型（红楼梦）：开引号是 “，直引号是**闭**引号。

        实测红楼梦 “ 5802 / ” 4202 / " 2426，且 `："` 出现 **0** 次、
        `？"` 915 次、`！"` 402 次 —— 直引号从不作开引号。

        ⚠️ 这条断言曾经是 `他说"你好"然后走了"`（把尾部的直引号判成开引号），
        它的前提已被语料证伪：直引号在红楼梦里一律是**闭**引号。
        """
        assert RE.fix_quotes('他说"你好"然后走了"') == "他说“你好”然后走了“"

    def test_pure_straight_type_is_paired(self):
        """纯直引号型（新书常见）：必须**成对**交替，不能把开引号改成闭引号。

        回归护栏：旧实现无条件套用"前接非空白 → 闭合引号"，
        会把 `他说"你好"` 整句改成 `他说”你好”` —— 开引号变闭引号，
        而且是写进索引的永久损坏。
        """
        assert RE.fix_quotes('他说"你好"然后走了') == "他说“你好”然后走了"
        # 多个引号也必须两两配对
        assert RE.fix_quotes('甲说"一"，乙说"二"') == "甲说“一”，乙说“二”"

    def test_closing_quote_after_sentence_punctuation_stays_closing(self):
        """语料接地：**句末标点后的直引号是闭引号**，不许被改成开引号。

        这里的三段都取自 books/ 的真实形态。实测（红楼梦）：
            `："`   0 次     开引号一律写 “
            `？"` 915 次     闭引号
            `！"` 402 次     闭引号
        旧实现把"直引号左侧紧跟句读"当作**必为开引号**的硬信号，于是这类
        闭引号被改成 “：“ 从 5802 涨到 7896、未配对 “ 从 1600 涨到 3362，
        把这个函数本想修的引号错位做成了一倍（`cache_v2/chunks.json` 里
        能直接看到 `意欲何往？“那僧笑道` 这样的坏样本）。
        """
        assert RE.fix_quotes(
            '只听道人问道：“你携了这蠢物，意欲何往？"那僧笑道：“你放心。”'
        ) == '只听道人问道：“你携了这蠢物，意欲何往？”那僧笑道：“你放心。”'
        assert RE.fix_quotes('道人问道：“你道好否？"石头听了，感谢不尽。') == \
            '道人问道：“你道好否？”石头听了，感谢不尽。'
        assert RE.fix_quotes('那僧还说：“舍我罢，舍我罢！"士隐不耐烦。') == \
            '那僧还说：“舍我罢，舍我罢！”士隐不耐烦。'
        # 单独一句"？\"" 在没有前置开引号时**只能**判成开引号 —— 状态机看不到
        # 上文（真实语料里它前面必有未闭合的 “）。这条断言固化该边界，
        # 免得有人为了"句末标点后的引号一定闭"再引入位置启发式。
        assert RE.fix_quotes('你道好否？"石头听了') == '你道好否？“石头听了'

    def test_noop_when_no_straight_quote(self):
        """无直引号时是纯空操作 —— 快速路径，且不得动弯引号。"""
        text = "他说：“你好。”然后走了"
        assert RE.fix_quotes(text) == text

    def test_jieba_punct_only_filter(self):
        """bm25_tokenize 丢弃纯标点 token（word 分词器不认它们）。"""
        out = RE.bm25_tokenize("武松喝了三碗酒，上了景阳冈打虎。")
        assert out
        assert not any(tok.strip() in RE._PUNCT_ONLY for tok in out.split())
        assert "武松" in out


# ============================================================================
# _enforce_token_limit：超长句逐级兜底
# ============================================================================
class TestEnforceTokenLimit:
    def test_short_text_untouched(self, fake_tok):
        assert RE._enforce_token_limit("短句。", fake_tok, 128) == ["短句。"]

    def test_respects_limit_with_punctuation(self, fake_tok):
        text = "".join(f"第{i}句。" for i in range(200))
        parts = RE._enforce_token_limit(text, fake_tok, 40)
        assert len(parts) > 1
        for p in parts:
            assert len(fake_tok.encode(p, add_special_tokens=False)) <= 40

    def test_terminator_only_at_tail_does_not_recurse_forever(self, fake_tok):
        """终止符恰好落在句尾时必须退出递归。

        出处：_enforce_token_limit 的注释 —— "若唯一终止符恰好落在句尾，
        _split_keep 会切出一个与输入等长的片段，不加判断就会无限自递归"。
        这里同时验证"不会 RecursionError"与"结果仍然不超限"。
        """
        text = "啊" * 100 + "。"
        parts = RE._enforce_token_limit(text, fake_tok, 10)
        for p in parts:
            assert len(fake_tok.encode(p, add_special_tokens=False)) <= 10
        assert "".join(parts) == text

    def test_no_punctuation_hard_split(self, fake_tok):
        """无任何标点可用时按 token 硬切（西游记无标点文言段，单句最长 673 字）。"""
        text = "混" * 250
        parts = RE._enforce_token_limit(text, fake_tok, 64)
        assert len(parts) == 4
        for p in parts:
            assert len(fake_tok.encode(p, add_special_tokens=False)) <= 64
        # 无损：拼回去必须还原原文
        assert "".join(parts) == text

    def test_lossless_for_all_levels(self, fake_tok):
        """逐级降级（标点→换行→分号→逗号）全过程都必须无损。"""
        text = "甲" * 30 + "，" + "乙" * 30 + "；" + "丙" * 30 + "\n" + "丁" * 30 + "。"
        parts = RE._enforce_token_limit(text, fake_tok, 35)
        assert "".join(parts) == text


# ============================================================================
# _split_into_children：子块尺寸与重叠
# ============================================================================
class TestSplitIntoChildren:
    def test_never_exceeds_child_max(self, fake_tok):
        """子块绝不超 CHILD_MAX_TOKENS。

        出处：_split_into_children 的注释 —— 重叠直接叠加曾让 254 个子块超限
        （旧实现 85 个）。超限的子块会在 embedding 时被 tokenizer 静默截断，
        检索到的其实是"半句话"。
        """
        sent = "字" * RE.CHILD_MAX_TOKENS + "。"
        parent = "".join([sent] * 12)
        children = RE._split_into_children(parent, fake_tok)
        assert children
        for c in children:
            n = len(fake_tok.encode(c, add_special_tokens=False))
            assert n <= RE.CHILD_MAX_TOKENS, f"子块 {n} token 超限: {c[:40]!r}"

    def test_overlap_present_between_siblings(self, fake_tok):
        """相邻子块之间应当有重叠（CHUNK_OVERLAP 的实际作用）。"""
        parent = "".join(f"第{i}句内容较长需要凑够一些字符。" for i in range(40))
        children = RE._split_into_children(parent, fake_tok)
        assert len(children) > 1
        # 前一块的尾部句应当出现在后一块里
        tail = children[0].split(" ")[-1]
        assert tail in children[1]

    def test_empty_parent(self, fake_tok):
        assert RE._split_into_children("", fake_tok) == []


# ============================================================================
# _hierarchical_split：父子结构
# ============================================================================
class TestHierarchicalSplit:
    def test_child_contained_in_parent(self, fake_tok, monkeypatch):
        """父子结构的核心契约：子块必须是父块的一部分。"""
        # 绕开语义分块（需要真实模型），直接考察分层组织这一段
        monkeypatch.setattr(RE, "_semantic_split", lambda text, tok: [text])
        text = "".join(f"这是第{i}句测试文本，用来填充长度。" for i in range(60))
        pairs = RE._hierarchical_split(text, fake_tok)
        assert pairs
        for child, parent in pairs:
            assert child in parent, f"子块不在父块中: {child[:30]!r}"

    def test_short_text_child_equals_parent(self, fake_tok, monkeypatch):
        """短文本走"子块=父块"分支。"""
        monkeypatch.setattr(RE, "_semantic_split", lambda text, tok: [text])
        pairs = RE._hierarchical_split("很短的一句话。", fake_tok)
        assert pairs == [("很短的一句话。", "很短的一句话。")]

    def test_parent_respects_limit(self, fake_tok, monkeypatch):
        """长文本必须被切成多个不超 PARENT_MAX_TOKENS 的父块。"""
        monkeypatch.setattr(RE, "_semantic_split", lambda text, tok: [text])
        text = "".join(f"第{i}句。" for i in range(400))
        pairs = RE._hierarchical_split(text, fake_tok)
        for _child, parent in pairs:
            n = len(fake_tok.encode(parent, add_special_tokens=False))
            assert n <= RE.PARENT_MAX_TOKENS, f"父块 {n} token 超限"


# ============================================================================
# sparse_encode / bm25_tokenize：文档侧与查询侧必须同构
# ============================================================================
class TestSparseEncoding:
    def test_encode_is_deterministic(self):
        """同一文本两次编码必须一致（文档侧/查询侧同构的前提）。"""
        a = RE.bm25_tokenize("武松打虎")
        b = RE.bm25_tokenize("武松打虎")
        assert a == b

    def test_empty_text(self):
        assert RE.bm25_tokenize("") == ""
        assert RE.bm25_tokenize(None) == ""
