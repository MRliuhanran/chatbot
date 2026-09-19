"""硬切兜底的无损性 —— 需要真实 tokenizer（只读词表，不加载权重）。

为什么单独成文件：这两条**必须用真分词器**才有意义（FakeTokenizer 的 decode
不会插空格，天然测不出问题），因此不能放进 L0。单列一档保持 L0
"无需任何模型"的契约不被破坏。

回归背景：_hard_split_by_tokens 旧实现用 `tokenizer.decode(ids[a:b])` 还原切片，
而 BERT 式分词器的 decode 会在 token 之间插入空格。中文基本一字一 token，
于是无标点的"二尊者即开报"被还原成"二 尊 者 即 开 报"。

实测现行 chunks.json 中该损伤的规模：
    子块 9 条、父块 6 条（全部来自西游记的经文清单与难数清单）
这些块的稠密向量与 BM25 稀疏向量都建立在被改坏的文本上 ——
"看起来只是多了些空格"，实际检索质量已被污染，且肉眼极难察觉。

运行: pytest -m needs_models
"""

import re

import pytest

import rag as RE

pytestmark = pytest.mark.needs_models

# 损伤特征：中文单字之间出现空格
CHAR_SPACING_RE = re.compile(r"[\u4e00-\u9fff] [\u4e00-\u9fff]")


class TestHardSplitLossless:
    def test_no_artificial_spaces(self, real_tokenizer):
        """硬切不得引入分词器自带的空格，且必须能拼回原文。"""
        text = "二尊者即开报现付去唐朝涅槃经四百卷菩萨经三百六十卷虚空藏经二十卷" * 20
        parts = RE._hard_split_by_tokens(text, real_tokenizer, 128)

        assert len(parts) > 1, "该文本应当被切成多片"
        assert "".join(parts) == text, "硬切后拼不回原文 = 有损"
        for p in parts:
            assert not CHAR_SPACING_RE.search(p), (
                f"硬切引入了逐字空格（文本被改坏）: {p[:40]!r}"
            )

    def test_respects_limit(self, real_tokenizer):
        """无损的同时仍必须守住 token 上限。"""
        text = "混" * 2000
        parts = RE._hard_split_by_tokens(text, real_tokenizer, 128)
        assert "".join(parts) == text
        for p in parts:
            n = len(real_tokenizer.encode(p, add_special_tokens=False))
            assert n <= 128, f"硬切后仍有 {n} token 超限"

    def test_short_text_untouched(self, real_tokenizer):
        assert RE._hard_split_by_tokens("短文本。", real_tokenizer, 128) == ["短文本。"]

    def test_lossless_via_enforce_token_limit(self, real_tokenizer):
        """经由 _enforce_token_limit 走完整降级链后，最终结果仍须无损。"""
        text = "混" * 2000  # 无任何标点 → 必然走到硬切
        parts = RE._enforce_token_limit(text, real_tokenizer, 128)
        assert "".join(parts) == text
        assert not any(CHAR_SPACING_RE.search(p) for p in parts)
