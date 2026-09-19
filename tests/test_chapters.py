"""章回体切分的测试：合成文本（L0）+ 真实语料不变量。

分两层，理由与项目其余测试一致 —— 资源缺失时 skip 而不是 fail：

  * `pytest.mark.unit`：纯合成文本，不碰 books/，毫秒级，可上门禁；
  * 真实语料层：读 books/ 下四个 .txt，用 os.path.exists 守卫。
    本仓库 pytest.ini 的 marker 表里没有"需要语料"这一档（只有
    needs_chunks / needs_qdrant / needs_models / slow），而本次任务要求
    不改动既有文件，故这一层不加 marker 留在默认集合里，
    靠 os.path.exists → pytest.skip 保证"没语料"不会变成"代码坏了"。

真实语料层守的是一条**合成文本测不出来**的性质：逐字无损。
分块那侧的同类缺陷（边界漂移吃掉一个字）只有在真书上才暴露，
切回同理 —— 合成文本里没有楔子、没有全角空格与半角空格混排、没有截断的非 UTF-8 字节。
"""

import functools
import os

import pytest

import chatbot as CP

BOOKS_DIR = "books"

# 实测回数。刻意写死而不是从文件反推：反推的话，正则漏掉一整回时
# 测试会跟着一起"通过"，等于把断言变成了同义反复。
EXPECTED_BOOKS = {
    "水浒传.txt": 23,
    "红楼梦.txt": 64,
    "三国演义.txt": 120,
    "西游记.txt": 100,
}


def _read_book(name):
    """按 errors="replace" 读取。

    水浒传/红楼梦 文件末尾各有 1 个被截断的 3 字节 UTF-8 序列，
    用默认的 errors="strict" 会抛 UnicodeDecodeError —— 这条路径本身也是被测对象，
    见 test_books_decode_with_replacement。
    """
    path = os.path.join(BOOKS_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"缺少语料 {path}")
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


@functools.lru_cache(maxsize=None)
def _cached_book(name):
    return _read_book(name)


# ============================================================================
# L0：中文数字
# ============================================================================
@pytest.mark.unit
class TestChineseToInt:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("一", 1),
            ("五", 5),
            ("九", 9),
            ("十", 10),           # 十前无数字按 1 算
            ("十一", 11),
            ("十九", 19),
            ("二十", 20),
            ("二十三", 23),
            ("九十九", 99),
            ("一百", 100),
            ("一百零二", 102),
            ("一百一", 101),      # 三国演义第 101 回：十位零省略
            ("一百十", 110),      # 三国演义第 110 回：写成"一百十"而非"一百一十"
            ("一百十一", 111),
            ("一百二十", 120),    # 任务明确要求的一条
            ("〇", 0),
            ("零", 0),
            ("二〇二四", 2024),   # 按位读
        ],
    )
    def test_parse(self, raw, expected):
        assert CP.chinese_to_int(raw) == expected

    def test_roundtrip_with_int_to_chinese(self):
        """label 归一化依赖内部 int→中文；与 chinese_to_int 必须互为逆运算。"""
        for n in [1, 5, 9, 10, 11, 19, 20, 23, 99, 100, 102, 105, 120]:
            assert CP.chinese_to_int(CP._int_to_chinese(n)) == n

    def test_int_to_chinese_canonical(self):
        assert CP._int_to_chinese(10) == "十"      # 不是"一十"
        assert CP._int_to_chinese(20) == "二十"
        assert CP._int_to_chinese(100) == "一百"
        assert CP._int_to_chinese(120) == "一百二十"

    @pytest.mark.parametrize("bad", ["", "   ", "abc", "第一回", "二点五", "一百二十x"])
    def test_unparsable_raises(self, bad):
        with pytest.raises(ValueError):
            CP.chinese_to_int(bad)

    @pytest.mark.parametrize("bad", ["十二三", "十十"])
    def test_bad_input_never_silently_guessed(self, bad):
        """坏输入必须报错。

        "十二三" 在"来一个字符就覆盖 number"的实现里会静默返回 13 —— 这正是
        回序号错位却无人发现的成因，因此这条断言是防回归的关键。
        """
        with pytest.raises(ValueError):
            CP.chinese_to_int(bad)


# ============================================================================
# L0：合成文本切回
# ============================================================================
@pytest.mark.unit
class TestParseChaptersSynthetic:
    def test_three_chapters(self):
        text = (
            "第一回　甲\n正文甲。\n"
            "第二回　乙\n正文乙。\n"
            "第三回　丙\n正文丙。\n"
        )
        preface, chapters = CP.parse_chapters(text)
        assert preface == ""
        assert [c.index for c in chapters] == [1, 2, 3]
        assert [c.label for c in chapters] == ["第一回", "第二回", "第三回"]
        assert [c.heading for c in chapters] == ["第一回　甲", "第二回　乙", "第三回　丙"]
        assert [c.title for c in chapters] == ["甲", "乙", "丙"]
        for c in chapters:
            assert c.body.startswith(c.heading)          # body 必须含标题行本身
            assert text[c.start:c.end] == c.body         # 核心不变量
        assert "".join(c.body for c in chapters) == text  # 无 preface 时逐字无损

    def test_body_is_lossless_with_preface(self):
        text = "书名页\n\n序言若干\n\n第一回　甲\n正文甲。\n第二回　乙\n正文乙。\n"
        preface, chapters = CP.parse_chapters(text)
        assert preface == "书名页\n\n序言若干\n\n"
        assert preface + "".join(c.body for c in chapters) == text
        # 第二回正文里不能混进前一回的尾巴
        assert chapters[1].body == "第二回　乙\n正文乙。\n"

    def test_preface_kept_verbatim(self):
        """水浒传的「楔子」就是这种形态：有实质内容但不含「第X回」。"""
        text = "《水浒传》施耐庵\n\n楔子　张天师祈禳瘟疫　洪太尉误走妖魔\n\n    话说大宋仁宗……\n\n第一回　王教头私走延安府\n正文。\n"
        preface, chapters = CP.parse_chapters(text)
        assert len(chapters) == 1
        assert preface.startswith("《水浒传》施耐庵")
        assert "楔子" in preface and "话说大宋仁宗" in preface  # 楔子整段不被丢弃
        assert preface + chapters[0].body == text

    def test_fullwidth_and_halfwidth_separator(self):
        """全角空格 U+3000（水浒传/红楼梦）与半角空格（三国演义/西游记）都要认。"""
        text = "第一回\u3000全角标题\n甲。\n第二回 半角标题\n乙。\n"
        _, chapters = CP.parse_chapters(text)
        assert [c.heading for c in chapters] == [
            "第一回\u3000全角标题",
            "第二回 半角标题",
        ]
        assert [c.title for c in chapters] == ["全角标题", "半角标题"]

    def test_preceding_blank_lines_not_swallowed(self):
        """回目行前的空行属于上一回/前言，不能被 \s 跨行吞进本回标题。"""
        text = "前言\n\n\n第一回　甲\n正文。\n"
        preface, chapters = CP.parse_chapters(text)
        assert preface == "前言\n\n\n"
        assert chapters[0].heading == "第一回　甲"
        assert chapters[0].start == len("前言\n\n\n")

    def test_empty_and_no_heading(self):
        """无回目即"不是章回体"，按接口约定返回 ("", []) —— 不是把原文丢掉不管。"""
        assert CP.parse_chapters("") == ("", [])
        assert CP.parse_chapters("无回目的短文。\n") == ("", [])
        assert CP.parse_chapters("只有前言\n") == ("", [])
        assert CP.parse_chapters("\n\n   \n") == ("", [])

    def test_last_body_runs_to_end_of_text(self):
        """最后一回取到文末，故尾部空白也归它 —— 逐字无损因此没有例外条款。"""
        text = "第一回　甲\n正文。\n\n   \n"
        _, chapters = CP.parse_chapters(text)
        assert chapters[0].end == len(text)
        assert chapters[0].body.endswith("\n\n   \n")

    def test_gap_raises_value_error(self):
        text = "第一回　甲\n甲。\n第三回　丙\n丙。\n"
        with pytest.raises(ValueError, match="不连续"):
            CP.parse_chapters(text)

    def test_not_starting_at_one_raises(self):
        with pytest.raises(ValueError, match="不连续"):
            CP.parse_chapters("第二回　乙\n乙。\n第三回　丙\n丙。\n")

    def test_duplicate_raises(self):
        with pytest.raises(ValueError, match="不连续"):
            CP.parse_chapters("第一回　甲\n甲。\n第一回　甲二\n乙。\n")

    def test_error_message_is_diagnosable(self):
        """报错消息要能直接定位：期望值 + 实际标题。"""
        with pytest.raises(ValueError) as ei:
            CP.parse_chapters("第一回　甲\n甲。\n第五回　戊\n戊。\n")
        msg = str(ei.value)
        assert "二" in msg and "第五回　戊" in msg


@pytest.mark.unit
class TestChapterOfOffset:
    def setup_method(self):
        self.text = "第一回　甲\n甲。\n第二回　乙\n乙。\n第三回　丙\n丙。\n"
        self.preface, self.chapters = CP.parse_chapters(self.text)

    def test_offset_inside_body(self):
        c0, c1 = self.chapters[0], self.chapters[1]
        assert CP.chapter_of_offset(self.chapters, c0.start + 1) is c0
        assert CP.chapter_of_offset(self.chapters, c1.start + 3) is c1

    def test_offset_exactly_at_start(self):
        for c in self.chapters:
            assert CP.chapter_of_offset(self.chapters, c.start) is c

    def test_offset_exactly_at_end_is_exclusive(self):
        """end 是开区间端点：恰好落在 end 属于下一回，最后一回的 end 则越界。"""
        c0, c1, c2 = self.chapters
        assert c0.end == c1.start
        assert CP.chapter_of_offset(self.chapters, c0.end) is c1
        assert CP.chapter_of_offset(self.chapters, c1.end) is c2
        assert CP.chapter_of_offset(self.chapters, c2.end) is None
        assert c2.end == len(self.text)

    def test_out_of_range(self):
        assert CP.chapter_of_offset(self.chapters, -1) is None
        assert CP.chapter_of_offset(self.chapters, len(self.text)) is None
        assert CP.chapter_of_offset(self.chapters, len(self.text) + 999) is None

    def test_empty_list(self):
        assert CP.chapter_of_offset([], 0) is None
        assert CP.chapter_of_offset([], 42) is None

    def test_preface_offsets_are_none(self):
        text = "序言\n\n第一回　甲\n甲。\n"
        preface, chapters = CP.parse_chapters(text)
        assert preface
        assert CP.chapter_of_offset(chapters, 0) is None   # 落在 preface 里

    def test_every_offset_in_text_hits_exactly_one_chapter(self):
        """全文每个偏移恰好命中一回（preface 非空时除外）—— 引用出处的正确性地基。"""
        text = "第一回　甲\n甲。\n第二回　乙\n乙。\n第三回　丙\n丙。\n"
        _, chapters = CP.parse_chapters(text)
        for off in range(len(text)):
            hit = CP.chapter_of_offset(chapters, off)
            assert hit is not None, f"偏移 {off} 没命中任何一回"
            assert hit.start <= off < hit.end


# ============================================================================
# 真实语料
# ============================================================================
class TestRealBooks:
    @pytest.mark.parametrize("name,expected", sorted(EXPECTED_BOOKS.items()))
    def test_chapter_count(self, name, expected):
        """各版本规模互不相同是事实而非 bug：水浒传是节本(23)、红楼梦是 64 回残抄。"""
        _, chapters = CP.parse_chapters(_cached_book(name))
        assert len(chapters) == expected, (
            f"{name} 实测 {len(chapters)} 回，期望 {expected} 回"
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_BOOKS))
    def test_index_continuous_from_one(self, name):
        _, chapters = CP.parse_chapters(_cached_book(name))
        assert [c.index for c in chapters] == list(
            range(1, EXPECTED_BOOKS[name] + 1)
        )
        assert [c.label for c in chapters][:3] == ["第一回", "第二回", "第三回"]
        assert [c.label for c in chapters][-1] == (
            "第" + CP._int_to_chinese(EXPECTED_BOOKS[name]) + "回"
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_BOOKS))
    def test_slice_invariant(self, name):
        """核心不变量：每一回 text[start:end] == body，body 以标题行开头。"""
        text = _cached_book(name)
        _, chapters = CP.parse_chapters(text)
        for c in chapters:
            assert text[c.start:c.end] == c.body, f"{name} 第{c.index}回切片不等"
            assert c.body.startswith(c.heading), f"{name} 第{c.index}回正文不含标题行"
            assert c.heading.startswith(c.label), f"{name} 第{c.index}回标题异常"
            assert c.start < c.end <= len(text)

    @pytest.mark.parametrize("name", sorted(EXPECTED_BOOKS))
    def test_lossless_concatenation(self, name):
        """preface + 所有 body 逐字节等于原文 —— 不丢字、不重叠、不重复。"""
        text = _cached_book(name)
        preface, chapters = CP.parse_chapters(text)
        assert preface + "".join(c.body for c in chapters) == text
        # 首尾相接：chapters[i].end == chapters[i+1].start，最后一回止于文末
        for a, b in zip(chapters, chapters[1:]):
            assert a.end == b.start
        assert chapters[-1].end == len(text)

    @pytest.mark.parametrize("name", sorted(EXPECTED_BOOKS))
    def test_chapter_of_offset_covers_whole_text(self, name):
        """真实全文里每个偏移都命中一回，且命中区间确实包含该偏移。"""
        text = _cached_book(name)
        preface, chapters = CP.parse_chapters(text)
        assert preface and CP.chapter_of_offset(chapters, 0) is None  # 正文前是书名行
        for off in range(chapters[0].start, len(text), 97):  # 抽样步进，控制耗时
            hit = CP.chapter_of_offset(chapters, off)
            assert hit is not None and hit.start <= off < hit.end
        assert CP.chapter_of_offset(chapters, len(text)) is None

    @pytest.mark.parametrize("name", ["水浒传.txt", "红楼梦.txt"])
    def test_books_decode_with_replacement(self, name):
        """水浒传/红楼梦 末尾各有 1 个被截断的 3 字节 UTF-8 序列。

        strict 解码会抛 UnicodeDecodeError；必须 errors="replace" 读到末尾那个
        U+FFFD，且切回照常工作（末回 body 同样含它，故逐字无损仍成立）。
        """
        path = os.path.join(BOOKS_DIR, name)
        if not os.path.exists(path):
            pytest.skip(f"缺少语料 {path}")
        with open(path, "rb") as f:
            raw = f.read()
        with pytest.raises(UnicodeDecodeError):
            raw.decode("utf-8")  # 证明"非法字节"这一前提为真，不是传说
        text = raw.decode("utf-8", errors="replace")
        assert "\ufffd" in text
        preface, chapters = CP.parse_chapters(text)
        assert preface + "".join(c.body for c in chapters) == text

    def test_shuihu_preface_holds_the_prologue(self):
        """水浒传第一回之前有「楔子」段：它必须整段落在 preface，不被丢弃。"""
        text = _cached_book("水浒传.txt")
        preface, chapters = CP.parse_chapters(text)
        assert preface.strip()
        assert "楔子" in preface
        assert chapters[0].heading.startswith("第一回")
        assert preface.startswith("《水浒传》")
