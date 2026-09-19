#!/usr/bin/env python3
"""
章回体切分 —— 与检索逻辑解耦的纯函数模块（模块顶层只用标准库 re/typing）。

用途：把 books/ 下的四大名著按「回」切开，给「限定某一回检索 / 引用出处」提供
offsets → Chapter 的映射。与 rag_engine 的分块是无重叠的两件事：
  * 分块（chunk）唯一目标是不超 token 上限，边界可以落在任意句子；
  * 切回（本章）唯一目标是**逐字无损**地还原原文，因此只认行首回目。

实测规模（本仓库 books/*.txt，本次实测复核）：

    水浒传  23 回   红楼梦  64 回   三国演义 120 回   西游记 100 回

四个版本的规模互不相同**不是 bug**：水浒传是节本（正文止于第二十三回），
红楼梦是 64 回残抄本。因此本模块**不做**「四大名著各 100/120 回」之类的
硬编码校验，只校验「从 1 开始、连续无空洞」这一条真正的不变量。
水浒传在第一回之前还有一段「楔子」（不含「第X回」，故不匹配回目正则），
按设计它整段落在 preface 里（实测 5743 字符），不会被丢弃。

中文数字写法的坑（实测）：三国演义把第 110 回写作「第一百十回」（十位上的"一"
省略了），而第 101 回写作「第一百一回」。naive 的解析器遇到前者会直接抛错，
把整本书卡住 —— 见 chinese_to_int 的 docstring。

非法字节：books/水浒传.txt 与 books/红楼梦.txt 的文件**末尾各有一个被截断的
3 字节 UTF-8 序列**（实测：水浒传 544768 字节处剩 b'\\xe4\\xb8'、红楼梦 1335294
字节处剩 b'\\xe6\\x9c'），直接 open(f).read() 会抛 UnicodeDecodeError。
调用方必须 `encoding="utf-8", errors="replace"` 读取（见 tests/test_chapters.py
的 _read_book），此时末尾变成一个 U+FFFD，切回逻辑照常工作、不会崩。

逐字无损的口径（**有回目时**没有例外条款）：设 preface 为第一个回目行之前的内容，
则

    preface + "".join(ch.body for ch in chapters) == text   # 恒等于，无例外

成立（四本书实测逐字相等）。原因是最后一回的 end 取 len(text)（而不是它的标题行
之后到下一个回目的某个位置），所以文件末尾的空白/换行也被归入最后一回的 body，
不存在「尾部空白被吃掉」的缺口。
（完全没有回目的文本按接口约定返回 ("", [])，此时该不变量不适用。）
"""

import re
from typing import NamedTuple

__all__ = ["Chapter", "parse_chapters", "chapter_of_offset", "chinese_to_int"]


class Chapter(NamedTuple):
    """一回。字段语义见任务接口约定。"""

    index: int    # 1-based 回序号（权威序号）
    label: str    # 规范化短标签，如 "第一回"；中文数字保留原文，阿拉伯数字转中文
    heading: str  # 完整回目标题行（去首尾空白）
    title: str    # 去掉"第X回"前缀后的标题正文
    body: str     # 该回完整正文，**包含 heading 行本身**，按原文原样
    start: int    # body 在传入 text 中的起始偏移
    end: int      # body 的结束偏移（不含），满足 text[start:end] == body


# 回目行：(行首缩进)(第X回)(分隔符)
#
# 四个细节都是被真实语料逼出来的：
#   1) 缩进用 [^\S\n] 而不是 \s：\s 会跨行吞掉回目前的空行，使 match.start()
#      落到空行上，body 就多出一截不属于本回的前导空白。限定「非换行的空白」
#      后 match.start() 精确落在回目行行首。
#   2) 分隔符同为 [^\S\n]：实测四种书里既有全角空格 U+3000（水浒传/红楼梦），
#      也有半角空格（三国演义/西游记），且西游记的回目行后紧跟正文。
#   3) 数字位兼容中文数字与半角/全角阿拉伯数字（全角 ０-９ 由 \uff10-\uff19 覆盖；
#      此前注释声称兼容全角，字符类里却只有 0-9，那条路径**永远不可达**）。
#      label 一律按 index 归一化成中文数字（见 _int_to_chinese）。
#   4) 分隔符改成**前瞻** `(?=[^\S\n]|$)`：旧写法 `回[^\S\n]` 要求"回"后必须
#      紧跟一个非换行空白，于是"第一回"单独成行（回后就是换行）不匹配。
#      漏掉中间某回会触发连续性检查、响亮报错；而**漏掉末回时它会被并进上一回
#      的 body、漏掉首回时会被并进 preface，parse_chapters 正常返回** ——
#      静默错位正是本模块最想拦住的事故。改成前瞻后行尾也能匹配，
#      且不消费字符（原实现把分隔符吃进匹配、靠 start 定位，改前瞻后行为不变）。
CHAPTER_RE = re.compile(
    r"^([^\S\n]*)第([〇零一二三四五六七八九十百千万两0-9\uff10-\uff19]+)回(?=[^\S\n]|$)",
    re.MULTILINE,
)

# 从 heading（已 strip）里剥掉"第X回"前缀。heading 首尾空白已去，故 ^ 直接可用。
_HEADING_PREFIX_RE = re.compile(
    r"^第[〇零一二三四五六七八九十百千万两0-9\uff10-\uff19]+回[^\S\n]*")

# "宽松回目"审计用：不要求分隔符，用于发现"正文里还藏着一个没被切开的回目"。
# 只用于**审计**，不参与切分 —— 真实语料里正文引用回目是常态
# （红楼梦「第四回中既将薛家母子…」、以及 2120 行那条缺分隔符的重复回目），
# 把它们当真回目切开会制造错位，故这里只在"编号恰好等于下一回"时硬失败。
_LOOSE_CHAPTER_RE = re.compile(
    r"^[^\S\n]*第([〇零一二三四五六七八九十百千万两0-9\uff10-\uff19]+)回", re.MULTILINE)


# ============================================================================
# 中文数字
# ============================================================================
_DIGITS = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}

_CN_DIGITS = "〇一二三四五六七八九"


def _int_to_chinese(n: int) -> str:
    """int → 中文数字（仅用于把 label 归一化，支持 0..9999）。

    规范形：10 → "十"（不是"一十"），11 → "十一"，120 → "一百二十"，
    105 → "一百零五"。本仓库最大回数为 120（三国演义），此范围绰绰有余。
    """
    if n == 0:
        return "〇"
    if n < 0 or n > 9999:
        raise ValueError(f"暂不支持的中文数字范围: {n}")
    units = ["", "十", "百", "千"]
    num = str(n)
    out = ""
    for i, ch in enumerate(num):
        d = int(ch)
        pos = len(num) - 1 - i
        if d == 0:
            # 中间的零只补一个，且后面还有非零数字时才补（100 不写成"一百零"）
            if out and not out.endswith("零") and any(int(c) for c in num[i + 1:]):
                out += "零"
            continue
        if d == 1 and pos == 1 and i == 0:
            out += "十"  # 十/十一 而非 一十/一十一
        else:
            out += _CN_DIGITS[d] + units[pos]
    return out


def chinese_to_int(s: str) -> int:
    """中文数字 → int。无法解析时抛 ValueError（绝不返回猜测值）。

    支持：一…九、十、十一…十九、二十…九十九、一百、一百二十、一百零二、〇/零。
    另外接受纯阿拉伯数字串（"12"），便于新增语料里出现"第12回"。

    两种读法：
      * **按位读**：整串都是数字字符（含 〇/零）时按位拼，如 "二〇二四" → 2024，
        "一〇" → 10。回序号不会长这样，但按位读是中文里唯一说得通的解释。
      * **按权读**：出现十/百/千/万时按权累加，如 "一百二十" → 120。

    严格性：混用两种读法的坏输入必须报错而不是猜。例如 "十二三"（连续两个
    非零数字字符且无单位间隔）在过去那种"来一个字符就覆盖 number"的实现里
    会静默返回 13 —— 这正是「回序号错位却没人发现」的成因，故此处直接
    抛 ValueError。

    相邻单位必须**递减**，这是被真实语料逼出来的规则：三国演义把第 110 回写成
    "第一百十回"（省略了十位上的"一"），实测全书 120 个回目里就有这一条；
    初版实现一律拒绝相邻单位，于是三国演义整本解析失败。因此允许
    "百→十"（递减）而拒绝 "十→十"（不递减，即坏输入）。
    注意"万"是**节**单位、不参与递减校验（否则 `十万`/`二十万` 会被误判成坏输入，
    而 `一万` 恰好通过 —— 同一规则对同类输入给出相反结论）。
    """
    if not isinstance(s, str):
        raise ValueError(f"chinese_to_int 需要 str，收到 {type(s).__name__}")
    s = s.strip()
    if not s:
        raise ValueError("chinese_to_int: 空字符串无法解析")

    # 纯阿拉伯数字（含全角）：交给 int()
    if s.isdigit():
        return int(s)

    # 按位读：整串只由数字字符构成
    if all(ch in _DIGITS for ch in s):
        return int("".join(str(_DIGITS[ch]) for ch in s))

    total = 0        # 已结算的部分（"万"以下已在 section 里）
    section = 0      # 当前小节
    number = 0       # 暂存待乘单位的数字
    prev_kind = ""   # "digit" / "zero" / "unit"，用于挡坏输入
    last_unit = 0    # 上一个单位值，用于校验单位递减
    for ch in s:
        if ch in _DIGITS:
            if prev_kind == "digit":
                raise ValueError(f"chinese_to_int: 无法解析 {s!r}（数字字符连续出现）")
            number = _DIGITS[ch]
            prev_kind = "zero" if number == 0 else "digit"
        elif ch in _UNITS:
            unit = _UNITS[ch]
            if unit == 10000:
                # "万"是**节**单位，必须先结算，且不能参与"相邻单位递减"校验：
                # 否则 `十万`/`二十万` 会被判成坏输入（10 后面跟着 10000 不算递减），
                # 而 `一万` 又恰好通过 —— 同一条规则对同类输入给出相反结论。
                section = (section + number) * unit
                total += section
                section = 0
                number = 0
                prev_kind = "unit"
                last_unit = unit
                continue
            # 相邻单位只允许递减（一百十 = 110，见 docstring）；不递减即坏输入
            if last_unit and unit >= last_unit:
                raise ValueError(f"chinese_to_int: 无法解析 {s!r}（单位字符未递减）")
            section += (number or 1) * unit  # "十" 前无数字时按 1 算
            number = 0
            prev_kind = "unit"
            last_unit = unit
        else:
            raise ValueError(f"chinese_to_int: 无法解析 {s!r}（非法字符 {ch!r}）")
    return total + section + number


# ============================================================================
# 切回
# ============================================================================
def _label_numeral(raw: str, index: int) -> str:
    """label 里的数字部分。

    中文数字**保留原文**，不换成规范形：三国演义把第 101 回写作"第一百一回"、
    第 110 回写作"第一百十回"，若强行规范化成"一百零一/一百一十"，label 就不再是
    heading 的前缀（heading.startswith(label) 会假），调用方想用 label 回切原文
    就得先做一次数字换算。保留原文后这条关系恒成立，而真正的权威序号是 index。
    只有阿拉伯数字（"第12回"这类新语料）才转成中文，因为"规范化"的底线是
    标签形态统一为中文数字。
    """
    return _int_to_chinese(index) if raw.isdigit() else raw


def parse_chapters(text: str) -> tuple[str, list[Chapter]]:
    """按行首回目把 text 切成 (preface, chapters)。

    preface 是第一个回目行之前的内容，**原样保留**（水浒传的「楔子」整段在此）。
    空输入或**完全没有回目**时返回 ("", [])（接口约定如此：无回目即"这不是章回体"，
    由调用方按无章节结构处理；此时逐字无损不变量自然无从谈起）。

    逐字无损不变量（**有回目时**对每本书都成立，无例外条款）：

        preface + "".join(ch.body for ch in chapters) == text
        且对所有 i：text[ch.start:ch.end] == ch.body，ch.body.startswith(ch.heading)

    回序号必须恰好是 1..N 连续无空洞，否则抛 ValueError 并给出可诊断消息
    （包含期望值、实际标题与解析结果）。宁可在切分阶段炸掉，也不能让
    「第五回丢了」这类问题一路静默传到索引里 —— 那会表现为检索结果莫名缺一段，
    排查成本远高于此刻报错。

    **漏尾审计**：连续性检查只能抓住"中间少了一回"。若**最后一回**没被匹配到
    （例如回目行写成"第一百二十回"后直接换行、既无空格也无标题），它会被并进
    上一回的 body，parse_chapters 正常返回 —— 静默错位，正是上面那段想拦的事。
    因此切分完后再扫一遍正文里的"宽松回目"：只有当它解析出的编号**恰好等于
    len(chapters)+1**（即"正文里藏着的正好是下一回"）时才硬失败。
    这个判据刻意收得很窄，因为正文引用回目是常态（红楼梦「第四回中既将薛家母子…」、
    以及那条缺分隔符的重复回目「第三十八回林潇湘魁夺菊花诗…」），放宽即误报。
    """
    matches = list(CHAPTER_RE.finditer(text))
    if not matches:
        return "", []

    preface = text[: matches[0].start()]

    chapters: list[Chapter] = []
    expected = 1
    for i, m in enumerate(matches):
        start = m.start()
        # 最后一回一直取到文末：这样尾部空白/换行也归入最后一回，
        # 拼接时就不会出现"结尾被吃掉"的例外。
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]

        line_end = body.find("\n")
        heading = (body if line_end < 0 else body[:line_end]).strip()

        prefix = _HEADING_PREFIX_RE.match(heading)
        title = heading[prefix.end():].strip() if prefix else heading

        index = chinese_to_int(m.group(2))
        if index != expected:
            raise ValueError(
                f"回序号不连续：期望 {_int_to_chinese(expected)}（index={expected}），"
                f"实际第 {len(chapters) + 1} 个回目是 {heading}（index={index}，"
                f"位于偏移 {start}）"
            )

        chapters.append(
            Chapter(
                index=index,
                label=f"第{_label_numeral(m.group(2), index)}回",
                heading=heading,
                title=title,
                body=body,
                start=start,
                end=end,
            )
        )
        expected += 1

    _audit_missing_tail(text, chapters, matches)
    return preface, chapters


def _audit_missing_tail(text, chapters, matches):
    """审计"最后一回没被识别成回目"（见 parse_chapters 的说明）。

    只在宽松回目的编号**恰好等于** len(chapters)+1 时抛错：那正是"正文里藏着
    一个本该是新一回的开头、却因为没有分隔符而被并进上一回"的签名。
    其它宽松命中一律放过 —— 正文引用回目（"第四回中…"）是常态。
    """
    parsed = {ch.index for ch in chapters}
    expected_next = len(chapters) + 1
    for m in _LOOSE_CHAPTER_RE.finditer(text):
        if any(m.start() == mm.start() for mm in matches):
            continue  # 就是已识别的那个回目行本身
        try:
            num = chinese_to_int(m.group(1))
        except ValueError:
            continue
        if num == expected_next and num not in parsed:
            line_end = text.find("\n", m.start())
            snippet = text[m.start():line_end if line_end > 0 else len(text)][:60]
            raise ValueError(
                f"疑似漏掉末回：正文里出现「{snippet}」，解析为第 {num} 回"
                f"（= 已识别回数 {len(chapters)} + 1），但它没有被当作回目。"
                f"多半是回目行在「回」字后没有分隔符、且直接换行。"
                f"请检查源文本，或显式处理该格式。"
            )


def chapter_of_offset(chapters: list[Chapter], offset: int) -> Chapter | None:
    """返回包含该偏移的 Chapter；越界或空列表返回 None。

    区间语义是左闭右开 [start, end)：offset 恰好等于某回 end 时属于**下一回**
    （因为 end == 下一回 start，同一位置不能同时属于两回）；恰好等于最后一回的
    end（== len(text)）则越界，返回 None。

    **preface 里的偏移同样返回 None** —— 它不属于任何一回（水浒传的楔子 5743
    字符全在这里）。上一版 docstring 写成"对任意 0 <= offset < len(text) 恰好
    命中一回"，与紧随其后的这句自相矛盾；代码一直是对的，承诺是错的。
    调用方若要覆盖 preface，必须自己判断 offset < chapters[0].start。
    """
    if not chapters or offset < 0:
        return None
    # 二分而非线性扫描：chapters 按 start 升序（切分顺序即升序），
    # 且逐回连续，故二分结果必然正确。
    lo, hi = 0, len(chapters) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if chapters[mid].start <= offset:
            lo = mid
        else:
            hi = mid - 1
    cand = chapters[lo]
    # 必须同时校验左端：offset 落在第一回 start 之前（即 preface 里）时，
    # 二分仍会给出 chapters[0]，只查右端就会把前言里的偏移误判成第一回。
    return cand if cand.start <= offset < cand.end else None
