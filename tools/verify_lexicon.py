#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四大名著 RAG —— 别名词典 / 古白话停用词表 校验器。

用法：
    python tools/verify_lexicon.py          # 全部通过 -> 退出码 0；发现问题 -> 1
    python tools/verify_lexicon.py -q       # 只打印结论行（CI 用）

它做四件事：
  1. 加载 data/aliases.txt 与 data/stopwords_classical.txt，校验格式合法性
     （空字段、组内重复、规范名重复出现、同一词面归属多个规范名等）。
  2. 统计每个词在 books/ 四个 .txt 中的真实出现次数（这是"语料接地"的核心：
     词典里任何一条"语料里根本没有的写法"都必须被判定为错误）。
  3. 子串冲突检测：本项目的归一方案是【对称替换 + 最长优先】，那么
     "别名 A 被更长的名字 L 包含" 有两种命运：
       * L 也在词典里  -> 最长优先会先吃掉 L，A 安全（本脚本报为"已遮蔽"）；
       * L 不在词典里  -> A 的这部分出现会被误改（本脚本用汉字扩展窗口
                          启发式把这类竞争串列成"人工复核清单"）。
  4. 停用词危险性检查：表里一旦出现否定词（不/无/未/莫/非/没/别…）或程度词
     （很/太/最/更/极/甚…），"宝玉不读书" 与 "宝玉读书" 会被归一成同一个查询。
     这是硬性错误 -> 退出码非 0。

【必读的语料坑】
    books/水浒传.txt 与 books/红楼梦.txt 的【末尾各有 2 个被截断的 UTF-8 字节】
    （文件是硬截断的，最后一个汉字只写了一半，例如 "…何消得\\xe4\\xb8"）。
    因此绝对不能按默认的 errors="strict" 读：
        open(p, encoding="utf-8")                 # -> UnicodeDecodeError
        open(p, encoding="utf-8", errors="replace")  # -> OK
    本脚本统一用 errors="replace"；替换出的 U+FFFD 只落在文件最末尾，
    不影响任何词频统计（统计前也会剔除 U+FFFD）。
    另外注意：本仓库的语料是【节本】——水浒传只到第 23 回、红楼梦只到第 64 回，
    （回数以 chapter_parse.parse_chapters 的实测结果为准；此处曾误写为 68 回）
    三国演义(120 回)/西游记(100 回) 完整。所以"李逵/花荣/甄宝玉/潘金莲"这些
    写法在语料里出现 0 次，词典里也就一个都不能收。

只用标准库。
"""

from __future__ import annotations

import glob
import os
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOKS_DIR = os.path.join(ROOT, "books")
DATA_DIR = os.path.join(ROOT, "data")
ALIAS_PATH = os.path.join(DATA_DIR, "aliases.txt")
STOP_PATH = os.path.join(DATA_DIR, "stopwords_classical.txt")

# 坑见模块 docstring：两份语料末尾有被截断的字节，必须 errors="replace"。
READ_KWARGS = dict(encoding="utf-8", errors="replace")

# --------------------------------------------------------------------------
# 危险词表（停用词表里出现即报错）
# --------------------------------------------------------------------------
FORBIDDEN_NEGATION = set("不无未莫非没别勿弗毋否亡沒無別罔靡")
FORBIDDEN_DEGREE = set("很太最更极極甚颇頗稍略愈越挺蛮太殊煞")
# 说明：判定方式是"停用词里出现了这些字就报错"，例如 "不在/不曾/莫不/无非"
# 都会因为含否定字被抓出来。宁可误报，不可漏报。

# --------------------------------------------------------------------------
# 用于抑制"竞争窗口"报告噪声的功能字/动词表。
# 只影响报告的噪声过滤，【不参与归一逻辑】。
# --------------------------------------------------------------------------
NOISE_CHARS = set(
    "的了着过们这那一二三四五六七八九十百千万见说道问答叫唤令教使与和在"
    "是有将把被大小老好众位个名字等来去出入上下前后里外中而又也都便就却"
    "只遂因故然若且乃其之于以为所曰我你他她它相再从向往到至及并同连皆亦"
    "复可会要想知听看望走行坐立笑哭怒喜心手头身口眼声气人家儿子们兮乎者"
    "也矣焉哉尔汝卿彼此谁每各另别样般些点儿"
)


def die(msg: str) -> None:
    print("错误：" + msg)


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------
def load_books():
    """返回 [(书名, 文本)]，按文件名排序。"""
    paths = sorted(glob.glob(os.path.join(BOOKS_DIR, "*.txt")))
    if not paths:
        raise SystemExit("找不到语料：%s/*.txt" % BOOKS_DIR)
    books = []
    for p in paths:
        with open(p, **READ_KWARGS) as fh:
            text = fh.read()
        # U+FFFD 是 errors="replace" 造出来的替换符，不参与统计
        text = text.replace("\ufffd", "")
        books.append((os.path.basename(p)[:-4], text))
    return books


def raw_byte_report():
    """报告两处非法 UTF-8 字节，证明 errors='replace' 不是可选项。"""
    lines = []
    for p in sorted(glob.glob(os.path.join(BOOKS_DIR, "*.txt"))):
        name = os.path.basename(p)[:-4]
        with open(p, "rb") as fh:
            data = fh.read()
        try:
            data.decode("utf-8")
            lines.append((name, len(data), "OK"))
        except UnicodeDecodeError as exc:
            lines.append(
                (name, len(data), "非法字节 @ %d-%d" % (exc.start, exc.end - 1))
            )
    return lines


def parse_aliases(path):
    """解析别名表。

    返回 (groups, problems, warnings)
      groups: [{"canonical":str, "aliases":[str], "line":int, "note":str}]
    格式：规范名 <TAB> 别名1 <TAB> 别名2 ...；'#' 之后为注释（整行或行内）。
    """
    groups, problems, warnings = [], [], []
    if not os.path.exists(path):
        problems.append("文件不存在：%s" % path)
        return groups, problems, warnings

    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            body = raw.split("#", 1)[0]
            if not body.strip():
                continue
            if raw.rstrip("\n").endswith("\t"):
                problems.append("第 %d 行：行尾有多余 TAB（空字段）" % lineno)
            fields = body.rstrip("\n").split("\t")
            fields = [f.strip() for f in fields]
            if any(f == "" for f in fields):
                problems.append("第 %d 行：存在空字段 %r" % (lineno, fields))
                fields = [f for f in fields if f]
            if len(fields) < 2:
                problems.append(
                    "第 %d 行：只有规范名没有别名 %r（不产生任何归一收益）"
                    % (lineno, fields)
                )
                continue
            canonical, aliases = fields[0], fields[1:]
            if len(set(aliases)) != len(aliases):
                dup = [a for a, n in Counter(aliases).items() if n > 1]
                problems.append("第 %d 行：组内别名重复 %s" % (lineno, dup))
            if canonical in aliases:
                problems.append("第 %d 行：规范名 %s 又出现在别名里" % (lineno, canonical))
            groups.append(
                {"canonical": canonical, "aliases": aliases, "line": lineno,
                 "note": raw.split("#", 1)[1].strip() if "#" in raw else ""}
            )

    # 跨组校验
    canon_seen = defaultdict(list)
    surface_owner = defaultdict(list)
    for g in groups:
        canon_seen[g["canonical"]].append(g["line"])
        for s in [g["canonical"]] + g["aliases"]:
            surface_owner[s].append(g["canonical"])
    for canon, lns in canon_seen.items():
        if len(lns) > 1:
            problems.append("规范名 %s 出现在多个组（行 %s）" % (canon, lns))
    for surface, owners in surface_owner.items():
        uniq = sorted(set(owners))
        if len(uniq) > 1:
            problems.append(
                "词面 %s 同时归属多个规范名 %s —— 归一目标冲突" % (surface, uniq)
            )
        elif len(owners) > 1:
            problems.append("词面 %s 在同一组内重复出现" % surface)
    return groups, problems, warnings


def parse_stopwords(path):
    """解析停用词表：line.split('#')[0].strip()。返回 (words, problems)。"""
    words, problems = [], []
    if not os.path.exists(path):
        problems.append("文件不存在：%s" % path)
        return words, problems
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            word = raw.split("#", 1)[0].strip()
            if not word:
                continue
            if any(ch.isspace() for ch in word):
                problems.append("第 %d 行：一个词里出现空白字符 %r" % (lineno, word))
            words.append((word, lineno))
    seen = Counter(w for w, _ in words)
    for w, n in seen.items():
        if n > 1:
            problems.append("停用词重复：%s（%d 次）" % (w, n))
    return words, problems


# --------------------------------------------------------------------------
# 统计与子串冲突
# --------------------------------------------------------------------------
def count_in_books(books, terms):
    """{term: {book: n}}，用 str.count（C 速度）。"""
    out = {}
    for t in terms:
        out[t] = {name: text.count(t) for name, text in books}
    return out


def longest_first_scan(books, surfaces):
    """模拟"最长优先、命中即消费"的单次扫描。

    返回 (matched, crossing)：
      matched[s]  = 扫描中被当作【独立命中】吃掉的次数
                    -> total(s) - matched[s] 就是 s 被遮蔽（没机会独立命中）的次数
      crossing[A] = Counter({更长词面 S: 实测次数})
                    这是【左侧重叠】型遮蔽：S 与 A 不构成包含关系，但 S 从更左的
                    位置被优先吃掉后，A 的那次出现一起消失。典型：
                    "南海观音菩萨" 里 南海观音 先命中，观音菩萨 的 2 次没了。
                    这是纯组合关系（"共享一个字"）判断不出来的，必须在真实文本里
                    逐位置实测，所以在这里顺手统计。
    """
    by_first = defaultdict(list)
    for s in surfaces:
        by_first[s[0]].append(s)
    for ch in by_first:
        by_first[ch].sort(key=len, reverse=True)

    matched = Counter()
    crossing = defaultdict(Counter)
    for _name, text in books:
        i, n = 0, len(text)
        while i < n:
            cands = by_first.get(text[i])
            hit = None
            if cands:
                for s in cands:
                    if text.startswith(s, i):
                        hit = s
                        break
            if hit is None:
                i += 1
                continue
            end = i + len(hit)
            # 命中区间 (i, end) 内部的每个位置：若有词面从那里起匹配且
            # 伸出 end 之外，那就是一次真实的左重叠遮蔽。
            for j in range(i + 1, min(end, n)):
                for a in by_first.get(text[j], ()):
                    if j + len(a) > end and text.startswith(a, j):
                        crossing[a][hit] += 1
                        break
            matched[hit] += 1
            i = end
    return matched, crossing


def competitor_windows(books, surface, surface_set, limit=4, min_count=2):
    """启发式：找表面串周围 1~2 个"非功能字"汉字构成的更长的竞争串。

    这些是【不在词典里】但会包含该词面的写法（例：通灵宝玉 之于 宝玉、
    猕猴王 之于 猴王）。它们就是对称替换的误伤来源，列出来供人工复核。
    """
    found = Counter()
    L = len(surface)
    for _name, text in books:
        start = 0
        while True:
            idx = text.find(surface, start)
            if idx < 0:
                break
            start = idx + 1
            for ext in (1, 2):
                s = idx - ext
                if s >= 0:
                    w = text[s:idx + L]
                    extra = w[:ext]
                    if len(w) == ext + L and all(
                        "\u4e00" <= c <= "\u9fff" for c in w
                    ) and all(c not in NOISE_CHARS for c in extra):
                        if w not in surface_set:
                            found[w] += 1
            r = idx + L
            if r < len(text):
                w = text[idx:r + 1]
                if len(w) == L + 1 and "\u4e00" <= text[r] <= "\u9fff" \
                        and text[r] not in NOISE_CHARS and w not in surface_set:
                    found[w] += 1
    return [(w, n) for w, n in found.most_common() if n >= min_count][:limit]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(argv):
    quiet = "-q" in argv or "--quiet" in argv
    # 可选位置参数：别名表路径 停用词表路径（便于对临时文件做负向测试）
    pos = [a for a in argv if not a.startswith("-")]
    alias_path = pos[0] if len(pos) > 0 else ALIAS_PATH
    stop_path = pos[1] if len(pos) > 1 else STOP_PATH

    def out(*a):
        if not quiet:
            print(*a)

    problems, warnings = [], []

    print("=" * 78)
    print("四大名著 RAG 词典校验（data/aliases.txt + data/stopwords_classical.txt）")
    print("=" * 78)

    books = load_books()
    total_chars = sum(len(t) for _, t in books)
    out("语料：%d 个文件，合计 %d 字符" % (len(books), total_chars))
    for name, nbytes, status in raw_byte_report():
        out("    %-8s %9d 字节   UTF-8: %s" % (name, nbytes, status))
    out("    读取方式：encoding='utf-8', errors='replace'"
        "（水浒传/红楼梦末尾各有 2 个截断字节，strict 会抛 UnicodeDecodeError）")

    # ---------------- 1. 别名表 ----------------
    out("")
    out("-" * 78)
    out("[1] data/aliases.txt")
    out("-" * 78)
    groups, ap, aw = parse_aliases(alias_path)
    problems += ap
    warnings += aw
    if not groups:
        die("别名表为空或无法解析")
        print("退出码 1")
        return 1

    surfaces = []
    for g in groups:
        surfaces.append(g["canonical"])
        surfaces.extend(g["aliases"])
    surface_set = set(surfaces)

    counts = count_in_books(books, surfaces)
    matched, crossing = longest_first_scan(books, surfaces)
    out("别名组数：%d；词面总数：%d" % (len(groups), len(surfaces)))
    out("最长优先扫描：命中 %d 次" % sum(matched.values()))

    # 语料接地：每个词面必须真实出现
    missing = [s for s in surfaces if sum(counts[s].values()) == 0]
    for s in missing:
        problems.append(
            "词面 %r 在 books/ 中出现 0 次 —— 违反「规范名与别名都必须真实出现」"
            % s
        )

    out("")
    out("    规范名                次数   别名（次数）")
    out("    " + "-" * 72)
    for g in groups:
        c = sum(counts[g["canonical"]].values())
        al = "  ".join(
            "%s(%d)" % (a, sum(counts[a].values())) for a in g["aliases"]
        )
        out("    %-10s %8d   %s" % (g["canonical"], c, al))

    # ---------------- 2. 子串冲突 ----------------
    out("")
    out("-" * 78)
    out("[2] 子串冲突 / 遮蔽检测（对称替换 + 最长优先）")
    out("-" * 78)
    out("    独立     = 按最长优先扫描，被当作独立词面吃掉的次数")
    out("    包含遮蔽 = 该词面是某个更长条目的子串，那几次归入更长条目的目标")
    out("               （同一目标=安全；不同目标=需确认，如 佛祖 ⊂ 东来佛祖→弥勒）")
    out("    左重叠保护 = 更左的名字抢先命中，把该词面那几次【挡掉】了（保护性）")
    out("               实测例：'赵云长子赵统' 里 赵云 先命中，云长 不会误改成关羽")
    out("")
    # 词面 -> 它所属的规范名（parse_aliases 已保证每个词面只归属一个规范名）
    owner_of = {}
    for g in groups:
        for s in [g["canonical"]] + g["aliases"]:
            owner_of[s] = g["canonical"]

    shadow_rows = []
    for g in groups:
        for a in g["aliases"]:
            tot = sum(counts[a].values())
            ind = matched[a]
            shadow = tot - ind
            if shadow > 0:
                contains = sorted(
                    s for s in surface_set
                    if s != a and a in s and len(s) > len(a)
                )
                # 左侧重叠：实测得到（不是组合推断）
                overlaps = [s for s, _c in crossing[a].most_common()]
                # 遮蔽源归属：同一规范名 -> 安全（只是又被归一回同一个人）；
                # 归属不同规范名 -> 该别名的部分出现属于别人，需人工确认。
                same = all(owner_of[s] == g["canonical"] for s in contains)
                shadow_rows.append(
                    (a, tot, shadow, ind, contains, overlaps, same, g["canonical"])
                )
    if shadow_rows:
        for a, tot, shadow, ind, contains, overlaps, same, canon in shadow_rows:
            flag = "安全(同归一目标)" if same else "注意(归入别人!)"
            cross_n = sum(crossing[a].values())
            contain_n = max(shadow - cross_n, 0)
            csrc = "%d:%s" % (contain_n, "/".join(contains)) if contains else "0:-"
            osrc = ("%d:%s" % (cross_n, "/".join(overlaps))) if overlaps else "0:-"
            out("    %-8s 总%5d 独立%5d 遮蔽%5d | 包含 %-22s | 左重叠保护 %-16s %s"
                % (a, tot, ind, shadow, csrc, osrc, flag))
            if not same:
                warnings.append(
                    "%s 的 %d 次出现被更长的词典条目 %s 先匹配走（分属 %s）——"
                    "最长优先下不会再落到 %s，请确认更长条目的归属无误"
                    % (a, contain_n, "/".join(contains),
                       "/".join(sorted({owner_of[s] for s in contains})),
                       canon)
                )
    else:
        out("    （无：本表不含「别名是另一词面子串」的情况）")

    out("")
    out("    竞争窗口（不在词典、但包含某别名的更长汉字串，>=2 次才列出）：")
    out("    注意：本清单是【启发式人工复核清单】，其中多数是动词/虚词紧邻造成的噪声")
    out("          （如 宝玉忙/云长不/孔明自），只有【本身像名字或专名的串】才是真陷阱。")
    out("          真陷阱示例：龙子龙孙(10) 之于 子龙 —— 已据此剔除 子龙；")
    out("          卧龙冈(8) 之于 卧龙 —— 同指诸葛亮，判定安全并保留。")
    any_comp = False
    for g in groups:
        for a in g["aliases"]:
            wins = competitor_windows(books, a, surface_set)
            if wins:
                any_comp = True
                out("      %-8s -> %s" % (a, "  ".join("%s(%d)" % w for w in wins)))
    if not any_comp:
        out("      （无）")

    # ---------------- 3. 归一自检 ----------------
    out("")
    out("-" * 78)
    out("[3] 归一自检：最长优先 + 规范名恒等映射（调用方加载逻辑的参考实现）")
    out("-" * 78)
    flat = {}
    for g in groups:
        for s in [g["canonical"]] + g["aliases"]:
            flat[s] = g["canonical"]
    first_idx = defaultdict(list)
    for s in flat:
        first_idx[s[0]].append(s)
    for ch in first_idx:
        first_idx[ch].sort(key=len, reverse=True)

    def normalize(text, identity_canonical=True):
        """参照实现：单次左->右扫描，命中即整体替换并跳过已消费字符。"""
        out_chars, i, n = [], 0, len(text)
        while i < n:
            hit = None
            for s in first_idx.get(text[i], ()):  # 已按长度降序
                if not identity_canonical and s in canon_set:
                    continue  # 模拟"忘了把规范名映射到自身"的错误实现
                if text.startswith(s, i):
                    hit = s
                    break
            if hit:
                out_chars.append(flat[hit])
                i += len(hit)
            else:
                out_chars.append(text[i])
                i += 1
        return "".join(out_chars)

    canon_set = {g["canonical"] for g in groups}
    sample = []
    for _name, text in books:
        step = max(1, len(text) // 120000)
        sample.append(text[::step])
    idem_bad, checked = [], 0
    for s in sample:
        once = normalize(s)
        twice = normalize(once)
        checked += len(s)
        if once != twice:
            bad = next(
                (i for i in range(min(len(once), len(twice))) if once[i] != twice[i]),
                min(len(once), len(twice)),
            )
            idem_bad.append((once[max(0, bad - 12):bad + 12],
                             twice[max(0, bad - 12):bad + 12]))
    out("抽样 %d 字符做归一，规范名是否参与匹配：是" % checked)
    if idem_bad:
        for a, b in idem_bad[:3]:
            out("    非幂等样例：%r -> %r" % (a, b))
        problems.append(
            "归一不是幂等的（规范名未按恒等映射参与匹配，会出现 鲁鲁智深 式二次替换）"
        )
    else:
        out("幂等性：通过 —— norm(norm(x)) == norm(x)，不会出现 鲁鲁智深 / 林林黛玉")

    # 反例：故意让规范名不参与匹配，展示它会坏在哪里（仅演示，不影响结论）
    demo = "鲁智深倒拔垂杨柳，金角大王与银角大王，孙悟空拜唐僧。"
    out("    反例演示（若加载方忘记把规范名映射到自身）：")
    out("      正确：%s" % normalize(demo))
    out("      错误：%s" % normalize(demo, identity_canonical=False))
    if normalize(demo) == normalize(demo, identity_canonical=False):
        warnings.append("参考实现的正/反例输出相同，说明该样例没覆盖到子串陷阱")

    # ---- 与**线上实现**对拍 ----
    # 上面那个 normalize 是本文件的参照实现，而真正决定检索的是
    # rag_engine.normalize_aliases。两份实现只要不同步，这里"通过"就毫无意义
    # （参照实现永远自洽）—— 这与"工具脚本各抄一份检索管线"是同一类缺陷。
    # 因此：rag_engine 可用时**必须**逐字对拍；不可用时要明说"没验"。
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import rag_engine as _RE
    except Exception as exc:
        warnings.append(
            "未能 import rag_engine（%s），本次**没有**校验线上归一实现 —— "
            "上面的结论只对这个参照实现成立" % type(exc).__name__)
    else:
        mismatch = []
        for s in sample[:2]:      # 全书抽样，逐字对拍
            got, want = normalize(s), _RE.normalize_aliases(s)
            if got != want:
                i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]),
                         min(len(got), len(want)))
                mismatch.append((got[max(0, i - 12):i + 12], want[max(0, i - 12):i + 12]))
        if mismatch:
            for a, b in mismatch[:2]:
                out("    与线上实现不一致：%r (本文件) vs %r (rag_engine)" % (a, b))
            problems.append(
                "本文件的参照实现与 rag_engine.normalize_aliases 输出不一致 —— "
                "两份实现已分叉，参照实现的自检结论不可信")
        else:
            out("与线上实现对拍：通过 —— rag_engine.normalize_aliases 输出逐字相同")

    # ---------------- 4. 停用词表 ----------------
    out("")
    out("-" * 78)
    out("[4] data/stopwords_classical.txt")
    out("-" * 78)
    swords, sp = parse_stopwords(stop_path)
    problems += sp
    words = [w for w, _ in swords]
    out("停用词数量：%d" % len(words))

    # 危险性检查
    danger = []
    for w in words:
        bad_neg = sorted(set(w) & FORBIDDEN_NEGATION)
        bad_deg = sorted(set(w) & FORBIDDEN_DEGREE)
        if bad_neg or bad_deg:
            danger.append((w, bad_neg, bad_deg))
    if danger:
        for w, bn, bd in danger:
            problems.append(
                "停用词 %r 含禁止字：否定%s 程度%s —— 过滤后会破坏语义"
                "（「宝玉不读书」与「宝玉读书」会变成同一个查询）"
                % (w, bn or "-", bd or "-")
            )
    else:
        out("危险性检查：通过 —— 无否定词（不/无/未/莫/非/没/别…）、"
            "无程度词（很/太/最/更/极/甚…）")

    # 词面冲突：停用词不该是人物名/别名
    name_conflict = sorted(set(words) & surface_set)
    for w in name_conflict:
        problems.append("停用词 %r 同时是人物词典里的词面（会把人物名过滤掉）" % w)
    if not name_conflict:
        out("交叉检查：停用词与人物词典无交集")

    zero_sw = [w for w in words if sum(count_in_books(books, [w])[w].values()) == 0]
    for w in zero_sw:
        warnings.append("停用词 %r 在语料中出现 0 次（留着无害，可删）" % w)

    out("")
    out("    停用词              次数    停用词              次数")
    out("    " + "-" * 60)
    sw_counts = count_in_books(books, words)
    pairs = [(w, sum(sw_counts[w].values())) for w in words]
    half = (len(pairs) + 1) // 2
    for i in range(half):
        left = "%-8s %8d" % pairs[i]
        right = "%-8s %8d" % pairs[i + half] if i + half < len(pairs) else ""
        out("    %s      %s" % (left, right))

    # ---------------- 结论 ----------------
    out("")
    out("=" * 78)
    if problems:
        print("发现问题 %d 处：" % len(problems))
        for p in problems:
            print("  [错误] " + p)
        print("=" * 78)
        print("退出码 1")
        return 1
    if warnings:
        print("提示 %d 处（不阻断）：" % len(warnings))
        for w in warnings:
            print("  [提示] " + w)
    print("全部通过：别名组 %d 组 / 词面 %d 个 / 停用词 %d 个，"
          "均已语料接地，无禁止类停用词。" % (len(groups), len(surfaces), len(words)))
    print("=" * 78)
    print("退出码 0")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
