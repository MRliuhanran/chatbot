#!/usr/bin/env python3
"""
分块结果 A/B 比较器 —— 用于验证分块改动是否真的"零影响"。

为什么要有这个工具：
  分块流水线（读取归一 → 分句 → 语义定界 → 父子分层 → 前缀装配）里，
  有的改动应该是**逐字节等价**的（例如删除冗余的 \n\n 段落预切），
  有的改动只应该影响**某个字段**（例如不再拼接上下文前缀，只应改 contextual_text）。
  没有这个工具，就只能靠肉眼看 chunks.json（45MB / 2 万余条）来"确认没变",
  而这类需要肉眼确认的地方恰恰是改动最容易出错的点。

比较口径：按**列表位置**逐条比对，而不是按 id 比对。
  因为如果分块边界发生了漂移，chunk_index 派生出的 id（书名_序号）会跟着整体错位，
  按 id 关联会把"第 100 条变成了第 101 条的内容"这种事故显示成"两处不相关的小差异"，
  反而掩盖了真正的问题。按位置比对能直接暴露边界漂移。

用法:
  python compare_chunks.py                       # 默认: chunks_before_simplify.json vs chunks.json
  python compare_chunks.py A.json B.json
  python compare_chunks.py A.json B.json --allow contextual_text
      # --allow F: 只允许字段 F 不同（其余字段必须逐字节相同），适合"只改前缀"这类改动
退出码: 0=通过（相同，或差异全部落在 --allow 字段内）  1=不通过
"""

import argparse
import json
import os
import sys
from collections import Counter

DEFAULT_A = "./cache_v2/chunks_before_simplify.json"
DEFAULT_B = "./cache_v2/chunks.json"

# 逐条记录里所有参与比对的字段（顺序即输出顺序）
FIELDS = [
    "child_text",
    "parent_text",
    "book",
    "chunk_index",
    "total_chunks",
    "contextual_text",
    "id",
]


def load(path):
    if not os.path.exists(path):
        sys.exit(f"❌ 找不到文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"❌ {path} 不是 JSON 列表（chunks.json 应为记录数组）")
    return data


def show(text, limit=90):
    """单行、限长的文本预览。"""
    text = text.replace("\n", "\\n")
    return text[:limit] + ("…" if len(text) > limit else "")


def compare(a, b, allow):
    print("=" * 78)
    print("分块结果比较")
    print("=" * 78)
    print(f"  A: {ARGS.a}")
    print(f"  B: {ARGS.b}")
    print(f"  允许不同的字段: {sorted(allow) if allow else '（无，要求逐字节相同）'}")
    print()

    ok = True

    # ---- 1. 总量 ----
    print(f"[1] 总量: A={len(a)} 条  B={len(b)} 条", end="")
    if len(a) == len(b):
        print("  ✅")
    else:
        print(f"  ❌ 相差 {len(b) - len(a):+d} 条")
        ok = False

    ca, cb = Counter(c.get("book", "?") for c in a), Counter(c.get("book", "?") for c in b)
    for book in sorted(set(ca) | set(cb)):
        na, nb = ca.get(book, 0), cb.get(book, 0)
        flag = "✅" if na == nb else f"❌ 相差 {nb - na:+d}"
        print(f"      {book}: A={na} B={nb} {flag}")
        if na != nb:
            ok = False

    # ---- 2. 字段级差异统计 ----
    n = min(len(a), len(b))
    diff = {f: [] for f in FIELDS}          # 字段 -> [位置...]
    for i in range(n):
        ra, rb = a[i], b[i]
        for f in FIELDS:
            if ra.get(f) != rb.get(f):
                diff[f].append(i)

    print("\n[2] 字段差异（按列表位置逐条比对）")
    for f in FIELDS:
        cnt = len(diff[f])
        if cnt == 0:
            print(f"      {f:16s} ✅ 完全一致")
            continue
        allowed = f in allow
        mark = "🟡 在允许范围内" if allowed else "❌ 不允许的改动"
        if not allowed:
            ok = False
        print(f"      {f:16s} {mark}  差异 {cnt} 条 ({cnt / n * 100:.2f}%)")
        for i in diff[f][:3]:
            print(f"          位置 {i}: A={show(str(a[i].get(f)))}")
            print(f"           {' ' * len(str(i))}      B={show(str(b[i].get(f)))}")

    # ---- 3. 边界漂移的额外提示 ----
    # 若 child_text 有差异，额外指出"差异位置是否沿列表向后传染"，
    # 因为分块边界一旦改动，后续所有 chunk_index 都会顺移。
    if diff["child_text"]:
        pos = diff["child_text"]
        print(f"\n[3] ⚠️  正文出现差异，首个差异位置 = {pos[0]}，"
              f"最后 = {pos[-1]}（共 {len(pos)} 条）")
        print("     若差异位置连续延伸到列表末尾，说明发生了分块边界漂移，")
        print("     而不是局部文本差异 —— 此时不应接受该改动。")

    print("\n" + "=" * 78)
    if ok:
        print("✅ 比较通过：差异全部落在允许范围内")
    else:
        print("❌ 比较不通过：出现了不允许的差异")
    print("=" * 78)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="分块结果 A/B 比较")
    ap.add_argument("a", nargs="?", default=DEFAULT_A, help=f"基线文件（默认 {DEFAULT_A}）")
    ap.add_argument("b", nargs="?", default=DEFAULT_B, help=f"对照文件（默认 {DEFAULT_B}）")
    ap.add_argument("--allow", action="append", default=[],
                    help="允许不同的字段名，可重复指定，例如 --allow contextual_text")
    global ARGS
    ARGS = ap.parse_args()

    unknown = [f for f in ARGS.allow if f not in FIELDS]
    if unknown:
        sys.exit(f"❌ --allow 出现未知字段: {unknown}（可用: {FIELDS}）")

    a, b = load(ARGS.a), load(ARGS.b)
    sys.exit(compare(a, b, set(ARGS.allow)))


if __name__ == "__main__":
    main()
