# ── lint 基线闸门（2026-11-12 审计 F8 落地）────────────────────────────────────
# 为什么要有它：ruff.toml 显式化后，lint 面受"新发现"与"存量债"双重影响——存量
# 债（chatbot.py 224 条，15 码）不能一刀切清（BLE001×121 是"观测绝不拖垮主链路"
# 哲学的记账位），但**新增**必须当场拦住，否则配置等于没配。
# 语义（与"new findings fail"的渐进门惯例一致）：
#   · 以 (文件, 规则码) 为键比较计数 —— 比"总数"键粗、比"每条 finding"键抗行号漂移；
#   · 当前 > 基线 或 基线没有的键 → **失败**（新发现/恶化，exit 1）；
#   · 当前 < 基线 或 键消失（文件删了/债清了）→ 放行并报"改善"（不许"修复越少越安全"）。
# 基线再生：python tools/lint_gate.py --regen（写 tools/ruff-baseline.json，记录
# ruff 版本与生成命令——再生必须走 commit 说明，不许顺手重生成抹掉新发现）。
# 用法：./start.sh lint（透传）；python tools/lint_gate.py 直接跑亦可。
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "tools", "ruff-baseline.json")
# --no-cache：基线语义要可复现（缓存可能持有旧 finding），禁用；
# --output-format=concise：一行一条 finding，正则可拆 (文件, 行, 码)。
RUFF_CMD = ["ruff", "check", ".", "--no-cache", "--output-format=concise"]
_FINDING_RE = re.compile(r"^(?P<file>.+?):\d+:\d+: (?P<code>[A-Z]+[0-9]+) ")


def _run_ruff():
    """跑 ruff 并返回 (file, code) → count；ruff 缺失/异常时 fail-closed 报错退出。"""
    try:
        # check=False 是刻意的：returncode 语义由本函数自己解读（1=有 finding 正常、
        # 2=配置/调用层错误要拦），交给 check=True 抛 CalledProcessError 反而绕远。
        proc = subprocess.run(RUFF_CMD, cwd=ROOT, capture_output=True, text=True, timeout=600, check=False)
    except FileNotFoundError:
        print("❌ lint_gate：找不到 ruff 命令（pip install ruff==0.16.7 或确认 PATH）")
        sys.exit(2)
    except subprocess.TimeoutExpired:
        print("❌ lint_gate：ruff 超 600s 未返回（异常，先手动 ruff check . 排查）")
        sys.exit(2)
    counts = {}
    for line in (proc.stdout or "").splitlines():
        m = _FINDING_RE.match(line)
        if m:
            counts.setdefault(m.group("file"), Counter())[m.group("code")] += 1
    # ruff 自身失败（配置坏/文件级语法错误）在 concise 输出里也可能零 finding：
    # 用 returncode 兜底区分"有 findings 但 0 处新增"与"ruff 根本没跑成"。
    return counts, proc.returncode


def _regen(counts):
    doc = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ruff": _ruff_version(),
        "command": " ".join(RUFF_CMD),
        # 语义注记：键 = (文件, 规则码)，值 = finding 数。再生是**显式动作**，
        # 必须随 commit 说明（新增了什么/为什么收编进基线）。
        "counts": {f: dict(c) for f, c in sorted(counts.items())},
    }
    tmp = BASELINE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, BASELINE)
    total = sum(sum(c.values()) for c in counts.values())
    files = len(counts)
    print(f"✅ 基线已再生：tools/ruff-baseline.json（{files} 个文件 / {total} 条 finding）")


def _ruff_version():
    # 精确异常型而非裸 Exception：版本探测是"尽力而为"，OSError（无 ruff）与
    # SubprocessError（超时/崩溃）之外没有值得吞的失败面。
    try:
        out = subprocess.run(["ruff", "--version"], capture_output=True, text=True, timeout=30, check=False)
        return (out.stdout or "").strip().split()[-1] if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description="lint 基线闸门：新发现失败，减少放行")
    ap.add_argument("--regen", action="store_true", help="再生基线（显式动作，随 commit 说明）")
    args = ap.parse_args()

    counts, rc = _run_ruff()
    if args.regen:
        _regen(counts)
        return

    if not os.path.exists(BASELINE):
        print("❌ lint_gate：基线缺失（tools/ruff-baseline.json）——先 python tools/lint_gate.py --regen")
        sys.exit(2)
    with open(BASELINE, encoding="utf-8") as fh:
        base = json.load(fh)
    base_counts = {f: Counter(c) for f, c in (base.get("counts") or {}).items()}

    regressions = []
    improved = []
    for f, codes in sorted(counts.items()):
        for code, n in sorted(codes.items()):
            old = base_counts.get(f, {}).get(code)
            if old is None:
                regressions.append(f"  ➕ {f}: {code} ×{n}（基线无此键——新发现）")
            elif n > old:
                regressions.append(f"  ➕ {f}: {code} {old}→{n}（恶化 +{n - old}）")
            elif n < old:
                improved.append(f"  ➖ {f}: {code} {old}→{n}")
    for f in sorted(base_counts):
        if f not in counts:
            improved.append(f"  ➖ {f}: 整文件无 finding（已清空或已删除）")

    cur_total = sum(sum(c.values()) for c in counts.values())
    base_total = sum(sum(c.values()) for c in base_counts.values())
    if regressions:
        print(f"❌ lint 失败：{len(regressions)} 处新发现/恶化（基线 {base_total} → 当前 {cur_total}）")
        for line in regressions:
            print(line)
        print("   处置：修掉新发现；确属有意引入时再生基线并随 commit 说明。")
        sys.exit(1)
    print(f"✅ lint 通过：无新发现（基线 {base_total} → 当前 {cur_total}）")
    if improved:
        print(f"   🎉 改善 {len(improved)} 处（减少即放行，顺手可在下次再生时收敛基线）：")
        for line in improved:
            print(line)
    # ruff 进程码非零 = 有 finding（存量债在基线内，正常）；2 = 配置/调用错误要拦。
    if rc == 2:
        print("❌ lint_gate：ruff 退出码 2（配置/调用层错误）， finding 层面无法背书")
        sys.exit(2)


if __name__ == "__main__":
    main()
