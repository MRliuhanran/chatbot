#!/usr/bin/env python3
"""重建索引后的验证编排 —— 一条命令跑完全部"必须重新确认"的事。

## 为什么需要它

分块/索引一变，下面四件事**全部失效**，必须重新确认，而它们分散在不同工具里：

  1. **确定性**：L3 基线用的是零容差判定，前提是"同一查询在不同进程里结果
     逐位相同"。这个前提必须每次都重新验证 —— 一旦它不成立，L3 的任何
     "回归"都可能是噪声，而不是退化。做法就是**连跑两次并 diff**。
  2. **拒答阈值**：`RAG_ABSTAIN_MEAN_HARD/LOW` 是在**旧索引**上校准的
     （误拒 0/24、拒答召回 6/10）。换索引后分数分布会变，阈值必须重新看。
  3. **A/B 结论**：rerank 粒度之类的对比结论同样绑定在具体索引上。
  4. **分类指标**：正向/多轮/负样本三组的新数值。

手工跑这四件事很容易漏掉一两件，而漏掉"确定性"最危险：它会让后续所有
零容差判定变成不可信。故固化成脚本。

## 用法

    python tools/verify_after_rebuild.py            # 全部检查（分钟级）
    python tools/verify_after_rebuild.py --skip-ab  # 跳过 A/B（省时间）
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_engine as RE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 在子进程里跑评测并把结果写到文件 —— 必须有独立的解释器进程，
# 同进程内重复跑证明不了"跨进程确定性"。
_DUMP = r"""
import json, sys
sys.path.insert(0, %(root)r)
import rag_engine as RE
from tests.eval_runner import run_probes, run_multiturn
eng = RE.RAGEngine()
out = {
    "collection": eng.collection_name,
    "chunks": eng.count(),
    "positive": run_probes(eng, top_k=RE.TOP_K),
    "multiturn": run_multiturn(eng, top_k=RE.TOP_K),
}
with open(%(out)r, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
"""


def _fail(msg):
    print(f"  ❌ {msg}")
    return 1


def check_determinism(runs=2):
    """连跑 N 次完整检索，逐条比对分数与召回集合。

    零容差基线的前提。**分数必须逐位相同**：只要有一位不同，L3 的
    "不低于基线"判定就会偶尔假报警，而假报警会训练人忽视报警。
    """
    print(f"\n[1] 确定性（连跑 {runs} 次独立进程并逐条 diff）")
    snaps = []
    for i in range(runs):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        code = _DUMP % {"root": ROOT, "out": path}
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return _fail(f"第 {i+1} 次评测失败: {proc.stderr[-400:]}")
        with open(path, encoding="utf-8") as f:
            snaps.append(json.load(f))
        os.unlink(path)
        print(f"    第 {i+1} 次完成（{time.time() - t0:.0f}s，"
              f"{snaps[-1]['chunks']} 条，集合 {snaps[-1]['collection']}）")

    bad = []
    base = snaps[0]
    for idx, snap in enumerate(snaps[1:], 2):
        for group in ("positive", "multiturn"):
            for q, va in base[group].items():
                vb = snap[group].get(q)
                if vb is None:
                    bad.append(f"[{group}] {q}: 第 {idx} 次缺该查询")
                    continue
                if va.get("top1_score") != vb.get("top1_score"):
                    # 用 repr 而非 float 比较：要的是"逐位相同"，不是"近似"
                    bad.append(f"[{group}] {q}: 首位分数 "
                               f"{va.get('top1_score')!r} vs {vb.get('top1_score')!r}")
                if va.get("books") != vb.get("books"):
                    bad.append(f"[{group}] {q}: 召回书目序列不同")
                if va.get("rerank_tokens") != vb.get("rerank_tokens"):
                    bad.append(f"[{group}] {q}: rerank 输入 token 数不同")
    if bad:
        print(f"    发现 {len(bad)} 处不一致（前 8 条）：")
        for b in bad[:8]:
            print(f"      {b}")
        return _fail("确定性不成立 —— 零容差基线的前提已被破坏，"
                     "请先查明原因再录基线")
    n = sum(len(s["positive"]) for s in snaps)
    print(f"    ✅ {len(snaps)} 次运行、{n} 组结果逐位一致")
    return 0


def calibrate_abstention():
    """在当前索引上重看拒答阈值：混淆矩阵 + 阈值扫描。

    阈值是在**旧索引**上校准的，换索引后分布会变。这里不做"自动选阈值"
    （那会过拟合到这套探针），只把可选阈值下的误拒/漏拒摆出来，由人决定。
    """
    print("\n[2] 拒答阈值校准（对照 RAG_ABSTAIN_MEAN_HARD / MEAN_LOW / MIN_BOOKS）")
    from tests.probes import NEGATIVE_PROBES, PROBES

    eng = RE.RAGEngine()

    def feats(q):
        res = eng.hybrid_search(q, top_k=RE.TOP_K)
        scores = [r.get("rerank_score", 0.0) for r in res]
        sig = RE.confidence_signal(scores, [r.get("book", "") for r in res])
        return sig

    pos = {p.query: feats(p.query) for p in PROBES}
    neg = {p.query: feats(p.query) for p in NEGATIVE_PROBES}

    def rate(group, key):
        vals = [v[key] for v in group.values() if v.get(key) is not None]
        return (min(vals), statistics.median(vals), max(vals)) if vals else (0, 0, 0)

    for name, grp in (("正例", pos), ("负例", neg)):
        lo, mid, hi = rate(grp, "mean")
        blo, bmid, bhi = rate(grp, "n_books")
        print(f"    {name} n={len(grp)}: mean {lo:.2f}~{hi:.2f}（中位 {mid:.2f}）  "
              f"n_books {blo:.0f}~{bhi:.0f}（中位 {bmid:.0f}）")

    tp = sum(1 for v in neg.values() if v["refuse"])
    fp = sum(1 for v in pos.values() if v["refuse"])
    print(f"    当前阈值下：误拒 {fp}/{len(pos)}，拒答召回 {tp}/{len(neg)}")
    missed = [q for q, v in neg.items() if not v["refuse"]]
    if missed:
        print(f"    漏拒（{len(missed)} 条）: {missed}")

    print("    阈值扫描（判据 A: mean < HARD；判据 B: mean < LOW 且 n_books >= 2）")
    print(f"      {'HARD':>6}{'LOW':>6}{'MINB':>6}{'误拒':>7}{'拒答召回':>10}")
    best = None
    for hard in (-4.0, -3.0, -2.0, -1.5, -1.0):
        for low in (0.0, 0.3, 0.5, 0.8, 1.2):
            for minb in (2, 3):
                fp2 = sum(1 for v in pos.values()
                          if v["mean"] < hard
                          or (v["n_books"] >= minb and v["mean"] < low))
                tp2 = sum(1 for v in neg.values()
                          if v["mean"] < hard
                          or (v["n_books"] >= minb and v["mean"] < low))
                flag = ""
                # "误拒为 0"是选阈值的第一原则：拒掉一个能答的问题，
                # 比多答一个答不好的问题更糟（前者用户无从绕过）。
                if fp2 == 0 and (best is None or tp2 > best[3]):
                    best = (hard, low, minb, tp2)
                    flag = "  ← 当前条件下最优（0 误拒）"
                if fp2 == 0 or (hard, low, minb) == (RE.ABSTAIN_MEAN_HARD,
                                                    RE.ABSTAIN_MEAN_LOW,
                                                    RE.ABSTAIN_MIN_BOOKS):
                    print(f"      {hard:>6}{low:>6}{minb:>6}{fp2:>7}{tp2:>10}{flag}")
    if best:
        print(f"    建议（0 误拒前提下的最大召回）: HARD={best[0]} LOW={best[1]} "
              f"MINB={best[2]} → 拒答召回 {best[3]}/{len(neg)}")
        if (best[0], best[1], best[2]) != (RE.ABSTAIN_MEAN_HARD,
                                           RE.ABSTAIN_MEAN_LOW, RE.ABSTAIN_MIN_BOOKS):
            print("    ⚠️  与当前默认值不同 —— 请人工确认后再改，"
                  "并把新阈值与本表一起写进 PROJECT_DOC")
    return 0


def report_metrics():
    """三类指标一次报出（与 L3 基线同一套判定）。"""
    print("\n[3] 分类指标")
    from tests.eval_runner import aggregate, run_multiturn, run_probes
    from tests.probes import NEGATIVE_PROBES, aggregate_multiturn

    eng = RE.RAGEngine()
    pos = run_probes(eng, top_k=RE.TOP_K)
    mt = run_multiturn(eng, top_k=RE.TOP_K)
    # 多轮必须用 probe 自己的多轮聚合器：它比单轮多一个 topic@k（"书对了但内容
    # 与指代对象无关"只有它能抓到。实测"他最后死在哪里"就是 book@1=Y 而 topic@k=N）。
    # 复用单轮的 aggregate() 会让 topic@k 静默变成 nan —— 本工具首版就是这么错的。
    agg, magg = aggregate(pos), aggregate_multiturn(mt)
    print(f"    正向 n={agg['n']}: book@1={agg['book@1']:.1%}  "
          f"book@k={agg['book@k']:.1%}  kw@k={agg['kw@k']:.1%}")
    print(f"    多轮 n={magg['n']}: book@1={magg['book@1']:.1%}  "
          f"book@k={magg['book@k']:.1%}  kw@k={magg['kw@k']:.1%}  "
          f"topic@k={magg.get('topic@k', float('nan')):.1%}")
    n = len(NEGATIVE_PROBES)
    print(f"    负样本 n={n}: 见 [2] 的拒答统计")
    return 0


def main():
    ap = argparse.ArgumentParser(description="重建索引后的验证编排")
    ap.add_argument("--runs", type=int, default=2, help="确定性检查跑几次（默认 2）")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--skip-ab", action="store_true", help="占位：A/B 见 tools/ab_retrieval.py")
    args = ap.parse_args()

    rc = 0
    if not args.skip_determinism:
        rc |= check_determinism(args.runs)
    if not args.skip_calibration:
        rc |= calibrate_abstention()
    rc |= report_metrics()

    print("\n" + "=" * 62)
    print("通过 ✅" if rc == 0 else "存在问题 ❌（见上）")
    print("=" * 62)
    if not args.skip_ab:
        print("A/B 请另行执行：")
        print("  RAG_RERANK_ON=child  python tools/ab_retrieval.py --label child  --json /tmp/a.json")
        print("  RAG_RERANK_ON=parent python tools/ab_retrieval.py --label parent --json /tmp/b.json")
        print("  python tools/ab_retrieval.py --compare /tmp/a.json /tmp/b.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
