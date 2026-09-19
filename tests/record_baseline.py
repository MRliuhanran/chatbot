#!/usr/bin/env python3
"""录制检索质量黄金基线 —— L3 回归的比对基准。

为什么需要它：`compare_ab.py` 只能做"两个集合谁更好"的相对比较，回答不了
"这次改动有没有让检索变差"。没有存档的基线，每次评测都在跟一个移动的靶子比
（AGENTS.md 里那些 100% / 87.5% 的指标也没有任何机器可读的存档）。

用法（需 Qdrant + 模型，分钟级）:
    python -m tests.record_baseline            # 写入 tests/golden/retrieval_baseline.json
    python -m tests.record_baseline --check    # 只跑一遍并打印，不写文件

录制后: pytest -m slow
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chatbot as RE
from tests.eval_runner import snapshot  # noqa: E402

BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "golden", "retrieval_baseline.json"
)


def print_report(data):
    """把快照渲染成人看的报表（纯函数，可 L0 测试）。

    抽出来的理由：这段逻辑此前内联在 main() 里，只有真的跑完一遍评测
    （需模型 + Qdrant，分钟级）才能验证它没写错 —— 而"报表渲染"恰恰是
    纯粹的字符串拼装，不值得为它付一次全栈评测的代价。
    """
    # 三类指标分开打印，不合成总分。分母不同的比率相加没有意义，
    # 合起来还会掩盖"长问句更差"这类分层事实（见 probes.aggregate_all 的注释）。
    agg = data["aggregate"]
    short = data["aggregate_short"]
    print(f"  正向 全部 n={agg['n']}:  book@1={agg['book@1']:.1%}  "
          f"book@k={agg['book@k']:.1%}  关键词@k={agg['kw@k']:.1%}")
    print(f"  正向 仅短探针 n={short['n']}:  book@1={short['book@1']:.1%}  "
          f"book@k={short['book@k']:.1%}  关键词@k={short['kw@k']:.1%}"
          f"   ← 只有这组能与旧基线按位置对拍")

    overall = data.get("aggregate_all", {})
    if "multiturn" in overall:
        m = overall["multiturn"]
        print(f"  多轮 n={m['n']}:  book@1={m['book@1']:.1%}  book@k={m['book@k']:.1%}  "
              f"kw@k={m['kw@k']:.1%}  topic@k={m.get('topic@k', float('nan')):.1%}"
              f"   (改写={'启用' if data.get('rewrite_enabled') else '关闭'})")
    if "negative" in overall:
        ng = overall["negative"]
        print(f"  负样本 n={ng['n']}:  拒答率={ng['refuse_rate']:.1%}  "
              f"平均书目分散={ng.get('mean_n_books', float('nan')):.2f} 本")
    print()

    for q, v in data["per_probe"].items():
        flag = "✅" if (v["book@1"] and v["kw@k"]) else ("🟡" if v["book@k"] else "❌")
        print(f"  {flag} {q:14s} book@1={'Y' if v['book@1'] else 'N'} "
              f"book@k={'Y' if v['book@k'] else 'N'} kw@k={'Y' if v['kw@k'] else 'N'} "
              f"top1={(v['top1_score'] if v['top1_score'] is not None else float('nan')):.3f}")

    if data.get("per_probe_multiturn"):
        print("\n  多轮逐条:")
        for q, v in data["per_probe_multiturn"].items():
            print(f"    {q:14s} book@1={'Y' if v['book@1'] else 'N'} "
                  f"topic@k={'Y' if v.get('topic@k') else 'N'}")
    if data.get("per_probe_negative"):
        print("\n  负样本逐条:")
        for q, v in data["per_probe_negative"].items():
            print(f"    {'🛑拒答' if v.get('refuse') else '⚠️作答'} {q:22s} "
                  f"top1={(v['top1_score'] if v['top1_score'] is not None else float('nan')):.3f}")


def main():
    ap = argparse.ArgumentParser(description="录制检索质量黄金基线")
    ap.add_argument("--check", action="store_true", help="只跑一遍并打印，不写文件")
    ap.add_argument("--top-k", type=int, default=RE.TOP_K)
    args = ap.parse_args()

    engine = RE.RAGEngine()
    n = engine.count()
    if n == 0:
        sys.exit(f"❌ 集合 {engine.collection_name} 为空，请先运行: python chatbot.py index")
    # 报实际访问的集合名（别名解析后可能是 books_current）：
    # 写 RE.COLLECTION_NAME 会让人以为基线录自 books_v3，而它可能早就没了
    print(f"集合 {engine.collection_name}: {n} 条，top_k={args.top_k}")

    t0 = time.time()
    data = snapshot(engine, top_k=args.top_k)
    data["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"评测完成，耗时 {time.time() - t0:.1f}s\n")

    print_report(data)

    if args.check:
        print("\n(--check：未写入基线文件)")
        return 0

    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n已写入基线: {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
