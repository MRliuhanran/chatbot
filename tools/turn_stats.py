#!/usr/bin/env python3
"""单轮台账摘要（stdlib-only，可选诊断工具）：读 turns.jsonl，聚合出分布。

与 chatbot.py 的 record_turn_stats 成对：那边每轮（done/error）追加一行 JSON，
这边聚合成可读指标。默认台账是关闭的（RAG_TURN_STATS=0）：项目不做基准测量，
需要排查某档性能时临时打开、看完即关。

主要读数：
· prefill 比值 = prompt_eval_count / prompt_tokens_est
  ≈1 → 本轮只有新增段被 prefill（KV 复用发生）；≫1 → 整段重算。
· retrieval_elapsed = 检索全链路耗时；gen_elapsed = 生成侧耗时
  （Stop 中断的轮次不落账）。

用法：
    python tools/turn_stats.py                  # 读 .env 配的 turns.jsonl（默认 artifacts/turns.jsonl）
    python tools/turn_stats.py 路径/turns.jsonl # 显式指定
    python tools/turn_stats.py --json           # 聚合结果以 JSON 输出（便于再加工）
"""

import json
import math
import os
import sys

DEFAULT_FILE = "artifacts/turns.jsonl"
_ENV_FILE = ".env"


def _env_get(key):
    """从项目 .env 读单键（tools 无 dotenv 依赖，与 boot_selftest 同款最小解析）。"""
    try:
        with open(_ENV_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return None


def _load_rows(path):
    """读台账，返回 (合法行列表, 跳过的坏行数)。坏行 = json 解析失败或缺 ts。"""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or "ts" not in row:
                    bad += 1
                    continue
                rows.append(row)
            except ValueError:
                bad += 1
    return rows, bad


def _pct(values, q):
    """最近邻秩分位：idx = ceil(q*N)-1。空列表返回 None。"""
    if not values:
        return None
    xs = sorted(values)
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


def _fmt(v):
    return "-" if v is None else f"{v:g}"


def summarize(rows):
    """聚合成 dict。字段含义见模块 docstring。"""
    status = {}
    done_reason = {}
    over_window = ctx_too_big = 0
    retrieval, gen = [], []
    evals, prefill, ratios = [], [], []
    for r in rows:
        status[r.get("status") or "?"] = status.get(r.get("status") or "?", 0) + 1
        dr = r.get("done_reason")
        if dr:
            done_reason[dr] = done_reason.get(dr, 0) + 1
        if r.get("over_window"):
            over_window += 1
        nc, pt = r.get("num_ctx"), r.get("prompt_tokens_est")
        if isinstance(nc, (int, float)) and nc > 0 and isinstance(pt, (int, float)) and pt > 0.9 * nc:
            ctx_too_big += 1
        for key, bucket in (("retrieval_elapsed", retrieval), ("gen_elapsed", gen),
                            ("eval_count", evals), ("prompt_eval_count", prefill)):
            v = r.get(key)
            if isinstance(v, (int, float)):
                bucket.append(v)
        pe = r.get("prompt_eval_count")
        if isinstance(pe, (int, float)) and pe > 0 and isinstance(pt, (int, float)) and pt > 0:
            ratios.append(pe / pt)
    return {
        "rows": len(rows),
        "statuses": status,
        "done_reason": done_reason,
        "over_window": over_window,
        "prompt_over_90pct_window": ctx_too_big,
        "retrieval_elapsed": {"p50": _pct(retrieval, 0.5), "p95": _pct(retrieval, 0.95)},
        "gen_elapsed": {"p50": _pct(gen, 0.5), "p95": _pct(gen, 0.95)},
        "eval_count": {"p50": _pct(evals, 0.5), "p95": _pct(evals, 0.95)},
        "prompt_eval_count": {"p50": _pct(prefill, 0.5), "p95": _pct(prefill, 0.95)},
        "prefill_ratio": {"p50": _pct(ratios, 0.5), "p95": _pct(ratios, 0.95)},
    }


def _print_human(s):
    print(f"台账行数: {s['rows']}   状态分布: {s['statuses']}")
    if s["done_reason"]:
        print(f"done_reason: {s['done_reason']}   (length = 有回答被 num_predict 截断)")
    print(f"over_window 轮次: {s['over_window']}   占用>90%窗口: {s['prompt_over_90pct_window']}")
    print("耗时（秒）:  p50 / p95")
    for k, label in (("retrieval_elapsed", "检索: "), ("gen_elapsed", "生成: ")):
        d = s[k]
        print(f"  {label:<9} { _fmt(d['p50']) } / { _fmt(d['p95']) }")
    print("token（服务端）:")
    for k, label in (("eval_count", "eval_count: "), ("prompt_eval_count", "prompt_eval_count: ")):
        d = s[k]
        print(f"  {label:<17} { _fmt(d['p50']) } / { _fmt(d['p95']) }")
    d = s["prefill_ratio"]
    print(f"  prefill 比值:  { _fmt(d['p50']) } / { _fmt(d['p95']) }")
    print("读法: 比值≫1 = 每轮整段重算（KV 未复用）；≈1 = 只有增量被 prefill（KV 复用发生）")


def main():
    args = [a for a in sys.argv[1:]]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    path = args[0] if args else (_env_get("RAG_TURN_STATS_FILE") or DEFAULT_FILE)
    try:
        rows, bad = _load_rows(path)
    except FileNotFoundError:
        print(f"还没有台账：{path} 不存在。临时打开 RAG_TURN_STATS=1 跑一轮问答后重试。")
        return 0 if os.path.exists(os.path.dirname(path) or ".") else 1
    except OSError as exc:
        print(f"❌ 台账不可读: {type(exc).__name__}: {exc}")
        return 1
    if bad:
        print(f"⚠️  跳过 {bad} 个坏行（写入半截等问题，不影响其余汇总）")
    if not rows:
        print(f"台账为空：{path}")
        return 0
    s = summarize(rows)
    s["file"] = path
    if as_json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        _print_human(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
