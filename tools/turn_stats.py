#!/usr/bin/env python3
"""单轮台账摘要（stdlib-only）：读 artifacts/turns.jsonl，聚合成本地排查的读数。

与 chatbot.py 的 record_turn_stats 成对：那边每轮（done/error）追加一行 JSON，
这边聚合成可行动的读数。台账默认关闭（RAG_TURN_STATS=0），排查期临时置 1，
看完即关——本项目不做基准测量，工具只服务排障：

· **零输出轮**（status=done 且 answer_chars=0）＝"只有输入无输出"的直接指纹：
  看 eval_count（服务端算没算）与 done_reason（stop/length）分辨是空回答还是
  中断；prompt_eval_count 顺带回答"prefill 是否真的发生过"。
· 错误分布按前缀归并（HTTP / [Ollama] / 超时 / 连接 / memory / 其他），末次错误全文。
· prefill 比值 = prompt_eval_count / prompt_tokens_est：≈1 → 只有本轮增量被
  prefill（服务端前缀缓存命中）；≫1 → 整段重算。
· 耗时 p50/p95：检索（retrieval_elapsed）与生成侧（gen_elapsed）。
· 首字延迟（ttft_s）与解码速率（tps）：前者含加载+prefill，后者只算 eval_count/eval_duration；
  老台账没有这两个字段时自动跳过（不补 0）。

用法：
    python tools/turn_stats.py                 # 读 .env 的 RAG_TURN_STATS_FILE（默认 artifacts/turns.jsonl）
    python tools/turn_stats.py [FILE]          # 显式指定
    python tools/turn_stats.py --tail 5        # 末 5 行原始台账
    python tools/turn_stats.py --json          # 聚合结果以 JSON 输出

坏行（写入半截等）跳过并计数，不中断汇总。
"""

import json
import math
import os
import sys

DEFAULT_FILE = "artifacts/turns.jsonl"
_ENV_FILE = ".env"


def _env_get(key):
    """从项目 .env 读单键（最小解析；进程环境优先由调用方自行覆盖）。"""
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
    """读台账，返回 (合法行列表, 跳过的坏行数)。坏行 = json 失败或缺 ts。"""
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
    if not values:
        return None
    xs = sorted(values)
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


def _err_bucket(text):
    """错误按前缀归桶（顺序即优先级，与页面错误映射同理）。"""
    t = (text or "").lower()
    if "timeout" in t or "timed out" in t:
        return "timeout"
    if "connection" in t or "connect" in t:
        return "connect"
    if t.startswith("http"):
        return "http"
    if t.startswith("[ollama]"):
        return "ollama"
    if "memory" in t or "killed" in t or "out of memory" in t:
        return "memory"
    return "other"


def summarize(rows):
    """聚合成 dict。"""
    status, done_reason, errors = {}, {}, {}
    zero_out = []
    retrieval, gen, evals, prefill, ratios = [], [], [], [], []
    ttfts, tpss = [], []
    over_window, big_window = [], []
    num_ctxs = set()
    for r in rows:
        st = r.get("status") or "?"
        status[st] = status.get(st, 0) + 1
        dr = r.get("done_reason")
        if dr:
            done_reason[dr] = done_reason.get(dr, 0) + 1
        err = r.get("error")
        if err:
            b = _err_bucket(str(err))
            entry = errors.setdefault(b, {"count": 0, "last": None})
            entry["count"] += 1
            entry["last"] = str(err)
        # 零输出轮 = 用了检索却一个字没回（done 落账，answer_chars 为 0）
        if st == "done" and not r.get("answer_chars"):
            zero_out.append(r)
        for key, bucket in (("retrieval_elapsed", retrieval), ("gen_elapsed", gen),
                            ("eval_count", evals), ("prompt_eval_count", prefill),
                            ("ttft_s", ttfts), ("tps", tpss)):
            v = r.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                bucket.append(v)
        pe, ee = r.get("prompt_eval_count"), r.get("prompt_tokens_est")
        if isinstance(pe, (int, float)) and pe > 0 and isinstance(ee, (int, float)) and ee > 0:
            ratios.append(pe / ee)
        if r.get("over_window"):
            over_window.append(r)
        nc, pt = r.get("num_ctx"), r.get("prompt_tokens_est")
        if isinstance(nc, (int, float)) and nc > 0:
            num_ctxs.add(int(nc))
            if isinstance(pt, (int, float)) and pt > 0.9 * nc:
                big_window.append(r)
    return {
        "rows": len(rows),
        "statuses": status,
        "done_reason": done_reason,
        "zero_output": {
            "count": len(zero_out),
            "rows": [{"ts": r.get("ts"), "eval_count": r.get("eval_count"),
                      "prompt_eval_count": r.get("prompt_eval_count"),
                      "done_reason": r.get("done_reason"),
                      "gen_elapsed": r.get("gen_elapsed"), "prompt_tokens_est": r.get("prompt_tokens_est")}
                     for r in zero_out[-5:]],
        },
        "errors": errors,
        "retrieval_elapsed": {"p50": _pct(retrieval, 0.5), "p95": _pct(retrieval, 0.95)},
        "gen_elapsed": {"p50": _pct(gen, 0.5), "p95": _pct(gen, 0.95)},
        "eval_count": {"p50": _pct(evals, 0.5), "p95": _pct(evals, 0.95)},
        "prompt_eval_count": {"p50": _pct(prefill, 0.5), "p95": _pct(prefill, 0.95)},
        "prefill_ratio": {"p50": _pct(ratios, 0.5), "p95": _pct(ratios, 0.95)},
        "ttft_s": {"p50": _pct(ttfts, 0.5), "p95": _pct(ttfts, 0.95)},
        "tps": {"p50": _pct(tpss, 0.5), "p95": _pct(tpss, 0.95)},
        "over_window": len(over_window),
        "prompt_over_90pct_window": len(big_window),
        "num_ctx_seen": sorted(num_ctxs),
        "first_ts": rows[0].get("ts") if rows else None,
        "last_ts": rows[-1].get("ts") if rows else None,
    }


def _fmt(v):
    return "-" if v is None else f"{v:g}"


def _print_human(s, path):
    span = ""
    if s["first_ts"] and s["last_ts"]:
        span = f"   区间 {s['first_ts']:.0f} → {s['last_ts']:.0f}"
    print(f"台账: {path}")
    print(f"行数: {s['rows']}{span}   状态: {s['statuses']}")
    if s["num_ctx_seen"]:
        print(f"num_ctx 现场: {s['num_ctx_seen']}   over_window: {s['over_window']}"
              f"   占用>90%窗口: {s['prompt_over_90pct_window']}")
    z = s["zero_output"]
    print(f"★ 零输出轮（done 且 0 字，'只有输入无输出'指纹）: {z['count']}")
    for r in z["rows"]:
        print(f"   ts={r['ts']} eval_count={r['eval_count']} prompt_eval_count={r['prompt_eval_count']}"
              f" done_reason={r['done_reason']!r} gen={_fmt(r['gen_elapsed'])}s est={r['prompt_tokens_est']}")
    print("   判读: eval_count=0/None 且 done_reason 出现 → 服务端空回；"
          "gen 耗时 ≈ OLLAMA_TIMEOUT 且无 done → 超时中断（error 行有记录）")
    if s["done_reason"]:
        print(f"done_reason: {s['done_reason']}   (length = 有回答被截断)")
    if s["errors"]:
        print("错误分布:")
        for b, e in sorted(s["errors"].items(), key=lambda kv: -kv[1]["count"]):
            print(f"   {b}: {e['count']} 次   末次: {e['last'][:160]}")
    print("耗时 p50 / p95（秒）:")
    for k, label in (("retrieval_elapsed", "检索: "), ("gen_elapsed", "生成: ")):
        d = s[k]
        print(f"  {label:<9} { _fmt(d['p50']) } / { _fmt(d['p95']) }")
    print("token（服务端）p50 / p95:")
    for k, label in (("eval_count", "eval_count: "), ("prompt_eval_count", "prompt_eval_count: ")):
        d = s[k]
        print(f"  {label:<17} { _fmt(d['p50']) } / { _fmt(d['p95']) }")
    d = s["prefill_ratio"]
    print(f"  prefill 比值:  { _fmt(d['p50']) } / { _fmt(d['p95']) }")
    print("读法: ≈1 = 只有本轮增量被 prefill（前缀缓存命中）；≫1 = 整段重算")
    print("首字 / 吞吐 p50 / p95:")
    for k, label in (("ttft_s", "首字(s): "), ("tps", "tok/s:  ")):
        d = s[k]
        print(f"  {label:<9} { _fmt(d['p50']) } / { _fmt(d['p95']) }")
    print("读法: 首字高而 tps 正常 = 加载/prefill 慢；tps 低 = 算力或换页（内存压力的滞后指标）")


def main():
    args = sys.argv[1:]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    tail = None
    if "--tail" in args:
        i = args.index("--tail")
        try:
            tail = int(args[i + 1])
            del args[i:i + 2]
        except (IndexError, ValueError):
            print("⚠️  --tail 需要跟一个数字，如 --tail 5")
            return 1
    path = args[0] if args else (_env_get("RAG_TURN_STATS_FILE") or DEFAULT_FILE)
    try:
        rows, bad = _load_rows(path)
    except FileNotFoundError:
        print(f"还没有台账：{path} 不存在。RAG_TURN_STATS=1 跑一轮（done/error 均计入）后重试。")
        return 0 if os.path.exists(os.path.dirname(path) or ".") else 1
    except OSError as exc:
        print(f"❌ 台账不可读: {type(exc).__name__}: {exc}")
        return 1
    if bad:
        print(f"⚠️  跳过 {bad} 个坏行（写入半截等，不影响其余汇总）")
    if not rows:
        print(f"台账为空：{path}")
        return 0
    if as_json:
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
        return 0
    if tail:
        for r in rows[-tail:]:
            print(json.dumps(r, ensure_ascii=False))
        return 0
    _print_human(summarize(rows), path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
