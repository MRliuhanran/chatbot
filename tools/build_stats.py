#!/usr/bin/env python3
"""构建台账摘要（stdlib-only）：读 artifacts/builds.jsonl，聚合 process/index/sparse 的离线读数。

与 chatbot.py 的 record_build_stats 成对：stage **成功收尾**时追加一行 JSON（失败不入账，
失败读数没有可比口径，只留在 rag.log），这边聚合成可行动读数。这工具不是基准测量，问的是
"缓存到底省了多少活、时间花在哪一段、这个派生并发度还成立吗"：

· process：复用率（文档分块缓存命中）与 chunks/s 趋势。复用率意外走低 = 切分参数或
  splitter 版本变了导致增量失效，与 stages 的 A 段互为印证。
· index：缓存命中率 hit/(hit+miss) 与 cache_saved —— 回答"这轮重算了多少、有没有整份重写
  npz"；embed_s 与 write_s 的比值定位慢段；写批构造/往返分解校验并发派生（构造被 GIL 串行、
  只有往返能真并发——两者比值偏离派生假设时该调 RAG_INDEX_WRITE_WORKERS）。
· sparse：点数、页并发与墙钟（词表调整频率与重建成本的对照）。
· 各 stage 墙钟 p50/p95：只比成功构建之间，失败不会伪装成"很慢的一次正常构建"。

用法：
    python tools/build_stats.py                 # 读 .env 的 RAG_BUILD_STATS_FILE（默认 artifacts/builds.jsonl）
    python tools/build_stats.py [FILE]          # 显式指定
    python tools/build_stats.py --tail 5        # 末 5 行原始台账
    python tools/build_stats.py --json          # 聚合结果以 JSON 输出

坏行（写入半截等）跳过并计数，不中断汇总；缺 ts/stage 的行按坏行计。
"""

import json
import math
import os
import sys

DEFAULT_FILE = "artifacts/builds.jsonl"
_ENV_FILE = ".env"
STAGES = ("process", "index", "sparse")


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
    """读台账，返回 (合法行列表, 跳过的坏行数)。坏行 = json 失败、缺 ts 或缺 stage。"""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or "ts" not in row or "stage" not in row:
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


def _num_field(rows, key):
    return [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]


def _hit_rates(index_rows):
    rates = []
    for r in index_rows:
        hit, miss = r.get("cache_hit"), r.get("cache_miss")
        if isinstance(hit, (int, float)) and isinstance(miss, (int, float)) and hit + miss > 0:
            rates.append(hit / (hit + miss))
    return rates


def summarize(rows):
    """聚合成 dict（--json 直接输出同一结构）。"""
    buckets = {stage: [] for stage in STAGES}
    unknown = []
    for r in rows:
        stage = r.get("stage")
        (buckets[stage] if stage in buckets else unknown).append(r)
    out = {
        "rows": len(rows),
        "unknown_stage_rows": len(unknown),
        "first_ts": rows[0].get("ts") if rows else None,
        "last_ts": rows[-1].get("ts") if rows else None,
    }
    for stage, bucket in buckets.items():
        if not bucket:
            continue
        latest = bucket[-1]
        entry = {
            "runs": len(bucket),
            "wall_s": {"p50": _pct(_num_field(bucket, "wall_s"), 0.5), "p95": _pct(_num_field(bucket, "wall_s"), 0.95)},
            "latest": latest,
        }
        if stage == "process":
            rates = [r.get("reused") / r.get("docs") for r in bucket
                     if isinstance(r.get("docs"), (int, float)) and r.get("docs") and isinstance(r.get("reused"), (int, float))]
            entry["reused_rate"] = {"p50": _pct(rates, 0.5), "p95": _pct(rates, 0.95),
                                    "latest": rates[-1] if rates else None}
            entry["chunks_per_s"] = {"p50": _pct(_num_field(bucket, "chunks_per_s"), 0.5),
                                     "p95": _pct(_num_field(bucket, "chunks_per_s"), 0.95)}
            entry["counts"] = {"docs": latest.get("docs"), "chunks": latest.get("chunks"),
                               "parents": latest.get("parents")}
        if stage == "index":
            rates = _hit_rates(bucket)
            entry["cache_hit_rate"] = {"p50": _pct(rates, 0.5), "p95": _pct(rates, 0.95),
                                       "latest": rates[-1] if rates else None}
            entry["embed_s"] = {"p50": _pct(_num_field(bucket, "embed_s"), 0.5), "p95": _pct(_num_field(bucket, "embed_s"), 0.95)}
            entry["write_s"] = {"p50": _pct(_num_field(bucket, "write_s"), 0.5), "p95": _pct(_num_field(bucket, "write_s"), 0.95)}
            entry["construct_rtt_s"] = {
                "p50": [_pct(_num_field(bucket, "construct_s"), 0.5), _pct(_num_field(bucket, "rtt_s"), 0.5)],
                "p95": [_pct(_num_field(bucket, "construct_s"), 0.95), _pct(_num_field(bucket, "rtt_s"), 0.95)],
            }
            entry["workers_seen"] = sorted({r.get("write_workers") for r in bucket if isinstance(r.get("write_workers"), int)})
            entry["precision_seen"] = sorted({r.get("precision") for r in bucket if r.get("precision")})
        if stage == "sparse":
            entry["points"] = {"p50": _pct(_num_field(bucket, "points"), 0.5), "p95": _pct(_num_field(bucket, "points"), 0.95)}
            entry["workers_seen"] = sorted({r.get("workers") for r in bucket if isinstance(r.get("workers"), int)})
        out[stage] = entry
    return out


def _fmt(v):
    return "-" if v is None else f"{v:g}"


def _print_human(s, path):
    span = ""
    if s["first_ts"] and s["last_ts"]:
        span = f"   区间 {s['first_ts']:.0f} → {s['last_ts']:.0f}"
    print(f"台账: {path}")
    print(f"行数: {s['rows']}{span}   未知 stage 行: {s['unknown_stage_rows']}")
    for stage, label in (("process", "process 分块"), ("index", "index 建索引"), ("sparse", "sparse 稀疏重建")):
        e = s.get(stage)
        if not e:
            print(f"\n[{label}] （台账里还没有成功行）")
            continue
        w = e.get("wall_s") or {}
        print(f"\n[{label}] 成功 {e['runs']} 次   墙钟 p50/p95: {_fmt(w['p50'])} / {_fmt(w['p95'])}s")
        latest = e.get("latest") or {}
        if stage == "process":
            rr = e.get("reused_rate") or {}
            cps = e.get("chunks_per_s") or {}
            print(f"  复用率 p50: {_fmt(rr['p50'])}（最新 {_fmt(rr['latest'])}）——低=增量失效：切分参数/语料都满血时异常")
            print(f"  chunks/s p50: {_fmt(cps['p50'])}   最新的量级: docs={_fmt(e['counts']['docs'])}"
                  f" chunks={_fmt(e['counts']['chunks'])} parents={_fmt(e['counts']['parents'])}")
        if stage == "index":
            cr = e.get("cache_hit_rate") or {}
            print(f"  稠密缓存命中率 p50: {_fmt(cr['p50'])}（最新 {_fmt(cr['latest'])}）——低=语料/嵌入代码/模型变动触发重算")
            emb, wr = e.get("embed_s") or {}, e.get("write_s") or {}
            print(f"  阶段分解 p50:  嵌入 {_fmt(emb['p50'])}s / 写批 {_fmt(wr['p50'])}s ——比值定位慢段")
            crt = e.get("construct_rtt_s") or {}
            p50c, p50r = (crt.get("p50") or [None, None])[0], (crt.get("p50") or [None, None])[1]
            print(f"  写批分解 p50:  构造 {_fmt(p50c)}s + 往返 {_fmt(p50r)}s ——构造≫往返时加大并发无收益"
                  f"（workers 见过: {e['workers_seen']}，precision: {e['precision_seen']}）")
        if stage == "sparse":
            pts = e.get("points") or {}
            print(f"  更新点数 p50: {_fmt(pts['p50'])}   页并发见过: {e['workers_seen']}")
    print("\n读数原则: 台账只记成功行（失败在 rag.log）；近期一次要比，看 [latest]。")


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
    path = args[0] if args else (_env_get("RAG_BUILD_STATS_FILE") or DEFAULT_FILE)
    try:
        rows, bad = _load_rows(path)
    except FileNotFoundError:
        print(f"还没有台账：{path} 不存在。跑一轮成功收尾的 process/index/sparse 后重试"
              f"（RAG_BUILD_STATS 默认开；关闭时设 1 或 RAG_BUILD_STATS_FILE 重定向）。")
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
