#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化对拍 · 判定端（parity）。

★ 判定阈值登记区（测量前写死，测量后不得改动）★
--------------------------------------------------
    THRESHOLDS = {
        "embed_cos_mean_min":  0.999,   # 嵌入保真：512 条真实块余弦均值
        "embed_cos_p1_min":    0.995,   # 嵌入保真：余弦 P1（最差 1% 的下界）
        "rerank_spearman_min": 0.99,    # 重排保真：>=100 组候选集内 Spearman 均值
        "rerank_top1_min":     0.95,    # 重排保真：top-1 一致率
        "speedup_min":         1.2,     # 收益：同批大小吞吐提升
        "rss_drop_min":        0.50,    # 收益：常驻内存下降比例
    }
    三项（嵌入保真 / 重排保真 / 收益）全过 → 采纳；任一不过 → 拒绝并给出失败数字。

数据来源：tools/quant_bench.py 落到 artifacts/quant/raw/*.json|*.npz 的原始测量。
本文件只做统计与判定，不产生任何测量。

用法：
    python tools/quant_parity.py --check-thresholds   # 打印登记阈值（登记时间戳）
    python tools/quant_parity.py --memo               # 打印决策备忘（markdown）
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "quant"
RAW = OUT / "raw"

# ─────────────── 阈值登记（测量前写死；改这里 = 改判定标准，必须重新登记） ───────────────
THRESHOLDS = {
    "embed_cos_mean_min": 0.999,
    "embed_cos_p1_min": 0.995,
    "rerank_spearman_min": 0.99,
    "rerank_top1_min": 0.95,
    "speedup_min": 1.2,
    "rss_drop_min": 0.50,
}
THRESHOLDS_REGISTERED_AT = "2026-10-17T00:05+08:00（首次提交，先于任何测量）"

# ─────────────── 共驻留算术的输入（来自任务书的背景算术，非本次测量） ───────────────
HW_MEM_MB = 8071          # 容器可用内存（HW_MEM_MB）
APP_BASELINE_MB = 2000    # 应用基线（python/streamlit/torch/句柄）
QDRANT_MB = 120           # qdrant 常驻
LLM_MB = 4350             # LLM = 权重 3.4GB + KV q8_0 1.06GB（任务书给定）
RETRIEVAL_BUDGET_MB = HW_MEM_MB - APP_BASELINE_MB - QDRANT_MB - LLM_MB

# 变体显示名 & 说明
PRETTY = {
    "fp32": "fp32 (CPU 现状)",
    "fp16": "fp16 (MPS 权重)",
    "fp16ac": "fp16 autocast (MPS 现状)",
    "int8": "int8 动态量化 (仅 Linear)",
    "int8e": "int8 动态量化 (Linear+Embedding)",
    "onnx_fp32": "ONNX fp32 (管线校验)",
    "onnx_int8": "ONNX int8 动态量化",
}
QUANT_VARIANTS = ["int8", "int8e", "onnx_int8"]


# ─────────────── 读取原始测量 ───────────────

def _raw(task: str, variant: str, phase: str):
    p = RAW / f"{task}__{variant}__{phase}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _mat(task: str, variant: str, phase: str):
    p = RAW / f"{task}__{variant}__{phase}.npz"
    if not p.exists():
        return None
    return np.load(p)["mat"]


# ─────────────── 保真统计 ───────────────

def cosine_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """逐行余弦（行已 L2 归一，仍显式归一以免疫实现差异）。"""
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    c = np.sum(a * b, axis=1)
    return {
        "n": int(c.size),
        "mean": float(np.mean(c)),
        "p1": float(np.percentile(c, 1)),
        "min": float(np.min(c)),
        "max_abs_delta": float(np.max(np.abs(a - b))),
        "p50": float(np.percentile(c, 50)),
    }


def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩（处理并列），不依赖 scipy。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman_rows(base: np.ndarray, other: np.ndarray) -> list:
    out = []
    for i in range(base.shape[0]):
        rb, ro = _rankdata(base[i]), _rankdata(other[i])
        rb = rb - rb.mean()
        ro = ro - ro.mean()
        den = np.sqrt(np.sum(rb ** 2) * np.sum(ro ** 2))
        out.append(float(np.sum(rb * ro) / den) if den > 0 else 1.0)
    return out


def rerank_stats(base: np.ndarray, other: np.ndarray) -> dict:
    sp = np.asarray(spearman_rows(base, other))
    top1 = float(np.mean(np.argmax(base, axis=1) == np.argmax(other, axis=1)))
    return {
        "groups": int(base.shape[0]),
        "spearman_mean": float(sp.mean()),
        "spearman_min": float(sp.min()),
        "spearman_p5": float(np.percentile(sp, 5)),
        "top1_agree": top1,
        "score_max_abs_delta": float(np.max(np.abs(base - other))),
    }


def collect() -> dict:
    d = {"embed": {}, "rerank": {}, "onnx_prepare": None}
    p = OUT / "onnx_prepare.json"
    if p.exists():
        d["onnx_prepare"] = json.loads(p.read_text(encoding="utf-8"))

    base_e = _mat("embed", "fp32", "fidelity")
    base_r = _mat("rerank", "fp32", "fidelity")
    for task, base in (("embed", base_e), ("rerank", base_r)):
        for variant in PRETTY:
            m = _mat(task, variant, "fidelity")
            if m is None:
                continue
            log = _raw(task, variant, "fidelity") or {}
            rec = {"log": log}
            if base is not None and variant != "fp32":
                rec["vs_fp32"] = cosine_stats(base, m) if task == "embed" else rerank_stats(base, m)
            d[task][variant] = rec
    # MPS 侧的量化对照：int8(CPU) vs fp16(MPS 现状)
    for task, mps_variant in (("embed", "fp16ac"), ("rerank", "fp16")):
        ref = _mat(task, mps_variant, "fidelity")
        if ref is None:
            continue
        for variant in QUANT_VARIANTS:
            m = _mat(task, variant, "fidelity")
            if m is None or variant not in d[task]:
                continue
            key = "vs_" + mps_variant
            d[task][variant][key] = (cosine_stats(ref, m) if task == "embed" else rerank_stats(ref, m))
    return d


# ─────────────── 表 1/2：内存与吞吐 ───────────────

def mem_rows(data: dict, task: str) -> list:
    rows = []
    for variant, rec in data[task].items():
        log = rec.get("log") or {}
        if not log:
            continue
        resident = log.get("rss_load_mb", float("nan")) - log.get("rss_before_load_mb", 0)
        mb = log.get("model_bytes", -1)
        rows.append({
            "variant": variant,
            "model_bytes_gb": (mb / 2 ** 30) if mb and mb > 0 else None,
            "resident_mb": resident,
            "peak_mb": log.get("peak_rss_mb"),
            "t_load_s": log.get("t_load_s"),
            "t_quant_s": log.get("t_quant_s"),
        })
    return rows


def speed_rows(data: dict, task: str) -> dict:
    out = {}
    for variant in data[task]:
        fid = _raw(task, variant, "fidelity")
        thr = _raw(task, variant, "throughput")
        if not fid:
            continue
        out[variant] = {
            "fid_threads": fid.get("threads"), "fid_units_per_s": fid.get("units_per_s"),
            "thr_threads": (thr or {}).get("threads"), "thr_units_per_s": (thr or {}).get("units_per_s"),
            "device": fid.get("device_actual", "cpu") or "cpu",
        }
    return out


# ─────────────── 判定 ───────────────

def _resident(task, variant):
    lg = _raw(task, variant, "fidelity") or {}
    if not lg:
        return None
    return lg.get("rss_load_mb", 0) - lg.get("rss_before_load_mb", 0)


def judge(data: dict) -> dict:
    T = THRESHOLDS
    verdicts = {}
    for variant in QUANT_VARIANTS:
        v = {"variant": variant, "embed": {}, "rerank": {}, "gain": {}, "fail": []}
        for key, label in (("vs_fp32", "vs fp32"), ("vs_fp16ac", "vs fp16ac(MPS现状)")):
            st = data["embed"].get(variant, {}).get(key)
            if not st:
                continue
            ok = st["mean"] >= T["embed_cos_mean_min"] and st["p1"] >= T["embed_cos_p1_min"]
            v["embed"][label] = {"mean": st["mean"], "p1": st["p1"], "min": st["min"], "pass": ok}
            if not ok:
                v["fail"].append(
                    "嵌入保真 " + label + ": mean=%.6f (需>=%s), P1=%.6f (需>=%s)"
                    % (st["mean"], T["embed_cos_mean_min"], st["p1"], T["embed_cos_p1_min"]))
        for key, label in (("vs_fp32", "vs fp32"), ("vs_fp16", "vs fp16(MPS现状)")):
            st = data["rerank"].get(variant, {}).get(key)
            if not st:
                continue
            ok = st["spearman_mean"] >= T["rerank_spearman_min"] and st["top1_agree"] >= T["rerank_top1_min"]
            v["rerank"][label] = {"spearman": st["spearman_mean"], "top1": st["top1_agree"], "pass": ok}
            if not ok:
                v["fail"].append(
                    "重排保真 " + label + ": Spearman=%.4f (需>=%s), top1=%.4f (需>=%s)"
                    % (st["spearman_mean"], T["rerank_spearman_min"], st["top1_agree"], T["rerank_top1_min"]))
        # 收益：吞吐（threads=2 容器代理，同批大小）或 常驻内存下降
        bth = _raw("embed", "fp32", "throughput") or {}
        qth = _raw("embed", variant, "throughput") or {}
        speedup = None
        if bth.get("units_per_s") and qth.get("units_per_s"):
            speedup = qth["units_per_s"] / bth["units_per_s"]
        r_fp32 = [_resident("embed", "fp32"), _resident("rerank", "fp32")]
        r_q = [_resident("embed", variant), _resident("rerank", variant)]
        rss_drop = None
        if all(x is not None for x in r_fp32 + r_q):
            base_sum, q_sum = sum(r_fp32), sum(r_q)
            rss_drop = 1.0 - q_sum / base_sum
            v["retrieval_mb"] = {"fp32": base_sum, "quant": q_sum}
        v["gain"] = {"speedup_embed": speedup, "rss_drop": rss_drop}
        gain_ok = ((speedup is not None and speedup >= T["speedup_min"]) or
                   (rss_drop is not None and rss_drop >= T["rss_drop_min"]))
        if not gain_ok:
            v["fail"].append("收益: 吞吐 %s (需>=%sx), 常驻下降 %s (需>=%s)" % (
                None if speedup is None else round(speedup, 3), T["speedup_min"],
                None if rss_drop is None else round(rss_drop, 3), T["rss_drop_min"]))
        v["pass"] = not v["fail"]
        verdicts[variant] = v
    return verdicts


# ─────────────── 备忘 ───────────────

def _fmt(x, nd=3, unit=""):
    if x is None:
        return "—"
    if isinstance(x, float):
        return ("%.*f" % (nd, x)) + unit
    return str(x) + unit


def memo(data: dict, verdicts: dict) -> str:
    L = []
    A = L.append
    A("# 决策备忘：bge-m3 + bge-reranker-v2-m3 量化对拍（int8 动态量化 / ONNX 量化）")
    A("")
    A("- 生成时间：%s" % datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"))
    A("- 阈值登记：%s" % THRESHOLDS_REGISTERED_AT)
    A("- 机器：Apple M2 / 8 核 / 8GB，本机纯原生（无 docker）")
    A("- 数据：artifacts/chunks.json 真实块 512 条（嵌入）；120 组 (query, 12 候选)（重排）")
    A("- 批大小：嵌入 64 / max_len 512；重排 32 / max_len 384（= chatbot.py 现役默认）")
    A("- 阈值：%s" % json.dumps(THRESHOLDS, ensure_ascii=False))
    A("")
    A("## 表 1 内存（每变体独立进程实测）")
    A("")
    A("| 任务 | 变体 | 权重字节 | 装载后常驻增量 | 峰值 RSS | 装载 s | 量化 s |")
    A("|---|---|---|---|---|---|---|")
    for task in ("embed", "rerank"):
        for r in mem_rows(data, task):
            A("| %s | %s | %s | %s | %s | %s | %s |" % (
                task, PRETTY.get(r["variant"], r["variant"]),
                _fmt(r["model_bytes_gb"], 3, " GB"), _fmt(r["resident_mb"], 0, " MB"),
                _fmt(r["peak_mb"], 0, " MB"), _fmt(r["t_load_s"], 1), _fmt(r["t_quant_s"], 1)))
    A("")
    A("## 表 2 吞吐（同批真实块；CPU 2 线程 = 2 vCPU 容器代理，MPS = 本机 GPU）")
    A("")
    A("| 任务 | 变体 | 设备 | threads=2 (units/s) | threads=8 (units/s) | 相对 fp32(CPU,2线程) |")
    A("|---|---|---|---|---|---|")
    for task in ("embed", "rerank"):
        sp = speed_rows(data, task)
        base2 = (sp.get("fp32") or {}).get("thr_units_per_s")
        for variant, r in sp.items():
            gain = (r["thr_units_per_s"] / base2) if (base2 and r["thr_units_per_s"]) else None
            A("| %s | %s | %s | %s | %s | %s |" % (
                task, PRETTY.get(variant, variant), r["device"],
                _fmt(r["thr_units_per_s"], 3), _fmt(r["fid_units_per_s"], 3), _fmt(gain, 3, "x")))
    A("")
    A("## 表 3 嵌入保真（512 条真实块逐条余弦）")
    A("")
    A("| 变体 | 基线 | 余弦均值 | P1 | 最小 | max|Δ| | 判定 |")
    A("|---|---|---|---|---|---|---|")
    for variant, rec in data["embed"].items():
        for key, label in (("vs_fp32", "fp32(CPU)"), ("vs_fp16ac", "fp16 autocast(MPS)")):
            st = rec.get(key)
            if not st:
                continue
            ok = st["mean"] >= THRESHOLDS["embed_cos_mean_min"] and st["p1"] >= THRESHOLDS["embed_cos_p1_min"]
            A("| %s | %s | %.6f | %.6f | %.6f | %.5f | %s |" % (
                PRETTY.get(variant, variant), label, st["mean"], st["p1"], st["min"],
                st["max_abs_delta"], "过" if ok else "不过"))
    A("")
    A("## 表 4 重排保真（120 组 × 12 候选）")
    A("")
    A("| 变体 | 基线 | Spearman 均值 | Spearman P5 | top-1 一致率 | 分数 max|Δ| | 判定 |")
    A("|---|---|---|---|---|---|---|")
    for variant, rec in data["rerank"].items():
        for key, label in (("vs_fp32", "fp32(CPU)"), ("vs_fp16", "fp16(MPS)")):
            st = rec.get(key)
            if not st:
                continue
            ok = st["spearman_mean"] >= THRESHOLDS["rerank_spearman_min"] and st["top1_agree"] >= THRESHOLDS["rerank_top1_min"]
            A("| %s | %s | %.4f | %.4f | %.1f%% | %.4f | %s |" % (
                PRETTY.get(variant, variant), label, st["spearman_mean"], st["spearman_p5"],
                st["top1_agree"] * 100, st["score_max_abs_delta"], "过" if ok else "不过"))
    A("")
    A("## 表 5 共驻留算术（单位 MB）")
    A("")
    A("容器 HW_MEM_MB = %d；应用基线 %d；qdrant %d；LLM(权重+KV q8_0) %d → 检索侧预算 = %d MB" % (
        HW_MEM_MB, APP_BASELINE_MB, QDRANT_MB, LLM_MB, RETRIEVAL_BUDGET_MB))
    A("")
    A("| 组合 | 嵌入常驻 | 重排常驻 | 检索侧合计 | 总占用 | 是否 <= HW_MEM_MB |")
    A("|---|---|---|---|---|---|")
    combos = [
        ("fp32 双模型（CPU 容器现状）", "embed", "fp32", "rerank", "fp32"),
        ("fp16 双模型（Mac 现状 fp16ac+fp16）", "embed", "fp16ac", "rerank", "fp16"),
        ("int8 双模型（仅 Linear）", "embed", "int8", "rerank", "int8"),
        ("int8 双模型（Linear+Embedding）", "embed", "int8e", "rerank", "int8e"),
        ("ONNX-int8 双模型", "embed", "onnx_int8", "rerank", "onnx_int8"),
    ]
    for label, t1, v1, t2, v2 in combos:
        r1, r2 = _resident(t1, v1), _resident(t2, v2)
        if r1 is None or r2 is None:
            A("| %s | — | — | — | — | 数据缺失 |" % label)
            continue
        tot = r1 + r2
        grand = tot + APP_BASELINE_MB + QDRANT_MB + LLM_MB
        A("| %s | %.0f | %.0f | **%.0f** | %.0f | %s |" % (
            label, r1, r2, tot, grand,
            "✅ 可行" if grand <= HW_MEM_MB else "❌ 超 %d MB" % int(grand - HW_MEM_MB)))
    A("")
    A("## 表 6 判定（登记阈值，三项全过才采纳）")
    A("")
    A("| 变体 | 嵌入保真 | 重排保真 | 收益 | 结论 | 失败数字 |")
    A("|---|---|---|---|---|---|")
    for variant, v in verdicts.items():
        e_ok = all(x["pass"] for x in v["embed"].values()) if v["embed"] else False
        r_ok = all(x["pass"] for x in v["rerank"].values()) if v["rerank"] else False
        gain_ok = not any(f.startswith("收益") for f in v["fail"])
        A("| %s | %s | %s | %s | **%s** | %s |" % (
            PRETTY.get(variant, variant), "过" if e_ok else "不过", "过" if r_ok else "不过",
            "过" if gain_ok else "不过", "采纳" if v["pass"] else "拒绝",
            "；".join(v["fail"]) if v["fail"] else "—"))
    A("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="量化对拍 · 判定端")
    ap.add_argument("--check-thresholds", action="store_true")
    ap.add_argument("--memo", action="store_true")
    ap.add_argument("--json", action="store_true", help="导出判定 JSON")
    a = ap.parse_args(argv)
    if a.check_thresholds:
        print(json.dumps({"registered_at": THRESHOLDS_REGISTERED_AT, "thresholds": THRESHOLDS},
                         ensure_ascii=False, indent=2))
        return
    data = collect()
    verdicts = judge(data)
    if a.json:
        (OUT / "verdict.json").write_text(json.dumps({"thresholds": THRESHOLDS, "verdicts": verdicts},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(verdicts, ensure_ascii=False, indent=2))
        return
    print(memo(data, verdicts))


if __name__ == "__main__":
    main()
