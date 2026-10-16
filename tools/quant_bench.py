#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化对拍 · 测量端（bench）。

回答的问题：BAAI/bge-m3 + BAAI/bge-reranker-v2-m3 换成 int8 动态量化 / ONNX 量化后，
内存能否从约 4.2GB 压到约 1.1GB，代价是多少精度与吞吐。

本脚本只**产数据**（内存 RSS / 模型字节 / 吞吐 / 原始向量与分数），
判定规则与阈值**不在本文件**，全部登记在 tools/quant_parity.py 的 THRESHOLDS（测量前已提交）。

设计要点
--------
* 一个变体 = 一个**独立子进程**：RSS 只在干净进程里有意义（量化模型和 fp32 模型不能同进程共存测量）。
* 保真用真实块：artifacts/chunks.json 的 contextual_text（与 chatbot.py::_index_text 同源），
  固定种子采样，样本清单落在 artifacts/quant/samples.json（判定脚本读同一份，保证可比）。
* 保真与吞吐分离：数值保真与线程数无关（同线程数下比对即可），故保真跑 threads=8 省时间；
  吞吐单独跑 threads=2 —— 2 vCPU 容器的代理（本机是 8 核 Apple M2，见 --threads 说明）。
* 不改仓库任何既有文件；产物全部落 artifacts/quant/（.gitignore 已忽略 artifacts/）。

用法
----
    python tools/quant_bench.py plan                     # 采样，写 samples.json
    python tools/quant_bench.py onnx-prepare             # 导出 + 量化 ONNX（需 onnx/onnxruntime）
    python tools/quant_bench.py run --tasks embed,rerank # 跑全部变体（子进程串行）
    python tools/quant_bench.py worker --task embed --variant int8 --phase fidelity
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "quant"
RAW = OUT / "raw"
ONNX_DIR = OUT / "onnx"
CHUNKS_JSON = ROOT / "artifacts" / "chunks.json"
EMBED_MODEL = ROOT / "models" / "bge-m3"
RERANK_MODEL = ROOT / "models" / "bge-reranker-v2-m3"

# ── 采样与批参数：与 chatbot.py 现役默认一致 ──
SEED = 20261016
N_EMBED = 512          # 硬约束：>=500 条真实块
N_GROUPS = 120         # 硬约束：>=100 组 (query, 候选集)
N_CANDS = 12           # 候选集大小（生产 RERANK_TOP_K=20 的 12 条子集；越大越慢，见 --cands）
EMBED_BATCH = 64       # chatbot.py RAG_EMBED_BATCH_SIZE 默认 64
RERANK_BATCH = 32      # chatbot.py RAG_RERANK_BATCH 默认 32
EMBED_MAX_LEN = 512    # chatbot.py::_embed_batch 硬编码 max_length=512
RERANK_MAX_LEN = 384   # RAG_RERANK_MAX_LENGTH 默认 384
N_THROUGHPUT_EMBED = 128   # 吞吐子集（同一批块的前 128 条，同批大小）
N_THROUGHPUT_GROUPS = 30

# ── 变体表 ──
# quant: None | "lin"（仅 nn.Linear，torch 标准配方）| "lin_emb"（Linear + nn.Embedding）
# dtype: 权重精度；device: cpu/mps；autocast: 复刻 chatbot.py 在 MPS 上的 fp16 autocast 现状
VARIANTS = {
    "fp32":      {"quant": None,      "device": "cpu", "dtype": "float32"},
    "int8":      {"quant": "lin",     "device": "cpu", "dtype": "float32"},
    "int8e":     {"quant": "lin_emb", "device": "cpu", "dtype": "float32"},
    "fp16":      {"quant": None,      "device": "mps", "dtype": "float16"},
    "fp16ac":    {"quant": None,      "device": "mps", "dtype": "float32", "autocast": True},
    "onnx_fp32": {"onnx": "fp32",     "device": "cpu"},
    "onnx_int8": {"onnx": "int8",     "device": "cpu"},
}
# 重排侧现状是 fp16 权重（chatbot.py::load_reranker_model 传了 torch_dtype=float16），
# 嵌入侧现状是 fp32 权重 + autocast（load_embedding_model 没传 dtype），故 fp16ac 只对 embed 有意义。
TASK_VARIANTS = {
    "embed": ["fp32", "int8", "int8e", "fp16", "fp16ac", "onnx_fp32", "onnx_int8"],
    "rerank": ["fp32", "int8", "int8e", "fp16", "onnx_fp32", "onnx_int8"],
}


# ───────────────────────── 内存/时间工具 ─────────────────────────

def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return float("nan")


def peak_rss_mb() -> float:
    """进程峰值 RSS。macOS 的 ru_maxrss 单位是字节，Linux 是 KB —— 按平台换算。"""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 2 ** 20 if sys.platform == "darwin" else v / 2 ** 10


def param_bytes(model) -> int:
    """权重实际字节数（int8 量化后 packed 权重按 1 字节/元素计）。"""
    import torch
    try:
        sd = model.state_dict()
    except Exception:
        return -1
    tot = 0
    for v in sd.values():
        if torch.is_tensor(v):
            tot += v.numel() * v.element_size()
    return tot


# ───────────────────────── 采样 ─────────────────────────

def build_samples() -> dict:
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    emb_idx = sorted(rng.sample(range(len(chunks)), N_EMBED))
    emb_texts = [chunks[i]["contextual_text"] for i in emb_idx]

    groups = []
    for g in range(N_GROUPS):
        q_i = rng.randrange(len(chunks))
        q_text = chunks[q_i]["contextual_text"].strip().splitlines()
        query = (q_text[0] if q_text else "").strip()[:64] or chunks[q_i]["contextual_text"][:64]
        cand = [q_i]
        while len(cand) < N_CANDS:
            c = rng.randrange(len(chunks))
            if c not in cand:
                cand.append(c)
        groups.append({
            "query": query,
            "cand_ids": cand,
            "cand_texts": [chunks[c]["contextual_text"] for c in cand],
        })

    samples = {
        "seed": SEED,
        "n_chunks_total": len(chunks),
        "embed": {"ids": emb_idx, "texts": emb_texts},
        "rerank": groups,
        "params": {
            "embed_batch": EMBED_BATCH, "rerank_batch": RERANK_BATCH,
            "embed_max_len": EMBED_MAX_LEN, "rerank_max_len": RERANK_MAX_LEN,
            "n_throughput_embed": N_THROUGHPUT_EMBED, "n_throughput_groups": N_THROUGHPUT_GROUPS,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    return samples


def load_samples() -> dict:
    p = OUT / "samples.json"
    if not p.exists():
        return build_samples()
    return json.loads(p.read_text(encoding="utf-8"))


# ───────────────────────── 模型装载 ─────────────────────────

def _quantize(model, spec: str, log: dict):
    """torch 动态量化。spec='lin' 只量化 nn.Linear（torch 标准配方）；
    spec='lin_emb' 额外量化 nn.Embedding 的权重表（bge-m3 词表 250002×1024 的
    fp32 权重表 = 1.02GB，是这两个模型里最大的一块内存，标准配方**碰不到它**）。"""
    import torch
    import torch.nn as nn
    from torch.ao.quantization import (
        float_qparams_weight_only_qconfig,
        per_channel_dynamic_qconfig,
        quantize_dynamic,
    )
    t0 = time.time()
    if spec == "lin":
        qm = quantize_dynamic(model, {nn.Linear: per_channel_dynamic_qconfig}, inplace=False)
    elif spec == "lin_emb":
        qm = quantize_dynamic(
            model,
            {nn.Linear: per_channel_dynamic_qconfig, nn.Embedding: float_qparams_weight_only_qconfig},
            inplace=False,
        )
    else:
        raise ValueError(spec)
    log["t_quant_s"] = round(time.time() - t0, 2)
    n_lin = sum(1 for m in qm.modules() if type(m).__module__.startswith("torch.ao.nn.quantized.dynamic")
                and type(m).__name__ == "Linear")
    n_emb = sum(1 for m in qm.modules() if type(m).__name__ == "Embedding"
                and type(m).__module__.startswith("torch.ao.nn.quantized"))
    log["n_quant_linear"] = n_lin
    log["n_quant_embedding"] = n_emb
    return qm


def load_torch_model(task: str, variant: str, log: dict):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    v = VARIANTS[variant]
    device, dtype_s = v["device"], v.get("dtype", "float32")
    path = EMBED_MODEL if task == "embed" else RERANK_MODEL
    device = device if (device != "mps" or torch.backends.mps.is_available()) else "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16}[dtype_s]

    tok = AutoTokenizer.from_pretrained(str(path))
    t0 = time.time()
    if task == "embed":
        model = AutoModel.from_pretrained(str(path), torch_dtype=dtype).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(path), torch_dtype=dtype).to(device)
    model.eval()
    log["t_load_s"] = round(time.time() - t0, 2)
    log["device_actual"] = device
    if v.get("quant"):
        model = _quantize(model, v["quant"], log)
    log["dtype"] = dtype_s
    log["autocast"] = bool(v.get("autocast"))
    return tok, model, device


# ───────────────────────── 推理（torch） ─────────────────────────

def embed_pass_torch(tok, model, texts, device, batch, autocast, timing: dict):
    import numpy as np
    import torch

    vecs, t_infer = [], 0.0
    for i in range(0, len(texts), batch):
        b = texts[i:i + batch]
        inputs = tok(b, padding=True, truncation=True, max_length=EMBED_MAX_LEN, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            if autocast:
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    out = model(**inputs)
            else:
                out = model(**inputs)
        cls = out.last_hidden_state[:, 0].float()
        vecs.append(torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy())
        t_infer += time.time() - t0
    timing["t_infer_s"] = round(t_infer, 3)
    return np.vstack(vecs)


def rerank_pass_torch(tok, model, groups, device, batch, timing: dict):
    import numpy as np
    import torch

    rows, t_infer = [], 0.0
    for g in groups:
        q, docs = g["query"], g["cand_texts"]
        scores = []
        for i in range(0, len(docs), batch):
            pairs = [(q, d) for d in docs[i:i + batch]]
            inputs = tok(pairs, padding=True, truncation=True, max_length=RERANK_MAX_LEN, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            t0 = time.time()
            with torch.no_grad():
                logits = model(**inputs).logits.squeeze(-1)
            t_infer += time.time() - t0
            scores.extend(logits.float().cpu().numpy().tolist())
        rows.append(scores)
    timing["t_infer_s"] = round(t_infer, 3)
    return np.asarray(rows, dtype=np.float32)


# ───────────────────────── ONNX 导出 / 量化 / 推理 ─────────────────────────

def _export_wrapper(task: str):
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoModelForSequenceClassification

    path = EMBED_MODEL if task == "embed" else RERANK_MODEL

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            if task == "embed":
                self.m = AutoModel.from_pretrained(str(path))
            else:
                self.m = AutoModelForSequenceClassification.from_pretrained(str(path))

        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            if task == "embed":
                cls = out.last_hidden_state[:, 0].float()
                return torch.nn.functional.normalize(cls, p=2, dim=1)
            return out.logits.squeeze(-1)

    return Wrap().eval()


def onnx_prepare(tasks, opset=17):
    """导出 fp32 ONNX（>2GB 用外部数据）并做 ORT 动态量化 → artifacts/quant/onnx/。"""
    import onnxruntime  # noqa: F401  仅探测可用性
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for task in tasks:
        tok = AutoTokenizer.from_pretrained(str(EMBED_MODEL if task == "embed" else RERANK_MODEL))
        fp32_path = ONNX_DIR / f"{task}_fp32.onnx"
        int8_path = ONNX_DIR / f"{task}_int8.onnx"
        rec = {"fp32_bytes": None, "int8_bytes": None, "t_export_s": None, "t_quant_s": None}
        if not fp32_path.exists():
            model = _export_wrapper(task)
            dummy = tok(["量化对拍的占位文本，用于导出计算图。"], return_tensors="pt")
            sig = inspect.signature(torch.onnx.export).parameters
            kw = {}
            if "dynamo" in sig:
                kw["dynamo"] = False           # 走 TorchScript 导出器（没装 onnxscript）
            if "external_data" in sig:
                kw["external_data"] = True     # >2GB 权重必须外置
            elif "use_external_data_format" in sig:
                kw["use_external_data_format"] = True
            t0 = time.time()
            torch.onnx.export(
                model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["vec" if task == "embed" else "logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    ("vec" if task == "embed" else "logits"): {0: "batch"},
                },
                opset_version=opset, do_constant_folding=True, **kw,
            )
            rec["t_export_s"] = round(time.time() - t0, 2)
            del model
            gc.collect()
        rec["fp32_bytes"] = sum(f.stat().st_size for f in ONNX_DIR.glob(f"{task}_fp32.onnx*"))
        if not int8_path.exists():
            t0 = time.time()
            quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                             per_channel=True, use_external_data_format=True)
            rec["t_quant_s"] = round(time.time() - t0, 2)
        rec["int8_bytes"] = sum(f.stat().st_size for f in ONNX_DIR.glob(f"{task}_int8.onnx*"))
        report[task] = rec
        print(f"[onnx] {task}: fp32={rec['fp32_bytes']} B int8={rec['int8_bytes']} B "
              f"export={rec['t_export_s']}s quant={rec['t_quant_s']}s", flush=True)
    (OUT / "onnx_prepare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _ort_session(task, which, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    path = ONNX_DIR / f"{task}_{which}.onnx"
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def embed_pass_ort(sess, tok, texts, batch, timing):
    import numpy as np

    out_name = sess.get_outputs()[0].name
    vecs, t_infer = [], 0.0
    for i in range(0, len(texts), batch):
        b = texts[i:i + batch]
        enc = tok(b, padding=True, truncation=True, max_length=EMBED_MAX_LEN, return_tensors="np")
        feed = {"input_ids": enc["input_ids"].astype("int64"),
                "attention_mask": enc["attention_mask"].astype("int64")}
        t0 = time.time()
        out = sess.run([out_name], feed)[0]
        t_infer += time.time() - t0
        v = np.asarray(out, dtype=np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        vecs.append(v / np.maximum(n, 1e-12))
    timing["t_infer_s"] = round(t_infer, 3)
    return np.vstack(vecs)


def rerank_pass_ort(sess, tok, groups, batch, timing):
    import numpy as np

    out_name = sess.get_outputs()[0].name
    rows, t_infer = [], 0.0
    for g in groups:
        q, docs = g["query"], g["cand_texts"]
        scores = []
        for i in range(0, len(docs), batch):
            pairs = [(q, d) for d in docs[i:i + batch]]
            enc = tok(pairs, padding=True, truncation=True, max_length=RERANK_MAX_LEN, return_tensors="np")
            feed = {"input_ids": enc["input_ids"].astype("int64"),
                    "attention_mask": enc["attention_mask"].astype("int64")}
            t0 = time.time()
            out = sess.run([out_name], feed)[0]
            t_infer += time.time() - t0
            scores.extend(np.asarray(out, dtype=np.float32).reshape(-1).tolist())
        rows.append(scores)
    timing["t_infer_s"] = round(t_infer, 3)
    return np.asarray(rows, dtype=np.float32)


# ───────────────────────── worker ─────────────────────────

def worker(task: str, variant: str, phase: str, threads: int, limit: int | None, cands: int | None,
           batch: int | None) -> dict:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    s = load_samples()
    v = VARIANTS[variant]
    is_onnx = "onnx" in v

    if task == "embed":
        texts = s["embed"]["texts"]
        if limit:
            texts = texts[:limit]
        bsz = batch or s["params"]["embed_batch"]
        units = len(texts)
    else:
        groups = s["rerank"]
        if cands:
            groups = [{"query": g["query"], "cand_texts": g["cand_texts"][:cands]} for g in groups]
        if limit:
            groups = groups[:limit]
        bsz = batch or s["params"]["rerank_batch"]
        units = sum(len(g["cand_texts"]) for g in groups)

    log = {
        "task": task, "variant": variant, "phase": phase, "threads": threads,
        "batch": bsz, "n_units": units, "n_items": len(texts) if task == "embed" else len(groups),
        "python": sys.version.split()[0], "platform": sys.platform,
        "torch": torch.__version__, "mps": bool(torch.backends.mps.is_available()),
        "rss_before_load_mb": round(rss_mb(), 1),
    }
    t0 = time.time()
    timing = {}
    if is_onnx:
        sess = _ort_session(task, v["onnx"], threads)
        tok = None
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(EMBED_MODEL if task == "embed" else RERANK_MODEL))
        log["t_load_s"] = round(time.time() - t0, 2)
    else:
        tok, model, device = load_torch_model(task, variant, log)
    gc.collect()
    log["rss_load_mb"] = round(rss_mb(), 1)
    log["model_bytes"] = (sum(f.stat().st_size for f in ONNX_DIR.glob(
        f"{task}_{v['onnx']}.onnx*")) if is_onnx else param_bytes(model))

    # 预热（不计入吞吐）
    if task == "embed":
        warm = texts[:min(bsz, len(texts))]
        if is_onnx:
            embed_pass_ort(sess, tok, warm, bsz, {})
        else:
            embed_pass_torch(tok, model, warm, device, bsz, log["autocast"], {})
    else:
        warm = [{"query": groups[0]["query"], "cand_texts": groups[0]["cand_texts"][:min(bsz, cands or N_CANDS)]}]
        if is_onnx:
            rerank_pass_ort(sess, tok, warm, bsz, {})
        else:
            rerank_pass_torch(tok, model, warm, device, bsz, {})

    t_run0 = time.time()
    if task == "embed":
        mat = (embed_pass_ort(sess, tok, texts, bsz, timing) if is_onnx
               else embed_pass_torch(tok, model, texts, device, bsz, log["autocast"], timing))
    else:
        mat = (rerank_pass_ort(sess, tok, groups, bsz, timing) if is_onnx
               else rerank_pass_torch(tok, model, groups, device, bsz, timing))
    log["t_wall_s"] = round(time.time() - t_run0, 3)
    log["t_infer_s"] = timing.get("t_infer_s")
    log["units_per_s"] = round(units / max(log["t_wall_s"], 1e-9), 3)
    log["rss_after_mb"] = round(rss_mb(), 1)
    log["peak_rss_mb"] = round(peak_rss_mb(), 1)
    log["rss_delta_mb"] = round(log["rss_after_mb"] - log["rss_before_load_mb"], 1)

    RAW.mkdir(parents=True, exist_ok=True)
    tag = f"{task}__{variant}__{phase}" + (f"__b{bsz}" if batch else "")
    np.savez_compressed(RAW / f"{tag}.npz", mat=mat)
    (RAW / f"{tag}.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False), flush=True)
    return log


# ───────────────────────── 驱动 ─────────────────────────

def run_all(tasks, phases, threads_fidelity, threads_throughput, skip_onnx):
    env = dict(os.environ)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + str(OUT / "pylibs")
    results = []
    for task in tasks:
        variants = list(TASK_VARIANTS[task])
        if skip_onnx:
            variants = [v for v in variants if "onnx" not in v]
        for variant in variants:
            for phase in phases:
                if phase == "fidelity":
                    th, lim, cd = threads_fidelity, None, None
                else:
                    th, lim = threads_throughput, (N_THROUGHPUT_EMBED if task == "embed" else N_THROUGHPUT_GROUPS)
                    cd = None
                cmd = [sys.executable, str(Path(__file__).resolve()), "worker",
                       "--task", task, "--variant", variant, "--phase", phase,
                       "--threads", str(th)]
                if lim:
                    cmd += ["--limit", str(lim)]
                print(f"\n>>> {' '.join(cmd)}", flush=True)
                p = subprocess.run(cmd, env=env, cwd=str(ROOT))
                results.append({"task": task, "variant": variant, "phase": phase, "rc": p.returncode})
                if p.returncode != 0:
                    print(f"!!! {task}/{variant}/{phase} 失败 rc={p.returncode}", flush=True)
    (OUT / "run_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="量化对拍 · 测量端")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("plan", help="采样并写 artifacts/quant/samples.json")

    w = sub.add_parser("worker", help="单变体单进程测量（内部用）")
    w.add_argument("--task", required=True, choices=["embed", "rerank"])
    w.add_argument("--variant", required=True, choices=list(VARIANTS))
    w.add_argument("--phase", default="fidelity", choices=["fidelity", "throughput"])
    w.add_argument("--threads", type=int, default=8)
    w.add_argument("--limit", type=int, default=None)
    w.add_argument("--cands", type=int, default=None)
    w.add_argument("--batch", type=int, default=None)

    r = sub.add_parser("run", help="串行跑全部变体")
    r.add_argument("--tasks", default="embed,rerank")
    r.add_argument("--phases", default="fidelity,throughput")
    r.add_argument("--threads-fidelity", type=int, default=8)
    r.add_argument("--threads-throughput", type=int, default=2)
    r.add_argument("--skip-onnx", action="store_true")

    o = sub.add_parser("onnx-prepare", help="导出+量化 ONNX")
    o.add_argument("--tasks", default="embed,rerank")

    a = ap.parse_args(argv)
    if a.cmd == "plan":
        s = build_samples()
        print(json.dumps({"embed_n": len(s["embed"]["ids"]), "rerank_groups": len(s["rerank"]),
                          "cands": N_CANDS, "seed": SEED}, ensure_ascii=False))
    elif a.cmd == "worker":
        worker(a.task, a.variant, a.phase, a.threads, a.limit, a.cands, a.batch)
    elif a.cmd == "run":
        run_all(a.tasks.split(","), a.phases.split(","), a.threads_fidelity, a.threads_throughput, a.skip_onnx)
    elif a.cmd == "onnx-prepare":
        onnx_prepare(a.tasks.split(","))


if __name__ == "__main__":
    main()
