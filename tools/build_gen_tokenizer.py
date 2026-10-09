#!/usr/bin/env python3
"""从 Ollama 的 GGUF 模型里抽出生成模型的 tokenizer，落成 tokenizer.json。

Ollama 不提供分词接口（/api/tokenize 404），本机又常无外网；GGUF 里却带完整 vocab +
merges。本工具**纯标准库**解析 GGUF 元数据，用 tokenizers 重建 GPT-2 式 BPE（含 Qwen
切分正则），产出可直接喂给 RAG_GEN_TOKENIZER_PATH 的 tokenizer.json。

产物是派生数据、可从模型重放，不入库：`artifacts/gen_tokenizer/` 已 gitignore。

用法：
  python tools/build_gen_tokenizer.py --model qwen3.5:4b-q4_K_M \
      --out artifacts/gen_tokenizer/tokenizer.json
  python tools/build_gen_tokenizer.py --gguf /path/to/model.gguf --out ...
"""
import argparse
import json
import os
import struct
import sys

FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
       10: "Q", 11: "q", 12: "d"}
SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
WANT = {
    "tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.model",
    "tokenizer.ggml.pre", "tokenizer.ggml.token_type",
}
# Qwen 系 tokenizer 的切分正则（`pre=qwen35` 在 llama.cpp 内是手写状态机，无正则串可挖；
# 已用模型真实 prompt_eval_count 逐条对照确认，含数字串与混合中文）。
QWEN_PRE = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SELF_CHECK = ["hello world", "你好，世界！", "第3回 贾宝玉初试云雨情",
              "1234567890", "《红楼梦》第一回\n\n此开卷第一回也。"]


def find_gguf(model):
    """按 Ollama manifest 定位 model 层 blob。"""
    base = os.path.expanduser("~/.ollama/models/manifests/registry.ollama.ai/library")
    name, tag = model.split(":", 1) if ":" in model else (model, "latest")
    man = os.path.join(base, name, tag)
    if not os.path.isfile(man):
        raise SystemExit(f"找不到 manifest: {man}；用 --gguf 直接给 GGUF 路径")
    layers = json.load(open(man))["layers"]
    digest = next(l["digest"].split(":", 1)[1] for l in layers
                  if l["mediaType"].endswith("image.model"))
    return os.path.expanduser("~/.ollama/models/blobs/sha256-" + digest)


def read_meta(path, want):
    """只读 GGUF 头部元数据，不碰张量数据。"""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"不是 GGUF: {path}")
        struct.unpack("<I", f.read(4))[0]          # version
        struct.unpack("<Q", f.read(8))[0]          # tensor_count
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        out = {}
        for _ in range(n_kv):
            key = rstr()
            t = struct.unpack("<I", f.read(4))[0]
            if t == 8:
                v = rstr()
                if key in want:
                    out[key] = v
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                cnt = struct.unpack("<Q", f.read(8))[0]
                if et == 8:
                    vals = [rstr() for _ in range(cnt)]
                    if key in want:
                        out[key] = vals
                else:
                    vals = list(struct.unpack("<" + FMT[et] * cnt, f.read(SIZE[et] * cnt)))
                    if key in want:
                        out[key] = vals
            else:
                v = struct.unpack("<" + FMT[t], f.read(SIZE[t]))[0]
                if key in want:
                    out[key] = v
    return out


def build_tokenizer(meta):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    if meta.get("tokenizer.ggml.model") != "gpt2":
        raise SystemExit(f"仅支持 gpt2 式 BPE，收到 {meta.get('tokenizer.ggml.model')!r}")
    tokens = meta["tokenizer.ggml.tokens"]
    merges = meta["tokenizer.ggml.merges"]
    vocab = {tok: i for i, tok in enumerate(tokens)}
    if len(vocab) != len(tokens):
        print(f"⚠️  词表有重复：{len(tokens)} 条里唯一 {len(vocab)} 个", file=sys.stderr)
    pairs = []
    for m in merges:
        a, _, b = m.partition(" ")
        if b:
            pairs.append((a, b))
    tk = Tokenizer(models.BPE(vocab=vocab, merges=pairs))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(QWEN_PRE, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    return tk, len(tokens), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b-q4_K_M", help="Ollama 模型名")
    ap.add_argument("--gguf", default=None, help="直接指定 GGUF，跳过 manifest")
    ap.add_argument("--out", default="artifacts/gen_tokenizer/tokenizer.json")
    args = ap.parse_args()

    path = args.gguf or find_gguf(args.model)
    print(f"GGUF: {path}")
    meta = read_meta(path, WANT)
    tk, n_tok, n_merge = build_tokenizer(meta)
    print(f"pre={meta.get('tokenizer.ggml.pre')} tokens={n_tok} merges={n_merge}")

    bad = 0
    for s in SELF_CHECK:
        if tk.decode(tk.encode(s).ids) != s:
            bad += 1
            print(f"  ✗ 往返失败: {s!r}")
    if bad:
        raise SystemExit(f"{bad} 条往返失败，词表/字节处理有问题，不写出")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tk.save(args.out)
    print(f"写出 {args.out}（{os.path.getsize(args.out)} 字节，往返自检 {len(SELF_CHECK)}/{len(SELF_CHECK)}）")
    print(f"启用：.env 设 RAG_GEN_TOKENIZER_PATH={args.out}")


if __name__ == "__main__":
    main()
