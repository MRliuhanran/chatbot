#!/usr/bin/env python3
"""从 Ollama 的 GGUF 模型里抽出生成模型的 tokenizer，落成 tokenizer.json。

Ollama 不提供分词接口（/api/tokenize 404），本机又常无外网；GGUF 里却带完整 vocab +
merges。本工具**纯标准库**解析 GGUF 元数据，用 tokenizers 重建 GPT-2 式 BPE，产出可直接
喂给 RAG_GEN_TOKENIZER_PATH 的 tokenizer.json。

切分规则**不再写死某一模型**：读 GGUF 的 `tokenizer.ggml.pre`，按 llama.cpp
(src/llama-vocab.cpp 的 llm_tokenizer_bpe) 同款正则自动匹配（Qwen / Llama3 / DeepSeek /
Falcon / Starcoder / GPT-2 系 等）。换 Ollama 模型只需重跑本工具，不必改代码。
`pre` 未知或属于 llama.cpp 里的手写状态机家族（gemma4 / kimi-k2 等）时，退回 GPT-2 默认
正则并**显式告警**，绝不静默产出错误切分。

产物是派生数据、可从模型重放；`artifacts/gen_tokenizer/` 在 .gitignore 里做了目录白名单
（tokenizer.json 与来源 meta 随仓库发布，容器构建需要），不是整体忽略。

用法：
  python tools/build_gen_tokenizer.py --model qwen3.5:4b-q4_K_M \
      --out artifacts/gen_tokenizer/tokenizer.json
  python tools/build_gen_tokenizer.py --gguf /path/to/model.gguf --out ...
  python tools/build_gen_tokenizer.py --model llama3.1:8b --pre llama-bpe   # 手动指定 pre
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time

FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
       10: "Q", 11: "q", 12: "d"}
SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
WANT = {
    "tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.model",
    "tokenizer.ggml.pre", "tokenizer.ggml.token_type",
}

# ── 切分规则表 ───────────────────────────────────────────────────────────────
# 逐条对自 llama.cpp `src/llama-vocab.cpp` 的 `llm_tokenizer_bpe` 构造函数（各 case 的适配
# 后版本，非 tokenizer.json 原始的 `(?i:...)` 写法）。llama.cpp 对多个正则的行为是"按序
# 逐层递归切分"，这里用 pre_tokenizers.Sequence 链式 Split 等价复现。
#
# 反复出现的两个缩略后缀片段，抽成变量避免整段重复（保持原样字符串，不加多余分组，
# 以免改变 `|` 的结合性）：
_CONTR = r"'s|'t|'re|'ve|'m|'ll|'d"                                      # 原版
_CONTR_CI = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"  # llama.cpp 适配版
_GPT2_TAIL = _CONTR + r"| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"

_DEFAULT = [r"[\p{P}\$\+<=>\^~\|]+", _GPT2_TAIL, r"\p{N}+", r"[0-9][0-9][0-9]"]
_GPT2 = [_GPT2_TAIL]
_STARCODER = [r"\p{N}", _GPT2_TAIL]
_FALCON = [r"[\p{P}\$\+<=>\^~\|`]+", _GPT2_TAIL, r"[0-9][0-9][0-9]"]
_LLAMA3 = [
    _CONTR_CI + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]
_JAIS2 = [
    _CONTR_CI + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}",
]
_DEEPSEEK_LLM = [
    r"[\r\n]",
    r"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
    r"\s?[!-/:-~！-／：-～'-‟　-。]+",
    r"\s+$",
    r"[一-龥ࠀ-一가-퟿]+",
    r"\p{N}+",
]
_DEEPSEEK_CODER = [
    r"[\r\n]",
    r"\s?\p{L}+",
    r"\s?\p{P}+",
    r"[一-龥ࠀ-一가-퟿]+",
    r"\p{N}",
]
_DEEPSEEK3 = [
    r"\p{N}{1,3}",
    r"[一-龥一-龥぀-ゟ゠-ヿ]+",
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
]
_SPARK2_5 = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+""",
    r"\p{N}",
]
_YOUTU = [
    r"[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥一-龥぀-ゟ゠-ヿ]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]
_QWEN2 = [
    _CONTR_CI + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]
_QWEN35 = [
    _CONTR_CI + r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]
_PORO = [r" ?[^(\s|.,!?…。，、।۔،)]+"]
_CHATGLM4 = [_CONTR_CI + _LLAMA3[0][len(_CONTR_CI):]]  # 与 llama3 同尾，仅前缀缩略形式不同
_VIKING = [r" ?[^(\s|.,!?…。，、।۔،)]+", r"\p{N}"]
_TEKKEN = [
    r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]
_CHAMELEON = [
    r"<sentinel:[0-9]+>",
    r"(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
    r"([\t\n]|    |  )",
    r"\p{N}",
    r"[\p{P}!-/:-@\[-`{-~]",
    _GPT2_TAIL,
]
_GPT4O = [
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]
_GRANITE_EMB_MULTI = [
    r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]
_TINY_AYA = [
    r"\d{1,3}(?=(?:\d{3})*\b)",
    _GPT4O[0],
]
_SUPERBPE = [r"\p{N}+", r"(?=(\d{3})+(?!\d))"]
_BAILINGMOE = [
    r"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+",
]
_SEED_CODER = [
    _CONTR_CI + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+",
]
_UFAKZEKA = [
    r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]
_GROK_2 = [_QWEN2[0]]
_LAGUNA = [r"[^\n]+|[\n]+", _QWEN2[0]]
_EXAONE_MOE = [
    _CONTR_CI + r"|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+",
]
_MINICPM5 = [
    r"\p{N}{1,3}",
    _CONTR_CI + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]
_K2_HORIZON = [
    _CONTR_CI + r"|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]

# pre 别名（空格分隔）→ 切分规则。分组严格对自 llama.cpp `get_pre_type()`；同一规则的别名
# 合并成一行（如 dbrx / smaug-bpe 与 llama3 同正则，故并列）。
_PRE_GROUPS = [
    ("default", _DEFAULT),
    ("llama3 llama-v3 llama-bpe falcon3 falcon-h1 pixtral midm-2.0 lfm2 jina-v5-nano "
     "dbrx smaug-bpe", _LLAMA3),
    ("jais-2", _JAIS2),
    ("deepseek-llm", _DEEPSEEK_LLM),
    ("deepseek-coder", _DEEPSEEK_CODER),
    ("deepseek-v3 hunyuan-dense joyai-llm hy_v4", _DEEPSEEK3),
    ("spark2_5", _SPARK2_5),
    ("youtu", _YOUTU),
    ("falcon", _FALCON),
    ("mpt gpt-2 phi-2 jina-es jina-de gigachat jina-v2-es jina-v2-de a.x-4.0 mellum "
     "modern-bert jina-v1-en jina-v2-code roberta-bpe olmo exaone4 trillion granite-docling", _GPT2),
    ("starcoder refact command-r smollm codeshell exaone minerva-7b mellum2", _STARCODER),
    ("qwen2 deepseek-r1-qwen kormo f2llmv2 megrez stablelm2 hunyuan solar-open", _QWEN2),
    ("qwen35", _QWEN35),
    ("poro-chat bloom gpt3-finnish", _PORO),
    ("chatglm-bpe glm4 glm5", _CHATGLM4),
    ("viking", _VIKING),
    ("tekken", _TEKKEN),
    ("chameleon", _CHAMELEON),
    ("gpt-4o llama4 kanana2 talkie minimax-m2", _GPT4O),
    ("granite-embed-multi-97m", _GRANITE_EMB_MULTI),
    ("tiny_aya cohere2moe", _TINY_AYA),
    ("superbpe", _SUPERBPE),
    ("bailingmoe bailingmoe2 llada-moe", _BAILINGMOE),
    ("seed-coder", _SEED_CODER),
    ("ufakzeka", _UFAKZEKA),
    ("grok-2", _GROK_2),
    ("laguna", _LAGUNA),
    ("exaone-moe", _EXAONE_MOE),
    ("minicpm5", _MINICPM5),
    ("k2-horizon", _K2_HORIZON),
]
PRE_REGEX = {alias: regexes for aliases, regexes in _PRE_GROUPS for alias in aliases.split()}

# llama.cpp 里走"手写状态机"、无正则串可复刻的 pre（这些家族的 BPE 还是 raw UTF-8 /
# 特殊字节词表，本工具的 ByteLevel 路径表示不了）。命中即显式告警 + 退回 default。
_NO_REGEX_PRE = {"gemma4", "granite-embed-multi-311m", "mmbert", "sarvam-moe",
                 "whitespace", "kimi-k2", "afmoe"}

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
    digest = next(layer["digest"].split(":", 1)[1] for layer in layers
                  if layer["mediaType"].endswith("image.model"))
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


def resolve_pre_regex(pre):
    """pre 字符串 → 正则列表。未知 / 手写状态机家族 → 告警并退回 default（不静默错切）。"""
    pre = (pre or "default").strip() or "default"
    if pre in PRE_REGEX:
        return PRE_REGEX[pre], pre
    kind = "手写状态机（无正则可复刻）" if pre in _NO_REGEX_PRE else "未知（llama.cpp 未收录或版本更新）"
    print(f"⚠️  pre={pre!r} 属于{kind}，退回 default 正则；token 计数可能有偏差", file=sys.stderr)
    return _DEFAULT, "default"


def build_tokenizer(meta, pre_override=None):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    if meta.get("tokenizer.ggml.model") != "gpt2":
        raise SystemExit(
            f"仅支持 gpt2 式 BPE，收到 {meta.get('tokenizer.ggml.model')!r}"
            f"（SentencePiece 家族如 Llama2/Mistral/Gemma）。"
            f"这类模型请改用 HF 的 tokenizer.json：设 RAG_GEN_TOKENIZER_PATH 指向它"
            f"（chatbot.py 的 _load_gen_tokenizer 直接吃文件/目录），无需本工具。"
        )
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

    pre = pre_override if pre_override is not None else meta.get("tokenizer.ggml.pre")
    regex_exprs, pre_used = resolve_pre_regex(pre)

    tk = Tokenizer(models.BPE(vocab=vocab, merges=pairs))
    # llama.cpp 的多正则 = 逐层递归切分；Sequence 链式 Split 等价。
    seq = [pre_tokenizers.Split(r, behavior="isolated") for r in regex_exprs]
    seq.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence(seq)
    tk.decoder = decoders.ByteLevel()
    return tk, len(tokens), len(pairs), pre_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b-q4_K_M", help="Ollama 模型名")
    ap.add_argument("--gguf", default=None, help="直接指定 GGUF，跳过 manifest")
    ap.add_argument("--pre", default=None, help="手动覆盖 tokenizer.ggml.pre（调试用）")
    ap.add_argument("--out", default="artifacts/gen_tokenizer/tokenizer.json")
    args = ap.parse_args()

    path = args.gguf or find_gguf(args.model)
    print(f"GGUF: {path}")
    meta = read_meta(path, WANT)
    tk, n_tok, n_merge, pre_used = build_tokenizer(meta, pre_override=args.pre)
    print(f"pre={meta.get('tokenizer.ggml.pre')!r} → 切分规则={pre_used!r} "
          f"tokens={n_tok} merges={n_merge}")

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

    # 来源 meta：记录 tokenizer.json 是从哪个 GGUF 生成的。运行时据此校验"换模型后忘了
    # 重跑本工具"的错配（文件名即 sha256 digest，不必重算 1.5GB 文件）。
    _gguf_base = os.path.basename(path)
    _digest = _gguf_base[len("sha256-"):] if _gguf_base.startswith("sha256-") else ""
    if not _digest:
        _h = hashlib.sha256()
        with open(path, "rb") as _fh:
            for _blk in iter(lambda: _fh.read(8 << 20), b""):
                _h.update(_blk)
        _digest = _h.hexdigest()
    _meta_out = os.path.splitext(args.out)[0] + ".meta.json"
    with open(_meta_out, "w", encoding="utf-8") as _mf:
        json.dump({
            "version": 1,
            "tool": "tools/build_gen_tokenizer.py",
            "model": args.model if not args.gguf else "",
            "gguf": _gguf_base,
            "gguf_sha256": _digest,
            "pre": meta.get("tokenizer.ggml.pre"),
            "pre_used": pre_used,
            "tokens": n_tok,
            "merges": n_merge,
            "out": os.path.basename(args.out),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, _mf, ensure_ascii=False, indent=2)
        _mf.write("\n")
    print(f"写出来源 meta {_meta_out}（gguf_sha256={_digest[:12]}…）")
    print(f"启用：.env 设 RAG_GEN_TOKENIZER_PATH={args.out}")


if __name__ == "__main__":
    main()
