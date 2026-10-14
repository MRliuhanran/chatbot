#!/usr/bin/env python3
"""为 infinity 准备"模型影子视图"：补上 sentence-transformers 需要、本地目录里却没有的模块。

背景（本机实测）：infinity 的 torch 嵌入引擎走 SentenceTransformer，它按 modules.json 逐个
加载模块；bge-m3 的 modules.json 要求 `1_Pooling`（CLS 池化）与 `2_Normalize`，而本项目从
ModelScope 拉的目录里只有权重 + tokenizer，**没有这两个子目录**。SentenceTransformer 在子目录
不存在时会退化成"把模型路径当 HF repo id 去下载"，于是抛
`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`，
模型服务根本起不来。

为什么不在真实模型目录里补：`_model_content_fingerprint` 会遍历模型目录做全量哈希，多一个
文件就让切分指纹漂移、索引被判过期 → 一次全量重建。影子目录只给模型服务用，真实目录一个
字节都不动（链接指向原文件）。

做法：目标目录 = 源目录的符号链接视图 + 缺失的模块子目录。池化方式与 chatbot.py 的
`_embed_batch` 对齐：**CLS + L2 归一**（`last_hidden_state[:, 0]` → `F.normalize`）。
其它类型的缺失模块直接报错、不猜。

用法：python tools/prepare_infinity_view.py <源模型目录> <视图目录> [--max-length N]

`--max-length`：把视图里的 `tokenizer_config.json` 覆写成指定的 `model_max_length`。
infinity 的重排按 tokenizer 的 model_max_length 截断（它代码里那行 max_length 是注释掉的），
而应用侧按 `RAG_RERANK_MAX_LENGTH`（默认 384）截断 —— 不对齐时**长候选的排序会变**（本机
实测：20 条全长候选 Spearman 0.85，截到 150 字符后 1.00）。改视图里的 tokenizer 配置即可
对齐，真实模型目录仍是一个字节不动。
"""
import json
import os
import sys

PATCHABLE = ("tokenizer_config.json",)


def _pooling_config(dim):
    if not isinstance(dim, int) or dim <= 0:
        raise SystemExit("源目录 config.json 里没有可用的 hidden_size，拒绝写一个维度未知的池化配置")
    return {
        "word_embedding_dimension": dim,
        "pooling_mode_cls_token": True,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }


def main(argv):
    argv = list(argv)
    max_length = 0
    if "--max-length" in argv:
        i = argv.index("--max-length")
        try:
            max_length = int(argv[i + 1])
        except (IndexError, ValueError):
            raise SystemExit("--max-length 需要一个整数") from None
        del argv[i:i + 2]
    if len(argv) != 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    src, dst = os.path.abspath(argv[1]), os.path.abspath(argv[2])
    if not os.path.isdir(src):
        raise SystemExit(f"源模型目录不存在: {src}")
    if src == dst:
        raise SystemExit("源与视图目录相同 —— 那会改动真实模型目录，拒绝执行")
    os.makedirs(dst, exist_ok=True)

    for name in sorted(os.listdir(src)):
        if max_length and name in PATCHABLE:
            continue  # 下面单独写补丁版，不建链接
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.lexists(d):
            if os.path.islink(d) and os.readlink(d) == s:
                continue
            raise SystemExit(f"视图目录已存在 {d}（且不是本脚本建的链接），拒绝覆盖")
        os.symlink(s, d)

    modules_file = os.path.join(dst, "modules.json")
    dim = None
    cfg_file = os.path.join(dst, "config.json")
    if os.path.isfile(cfg_file):
        with open(cfg_file, encoding="utf-8") as f:
            dim = json.load(f).get("hidden_size")
    if os.path.isfile(modules_file):
        with open(modules_file, encoding="utf-8") as f:
            modules = json.load(f)
        for module in modules:
            path = (module.get("path") or "").strip()
            if path in ("", "."):
                continue
            target = os.path.join(dst, path)
            if os.path.isdir(target):
                continue
            mtype = module.get("type", "")
            if mtype.endswith("Pooling"):
                os.makedirs(target, exist_ok=True)
                with open(os.path.join(target, "config.json"), "w", encoding="utf-8") as f:
                    json.dump(_pooling_config(dim), f, indent=2)
                print(f"  + {path}/config.json（CLS 池化 + 归一，dim={dim}）")
            elif mtype.endswith("Normalize"):
                os.makedirs(target, exist_ok=True)
                print(f"  + {path}/（Normalize 不需要文件）")
            else:
                raise SystemExit(f"模块 {path!r}（{mtype}）在源目录缺失，本脚本不知道如何补齐 —— 不猜")
    if max_length:
        tc_src = os.path.join(src, "tokenizer_config.json")
        if not os.path.isfile(tc_src):
            raise SystemExit(f"要求 --max-length，但源目录没有 tokenizer_config.json: {tc_src}")
        with open(tc_src, encoding="utf-8") as f:
            tc = json.load(f)
        old = tc.get("model_max_length")
        tc["model_max_length"] = max_length
        tc_dst = os.path.join(dst, "tokenizer_config.json")
        if os.path.islink(tc_dst) or os.path.exists(tc_dst):
            os.remove(tc_dst)
        with open(tc_dst, "w", encoding="utf-8") as f:
            json.dump(tc, f, ensure_ascii=False, indent=2)
        print(f"  ~ tokenizer_config.json: model_max_length {old} → {max_length}（与应用侧截断对齐）")
    print(f"视图就绪: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
