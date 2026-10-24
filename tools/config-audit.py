#!/usr/bin/env python3
"""配置一致性审计（以代码默认值为准）：部署前拦下"写了却没人读"的旧键与模型键漂移。

判据：
1) .env 里每个生效键，必须能在代码/启动脚本/配置消费方里找到引用；
   第三方运行时自己消费的变量（PYTORCH_*/OLLAMA_*/... 前缀）在允许名单里。
2) 模型选择键在 .env 与 tools/bake.env 两处出现时值必须一致（构建期/运行期同源）。
3) 明确废弃的键（见 REMOVED_KEYS）一旦在 .env 生效即报错。

退出码：0 = 一致；1 = 有漂移/旧键（deploy-check 据此 fail-closed）。
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(ROOT, ".env")
BAKE_FILE = os.path.join(ROOT, "tools", "bake.env")

CONSUMER_FILES = [
    "chatbot.py", "entrypoint.sh", "Dockerfile",
    "docker-compose.yml", "docker-compose.gpu.yml", "embed.Dockerfile",
]
CONSUMER_DIRS = ["tools", "qdrant_config", ".streamlit", ".devcontainer"]

# 第三方运行时自己消费的变量：代码里 grep 不到是正常的。
THIRD_PARTY_PREFIXES = (
    "PYTORCH_", "OLLAMA_", "QDRANT__", "HF_", "TRANSFORMERS_", "TOKENIZERS_",
    "CUDA_", "MALLOC_", "OPENBLAS_", "OMP_", "RAYON_", "LLAMA_", "MODELSCOPE_",
    "HOME", "PATH", "TMPDIR",
)

# 明确废弃：代码已不再读取，.env 里出现就是漂移。
REMOVED_KEYS = {
    "RAG_EXCLUSIVE_MODELS",
    "RAG_GEN_KEEP_ALIVE",
    "RAG_EXCLUSIVE_AUTO_FREE_MB",
    "RAG_TURN_MIN_FREE_MB",
}

_MODEL_KEYS = ("MODEL", "RAG_EMBED_MODEL_NAME", "RAG_RERANK_MODEL_NAME",
               "RAG_EMBED_MODEL_PATH", "RAG_RERANK_MODEL_PATH", "RAG_MODEL_DL_CONNS")


def read_env(path):
    """返回 {键: 值}；只认生效行（^KEY=...，忽略注释/空行）。"""
    out = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.rstrip("\n"))
                if m:
                    out[m.group(1)] = m.group(2).strip().strip("\"'")
    except OSError as exc:
        print(f"⚠️  读不到 {path}: {exc}")
    return out


def consumer_blob():
    chunks = []
    for rel in CONSUMER_FILES:
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p):
            chunks.append(open(p, encoding="utf-8", errors="ignore").read())
    for d in CONSUMER_DIRS:
        base = os.path.join(ROOT, d)
        for cur, _dirs, files in os.walk(base):
            for name in files:
                if name.endswith((".py", ".sh", ".yml", ".yaml", ".toml", ".json")):
                    p = os.path.join(cur, name)
                    chunks.append(open(p, encoding="utf-8", errors="ignore").read())
    return "\n".join(chunks)


def main():
    env = read_env(ENV_FILE)
    if not env:
        print("❌ .env 不存在或没有生效行")
        return 1
    blob = consumer_blob()

    problems = []

    removed = sorted(k for k in env if k in REMOVED_KEYS)
    for k in removed:
        problems.append(f"废弃键仍在 .env 生效：{k}（代码已不读取，删掉）")

    stale = []
    for k in sorted(env):
        if k in REMOVED_KEYS:
            continue
        if k.startswith(THIRD_PARTY_PREFIXES):
            continue
        if k not in blob:
            stale.append(k)
    for k in stale:
        problems.append(f".env 键没有消费方（代码/脚本都不读）：{k}")

    bake = read_env(BAKE_FILE)
    if bake:
        for k in _MODEL_KEYS:
            if k in env and k in bake and env[k] != bake[k]:
                problems.append(
                    f"模型键两源不一致：{k}  .env={env[k]!r}  tools/bake.env={bake[k]!r}"
                )

    print(f"配置审计：.env 生效键 {len(env)} 个；bake.env 生效键 {len(bake)} 个")
    if problems:
        print("❌ 发现 %d 处漂移（以代码默认值为准，逐条修）：" % len(problems))
        for p in problems:
            print("   -", p)
        return 1
    print("✅ 一致：无废弃键、无无消费方键、模型键两源一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
