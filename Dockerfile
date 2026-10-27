# ── 单容器 ModelScope 镜像（无 compose、无 shell 脚本）──────────────────
# 一个容器里跑 Qdrant + Ollama + Streamlit，入口是 chatbot.py 的 all 命令；
# 与本地源码运行共用同一份 .env / models / artifacts / storage，功能与效果同源。
# 所有 RUN 使用 exec form；构建期模型下载复用仓库唯一的 start.sh（最终镜像不 COPY 它）。
#
# 构建:
#   docker build -t rag-chatbot:modelscope .
# 运行:
#   docker run --rm -p 7860:7860 rag-chatbot:modelscope
ARG PYTHON_VERSION=3.10

# ── 基础镜像：Debian 源切到国内镜像（构建机访问 deb.debian.org 会超时） ──
FROM python:${PYTHON_VERSION} AS base
RUN ["python","-c","import pathlib; [p.write_text(p.read_text().replace('deb.debian.org','mirrors.aliyun.com').replace('security.debian.org','mirrors.aliyun.com')) for p in [pathlib.Path('/etc/apt/sources.list'),*pathlib.Path('/etc/apt/sources.list.d').glob('*')] if p.is_file()]"]

# ── 直接复用官方镜像里的二进制，避免 RUN 下载/解压 ──
FROM qdrant/qdrant:v1.19.1 AS qdrant
FROM ollama/ollama:0.34.4 AS ollama

# ── 依赖 stage：完整 python 镜像自带编译工具，供 llama-cpp-python 编译 ──
FROM base AS pydeps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple \
    PIP_EXTRA_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    CMAKE_ARGS="-DGGML_OPENMP=OFF"
RUN ["pip","install","--no-cache-dir","cmake","ninja"]
COPY requirements.txt /tmp/requirements.txt
RUN ["pip","install","--no-cache-dir","https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.7.1%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl"]
RUN ["pip","install","--no-cache-dir","-r","/tmp/requirements.txt"]

# ── 检索模型 + GGUF stage：唯一 shell start.sh 构建期下载 ───────────────────
# 为什么不在 Dockerfile 里写 17 个 ADD：远程 ADD 不走层缓存，每次部署重下 4GB；
# 这里一个 RUN 可缓存，且下载逻辑并入本仓库唯一的 start.sh（Docker/本机共用）。
FROM base AS fetchmodels
ARG MODELS_REV=2026-10-26
ENV MODELS_REV=${MODELS_REV}
COPY start.sh /tmp/start.sh
RUN ["apt-get","update"]
RUN ["apt-get","install","-y","--no-install-recommends","aria2","ca-certificates"]
RUN ["bash","/tmp/start.sh","fetch-models","/models"]

# ── 资产 stage：与 pydeps 并行；源码/产物/配置 ──
# ollama_models 可选：留 .keep 也能构建；正式生成模型由 fetchmodels 下载，all 里执行 ollama create。
FROM scratch AS assets
COPY .env /app/.env
COPY corpus/ /app/corpus/
COPY artifacts/ /app/artifacts/
COPY lexicon/ /app/lexicon/
COPY .streamlit/ /app/.streamlit/
COPY ollama_models/ /app/ollama_models/
COPY qdrant_config/ /app/qdrant_config/
COPY storage/ /app/storage/

# ── 最终镜像：一个容器、一个入口、三个子进程由 chatbot.py all 管理 ──
FROM base AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN ["apt-get","update"]
RUN ["apt-get","install","-y","--no-install-recommends","libunwind8"]
COPY --from=pydeps /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=pydeps /usr/local/bin/ /usr/local/bin/
COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant
COPY --from=ollama /usr/bin/ollama /usr/local/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/local/lib/ollama
RUN ["test","-x","/usr/local/lib/ollama/llama-server"]
COPY --from=fetchmodels /models/bge-m3 /app/models/bge-m3
COPY --from=fetchmodels /models/bge-reranker-v2-m3 /app/models/bge-reranker-v2-m3
COPY --from=assets /app/ /app/
# 生成模型：直接把官方 GGUF 放进 ollama_models；容器启动后由 all 执行 ollama create。
COPY --from=fetchmodels /models/_gguf/MiniCPM5-2B-Q4_K_M.gguf /app/ollama_models/MiniCPM5-2B-Q4_K_M.gguf
RUN ["python","-c","import os,sys; p='/app/ollama_models/MiniCPM5-2B-Q4_K_M.gguf'; n=os.path.getsize(p); print('GGUF bytes:', n); sys.exit(0 if n > 100000000 else 1)"]
COPY chatbot.py /app/chatbot.py
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD ["python","-c","import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3)"]
ENTRYPOINT ["python","chatbot.py","all"]
