# ── 单容器 ModelScope 镜像（无 compose、无 shell 脚本）──────────────────
# 一个容器里跑 Qdrant + Ollama + Streamlit，入口是 chatbot.py 的 all 命令；
# 与本地源码运行共用同一份 .env / models / artifacts / storage，功能与效果同源。
# 所有 RUN 使用 exec form，不经过 /bin/sh；不 COPY 任何 shell 脚本。
#
# 构建:
#   docker build -t rag-chatbot:modelscope .
# 运行:
#   docker run --rm -p 7860:7860 rag-chatbot:modelscope
ARG PYTHON_VERSION=3.10
ARG HF_ENDPOINT=https://hf-mirror.com
ARG OLLAMA_GGUF_URL=https://modelscope.cn/models/OpenBMB/MiniCPM5-2B-gguf/resolve/master/MiniCPM5-2B-Q4_K_M.gguf

# ── 直接复用官方镜像里的二进制，避免 RUN 下载/解压 ──
FROM qdrant/qdrant:v1.19.1 AS qdrant
FROM ollama/ollama:0.34.4 AS ollama

# ── 依赖 stage：完整 python 镜像自带编译工具，供 llama-cpp-python 编译 ──
FROM python:${PYTHON_VERSION} AS pydeps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_ARGS="-DGGML_OPENMP=OFF"
COPY requirements.txt /tmp/requirements.txt
RUN ["pip","install","--no-cache-dir","cmake","ninja"]
RUN ["pip","install","--no-cache-dir","--index-url","https://download.pytorch.org/whl/cpu","torch==2.7.1"]
RUN ["pip","install","--no-cache-dir","-r","/tmp/requirements.txt"]

# ── 检索模型 stage：不依赖 git 里是否有 models/，构建期从 HF 镜像下载 ──
# 与 pydeps / assets 并行；产物目录与本地 models/ 同名，指纹保持一致。
FROM python:${PYTHON_VERSION} AS fetchmodels
ENV HF_ENDPOINT=${HF_ENDPOINT}
RUN ["pip","install","--no-cache-dir","huggingface_hub==0.33.2"]
RUN ["python","-c","from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3', local_dir='/models/bge-m3', local_dir_use_symlinks=False); snapshot_download(repo_id='BAAI/bge-reranker-v2-m3', local_dir='/models/bge-reranker-v2-m3', local_dir_use_symlinks=False)"]

# ── 资产 stage：与 pydeps 并行；源码/产物/配置 ──
# ollama_models 必须在构建上下文里（ModelScope 需 LFS/持久盘；本地可从 ~/.ollama/models 复制）。
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
FROM python:${PYTHON_VERSION} AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_DATA_DIR=corpus \
    RAG_ARTIFACTS_DIR=./artifacts \
    RAG_LEXICON_DIR=./lexicon \
    RAG_HTTP_PORT=7860 \
    OLLAMA_MODELS=/app/ollama_models
WORKDIR /app
RUN ["apt-get","update"]
RUN ["apt-get","install","-y","--no-install-recommends","libunwind8"]
COPY --from=pydeps /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=pydeps /usr/local/bin/ /usr/local/bin/
COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant
COPY --from=ollama /usr/bin/ollama /usr/local/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama
COPY --from=fetchmodels /models /app/models
COPY --from=assets /app/ /app/
# 生成模型：直接把官方 GGUF 放进 ollama_models；容器启动后由 all 执行 ollama create。
ADD ${OLLAMA_GGUF_URL} /app/ollama_models/MiniCPM5-2B-Q4_K_M.gguf
COPY chatbot.py /app/chatbot.py
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD ["python","-c","import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3)"]
ENTRYPOINT ["python","chatbot.py","all"]
