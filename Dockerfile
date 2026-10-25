# ── 无 shell 的 Dockerfile ─────────────────────────────────────────────
# 约束：
#   · 所有 RUN 使用 exec form（["cmd","arg"]），不经过 /bin/sh；
#   · 不 COPY 任何 shell 脚本，不使用 ENTRYPOINT/CMD 的 shell 形式；
#   · 独立 stage（pydeps / assets）由 BuildKit 默认并行构建；
#   · 模型权重不进镜像，由 compose 以只读卷挂载（避免 6.9G 构建上下文）。
#
# 构建（并行）:
#   docker buildx build --progress=plain -t rag-chatbot:local .
# 或用 compose（并行拉取/构建/启动）:
#   docker compose up -d --build
ARG PYTHON_VERSION=3.10

# ── 依赖 stage：完整 python 镜像自带编译工具，供 llama-cpp-python 源码编译 ──
FROM python:${PYTHON_VERSION} AS pydeps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_ARGS="-DGGML_OPENMP=OFF"
COPY requirements.txt /tmp/requirements.txt
RUN ["pip","install","--no-cache-dir","cmake","ninja"]
RUN ["pip","install","--no-cache-dir","--index-url","https://download.pytorch.org/whl/cpu","torch==2.7.1"]
RUN ["pip","install","--no-cache-dir","-r","/tmp/requirements.txt"]

# ── 资产 stage：与 pydeps 并行；只做 COPY，不执行任何命令 ──
FROM scratch AS assets
COPY corpus/ /app/corpus/
COPY artifacts/ /app/artifacts/
COPY lexicon/ /app/lexicon/
COPY .streamlit/ /app/.streamlit/

# ── 最终镜像：slim + 只装 libgomp1（torch 运行期依赖），不装 shell 脚本 ──
FROM python:${PYTHON_VERSION}-slim-bookworm AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_DATA_DIR=corpus \
    RAG_ARTIFACTS_DIR=./artifacts \
    RAG_LEXICON_DIR=./lexicon
WORKDIR /app
RUN ["apt-get","update"]
RUN ["apt-get","install","-y","--no-install-recommends","libgomp1"]
COPY --from=pydeps /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=pydeps /usr/local/bin/ /usr/local/bin/
COPY --from=assets /app/ /app/
COPY chatbot.py /app/chatbot.py
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 CMD ["python","-c","import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3)"]
ENTRYPOINT ["python","-m","streamlit","run","chatbot.py","--server.address","0.0.0.0","--server.port","7860","--server.headless","true"]
