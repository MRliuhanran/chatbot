# 单镜像全栈：Qdrant + Ollama(CPU) + RAG 应用（魔搭创空间 Docker 类型用，端口 7860）
#
# ★ 最小化策略（2026-09-27）★
#   1) 服务器可下载的一律不打包：Ollama 生成模型、bge 嵌入/重排权重都由 entrypoint
#      在容器启动时按需下载（见 entrypoint.sh；models/ 已在 .dockerignore 排除）。
#   2) Ollama 自带 CUDA/ROCm 运行库在无 GPU 服务器上是纯浪费，构建时删除（约省 2.1G）。
#   3) 本地数据全部打进镜像：.env / artifacts / qdrant_storage / corpus / lexicon，
#      开箱即用、无需在服务器重建（2 核重建索引远超 73 分钟）。
#   4) 权限用 COPY --chmod 与一个只改目录的小 RUN 设定，避免旧写法里 chmod -R 产生的
#      整份文件复制层（旧构建该层 ~1.2G）。
#
# 环境一致性：
#   - requirements.txt 全 == 锁定本地现役版本；基础镜像 python:3.10（本地 3.10.0，
#     slim 官方只有 3.10 系列最新 patch，patch 差异见 requirements.txt 头注释）。
#   - 本地 macOS 的 torch 是 MPS 版，镜像走 CPU 专用索引装 +cpu 版；这是操作系统差异，
#     由 chatbot.py 的 get_device()（MPS→CUDA→CPU）自动对齐，无需也不应统一。
#
# 构建上下文：必须含 .env / artifacts / qdrant_storage / corpus（被 .gitignore 忽略，
#   推送前需 git add -f）。models/ 不再需要进入上下文。
#
# 本地构建: docker build --platform linux/arm64 -t rag-chatbot .   （M 系 Mac）
#           docker build --platform linux/amd64 -t rag-chatbot .   （x86 / 服务器）
# 本地运行: docker run --rm -p 7860:7860 \
#             -e RAG_SKIP_MODEL_PULL=1 -e OLLAMA_MODELS=/root/.ollama/models \
#             -v ~/.ollama/models:/root/.ollama/models:ro \
#             -v "$PWD/models:/app/models:ro" rag-chatbot
#           （挂载本地模型仅为离线自测；不挂载则启动时自动下载）

FROM python:3.10-slim-bookworm

ARG TARGETARCH
ARG QDRANT_VERSION=v1.19.1
ARG OLLAMA_VERSION=0.34.4
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_FALLBACK_INDEX=https://mirrors.aliyun.com/pypi/simple

# 系统依赖单独成层（这个层很慢但很稳定，与下面的下载分层，改下载逻辑不必重装 apt）
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl zstd procps musl; \
    rm -rf /var/lib/apt/lists/*

# Qdrant + Ollama(CPU)：
# - Qdrant 两架构统一用官方 musl 静态版：gnu 版链接 glibc≥2.38，bookworm 基座只有 2.36
#   （实测 amd64 gnu 版起不来：GLIBC_2.38 not found），musl 静态版不挑发行版；
# - Ollama 只发 .tar.zst（.tgz 已 404），解压需要 zstd；
# - 删掉 Ollama 的 CUDA/ROCm 运行库：CPU 服务器只用 libggml-cpu-*，删后 ollama 仍可用；
# - 多源下载：直连 GitHub 常被限速到 <100KB/s，低于 200KB/s 持续 30s 即判慢，自动改走
#   gh-proxy.com 镜像（只是加前缀，真实资产仍在 GitHub）。
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) _qarch="x86_64-unknown-linux-musl" ;; \
      arm64) _qarch="aarch64-unknown-linux-musl" ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    _dl() { \
      _out="$1"; shift; \
      for _u in "$@"; do \
        echo "== 下载 ${_u}"; \
        if curl -fSL --http1.1 --connect-timeout 20 --speed-limit 204800 --speed-time 30 "${_u}" -o "${_out}"; then \
          return 0; \
        fi; \
        rm -f "${_out}"; \
      done; \
      echo "all sources failed" >&2; return 1; \
    }; \
    _dl /tmp/qdrant.tar.gz \
        "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-${_qarch}.tar.gz" \
        "https://gh-proxy.com/https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-${_qarch}.tar.gz"; \
    mkdir -p /opt/qdrant; \
    tar -xzf /tmp/qdrant.tar.gz -C /opt/qdrant; \
    rm -f /tmp/qdrant.tar.gz; \
    chmod +x /opt/qdrant/qdrant; \
    test -x /opt/qdrant/qdrant; \
    _dl /tmp/ollama.tar.zst \
        "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-${TARGETARCH}.tar.zst" \
        "https://gh-proxy.com/https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-${TARGETARCH}.tar.zst" \
        "https://ollama.com/download/ollama-linux-${TARGETARCH}.tar.zst"; \
    tar --zstd -xf /tmp/ollama.tar.zst -C /usr/local; \
    rm -f /tmp/ollama.tar.zst; \
    rm -rf /usr/local/lib/ollama/cuda_* /usr/local/lib/ollama/rocm*; \
    du -sh /usr/local/lib/ollama; \
    ollama --version

WORKDIR /app

# 依赖分两步：先装 CPU 版 torch —— PyPI 默认 linux wheel 会连带拉数 GB 的 CUDA 依赖。
# requirements 里的 torch==2.7.1 此时已被 2.7.1+cpu 满足（PEP 440：无 local 标记的
# == 比较会忽略候选的 local 标记），不会被重装。装完 pip check 自检，再清字节码与缓存。
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --index-url "${TORCH_INDEX_URL}" \
        --extra-index-url "${PIP_INDEX_URL}" \
        --extra-index-url "${PIP_FALLBACK_INDEX}" \
        "torch==2.7.1+cpu" \
    && pip install --no-cache-dir \
        --index-url "${PIP_INDEX_URL}" \
        --extra-index-url "${PIP_FALLBACK_INDEX}" \
        -r requirements.txt \
    && pip check \
    && find /usr/local/lib/python3.10/site-packages -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && rm -rf /root/.cache

# 本地数据全量进镜像：--chmod=0777 让平台以非 root 跑时也能写 qdrant_storage/artifacts/rag.log，
# 且不额外产生整份文件的 chmod 层。models/ 已排除，由 entrypoint 运行时下载。
COPY --chmod=0777 . .

# 只改目录属性（不复制文件，层极小）：预建运行期可写目录，并确保入口可执行。
RUN chmod 0777 /app \
    && mkdir -p /app/models /app/ollama_models \
    && chmod 0777 /app/models /app/ollama_models \
    && chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
