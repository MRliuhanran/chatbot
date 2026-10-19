# 单镜像全栈：Qdrant + Ollama(CPU) + RAG 应用（魔搭创空间 Docker 类型用，端口 7860）
#
# ★ 最小化策略（2026-09-27）★
#   1) 模型默认在**构建期烘焙**进镜像（BAKE_MODELS/BAKE_OLLAMA，见下方 modelbake 阶段）：
#      容器可写层不持久，运行期下载=每次重启重下数 GB、且"构建完"并不"能用"。置 0 退回运行期
#      按需下载（镜像回到 ~2.5G，但首启要等下载）。清单唯一来源 tools/fetch_models.sh。
#   2) Ollama 自带 CUDA/ROCm 运行库在无 GPU 服务器上是纯浪费，构建时删除（约省 2.1G）。
#   3) 本地数据全部打进镜像：.env / artifacts / storage / corpus / lexicon，
#      开箱即用、无需在服务器重建（2 核重建索引远超 73 分钟）。
#   4) 权限用 COPY --chmod 与一个只改目录的小 RUN 设定，避免旧写法里 chmod -R 产生的
#      整份文件复制层（旧构建该层 ~1.2G）。
#
# ★ 构建压榨（2026-10-16）★ —— 只改「怎么构建」，不改「构建出什么」
#   A) ENV 全部退场：原先 OLLAMA_KV_CACHE_TYPE/FLASH_ATTENTION/NUM_PARALLEL 站在任何模型层
#      **之前**，改一个数值就让其后全部层失效、重下 7.7G（实测每次部署约 10 分钟）；后来
#      又把 MALLOC_ARENA_MAX / OMP / MKL / OPENBLAS 也写成死值。现在**本文件一个 ENV 都不再
#      承载这些旋钮**，改为按性质分两类、各自单一来源（见下方"服务端参数"）：
#        · 语义项（KV 量化 / flash attention / 槽位并发）→ .env（定义行为，与机器无关）
#        · 硬件项（线程数 / arena / 并行槽）→ tools/hw_profile.py 按 cgroup 探测**派生**
#      收益：改任一项都只失效元数据层级的层，不碰 6.8G 模型层；且 8 核 Mac 与 2vCPU 容器
#      各自算出各自最优值（写死必然有一边是错的）。
#   B) 三段并行烘焙：bge-m3 / bge-reranker-v2-m3 / ollama 生成模型在**同一个 RUN** 里后台
#      同时跑、最后一次 wait 收口 → 关键路径从 sum(2.1G, 2.1G, 2.6G) 变 max(...)。
#      ⚠️ 为什么不直接合成一个 RUN：一个 RUN = 一层，三段合一 = ~6.8G 单层，而 2026-10-16
#      两次实测推镜像正是在 5.95G 单层上卡死 >75 分钟（无一层完成、无错误）。故改用
#      **builder 阶段**并行下载，再由最终阶段 COPY --from 拆成 embed/rerank/ollama 三层推
#      （各 ~2.1G，与今天的层大小一致）。代价：构建期多占 ~6.8G 磁盘，故先做 df 预检。
#      失败语义不变：任一失败 → 非零退出 → 整层失败；ollama pull 的 3 次重试与失败日志
#      转印保持原样（都在 tools/fetch_models.sh 里，本文件一行不动它）。
#   C) COPY 显式化：COPY --chmod=0777 . . 改成逐路径 COPY + 逐路径 --chmod。
#      **文件集合与今天逐条相同**（下方 COPY 段有完整清单与核对方式），但：
#      · 改一个文件不再让 476MB 上下文整块重建（今天改 chatbot.py 会整块重来）；
#      · 不再给全部数据文件 0777，只给运行期真的需要写的两棵树 0777。
#      代价：新增构建上下文里的顶层条目时，必须同步往下面 COPY 清单里加一行。
#   D) 同层清理 + 度量：apt lists / pip+uv 缓存 / .part 暂存 / /tmp/*.log 都在**创建它们的
#      同一个 RUN** 里删；每个阶段打印耗时与 du -sh，构建日志可直接判读关键路径。
#
# ★ 服务端参数：两边同源、且按性质分两类（2026-10-16 统一）★
#   过去是**两套注入机制** —— 本地 launchctl setenv、容器 Dockerfile ENV；效果相同、来源不同，
#   改一处必漏另一处。现在两条链，本地与容器**走的是同一份实现**：
#     · 语义项（KV_CACHE_TYPE / FLASH_ATTENTION / NUM_PARALLEL）—— 定义行为，与核数无关：
#           .env → _load_dotenv（tools/dotenv.sh，唯一实现） → 各自那份 ollama serve
#     · 硬件项（OLLAMA_NUM_THREADS / OMP / MKL / OPENBLAS / MALLOC_ARENA_MAX）
#           tools/hw_profile.py 按 cgroup（v2 cpu.max/memory.max → v1 → nproc/meminfo/sysctl）
#           探测后**派生** → 各自那份 ollama serve / torch（shell 侧只是一层薄封装：
#           eval 它的 --sh 输出，因为导出必须在**当前 shell** 生效才能传给子进程）
#   本地：tools/run_local.sh 与 tools/run_ollama.sh（纯原生）；容器：entrypoint.sh。
#   三者都**先 _load_dotenv 再 _load_hw_profile**（显式覆盖优先），并各自打印一行 [hw] 审计。
#   entrypoint 对 KV_CACHE_TYPE 为空做了 fail-closed：宁可起不来，也不静默退回 f16 KV 把 8G
#   档 OOM 掉；对 hw_profile.sh 缺失、或 _load_hw_profile 返回非 0，同样 fail-closed
#   （线程/arena 静默偏离是同一类风险）。
#
# 环境一致性：
#   - requirements.txt 全 == 锁定本地现役版本；基础镜像 python:3.10（本地 3.10.0，
#     slim 官方只有 3.10 系列最新 patch，patch 差异见 requirements.txt 头注释）。
#   - 本地 macOS 的 torch 是 MPS 版，镜像走 CPU 专用索引装 +cpu 版；这是操作系统差异，
#     由 chatbot.py 的 get_device()（MPS→CUDA→CPU）自动对齐，无需也不应统一。
#
# 构建上下文：必须含 .env / artifacts / storage / corpus（被 .gitignore 忽略，
#   推送前需 git add -f）。models/ 不进入上下文（.dockerignore），由 modelbake 阶段下载。
#
# 本地构建: docker build --platform linux/arm64 -t rag-chatbot .   （M 系 Mac）
#           docker build --platform linux/amd64 -t rag-chatbot .   （x86 / 服务器）
# 本地运行: docker run --rm -p 7860:7860 \
#             -e RAG_SKIP_MODEL_PULL=1 -e OLLAMA_MODELS=/root/.ollama/models \
#             -v ~/.ollama/models:/root/.ollama/models:ro \
#             -v "$PWD/models:/app/models:ro" rag-chatbot
#           （挂载本地模型仅为离线自测；不挂载则启动时自动下载）

# ── 生成侧精确分词 wheel：源码编译 llama-cpp-python 的 GLIBC 版（独立阶段，不进最终镜像）──
# 为什么必须自编：llama-cpp-python **0.3.x 全系列都没有 glibc wheel** —— PyPI 只有 sdist，
# abetlen 的 cpu 索引与 GitHub Release 只发 musl 版（linux_x86_64，DT_NEEDED=libc.musl-x86_64.so.1）。
# 在 python:3.10-slim-bookworm（glibc）上加载该 musl wheel 直接失败：
#   ⚠️ llama.cpp 分词不可用 … libc.musl-x86_64.so.1: cannot open shared object file
#   → 生成侧精确分词**静默**退化成 tokenizer.json（2026-10-17 云端运行日志实测，违反"本地=云端"）。
# 修法：用 build-essential+cmake 从 sdist 编译出 glibc wheel（实测 NEEDED=libc.so.6 /
# libstdc++.so.6 / libgcc_s.so.1，RUNPATH=$ORIGIN → 自带 libggml*/libllama.so，不依赖系统额外库）。
# 只产出 wheel，编译器不进最终镜像；版本**从 requirements.txt 现取**，避免与 pin 漂移（取不到即报错）。
# 本阶段较慢，但输入仅 requirements.txt —— 层缓存命中即可整段跳过（不会拖慢日常重建）。
FROM python:3.10-slim-bookworm AS lcpwheel
# 与 appbase 同名 ARG 的默认值保持一致（本阶段在其之前，无法继承 appbase 的 ARG）。
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_FALLBACK_INDEX=https://repo.huaweicloud.com/repository/pypi/simple
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN set -eux; \
    _v="$(sed -nE 's/^llama-cpp-python==([^[:space:]]+).*$/\1/p' requirements.txt | head -n1)"; \
    [ -n "${_v}" ] || { echo "[lcpwheel] ❌ requirements.txt 未找到 llama-cpp-python==<版本>"; exit 1; }; \
    echo "[lcpwheel] 从源码编译 llama-cpp-python==${_v}（glibc）"; \
    pip wheel --no-cache-dir --no-deps --wheel-dir /lcpwheel \
        --index-url "${PIP_INDEX_URL}" --extra-index-url "${PIP_FALLBACK_INDEX}" \
        "llama-cpp-python==${_v}"; \
    ls -l /lcpwheel

# ── 基座与依赖：appbase 同时是 modelbake 与最终镜像的父层（同一份 apt/pip/ollama 只构建一次）
FROM python:3.10-slim-bookworm AS appbase

# stdout 不缓冲：非 logging 的 print（jieba 初始化行等）在容器里默认块缓冲，
# 崩溃/OOM 时最后一批输出整段丢失，而运行日志是唯一的现场。
# OLLAMA_MODELS 是**路径**不是旋钮（值稳定、且 modelbake 阶段要用），故留在最前；
# 它变化也会让后续层失效，但它只随目录布局变，不随调参变。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_MODELS=/app/ollama_models

ARG TARGETARCH
ARG QDRANT_VERSION=v1.19.1
ARG OLLAMA_VERSION=0.34.4
# uv 版本固定到本机现役版，避免构建工具本身漂移。
ARG UV_VERSION=0.12.13
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# 兜底源不能再用阿里云：其 PEP 658 元数据文件（*.whl.metadata）对部分包返回 404，
# 而 pip 一旦选中该候选会直接失败、不会回退到其它索引（本地 pip 26.1.2 可复现）。
# 2026-10-10 魔塔构建正是因 streamlit==1.62.0 的 metadata 404 中断经验证。华为云镜像
# 元数据完整且在国内，速度与可靠性兼得；若它再滞后可改官方 https://pypi.org/simple。
ARG PIP_FALLBACK_INDEX=https://repo.huaweicloud.com/repository/pypi/simple

# 系统依赖单独成层（这个层很慢但很稳定，与下面的下载分层，改下载逻辑不必重装 apt）
# apt lists 在**同一个 RUN** 里删：换层删只减可见性、不减体积。
RUN set -eux; \
    _t0=$(date +%s); \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl zstd procps musl; \
    rm -rf /var/lib/apt/lists/*; \
    echo "[timing] apt 系统依赖 $(( $(date +%s) - _t0 ))s（apt lists 已同层删除）"

# Qdrant + Ollama(CPU)：
# - Qdrant 两架构统一用官方 musl 静态版：gnu 版链接 glibc≥2.38，bookworm 基座只有 2.36
#   （实测 amd64 gnu 版起不来：GLIBC_2.38 not found），musl 静态版不挑发行版；
# - Ollama 只发 .tar.zst（.tgz 已 404），解压需要 zstd；
# - 删掉 Ollama 的 CUDA/ROCm 运行库：CPU 服务器只用 libggml-cpu-*，删后 ollama 仍可用；
# - 多源下载：每个源失败/超时后自动换下一个（gh-proxy.com 只是加前缀，真实资产仍在 GitHub）。
#   必须带 --max-time：curl 的 --speed-time 在「连接已建立但收不到任何字节」时不会生效，
#   ollama 是 ~1.4G 的大资产，一旦卡在 0 字节会无限期挂住整个构建（2026-10-10 魔塔构建
#   卡在第 7 层 16 分钟、日志停在 0 字节，即此因）。大资产优先 gh-proxy（实测 9MB/s），
#   直连 GitHub 实测卡死、仅作兜底；qdrant 小资产直连更快，顺序不变。
RUN set -eux; \
    _t0=$(date +%s); \
    case "${TARGETARCH}" in \
      amd64) _qarch="x86_64-unknown-linux-musl" ;; \
      arm64) _qarch="aarch64-unknown-linux-musl" ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    _dl() { \
      _out="$1"; shift; \
      for _u in "$@"; do \
        echo "== 下载 ${_u}"; \
        if curl -fSL --http1.1 --connect-timeout 20 --max-time 420 --speed-limit 51200 --speed-time 60 "${_u}" -o "${_out}"; then \
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
        "https://gh-proxy.com/https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-${TARGETARCH}.tar.zst" \
        "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-${TARGETARCH}.tar.zst" \
        "https://ollama.com/download/ollama-linux-${TARGETARCH}.tar.zst"; \
    tar --zstd -xf /tmp/ollama.tar.zst -C /usr/local; \
    rm -f /tmp/ollama.tar.zst; \
    rm -rf /usr/local/lib/ollama/cuda_* /usr/local/lib/ollama/rocm*; \
    du -sh /usr/local/lib/ollama; \
    ollama --version; \
    echo "[timing] Qdrant+Ollama 安装 $(( $(date +%s) - _t0 ))s"

WORKDIR /app

# 依赖分两步：先装 CPU 版 torch —— PyPI 默认 linux wheel 会连带拉数 GB 的 CUDA 依赖。
# requirements 里的 torch==2.7.1 此时已被 2.7.1+cpu 满足（PEP 440：无 local 标记的
# == 比较会忽略候选的 local 标记），不会被重装。装完 uv pip check 自检，再清字节码与缓存。
#
# pip → uv：--index-url/--extra-index-url/--find-links 语义一致，但解析与下载并发度远高，
# 2vCPU 构建机上这一阶段明显更快（依赖已全 == 锁定，解析量本就接近零）。uv 用 --system
# 装进系统 site-packages（官方 python 镜像无 venv、也无 EXTERNALLY-MANAGED 标记），
# --break-system-packages 仅为将来基座变化兜底；--no-cache 与末尾 rm -rf /root/.cache 呼应。
# uv 本身也走主源+兜底源，避免镜像滞后导致构建工具装不上。
COPY requirements.txt .
COPY wheels/ /tmp/wheels/
# 生成侧精确分词的 glibc wheel 由文件顶部的 lcpwheel 阶段源码编译产出，放进同一个 find-links 目录；
# wheels/ 里**不再放** llama-cpp-python 的预编译 wheel（0.3.x 的官方预编译只有 musl 版，在 bookworm 上不可用）。
COPY --from=lcpwheel /lcpwheel/ /tmp/wheels/
RUN set -eux; \
    _t0=$(date +%s); \
    pip install --no-cache-dir \
        --index-url "${PIP_INDEX_URL}" \
        --extra-index-url "${PIP_FALLBACK_INDEX}" \
        "uv==${UV_VERSION}" \
    && uv pip install --system --break-system-packages --no-cache \
        --index-strategy unsafe-best-match \
        --index-url "${TORCH_INDEX_URL}" \
        --extra-index-url "${PIP_INDEX_URL}" \
        --extra-index-url "${PIP_FALLBACK_INDEX}" \
        "torch==2.7.1+cpu" \
    && uv pip install --system --break-system-packages --no-cache \
        --index-strategy unsafe-best-match \
        --find-links /tmp/wheels \
        --index-url "${PIP_INDEX_URL}" \
        --extra-index-url "${PIP_FALLBACK_INDEX}" \
        --only-binary=llama-cpp-python \
        -r requirements.txt \
    && uv pip check --system \
    && find /usr/local/lib/python3.10/site-packages -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && rm -rf /root/.cache \
    && echo "[timing] Python 依赖 $(( $(date +%s) - _t0 ))s（pip/uv 缓存与 __pycache__ 已同层删除）"

# 检索模型服务（infinity）**不装进本镜像**：魔搭是 8G 单容器，运行档定为
# RAG_EMBED_BACKEND/RAG_RERANK_BACKEND = torch（进程内），entrypoint 仅在 =http 时才拉起
# infinity，故这层对云端是纯负担；且 infinity-emb[server]==0.0.77 要求 uvicorn<0.33.0，
# 与 requirements.txt 锁定的 uvicorn==0.48.0 直接冲突，带 --constraint 的安装**必然构建失败**
# （2026-10-15 构建实测）。compose 路径仍用独立的 embed.Dockerfile（不带本仓约束、可解析）。

# ⚠️ 必须先把 RUN 的 shell 换成 bash：Docker 默认 /bin/sh，Debian 上是 dash，而
# tools/fetch_models.sh 用了 bash 专有语法（declare -F / 数组 local -a / ${!arr[@]}），
# dash 在**解析阶段**就报 `Syntax error: "(" unexpected`，整层从未执行
# （2026-10-10 构建实测的失败点）。只影响其后的 RUN；entrypoint.sh 自带 bash shebang。
# 放在 appbase 里 → modelbake 与最终阶段一起继承。
SHELL ["/bin/bash", "-c"]

# ── 构建期烘焙模型（"构建完就能用"）────────────────────────────────────────────
# 容器的可写层不跨重启持久：模型若在运行期下载，每次冷启都要重下数 GB，且"构建完成"
# 并不等于"可服务"。烘焙进镜像层后，entrypoint 的清单判据直接判"已就绪"，启动即用，
# 运行日志里不再有下载过程。清单与实现唯一来源 tools/fetch_models.sh（与运行期共用，
# 避免两处清单漂移）；BAKE_MODELS / BAKE_OLLAMA 任一置 0 即可关掉对应烘焙、退回运行期下载。
#
# 本阶段**只下载**，不产出镜像层：它把三份模型放 /bake/{embed,rerank,ollama_models}，
# 由最终阶段 COPY --from 拆成三层（见文件头 B 条）。
FROM appbase AS modelbake

# SHELL / WORKDIR 在 FROM <stage> 时按 Docker 语义继承自 appbase，但本机无法实跑构建验证，
# 故**显式重申**而不是依赖隐含继承：SHELL 决定本阶段能不能用 bash 数组/declare（fetch_models.sh
# 必需），WORKDIR 决定 COPY .env ./ 落到 /app 与 _load_dotenv 读的 ./.env 是不是同一份。
SHELL ["/bin/bash", "-c"]
WORKDIR /app

# ⚠️ 这里**有意不 COPY 整个 .env**，而是用只含"模型选择键"的 tools/bake.env 顶替成 .env：
# modelbake 只关心"下哪些模型"（MODEL / RAG_*_MODEL_PATH / RAG_GEN_MODEL_MODELSCOPE /
# RAG_OLLAMA_DL_CONNS）。COPY 整个 .env 会让任何业务配置（RAG_THINK / RAG_NUM_CTX…）
# 一变就使烘焙层缓存失效、重下 ~7.7G；改用 bake.env 后，业务配置改动命中缓存、整段跳过。
# 一致性由下方 app 阶段的断言兜底（.env ↔ tools/bake.env 逐键比对，不一致即构建失败）。
COPY tools/bake.env ./.env
# dotenv.sh 与 fetch_models.sh 一起进构建期：Ollama 服务端参数在构建期也走**同一个**
# _load_dotenv，与运行期逐字同源（避免"构建期一套、运行期一套"的第二条注入路径）。
COPY tools/fetch_models.sh tools/dotenv.sh ./tools/

ARG BAKE_MODELS=1
ARG BAKE_OLLAMA=1
# 磁盘预检下限（GB）：并行烘焙同时容纳"下载区 + 最终镜像层"两份，低于下限直接判失败
# （半途 ENOSPC 只会产出更难归因的残层）。构建机磁盘紧时：清缓存，或置 BAKE_*=0。
ARG BAKE_MIN_FREE_GB=8
# 体积下限（MB）：清单齐备但明显偏小 = 半截下载。取值远低于实测（bge-m3 ≈2.1G、
# rerank ≈2.1G、ollama Q4_K_M：qwen3.5-4b ≈2.5G / minicpm5-2b ≈1.5G），只挡"静默小文件"。
# 换成更小的生成模型时必须同步下调本下限，否则校验会把完整下载误判成"半截下载"。
ARG BAKE_MIN_EMBED_MB=1500
ARG BAKE_MIN_RERANK_MB=1500
ARG BAKE_MIN_OLLAMA_MB=1200

RUN set -eux; \
    _T_ALL=$(date +%s); \
    mkdir -p /bake/embed /bake/rerank /bake/ollama_models; \
    echo "[bake] ==================== 磁盘预检（要下 ~7.7G）===================="; \
    df -h / /bake; \
    _free=$(df -Pk /bake | awk 'NR==2{printf "%d", $4/1048576}'); \
    echo "[bake] /bake 可用 ${_free}G（下限 BAKE_MIN_FREE_GB=${BAKE_MIN_FREE_GB}G）"; \
    if [ "${_free}" -lt "${BAKE_MIN_FREE_GB}" ]; then \
        echo "[bake] ❌ 可用空间 ${_free}G < ${BAKE_MIN_FREE_GB}G：并行烘焙需同时容纳下载区与最终镜像层" >&2; \
        echo "[bake]    出路：① 清理构建机磁盘 ② BAKE_MODELS=0 / BAKE_OLLAMA=0 退回运行期下载。" >&2; \
        exit 1; \
    fi; \
    . ./tools/dotenv.sh; \
    _load_dotenv; \
    . ./tools/fetch_models.sh; \
    echo "[bake] 检索模型（来自 tools/bake.env）：embed=$(basename "$(_env_get RAG_EMBED_MODEL_PATH)") rerank=$(basename "$(_env_get RAG_RERANK_MODEL_PATH)") 生成模型=$(_env_get MODEL)"; \
    echo "[bake] 生成侧语义参数：本层用的是 tools/bake.env（只含模型选择键），故 KV_CACHE_TYPE/FLASH_ATTENTION/NUM_PARALLEL 之类的 OLLAMA_* 在此为未设 —— 它们只影响**运行期**服务，且由最终阶段的 .env 提供（构建期烘焙只看模型，不看服务参数）。"; \
    echo "[bake] ==================== 三段并行下载（最后一次 wait 收口）===================="; \
    _pids=(); _tags=(); \
    if [ "${BAKE_MODELS}" = "1" ]; then \
        ( _t0=$(date +%s); _rc=0; \
          MODELS_DIR=/bake/embed _download_side embed || _rc=$?; \
          echo "[timing] embed(bge-m3) 用时 $(( $(date +%s) - _t0 ))s rc=${_rc}"; \
          exit "${_rc}" ) & \
        _pids+=("$!"); _tags+=("embed"); \
        ( _t0=$(date +%s); _rc=0; \
          MODELS_DIR=/bake/rerank _download_side rerank || _rc=$?; \
          echo "[timing] rerank(bge-reranker-v2-m3) 用时 $(( $(date +%s) - _t0 ))s rc=${_rc}"; \
          exit "${_rc}" ) & \
        _pids+=("$!"); _tags+=("rerank"); \
    fi; \
    if [ "${BAKE_OLLAMA}" = "1" ]; then \
        ( _t0=$(date +%s); _rc=0; \
          OLLAMA_MODELS=/bake/ollama_models _pull_ollama_model "$(_env_get MODEL)" || _rc=$?; \
          echo "[timing] ollama($(_env_get MODEL)) 用时 $(( $(date +%s) - _t0 ))s rc=${_rc}"; \
          exit "${_rc}" ) & \
        _pids+=("$!"); _tags+=("ollama"); \
    fi; \
    _rc=0; \
    for _i in "${!_pids[@]}"; do \
        if wait "${_pids[$_i]}"; then \
            echo "[bake] ✅ ${_tags[$_i]} 完成"; \
        else \
            echo "[bake] ❌ ${_tags[$_i]} 失败（rc≠0；失败原因见上方该段日志，ollama pull 已内建 3 次重试）" >&2; \
            _rc=1; \
        fi; \
    done; \
    if [ "${_rc}" != 0 ]; then \
        echo "[bake] ❌ 有失败项 → 不产出「模型缺失却照样成功」的半成品镜像" >&2; \
        exit 1; \
    fi; \
    echo "[bake] ==================== 清单 / 尺寸校验（清单来源 _model_spec）===================="; \
    _verify_side() { \
        local _side="$1" _root="$2" _min_mb="$3" _name _spec _files _f _bad=0 _mb; \
        case "${_side}" in \
            embed)  _name="$(basename "$(_env_get RAG_EMBED_MODEL_PATH)")" ;; \
            rerank) _name="$(basename "$(_env_get RAG_RERANK_MODEL_PATH)")" ;; \
            *) echo "[verify] ❌ 未知侧 ${_side}" >&2; return 1 ;; \
        esac; \
        _spec="$(_model_spec "${_name}")" || { echo "[verify] ❌ ${_name} 未登记于 _model_spec" >&2; return 1; }; \
        _files="${_spec#*|}"; _files="${_files#*|}"; \
        echo "[verify] ${_name} 清单（tools/fetch_models.sh:_model_spec）："; \
        for _f in ${_files}; do \
            if [ -s "${_root}/${_name}/${_f}" ]; then \
                printf '[verify]   %12s  %s\n' "$(wc -c < "${_root}/${_name}/${_f}")" "${_f}"; \
            else \
                echo "[verify] ❌ 缺失或空文件: ${_name}/${_f}" >&2; _bad=1; \
            fi; \
        done; \
        _mb=$(du -sm "${_root}/${_name}" | awk '{print $1}'); \
        echo "[verify] ${_name} 合计 ${_mb}MB（下限 ${_min_mb}MB）"; \
        if [ "${_mb}" -lt "${_min_mb}" ]; then \
            echo "[verify] ❌ ${_name} 体积 ${_mb}MB < ${_min_mb}MB：疑似半截下载" >&2; _bad=1; \
        fi; \
        [ "${_bad}" = 0 ] || return 1; \
        du -sh "${_root}/${_name}"; \
    }; \
    if [ "${BAKE_MODELS}" = "1" ]; then \
        _verify_side embed  /bake/embed  "${BAKE_MIN_EMBED_MB}"; \
        _verify_side rerank /bake/rerank "${BAKE_MIN_RERANK_MB}"; \
    else \
        echo "[verify] BAKE_MODELS=0：跳过检索模型校验（退回运行期下载，镜像里 /app/models 为空目录）"; \
    fi; \
    if [ "${BAKE_OLLAMA}" = "1" ]; then \
        _mb=$(du -sm /bake/ollama_models/blobs 2>/dev/null | awk '{print $1}'); \
        _mb="${_mb:-0}"; \
        echo "[verify] ollama blobs ${_mb}MB（下限 ${BAKE_MIN_OLLAMA_MB}MB）；manifests："; \
        find /bake/ollama_models/manifests -type f 2>/dev/null || true; \
        if [ "${_mb}" -lt "${BAKE_MIN_OLLAMA_MB}" ]; then \
            echo "[verify] ❌ ollama 模型体积 ${_mb}MB < ${BAKE_MIN_OLLAMA_MB}MB：疑似半截下载" >&2; \
            exit 1; \
        fi; \
    else \
        echo "[verify] BAKE_OLLAMA=0：跳过生成模型校验（退回运行期拉取）"; \
    fi; \
    echo "[bake] ==================== 同层清理（创建者即删除者）===================="; \
    rm -f /tmp/ollama-pull.log /tmp/ollama-bake.log /tmp/model-download.status /tmp/mkdir.log; \
    rm -rf /bake/embed/*.part /bake/rerank/*.part /bake/embed/*.old /bake/rerank/*.old; \
    chmod -R a+rX /bake/embed /bake/rerank /bake/ollama_models; \
    echo "[bake] ==================== 产物汇总 ===================="; \
    du -sh /bake/embed /bake/rerank /bake/ollama_models; \
    echo "[timing] 烘焙总用时 $(( $(date +%s) - _T_ALL ))s（三段并行的墙钟，非各自之和）"

# ── 最终镜像 ─────────────────────────────────────────────────────────────────
# 与 appbase 同一父层：apt/pip/ollama 的层**直接复用**，此处只落模型与数据。
FROM appbase AS app

# 同上：显式重申 WORKDIR（下面 COPY 的相对目标 ./ 与运行期 CWD 都依赖它）。
WORKDIR /app

# 模型三层（每层 ~2.1G）：层大小与今天的「按侧拆三个 RUN」逐层一致，推镜像行为不变；
# 但下载已在 modelbake 里并行完成，墙钟 = max(三段) 而不是 sum。
# 目录名从 .env 推导、写在 modelbake 里，这里按**目录本身**搬（/bake/embed/ 的内容 →
# /app/models/），故 .env 换模型不需要改本文件。
COPY --from=modelbake /bake/embed/         /app/models/
COPY --from=modelbake /bake/rerank/        /app/models/
COPY --from=modelbake /bake/ollama_models/ /app/ollama_models/

# 本地数据全量进镜像。**文件集合与旧写法 COPY --chmod=0777 . . 只差下面点名的 3 个**（旧写法
# 26 个顶层条目、1302 个路径），.dockerignore 仍在生效（artifacts/embeddings、artifacts/docs、
# storage/**/wal/{closed,open}-*、*.log 等照旧被排除）。改动只有三点：
#   ① 权限从"全部 0777"收敛成"按用途给"：运行期要写的两棵树 0777；只读的给 a+rX；
#      根文件 0644；入口脚本 0755（旧写法连 .env/chatbot.py/corpus 都是 0777）。
#      已验证无写入点：corpus/lexicon/tools/wheels/qdrant_config/.streamlit 运行期只读
#      （chatbot.py 的 RAG_DATA_DIR 只 glob+open 读，RAG_LEXICON_DIR 只读，artifacts 才是写目标）。
#   ② 逐路径 COPY：改一个文件只重建它那一层，不再整块重来 476MB。
#   ③ **有意不复制 3 个宿主遗留文件**：.qdrant-initialized / .qdrant.pid / .streamlit.pid。
#      它们是宿主运行期的残留（.qdrant.pid 只有 tools/run_qdrant.sh 读、.streamlit.pid 只有
#      tools/run_local.sh 读），容器内没有任何读取方；旧写法靠 COPY . . 顺手带进去，属噪声。
# ⚠️ **决定构建上下文的是 .dockerignore，不是 .gitignore** —— 这两份名单不同，别混：
#    上面那 3 个文件在 .gitignore 里（`*.pid` / `.qdrant-initialized`）**但 .dockerignore 里
#    没有任何规则**，所以它们**确实进得了上下文**（2026-10-16 用真实 docker build 探针
#    （FROM python:3.10-slim-bookworm + COPY . /ctx/）拉出的上下文清单里就有这三项，各 0~5 字节）。
#    换句话说：把不存在的文件列进 COPY 才会 `failed to calculate checksum ... "not found"`。
#    现在不列它们，是为了**不再依赖 .dockerignore 永远保持宽松** —— 将来若有人照 .gitignore
#    给 .dockerignore 补一条 `*.pid`，列在 COPY 里的它们就会立刻 kill 构建。
# ⚠️ 往构建上下文加新顶层条目时必须同步加到这里，否则该文件不进镜像（旧写法是自动带上）；
#    但只列 .dockerignore **没有排除**的文件。判断口径：`git check-ignore` 看的是 .gitignore，
#    对这里没有意义 —— 要看 .dockerignore 里有没有匹配的 pattern。
#    .infinity_cache/ 是空目录（COPY 空目录的语义在各 Docker 版本上不一致），故改由下方 RUN mkdir。
COPY --chmod=0644 \
     .dockerignore .env \
     Dockerfile README.md chatbot.py \
     docker-compose.gpu.yml docker-compose.yml embed.Dockerfile \
     ms_deploy.json pyproject.toml requirements-dev.txt requirements.txt \
     ./
COPY --chmod=0755 entrypoint.sh ./
COPY --chmod=0755 .devcontainer/  ./.devcontainer/
COPY --chmod=0755 .streamlit/     ./.streamlit/
COPY --chmod=0755 corpus/         ./corpus/
COPY --chmod=0755 lexicon/        ./lexicon/
COPY --chmod=0755 qdrant_config/  ./qdrant_config/
COPY --chmod=0755 tools/          ./tools/
COPY --chmod=0755 wheels/         ./wheels/
COPY --chmod=0777 artifacts/      ./artifacts/
COPY --chmod=0777 storage/ ./storage/

# 模型清单一致性断言（.env ↔ tools/bake.env）─────────────────────────────────────
# modelbake 只 COPY tools/bake.env（为的是业务配置改动不打穿烘焙层缓存），而运行期读 .env。
# 二者对"模型选择键"必须逐键一致，否则会出现"构建期烘焙 A、运行期按 .env 加载 B"。
# 这里在最终镜像里把它们比一遍：不一致即构建失败（响亮，不静默）。用的 _env_get 与
# 构建/运行期**同一解析器**（tools/fetch_models.sh），不另造一份 .env 解析。
RUN set -eux; \
    . ./tools/fetch_models.sh; \
    for _k in MODEL RAG_GEN_MODEL_MODELSCOPE RAG_OLLAMA_DL_CONNS RAG_EMBED_MODEL_PATH RAG_RERANK_MODEL_PATH; do \
        _a="$(_env_get "${_k}" .env)"; \
        _b="$(_env_get "${_k}" tools/bake.env)"; \
        if [ "${_a}" != "${_b}" ]; then \
            echo "[check] ❌ 模型清单漂移：${_k} 在 .env='${_a}' 与 tools/bake.env='${_b}' 不一致" >&2; \
            echo "[check]    .env 决定运行期加载、tools/bake.env 决定构建期烘焙 —— 两处必须同步改" >&2; \
            exit 1; \
        fi; \
    done; \
    echo "[check] ✅ 模型清单一致（.env ↔ tools/bake.env）"

# 只改目录属性（不复制文件，层极小）：预建运行期可写目录。**非递归** —— 递归 chmod 会让
# overlay 把每个文件 copy-up 成新层（旧构建 ~1.2G 的"整份文件复制层"就是这么来的）。
# models/ollama_models 由 entrypoint 在运行期按需补下，故这两个目录必须可写。
RUN set -eux; \
    mkdir -p /app/models /app/ollama_models /app/.infinity_cache; \
    chmod 0777 /app /app/models /app/ollama_models /app/.infinity_cache; \
    test -x /app/entrypoint.sh; \
    echo "[layer] /app 顶层："; ls -la /app; \
    du -sh /app/models /app/ollama_models /app/artifacts /app/storage /app/corpus

# 这里**故意没有任何 ENV**：MALLOC_ARENA_MAX / OMP_NUM_THREADS / MKL_NUM_THREADS /
# OPENBLAS_NUM_THREADS / OLLAMA_NUM_THREADS / OLLAMA_NUM_PARALLEL 这些硬件相关旋钮全部由
# tools/hw_profile.py 在**运行期**按真实配额派生（entrypoint.sh 第 0.2b 节经 hw_profile.sh
# 薄封装加载并导出）。
# 为什么不写成 ENV：
#   · 写死就是"同一份镜像只能适配一种机器"——2vCPU 容器与 8 核 Mac 必有一边被拖慢；
#   · ENV 还在**构建期**生效，会让 pip/烘焙阶段也跟着变线程数（与产物无关的抖动）；
#   · 改成派生后，本文件的层不再随这些数值变化，缓存命中率最高。
# ⚠️ 若将来确实要固定某个值：写进 .env（语义项）或用平台环境变量注入（进程环境优先，
#   hw_profile 会尊重它）—— **不要**在这里加回去，那会把"单一来源"重新劈成两条。

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
