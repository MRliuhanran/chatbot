# ── 单容器 ModelScope 镜像（无 compose）──────────────────────────────
# 一个容器里跑 Qdrant + Ollama + Streamlit，入口是 chatbot.py 的 all 命令；
# 与本地源码运行共用同一份 .env / artifacts / storage，功能与效果同源。
# 所有 RUN 使用 exec form；构建期不再下载模型 —— 4.9G 权重（bge-m3 /
# bge-reranker/GGUF）由运行期 cmd_all → _ensure_runtime_models 用仓库唯一的
# start.sh fetch-models 装到持久盘（/mnt/workspace/models），冷启动只下一次。
#
# 构建:
#   docker build -t rag-chatbot:modelscope .
# 运行（本机需先自行准备 ~/.ollama/models 到运行目录，或用持久盘）:
#   docker run --rm -p 7860:7860 rag-chatbot:modelscope
ARG PYTHON_VERSION=3.10

# ── 基础镜像：Debian 源切到国内镜像（构建机访问 deb.debian.org 会超时） ──
FROM python:${PYTHON_VERSION} AS base
RUN ["python","-c","import pathlib; [p.write_text(p.read_text().replace('deb.debian.org','mirrors.aliyun.com').replace('security.debian.org','mirrors.aliyun.com')) for p in [pathlib.Path('/etc/apt/sources.list'),*pathlib.Path('/etc/apt/sources.list.d').glob('*')] if p.is_file()]"]

# ── 直接复用官方镜像里的二进制，避免 RUN 下载/解压 ──
FROM qdrant/qdrant:v1.19.1 AS qdrant
FROM ollama/ollama:0.34.4 AS ollama

# ── 依赖 stage ─────────────────────────────────────────────
# 2026-10-27 起 llama-cpp-python 已下线（见 requirements.txt 注释）：生成 tokenizer
# 主路径是 artifacts/gen_tokenizer（sha 校验），GGUF 仅兜底且缺包可优雅降级 ——
# 依赖全部是 manylinux wheel，不再有源码编译项（cmake/ninja 阶段已删）。
# base=python:3.10 完整镜像保留（torch 运行时要 libgomp1 等，slim 需另配未验证，
# 不值得在这里赌一次镜像换基座）。
FROM base AS pydeps
# ★ 阶段内必须重声明 ARG：全局 ARG 只进 FROM 行；不重声明时 bash 里读到的是
#   基础镜像的 ENV PYTHON_VERSION=3.10.22（完整三元版本）——把它去掉所有点拼出
#   cp31022 的轮子名，pip 当场拒装（2026-10-27 22:01 构建实锤）。重声明后拿到
#   全局默认 3.10 → cp310；将来全局 ARG 改 3.11 这里也会联动。
ARG PYTHON_VERSION=3.10
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple \
    PIP_EXTRA_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10
COPY requirements.txt /tmp/requirements.txt
# 唯一 need-shell 的 RUN：wheel 文件名里的 cpXR 与 ARG PYTHON_VERSION 绑定（此前写死
# cp310 —— 有人改 ARG 就会往不匹配的 Python 装轮子）。ARG 在构建容器里是环境变量，
# bash 形参替换 3.10 → 310；镜像自带的 /bin/bash 可用。
RUN ["bash","-c","exec pip install --no-cache-dir \"https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.7.1%2Bcpu-cp${PYTHON_VERSION//./}-cp${PYTHON_VERSION//./}-manylinux_2_28_x86_64.whl\""]
RUN ["pip","install","--no-cache-dir","-r","/tmp/requirements.txt"]

# ── 资产 stage：与 pydeps 并行；源码/产物/配置 ──
# 模型（4.9G）不烤进镜像：运行期由 cmd_all → _ensure_runtime_models 用仓库唯一
# start.sh fetch-models 装到持久盘（/mnt/workspace/models），冷启动只下一次。
# ollama_models 留空目录占位（本地直跑舒服；正式 GGUF 运行期落到 OLLAMA_MODELS）。
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
RUN ["apt-get","install","-y","--no-install-recommends","libunwind8","aria2","ca-certificates"]
COPY --from=pydeps /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=pydeps /usr/local/bin/ /usr/local/bin/
COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant
COPY --from=ollama /usr/bin/ollama /usr/local/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/local/lib/ollama
RUN ["test","-x","/usr/local/lib/ollama/llama-server"]
COPY --from=assets /app/ /app/
# start.sh 进入镜像：运行期 fetch-models 的唯一下载器（构建期已不下载模型）。
COPY start.sh /app/start.sh
COPY chatbot.py /app/chatbot.py
EXPOSE 7860
# start-period 600s：冷启动（持久盘无权重）要先下载 4.9G 模型再到 7860，120s 的
# 老窗口会把"正在装模型"的健康等待误判成不健康。
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 CMD ["python","-c","import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3)"]
ENTRYPOINT ["python","chatbot.py","all"]
