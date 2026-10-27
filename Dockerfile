# ── 单容器 ModelScope 镜像（无 compose）──────────────────────────────
# 一个容器里跑 Qdrant + Ollama + Streamlit，入口是 chatbot.py 的 all 命令；
# 与本地源码运行共用同一份 .env / artifacts / storage，功能与效果同源。
# 所有 RUN 使用 exec form；模型部署三层各司其职（2026-10-27 最快化定案）：
#   ① 镜像层烤权重（fetchmodels stage，构建机走魔搭 CDN + 组级层缓存命中=零下载）
#      —— Running 后 ~12s 可用，没有冷启动 4.9G 下载；
#   ② 持久盘 /mnt/workspace 暖缓存（ollama blobs 必落盘，重启零拷贝）；
#   ③ 运行期兜底 cmd_all → _ensure_runtime_models——镜像层/持久盘任一缺失时用
#      start.sh fetch-models 拉齐（缺组才下、字节幂等），绝不启动死循环。
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

# ── 检索模型 + GGUF stage：构建期从魔搭 CDN 拉取（烤镜像=Running 后 12s 可用） ──
# 2026-10-27 部署最快化的定案：**烤镜像为主 + 运行期兜底**（三层各司其职）：
#   - 模型层组级缓存（ARG MODELS_REV 不变=命中层缓存零下载，参见 22:02→22:08 实测
#     5m47s 的那场）；首次无缓存时 16 连接 aria2（构建机走魔搭 CDN 最快通道）；
#   - 生成侧 manifest/blobs 经运行期 ollama create 落持久盘（跨重建零拷贝收益）；
#     bge 权重不暖盘——烤镜像层已在，拷一份纯耗时没有收益。
#   - 镜像层/持久盘任一丢失都由运行期下兜底拉齐，绝不出现启动死循环。
# 模型清单变化时 bump MODELS_REV 强制重拉；组内单文件变更由字节校验兜住。
FROM base AS fetchmodels
ARG MODELS_REV=2026-10-26
ENV MODELS_REV=${MODELS_REV}
COPY start.sh /tmp/start.sh
RUN ["apt-get","update"]
RUN ["apt-get","install","-y","--no-install-recommends","aria2","ca-certificates"]
RUN ["bash","/tmp/start.sh","fetch-models","/models","gguf"]
RUN ["bash","/tmp/start.sh","fetch-models","/models","bge-m3"]
RUN ["bash","/tmp/start.sh","fetch-models","/models","bge-reranker-v2-m3"]

# ── 资产 stage：与 pydeps / fetchmodels 并行；语料/产物/配置 ──
# artifacts/chunks.json 只服务本地重建，不出镜像（serve 只读 meta+collection）。
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
# ── 烤镜像层：权重进镜像（Running 即用；运行期兜底只在缺失时才动）──
COPY --from=fetchmodels /models/bge-m3 /app/models/bge-m3
COPY --from=fetchmodels /models/bge-reranker-v2-m3 /app/models/bge-reranker-v2-m3
# 生成模型 GGUF 直落 /app/ollama_models；启动后 all 执行 ollama create（manifest
# 落持久盘 OLLAMA_MODELS，重启零拷贝）。>100MB 断言：GGUF 无稳定清单大小，构建期
# 就 fail-fast，不让半截文件进镜像。
COPY --from=fetchmodels /models/_gguf/MiniCPM5-2B-Q4_K_M.gguf /app/ollama_models/MiniCPM5-2B-Q4_K_M.gguf
RUN ["python","-c","import os,sys; p='/app/ollama_models/MiniCPM5-2B-Q4_K_M.gguf'; n=os.path.getsize(p); print('GGUF bytes:', n); sys.exit(0 if n > 100000000 else 1)"]
# start.sh 进入镜像：运行期兜底的唯一下载器（主路径=烤镜像层）。
COPY start.sh /app/start.sh
COPY chatbot.py /app/chatbot.py
EXPOSE 7860
# start-period 600s：烤镜像主路径 ~12s 就绪；只有兜底路径（镜像层/持久盘双缺失）
# 才需要先下 4.9G 再到 7860——宽窗口只服务那种极端场景（120s 老窗口曾把"装模型"
# 的健康等待误判成不健康）。
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 CMD ["python","-c","import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3)"]
ENTRYPOINT ["python","chatbot.py","all"]
