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
ARG OLLAMA_GGUF_URL=https://modelscope.cn/models/OpenBMB/MiniCPM5-2B-gguf/resolve/master/MiniCPM5-2B-Q4_K_M.gguf

# ── 直接复用官方镜像里的二进制，避免 RUN 下载/解压 ──
FROM qdrant/qdrant:v1.19.1 AS qdrant
FROM ollama/ollama:0.34.4 AS ollama

# ── 依赖 stage：完整 python 镜像自带编译工具，供 llama-cpp-python 编译 ──
FROM python:${PYTHON_VERSION} AS pydeps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    CMAKE_ARGS="-DGGML_OPENMP=OFF"
COPY requirements.txt /tmp/requirements.txt
RUN ["pip","install","--no-cache-dir","cmake","ninja"]
RUN ["pip","install","--no-cache-dir","--index-url","https://download.pytorch.org/whl/cpu","torch==2.7.1"]
RUN ["pip","install","--no-cache-dir","-r","/tmp/requirements.txt"]

# ── 检索模型 stage：按本地同一批文件从 ModelScope 下载（指纹一致）──
# 只取本地已存在的文件；不下载 README/configuration.json 等额外文件，避免
# _model_content_fingerprint 与本地/已建索引不一致。与 pydeps / assets 并行。
FROM scratch AS fetchmodels
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/colbert_linear.pt /models/bge-m3/colbert_linear.pt
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/config.json /models/bge-m3/config.json
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/config_sentence_transformers.json /models/bge-m3/config_sentence_transformers.json
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/modules.json /models/bge-m3/modules.json
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/pytorch_model.bin /models/bge-m3/pytorch_model.bin
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/sentence_bert_config.json /models/bge-m3/sentence_bert_config.json
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/sentencepiece.bpe.model /models/bge-m3/sentencepiece.bpe.model
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/sparse_linear.pt /models/bge-m3/sparse_linear.pt
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/special_tokens_map.json /models/bge-m3/special_tokens_map.json
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/tokenizer.json /models/bge-m3/tokenizer.json
ADD https://modelscope.cn/models/BAAI/bge-m3/resolve/master/tokenizer_config.json /models/bge-m3/tokenizer_config.json
ADD https://modelscope.cn/models/BAAI/bge-reranker-v2-m3/resolve/master/config.json /models/bge-reranker-v2-m3/config.json
ADD https://modelscope.cn/models/BAAI/bge-reranker-v2-m3/resolve/master/model.safetensors /models/bge-reranker-v2-m3/model.safetensors
ADD https://modelscope.cn/models/BAAI/bge-reranker-v2-m3/resolve/master/sentencepiece.bpe.model /models/bge-reranker-v2-m3/sentencepiece.bpe.model
ADD https://modelscope.cn/models/BAAI/bge-reranker-v2-m3/resolve/master/special_tokens_map.json /models/bge-reranker-v2-m3/special_tokens_map.json
ADD https://modelscope.cn/models/BAAI/bge-reranker-v2-m3/resolve/master/tokenizer.json /models/bge-reranker-v2-m3/tokenizer.json
ADD https://modelscope.cn/models/BAAI/bge-reranker-v2-m3/resolve/master/tokenizer_config.json /models/bge-reranker-v2-m3/tokenizer_config.json

# ── 资产 stage：与 pydeps 并行；源码/产物/配置 ──
# ollama_models 可选：留 .keep 也能构建；正式生成模型由下方 ADD GGUF + all 里的 ollama create 完成。
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
    PIP_DISABLE_PIP_VERSION_CHECK=1
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
