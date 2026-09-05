FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 内置 Ollama（未设置外部 OLLAMA_BASE_URL 时由 entrypoint 启动）
RUN curl -fsSL https://ollama.com/install.sh | sh

# 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码 + 源文本 + 模型 + 预构建索引（免首次冷启动重建）
COPY app.py rag_engine.py entrypoint.sh ./
COPY books/ ./books/
COPY models/ ./models/
COPY chroma_db_v2/ ./chroma_db_v2/
COPY cache_v2/ ./cache_v2/

RUN chmod +x /entrypoint.sh

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENV OLLAMA_BASE_URL=http://localhost:11434
ENV MODEL=qwen3.5:2b-q4_K_M
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

ENTRYPOINT ["/entrypoint.sh"]
