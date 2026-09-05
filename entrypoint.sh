#!/bin/sh
set -e

# 1) 若未指定外部 Ollama，则启动内置 Ollama 并拉取模型
if [ "$OLLAMA_BASE_URL" = "http://localhost:11434" ]; then
  echo "No external OLLAMA_BASE_URL set -> starting built-in Ollama..."
  ollama serve > /var/log/ollama.log 2>&1 &
  for i in $(seq 1 30); do
    curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
  echo "Pulling model $MODEL ..."
  ollama pull "$MODEL" || echo "WARN: 模型拉取失败，将使用外部 Ollama 或稍后重试"
fi

# 2) 索引缺失时自动构建（正常情况下镜像已带预构建索引）
if [ ! -d "/app/chroma_db_v2" ] || [ -z "$(ls -A /app/chroma_db_v2 2>/dev/null)" ]; then
  echo "索引缺失，开始构建..."
  python app.py process
  python app.py index
fi

# 3) 启动 Streamlit
exec streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
