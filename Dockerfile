# ── 单容器 ModelScope 镜像（快速发布：预构建 base + 只替换应用层）────────────
# 重层（ollama/qdrant/python 依赖/模型/storage）已在以下镜像中构建并推送：
#   mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/independent-service:366944-7e8f3a80-2026-10-28-22-16-27
# 本 Dockerfile 只基于它替换 .env / start.sh / chatbot.py，构建与 push 只涉及应用 delta。
# 重层需要更新时，重新构建一次 base（或换 BASE_IMAGE build-arg），不必每次发布重跑。
ARG BASE_IMAGE=mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/independent-service:366944-7e8f3a80-2026-10-28-22-16-27
FROM ${BASE_IMAGE}

WORKDIR /app
COPY .env /app/.env
COPY start.sh /app/start.sh
COPY chatbot.py /app/chatbot.py

EXPOSE 7860
ENTRYPOINT ["python","chatbot.py","all"]
