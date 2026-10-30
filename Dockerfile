# ── 单容器 ModelScope 镜像（快速发布：预构建 base + 只替换应用层）────────────
# 重层（ollama/qdrant/python 依赖/模型/storage）已在以下镜像中构建并推送：
#   mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/independent-service:366944-7e8f3a80-2026-10-28-22-16-27
# 本 Dockerfile 只基于它替换 .env / start.sh / chatbot.py，构建与 push 只涉及应用 delta。
# 2026-10-30：.env 烧 http/http，检索侧推理服务 infinity_emb 不在 base 重层里——
# 应用层补装小包（torch/transformers/numpy 等重件留在 base，不动重层）。
# 重层需要更新时，重新构建一次 base（或换 BASE_IMAGE build-arg），不必每次发布重跑。
ARG BASE_IMAGE=mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/independent-service:366944-7e8f3a80-2026-10-28-22-16-27
FROM ${BASE_IMAGE}

WORKDIR /app
COPY .env /app/.env
COPY start.sh /app/start.sh
COPY chatbot.py /app/chatbot.py

# http 后端两个 extras：[torch]=sentence-transformers+torch（base 已有 torch 2.7.1，
# 不重装）、[server]=fastapi/uvicorn/pydantic 等 HTTP 面板。版本 = 本地实测组合 0.0.76。
# psutil：内存熔断（RAG_MEM_GUARD=1）的唯一水位口径，base 也没有。
RUN ["pip", "install", "--no-cache-dir", "-i", "https://mirrors.aliyun.com/pypi/simple/", "infinity-emb[torch,server]==0.0.76", "psutil==7.0.0"]

EXPOSE 7860
ENTRYPOINT ["python","chatbot.py","all"]
