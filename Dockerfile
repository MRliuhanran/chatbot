# ── 单容器 ModelScope 镜像（快速发布：预构建 base + 只替换应用层）────────────
# 重层（ollama/qdrant/python 依赖/模型/storage）已在以下镜像中构建并推送：
#   mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/independent-service:366944-7e8f3a80-2026-10-28-22-16-27
# 本 Dockerfile 只基于它做应用层 delta：COPY .env / start.sh / chatbot.py + bge-m3
# 池化元文件、RUN pip 补装 infinity-emb/psutil 小包（torch 等重件留在 base，不动重层）。
# 重层需要更新时，重新构建一次 base（或换 BASE_IMAGE build-arg），不必每次发布重跑。
ARG BASE_IMAGE=mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/independent-service:366944-7e8f3a80-2026-10-28-22-16-27
FROM ${BASE_IMAGE}

WORKDIR /app
COPY .env /app/.env
COPY start.sh /app/start.sh
COPY chatbot.py /app/chatbot.py
# base 烤入层（10-28）的 bge-m3 缺 ST 池化元文件——modules.json 要求 1_Pooling/config.json，
# 缺了 infinity 的 ST 引擎把模型路径当 HF repo id → HFValidationError，rc=3 启动即崩
# （2026-10-31 本地实证）。应用层补 191B；将来重层换 base 时此覆盖幂等可留。
COPY models/bge-m3/1_Pooling/config.json /app/models/bge-m3/1_Pooling/config.json

# http 后端两个 extras：[torch]=sentence-transformers+torch（base 已有 torch 2.7.1，
# 不重装）、[server]=fastapi/uvicorn/pydantic 等 HTTP 面板。版本 = 本地实测组合 0.0.76。
# psutil：内存熔断（RAG_MEM_GUARD=1）的唯一水位口径，base 也没有。
RUN ["pip", "install", "--no-cache-dir", "-i", "https://mirrors.aliyun.com/pypi/simple/", "infinity-emb[torch,server]==0.0.76", "psutil==7.0.0"]

EXPOSE 7860
ENTRYPOINT ["python","chatbot.py","all"]
