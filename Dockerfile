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

# http 后端依赖，一句话三块：
#   ① [torch,server]=infinity 引擎+HTTP 面板（typer<0.13 钉死 → 配套 click==8.1.8 防
#      未来 base 重建时 pip 现解 newer click 与 typer 0.12.5 CLI 解析不兼容
#      TypeError "Secondary flag is not valid for non-boolean flag"，2026-10-31 cage 实证）
#   ② optimum==1.22.0：bge-m3 走 torch 引擎时 EngineArgs.bettertransformer 默认 True，
#      acceleration.py:46 解引用 BetterTransformerManager；该符号属 optimum 可选 extra
#      （不在 [torch,server] 里），容器缺它 → NameError rc=3 启动即崩 + 30s 循环
#      （2026-10-31 cage 全量 traceback 实证：transformer/embedder/sentence_transformer.py:62
#      → acceleration.py:46）。1.22.0 = 本地实证组合的同版本。
#   ③ psutil：内存熔断（RAG_MEM_GUARD=1）的水位口径；torch==2.7.1 与 base 同版本锁死
#      防 pip 现解漂移。版本 = 本地实测组合 0.0.76。
RUN ["pip", "install", "--no-cache-dir", "-i", "https://mirrors.aliyun.com/pypi/simple/", "infinity-emb[torch,server]==0.0.76", "psutil==7.0.0", "torch==2.7.1", "optimum==1.22.0", "click==8.1.8"]

EXPOSE 7860
ENTRYPOINT ["python","chatbot.py","all"]
