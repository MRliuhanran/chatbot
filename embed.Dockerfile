# 检索模型服务镜像（现成组件 infinity，非自建微服务）：
# **一个进程托管嵌入与重排两个模型**（共享一份 torch 运行时，请求按 body 里的 model 字段
# 路由到对应 engine），故 compose 里这一个容器同时服务 /v1/embeddings 与 /v1/rerank。
# 只装 infinity 的 torch 运行时，不复制应用依赖；模型权重由 compose 挂载 ./models 只读。
# 与 app 的全栈镜像相互独立 —— app 仍从 Dockerfile 构建，本镜像只被该服务使用。
# 说明：
#   - --no-bettertransformer 必须显式关：infinity 默认开启会走 optimum/bettertransformer
#     路径，而 torch 依赖集不装 optimum（0.0.76 实测 NameError，模型根本加载不了）。
#   - 健康检查用 curl：对比 qdrant 官方镜像无 curl 的情况，本镜像显式装了 curl。
#   - 镜像源与 Dockerfile 同方针：tuna 清华源优先（国内可拉），失败回退官方 PyPI。
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100

RUN pip install --no-cache-dir \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        "infinity-emb[server,torch]==0.0.77" \
    || pip install --no-cache-dir "infinity-emb[server,torch]==0.0.77"

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /models

EXPOSE 8001

# 两个模型都挂 ./models：目录 basename 恰好等于要注册的模型名（bge-m3 / bge-reranker-v2-m3），
# --served-model-name 显式写出来只为把契约钉住（compose 的 command 与此保持一致）。
CMD ["infinity_emb", "v2", "--model-id", "/models/bge-m3", "--served-model-name", "bge-m3", "--model-id", "/models/bge-reranker-v2-m3", "--served-model-name", "bge-reranker-v2-m3", "--engine", "torch", "--device", "cpu", "--no-bettertransformer", "--url-prefix", "/v1", "--host", "0.0.0.0", "--port", "8001"]