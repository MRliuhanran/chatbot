# RAG 垂直领域知识库问答（单文件版）

本仓库是一个**本机单用户**的 RAG 问答应用：对本地 `corpus/` 的 `*.txt` 做分块、建稠密 + BM25 稀疏索引，在线用双通道召回 → RRF → 重排 → 上下文预算装配，交给 Ollama 流式生成；前端是 Streamlit。

代码与脚本按“一个 shell、一个 Python、一个 README”组织；Docker 部署只额外增加 Dockerfile / compose 配置：

| 文件 | 作用 |
|---|---|
| **chatbot.py** | 唯一 Python 文件：CLI + Streamlit + 全部业务逻辑 |
| **start.sh** | 唯一 shell 文件：Qdrant / Ollama / 应用起停、日志、命令透传、.env 加载 |
| **README.md** | 唯一 Markdown 文档 |
| **Dockerfile** | 无 shell 的镜像构建（RUN 全 exec-form，无 entrypoint.sh） |
| **docker-compose.yml** | qdrant / ollama / app 三容器并行启动 |
| `.env` | 配置覆盖层（代码默认值为唯一权威） |
| `requirements.txt` | 运行时依赖（全量 `==` 锁定） |
| `corpus/` `models/` `artifacts/` `storage/` `lexicon/` | 语料 / 模型 / 分块产物 / Qdrant 数据 / 词表 |
| `qdrant_config/` `.streamlit/` | 本地 Qdrant 与应用配置 |

> 支持两种运行方式：本机原生（`start.sh`）或 Docker Compose（`Dockerfile` + `docker-compose.yml`）。完整版（含 tools/ 辅助脚本与旧部署文件）保留在 git tag `backup/pre-minimal-20261025`。

## 1. 环境准备

1. Python 3.10：
   ~~~bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ~~~
2. Ollama 已安装，并准备 `.env` 中 `MODEL` 对应的模型（例如 MiniCPM5-2B）：
   ~~~bash
   ollama list
   # 没有就按模型名拉取；模型文件也可用本地导入
   ~~~
3. Qdrant 可执行文件放到 `.bin/qdrant/qdrant`（或设置 `QDRANT_BIN`）：
   ~~~bash
   chmod +x .bin/qdrant/qdrant
   ~~~
4. 检索模型目录存在：默认 `./models/bge-m3`（嵌入）与 `./models/bge-reranker-v2-m3`（重排）。

## 2. 快速开始

~~~bash
# 1) 起 Qdrant + Ollama + 应用（已在跑的复用）
./start.sh start

# 2) 首次或语料变化后：分块 + 建索引（需要 Qdrant 在跑）
./start.sh process
./start.sh index

# 3) 打开浏览器
open http://127.0.0.1:8501
~~~

应用启动时校验索引与分块指纹；不一致会拒绝检索并提示重跑命令，这是有意的 fail-closed。

## 3. 常用命令

| 命令 | 作用 |
|---|---|
| `./start.sh start` | 起 Qdrant / Ollama / Streamlit（复用已运行实例） |
| `./start.sh stop` | 停本脚本记录的三个进程 |
| `./start.sh restart` | 先停后起（改代码/改 `.env` 后用） |
| `./start.sh status` | 三个组件的健康状态与 pid |
| `./start.sh logs` | `tail -F` Qdrant / Ollama / Streamlit 日志 |
| `./start.sh serve` | 只前台起 Streamlit（调试） |
| `./start.sh process` | 分块（等价 `python chatbot.py process`） |
| `./start.sh index` | 建索引 |
| `./start.sh sparse` | 只重建稀疏向量（换词表后） |
| `./start.sh stages` | 四阶段新鲜度检查，给出最小重跑命令 |
| `./start.sh lexicon` | 词表自检 |
| `./start.sh compare-chunks A [B]` | 两份分块产物对比 |

所有命令最终都落到 `chatbot.py` 的 CLI；`start.sh` 只做服务编排与命令透传。

## 4. 配置

配置规则：**代码默认值为准，`.env` 只做覆盖层**。缺行 = 用默认值；写了代码不读取的键 = 失效；非法值启动即报错。

常用变量：

| 变量 | 说明 |
|---|---|
| `MODEL` / `OLLAMA_BASE_URL` | 生成模型与 Ollama 地址 |
| `RAG_NUM_CTX` / `RAG_NUM_PREDICT` | 上下文窗口 / 输出上限（0=不限长） |
| `RAG_DATA_DIR` / `RAG_DOC_GLOB` | 语料目录与匹配模式（默认 `corpus/*.txt`） |
| `RAG_EMBED_BACKEND` / `RAG_RERANK_BACKEND` | `torch`（进程内，默认）或 `http`（外部服务，本版本不内置） |
| `RAG_EMBED_MODEL_PATH` / `RAG_RERANK_MODEL_PATH` | 本地检索模型目录 |
| `RAG_TOP_K` / `RAG_RERANK_TOP_K` | 返回条数 / 重排候选池 |
| `RAG_LOGIN_ENABLED` | 登录门开关（本地无 OAuth 时自动关闭） |
| `RAG_TURN_STATS` | 单轮台账（排查性能时临时开） |

## 5. 架构一句话版

单文件内按依赖方向分层，**业务代码不按 CPU/GPU 或后端分叉**：

~~~text
L1 配置/日志/硬件/RuntimeProfile
L2 纯函数：归一化、分块、token 预算、RRF、去重、上下文格式
L3 适配器：EmbeddingBackend / RerankBackend / OllamaChatBackend（Torch/HTTP 实现）
L4 RAGEngine：只依赖 L3 协议，构造注入
L5 编排：run_turn、CLI 命令
L6 Streamlit UI（唯一 import streamlit 的层）
~~~

在线链路：空查询守卫 → 分块/词表指纹闸门 → 稠密 + 稀疏并发召回 → 加权 RRF → 按 id 回取 → 父块去重 → 全候选重排 → token 硬上界装配 → Ollama 流式生成。失败不入历史，Stop 主动断开上游流。

## 6. 数据重建与指纹

~~~bash
./start.sh process   # 改语料/切分参数
./start.sh index     # 改模型/新增内容（内容哈希增量）
./start.sh sparse    # 改词表/BM25
./start.sh stages    # 判断该跑哪个，退出码 0=全新鲜
~~~

产物由指纹绑定：切分指纹、词表指纹、稠密缓存键。**不要改名/加装饰器/改空白**到 `chatbot.py` 中 `_CHUNKING_CODE_UNITS`、`_embed_code_fingerprint`、`lexicon_fingerprint` 列出的函数——`inspect.getsource` 变化会触发全量重建。

## 7. Docker 部署（无 shell + 并行）

### 7.1 Dockerfile 里没有 shell

- 所有 `RUN` 都是 **exec form**：`RUN ["pip","install",...]`，不经过 `/bin/sh -c`；
- 不 `COPY` 任何 shell 脚本；没有 `entrypoint.sh`；
- 容器入口直接 exec Streamlit：`ENTRYPOINT ["python","-m","streamlit",...]`；
- `HEALTHCHECK` 同样是 exec form（`python -c ...`）。

> 说明：pip/apt 内部仍会执行各自脚本，但 Dockerfile 本身没有 shell 形式的命令行。

### 7.2 并行在哪里

- **BuildKit 并行 stage**：`pydeps`（依赖安装）与 `assets`（语料/产物/词表复制）互不依赖，默认并行构建；最后 `app` 合并两者。
- **Compose 并行服务**：`qdrant` 与 `ollama` 同时启动并各自健康检查；`app` 等两者健康后启动。
- 构建与启动都走 Docker 命令：

~~~bash
# 并行拉取/构建/启动
docker compose up -d --build
docker compose ps
docker compose logs -f app

# 单独观察并行构建
docker buildx build --progress=plain -t rag-chatbot:local .
~~~

### 7.3 首次/数据更新（全部 Docker 命令）

~~~bash
docker compose up -d --build

# 分块 + 建索引；artifacts 以 rw 卷挂载，结果直接落宿主
docker compose run --rm --entrypoint python app chatbot.py process
docker compose run --rm --entrypoint python app chatbot.py index

# 自检
docker compose run --rm --entrypoint python app chatbot.py stages
~~~

- 卷挂载：`corpus:ro`、`artifacts:rw`、`lexicon:ro`、`models:ro`；Qdrant 数据在 `./storage`。
- Ollama 模型挂宿主 `~/.ollama/models:ro`，请先在宿主机 `ollama pull <MODEL>`。
- `.env` 通过 compose `env_file` 注入；`QDRANT_HOST` / `OLLAMA_BASE_URL` 由 compose 覆盖为服务名。
- `.dockerignore` 排除了 `models/`、`storage/`、`artifacts/embeddings`、`artifacts/docs` 等，构建上下文只有源码与必要产物。
- 当前 `.env` 是 `torch` 后端，容器内进程内推理，不需要 infinity。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `缺少 Qdrant 可执行文件` | 下载对应平台 Qdrant 放到 `.bin/qdrant/qdrant`，或 `QDRANT_BIN=/path/to/qdrant ./start.sh start` |
| 生成 404 / 模型不在 Ollama | `ollama list` 确认 `.env` 的 `MODEL` 已存在；冷启动拉取期间会失败 |
| 页面提示向量库为空 | 先 `./start.sh process && ./start.sh index` |
| `index` 提示可开内存不足 | 先 `./start.sh stop` 再重建（8G 档尤其注意） |
| 改代码/`.env` 不生效 | 本脚本会复用已在跑的进程，`./start.sh stop && ./start.sh start` |
| 配了 `http` 检索后端但查询报错 | 本版本不内置 infinity，改回 `torch` 或自行启动外部服务 |
| Docker 构建慢 | 首次要下载 torch 并编译 llama-cpp-python；之后 BuildKit 缓存命中会快很多 |

## 9. 边界

- 单机、单用户；无多租户、无并发队列、无鉴权（默认登录门关闭）。
- 语料仅支持 `*.txt`。
- 生成侧不重试、不拒答、无输出长度闸（窗口是唯一上限）；这是显式设计。
- 没有测试框架与 CI；质量靠指纹闸门、`start.sh stages` 与人工验收。
