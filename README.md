# RAG 垂直领域知识库问答（单文件版）

本仓库是一个**本机单用户**的 RAG 问答应用：对本地 `corpus/` 的 `*.txt` 做分块、建稠密 + BM25 稀疏索引，在线用双通道召回 → RRF → 重排 → 上下文预算装配，交给 Ollama 流式生成；前端是 Streamlit。

代码与脚本按“一个 shell、一个 Python、一个 README”组织；ModelScope 部署只额外增加**一个 Dockerfile**（不使用 compose）：

| 文件 | 作用 |
|---|---|
| **chatbot.py** | 唯一 Python 文件：CLI + Streamlit + 全部业务逻辑 |
| **start.sh** | 唯一 shell 文件：Qdrant / Ollama / 应用起停、日志、命令透传、.env 加载 |
| **README.md** | 唯一 Markdown 文档 |
| **Dockerfile** | 单容器 ModelScope 镜像（Qdrant + Ollama + Streamlit；构建期复用唯一 shell `start.sh fetch-models`，最终镜像不含 shell） |
| `.env` | 配置覆盖层（代码默认值为唯一权威） |
| `requirements.txt` | 运行时依赖（全量 `==` 锁定） |
| `corpus/` `models/` `ollama_models/` `artifacts/` `storage/` `lexicon/` | 语料 / 检索模型 / Ollama 模型 / 分块产物 / Qdrant 数据 / 词表 |
| `qdrant_config/` `.streamlit/` | 本地 Qdrant 与应用配置 |

> 支持两种运行方式：本机原生（`start.sh`）或单容器 Docker（`Dockerfile`）。完整版（含 tools/ 辅助脚本与旧部署文件）保留在 git tag `backup/pre-minimal-20261025`。

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
# 1) 起 Qdrant + Ollama + 应用（只复用本脚本启动的进程；外部实例会报错）
./start.sh start

# 2) 首次或语料变化后：分块 + 建索引（需要 Qdrant 在跑）
./start.sh process
./start.sh index

# 3) 打开浏览器
open http://127.0.0.1:8501
~~~

应用启动时校验索引与分块指纹；不一致会拒绝检索并提示重跑命令，这是有意的 fail-closed。

> `start.sh` 的复用判据是“pid 文件 + 命令行匹配”，不是只看端口健康：外部/旧进程占着 8501/6333/11434 会被拒绝，避免假发布。

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
| `./start.sh evaluate <评测集> [...]` | 离线检索评估（Recall@k / MRR@k / nDCG@k；不依赖 Ollama，只测检索链路） |

所有命令最终都落到 `chatbot.py` 的 CLI；`start.sh` 只做服务编排与命令透传。

构建台账默认开（`RAG_BUILD_STATS=0` 关）：process / index / sparse 每次成功收尾在
`artifacts/builds.jsonl` 追加一行结构化读数（复用率、缓存命中率/是否重写、嵌入与写批墙钟、
写批构造-往返分解），摘要工具 `python tools/build_stats.py`（`--json` / `--tail N`）。

## 4. 配置

配置规则：**代码默认值为准，`.env` 只做覆盖层**。缺行 = 用默认值；写了代码不读取的键 = 失效；非法值启动即报错。

常用变量：

| 变量 | 说明 |
|---|---|
| `MODEL` / `OLLAMA_BASE_URL` | 生成模型与 Ollama 地址 |
| `RAG_NUM_CTX` / `RAG_NUM_PREDICT` | 上下文窗口 / 输出上限（0=不限长） |
| `RAG_DATA_DIR` / `RAG_DOC_GLOB` | 语料目录与匹配模式（默认 `corpus/*.txt`） |
| `RAG_EMBED_BACKEND` / `RAG_RERANK_BACKEND` | `http`（现行：同一 infinity 实例双模型，GPU=auto）/ `torch`（进程内降档） |
| `RAG_EMBED_MODEL_PATH` / `RAG_RERANK_MODEL_PATH` | 本地检索模型目录 |
| `RAG_TOP_K` / `RAG_RECALL_LIMIT` | 返回条数 / 每通道召回深度（重排候选不截断） |
| `RAG_LOGIN_ENABLED` / `RAG_LOGIN_IDLE_TTL_S` | 登录门开关（本地手机号模式，无第三方 IdP）/ 闲置过期秒数（默认 3600，每次使用顺延） |
| `RAG_MEM_GUARD` / `RAG_MEM_KILL_PCT` | 内存熔断开关 / 触发水位（默认 95%：触线中止本轮，等同"停止生成"） |
| `RAG_TURN_STATS` | 单轮台账（排查性能时临时开） |
| `RAG_BUILD_STATS` / `RAG_BUILD_STATS_FILE` | 构建台账（默认开；每轮 process/index/sparse 成功收尾各一行）/ 落盘路径（默认 `artifacts/builds.jsonl`） |

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

在线链路：分块/词表指纹闸门 → 稠密 + 稀疏并发召回 → 加权 RRF → 按 id 回取 → 父块去重 → 全候选重排 → token 硬上界装配 → Ollama 流式生成。失败不入历史，Stop 主动断开上游流。

## 6. 数据重建与指纹

~~~bash
./start.sh process   # 改语料/切分参数
./start.sh index     # 改模型/新增内容（内容哈希增量）
./start.sh sparse    # 改词表/BM25
./start.sh stages    # 判断该跑哪个，退出码 0=全新鲜
~~~

产物由指纹绑定：切分指纹、词表指纹、稠密缓存键。**不要改名/加装饰器/改空白**到 `chatbot.py` 中 `_CHUNKING_CODE_UNITS`、`_embed_code_fingerprint`、`lexicon_fingerprint` 列出的函数——`inspect.getsource` 变化会触发全量重建。

嵌入缓存的两条运行期约定（2026-10-29 起）：

- **落盘位宽与文件名标签同源**：`fp16` 计算路径（GPU half 权重）的缓存真的存 float16（磁盘减半），
  `fp32`（CPU）保持 float32；读侧统一升 float32，在线 upsert 与质量口径不变。文件名四段键
  （治路径/内容/代码/精度）不受影响，**已有缓存不失效**。
- **index 成功收尾自动 GC**：换嵌入模型或改嵌入代码后，旧的 `code_key`/模型指纹缓存文件不再被
  任何一次运行读取，index 会顺手删掉它们（`*.tmp.npz` 中断残留照旧在起手时清）。不再手工删旧 npz。

## 7. ModelScope 部署：一个 Dockerfile

本地运行 = 源文件 + start.sh；ModelScope 运行 = **一个 Dockerfile + 一个容器**。容器内 qdrant / ollama / streamlit 三进程由 `chatbot.py all` 管理（源码里的进程管理器）；**http 后端多托一个 infinity 作为第 4 进程**（2026-10-30 起 `.env=http/http`：应用层 pip 补装 `infinity-emb[torch,server]==0.0.76` + `psutil`，并 COPY 补齐 bge-m3 缺失的 `1_Pooling/config.json` 池化元文件——缺它 infinity 的 ST 引擎启动即崩 rc=3）。`cmd_all` 按 start.sh 同构 argv 编排：app 依赖就绪后立即放量（平台健康探测打 7860），infinity 就绪由后台线程两段式探测（health → embeddings+rerank 真打一发 ≤ `RAG_INFINITY_WAIT`），未就绪 fail-closed 退出交平台重启。不依赖 compose。

### 7.1 Dockerfile 约束

- 所有 `RUN` 都是 **exec form**：`RUN ["pip","install",...]`，不经过 `/bin/sh -c`；
- 唯一入口：`ENTRYPOINT ["python","chatbot.py","all"]`（start.sh 只作为运行期模型下载器进镜像）；
- qdrant / ollama 二进制直接从官方镜像 COPY，不在 Dockerfile 里下载/解压；
- 应用层只做小 delta：COPY 三配置文件 + 补 COPY `models/bge-m3/1_Pooling/config.json`（191B 池化元文件，`.dockerignore` 白名单放行）+ RUN exec-form pip 装 infinity-emb/psutil 小包；torch 等重件留在 base；
- `HEALTHCHECK` 也是 exec form（`python -c ...`）。

> pip 等包管理器内部仍会执行自己的脚本，但 Dockerfile 本身没有 shell 形式的命令行。

### 7.2 构建与运行

**模型三层部署（2026-10-27 最快化定案）**：① 镜像层烤权重——`fetchmodels` stage 构建期拉魔搭 CDN，组级层缓存命中即零下载（`ARG MODELS_REV` 不变即命中）；② 持久盘 `/mnt/workspace` 暖 ollama blobs（重启零拷贝）；③ 运行期兜底 `cmd_all → _ensure_runtime_models`——镜像层/持久盘任一缺失才用 `start.sh fetch-models` 补齐（缺组才下、字节级幂等，绝不启动死循环）。Running 后约 12s 可用；只有双层尽失的极端场景才先下 4.9G。

| 资产 | 来源 |
|---|---|
| 检索/重排模型 | 镜像层（与本地同文件清单、逐文件字节校验）；缺失时运行期 fetch-models 补 |
| GGUF 生成模型 | 镜像层 `/app/ollama_models`；启动后 `ollama create`（manifest 落持久盘，重启零拷贝） |
| `artifacts/` + `storage/` | 本地已建好的分块产物与 Qdrant 数据（保证指纹一致；`artifacts/chunks.json` 只服务本地重建，不出镜像） |
| `ollama_models/` | 可选复用宿主机 Ollama 模型：`-v ~/.ollama/models:/app/ollama_models:ro`；否则运行期把 GGUF 落 `OLLAMA_MODELS` 再 `ollama create` |

~~~bash
docker build -t rag-chatbot:modelscope .
docker run --rm -p 7860:7860 rag-chatbot:modelscope
# 打开 http://127.0.0.1:7860
~~~

容器内 `chatbot.py all` 会：

1. **先校验模型**（烤镜像命中=零下载；缺组才 `start.sh fetch-models` 兜底）；
2. 并行启动 Qdrant 与 Ollama；等 `/healthz` 与 `/api/version` 就绪；
3. 若没有现成 manifest，用 `OLLAMA_MODELS/*.gguf` 执行 `ollama create`（持久盘时限一次拷贝，之后零启动成本）；
4. 启动 Streamlit（`0.0.0.0:7860`）；
5. 任一子进程退出 → 指数退避后就地重启（最长 30s）；收到 SIGTERM 优雅终止；
6. 模型装载失败/依赖 120s 不健康/模型创建失败 → 容器退出（平台重启自动重试，已下载字节续传）。

### 7.3 数据更新

推荐与本地一致的路径：**本地 process/index → 重新 build 镜像 → 重新部署**，这样 `artifacts/` 与 `storage/` 始终同源。

也可以进入运行中的容器执行（注意 8G 档内存，重建期间会与生成/检索争内存）：

~~~bash
docker exec -it <container> python chatbot.py process
docker exec -it <container> python chatbot.py index
docker exec -it <container> python chatbot.py stages
~~~

### 7.4 功能、性能、效果

- **功能同源**：同一份 `chatbot.py`、`.env`、`artifacts/`、`storage/`；检索/生成模型与本地同一文件清单（运行期下载+字节校验）；没有 torch/http 或 CPU/GPU 的代码分叉。
- **效果一致**：检索模型、重排模型、生成模型、切分/索引参数都相同，质量口径一致。
- **性能不可能完全一致**：本地是 Metal/MPS，ModelScope 是 2vCPU 纯 CPU；重排、索引、首字延迟会明显慢于本机。这是硬件差异，不是代码差异。
- 单容器 8G 档注意 `RAG_NUM_CTX`；OOM 时 `all` 会重启被杀的子进程（与本地 start.sh 的巡检语义一致）。

### 7.5 并行

- **构建并行**：`pydeps`（依赖）、`fetchmodels`（烤模型）、`assets`（语料/产物）三个 stage 互不依赖并行执行，最终 `app` 合并。
- **运行并行**：`chatbot.py all` 先装载模型，再并行拉起 Qdrant 与 Ollama，健康后启动 Streamlit。

### 7.6 一键发布（tools/ms_release.sh）

~~~bash
bash tools/ms_release.sh               # push(魔搭+GitHub) → deploy → 轮询 → 运行日志验收
bash tools/ms_release.sh --verify-only # 只对当前运行日志做验收
bash tools/ms_release.sh --no-deploy   # 只 push 不触发构建
~~~

验收判据与部署事故复盘同源：`🔧 分块指纹` 两侧一致；`📦 [模型装载]`/`[模型落位]` 行；无 `RuntimeError` 与 ❌ 新行（storage 运行时 churn 不挡发布）。构建超时默认自动重触发一次。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `缺少 Qdrant 可执行文件` | 下载对应平台 Qdrant 放到 `.bin/qdrant/qdrant`，或 `QDRANT_BIN=/path/to/qdrant ./start.sh start` |
| 生成 404 / 模型不在 Ollama | `ollama list` 确认 `.env` 的 `MODEL` 已存在；冷启动拉取期间会失败 |
| 页面提示向量库为空 | 先 `./start.sh process && ./start.sh index` |
| `index` 提示可开内存不足 | 先 `./start.sh stop` 再重建（8G 档尤其注意） |
| 改代码/`.env` 不生效 | 本脚本会复用已在跑的进程，`./start.sh stop && ./start.sh start` |
| 配了 `http` 检索后端但查询报错 | 本地：`./start.sh status` 看 infinity 是否健康，起服务后先跑 `./start.sh embedpairs`（对拍 cos≥0.999 才准入）；容器：`cmd_all` 自行编排并等就绪，看运行日志的 `[all] infinity 未就绪` 行 |
| 容器冷启动慢（首启） | 持久盘无权重时先下载 4.9G（运行日志看 `📦 [模型装载]` 进度），下载完成才到 7860；之后重启零下载 |

## 9. 边界

- 单机、单用户；无多租户、无并发队列。登录门是**自报手机号识别**（无密码/短信/IdP），不是认证；会话存进程内，重启即全员重登。
- 不限制在线人数与并发数：内存是唯一闸门（`RAG_MEM_*`）。生成中触到水位线即中止本轮（等同"停止生成"，残缺答案不入历史）；不排队、不拒绝登录、不卸载模型（驻留交给 Ollama keep_alive）。
- 语料仅支持 `*.txt`。
- 生成侧不重试、不拒答、无输出长度闸（窗口是唯一上限）；这是显式设计。
- 没有测试框架与 CI；质量靠指纹闸门、`start.sh stages` 与人工验收。检索质量可用离线评估
  `evaluate` 命令（手标评测集，只测检索链路、不依赖 Ollama），但它不是常驻的回归防线，评测集是人工维护的。
