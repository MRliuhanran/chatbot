# RAG 垂直领域知识库问答（单文件版）

本仓库是一个**本机单用户**的 RAG 问答应用：对本地 `corpus/` 的 `*.txt` 做分块、建稠密 + BM25 稀疏索引，在线用双通道召回 → RRF → 重排 → 上下文预算装配，交给 Ollama 流式生成；前端是 Streamlit。

> 📁 **docs/ 目录已归档**：`docs/architecture/` 四篇架构文档（2026-11-04 定稿）整体移入
> `docs/archive/architecture-2026-11-04/`（原样冻结，不再更新——行号与多处机制已被
> 2026-11-10/11 的组件解耦取代）。现行架构真相 = 本 README（§5 一句话版 + §10 已了结项
> 带证据）+ 代码本身；见 [`docs/README.md`](docs/README.md) 的取舍说明。

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
| `./start.sh index` | 建索引（成功收尾写 `artifacts/index_manifest.json`） |
| `./start.sh verify-build` | 构建期防呆：校验 `artifacts/index_manifest.json` 的 `chunk_fingerprint` 与当前语料/切分配置现算指纹一致（不等退出非 0；Dockerfile 与发布脚本都调它） |
| `./start.sh sparse` | 只重建稀疏向量（换词表后） |
| `./start.sh stages` | 四阶段新鲜度检查，给出最小重跑命令 |
| `./start.sh lexicon` | 词表自检 |
| `./start.sh compare-chunks A [B]` | 两份分块产物对比 |
| `./start.sh evaluate <评测集> [...]` | 离线检索评估（Recall@k / MRR@k / nDCG@k；不依赖 Ollama，只测检索链路）；`--compare-rewrite` 对同一集跑 raw/gold 改写对拍（评测集每行须带 `gold_rewrite`） |
| `python tools/turn_stats.py` | 单轮台账摘要（`--json` / `--tail N`；读 `.env` 的 `RAG_TURN_STATS_FILE`，排查性能时临时开 `RAG_TURN_STATS=1`） |
| `python tools/build_stats.py` | 构建台账摘要（读 `artifacts/builds.jsonl`：复用率/缓存命中/嵌入与写批墙钟） |

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
| `RAG_NUM_PREDICT` | 输出上限（0=不限长）；上下文窗口**不再配置**：运行期读 Ollama `/api/show` 的模型原生窗口，再按整机内存档位钳制（≤9G→32768、≤18G→65536、更大→原生值），输入上界/历史预算随之派生 |
| `RAG_DATA_DIR` / `RAG_DOC_GLOB` | 语料目录与匹配模式（默认 `corpus/*.txt`） |
| `RAG_EMBED_BACKEND` / `RAG_RERANK_BACKEND` | `http`（现行：同一 infinity 实例双模型，GPU=auto）/ `torch`（进程内降档） |
| `RAG_EMBED_MODEL_PATH` / `RAG_RERANK_MODEL_PATH` | 本地检索模型目录 |
| `RAG_TOP_K` / `RAG_RECALL_LIMIT` | 最终返回条数（默认 10：融合→去重→全体重排后按分数取前 10）/ 每通道召回深度（默认 10，重排候选不截断） |
| `RAG_REWRITE` / `RAG_REWRITE_MAX_TURNS` / `RAG_REWRITE_MAX_TOKENS` | 多轮检索改写（默认 0）：RAG_REWRITE=1 且有历史时，生成模型先把本轮问与最近几轮合成一条独立完整查询再进召回（`rewrite_query`，失败退原词、失败永不阻断本轮）；进改写上下文的最大完整轮数 / 改写输出 token 上界 |
| `RAG_SERVER_FUSION` | 检索编排服务端融合（默认 1）：双路召回 + RRF 融合 + 随交 payload = Qdrant Query API 一次调用（§10.2-D1）；0 回落客户端融合；权重非默认/深追踪自动走旧路径 |
| `RAG_QUERY_CACHE_TTL` / `RAG_QUERY_CACHE_MAX` | 查询级缓存（cachetools TTLCache，默认 60s / 64 条；§10.2-D2）：同查询查中即零 RTT 返回；0 = 关；索引重建后 TTL 窗口内旧命中最长存留 TTL 秒 |
| `RAG_OTLP_ENDPOINT` / `RAG_OTEL_SERVICE` | 单轮 trace 推送（OTel OTLP HTTP，未设 = 零成本 no-op，连包都不 import）：Logfire / OTel Collector / 任何 OTLP 后端通用；服务名（默认 `rag-chatbot`），与 turns 台账同口径（§10.2-D3） |
| `RAG_LOGIN_ENABLED` / `RAG_LOGIN_IDLE_TTL_S` | 登录门开关（本地手机号模式，无第三方 IdP）/ 闲置过期秒数（默认 3600，每次使用顺延） |
| `RAG_MEM_GUARD` / `RAG_MEM_KILL_PCT` | 内存熔断开关 / 触发水位（默认 95%：触线中止本轮，等同"停止生成"） |
| `RAG_QDRANT_WAIT` | 容器启动期等 Qdrant `/healthz` 的预算秒数（默认 300；新数据层首启 + 2vCPU 冷载入实测超过旧 120s） |
| `RAG_TURN_STATS` | 单轮台账（**常开**定档：每轮一行 ~200B，持续留证据；排查期可临时置 0 回默认） |
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

在线链路：分块/词表指纹闸门 → （`RAG_REWRITE=1` 且有历史时：多轮指代改写 `rewrite_query`）→ 双路召回 + **服务端 RRF 融合**（Qdrant Query API，`RAG_SERVER_FUSION`；深追踪/非默认权重/失败兜底回落客户端 weighted_rrf）→ payload 随召回一并交付（不再有独立"回取段"）→ 父块去重 → 全候选重排 → token 硬上界装配 → Ollama 流式生成。失败不入历史，Stop 主动断开上游流（Ollama 随断连取消服务端生成，2026-11-10 实测）。同查询 TTL 缓存查中即回（`RAG_QUERY_CACHE_*`，读数与直查逐条相同）；每轮收尾在台账之外同步发 OTLP trace（`RAG_OTLP_ENDPOINT` 未设则零成本跳过）。

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

`artifacts/index_manifest.json` 是索引的**发布清单**：`index` 成功收尾（点数量不变量与别名切换都通过之后）写入 `chunk_fingerprint` / `lexicon_id` / 集合名 / 分块数 / 模型内容指纹 / Qdrant 服务端版本；`verify-build` 与发布脚本以它的 `chunk_fingerprint` 为**唯一 EXPECTED_FP 来源**。它与 `storage/`、`chunks.meta.json` 是同一次 index 的产物，必须一起入仓、一起进镜像——这是部署侧“数据落后于应用”事故的根治点（见 §7）。清单缺失/读不到 = 发布前置失败，不允许“没有清单就放行”。

## 7. ModelScope 部署：一个 Dockerfile

本地运行 = 源文件 + start.sh；ModelScope 运行 = **一个 Dockerfile + 一个容器**。容器内 qdrant / ollama / streamlit 三进程由 `chatbot.py all` 管理（源码里的进程管理器）；**http 后端多托一个 infinity 作为第 4 进程**（2026-10-30 起 `.env=http/http`：应用层 pip 装 `infinity-emb[torch,server]==0.0.76` + `psutil` + `optimum==1.22.0`（bettertransformer 默认开，缺 optimum → acceleration.py NameError rc=3 起即崩）+ `click==8.1.8`（typer<0.13 的解析关）+ `torch==2.7.1` 锁本，并 COPY 补齐 bge-m3 缺失的 `1_Pooling/config.json` 池化元文件——缺它 ST 引擎把路径当 HF repo id 启即崩）。`cmd_all` 按 start.sh 同构 argv 编排：app 依赖就绪后立即放量（平台健康探测打 7860），infinity 就绪由后台线程两段式探测（health → embeddings+rerank 真打一发 ≤ `RAG_INFINITY_WAIT`），未就绪 fail-closed 退出交平台重启。不依赖 compose。运行环境是 FC 实例**每 1800s 强杀重启**（2026-11-1 运行日志实证）：boot 侧 infinity 跳过 warmup calibration（就绪探测真打一发即预热，冷加载实测 14-17s），Ollama 三驻留旋钮（`KEEP_ALIVE=-1`/`NUM_PARALLEL=1`/`MAX_LOADED_MODELS=1`）+ **编排层自醒驻留探针**（不依赖访客：预热链原挂在 Streamlit 脚本触发上，无人访问时 get_engine 不执行 → llama 零装载 → 访客首问必背冷加载；探针 /api/chat + num_ctx=model_context_window() 与回合链逐字同源自醒，访客到达后参数一致不触发二次装载）——模型在周期内驻留正确 ctx，用户第一问零重载。

**镜像分两层（2026-10-31 定案，指纹事故的根治）**：base 只放“重且不常变”的二进制与模型（qdrant/ollama、python 依赖、双检索模型权重、生成模型 GGUF）；`.env`/`start.sh`/`chatbot.py`/`1_Pooling` 与**全部可变数据**（`storage/`、`corpus/`、`lexicon/`、`qdrant_config/`、`artifacts/index_manifest.json` 及同代的 `artifacts/chunks.meta.json`）在同一次发布的应用层里一起 COPY，仓库是唯一事实源。此前 base 烤了 10-28 的旧 `storage/artifacts`、应用层却只换代码 → 数据永远落后于应用：2026-10-31 远程启动期分块指纹 `索引=923236fe7e4acc7c / 当前=3502968fe6f7039e` 不一致，检索闸门 fail-closed。可变数据随应用层发布后这个结构性缺口消失，另有构建期 `verify-build` 兜底。

### 7.1 Dockerfile 约束

- 所有 `RUN` 都是 **exec form 单行 JSON 数组**：`RUN ["pip","install",...]`，不经过 `/bin/sh -c`。**每个 RUN 的 JSON 数组必须写在同一行**——魔搭 dockerfile parser 会把跨行续行当成独立指令（2026-10-31 23:27 场 parse error line 20 实证）；
- 唯一入口：`ENTRYPOINT ["python","chatbot.py","all"]`（start.sh 只作为运行期模型下载器进镜像）；
- qdrant / ollama 二进制直接从官方镜像 COPY，不在 Dockerfile 里下载/解压；
- 层顺序固定：**base → pip RUN → 应用文件（.env/start.sh/chatbot.py/池化元文件）→ 可变数据层 → verify-build**。应用文件与数据 COPY 都在 pip RUN 之后：改代码/改数据都不会重跑 16min 级依赖层，日常发布只重打后三层；torch 重件留在 base；
- 数据层先 `RUN ["rm","-rf","/app/storage","/app/corpus","/app/lexicon"]` 再 COPY：`COPY 目录` 是 merge 语义，不删会留下 base 旧 segment / 已删语料，容器内现算指纹与清单永远对不上；
- COPY `storage` 后必须**补空 `wal/` 目录**：git 不跟踪空目录、`.dockerignore` 又排除 wal 大文件（closed-*/open-*），缺目录时 Qdrant 1.19.1 启动直接 panic rc=101（`Can't init WAL: NotFound ... /wal`，2026-10-31 远程 DeployFailed 真因，已用官方 qdrant 容器同 storage 复现）；WAL 内容不必进镜像，正常 stop 已 flush 到 segment；`verify-build` 会复核每个 shard 的 `wal/` 存在；
- 构建最后一步 `RUN ["python","chatbot.py","verify-build"]`：清单指纹 ≠ 现算指纹直接构建失败（fail-fast，不把问题留到运行期闸门）；
- `HEALTHCHECK` 也是 exec form（`python -c ...`）。

> pip 等包管理器内部仍会执行自己的脚本，但 Dockerfile 本身没有 shell 形式的命令行。

### 7.2 构建与运行

**模型三层部署（2026-10-27 最快化定案；属于 base 的构建期，日常发布不重跑）**：① 镜像层烤权重——`fetchmodels` stage 构建期拉魔搭 CDN，组级层缓存命中即零下载（`ARG MODELS_REV` 不变即命中）；② 持久盘 `/mnt/workspace` 暖 ollama blobs（重启零拷贝）；③ 运行期兜底 `cmd_all → _ensure_runtime_models`——镜像层/持久盘任一缺失才用 `start.sh fetch-models` 补齐（缺组才下、字节级幂等，绝不启动死循环）。Running 后启动期即打印 `🔧 分块指纹`（发布脚本轮询该行做验收，上限 300s，不再靠固定 sleep 猜）；只有双层尽失的极端场景才先下 4.9G。启动编排在 2026-10-31 远程 DeployFailed 后加固：**app 在依赖就绪前就 spawn**（7860 先监听，不占部署探测窗口；检索仍由指纹/词表闸门和依赖就绪门兜底），Qdrant 冷载入预算提到 `RAG_QDRANT_WAIT`（默认 300s），且 `_wait_http_ready` 会检测子进程提前退出并立刻归因，不再把「进程崩了」混进「载入慢」。

| 资产 | 来源 |
|---|---|
| 检索/重排模型 | 镜像层（与本地同文件清单、逐文件字节校验）；缺失时运行期 fetch-models 补 |
| GGUF 生成模型 | 镜像层 `/app/ollama_models`；启动后 `ollama create`（manifest 落持久盘，重启零拷贝） |
| `storage/` + `corpus/` + `lexicon/` + `qdrant_config/` + `artifacts/index_manifest.json` + `artifacts/chunks.meta.json` | **应用层 COPY**（不再是 base 快照）：本地 process/index 的同一次产物，与清单同时进镜像，指纹天然同源；`chunks.meta.json` 还让构建期 verify-build 能核对“process 与 index 同代”；`artifacts/chunks.json` 只服务本地重建，不出镜像 |
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

唯一推荐路径（数据与应用永远同源，根因修复的落地步骤）：

~~~bash
./start.sh start            # process/index 需要 Qdrant 在跑
./start.sh process          # 语料/切分参数变了
./start.sh index            # 模型/内容变了；成功收尾会写 artifacts/index_manifest.json
./start.sh verify-build     # 本地先自检（镜像构建期、发布 preflight 还会各跑一次）
./start.sh stop             # 必须：让 Qdrant 把 WAL 刷盘，避免把运行中的 storage 打进镜像
# storage/ 在 .gitignore 里，新索引文件必须 -f；漏了它镜像里的索引还是旧快照
git add -f storage
git add artifacts/index_manifest.json artifacts/chunks.meta.json
git commit -m "data: 重建索引并同步 index_manifest"
bash tools/ms_release.sh    # preflight → push → 构建 → 运行日志验收
~~~

三道闸门用**同一个判据**：`index` 写的 `chunk_fingerprint`（清单）vs 当前语料/切分配置现算指纹。任何一处不一致都在本地/构建期 fail-fast，而不是上线后由检索闸门拒服务。2026-10-31 的 923/350 事故之所以能上到远程，就是因为旧结构里“数据在 base、代码在应用层”，两者根本没有同源校验。

进入运行中的容器重建**只作应急**（改的是当次容器，下次发布仍回到镜像数据；且 8G 档会与生成/检索争内存）：

~~~bash
docker exec -it <container> python chatbot.py process
docker exec -it <container> python chatbot.py index
docker exec -it <container> python chatbot.py verify-build
docker exec -it <container> python chatbot.py stages
~~~

### 7.4 功能、性能、效果

- **功能同源**：同一份 `chatbot.py`、`.env`、`storage/`、`corpus/`、`artifacts/index_manifest.json`（全部经应用层 COPY，且数据与清单来自同一次 index）；检索/生成模型与本地同一文件清单（运行期下载+字节校验）；没有 torch/http 或 CPU/GPU 的代码分叉。
- **效果一致**：检索模型、重排模型、生成模型、切分/索引参数都相同，质量口径一致。
- **性能不可能完全一致**：本地是 Metal/MPS，ModelScope 是 2vCPU 纯 CPU；重排、索引、首字延迟会明显慢于本机。这是硬件差异，不是代码差异。
- 上下文窗口自 2026-10-31 起**动态派生**（不再有 `RAG_NUM_CTX` 配置）：窗口 = min(模型原生上限，整机内存档位) —— ≤9G 内存取 32768（8G 档实测稳定），≤18G 取 65536，更大才用满模型原生 131072（8G 上直接用 131072 会内存熔断，实测 9.3s 触 95.9%）。OOM 时 `all` 会重启被杀的子进程（与本地 start.sh 的巡检语义一致）。

### 7.5 并行

- **构建**：`pydeps`（依赖）/ `fetchmodels`（烤模型）/ `assets` 三个 stage 只在**重烤 base** 时并行；日常发布是单 stage：pip RUN（命中缓存）→ 应用文件层 → 可变数据层 → verify-build，只重打应用+数据层（改代码也不重装依赖）。
- **运行并行**：`chatbot.py all` 先装载模型，再并行拉起 Qdrant 与 Ollama，健康后启动 Streamlit。

### 7.6 一键发布（tools/ms_release.sh）

~~~bash
bash tools/ms_release.sh                  # preflight → push(魔搭+GitHub) → deploy → 轮询 → 运行日志验收
bash tools/ms_release.sh --verify-only    # 只对当前运行日志做验收（不 push 不 deploy）
bash tools/ms_release.sh --no-deploy      # preflight + 只 push 不触发构建
bash tools/ms_release.sh --skip-preflight # 应急：跳过 preflight 全部闸门（EXPECTED_FP 仍尽量从清单取）
~~~

**preflight（默认开）**：① 本机 Qdrant 仍在跑则拒绝发布（提示先 `./start.sh stop`，防止把正在写的 storage 打进镜像）；② `python chatbot.py verify-build` 失败即退出；③ 从 `artifacts/index_manifest.json` 取 `chunk_fingerprint` 作 `EXPECTED_FP`，取不到即退出；④ git 追踪完整性——清单必须已跟踪、`storage/` 新索引文件必须 `git add -f`（`git push` 只送已跟踪文件，漏了镜像还是旧 storage），否则拒绝发布。

**Running 后验收（轮询运行日志，默认最多 300s / 每 10s；`VERIFY_TIMEOUT` / `VERIFY_INTERVAL` 可调）**：

1. 必须见到启动期 `🔧 分块指纹: 索引=… / 当前=…`（`cmd_all` 启动即打）；**到点未见 = FAIL**（不再按“未见=ABSENT”放行），两侧必须相等且索引侧 = `EXPECTED_FP`；
2. `📦 [模型装载]` / `[模型落位]` 行；
3. 无 `RuntimeError`、无 `❌ [all]`（启动编排失败）；infinity 就绪行尽力检查；启动期 `🧮 词表指纹` 行**两侧相等则 PASS、不等则 FAIL**（未见行不阻断——平台日志行会被 ~2KB 截断/丢行，故 `_run_logs` 用 `page_size=100` 取证，稀疏侧另有检索期闸门兜底）。

构建超时默认自动重触发一次；`storage` 运行时 churn 不挡发布。

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

- 单机、**共享单库的多用户并发**：不做账号隔离/多租户、没有按用户的配额与队列；多会话可同时检索，生成槽位受 `OLLAMA_NUM_PARALLEL` 限制。登录门是**自报手机号识别**（无密码/短信/IdP），不是认证；会话存进程内，重启即全员重登。
- 不限制在线人数与并发数：内存是唯一闸门（`RAG_MEM_*`）。生成中触到水位线即中止本轮（等同"停止生成"，残缺答案不入历史）；不排队、不拒绝登录、不卸载模型（驻留交给 Ollama keep_alive）。
- 语料仅支持 `*.txt`。
- 生成侧不重试、不拒答、无输出长度闸（窗口是唯一上限）；这是显式设计。
- 没有测试框架与 CI；质量靠指纹闸门、`start.sh stages` 与人工验收。检索质量可用离线评估
  `evaluate` 命令（手标评测集，只测检索链路、不依赖 Ollama），但它不是常驻的回归防线，评测集是人工维护的。
## 10. 关注领域与缺口（基准 2026-11-10，锚函数名不锚行号）

> 本节只记"盯什么、缺什么、何时动"：已了结项一句话带证据归档，不展开；随变更更新 §10.4–§10.6
> （§10.1–§10.3 是快照）。锚点全部是当前代码里的真实符号。

### 10.1 六大领域 × 现状

| 领域 | 现状 |
|---|---|
| 模型内存 | 三闸定档：动态窗口（`model_context_window`，≤9G→32768）、`RAG_MEM_GUARD` 95% 熔断、Ollama 驻留旋钮（容器内 `OLLAMA_NUM_PARALLEL=1`）；32K 窗口单模型驻留实测 2.9G（§10.2-B3） |
| 检索构建 | 四阶段 + 指纹 fail-closed、`index_manifest.json` 发布清单（§7）、内容哈希增量缓存（fp16/fp32 位宽同源落盘 + index 收尾自动 GC） |
| 检索查询 | 双路召回 + 服务端 RRF 融合（Query API，§10.2-D1）→ 父块去重 → 全候选重排；同查询 TTL 缓存（§10.2-D2）；多轮指代改写已实装（默认关，§10.4） |
| 构建优化 | fp32 质量档 + 断点续算 + 自适应资源 + GPU 满档/gRPC 写入；双边台账：`artifacts/builds.jsonl`（`python tools/build_stats.py`）与 `turns.jsonl`（`python tools/turn_stats.py`，`RAG_TURN_STATS=1` 开） |
| 多轮上下文 | 令牌硬上界、整轮淘汰、失败不入历史；检索侧改写见 §10.4 |
| 多用户并行 | §9 定档：松散并发（内存是唯一闸门）；隔离/配额/队列有意不做（§10.5） |

### 10.2 已了结项（快照，一句话证据）

- **A2 在线台账**：`RAG_TURN_STATS` → `artifacts/turns.jsonl`（run_turn 尾部 `record_turn_stats`：TTFT/tok/s/prefill 比值/上下文取舍/水位提示），摘要 `python tools/turn_stats.py`；构建侧 `RAG_BUILD_STATS` 同构。
- **A3 检索侧评测**：`evaluate`（Recall@k/MRR@k/nDCG@k + `--compare-rewrite` raw/gold 对拍）；**生成侧"答得对不对"评测按 2026-11-10 定档不做**——若做也只是一次性功能工具（对拍 `quant_bench` 一类），不进闸门、非常驻防线。
- **A4 生成决策记录**：§9 已落（"不重试、不拒答、无输出长度闸；这是显式设计"）。
- **A6 Stop 服务端残留**（实测 2026-11-10，本地 minicpm5-2b）：2000 字流 6s 断开（等同 `Stop`/熔断走的 `r.close()` 路径）→ 静置 10s 探针 **3.7s 返回**（若服务端继续烧需等剩余生成时长的排队）→ **Ollama 随客户端断连取消生成**，条目关闭。
- **A7 数据同源**：两层镜像（§7.2）使可变数据随应用层同 COPY、`index_manifest.json` 是唯一 `EXPECTED_FP` 源（`verify-build` fail-fast + 发布脚本轮询 `🔧 分块指纹` 行）→ 结构性漂移缺口消失；旧三源（`bake.env`/`fetch_models.sh`/`deploy-check.sh`）已随极简化移除（tag `backup/pre-minimal-20261025`）。
- **C4 冷启动**：Ollama 三驻留旋钮 + 编排层自醒驻留探针（回报链 num_ctx 同源，访客到达不二次装载）+ infinity 两段式后台就绪（不做 warmup calibration）→ FC 每 1800s 强杀周期内首问零重载。
- **B3 内存并发**（实测 2026-11-10，`ollama ps` SIZE）：ctx2048 基线 1.7G → ctx32768 **2.9G**（KV≈+1.2G，预分配在加载期完成、并发请求不翻倍内存；6.2k token prefill 17.2s、解码 31 tok/s 正常）；本地外部 Ollama **未设** `OLLAMA_NUM_PARALLEL=1`（两路并发几乎同时完成、第二路吃前缀缓存），容器内 `all` 钉 1 = 单槽排队；触线由 `MEM_GUARD` 兜底 → 条目关闭（32768 窗口 ≥1 路生成 + 全组件 ≤8G 已实测）。
- **D1 检索编排 → Qdrant Query API 服务端融合**（2026-11-10 "拿来即用"解耦第①项）：检索三段（双通道并发 + 客户端 RRF + 按 id 回取）里的**融合段**归位存储引擎——`RAGEngine._server_fusion_points`（dense/sparse 两个 prefetch + `FusionQuery("rrf")`，payload 随交，墙钟 = embed + 1×RTT）。**验收 = 四组对拍全平**（live stub/rerank + multi_turn raw/gold × stub/rerank，全部 recall/mrr/ndcg 读数键逐个相等，0 漂移）；三档回落边界 = ① 非默认通道权重（服务端 RRF 不接受权重，告警一次走客户端融合）② `return_steps` 深追踪（保"每通道名次"可追溯，两路径互斥）③ 融合调用失败兜底纯稠密（对齐旧 `tolerate_sparse_error`；稠密真坏在兜底里照常抛 = fail-closed）。旧路径完全保留，`RAG_SERVER_FUSION=0` 随时重新对拍。
- **D2 查询缓存 → cachetools TTLCache**（第②项）：`RAG_QUERY_CACHE_TTL/MAX`（默认 60s / 64 条，0=关）。键=(query, top_k, source)，命中 = 零 RTT 零重排（实测 **13.67s → 0.0001s**，读数与直查逐条相同）；命中值 deepcopy 快照隔离（下游改写不污染缓存）；空结果与深追踪不进缓存；cachetools 缺装 = 一次性告警后缓存退场（增收益位不是正确性依赖）；指纹闸门照常每轮先行，fail-closed 不因缓存放开。
- **D3 观测 → OTel 核心（api/sdk/otlp-http 全家 1.42.1 锁齐）**（第③项）：`RAG_OTLP_ENDPOINT` 设了才生效——未设 = 每轮零成本 no-op（连 opentelemetry 都不 import，懒加载）；Logfire / OTel Collector / 任何 OTLP HTTP 后端**零代码**接入（Logfire 本身就是 OTel 包装，底层选 OTel 后"面板"永远只是配置项）。span `rag.turn` 与 turns 台账吃**同一份 fields**（字段单源加工不复读），起止 = `turn_start`（台账新字段）→ 收尾；全真链路冒烟：本地 OTLP 接收器收 655B protobuf span，`turn_start→ts` 差 = 检索+生成墙钟。治理记录：本机曾 api 1.45.1 + sdk 1.42.1 混装（`_ExtendedAttributes` import 崩）→ 重装锁齐 1.42.1 后通过。
- **D5 配置校验 → pydantic TypeAdapter**（第⑥项，2026-11-11）：`env_int`/`env_float` 的解析内核换 `TypeAdapter(int/float)`（pydantic-core 的 Rust 实现，pydantic-settings 的底层即 pydantic）——下划线千分位/首尾空白/`nan`/`inf`/上溢到 `1e400=inf` 与内建全同。**唯一语义分叉 = 非 ASCII Unicode 十进制数字**（全角 `１２３`、阿拉伯-印度 `٣٤٥`：pydantic 拒、内建 `int()/float()` 收，`.env` 手改时全角是输入法事故常见产物）——为零漂移在 `except ValidationError` 里垫内建 `int()/float()` 兜底，兜底也失败才抛原 ValueError 具名文案；**函数签名与报错文案逐字节不动**。其余五函数（`env_bool` 词表语义与 pydantic BoolValidator 不同、`env_choice`/`env_int_min`/`env_float_min`/`env_int_any`）保持显式手写——配置边界留在能读的代码里。范围收缩记档：pydantic-settings 只作 pydantic 本体的家族落点，**BaseSettings 组模型未启用**（.env 键位散点手写依旧够用，触发制另议）。验收 = 行为矩阵 **69 case 新旧对拍全平**（含异常消息逐字节）+ stages 四绿（指纹 `3502968fe6f7039e` 不变）+ live 探针与 D1 收工基线全平；requirements 里 pydantic==2.11.7 同步升出传递闭包区。
- **D6 语料格式解析器终裁 docling**（第④项，选型记录不实施）：候选 docling（IBM，MIT）vs unstructured —— 定 **docling**：① PDF/DOCX/表格版式还原强（TableFormer）且统一导出 markdown/JSON，正好接现有 `*.txt` 产物链；② 纯 python 库，无独立服务/API 形态分叉，与"进程外只有三个推理服务"的项目边界一致；③ MIT 干净。unstructured 落选：依赖链重、优先服务端/API 形态、表格还原弱。触发时的动作：requirements 加 `docling==<本地快照>`；解析入口放 build 阶段（`*.txt` 管线在解析产物之后不变）；**解析器版本升格进 `_CHUNKING_CODE_UNITS` 指纹输入**——换解析器 = 全量重建，把"同语料同产物"守住闸门里。
- **全量组件审计**（2026-11-11，"能外部的都外部了吗"复核）：手写逻辑逐区过筛后分三类。**已外部**（16 领域全在跑）：streamlit / requests / qdrant-client+Query API（含服务端 RRF 与 qdrant/bm25）/ infinity-emb / semantic-text-splitter / ftfy / OpenCC / pyahocorasick / cachetools / opentelemetry 全家 / supervisord / pydantic TypeAdapter + pydantic-settings / python-dotenv / psutil / tokenizers。**手写有硬理由**（换库负收益，逐条）：evaluate 指标（折叠去重+四层粒度是自定义口径，见 `_eval_row_metrics`）、文件锁陈旧接管+PID 复用判定（filelock 不覆盖）、upsert 立即重发（无睡眠，与退避模型相反）、别名切换验证重试、NO_PROXY 双向写入、Ollama NDJSON 流解析（非 SSE，sseclient 不适用）、cgroup v2 cpu.max（psutil 不暴露）、生成/检索显式无重试（设计哲学，§9）。**两项闲置依赖退场**：tenacity==9.1.4（全仓库零 import，重试点全是业务语义）、sentencex==1.0.30（分句/分块 2026-10-14 起已走 TextSplitter，清单忘删；`RAG_SENTENCE_LANG` 同步退役）。卸载后 stages 四绿指纹不变 = 零作用实锤。**剩余候选同日落地（2026-11-11，用户拍板即时换）**：evaluate 指标计分内核换 ir_measures/pytrec_eval（TREC 官方 trec_eval 的 C 实现）——`_eval_row_metrics` 签名/返回型不变，折叠去重、四层粒度、Δ(gold−raw)、hit@k(=Success@k) 口径全留适配层；**pytrec_eval 对同分按 docid 字典序重排（实测同分 `[zz,d1]` 把 d1 提前、RR 0.5→1.0）→ run 分数严格递减（-rank）让库读到真实序，tie 语义不进读数**；本地边界样例 **207 组新旧公式对拍全平（最大浮点差 0.00e+00，连浮点都同源）**，live 冒烟读数与换芯前逐键相同（live 仅 2 条且未踩 tie 位，样本有限；计分等价性由 207 组边界对拍背书——读数级对拍的噪声问题见下条）；依赖 ir_measures==0.4.3 + pytrec-eval-terrier==0.5.11 联动锁（适配层按 Measure 对象取值，不依赖随版本漂的字符串形态）。
- **检索读数非确定的两层噪声（2026-11-11 深反思发现）**：evaluate 连跑读数会漂——**(a) 服务端 RRF tie 决序**：Qdrant Query API 融合对并列 RRF 分（两文档各在一路 prefetch 同名次，归一化分逐位相等，实测 `0.5/0.5`）的排序不稳定，同进程连跑可互换 top1/top2（multi_turn lineno1 实测 3 连跑名次 2/2/1 交替，互换对为真值/非真值 → sources 层宏 mrr/ndcg ±0.02 级抖）；排除链 = embed 4 次逐位全等（差 0）→ 稠密路 8 次逐位全等 → 融合调用 tie 互换实锤 → 客户端 RRF 连跑两次全平（客户端按"首通道先见序"决并列 = 确定）。**(b) Qdrant 状态面偏移**：lineno9 跨天名次 3→4 但状态内三连跑全稳——tie 解释不了状态内稳定，应是 Qdrant 内部状态（如 segment 合并）改变固有名次，机制未深究。**影响与对策**：在线无感（互换文档都在最前）；**读数级对拍必须 `RAG_SERVER_FUSION=0`**（或把 tie 互换视为等价）；(b) 层在旧路径下同样存在——跨天对拍要么在同一状态窗口内完成，要么先跑已知金查询校准 Qdrant 状态再比；D1 当年"四组对拍全平"未踩抖动、结论仍成立，但"检索逐字节确定"这一对拍前提自此成文——今后任何读数对拍遇漂移，先分路定位（embed→单通道→融合）再定性，勿直接判回归。

### 10.3 故障矩阵（组件 × 当前行为，全部已取证）

| 组件 / 故障 | 现行为 | 凭据 |
|---|---|---|
| Qdrant 未起 / 崩 | 指纹与词表闸门 fail-closed 拒检索；`all` / start.sh 巡检退避重启 | 闸门 + `cmd_all` |
| Ollama 连不上 | connect 10s → `EVENT_ERROR`，本轮不入历史 | `stream_chat` |
| Ollama HTTP 5xx | 非 200 → `EVENT_ERROR HTTP xx`；流中 error 字段 → `[Ollama] …` | `stream_chat` |
| Ollama 卡死 | `RAG_STREAM_READ_TIMEOUT`（默认 300s）与 `RAG_TURN_TIMEOUT`（默认 1800s）；均 0 = 不限（"只能手动停止"是显式代价） | `stream_chat` docstring |
| 内存触线 95% | 生成中每次采样 → `EVENT_ERROR` 中止本轮（与"停止生成"同效） | `_mem_over_line` |
| http 嵌入/重排服务故障 | `_model_http_json` 三重合同 + 嵌入三条校验（条数/模型名/维度）→ `RuntimeError` 整轮失败，`st.error` + 不入历史，**无降级**（有意 fail-closed）；`RAG_MODEL_HTTP_TIMEOUT=60s` | `_model_http_json` |
| infinity boot 未就绪 | 两段式探测 → fail-closed 容器退出交平台重启 | `cmd_all` |
| 应用进程崩 | `all` / start.sh 巡检就地重启 | `cmd_all` |

### 10.4 A1 多轮检索改写：已实装（默认关）

- **上限收益量化**（`--compare-rewrite`，2026-11-10，20 条手标指代题 raw 与 gold 各跑一遍）：ΔRecall@5 来源层 **+0.45~0.55**、父块层 +0.30~0.40、块层 +0.20~0.25（真重排自行补偿一部分但仍显著）；"改写零收益"样本仅 6~8/20。
- **实现**：`rewrite_query`（`run_turn` 检索前置）——仅 `RAG_REWRITE=1` 且 history>0 触发（首轮零开销）；最近 `RAG_REWRITE_MAX_TURNS` 轮 + 本轮问交生成模型一次性小请求（温度 0、`num_predict` 封顶、think 关），产出独立完整查询再进 `hybrid_search`；**失败崩了 / 空输出 → 退回原词继续检索**（只告警，永不阻断本轮）；端到端冒烟：指代查询正确改写且首条来源命中（warm ≈1.4s，冷首载一次 ≈4.9s）。
- **台账/事件**：`turns.jsonl` 加 `rewrite_used / rewrite_ms / rewrite_chars`；`retrieval` 事件带 `rewrite` 元。
- **启用**：**留库存定档（2026-11-10）：默认 0**——`.env` 置 `RAG_REWRITE=1` 随时通电；想到它的时机 = 真实多轮使用里出现"指代问题召回很差"的体感时，可先 `evaluate --evalset evalsets/multi_turn.sample.jsonl --compare-rewrite` 复核收益再开。

### 10.5 显式不做与缓发（随 §9/§10.4 定档，重议条件见各行）

- **B1 会话持久化（缓发，2026-11-10 定档暂不做）**：聊天历史在 `st.session_state`（进程内存）——刷新页面/重启进程即清零，重启后还需重登；自用可接受。**触发信号：服务给他人日常使用或重启频繁时**，方案为按登录 ID 落盘（半天量级）。
- **B4 数据面**：无文档级权限（全库对登录者可见）、无限流（`OLLAMA_NUM_PARALLEL` 与内存是闸门）、语料正文不做提示词注入防护——三者与配额/多租户一道"有意不做"。
- **B5 日志留存**：`rag.log` 会记录用户提问与检索出的正文（10MB×3 轮转），自用可接受，未做脱敏/期限约定。
- **排队 / 账号隔离 / 多租户**：§9"内存是唯一闸门"的背面，全是有意不做。

### 10.6 触发制（记档不排期）

- **C1 索引增长**：HNSW/量化/mmap 全走默认，上量再测再定量。
- **C2 语料格式**：**已终裁 docling（§10.2-D6）**——触发（真来 pdf/docx 语料）即装即接，解析器版本入指纹链。
- **C3 查询级缓存**：**已落地（§10.2-D2，默认开）**——`RAG_QUERY_CACHE_TTL=0` 一键回收；`turns.jsonl` 复查命中率与收益后重议。
- **C5 上下文高级形态**（摘要压缩/选择性保留）：淘汰式够用，不做。
- **开关档案**：`RAG_REWRITE=0` 留库存（§10.4）；`RAG_TURN_STATS=1` 常开（§10.2-A2）；`RAG_SERVER_FUSION=1`（§10.2-D1，0=恒走旧路径）；`RAG_OTLP_ENDPOINT` 未设 = 观测零成本 no-op（§10.2-D3）。
- **D5 配置校验**：**已落地（§10.2-D5，2026-11-11）**——`env_int`/`env_float` 换 pydantic TypeAdapter 内核（pydantic-settings 家族），签名/文案逐字节不变、69 case 行为矩阵对拍全平；BaseSettings 组模型 = 触发制（.env 键位散点手写依旧够用）。
