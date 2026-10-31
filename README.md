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
| `./start.sh index` | 建索引（成功收尾写 `artifacts/index_manifest.json`） |
| `./start.sh verify-build` | 构建期防呆：校验 `artifacts/index_manifest.json` 的 `chunk_fingerprint` 与当前语料/切分配置现算指纹一致（不等退出非 0；Dockerfile 与发布脚本都调它） |
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

`artifacts/index_manifest.json` 是索引的**发布清单**：`index` 成功收尾（点数量不变量与别名切换都通过之后）写入 `chunk_fingerprint` / `lexicon_id` / 集合名 / 分块数 / 模型内容指纹 / Qdrant 服务端版本；`verify-build` 与发布脚本以它的 `chunk_fingerprint` 为**唯一 EXPECTED_FP 来源**。它与 `storage/`、`chunks.meta.json` 是同一次 index 的产物，必须一起入仓、一起进镜像——这是部署侧“数据落后于应用”事故的根治点（见 §7）。清单缺失/读不到 = 发布前置失败，不允许“没有清单就放行”。

## 7. ModelScope 部署：一个 Dockerfile

本地运行 = 源文件 + start.sh；ModelScope 运行 = **一个 Dockerfile + 一个容器**。容器内 qdrant / ollama / streamlit 三进程由 `chatbot.py all` 管理（源码里的进程管理器）；**http 后端多托一个 infinity 作为第 4 进程**（2026-10-30 起 `.env=http/http`：应用层 pip 装 `infinity-emb[torch,server]==0.0.76` + `psutil` + `optimum==1.22.0`（bettertransformer 默认开，缺 optimum → acceleration.py NameError rc=3 起即崩）+ `click==8.1.8`（typer<0.13 的解析关）+ `torch==2.7.1` 锁本，并 COPY 补齐 bge-m3 缺失的 `1_Pooling/config.json` 池化元文件——缺它 ST 引擎把路径当 HF repo id 启即崩）。`cmd_all` 按 start.sh 同构 argv 编排：app 依赖就绪后立即放量（平台健康探测打 7860），infinity 就绪由后台线程两段式探测（health → embeddings+rerank 真打一发 ≤ `RAG_INFINITY_WAIT`），未就绪 fail-closed 退出交平台重启。不依赖 compose。

**镜像分两层（2026-10-31 定案，指纹事故的根治）**：base 只放“重且不常变”的二进制与模型（qdrant/ollama、python 依赖、双检索模型权重、生成模型 GGUF）；`.env`/`start.sh`/`chatbot.py`/`1_Pooling` 与**全部可变数据**（`storage/`、`corpus/`、`lexicon/`、`qdrant_config/`、`artifacts/index_manifest.json` 及同代的 `artifacts/chunks.meta.json`）在同一次发布的应用层里一起 COPY，仓库是唯一事实源。此前 base 烤了 10-28 的旧 `storage/artifacts`、应用层却只换代码 → 数据永远落后于应用：2026-10-31 远程启动期分块指纹 `索引=923236fe7e4acc7c / 当前=3502968fe6f7039e` 不一致，检索闸门 fail-closed。可变数据随应用层发布后这个结构性缺口消失，另有构建期 `verify-build` 兜底。

### 7.1 Dockerfile 约束

- 所有 `RUN` 都是 **exec form 单行 JSON 数组**：`RUN ["pip","install",...]`，不经过 `/bin/sh -c`。**每个 RUN 的 JSON 数组必须写在同一行**——魔搭 dockerfile parser 会把跨行续行当成独立指令（2026-10-31 23:27 场 parse error line 20 实证）；
- 唯一入口：`ENTRYPOINT ["python","chatbot.py","all"]`（start.sh 只作为运行期模型下载器进镜像）；
- qdrant / ollama 二进制直接从官方镜像 COPY，不在 Dockerfile 里下载/解压；
- 层顺序固定：**base → 应用文件（.env/start.sh/chatbot.py/池化元文件）→ pip RUN → 可变数据层 → verify-build**。数据 COPY 必须在 pip RUN 之后，改数据才不会重跑 16min 级依赖层；torch 重件留在 base；
- 数据层先 `RUN ["rm","-rf","/app/storage","/app/corpus","/app/lexicon"]` 再 COPY：`COPY 目录` 是 merge 语义，不删会留下 base 旧 segment / 已删语料，容器内现算指纹与清单永远对不上；
- 构建最后一步 `RUN ["python","chatbot.py","verify-build"]`：清单指纹 ≠ 现算指纹直接构建失败（fail-fast，不把问题留到运行期闸门）；
- `HEALTHCHECK` 也是 exec form（`python -c ...`）。

> pip 等包管理器内部仍会执行自己的脚本，但 Dockerfile 本身没有 shell 形式的命令行。

### 7.2 构建与运行

**模型三层部署（2026-10-27 最快化定案；属于 base 的构建期，日常发布不重跑）**：① 镜像层烤权重——`fetchmodels` stage 构建期拉魔搭 CDN，组级层缓存命中即零下载（`ARG MODELS_REV` 不变即命中）；② 持久盘 `/mnt/workspace` 暖 ollama blobs（重启零拷贝）；③ 运行期兜底 `cmd_all → _ensure_runtime_models`——镜像层/持久盘任一缺失才用 `start.sh fetch-models` 补齐（缺组才下、字节级幂等，绝不启动死循环）。Running 后启动期即打印 `🔧 分块指纹`（发布脚本轮询该行做验收，上限 300s，不再靠固定 sleep 猜）；只有双层尽失的极端场景才先下 4.9G。

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
git add artifacts/index_manifest.json artifacts/chunks.json artifacts/chunks.meta.json
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
- 单容器 8G 档注意 `RAG_NUM_CTX`；OOM 时 `all` 会重启被杀的子进程（与本地 start.sh 的巡检语义一致）。

### 7.5 并行

- **构建**：`pydeps`（依赖）/ `fetchmodels`（烤模型）/ `assets` 三个 stage 只在**重烤 base** 时并行；日常发布是单 stage 的应用层：应用文件 → pip RUN（命中缓存）→ 可变数据层 → verify-build，只重打数据层。
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
3. 无 `RuntimeError`、无 `❌ [all]`（启动编排失败）；infinity 就绪行尽力检查；启动期 `🧮 词表指纹` 行仅记录不阻断。

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

- 单机、单用户；无多租户、无并发队列。登录门是**自报手机号识别**（无密码/短信/IdP），不是认证；会话存进程内，重启即全员重登。
- 不限制在线人数与并发数：内存是唯一闸门（`RAG_MEM_*`）。生成中触到水位线即中止本轮（等同"停止生成"，残缺答案不入历史）；不排队、不拒绝登录、不卸载模型（驻留交给 Ollama keep_alive）。
- 语料仅支持 `*.txt`。
- 生成侧不重试、不拒答、无输出长度闸（窗口是唯一上限）；这是显式设计。
- 没有测试框架与 CI；质量靠指纹闸门、`start.sh stages` 与人工验收。检索质量可用离线评估
  `evaluate` 命令（手标评测集，只测检索链路、不依赖 Ollama），但它不是常驻的回归防线，评测集是人工维护的。
