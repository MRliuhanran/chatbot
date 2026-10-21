---
# 详细文档见https://modelscope.cn/docs/%E5%88%9B%E7%A9%BA%E9%97%B4%E5%8D%A1%E7%89%87
domain: #领域：cv/nlp/audio/multi-modal/AutoML
# - cv
tags: #自定义标签
-
datasets: #关联数据集
  evaluation:
  #- iic/ICDAR13_HCTR_Dataset
  test:
  #- iic/MTWI
  train:
  #- iic/SIBR
models: #关联模型
#- iic/ofa_ocr-recognition_general_base_zh

## 启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
# deployspec:
#   entry_file: app.py
license: apache-2.0
---
#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/studios/liuhanran/chatbot.git
```

# RAG 垂直领域知识库问答服务

单文件 Python 应用（`chatbot.py`）实现的检索增强生成（RAG）服务：对本地语料分块、建立稠密+稀疏双通道索引，问答时双路召回 → RRF 融合 → 重排 → 拼装上下文 → 流式生成。前端为 Streamlit，向量库为 Qdrant，生成模型经 Ollama 运行。

> 本文档描述系统结构与使用方式，并记录关注领域与缺口（§11）。文中行号为 2026-10-15 版本快照，仅作导航参考（§11 锚函数名，不受行号快照影响）。

## 1. 系统组成

| 组件 | 职责 | 形态 / 地址 |
|---|---|---|
| `chatbot.py` | 全部业务逻辑与唯一入口（CLI + Streamlit） | 本地 `:8501`，魔搭容器 `:7860` |
| Qdrant | 稠密向量 + BM25 稀疏向量存储 | `localhost:6333`，经别名 `rag_current` 访问 |
| Ollama | 生成模型（`minicpm5-2b`，Q4_K_M） | `localhost:11434` |
| 检索模型 | 嵌入 `bge-m3` + 重排 `bge-reranker-v2-m3` | 默认进程内 torch（MPS/CPU）；`RAG_*_BACKEND=http` 时外置为 infinity 服务（`:8001`） |

本地四组件用 `tools/run_local.sh {start|stop|status|logs}` 一键起停，全部原生进程，不依赖 Docker。

## 2. 目录结构

| 路径 | 说明 |
|---|---|
| `chatbot.py` | 业务代码单文件（约 4100 行），按依赖顺序分节，见第 3 节 |
| `.env` | 配置生效值（唯一配置权威，见第 7 节） |
| `corpus/` | 语料目录（默认 `RAG_DATA_DIR`，`*.txt`） |
| `artifacts/` | 离线产物：`chunks.json` + `chunks.meta.json`（分块与切分指纹）、`docs/`（逐文档分块缓存）、`embeddings/`（稠密向量缓存）、`gen_tokenizer/`（生成模型 tokenizer） |
| `lexicon/` | 领域词表：`aliases.txt`（别名归一）、`stopwords.txt`，整表替换即可换领域 |
| `models/` | 本地模型权重（嵌入、重排） |
| `storage/`、`qdrant_config/` | Qdrant 数据与配置 |
| `tools/` | 运维与验证脚本，见第 9 节 |
| `Dockerfile`、`entrypoint.sh` | 魔搭创空间单容器（qdrant + ollama + streamlit 三进程） |
| `docker-compose.yml`、`docker-compose.gpu.yml`、`embed.Dockerfile` | 容器化路径（本地 Docker / 自有服务器；gpu 覆盖仅把生成模型交给 GPU） |
| `ms_deploy.json` | 魔搭部署规格（docker 类型、2vCPU/8G、端口 7860） |
| `rag.log` | 运行日志（10MB × 3 轮转） |

## 3. `chatbot.py` 文件结构

分节顺序即数据流顺序，节分隔符为 `# ── 名称 ──`：

| 节 | 行号 | 职责 |
|---|---|---|
| 基础设施：环境配置与日志 | 1 | `env_*` 配置解析（非法值启动即报错）、双写日志 |
| 运行态单例 | 184 | `_RuntimeState`：模块级缓存/锁（会话、tokenizer、词表） |
| 采样/惩罚、生成客户端 | 239 | `_chat_options` 单点组装 Ollama options；`stream_chat` 流式请求 |
| 单例协调（模型互斥驻留） | 424 | `RAG_EXCLUSIVE_MODELS`：检索与生成模型交替装卸 |
| 领域与检索侧配置 | 494 | 语料/来源字段/检索/嵌入后端等配置常量 |
| 模型加载与向量化 | 728 | 嵌入/重排模型懒加载、`embed()`、设备选型 |
| 分块 | 830 | 归一化（ftfy + OpenCC）→ `semantic-text-splitter` 父子分块 → `build_chunks()` |
| 分块产物与门禁 | 1179 | `chunks.json` 读写、切分指纹、`verify_chunks_freshness()` |
| 词表 | 1301 | 别名归一、停用词、词表指纹 |
| 索引构建 | 1746 | 内容哈希嵌入缓存、稠密+稀疏双写、建索引、原子切别名 |
| 检索原语 | 2047 | `dual_channel_recall` 双通道并发召回、`rerank_indices` |
| 生成装配 | 2195 | `plan_generation` / `build_generation_messages`：token 硬上界、历史整轮淘汰 |
| 检索引擎 | 2565 | `RAGEngine.hybrid_search`：闸门 → 召回 → RRF → payload 回取 → 去重 → 重排 |
| 单轮编排 | 2969 | `run_turn`：一轮问答全流程与事件流 |
| 交互入口：Streamlit | 3127 | 页面、会话历史、Stop 按钮（确定性中止生成） |
| 检索引擎单例与预热 | 3318 | `_warm_engine` 后台预热 |
| 工具节 ×4 | 3391 | `compare-chunks`、`sparse`、`lexicon`、`stages` |
| 命令分发 | 4026 | `COMMANDS` 表与 `main()` |

## 4. 离线流水线

```
corpus/*.txt
   │ python chatbot.py process        归一化 → 父子分块（逐字无损）
   ▼
artifacts/chunks.json (+.meta.json 携带切分指纹；docs/ 逐文档缓存)
   │ python chatbot.py index          指纹校验 → 向量化(内容哈希增量) → 稠密+稀疏双写 → 建字段索引 → 原子切别名 → 删旧集合
   ▼
Qdrant 集合（别名 rag_current）

python chatbot.py sparse              仅重建稀疏向量与词表指纹（换词表时）
python chatbot.py stages              四阶段指纹比对：A 分块 / B 稠密 / C 稀疏 / D 入库
                                      输出最小重跑命令；退出码 0 = 全新鲜，1 = 有阶段过期或未验证
```

产物由指纹绑定：**切分指纹**（切分代码 + 分块参数 + 源文本 + 模型内容指纹）写入 `chunks.meta.json`，**词表指纹**写入索引 payload，**稠密缓存**按 `模型内容指纹_嵌入代码指纹_精度` 命名。任一不匹配时门禁 fail-closed（报错而非沿用旧产物）。

增量语义：`process` 按文档内容 sha 缓存，未变文档整篇复用；`index` 按块文本哈希复用向量。未变部分不重算。

## 5. 在线问答链路

```
提问 → 空查询守卫（纯标点/空白必拦）
     → 闸门：切分指纹 + 词表指纹（fail-closed）
     → 双通道召回：稠密 + 稀疏并发查询（零权重通道不融合也不召回）
     → 客户端加权 RRF → 按命中 id 精确回取 payload → 父块去重
     → rerank（对全体候选统一打分）
     → 上下文装配：资料以结构化 `<document index= source= id= chunk= score=>正文</document>`
       注入（RAG_CONTEXT_META=1）；system 恒为空，资料恒以 user 角色随本轮提问发出（不寄生 system）
       历史超预算按整轮淘汰最旧；资料按整条从尾部丢
     → Ollama 流式生成（事件：retrieval / thinking / delta / error / done）
     → 仅 done 时写入会话历史；失败不入历史
```

设计约定：生成**无长度/时间闸、不重试**（上下文窗口是唯一上限）；检索为空也照常生成（无拒答逻辑）；Stop 按钮经 `close_events()` 关闭上游流确定性中止。

## 6. 命令

```bash
python chatbot.py process          # 分块（改语料/切分参数后）
python chatbot.py index            # 建索引（改模型/新增内容后）
python chatbot.py sparse           # 重建稀疏向量（改词表后）
python chatbot.py stages           # 四阶段新鲜度检查，给出最小重跑命令
python chatbot.py lexicon          # 词表自检
python chatbot.py compare-chunks A [B]   # 分块产物 A/B 比较
python chatbot.py serve             # Web 界面（先校验索引再启动 Streamlit，:8501）
```

## 7. 配置

配置以 `.env` 为准：代码默认值权威，同名环境变量覆盖，非法值启动即报错。主要分组（完整列表见 `.env` 注释）：

| 分组 | 代表变量 |
|---|---|
| 语料与领域 | `RAG_DATA_DIR`、`RAG_DOC_GLOB`、`RAG_SOURCE_FIELD`、`RAG_SOURCE_LABEL` |
| 分块 | `RAG_PARENT_MAX_TOKENS`(512)、`RAG_CHILD_MAX_TOKENS`(128)、`RAG_CHUNK_OVERLAP`(32) |
| 检索 | `RAG_TOP_K`、`RAG_RERANK_TOP_K`、`RAG_RERANK_BATCH`、`RAG_RRF_SPARSE_WEIGHT` |
| 检索模型后端 | `RAG_EMBED_BACKEND` / `RAG_RERANK_BACKEND`（`torch`=进程内，`http`=infinity 服务）、`RAG_EMBED_URL`、`RAG_EMBED_MODEL_PATH`、`RAG_RERANK_MODEL_PATH` |
| 生成 | `MODEL`、`RAG_NUM_CTX`、`RAG_NUM_PREDICT`(0=不限长)、`RAG_THINK`、`RAG_TEMPERATURE`(0=贪心)、`RAG_GEN_NUM_GPU`（`auto`=GPU 优先失败回落 CPU）、`OLLAMA_TIMEOUT` |
| 内存与驻留 | `RAG_EXCLUSIVE_MODELS`（模型互斥驻留）、`RAG_GEN_KEEP_ALIVE`、`RAG_PRELOAD` |
| 索引与路径 | `RAG_COLLECTION_BASE`、`RAG_COLLECTION_ALIAS`、`RAG_ARTIFACTS_DIR`、`RAG_LEXICON_DIR` |

## 8. 部署

| 路径 | 说明 |
|---|---|
| 本地原生（默认） | `tools/run_local.sh start` 依次拉起 Qdrant / Ollama / 应用（GPU 档：Metal + MPS），已在跑的实例复用 |
| 容器（本地 Docker / 自有服务器） | `docker-compose.yml` 四容器：qdrant / ollama / app / embed；NVIDIA 主机叠加 `docker-compose.gpu.yml` |
| 魔搭创空间 | 平台不支持 compose：仓库根 `Dockerfile` 单容器跑三进程，`entrypoint.sh` 拉起 + 健康等待 + 存活哨兵，仅暴露 7860；规格见 `ms_deploy.json` |

发布前闸门：`tools/deploy-check.sh`。

## 9. tools/ 脚本

| 脚本 | 用途 |
|---|---|
| `run_local.sh` / `run_qdrant.sh` | 本地组件一键起停（Qdrant 原生二进制版本与 `Dockerfile` 同源） |
| `deploy-check.sh` | 发布前检查（lint、产物、索引连通、进程内冒烟检索） |
| `fetch_models.sh` | 按 `.env` 的模型路径查表下载权重（构建期/容器内） |
| `eval_retrieval.py` | 检索基线对拍（来源命中率 / top-k 一致性） |
| `rerank_parity.py` | 进程内重排与 `/v1/rerank` 分数比对 |
| `prepare_infinity_view.py` | 为 infinity 生成模型影子视图（不动真实权重目录） |
| `build_gen_tokenizer.py` | 从 GGUF 离线抽出生成模型 tokenizer |

## 10. 约束与不变量

- **指纹区禁改**：`chatbot.py` 中指纹相关注释与常量是切分指纹的输入，改动 = 全量重建；`read_book_text` 名字是 `_CHUNKING_CODE_UNITS` 成员，改名即改指纹。
- **核心不变量**：切分逐字无损；索引点数 = 产物记录数；空查询必拦；重排对全体候选统一打分；payload 按命中 id 精确回取；资料不寄生 system；生成装配估算 token 恒 ≤ 输入上界；生成失败不入历史。
- **变更代价**：改切分/分块参数 → `process` + `index`；改语料 → `process` + `index`；改词表 → `sparse`；改模型 → `index` + 重新度量；改检索参数 → 重新度量。以 `python chatbot.py stages` 输出为准。
- **不支持并发访问**（单用户本地/单容器形态；多用户规划见 §11 B 档，属改不变量的版本迁移）。
- **第三方依赖全部惰性导入**（模块级仅标准库，`dotenv` 例外且须先于配置常量）。

## 11. 关注领域与缺口（基准日 2026-10-21，锚函数名不锚行号）

> 本节只记"盯什么、缺什么、何时动"，随需求变更更新；操作与结构见 §1–§10。

### 11.1 六大领域 × 现状

| 领域 | 已闭环 | 缺口 |
|---|---|---|
| 模型内存 | 互斥驻留（`RAG_EXCLUSIVE_MODELS`、`RAG_GEN_KEEP_ALIVE`）、MPS 分配闸门、KV 档位 `.env` 单点管 | 并发下总预算 → B3 |
| 检索构建（离线） | 四阶段 + 指纹 fail-closed、内容哈希增量缓存、原子切别名 | 规模增长应对 → C1 |
| 检索查询（在线） | 空查询守卫 → 双路召回 → RRF → 按 id 回取 → 父块去重 → 重排 | 多轮查询未改写 → A1 |
| 构建优化 | 嵌入/分块缓存 + 离线度量（`quant_bench`、`hw_profile`、`quant_parity`） | 每轮在线指标 → A2 |
| 多轮上下文 | 令牌硬上界、整轮淘汰、失败不入历史（`_truncate_history_to_budget`） | 仅淘汰一策，当前够用 → C5 |
| 多用户并行 | 仅浏览器会话隔离；§10 明文"不支持并发访问" | B 档整组（版本迁移） |

### 11.2 缺口清单

**A 档：单用户形态就有价值**

| # | 缺口 | 处置 |
|---|---|---|
| A1 | `run_turn` 直用原始 query 进 `hybrid_search`，历史不参与检索："那它呢？"类指代必差，多轮×检索间缺桥 | 短提示词把"历史+当前问"改写成独立完整查询；命中率变化由 A3 验收 |
| A2 | 检索/重排耗时、TTFT、上下文取舍数无结构化记录，一切优化结论不可证 | `run_turn` 采固定字段 → 轮转 JSONL → 工具出摘要 |
| A3 | `eval_retrieval.py`（重建前后自拍）与 `rerank_parity.py`（分数对拍）都答不了"答得好不好" | 30~50 条 golden QA（问题+期望来源+关键点），进发布闸门 |
| A4 | 生成侧"不重试/不拒答/无时间闸"是显式设计但未录缘由，出现答非所问时无决策依据可查 | 200 字决策记录落档，改动先过它 |
| A5 | 故障"体感"未定义：组件挂 = 重启/报错气泡/空白？Ollama 卡死是否无限等待？ | 故障 × 行为 × 期望矩阵，改码顺手补齐 |
| A6 | Stop 只关上游连接（`close_events`），Ollama 服务端跑完本轮才释放模型 | 调研会话级取消；否则监控连续 Stop 的资源占用 |
| A7 | 模型选择键三源（`.env`/`tools/bake.env`/Dockerfile）靠人工同步，已出过一次清单漂移 | `tools/deploy-check.sh` 加三源一致性断言 |

**B 档：动多用户前的版本迁移**（连带改 §10"不支持并发访问"不变量，README、单容器形态、部署规格同步重审）

| # | 缺口 | 处置 |
|---|---|---|
| B1 | `st.session_state` 刷新/重启即失；登录门 `RAG_LOGIN_ENABLED=0` 当前关 | 会话按登录 ID 持久化；登录门变硬前提 |
| B2 | 互斥驻留靠锁串行：并发即排队——上限/公平/超时/装卸抖动全未定义 | 核心技术风险位：排队与背压设计 |
| B3 | KV cache 随并发×131K 窗口增长；三进程同容器 8G（2vCPU） | 先立 KV≈f(并发, ctx) 预算公式，再定并发上限（估 1~2） |
| B4 | 全库对所有用户可见、无限流、资料正文未防提示注入 | Qdrant 租户 filter；限流与 B2 合流计数；提示词隔离约定 |
| B5 | `rag.log` 全记用户查询与资料正文（10MB×3） | 启用前定留存/脱敏策略，写进交付说明 |

**C 档：规模/形态变化才触发（记档不排期）**：C1 索引增长（HNSW/量化/mmap 全走默认，上量再测再定量）；C2 语料格式（现仅 `txt`，引 pdf/docx 时解析环节进入指纹链）；C3 查询级缓存（A2 数据说话）；C4 冷启动预热（先观测首问延迟）；C5 上下文高级形态（摘要压缩/选择性保留）。

### 11.3 执行顺序与冲突提醒

- **P0**：A2（其余一切的测量前提）、A7（有真实事故背书，成本极低）。
- **P1**：A3 → A1（先立验收尺，再改检索行为）；A4/A5/A6 各 2h 级。
- **触发条件**：B 档以"决定做多用户"的评审为起点；C 档随对应变化启动。
- **冲突提醒**：改生成侧"不重试/不拒答/无闸"（§5）必先过 A4 决策记录；A3 接入 `tools/deploy-check.sh` 时与指纹新鲜度检查分开执行，不得变成每次提交必跑。

开发校验：`ruff check chatbot.py`、`python -m py_compile chatbot.py`、`bash -n tools/*.sh`。
