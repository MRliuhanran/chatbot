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

> 本文档只描述系统结构与使用方式。文中行号为 2026-10-15 版本快照，仅作导航参考。

## 1. 系统组成

| 组件 | 职责 | 形态 / 地址 |
|---|---|---|
| `chatbot.py` | 全部业务逻辑与唯一入口（CLI + Streamlit） | 本地 `:8501`，魔搭容器 `:7860` |
| Qdrant | 稠密向量 + BM25 稀疏向量存储 | `localhost:6333`，经别名 `rag_current` 访问 |
| Ollama | 生成模型（`qwen3.5:4b-q4_K_M`） | `localhost:11434` |
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
     → 上下文装配：parent_text + <document> 边界标签 + [n]（来源：xxx）标头
       system 恒为空字符串；整条 prompt 估算 token ≤ RAG_NUM_CTX − RAG_INPUT_RESERVE
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
- **不支持并发访问**（单用户本地/单容器形态）。
- **第三方依赖全部惰性导入**（模块级仅标准库，`dotenv` 例外且须先于配置常量）。

开发校验：`ruff check chatbot.py`、`python -m py_compile chatbot.py`、`bash -n tools/*.sh`。
