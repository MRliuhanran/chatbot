# ARCHITECTURE.md —— 配置驱动的多数据集对话机器人（定稿）

> **状态**：定稿（按 2026-09-19 决策会议修订）。
> **本轮不修改任何代码**，只交付本文档。
> **修订说明**：初稿提出了一套 `DomainPack` 协议 + 元数据重构 + 探针域化 + 架构不变量，
> 属于**过度设计**。本文档已按你的真实意图大幅收敛，被砍掉的部分记在 §7，并说明为什么砍得对。

---

## 1. 你的意图（先复述准，再谈方案）

拆成两条流水线，**都只由一个配置文件决定**：

```
【离线】配置文件指定文件夹 ──▶ 清洗/分块 ──▶ 向量化 ──▶ 写入某个「向量库分区」
【在线】配置文件指定分区   ──▶ 算相似度召回 ──▶ 装配 prompt ──▶ 生成
```

- **通用**：检索链路、融合、重排、prompt 装配、token 预算、生成、前后端接口、参数。
- **垂直**：*数据从哪个文件夹来*、*去哪个分区读*。
- 你说的"配置参数提前离线清洗好数据" = 清洗是你自己的工序，平台只负责"按配置读那个文件夹"。

**本文档的全部内容，就是把上面这张图变成可运行的配置项。**

---

## 2. 核心问题：向量库要分吗？

**要分，但不需要发明任何新概念——Qdrant 的 collection 就是分区。**

你说的"去哪个向量化的数据库部分读"，在 Qdrant 里的正确实现就是**配置指定 collection 名**。这件事你已经在做了：现在跑着的就是 `books_v3__71371e82` 这个 collection。

### 2.1 为什么必须物理分 collection（不能靠 payload/filter 区分）

| # | 原因 | 依据 |
|---|---|---|
| 1 | **BM25 的 IDF 是按整个集合统计的** | Qdrant 正为此在做 [per-query IDF corpus](https://github.com/qdrant/qdrant/pull/9661)（PR #9661）。两套语料混装 → 同一个词的 IDF 被另一套的文档频率污染 → 稀疏打分失真，**且不报错** |
| 2 | **向量维度与模型必须全集合一致** | 换 embedding 必须换集合——AGENTS.md 已写明 `books_v3` 这个名字即由此而来 |
| 3 | **别名原子切换是 per-collection 的** | `books_current` 只能指向一个集合；混装则"重建 A"必须停掉 B |
| 4 | **分块指纹门禁是全局一份** | `cache_v2/chunks.meta.json` 只有一份 → A 改配置会逼 B 重建；更糟的是 B 的产物被 A 的指纹判定，**静默放行或静默失败** |
| 5 | **拒答阈值是相对分位标定，不是绝对分** | `-1.5 / 0.5` 是在四大名著的分数分布上标的；混装后分布被平均，拒答永久失效 |

### 2.2 反过来说：什么时候**可以**同集合？

判据只有一条——**稀疏通道的词表与 IDF 能不能共享**：

| 场景 | 做法 |
|---|---|
| 同语体、同词表（四大名著的 4 本书） | **同集合 + `book` filter 区分**（现状正确，别动） |
| 不同语体 / 不同词表 / 不同模型（四大名著 ↔ 项目文档） | **必须分集合** |

所以准确说法是：**不是"分域"，是"分集合"**。这也是为什么本文档不引入"域"这个新概念——它只会让人以为要写一套插件框架。

---

## 3. 最小设计：一个 profile = 一个数据集

**不叫"域"，因为它的确只是一份"数据源 + 读法"的声明。** 一个 YAML 文件，没有插件、没有协议、没有注册表。

```yaml
# profiles/classical_novels.yaml
name: classical_novels
display_name: 四大名著

# ---- 离线：读哪里、产物写哪里 ----
source_dir: books                          # 源文件夹
include: ["*.txt"]
artifacts_dir: cache/classical_novels      # 分块产物 + 向量缓存（隔离，防互相覆盖）

# ---- 在线：读哪个集合 ----
collection: books_v3                       # = 分区
alias: books_current                       # 检索端走别名，重建时原子切换

# ---- 可选资产（缺省即关闭）----
lexicon_dir: data                          # 为空 = 无词表，稀疏通道跳过归一
prompts_dir: prompts                       # prompt 文件所在目录

# ---- 参数（不写则用平台默认）----
retrieval:
  top_k: 5
  rerank_top_k: 20
  rrf_k: 60
  dedup_by_parent: true
generation:
  model: qwen3.5:4b-q4_K_M
  num_ctx: 8192
  num_predict: 4096
```

### 3.1 三条命令共用同一个 profile

```bash
python app.py process --profile=classical_novels   # 读 source_dir → 分块 → artifacts_dir
python app.py index   --profile=classical_novels   # 向量化 → 写入 collection
python app.py serve   --profile=classical_novels   # 在线：只读该 collection
python api.py         --profile=classical_novels   # 同上，HTTP 形态
```

也可以用环境变量：`RAG_PROFILE=project_docs python app.py serve`。

### 3.2 换一个数据集 = 加一个 YAML

```yaml
# profiles/project_docs.yaml
name: project_docs
source_dir: docs
include: ["*.md"]
artifacts_dir: cache/project_docs
collection: project_docs_v1
alias: project_docs_current
lexicon_dir: ""            # 无词表
```

然后 `process` + `index` + `serve` 三条命令照跑。**新数据集不需要改任何 Python 代码**——这正是你要的"通用"。

---

## 4. 需要改的代码清单（带行号，可直接施工）

### 4.1 必须改（不改则两个数据集会互相破坏）

| # | 位置 | 现状 | 改成 |
|---|---|---|---|
| 1 | `rag_engine.py:242` | `CHUNKS_JSON = "./cache_v2/chunks.json"` **硬编码，连 env 都没有** | 由 profile 的 `artifacts_dir` 派生 |
| 2 | `rag_engine.py:262` | `EMBED_CACHE_DIR = "./cache_v2/embeddings"` 硬编码 | 同上 |
| 3 | `rag_engine.py:266` | `BM25_CACHE_DIR = "./cache_v2"` 硬编码 | 同上 |
| 4 | `rag_engine.py:225` | `COLLECTION_NAME = "books_v3"` 字面量 | 由 profile 的 `collection` 读 |
| 5 | `rag_engine.py:231` | `COLLECTION_ALIAS` 仅 env | 由 profile 的 `alias` 读 |
| 6 | `rag_engine.py:248` | `_chunks_meta_path()` 硬编码 `./cache_v2/` | 随 #1 一起走 `artifacts_dir` |
| 7 | `rag_engine.py:1172` | `_books_fingerprint` 只扫 `BOOKS_DIR/*.txt` | 扫 profile 的 `source_dir` + `include` |
| 8 | `rag_engine.py:124` | `BOOKS_DIR` 仅 env | 纳入 profile |
| 9 | `rag_engine.py:268` | `LEXICON_DIR` 仅 env，**没有"无词表"这个状态** | 纳入 profile，允许为空 = 关闭词表 |
| 10 | 全局 | 无 profile 加载器，各处各自 `os.getenv` | 新增 `load_profile()`：解析 + **未知键报错** + 校验 |

> **#1–#3 是最要命的三条。** 现在 `chunks.json` 44MB 是硬编码单一路径，两个数据集会直接互相覆盖，然后被指纹门禁硬失败——而失败信息指向"请重跑 process"，完全看不出真正原因是**路径撞车**。

### 4.2 可延后（不做也不破坏隔离）

| 项 | 何时必须做 |
|---|---|
| `rag_engine.py:1105` `_chunking_code_fingerprint()` **硬 import `chapter_parse`** | **引入第二种解析器的那一天**（现在所有数据集都是 `.txt`，可以不动） |
| 恢复 prompt / 接入检索结果（P0） | **已做**，2026-09-19：规则进 system、正文走独立消息，见 §5 |
| `book` 元数据降级 / 探针域化 / 事件流统一 / 架构不变量 | 暂不考虑（§7） |

---

## 5. P0 与文档漂移（§5.1 已于 2026-09-19 处理）

### 5.1 P0 事实（记录在案，**现已修复**）

`rag_engine.py:2331-2336` 三段提示词为空串，`build_system_prompt()` **所有分支都返回 `""`**（本轮已实测：`assembled len = 0`）。

**当时的决策：暂不动 prompt 内容。** 后果必须明确记录：

> **当前生成侧未接入检索结果。** 检索链路完整运行、推理链完整展示，但召回的正文不进模型，回答全部来自模型参数化记忆。UI 上表现完全正常，无法从回答分辨。

本文档保留此条，避免"因为没人提就以为已修好"。

**2026-09-19 的修法（不是简单把三段填回去）**：正文**不再拼进 system**，
改为 `build_system_prompt` 只返回规则、`build_context_message` 把正文装配成
一条独立消息（默认 role=`user`，见 `RAG_CONTEXT_ROLE`），装配顺序
`system → 历史 → 资料 → 本轮提问`；无结果时三个入口统一硬拒答
（`NO_CONTEXT_ANSWER`）。这样"资料进没进 prompt"变成可直接断言的事实，
而不再依赖"system 里有没有某个子串" —— 后者正是这次失效能静默发生的原因。

### 5.2 文档漂移

**事实**：AGENTS.md 本次更新已写明"生成侧**已真正接入**检索结果"，而代码仍是空的——同类问题**第四次**（前三次：`RAG_QDRANT_HOST`、`RAG_ABSTAIN_RATIO`、D9 import 顺序）。

**决策**：
- **规则采纳**：AGENTS.md 中凡是"状态声称"，必须是**可被工具验证**的陈述，否则不写进文档。
- **实际修改延后**：与阶段 1（prompt 恢复）一起做，不单独动。
- **2026-09-19 更新**：该条已随 §5.1 一并落地；措辞也改了 —— 不再声称"已接入"，
  而是把**装配顺序**（system → 历史 → 资料 → 提问）写出来，它可被直接核对。

---

## 6. 自测（本次决策：先简单自测就行）

不搞三道门，用**一条**即可证明隔离成立：

> 建两个 profile，各跑一遍 `process` + `index`，然后互相查：
> 1. A 的检索结果里**不得出现** B 的任何内容；
> 2. A 的产物文件**不得被 B 覆盖**（对比 `artifacts_dir` 下的文件与指纹）；
> 3. 切 profile 重启后，集合名与点数正确切换。

这一条同时覆盖了 §4.1 里 #1–#7 全部七项改动，性价比最高。要做更严格的回归时再补 §7 提到的逐字节对拍。

---

## 7. 砍掉的部分（初稿的过度设计）及原因

诚实记录，避免以后有人重新提出又被绕进去：

| 初稿提出 | 砍掉的理由 |
|---|---|
| `DomainPack` 协议（7 个方法的插件接口） | 你要的是"配置读哪个文件夹"，不是"写插件框架"。**一个 YAML 覆盖 90% 场景**，真需要自定义解析器时再加 |
| `book` 从 schema 一等公民降级为 `meta` 字典 | 为"零域名词"付出的代价是重建索引 + 全链路改造；而当前只有 `.txt` 一种数据集，收益远小于风险 |
| 探针/基线按域拆包 | 你不需要多域质量门禁；**探针唯一来源**这条纪律本身保持不动即可 |
| INV-1/2/3 架构不变量 + L0 lint 测试 | 单数据集场景下这些不变量大部分自动成立；等真有 2 个以上数据集时再加 |
| 黄金链路逐字节对拍 + L3 零容差 | 本次只用 §6 一条自测；重构成规模时再启用 |
| 第二域选仓库文档 + 对抗域 | 不考虑（本次决策） |
| `ChatService` 事件流统一前后端 | 有价值，但与"配置驱动多数据集"这个目标正交，可以完全独立地以后再做 |

**保留下来的一条纪律**（唯一保留）：`artifacts_dir` 与 `collection` 必须由 profile 派生、**跨 profile 绝不共享**——这条是 §2.1 那五条原因的机械保障，也是整个方案的地基。

---

## 8. 执行顺序

| 步 | 内容 | 验收 |
|---|---|---|
| 1 | 抽出 `load_profile()` + 校验（未知键/非法值报错） | 无 profile 时报错并列出可用 profile |
| 2 | 参数化 §4.1 的 #1–#6（路径与集合名） | 用现有 `classical_novels` profile 跑通 `process`/`index`/`serve`，行为与现在**完全一致** |
| 3 | 参数化 #7–#9（源指纹 / BOOKS_DIR / LEXICON_DIR 可空） | 同上 |
| 4 | 加 `profiles/project_docs.yaml`（第二数据集） | §6 的三条自测全绿 |
| 5 | （可选，独立）恢复 prompt / 逐字节对拍 / 事件流统一 | 各自单独实测 |

**第 2 步的关键验收是"行为与现在完全一致"**——先把配置化做成纯搬家，不夹带任何行为变更；否则一旦出问题，无法判断是配置化引入的还是原有的。

---

## 9. 决策记录（本次）

| 议题 | 决策 |
|---|---|
| P0 提示词 | **暂不动**（记录事实：生成侧当前未接入） |
| 文档漂移 | **采纳"可验证陈述"规则**；实际修改并入阶段 1 |
| 边界判据 / 是否分域 | 澄清：**不是分域，是分集合**（Qdrant collection 即分区），判据 = 词表与 IDF 能否共享 |
| 是否能在向量库下"分域" | 能，且必须——**就是 collection**；不能靠 filter 代替（§2.1 五条） |
| DomainPack 协议 | 后面再说 → 实际是**砍掉**，用 YAML 替代 |
| `book` → meta 降级 | 后面再说 → **暂不做** |
| 四块垂直资产 | 改为：**prompt 目录 / 词表目录可为空 / 参数覆盖**，不做探针与拒答域化 |
| 质量四样域化 | 暂不考虑 |
| 架构不变量 | 暂不考虑；保留"产物与集合跨 profile 不共享"一条 |
| 验收门 | 先简单自测（§6 一条） |
| 第二域 / 对抗域 | 不考虑 |
| 本轮范围 | **设计文档定稿**（即本文档） |

### 待办（下次开工）

1. `/tmp/prompt_backup/rag_engine.py` 是否还存在（P0 恢复时需要）。
2. `RAG_USE_COLLECTION_ALIAS=1` 但别名 `books_current` **实际不存在**——`resolve_collection_name()`（`rag_engine.py:2243`）的回退行为需确认，这直接关系到 §3 里 `alias` 字段能否照现有语义工作。
3. 你那张"离线清洗"的具体工序（清洗到什么程度、是否已是纯文本），决定 `source_dir` 里应该放什么。
