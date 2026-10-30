# 02 离线数据与索引链路：分块 / 词表 / 嵌入缓存 / BM25 / Qdrant 建索引

> 分析对象：`chatbot.py`（5872 行，仓库唯一 Python 文件）。行号基准为当前工作区版本。
> 任务：task-2。只读分析，未改动任何源码；本文只描述"离线产物如何生成、如何判定新鲜、失败如何收场"。
> 覆盖范围：L1161-L1274 模型加载与向量化、L1274-L1677 归一化与父子分块、L1677-L1922 分块产物与新鲜度门禁、L1922-L2426 词表与稀疏编码、L2426-L2872 建索引、三个指纹族（`_chunking_code_fingerprint` / `_embed_code_fingerprint` / `lexicon_fingerprint`）、L824-L1032 配置基线。
> 行号漂移提示：2026-10-29 起构建链路有增量改动（嵌入缓存 fp16 真存储 + `_gc_embed_cache_files`、
> `build_index`/`build_chunks`/`_sparse_rebuild` 收尾的构建台账、`evaluate` 离线评估命令），涉及函数与配置段
> 的行号会相对本文有所漂移，涉及函数的语义见代码注释（与本文冲突时以代码注释为准）。

---

## 0. 结论先行：这条链路的四个设计支点

1. **三层指纹，各管一段**：切分指纹（`_chunking_fingerprint`）只决定"分块产物是否要重算"，嵌入缓存键（`_embed_cache_file`）只决定"稠密向量是否要重算"，词表指纹（`lexicon_fingerprint`）只决定"稀疏向量是否要重算"。三层互不牵连，于是"改一个别名"只跑 `sparse`，不动 21255 条稠密向量。
2. **指纹与"真正决定结果的东西"同源**：源码（`inspect.getsource`）＋配置快照＋文件内容哈希（不是 mtime）＋Qdrant 服务端版本＋第三方包版本。文件头 L8-L9 明确写死了一条纪律：指纹单元里的函数"改名/加装饰器/改空白都会让源码哈希变化 → 全量重建"。
3. **fail-closed + "判据不能恒真"**：检索前拿 payload 里的指纹与"当前"比对，对不上、或**无从判断**（字段缺失）都显式抛错，绝不"看起来正常地继续服务旧库"。全文件反复出现的"失效原型"注释（静默 / 判据恒真 / 抽样恒真 / 归因指反）是这套设计的动机来源。
4. **所有落盘都是原子写**：`<path>.tmp` + `os.replace`（chunks.json、doc 缓存、npz 缓存同款），任何时刻被杀都只留下"整份旧产物"或"完整新产物"，不会出现能通过校验的撕裂文件。
5. **编码来源单一化（2026-10-30 起）**：稠密 npz 缓存**只由本地 torch 写入**（build_index 不加 http 分支——靠"缓存完整命中"的策略，重建时再在开发机 MPS/CUDA 编码一次并发布缓存）；查询侧已是 http（infinity），**不写缓存**。"同缓存键混两种编码来源"的结构性风险因此不存在。查询编码与入库编码的池化/精度差用 `embedpairs` 对拍探针（cos≥0.999）+ `_http_embed` 合同三查（model 名/条数/维度）两端防守。

本仓库当前产物实测：**997 篇文档 → 21255 个子块 / 4852 个父块**，切分指纹 `923236fe7e4acc7c`；`artifacts/chunks.json` 61MB；稠密缓存 `artifacts/embeddings/*.npz` 每份 92.5MB（21255 × 1024 维）。链路上任何一层出问题，都能从这三组数字与三个指纹定位到具体阶段。

### 读代码导航

| 关注点 | 入口 | 行号 |
| --- | --- | --- |
| 配置基线 | 模块级常量 | L824-L1032 |
| 机器规格/并发派生 | `offline_workers` / `embed_batch_size` / `qdrant_write_workers` | L285-L346 |
| 模型加载 | `load_embedding_model` / `embed` | L1197-L1271 |
| 归一化 | `normalize_text` / `fix_quotes` / `read_book_text` | L1295-L1468 |
| 父子切分 | `_hierarchical_split` / `_merge_tiny_parents` | L1395-L1441 |
| 分块指纹 | `_chunking_*_fingerprint` / `_sources_fingerprint` | L1478-L1555 |
| 分块主流程 | `build_chunks` / `_chunk_one_document` | L1705-L1893 |
| 产物新鲜度 | `verify_chunks_freshness` | L1896-L1919 |
| 词表 | `load_lexicon` / `normalize_aliases` / `sparse_text` | L1922-L2206 |
| 词表指纹 | `lexicon_fingerprint` / `qdrant_server_version` | L2076-L2142 |
| 嵌入缓存 | `_embed_cache_file` / `_inspect_embed_caches` / `_save_embed_cache` | L2227-L2423 |
| 建索引 | `build_index` | L2471-L2869 |
| 稀疏重建 | `_sparse_rebuild` | L5198-L5320 |
| 索引单例锁 | `acquire_index_lock` | L4232-L4356 |
| 检索侧闸门 | `_check_chunk_freshness` / `_check_lexicon_consistency` | L3823-L3921 |
| 阶段体检 | `cmd_stages` | L5675-L5797 |

---

## 1. 端到端数据流

### 1.1 管线总图

~~~mermaid
flowchart TD
    subgraph OFF["离线：四条命令"]
        C1["corpus/*.txt<br/>(RAG_DATA_DIR / RAG_DOC_GLOB)"]
        P["python chatbot.py process<br/>build_chunks()"]
        I["python chatbot.py index<br/>build_index()"]
        S["python chatbot.py sparse<br/>_sparse_rebuild()"]
        T["python chatbot.py stages<br/>cmd_stages()"]
    end

    C1 -->|"读文件 + sha256(16)"| P
    P -->|"原子写"| A1["artifacts/docs/&lt;source&gt;.json<br/>单文档分块缓存"]
    P -->|"原子写"| A2["artifacts/chunks.json<br/>61MB / 21255 条"]
    P -->|"原子写"| A3["artifacts/chunks.meta.json<br/>fingerprint + config + sources"]
    A3 -->|"verify_chunks_freshness()"| I
    A2 --> I

    I -->|"文本 sha256 → key"| E1["artifacts/embeddings/<br/>&lt;model&gt;_&lt;content&gt;_&lt;code&gt;_&lt;prec&gt;.npz"]
    I -->|"每 1000 点一批 / 4 路并发"| Q[("Qdrant<br/>集合名__build_id[:8]")]
    I -->|"别名原子切换"| AL["alias: rag_current"]
    S -->|"滚动 + 逐页更新"| Q
    T -->|"只读比对"| A2
    T -->|"只读比对"| A3
    T -->|"count / 全量滚动"| Q
~~~

### 1.2 磁盘布局（实测）

~~~text
artifacts/
├── chunks.json          61MB  21255 条分块记录（JSON 数组，indent=2）
├── chunks.meta.json     96KB  指纹 + 配置快照 + 每文档 sha + 统计
├── builds.jsonl         构建台账（RAG_BUILD_STATS 默认开；process/index/sparse 成功收尾各一行，摘要 tools/build_stats.py）
├── docs/                62MB  997 个 <source>.json（单文档分块缓存）
├── embeddings/          185MB 2 份 npz（同 model/content，不同 code_key）
│   ├── c65e1380556b_23d8f4c1c39f_5e2567e2b1de_fp16.npz
│   └── c65e1380556b_23d8f4c1c39f_ae752abbbbc0_fp16.npz
└── .index.lock                 index/sparse 单例锁（运行期存在）
~~~

第二份 npz 是"代码指纹变化后旧缓存未被回收"的直接证据（见 `9 取舍 5）：文件名中的 `code_key` 变了，旧文件不再被任何一次运行读取，但也没有任何 GC 删除它。

### 1.3 为什么分成 process / index / sparse 三条命令

- **process 只依赖 CPU 与 tokenizer**（`_chunk_worker_tokenizer` L1695-L1702，进程内懒加载），不碰 Qdrant、不碰嵌入模型权重，因此可以在语料变动时独立、快速、可增量地重跑。
- **index 依赖 GPU/CPU 嵌入、Qdrant、别名**，是唯一会改"对外可见知识库"的阶段。
- **sparse 是 index 的"外科手术版"**：词表/BM25 变了，稠密向量与 payload 正文都没变，只需重算稀疏向量并回写 `lexicon_id`（`_sparse_rebuild` L5198-L5320）。这是三层指纹拆分的直接收益：改一个别名不必重算 21255×1024 个浮点。

---

## 2. 配置基线（L824-L1032）

### 2.1 领域与来源字段

`SOURCE_FIELD`（默认 `source`，L829）是"来源"在 payload 里的键名。它有一份历史别名表 `SOURCE_PAYLOAD_FIELDS = (SOURCE_FIELD, "source", "book")`（L1029-L1032），读 payload、来源过滤、比较器放行全部从这一处派生。同时 L831-L846 有一张**保留字表**并在启动时校验：若 `RAG_SOURCE_FIELD` 撞上 `score/text/context/n/rerank_score/source_label/point_id` 等"对外输出键"，会**启动即报错**——因为这些键会被来源名就地覆盖，响应看起来正常但字段全错（注释 L836-L841 点名的静默失效）。

### 2.2 分块参数（子块 ≤128 / 父块 ≤512 token）

| 参数 | 环境变量 | 默认 | 行号 | 说明 |
| --- | --- | --- | --- | --- |
| 子块上限 | `RAG_CHILD_MAX_TOKENS` | 128 | L953 | 子块 capacity；也是 RERANK_ON=child 的评分单位 |
| 父块上限 | `RAG_PARENT_MAX_TOKENS` | 512 | L954 | 父块 capacity，硬不变量 |
| 子块重叠 | `RAG_CHUNK_OVERLAP` | 32 | L957 | 必须 < 子块上限，否则每个文档都抛错 |
| 父块下限 | `RAG_PARENT_MIN_TOKENS` | 192 | L958 | 短于它的父块尝试并入下一块（仅在合并后仍 ≤ 父上限时） |
| 子块下限 | `RAG_CHILD_MIN_TOKENS` | 32 | L962 | **死配置**：无消费点，已从指纹里移除（L959-L962）；保留解析只为对历史 .env 做具名校验 |
| 直引号归一 | `NORMALIZE_QUOTES` | true | L1279 | 见 `4.2 |
| 繁简归一 | `RAG_TEXT_OPENCC` | `t2s` | L1290 | 空串 = 跳过 OpenCC |

L963-L969 在启动期做两条交叉校验：子 ≤ 父、overlap < 子上限。注释明确写了不这样做的后果——"负的重叠/最小值能一路进到 splitter，每个文档都抛 ValueError 让 process 全量失败（报错点还落在组件内部，指不到配置）"。

### 2.3 索引/模型/产物路径

| 参数 | 环境变量 | 默认 | 行号 |
| --- | --- | --- | --- |
| 数据目录 | `RAG_DATA_DIR`（遗留别名 `RAG_BOOKS_DIR`） | `corpus` | L827 |
| 文档 glob | `RAG_DOC_GLOB` | `*.txt` | L828 |
| 嵌入权重目录 | `RAG_EMBED_MODEL_PATH` | `./models/bge-base-zh-v1.5` | L1002 |
| 索引设备 | `INDEX_DEVICE` | 空 = `get_device()` 自动 | L977 |
| 嵌入批大小 | `RAG_EMBED_BATCH_SIZE` | 空 = 派生（L292-L324） | L973-L974 |
| 索引内存闸门 | `RAG_INDEX_MIN_FREE_MB` | 0 = 关闭 | L983 |
| Qdrant 连接 | `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_API_KEY` / `RAG_QDRANT_TIMEOUT` | localhost / 6333 / None / 60s | L984-L991 |
| 写批并发 | `RAG_INDEX_WRITE_WORKERS` | 0 = 派生（L341-L346） | L994 |
| 集合基名 | `RAG_COLLECTION_BASE` | `rag_documents_v1` | L998 |
| 别名 | `RAG_COLLECTION_ALIAS` / `RAG_USE_COLLECTION_ALIAS` | `rag_current` / true | L1000-L1001 |
| 产物目录 | `RAG_ARTIFACTS_DIR` | `./artifacts` | L1010 |
| 分块产物 | `artifacts/chunks.json` + `chunks.meta.json` | meta_version=1 | L1011-L1015 |
| 稠密缓存 | `RAG_EMBED_CACHE` | true | L1017 |
| 单文档缓存 | `artifacts/docs` | — | L1018 |
| 词表目录 | `RAG_LEXICON_DIR` | `./lexicon` | L1019 |
| 写批大小 | 常量 `WRITE_BATCH` | 1000 | L1020 |
| 权重后台下载等待 | `RAG_WEIGHTS_BACKGROUND` / `RAG_MODEL_WAIT_SECS` | false / 900s | L1008-L1009 |

### 2.4 机器规格 → 并发度（"依据"必须可审计）

离线链路有三处并行，全部由机器事实派生，并**把依据打进日志**（`hw_report()` L233-L236）：

| 派生函数 | 公式 | 行号 | 用在哪 |
| --- | --- | --- | --- |
| `offline_workers` | `min(核数, 可用内存 // 768MB)` | L285-L289 | process 的分块进程池 |
| `embed_batch_size` | MPS 16 / CUDA `vram/8/40` 夹到 8..128 / CPU 32；内存 <4GB 减半（保底 8） | L292-L324 | index 的嵌入批 |
| `qdrant_write_workers` | `min(核数, 内存//512MB, 批数, max(4, 2×Little))`，Little = ⌈1000pps × 1.5s ÷ 1000点/批⌉ = 2 | L327-L346 | index 写批 + sparse 页并发 |

核数/内存探测顺序（L127-L220）：环境变量 `HW_CORES`/`HW_MEM_MB` → cgroup v2/v1 → 线程族回退 → 安全回退（4 核/内存按紧张）。`_mem_available_mb`（L239-L275）同时读 `/proc/meminfo` 与 cgroup 头寸并**取 min**，注释写明"报小的失真方向是安全的：……不会导向'误判有余量'的死亡方向"。

---

## 3. process 阶段：归一化与父子分块

### 3.1 输入/输出

- 输入：`DATA_DIR` 下匹配 `DOC_GLOB` 的文件（排序后，L1561-L1564）；嵌入模型目录（只为 tokenizer 与内容指纹）。
- 输出：`artifacts/chunks.json`（记录数组）、`artifacts/chunks.meta.json`（指纹/配置/每文档 sha/统计）、`artifacts/docs/<source>.json`（逐文档缓存）。
- 退出码：0 = 产物已写；1 = 有文档失败，**产物保持上一版不覆盖**（L1855-L1862）。

### 3.2 文档发现与 source 命名（L1558-L1606）

`_document_paths`：目录不存在 → 报错；glob 无通配符且零命中 → 报错（等于写死文件名，是配置指错）；glob 带通配符且零命中 → 返回空列表，走"语料清空"合法边界（L1565-L1574）。这条区分很关键：旧写法一律 raise，会让"清空知识库"变成不可达状态。

`_document_names`：候选名 = 相对路径去扩展名、路径分隔符编码成 `__`；**冲突组内**（同 stem 不同扩展名、或候选里本来就有 `__`）一律追加相对路径 sha256 前 16 位（L1598-L1604）。注释记录了旧实现的静默覆盖：`a.txt`/`a.md` 会派生出同一 source/id/parent_id，两篇文档互相覆盖且每次 process 都重算。名字映射用单条缓存 `_DOC_NAME_CACHE`（L1577、L1604），只保留最近 (DATA_DIR, DOC_GLOB, paths) 组合。

### 3.3 归一化链（L1274-L1468）

顺序固定为 **ftfy 修编码 → OpenCC 繁简 → 引号归一**（`read_book_text` L1465-L1467）：

1. **严格解码**（L1455-L1464）：`utf-8` 解码失败时记录偏移与坏字节，再用 `errors="replace"` 降级为 U+FFFD。不用 `ignore` 静默丢字节（"损坏位置和内容都无从追查"）。
2. **不做 CRLF 归一**（L1450-L1452）：实测 sentencex 自身把 `\r\n\r\n` 当句边界且输出不含 `\r`，归一化是多余变换，反而破坏"逐字无损"。
3. **ftfy 只修编码**（`FTFY_FIX` L1289）：`fix_encoding=True`，其余全关——尤其 `uncurl_quotes` 会与 `fix_quotes` 打架、`fix_line_breaks` 会与"不归一 CRLF"冲突。判据是"一个功能只能有一个实现"。
4. **OpenCC t2s**（`_opencc` L1295-L1309，双检锁惰性单例）：让同一实体的繁/简两种写法在稀疏与稠密两侧不再分叉；简化字语料上是恒等变换。
5. **`fix_quotes`（L1321-L1371）**：单遍二值状态机，`"`、`“`、`”` 共同参与"是否处于已开未闭引号内"的状态。注释用实测数据否定了两种旧思路：按"直引号左侧紧邻句读"判开引号在真实语料上恰好相反（`？"` 915 次、`！"` 402 次都是闭引号），旧规则把不配平从 1600 推到 3362；只让成对弯引号参与状态则无法承认直引号是闭合者。`NORMALIZE_QUOTES=0` 是逃生开关。

### 3.4 父子切分（L1395-L1441）

`_hierarchical_split` 把机械切分交给 `semantic-text-splitter`：

- 父块：`TextSplitter.from_callback(count, PARENT_MAX_TOKENS, trim=False)`；
- 子块：在**每个父块内部**再切，`capacity=CHILD_MAX_TOKENS, overlap=CHUNK_OVERLAP, trim=False`；
- 容量用本仓库 tokenizer 度量（`_token_counter` L1388-L1392，`tokenizer.encode(..., add_special_tokens=False)`），所以"父 ≤512 / 子 ≤128 token"与改造前同义；
- `trim=False` 是"切分逐字无损"（`join(chunks) == text`）的必要条件；
- `_merge_tiny_parents`（L1395-L1413）：只做拼接，且只在合并后仍 ≤ 父上限时合并——上限是硬不变量，块短只是质量问题（注释：9.8% 的父块曾短于 40 token，"短父块没有上下文可给"）；
- 只给上限、不给区间：注释 L1434-L1436 记录实测，父块区间 (192,512) 会把父块从 411 涨到 672、子块区间更会撞 `ValueError(overlap >= 容量下界)` 让 process 全量失败。

### 3.5 记录结构（写入 chunks.json 的字段）

`_chunk_one_document` L1746-L1749 构造：

~~~python
{
  "id": "<source>_<chunk_idx>",       # 记录键；payload 键是 chunk_id（见 _PAYLOAD_SPEC）
  "child_text": ..., "parent_text": ...,
  "chunk_index": i, "total_chunks": n,
  "parent_id": "<source>_p<k>",       # 同父块文本共享一个 parent_id
  "parent_chunk_count": 0,            # 全量汇总后回填（L1865-L1867）
  "contextual_text": child_text,      # 当前与 child_text 逐字相同
  SOURCE_FIELD: "<source>"            # 默认键名 source
}
~~~

`_RECORD_FIELDS`（L1027-L1028）与 `_PAYLOAD_SPEC`（L1024）互为镜像，并在 process（L1863-L1864）与 index（L2512-L2515）两处做**键集全等校验**。这条校验防的是：改 `RAG_SOURCE_FIELD` 后只跑 index（没跑 process），切分指纹与源 sha 都不变、stages 四段全绿，但旧键读不到 → payload 来源全空、来源过滤恒零命中（注释 L2509-L2515）。

### 3.6 单文档增量缓存（L1609-L1674）

每个文档一份 `artifacts/docs/<source>.json`，内容 `{source, sha256, chunking_fingerprint, chunks}`。命中条件（`_load_doc_chunks` L1613-L1633）三条全满足：`source` 一致、`sha256` 一致、`chunking_fingerprint` 一致，且 `chunks[0]` 的键集 == `_RECORD_FIELDS`。任何一条不满足 → 返回 None → 重算并覆盖自己那份。

- 写：`tmp + os.replace`（L1636-L1649），失败只告警并清 tmp，"本次分块不受影响"。
- GC：`_gc_doc_chunks`（L1652-L1674）删除已消失文档的缓存（增删改里的"删"）与中断遗留的 `.json.tmp`。
- 锁：**不需要**。每个 worker 只写自己的文件，主进程只汇总；这也是 process 能并行的前提（L1677-L1680）。

### 3.7 并行模型：按文档分发、按原顺序收集（L1760-L1893）

- 并行度 `workers = min(total, offline_workers())`（L1797），日志打印 `hw_report()` 说明"为什么是 N 路"。
- 用 `ProcessPoolExecutor`（L1821-L1827），**窗口化提交**（在飞 ≤ `2×workers`，L1834-L1838）：结果不必全量驻留，Ctrl-C 只消等在飞的那几篇。
- 每个 worker 内 `_chunk_one_document`（L1705-L1757）：命中缓存直接返回；否则读文本、切分、丢弃纯空白块（留痕条数）、分配 parent_id、写自己的 doc 缓存。
- **顺序性由主进程的"前缀吸收"保证**：`done[]` 按下标存结果，`_absorb_ready`（L1802-L1819）只在"当前下标已完成"时并入 `all_chunks` 并打日志，因此产物与日志顺序和串行**逐字节一致**。
- 日志接管：`_chunk_capture_log`（L1685-L1692）把 worker 内的 `log` 收进缓冲区，由主进程按文档顺序统一打印，避免并行日志交错。
- 单篇异常返回错误串，不拖垮全局；worker 崩溃/结果不可 pickle 也按单篇失败结算（L1842-L1846）。
- 进程池创建失败自动回退串行（L1825-L1827）。

### 3.8 产物落盘与失败语义（L1855-L1881）

- 只要有失败：打印清单 + "chunks.json 与 meta 保持上一版（残缺产物不入库）"，返回 1。理由：残缺产物一旦入库，索引就会静默少文档。
- 成功：先回填 `parent_chunk_count`，再写两份 tmp，然后 `os.replace` 依次替换。注释 L1868-L1872 说明"任一步崩溃都只剩（新 chunks + 旧 meta）或整份旧产物"，而 `verify_chunks_freshness` 的指纹/语料比对都能拦下。
- meta 字段：`meta_version / fingerprint / chunking_config / sources / total_chunks / total_parents / reused_documents / build_time`（L1877）。

### 3.9 空文档与空语料是合法边界

- 0 块的文档（空文件/纯空白）：按 0 块正常结算并写空缓存，不再 raise（L1731-L1737）。旧写法一篇空文件就让整轮拒绝写产物、process 永远退出 1。
- 0 文档（语料被清空）：写出空 `chunks.json` + 空 `sources`（L1783-L1786）。

---

## 4. 指纹体系：三层绑定与最小重跑

### 4.1 切分指纹（决定 process 产物是否新鲜）

~~~mermaid
flowchart LR
    CFG["配置快照<br/>token 上限 / overlap / 归一化开关<br/>OpenCC / ftfy / embed 模型路径+内容<br/>splitter 包版本 / 分块源码哈希"] --> CFG_FP["config_fp (16hex)"]
    SRC["每文档 sha256(内容) 16hex<br/>键=相对路径"] --> SRC_FP["sources"]
    CFG_FP --> FP["_chunking_fingerprint<br/>= sha256(config_fp | sources)[:16]"]
    SRC_FP --> FP
    FP --> META["chunks.meta.json.fingerprint"]
    FP --> PAYLOAD["Qdrant payload.chunk_fingerprint"]
~~~

- `_chunking_code_fingerprint`（L1497-L1506）对 `_CHUNKING_CODE_UNITS`（L1478-L1486）的 7 个函数取 `inspect.getsource` 后 sha256 前 16：`fix_quotes, read_book_text, normalize_text, _opencc, _hierarchical_split, _token_counter, _merge_tiny_parents`。注释解释为何 `normalize_text`/`_opencc` 必须在列：`read_book_text` 只是**调用**它们，改归一化实现不改调用方源码 → 漏掉就是"改了归一化、指纹不动、旧分块静默沿用"。
- `_chunking_config_fingerprint`（L1509-L1523）的 payload 同时包含：子/父 token 上限、父下限、overlap、引号开关、OpenCC 配置、ftfy 配置 JSON、嵌入模型路径、**嵌入模型内容指纹**、`semantic-text-splitter` 包版本、分块代码指纹。返回 `(指纹, 快照)`，快照进 meta，供 `verify_chunks_freshness` 打出"哪一项从旧变新"。
- `_sources_fingerprint`（L1540-L1555）：1MB 流式 sha256，键为相对路径，截 16 位。
- `_chunking_fingerprint`（L1526-L1537）= `sha256(config_fp | json(sources))[:16]`。**源文本必须并进来**：payload 里的 `chunk_fingerprint` 是检索侧闸门的唯一判据，只算配置时"改语料 + 只跑 process"会让索引里的旧指纹与"当前"仍然相等 → 闸门恒等放行（注释 L1528-L1533）。
- 模型目录缺失时，`_chunking_config_fingerprint` 单独报"嵌入模型目录缺失"（L1513-L1519），而不是让调用方去比指纹（那条路正需要这个目录，归因不能指反）。

### 4.2 嵌入缓存键（决定稠密向量是否新鲜）

`_embed_cache_file(precision)`（L2309-L2313）的文件名是四段拼接：

`artifacts/embeddings/<model_key>_<content_key>_<code_key>_<precision>.npz`

| 段 | 来源 | 长度 | 变更含义 |
| --- | --- | --- | --- |
| `model_key` | `sha256(abspath(EMBED_MODEL_PATH))` | 12 | 换模型目录 → 新文件 |
| `content_key` | `_model_content_fingerprint()`（L2262-L2306） | 12 | 同路径换权重 → 新文件 |
| `code_key` | `_embed_code_fingerprint()`（L2234-L2250），对 `_embed_batch/embed/load_embedding_model` 源码取哈希 | 12 | 改池化/归一/截断/精度策略 → 新文件 |
| `precision` | 本次计算路径：`fp16` 或 `fp32` | — | 只决定"读哪份" |

`_model_content_fingerprint` 的设计细节（全部来自注释 L2262-L2277 的踩坑记录）：

- 不用 mtime：容器启动时下载模型会更新 mtime → 指纹漂移 → 触发本不该有的"不一致"闸门；
- **跳过隐藏文件/目录**：HF 的下载记账目录与 `.part` 暂存与模型内容无关；
- **目录不可用返回哨兵 `"unknown"`**（L2256），不返回空列表的哈希常量——否则"模型目录没了"看起来像一个正常指纹，一路走到"请重跑 process+index"；
- 小文件（≤4MB：config/tokenizer/池化头/词表）全量哈希；大文件取头 4KB + 尾 64KB + **文件长度**（防"头尾相同、中段不同"）。

注意一个实现细节（2026-10-29 起语义收口）：文件名写 `fp16`，`_save_embed_cache` 现在也**真的存
float16**（`precision=="fp16"` → `matrix.astype("float16")`；`fp32` 路径仍存 float32）——fp16 计算路径的
向量本来就产自 half 权重，二次位宽只是同阶损耗（见 `load_embedding_model` 的量化对拍：余弦 0.999998），
磁盘减半；`_load_embed_cache` 读侧统一升 float32，进程内的 vstack / 维度校验 / upsert 口径不变。
`precision` 的语义不变：描述**计算路径/产出者**（fp16 权重 vs fp32 权重），只是现在文件字节数与标签
自洽了。旧 float32 格式的 npz 照常读出（缓存键四段不含存储位宽），已有缓存不失效。

缓存选择规则只有一份：`_inspect_embed_caches`（L2337-L2360）+ `_full_embed_cache`（L2363-L2368）+ `_EMBED_CACHE_PRECEDENCE = ("fp16","fp32")`（L2334）。index（L2536）与 stages（L5724-L5725）**共用**这套规则；注释明确写了各写一份必然分叉，正是"stages 说新鲜 / index 却重算"两个结论的来源。

### 4.3 词表指纹（决定稀疏向量是否新鲜）

`lexicon_fingerprint`（L2104-L2142）的组成部分：

1. `aliases.txt` 文件内容 sha256（读不到记 `"missing"`，而非静默跳过）；
2. `ALIASES_FILE` 路径、`RAG_ALIASES`/`RAG_STOPWORDS` 开关（防"环境变量变了但文件内容相同"）；
3. `bm25_options()` 的 JSON（tokenizer/stemmer/stopwords/ascii_folding，L1106-L1120）；
4. **Qdrant 服务端版本**（`qdrant_server_version` L2076-L2101，取不到直接抛错）：BM25 分词器、停用词表、IDF 都在服务端实现，升级就可能换词空间；
5. `BM25_MODEL_NAME = "qdrant/bm25"`；
6. 6 个源码单元：`parse_aliases, _read_aliases, load_lexicon, normalize_aliases, sparse_text, sparse_encode`（L2135-L2140）。

**故意不纳入**的：`lexicon/stopwords.txt`。运行期停用词走 Qdrant 内置 `chinese` 表（`bm25_options`），该文件只服务 `python chatbot.py lexicon` 自检。注释 L2107-L2113 明确：旧写法把它哈希进指纹，改它检索行为一点不变却拦住全部检索并强制跑一次 sparse（"判据与实现不同源"）；若将来真把它下发给 BM25，必须重新纳入指纹。

### 4.4 三层指纹的职责矩阵（改了 X 要跑什么）

| 变更 | 变化的指纹 | 最小重跑 | 代码依据 |
| --- | --- | --- | --- |
| 语料增/删/改 | 源 sha → 切分指纹 | `process && index` | L1540-L1555、L1909-L1918 |
| 子/父 token 上限、overlap、归一化开关、OpenCC、ftfy | 配置指纹 | `process && index` | L1509-L1523 |
| 分块源码（7 个函数）、splitter 包版本、词表归一化源码 | 分块/词表代码指纹 | `process && index` / `sparse` | L1478-L1486、L2135-L2140 |
| 同路径换嵌入权重 | `content_key` + 切分配置指纹 | `process && index`（全量重算） | L2262-L2306 |
| 改嵌入代码（池化/归一/截断） | `code_key` | `index` | L2234-L2250 |
| 改 `aliases.txt` / BM25 options / Qdrant 版本 | 词表指纹 | `sparse` | L2104-L2142 |
| `RAG_SOURCE_FIELD` 等 payload 契约 | 记录键集 | `process && index` | L1863、L2512 |
| 集合丢失 / 点数不符 | 无（D 阶段） | `index` | L5752-L5788 |

### 4.5 指纹怎么被消费：检索侧 fail-closed 闸门

`RAGEngine._gate_check_collection_fingerprint`（L3889-L3921）在检索开跑前抽 1 个点取 payload：

- `_check_chunk_freshness`（L3823-L3853）：`index_fp is None` → 抛"索引未记录分块指纹（旧索引）"；否则**本地**重算当前切分指纹（`_document_paths` + `_sources_fingerprint` + `_chunking_fingerprint`），相等才置位 `_chunk_fp_checked`。失败不置位，后续每轮重试。
- `_check_lexicon_consistency`（L3854-L3888）：`_distinct_lexicon_ids` 全量滚动（L2053-L2073）收集集合里出现过的所有 `lexicon_id`。**必须全量**：sparse 逐页原地更新，中断会留下"已更新前缀 + 旧指纹后缀"，而 scroll 首点恒属已更新前缀 → 抽样必过（注释 L2053-L2059）。
- 抽样 vs 全量的边界写得很清楚（L3903-L3905）：分块指纹"抽首点成立"——一个集合只由一次 index 写成（新集合 + 原子切别名），集合内恒一；词表指纹不成立，必须全量。
- 全量滚动若发现**多个** `lexicon_id`（sparse 中断/跳页留下的混合态），直接判不一致并抛错（L3912-L3921），与 `_check_*` 的"通过才置位"同规则。
- 唯一的宽松口径是网络抖动：词表指纹计算失败若是连接/超时类，记一条告警并**暂缓本轮**（L3868-L3880），因为"状态未知 ≠ 已知不一致"；分块指纹走本地文件，无此豁免。

---

## 5. index 阶段：向量化与入库（L2471-L2869）

### 5.1 前置闸门

1. `verify_chunks_freshness()`（L2489 → L1896-L1919）：产物存在、`meta_version` 一致、当前指纹与 meta 一致。指纹不一致时**分两种文案**：源文档变了 → 列出变更文件名；否则 → 列出配置差异（旧→新）。
2. 记录键集全等校验（L2512-L2515）。
3. 内存闸门（L2490-L2506）：`RAG_INDEX_MIN_FREE_MB>0` 时，`_mem_available_mb()` 低于门槛直接报错并提示"按 README `11.4 的隔离规则先停 serve/生成"，注释写明嵌入阶段实测峰值 2.2-3.0GB。macOS 读不到 MemAvailable 时**显式说明跳过**，不冒充"检查过了"。
4. index 单例锁（`cmd_index` L4359-L4372 → `acquire_index_lock` L4300-L4341）：`artifacts/.index.lock` 用 `O_CREAT|O_EXCL` 独占，拿不到直接退出（**不等待、不跳过**）。理由（L4223-L4232）：两个 index 会写同一份 `.tmp.npz`，一方 rename 后得到"维度对、条数对但少一批向量"的坏缓存；sparse 也必须进这把锁，否则会往被 index 删掉的旧集合上写。陈旧锁用 `pid + 锁写入时刻` 双判据自动接管（防 pid 复用导致永久死锁），释放时只删自己那份。

### 5.2 文本唯一来源 `_index_text`（L2209-L2224）

稠密向量、稀疏 Document、稠密缓存键全部取 `rec["contextual_text"]`。注释记录旧代码"稠密取 contextual_text、稀疏取 child_text、缓存键写 `contextual_text or child_text`"三份规则，今天两字段逐字相同所以差异被掩盖，一旦分叉同一个点的两条向量就描述不同文本。缺字段直接 raise，不悄悄换字段。

### 5.3 稠密向量：三条路径

`build_index` 先算 `keys = [_embed_cache_key(_index_text(c)) ...]`（L2530，`key = sha256(text)` L2227-L2228），然后 `_inspect_embed_caches(keys)` 拿到候选，分三支：

**A. 空产物分支（total == 0，L2538-L2554）**：不加载模型，直接读 `config.json.hidden_size` 作为 `dense_dim`（`_expected_embed_dim` L2316-L2329）；读不出就报错——空集合同样需要维度。然后写 0 点集合并切别名，使"清空知识库"可达。

**B. 完整命中分支（L2555-L2560）**：某份 npz 覆盖全部 keys → 跳过设备选择与模型加载（省约 600MB 峰值），`precision/device/tokenizer/model` 全部为空。

**C. 缺失分支（L2561-L2597）**：
- 设备 = `INDEX_DEVICE` 或 `get_device()`；GPU 路径先尝试 `load_embedding_model` + `model.half()`，失败**回退 CPU**（L2565-L2571）；
- CPU 路径设 `torch.set_num_threads(hw_cores())`（L2582-L2583，旧写法写死 6）；
- 批大小 = 显式 `RAG_EMBED_BATCH_SIZE` 或派生（L2573/L2584）；
- `precision = "fp16"/"fp32"` 决定缓存文件名；
- 维度：优先 `model.config.hidden_size`，缺失时用一次前向推断（L2588-L2593）；
- 该精度的缓存复用 `_inspect_embed_caches` 已读出的结论，避免同一文件读两遍（L2594-L2597）。

**逐批计算（L2604-L2616）**：只对 `missing` 下标切批，每批 `embed(...)`（`is_query=False`，无查询指令前缀），MPS 每批后 `empty_cache()`，日志按"百分比 + 条/秒"打进度。

**矩阵与校验（L2617-L2643）**：
- `holes`：任何槽位仍为 None → 报"缓存命中数与待计算集合不一致"；
- 维度集合必须只有 1 个 → 否则提示"缓存里混入别的模型的向量，删掉 cache_path 或设 `RAG_EMBED_CACHE=0`"；
- `np.vstack(float32)` 成矩阵，行数必须 == total（显式 raise 不用 assert，因为 `-O` 会剥离 assert）；
- 缓存写回判据：`missing or cache_entries != len(set(keys))`（L2639）。两条缺一不可：`missing==0` 保证每个 key 都在；条数相等才排除"缓存里残留已删文档的旧条目"。**两侧都必须去重**：keys 是逐块列表，重复段落会让 `len(keys) > len(cache)`，判据恒真 → 每次都整份重写 npz（"判据恒真式浪费"）。
- 内存：先记 `cache_entries` 再 `del cache`（12MB 量级），`embeddings` 取完值后 `del keys/model/tokenizer`，MPS 再清一次缓存（L2598-L2603、L2644-L2646）。

### 5.4 稀疏向量：把分词交给 Qdrant（L1106-L1120、L2176-L2206）

- `bm25_options()` 返回 `{tokenizer:"multilingual", stemmer:{type:"none"}, stopwords:"chinese"（关掉时用 {}）, ascii_folding:true}`。注释记录实测：`"word"/"whitespace"/"prefix"` 对中文整句零召回（无空格 → 整句 1 个 token），`multilingual` 正确命中（相关 5.069 vs 无关 0.221），于是 Python 侧 jieba 预分词整层删除；`stopwords` 不能用字符串 `"none"`（实测 400）。
- `sparse_text`（L2176-L2186）只做领域别名归一；纯标点/空白返回零宽空格哨兵 `_EMPTY_SPARSE_TEXT`（L2189），因为空文档会让 BM25 静默失败（"用哨兵占位"）。
- `sparse_encode`（L2192-L2194）返回 `models.Document(text=..., model="qdrant/bm25", options=...)`——真正的切词/IDF 在服务端，Python 侧拿不到 term 列表，`sparse_terms_of`（L2197-L2206）因此退化为"归一后还剩字母/数字就有内容"。
- 服务端编码的代价被两处补偿：Qdrant 版本进词表指纹（`lexicon_fingerprint` 第 4 项）、`sparse` 重建前对空短语做**哨兵自检**（L5210-L5236：空短语必须 0 命中或被服务端拒收，否则中止重建）。
- `normalize_aliases`（L2145-L2173）：Aho-Corasick 最长优先、从左到右不重叠；`load_lexicon`（L1933-L1955）懒加载 + 双检锁 + 无锁快路径（2026-10-30 起热路径不进锁，见 docs 04 §6 锁账），返回 `(alias_map, automaton, stopwords)` 三元组。

### 5.5 Qdrant collection 与 payload schema

**集合命名（L2647-L2662）**：启用别名时目标集合 `<COLLECTION_BASE>__<build_id[:8]>`（蓝绿新集合）；不支持别名时退化为 `COLLECTION_NAME` 且**先删旧集合**。

**向量空间（L2664）**：

~~~python
vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)}
sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)}
~~~

**payload 索引（L2665-L2669）**：`SOURCE_FIELD` 与 `parent_id` 建 KEYWORD 索引；失败只告警（"过滤会退化为扫描"）。

**payload 字段（`chunk_payload` L1079-L1088）**：

| payload 键 | 来源记录键 | 类型 | 用途 |
| --- | --- | --- | --- |
| `chunk_id` | `id` | str | 分块业务 id（`<source>_<n>`） |
| `child_text` / `parent_text` | 同名 | str | 子块命中、父块回填上下文 |
| `source`（可配） | `source` | str | 来源过滤/展示 |
| `chunk_index` / `total_chunks` | 同名 | int | 位置与引用编号 |
| `contextual_text` | 同名 | str | **入库文本唯一来源**（稠密/稀疏/缓存键） |
| `parent_id` / `parent_chunk_count` | 同名 | str/int | 父块去重（DEDUP_BY_PARENT） |
| `build_id` | 本次 `uuid4().hex` | str | 本次构建标识，便于对账 |
| `lexicon_id` | `lexicon_fingerprint()` | str | 检索侧词表闸门判据 |
| `chunk_fingerprint` | `meta["fingerprint"]` | str | 检索侧切分闸门判据（必填，L1083-L1087） |

**点 id（L2699）**：`PointStruct(id=i + j, ...)`，其中 `i` 是批起始下标、`j` 是批内偏移——即**全局分块序号**（0..total-1），整数、与写入先后无关。这个选择让"清残留点"可以直接按 `id >= total` 判定（见 5.7）。

### 5.6 写批并发与失败处理（L2670-L2766）

- 分批：`WRITE_BATCH = 1000`，`batch_starts = range(0, total, 1000)`（本仓库 22 批）。
- **每 worker 一个 QdrantClient**（`threading.local`，L2677-L2684）：qdrant-client 的 `ModelEmbedder`（本地拆包 `Document`）是**每客户端一份且内部无锁**的，共用客户端时多线程会互相 drain `_embed_storage`（实测 `IndexError: pop from empty list` / 422 "dense vector must not be empty"）。旧修法用一把模块锁把整次 upsert（含 HTTP）串行化 → 4 路退化成 1 路。分开客户端后，本地推理天然串行、HTTP 往返真正并发。（模块级 `_QDRANT_INFERENCE_LOCK` L2435 只留给"共用同一 client"的检索侧稀疏分支。）
- 并发度 = `INDEX_WRITE_WORKERS` 或 `qdrant_write_workers(batch_count)`；公式依据是实测的 Little 定律 + "客户端构造 points 受 GIL 限制需要额外线程搭桥"（L327-L346）。
- 单批 `_upsert_batch_with_retry`（L2438-L2455）：失败立即重试 1 次（**不睡眠**，等待只会占住 worker 槽），仍失败返回错误串。
- `_write_batch` 用 `except BaseException` 把一切异常转成错误串（L2692-L2705），因为"从 `fut.result()` 抛出会跳过整段 `write_failures` 收尾（不删孤儿集合、不报谁在服务、连集合名都不留）"。
- 失败后：继续写其余批（尽快暴露完整失败面）→ 结束时统一 raise（L2744-L2766），**不切别名**，并按模式给相反的"谁在服务"文案：别名模式下旧集合继续服务；非别名模式下旧集合已删、残缺集合就是服务集合。
- 孤儿集合：能删就删，删不掉给出具体 `curl` 回收命令（L2748-L2754）。
- 观测：每批记 `(构造, 往返)` 秒数，末尾打平均分解（L2737-L2742），让"要几路并发"可被复核。

### 5.7 点数不变量与残留点清理（L2767-L2797）

写入后 `client.count(exact=True)` 必须 == `total`。不相等时**先修再判**：滚动扫出 `id >= total` 的残留点并删除，再复核。注释把方向写清楚了（L2769-L2772）：残留点"不是检索不到"——检索是向量搜索、没有 id 过滤，残留点会被正常召回并返回陈旧正文。最后再校验集合稠密维度 == 本次 `dense_dim`（显式 raise，不用 assert）。这套不变量把"索引点数 = 产物记录数"从"stages 事后发现"提前到"本次当场把关"。

### 5.8 别名切换：非事务双操作的重试 + 验证 + 恢复（L2799-L2865）

`DeleteAlias` + `CreateAlias` 是两个非事务操作，极端情况下 Delete 成功而 Create 失败会把别名删空。实现采取四层防护：

1. **只重试 2 次**（`max_retries = 2`），每次切换后 `get_aliases()` 验证指向新集合；
2. 空别名冷启动时**不提交 Delete**（部分 Qdrant 版本对不存在的 alias 整批失败），Create 恒提交；
3. 失败分支按"**本轮之前的目标**"先发一次恢复性 CreateAlias 把检索端救回来（L2837-L2845），再抛错，且文案按 `get_aliases()` 的**实际**结果组织；
4. 成功后再用"本轮之前的目标"删旧集合（不能用循环内变量——重试后它可能已变成新 target），并扫描同前缀 stragglers 给出手动回收命令（L2857-L2865）。

检索侧 `resolve_collection_name`（L3005-L3027）缓存的是**别名字符串本身**，Qdrant 查询时再路由到当前集合，所以 reindex 切别名后新查询自动跟随、无需重启。

非别名模式：L2658-L2662 写入前删旧集合、L2867 明示"重建期间该集合会短暂不可用"。

### 5.9 index 的增量语义

- 断点粒度 = **单个分块**（`_embed_cache_key` 是文本 sha）；重复文本（重复段落/标题）天然共享一条向量。
- 重跑 index 时若语料只改了几篇，process 只重算那几篇，index 只算新增/变更文本对应的向量；其余从 npz 命中。
- 没有"部分失败续传"：写批失败会整体失败并清理孤儿集合，但稠密缓存已经保留（下次重跑直接命中），所以重试代价 = 剩余向量 + 全部写批。

---

## 6. sparse 阶段：词表变化的最小重建（L5198-L5320）

- 目标集合用 `resolve_collection_name` 解析（可能是指向别名的名字）。
- 前置：集合为空 → 退出 1 并提示先 process+index；空短语哨兵自检（5.4 已述）。
- 更新方式：主线程串行推进 `scroll` 游标（游标链本身必须串行），**页内更新交给线程池并发**：每页 `update_vectors` 重算稀疏向量 + `set_payload` 回写新 `lexicon_id`（L5256-L5267）。每线程独立 client，理由同 5.6。
- 窗口化：在飞页 ≤ `2×workers`，不把 21255 条 payload 全摊在内存（L5275-L5276）。
- `scroll` 失败只重试同一页 1 次，连败 2 次熔断（L5279-L5287）。
- **EOF 必须在 `offset is None` 时判定**（L5293-L5296）：Qdrant 的 scroll 在 `offset=None` 时是"从头再来"，只看空页会绕回集合开头无限重扫（注释记录实测 done 涨到 130 万）。
- 结果：`done != total` → 退出 1；末尾抽 5 条打稠密维数/稀疏 term 数（明确标注"非全量契约，全量契约由词表指纹闸门承担"，L5314-L5318）。

---

## 7. stages：只读体检与最小重跑建议（L5675-L5797）

四段检查对应三层指纹 + 入库：

| 段 | 检查 | 过期判据 |
| --- | --- | --- |
| A 分块 | `chunks.json` 可读 + `verify_chunks_freshness` | 读坏 / 指纹不一致 |
| B 稠密 | 与 index 共用 `_inspect_embed_caches + _full_embed_cache` | 没有任何一份缓存完整覆盖 |
| C 稀疏 | `_distinct_lexicon_ids` 全量滚动 | 集合不存在 / 混合指纹 / 未记录 / 与当前不一致 |
| D 入库 | `client.count(exact=True)` | 集合不存在 / 点数 ≠ 产物条数 |

细节：`chunks.json` 只读一次供 A/C/D 共用（防同一份文件半截读得出、半截读不出）；读坏时 A 判过期且 `cd_unknown=True`，**绝不**让 B/D 落到"✅"；检查抛异常也算"未验证"（`cd_unknown`），与"已过期"同权，最终 `_stage_next_action`（L5667-L5672）优先给出 `process && index` 或 `sparse`；退出码 1 = 有阶段过期或未验证（L5795-L5797），便于自动化判断。

---

## 8. 失败 / 重跑语义一览

| 场景 | 产物状态 | 对外行为 | 依据 |
| --- | --- | --- | --- |
| 单篇文档分块异常 | 该文档不写缓存；chunks.json/meta 不覆盖 | process 退出 1，列出失败文档；修复后重跑只重算失败与变更文档 | L1752-L1757、L1855-L1862 |
| 空文档 / 空语料 | 正常写 0 块记录 / 空数组 | 退出 0（合法边界） | L1731-L1737、L1783-L1786 |
| process 中途被杀 | 可能留 .tmp；下次启动先清 | 旧产物完整可用 | L1761-L1767 |
| 切分指纹不一致时跑 index | 拒绝 | 报"源文档变了"或"配置差异（旧→新）" | L1911-L1918 |
| 嵌入缓存损坏/维度不符 | 该缓存被忽略 | 告警 + 全量重算 | L2371-L2391 |
| 缓存混入异维度/异模型向量 | 拒绝 | 报维度不一致并提示删缓存 | L2621-L2623 |
| 单个写批失败 | 继续写其余批，最后整体失败 | 不切别名；能删则删孤儿集合，否则给 curl | L2729-L2766 |
| 别名切换失败 | 新集合已建好但不可见 | 先恢复旧别名；报**实际**指向 | L2835-L2850 |
| 非别名模式写入失败 | 旧集合已删 | 明确告警"当前点数残缺，检索会吃到不完整结果" | L2757-L2761 |
| sparse 中断 | 集合内混合 `lexicon_id` | 下次 stages 判 C 过期；检索闸门全量滚动拒绝 | L2053-L2073、L5768-L5777 |
| 检索时指纹缺失/不一致 | 拒绝检索 | 分别报"旧索引/请跑 process+index/sparse" | L3823-L3888 |

一条贯穿全链的纪律：**"没有记录"不等于"一致"**。`index_fp is None`、`lexicon_id is None`、`qdrant_server_version` 取不到，全部按 fail-closed 处理，因为"无从判断"被当成"通过"正是本项目最常见的静默失效形态。

---

## 9. 设计取舍与已知代价

1. **源码哈希作为指纹**：换来"改实现即自动全量重建"的强正确性，代价是重命名/加装饰器/改空白都会触发重建（文件头 L8-L9 把这条写成了约定：指纹单元禁改）。替代方案（显式版本号）省重建但依赖人工纪律，这里选择了"宁可多算一次，不可静默用旧"。
2. **三层指纹而非一条总版本号**：换来最小重跑（改别名只跑 sparse），代价是三套判据、三种失败文案、以及"必须保证每层与实现同源"的持续维护成本（`stopwords.txt` 被移出指纹就是一个纠偏案例）。
3. **全量 JSON 产物（chunks.json 61MB）**：人类可读、可 diff、可用 jq 排查；代价是加载/解析慢、内存峰值高（21255 个 dict）。缓解手段是流式 sha、按需重取 `_index_text`、及时 `del` 中间结构，但"全量读入内存"这一点没有改变。
4. **process 按文档并行 + 前缀顺序吸收**：产物与串行逐字节一致、可复现；代价是"最慢的那篇"决定整体墙钟（前缀吸收会等待慢文档），以及 worker 结果需要 pickle。
5. **嵌入缓存按 `code_key` 分文件**：正确性优先（旧向量永不被新代码误用）。2026-10-29 起补上收尾
   GC（`_gc_embed_cache_files`，index 成功 + 别名切换通过后调用）：只保留当前
   `<model>_<content>_<code>_` 前缀的文件（fp16/fp32 两条计算路径都合法），其余 npz 删除——
   "换模型/换嵌入代码后旧缓存需人工回收"的负担取消。GC 只删文件不动任何指纹（cache 文件名
   四段不含存储位宽），也不清 `.tmp.npz`（那个仍归 `_gc_embed_cache_tmp` 在 index 起手时处理）。
6. **`precision` 是计算路径标签；落盘位宽自 2026-10-29 起与标签同源**：fp16 文件存 float16（磁盘减半，
   读侧升 float32）；fp32 文件（CPU 全精度路径）仍存 float32——同一模型下 CPU 构建保留更高的存储
   保真。代价评估：fp16 存储的二次量化与 fp16 计算路径的既有误差同阶（对拍实测 top-10 近邻
   recall 0.998），且缓存键不含存储位宽 → 已有缓存无需重建。
7. **BM25 交给 Qdrant 服务端**：删掉 jieba、别名以外的中文分词逻辑与串行锁，换来正确的中文召回与更简单的代码；代价是三个新约束——Qdrant 版本必须进指纹、空文本必须用哨兵、Python 拿不到 term 列表（`sparse_terms_of` 退化为布尔）。
8. **写批"继续写其余批再统一失败"**：能一次暴露完整失败面、避免只报第一个错误；代价是可能留下孤儿集合（有清理与回收命令）。
9. **别名切换是"尽力而为的原子发布"**：重试 + 验证 + 恢复能把"别名删空"窗口压到很小，但本质上仍非事务；单实例、人工运维的前提下被接受，代码用"实际指向"文案避免把排查方向指反。
10. **内存闸门默认关闭、并发/批大小靠派生**：默认行为对老部署零变化；8G 容器需显式设 `RAG_INDEX_MIN_FREE_MB`。并发度宁可偏保守（写批至少 4 路但受核数/内存/批数封顶），因为在 8G 档"更快"往往意味着 OOM。
11. **空语料/空文档作为合法边界**：让"清空知识库"可达；代价是 index 多一条"total==0 仍需 dense_dim"的分支，以及 stages 里"空集合无数可比"的专门文案。

---

## 10. 关键参数与常量总表（离线侧）

| 名称 | 类型/环境变量 | 默认 | 行号 |
| --- | --- | --- | --- |
| `DATA_DIR` | `RAG_DATA_DIR` / `RAG_BOOKS_DIR` | `corpus` | L827 |
| `DOC_GLOB` | `RAG_DOC_GLOB` | `*.txt` | L828 |
| `SOURCE_FIELD` | `RAG_SOURCE_FIELD` | `source` | L829 |
| `CHILD_MAX_TOKENS` | `RAG_CHILD_MAX_TOKENS` | 128 | L953 |
| `PARENT_MAX_TOKENS` | `RAG_PARENT_MAX_TOKENS` | 512 | L954 |
| `CHUNK_OVERLAP` | `RAG_CHUNK_OVERLAP` | 32 | L957 |
| `PARENT_MIN_TOKENS` | `RAG_PARENT_MIN_TOKENS` | 192 | L958 |
| `CHILD_MIN_TOKENS` | `RAG_CHILD_MIN_TOKENS` | 32（死配置） | L962 |
| `NORMALIZE_QUOTES` | `NORMALIZE_QUOTES` | true | L1279 |
| `TEXT_OPENCC` | `RAG_TEXT_OPENCC` | `t2s` | L1290 |
| `FTFY_FIX` | 常量 | 仅 `fix_encoding` | L1289 |
| `EMBED_MODEL_PATH` | `RAG_EMBED_MODEL_PATH` | `./models/bge-base-zh-v1.5` | L1002 |
| `EMBED_BATCH_SIZE` | `RAG_EMBED_BATCH_SIZE` | 0 → 派生 | L973-L974 |
| `INDEX_DEVICE` | `INDEX_DEVICE` | 空 → 自动 | L977 |
| `INDEX_MIN_FREE_MB` | `RAG_INDEX_MIN_FREE_MB` | 0（关闭） | L983 |
| `QDRANT_TIMEOUT` | `RAG_QDRANT_TIMEOUT` | 60s | L991 |
| `INDEX_WRITE_WORKERS` | `RAG_INDEX_WRITE_WORKERS` | 0 → 派生 | L994 |
| `COLLECTION_BASE` | `RAG_COLLECTION_BASE` | `rag_documents_v1` | L998 |
| `COLLECTION_ALIAS` | `RAG_COLLECTION_ALIAS` | `rag_current` | L1000 |
| `USE_COLLECTION_ALIAS` | `RAG_USE_COLLECTION_ALIAS` | true | L1001 |
| `_ARTIFACTS_DIR` | `RAG_ARTIFACTS_DIR` | `./artifacts` | L1010 |
| `CHUNKS_JSON` / `CHUNKS_META_JSON` | 派生常量 | `artifacts/chunks.json` / `.meta.json` | L1011-L1014 |
| `CHUNKS_META_VERSION` | 常量 | 1 | L1015 |
| `EMBED_CACHE_DIR` | 派生常量 | `artifacts/embeddings` | L1016 |
| `EMBED_CACHE_ENABLED` | `RAG_EMBED_CACHE` | true | L1017 |
| `DOC_CHUNK_DIR` | 派生常量 | `artifacts/docs` | L1018 |
| `LEXICON_DIR` | `RAG_LEXICON_DIR` | `./lexicon` | L1019 |
| `WRITE_BATCH` | 常量 | 1000 | L1020 |
| `DENSE_VECTOR_NAME` / `SPARSE_VECTOR_NAME` | 常量 | `dense` / `sparse` | L1101-L1102 |
| `BM25_MODEL_NAME` | 常量 | `qdrant/bm25` | L1103 |
| `ALIASES_ENABLED` / `STOPWORDS_ENABLED` | `RAG_ALIASES` / `RAG_STOPWORDS` | true / true | L1923-L1924 |
| `ALIASES_FILE` / `STOPWORDS_FILE` | `RAG_ALIASES_FILE` / `RAG_STOPWORDS_FILE` | `lexicon/aliases.txt` / `stopwords.txt` | L1925-L1926 |
| `QDRANT_BATCH_RTT_S` / `QDRANT_TARGET_PPS` | 常量 | 1.5s / 1000pps | L337-L338 |
| `_OFFLINE_WORKER_MEM_MB` / `_WRITE_WORKER_MEM_MB` | 常量 | 768 / 512 | L280-L282 |
| `_MODEL_FULL_HASH_MAX` / `_MODEL_HEAD_BYTES` / `_MODEL_TAIL_BYTES` | 常量 | 4MB / 4KB / 64KB | L2257-L2259 |
| `_EMBED_CACHE_PRECEDENCE` | 常量 | `("fp16","fp32")` | L2334 |
| `_EMPTY_SPARSE_TEXT` | 常量 | 零宽空格 U+200B | L2189 |

---

## 11. 一句话记住每一层

- **process**：文档 → 归一化 → 父子切分 → 逐文档缓存 → 原子写 chunks.json；指纹 = 分块源码 + 配置 + 每条源文本 sha。
- **index**：chunks.json → 文本 sha 做键 → 稠密缓存命中/重算 + 服务端 BM25 → 新集合（蓝绿）→ 点数不变量 → 原子切别名。
- **sparse**：词表变了 → 只滚动重算稀疏向量 + 回写 `lexicon_id`，不碰稠密。
- **门禁**：任何一层"对不上"或"无从判断"都显式报错；全部写入原子化；失败时旧知识库继续服务。
