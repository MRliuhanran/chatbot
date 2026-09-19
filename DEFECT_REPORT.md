# 功能缺陷测试报告

被测版本：修复前的代码快照（当时 `rag_engine.py` 1222 行 / `app.py` 426 行；报告下方各节描述的是**修复前**的现象与复现方式）
环境：macOS + Python 3.10，Qdrant（books_v3，21137 点）在线，Ollama 在线，两个模型目录齐备
测试方式：分层回归（L0~L3）+ 针对未覆盖面的对抗性探针
产出：`tests/test_defects_regression.py`（护栏。修复后**只剩 D3 一条**仍带 `xfail(strict=True)`，因为它要等重跑 `process` 才通过；其余 7 条已摘标记）


> **编号权威性声明（2026-09-17 补）**
>
> 本报告的 D1~D8 编号与 `tests/test_defects_regression.py` 的编号**曾经冲突**
> （只有 D1/D7 相同），两份"D 编号"会持续互相污染 —— 引用"D3"时无法确定指向
> 哪一条。现已统一：**`tests/test_defects_regression.py` 是唯一权威编号**，
> 因为它有 `xfail(strict=True)` 护栏在强制执行（修好必须摘标记，否则 XPASS 报错）。
>
> 下表已按权威编号重排。本报告额外记录的、**没有测试护栏**的发现放在 §3。
> 引用缺陷时请写"护栏 D3"或"§3 第 N 条"以消歧。


## 0. 结论摘要

先说好消息：**现有测试体系是可信的**。L0/L1/L2/L3 四层共 61 条全部通过，
其中含真实 tokenizer 的 128/512 token 上限、索引与 chunks.json 逐条一致、
Qdrant payload 与分块源**逐字段零差异**（21137/21137 条）、L3 检索质量
`book@1 91.7% / book@k 100% / kw@k 100%`。核心检索链路没有发现正确性缺陷。

发现 **8 类已复现的功能缺陷**，分布在"入库健壮性 / 生成侧错误处理 / 可诊断性 / 文档一致性"上。
按严重度：

下表编号 = `tests/test_defects_regression.py` 的护栏编号（权威）。"状态"一列是本次修复后的实测状态。

| 护栏编号 | 严重度 | 缺陷 | 状态 |
|---------|--------|------|------|
| D1 | **高** | 句内软换行被判为"分句器跨段"，`process` 整库崩溃 | ✅ 已修（段落判据改为空行） |
| D2 | 中 | `fix_quotes` 混合型判定"全有全无"，一个 `“` 就改坏整本引号 | ✅ 已修（硬信号 + 状态机） |
| D3 | 中 | 源文本被静默丢弃（`len(child_text) > 5` 过滤） | ✅ 已修（判据改为"有无正文"；**需重跑 process 生效**） |
| D4 | 中 | `.env` 写 `RAG_QDRANT_HOST`，代码只读 `QDRANT_HOST`，配置静默无效 | ✅ 已修（两个名字都接受） |
| D5 | 中 | 空查询 / 纯标点返回 5 条"看似正常"的结果 | ✅ 已修（`_has_searchable_content` 守护） |
| D6 | 低 | `requirements.txt` 描述了一个不存在的降级函数 | ✅ 已修（改为"硬依赖、无降级"） |
| D7 | 低-中 | `cmd_serve` 把"连不上 Qdrant"误报成"向量数据库为空" | ✅ 已修（两类错误分开报） |
| D8 | 中 | 生成失败被当成"模型答案"写进会话历史并回灌模型 | ✅ 已修（独立 `error` kind） |

本报告另外记录的、**没有测试护栏**的发现（本轮也已全部处理）：

| 本报告原编号 | 缺陷 | 状态 |
|-------------|------|------|
| 原 D3 | `check_health.py` 第 6 项超时 30s < 冷启动实测首字节，把"没预热"报成"检查失败" | ✅ 已修（默认 120s，且超时与故障分开报并给出预热命令） |
| §3 | `top_k > RERANK_TOP_K` 静默截断 | ✅ 已修（显式告警并指出该调哪个开关） |
| §3 | `build_index` 非原子（delete+create，重建期间不可用、无法回滚） | ✅ 已修（写新集合 + 别名原子切换） |
| §3 | 生成轮次远超文档值 | ✅ 已修（PROJECT_DOC 已按实测重写） |

---

## 1. D1（高）句内软换行 → 整个 `process` 崩溃

**位置**：`rag_engine._split_sentences` 的段落守护断言

```python
parts = [p.strip() for p in segment("zh", s) if p.strip()]
bad = next((p for p in parts if "\n" in p or "\r" in p), None)
if bad is not None:
    raise ValueError(f"分句器跨段了，段落硬边界保证已失效（sentencex 行为可能已变）：{bad[:60]!r}")
```

**问题**：sentencex 会**在句子内部保留单个 `\n`**（软换行）。该断言把这种**合法输入**
判成了"第三方库行为变更"，直接抛异常。

**复现**：

```
$ python -c "import rag_engine as RE; print(RE._split_sentences('第一句。\n第二句。'))"
ValueError: 分句器跨段了，段落硬边界保证已失效（sentencex 行为可能已变）：'第一句。\n第二句。'
```

**为什么现在没暴露**：实测四本典籍的句内残留换行**都是 0** —— 现有语料恰好没有软换行。
而这个断言**恰好是最容易被真实数据触发的**：任何"按 40/76 列硬折行"的 txt
（网上 txt 最常见的排版）都会命中。实测：

```
硬折行文本（每 40 字换行）→ _hierarchical_split(...)
ValueError: 分句器跨段了……：'话说天下大势\n，分久必合，合久必分。'
```

**影响放大**：`build_chunks` 是"读完所有书 → 最后一次性写 `chunks.json`"。
按 `AGENTS.md` 的"添加新书"流程，用户把一本折行的书放进 `books/` 后跑 `process`，
会在 55 分钟的流程里崩掉，且**不是崩在第一本书**——前几本书的结果全部作废。

**误诊**：报错信息把输入问题归因为"换版本/换库"，会把人带到完全错误的方向。

**建议**：把软换行归一（`\n`/`\r` → 空或空格）后再做守护断言；
断言只在"确实出现了跨段"时才响（即句子同时包含换行与多个段落内容）。

护栏：`TestSoftWrappedTextCrashesIngest`（3 条 xfail）+ 1 条记录现状的通过用例。

---

## 2. D2（中）生成失败被当作"模型答案"存进会话历史

**位置**：`app._stream_ollama` + `app.run_streamlit`

```python
except Exception as e:
    yield "content", f"请求失败: {e}"      # ← 错误走的是 content 通道
```

`run_streamlit` 收到 `content` 就累积进 `answer`，回合末尾：

```python
st.session_state.messages.append({"role": "assistant", "content": answer, ...})
```

**后果有两层**：
1. 用户看到一段"回答"，其实是报错文本，与模型输出无法区分；
2. 下一轮 `ollama_messages` 会把这条报错当**助手历史回灌给模型**。

**实测触发路径**（本机各跑过一次，均稳定复现）：

| 场景 | 结果 |
|------|------|
| 模型名写错 | `HTTP 404 {"error":"model 'no-such-model:1b' not found"}` → 变成助手消息 |
| 模型冷启动 | 连续两次请求在 **180.0s 整点** ReadTimeout → 变成助手消息 |
| Ollama 未启动 | `ConnectionError` → 变成助手消息 |

**建议**：错误单独走 `kind="error"`，由 UI 以 `st.error` 呈现且**不入** `session_state.messages`。

护栏：`TestGenerationErrorStoredAsAnswer`。

---

## 3. D3（中）`check_health.py` 的生成侧检查在本机恒失败

**位置**：`check_health.py` 第 6 项，`parser.add_argument("--timeout", type=int, default=30)`

用**非流式** `requests.post(..., timeout=30)` 做一次简短问答。而本机实测：

```
第1次: 34.7s 失败 -> ReadTimeout: read timeout=30
第2次: 30.3s 失败 -> ReadTimeout: read timeout=30
```

同一模型、同一 prompt，`app.py` 走**流式 + 180s** 读超时是能成功的。
即：**健康检查的判据（30s 非流式）比线上路径（180s 流式）严苛 6 倍**，
于是 `python check_health.py` 在本机恒定返回 `exit 1`，第 6 项永远是 ❌ ——
健康检查一旦"常红"，就再也没人看它了。

**建议**：默认超时对齐线上预算（≥180s），或改用与 `app.py` 相同的流式请求。

---

## 4. D4（中）空查询 / 纯标点查询返回 5 条"看似正常"的结果

**位置**：`RAGEngine.hybrid_search` 不校验 `query`

实测（真实索引，top_k=5）：

| query | 返回 |
|-------|------|
| `""` | 5 条（跨 三国演义/水浒传/红楼梦） |
| `"   "` | 5 条 |
| `"\n\n"` | 5 条 |
| `"？？？"` | 5 条（jieba 把标点全过滤掉 → 稀疏通道为空） |

调用方无从区分"召回了"与"输入是空的"。`check_health.py` 里专门为"集合不存在"
写过守卫注释，这里是同一个道理的调用侧镜像。

**建议**：`query.strip()` 为空（或 jieba 分词后 term 数为 0）时直接返回 `[]`。

护栏：`TestEmptyQueryHasNoGuard`。

---

## 5. D5（中）源文本被静默丢弃

**位置**：`build_chunks`

```python
kept = [(c, p) for c, p in chunks if len(c) > 5]
```

被丢掉的块走的是"父子相同"分支（`parent_text == child_text`），
所以丢的**不是重复内容，而是正文本身**。

**实测**：把各书父块按序在原文中贪心定位，父块序列之间出现空隙：

| 书 | 丢失文本 | 字数 |
|----|----------|------|
| 红楼梦 | `宝玉又道：` | 5 |
| 西游记 | `道士云：`、`莫念！`、`难！` | 4+3+2 |
| 合计 | 4 处 | **14 字** |

复现：这 4 个片段在 `chunks.json` 中**作为 `child_text` 一条都不存在**
（`grep` 到的都是别的块里的子串）。

**现有测试反而固化了这个行为**：`test_chunk_invariants.test_child_text_meets_min_length`
断言"产物里不该有 ≤5 字的块"——把丢字当成了不变量。

**建议**：短块不丢弃，改为并入相邻块（或只丢弃**父块也 ≤5 字**的块并计数上报）。
至少要在日志里打印丢弃条数与总字数。

护栏：`TestSourceTextIsDropped`（逐书贪心定位，断言父块序列无空隙）。

---

## 6. D6（中）`.env` 的 Qdrant 变量名与代码不一致

`.env` 底部：

```
# Qdrant 服务器配置（多人模式）
# RAG_QDRANT_HOST=localhost
# RAG_QDRANT_PORT=6333
```

`rag_engine` 读的却是 `QDRANT_HOST` / `QDRANT_PORT`（无 `RAG_` 前缀，全仓库无一处读 `RAG_QDRANT_*`）。
用户按 `.env` 的写法取消注释后，配置**静默无效** ——
正是 `.env` 顶部那段"缺失时直接报错而不是静默跳过"注释想避免的问题类型。

实测：

```
$ RAG_QDRANT_HOST=example.invalid python -c "import rag_engine as RE; print(RE.QDRANT_HOST)"
localhost        # 期望 example.invalid
```

护栏：`TestEnvQdrantKeyMismatch`（子进程实测 + 文档一致性）。

---

## 7. D7（低-中）`cmd_serve` 把连接失败误报成"向量库为空"

```python
try:
    count = client.count(...).count
except Exception:
    count = 0
if count == 0:
    print("错误: 向量数据库为空"); print("请先运行: python app.py process / index")
```

实测：

```
$ QDRANT_PORT=6999 python app.py serve
错误: 向量数据库为空
请先运行: python app.py process
         python app.py index
```

Docker 没起时，用户被引导去**重跑 55 分钟的分块和建索引**，而真正原因是连接失败；
异常本身一个字都没打印。

讽刺的是 `check_health.py` 与 `RAGEngine.count()` 都专门写注释防这个坑
（"把'索引没建'误报成'Docker 挂了'"）——这里是它的镜像错误，反而漏了。

护栏：`TestServeMisdiagnosesConnectionFailure`。

---

## 8. D8（低）`requirements.txt` 引用不存在的降级函数

```
# 缺失时会自动降级为内置标点分句（rag_engine._split_sentences_rule），不会报错。
```

实际：`rag_engine` 顶层 `from sentencex import segment` 是**硬依赖**，
且 `_split_sentences_rule` 在整个仓库**不存在**（`AGENTS.md` 与 `PROJECT_DOC.md`
的说法才是对的："缺库直接 ImportError，刻意不降级"）。
照 requirements 理解去卸载 sentencex，得到的是模块起不来，不是"降级"。

护栏：`TestDocsReferenceNonexistentCode`（自动校验 requirements 里引用的
`rag_engine.<符号>` 都真实存在）。

---

## 9. 其他观察（未写成护栏，均已实测）

| 项 | 实测 | 评估 |
|----|------|------|
| `top_k` 上限静默截断 | `hybrid_search(q, top_k=100)` 返回 **20** 条，无任何提示 | 低。`RERANK_TOP_K=20` 是设计上限，但应警告或抛错 |
| `fix_quotes` 混合型分支前提错误 | 红楼梦 2426 个直引号里 **1722 个**前面是句读/空白/右引号，即真正的**开引号**（如 `而借"通灵"之说`）；`(?<=[^\s])` 在中文里几乎恒真，实际等价于"全部转闭引号"。归一后配平差从 `+1600` 变成 `-826` | 低。**回归验证过**：它仍优于"不归一"（最长句 964→705 字）和"交替配对"（最长 741 字），所以是**注释里的论证错了**，不是行为需要回退。建议改注释并补一条说明为何接受 |
| `build_index` 非原子 | 先 `delete_collection` 再 `create` 再批量 upsert；中途失败会留下半成品集合 | 低。`_get_chunks` 的条数自校验能发现（会显式报错而非静默漏召回），但服务不可用且需重跑 |
| 生成轮次耗时 | 生产默认（`think=true, num_ctx=8192, num_predict=4096`）实测 **439s**，`eval_count=2364`；`app.py` 注释写的是"整轮 ~135s" | 低。3 倍偏差；`think=false` 的短问答首字节实测可达 **106s**，是 D2/D3 的共同成因 |
| `_semantic_split` 编码截断 | 句子编码 `truncation=True, max_length=512`，而语料最长单句 705 字 | 低。只影响语义边界的相似度计算，不影响落库文本 |
| 并发 | `st.cache_resource` 让所有会话共享一个 `RAGEngine`（内含 torch 模型 + Qdrant client），懒加载无锁 | 低-中。多标签页并发时会重复加载模型/竞态；未实测复现，故未列为缺陷 |

---

## 10. 明确验证为**正常**的部分

- 四层测试 61 条全绿；L3 检索质量 `book@1 91.7% / book@k 100% / kw@k 100%`，
  唯一 `book@1` 未命中的是"火烧赤壁"（top1 是西游记），属排序质量而非缺陷。
- Qdrant payload 与 `chunks.json` **21137/21137 条逐字段零差异**
  （`chunk_id`/`child_text`/`parent_text`/`book`/`chunk_index`/`total_chunks`）。
- `total_chunks` 与各书实际条数一致；point id 连续；build_id 唯一。
- 稀疏单通道可召回；稠密向量已 L2 归一化；集合配平无误。
- 异常输入不崩：超长 query（2 万字符）、emoji、纯拉丁、单个引号都能正常返回。
- `check_health.py --offline` 6 项全部通过。
- `fix_quotes` 对无直引号的三本书是**逐字节等价的空操作**（红楼梦除外）。

## 11. 建议的修复顺序

1. **D1** —— 唯一会"整库崩掉 + 丢 55 分钟工作"的缺陷，且极易被新书触发。
2. **D2 / D3** —— 生成侧的错与"假失败"直接决定用户是否信任这个系统。
3. **D5 / D6** —— 静默丢数据、静默无效配置，都属于"最难排查"的一类。
4. **D4 / D7 / D8** —— 接口与文案的守卫，改动成本低。

修好后把 `tests/test_defects_regression.py` 里对应的 `xfail(strict=True)`
摘掉即可 —— 该文件同时就是这批缺陷的回归护栏。
