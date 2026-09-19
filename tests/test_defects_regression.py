"""缺陷回归护栏 —— 每条对应一个**已复现**的功能缺陷。

写法约定：断言的是**正确行为**。缺陷未修时标 `xfail(strict=True)`；
**本文件当前 8 条缺陷已全部修复，故已无 xfail 标记**（摘标记这个动作本身
由 strict xfail 强制：修好后它会变成 XPASS 失败，不摘就一直红着）。
这样做的两个好处：
  1. 缺陷未修时，测试套件保持绿色（不会把"已知缺陷"和"新回归"混在一起）；
  2. 缺陷被修好时，strict xfail 会立刻变成 XPASS 失败，强制把标记摘掉 ——
     否则这类测试会永远停在那里假装还有问题。

按 pytest.ini 的分层 marker 归类。运行：
    pytest -m unit            # D1/D2(引号)/D4/.env 名单/文档一致性
    pytest -m needs_chunks    # D3 语料丢字
    pytest -m needs_qdrant    # D5 空查询
每条 docstring 里写了复现方式与实测数据。
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import rag_engine as RE

ROOT = Path(__file__).resolve().parent.parent

# D1 用的硬折行文本（每 20 字换一行，模拟最常见的 txt 排版）
_WRAPPED_BODY = "话说天下大势，分久必合，合久必分。" * 4
WRAPPED_TEXT = "\n".join(
    _WRAPPED_BODY[i:i + 20] for i in range(0, len(_WRAPPED_BODY), 20)
)


# ============================================================================
# D1（HIGH）句内软换行把整个 process 流程打崩
# ============================================================================
class TestSoftWrappedTextCrashesIngest:
    """`_split_sentences` 的段落守护断言把**合法输入**判成了分句器失效。

    sentencex 的输出里，句子内部保留单个 \\n（软换行）。该函数 `.strip()` 后
    仍发现 "\\n" 就抛 ValueError，于是任何"段落被硬折行"的 txt（每 40/76 列
    换行，是网上 txt 最常见的排版）都会让 `python app.py process` 直接崩掉。

    实测（本机）：
        四本典籍 句内残留换行 = 0，所以现有语料不触发；
        "\n".join(每40字一段) 的同一段文字立刻 ValueError。
    """

    pytestmark = pytest.mark.unit

    TEXT = WRAPPED_TEXT

    def test_soft_wrap_does_not_raise(self):
        parts = RE._split_sentences(self.TEXT)
        assert parts, "软换行文本应当仍能分句"

    def test_hierarchical_split_does_not_raise(self, fake_tok):
        pairs = RE._hierarchical_split(self.TEXT, fake_tok)
        assert pairs

    def test_error_message_does_not_blame_sentencex(self):
        try:
            RE._split_sentences(self.TEXT)
        except ValueError as exc:
            assert "sentencex" not in str(exc), (
                f"输入里只是有软换行，不是 sentencex 行为变更，却报: {exc}"
            )

    def test_current_corpus_has_no_interior_newline(self):
        """记录现状：现有四本书恰好处处没有句内换行，所以缺陷被掩盖了。

        这条是**通过**的，作用是把这个前提写进测试 —— 一旦换成别的书，
        D1 就会在真实数据上复现。
        """
        from sentencex import segment

        book = ROOT / RE.BOOKS_DIR / "水浒传.txt"
        if not book.exists():
            pytest.skip("缺少语料")
        text = RE.read_book_text(str(book))
        bad = [p for p in segment("zh", text) if "\n" in p.strip() or "\r" in p.strip()]
        assert bad == [], f"语料里出现了句内换行，D1 会在真实数据上触发: {bad[:2]}"


# ============================================================================
# D2（LOW-MED）fix_quotes 的"混合型"判定是全有全无，一个 “ 就改坏整本
# ============================================================================
class TestFixQuotesMixedModeTrap:
    """`fix_quotes` 只按"全文是否出现过 “" 二选一。

    混合型分支里的位置规则 `(?<=[^\\s])"` 在中文里几乎恒真（汉字之间没有空格），
    所以它实际等价于"所有直引号 → 闭引号"。于是：
      * 一本本来正常使用直引号的书，只要正文里出现过**一个** “（引号、
        注释、错字都可能），整本的直引号会被全部改成闭引号；
      * 红楼梦实测 2426 个直引号里 1722 个前面是句读/空白/右引号
        （即真正的**开引号**，如 `而借"通灵"之说`），它们全被改成了 ”。
    """

    pytestmark = pytest.mark.unit

    def test_single_stray_curly_quote_does_not_flip_whole_document(self):
        text = '他说"你好"，她说"再见"。序言里出现过“这个字。'
        out = RE.fix_quotes(text)
        assert out.startswith("他说“你好”，她说“再见”。"), (
            f"开引号被改成了闭引号: {out!r}"
        )

    def test_mixed_mode_rule_is_not_degenerate(self):
        """中文文本里开引号前也是汉字，`(?<=[^\\s])` 无法区分开/闭。

        注意必须带一个 “ 才会进入（旧实现的）混合型分支。

        ⚠️ 期望值在 2026-09 的语料复核后改变：`序言里有“这个字。而借"通灵"之说`
        里那个 “ **没有**闭合的 ”，按"是否处在未闭合引号内"的判据，`"` 应当
        闭掉它 —— 旧期望（`而借“通灵”之说`）建立在"汉字后的直引号必为开引号"
        这个前提上，而该前提已被 books/红楼梦.txt 证伪（`："` 出现 0 次、
        `？"` 915 次、`！"` 402 次，直引号一律是闭引号）。
        真正防止"全有全无改坏整本"的是**状态机**，不是位置启发式：
        下面第二条断言证明了直引号仍能被成对处理。
        """
        out = RE.fix_quotes('序言里有“这个字。而借"通灵"之说')
        assert out == '序言里有“这个字。而借”通灵”之说', (
            f"未闭合的 “ 应当被下一个直引号闭合: {out!r}"
        )
        # 没有未闭合开引号时，直引号仍然是**开**引号（不许退化成"全部转闭引号"）
        out2 = RE.fix_quotes('序言里有“这个字。”而借"通灵"之说')
        assert out2.endswith('而借“通灵”之说'), f"开引号被改坏: {out2!r}"


# ============================================================================
# D3（MED）源文本被静默丢弃
# ============================================================================
class TestSourceTextIsDropped:
    """`build_chunks` 用 `len(child_text) > 5` 过滤，过滤掉的是**原文本身**。

    被丢的块 `parent_text == child_text`（短块走"父子相同"分支），
    所以丢的不是重复内容，而是正文。实测四本书合计丢 14 字：
        红楼梦 `宝玉又道：`（5 字，len>5 不成立）
        西游记 `道士云：`（4 字）、`莫念！`（3 字）、`难！`（2 字）

    现有 L1 用例 `test_child_text_meets_min_length` 反而**断言了**这些块不存在，
    等于把丢字当成了不变量。
    """

    pytestmark = pytest.mark.needs_chunks

    def _find_gaps(self, book, chunks):
        """贪心定位：父块按顺序在原文中查找，缝隙即被丢弃的正文。"""
        norm = lambda s: re.sub(r"\s+", "", s)
        orig = norm(RE.read_book_text(str(ROOT / RE.BOOKS_DIR / f"{book}.txt")))
        pos, gaps, last = 0, [], None
        for c in chunks:
            p = norm(c["parent_text"])
            # 同一父块被它下面的多个子块重复引用，只消费一次
            if not p or p == last:
                continue
            last = p
            idx = orig.find(p, pos)
            if idx < 0:
                continue
            if idx > pos:
                gaps.append(orig[pos:idx])
            pos = idx + len(p)
        return gaps

    def test_no_source_gap_between_parent_chunks(self, chunks):
        from collections import defaultdict

        by_book = defaultdict(list)
        for c in chunks:
            by_book[c["book"]].append(c)

        lost = {}
        for book, cs in by_book.items():
            g = self._find_gaps(book, cs)
            if g:
                lost[book] = g
        total = sum(len(t) for g in lost.values() for t in g)
        assert not lost, (
            f"分块产物丢失了 {total} 字原文（父块序列出现 {sum(len(v) for v in lost.values())} 处空隙）: "
            f"{ {b: v[:3] for b, v in list(lost.items())[:2]} }"
        )


# ============================================================================
# D4（MED）`.env` 里 Qdrant 的变量名与代码读取的名字不一致
# ============================================================================
class TestEnvQdrantKeyMismatch:
    """`.env` 底部写的是 `RAG_QDRANT_HOST` / `RAG_QDRANT_PORT`，
    而 `rag_engine` 读的是 `QDRANT_HOST` / `QDRANT_PORT`。

    用户按 `.env` 的注释取消注释后，配置**静默无效** —— 这正是 .env 顶部
    "缺失时直接报错而不是静默跳过"那段注释想避免的问题。
    """

    pytestmark = pytest.mark.unit

    def test_documented_env_key_is_actually_read(self):
        env = dict(os.environ, RAG_QDRANT_HOST="example.invalid", RAG_QDRANT_PORT="7999")
        code = "import rag_engine as RE; print(RE.QDRANT_HOST, RE.QDRANT_PORT)"
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr[-500:]
        assert out.stdout.split() == ["example.invalid", "7999"], (
            f"按 .env 的写法设置 RAG_QDRANT_HOST 后，代码仍然用 {out.stdout.strip()!r}"
        )

    def test_env_documents_the_effective_key(self):
        """.env 必须给出代码真正读取的名字，否则用户无从得知正确写法。"""
        text = (ROOT / ".env").read_text(encoding="utf-8")
        assert re.search(r"^#?\s*RAG_QDRANT_HOST=", text, re.M) is None, (
            ".env 把 RAG_QDRANT_HOST 写成了生效写法，但代码读的是 QDRANT_HOST"
        )
        assert re.search(r"^#?\s*QDRANT_HOST=", text, re.M), "未记录 QDRANT_HOST"


# ============================================================================
# D5（MED）空查询 / 纯标点查询返回 5 条"看似正常"的结果
# ============================================================================
class TestEmptyQueryHasNoGuard:
    """`hybrid_search` 不校验 query。实测 `hybrid_search("")` 返回 5 条结果，
    跨 3 本书、带 rerank 分数，调用方无从区分"召回了"和"输入是空的"。

    实测："" / "   " / "？？？" / "\\n\\n" 全部返回 5 条命中。
    """

    pytestmark = pytest.mark.needs_qdrant

    def test_empty_query_returns_nothing(self, engine):
        for bad in ("", "   ", "\n\n", "？？？"):
            out = engine.hybrid_search(bad, top_k=RE.TOP_K)
            assert out == [], f"query={bad!r} 返回了 {len(out)} 条结果"


# ============================================================================
# D6（LOW-MED）requirements.txt 描述了一个不存在的降级函数
# ============================================================================
class TestDocsReferenceNonexistentCode:
    """requirements.txt 写"sentencex 缺失时会自动降级为内置标点分句
    （rag_engine._split_sentences_rule），不会报错"，但：

      * `rag_engine` 顶层 `from sentencex import segment`，是硬依赖；
      * `_split_sentences_rule` 在仓库里根本不存在。

    照这份文档理解，运维会以为删掉 sentencex 只是"降级"，实际是整个模块
    ImportError 起不来。
    """

    pytestmark = pytest.mark.unit

    def test_referenced_symbols_exist(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        refs = set(re.findall(r"rag_engine\.([A-Za-z_][A-Za-z0-9_]*)", text))
        missing = sorted(r for r in refs if not hasattr(RE, r))
        assert not missing, f"requirements.txt 引用了不存在的 rag_engine.{missing}"

    def test_sentencex_is_declared_hard_dependency(self):
        """代码是硬依赖，文档就不能说"缺失会自动降级"。"""
        import inspect

        src = inspect.getsource(RE)
        assert "from sentencex import segment" in src
        assert "_split_sentences_rule" not in src, (
            "rag_engine 里并不存在降级函数，requirements.txt 的说法与代码不符"
        )


# ============================================================================
# D7（LOW-MED）cmd_serve 把"连不上 Qdrant"误报成"向量库为空"
# ============================================================================
class TestServeMisdiagnosesConnectionFailure:
    """`cmd_serve` 用 `except Exception: count = 0` 把连接失败吞成 0，
    于是 Docker 没起时用户看到的是"向量数据库为空，请先 process + index" ——
    排查方向完全错误。

    同一个项目在 check_health.py 里专门写注释防这个坑（"把索引没建误报成
    Docker 挂了"），这里恰好是它的镜像错误。
    """

    pytestmark = pytest.mark.unit

    def test_unreachable_qdrant_reports_connection_error(self):
        env = dict(os.environ, QDRANT_PORT="6999")   # 无人监听
        out = subprocess.run(
            [sys.executable, "app.py", "serve"], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=180,
        )
        blob = out.stdout + out.stderr
        assert "向量数据库为空" not in blob, (
            f"Qdrant 连不上，却报成'向量数据库为空':\n{blob[-500:]}"
        )


# ============================================================================
# D8（MED）生成失败被当成"模型答案"写进会话历史
# ============================================================================
class TestGenerationErrorStoredAsAnswer:
    """`stream_chat` 曾把异常/HTTP 错误都 `yield ("content", ...)`。

    `run_streamlit` 收到 "content" 就当成模型输出累积进 `answer`，并在回合末尾
    `st.session_state.messages.append({"role": "assistant", "content": answer})`。
    后果有两层：
      1. 用户看到"回答"其实是一条报错文本，与模型输出无法区分；
      2. 下一轮 `ollama_messages` 会把这条报错当**助手历史**回灌给模型。

    实测触发：模型冷启动时首字节等待超过 `requests` 的 180s 读超时
    （本机连跑两次都是 180.0s 整点失败）；模型名写错时是 HTTP 404。
    """

    pytestmark = pytest.mark.unit

    def test_error_is_not_delivered_as_content(self, monkeypatch):
        import ollama_client as OC

        # 指向一个必然拒绝连接的端口：等价于 Ollama 挂掉 / 超时
        monkeypatch.setattr(OC, "API_URL", "http://127.0.0.1:1/api/chat")
        messages = [{"role": "user", "content": "hi"}]

        kinds = [k for k, _ in OC.stream_chat(messages)]
        assert "content" not in kinds, (
            "生成失败被当作模型答案（kind=content）返回，UI 会把它存成助手消息并回灌"
        )


# ============================================================================
# D9（HIGH）配置是否生效取决于 import 顺序
# ============================================================================
class TestEnvNotLoadedByImportOrder:
    """.env 里的 RAG_QUERY_REWRITE=0 会被"先 import 了 rag_engine"这件事抵消。

    `query_rewrite` 在 **import 时**就把开关读成模块常量，而 `rag_engine` 顶层
    就 import 它。于是：

        python -c "import rag_engine, query_rewrite"   → REWRITE_ENABLED = True
        python -c "import app, query_rewrite"          → REWRITE_ENABLED = False

    前者忽略了 .env 里的 0，改写照跑。受害面是**同一份配置在两个入口表现不同**：

      * app.py（UI）与 api.py 恰好是对的 —— app.py 在 import rag_engine 之前先
        `load_dotenv()`，而在本条修复之前 api.py 是靠 `import app as A` 顺带加载的
        （那次依赖已随 D13 的单源改造一并删除）；
      * `check_health.py` 是错的：它先 import rag_engine，于是把"已关闭"
        **报成"开启"**（健康检查本身给出错误结论）；
      * `python -m tests.record_baseline` 与 `tools/ab_retrieval.py` /
        `compare_ab.py` / `verify_qdrant.py` 等一切先 import rag_engine 的工具，
        .env 里的 0 一律失效 —— "关掉改写再跑一遍做 A/B"实际上根本没关掉。
        （例外：命令行环境变量仍优先，因为 load_dotenv 默认不覆盖已存在的变量，
        所以 `RAG_QUERY_REWRITE=0 python ...` 是灵验的。）

    这与 D4（`RAG_QDRANT_HOST` 写进 .env 却静默无效）是同一类缺陷，
    只是成因从"变量名写错"变成了"加载顺序"。

    这里用**子进程**复现，因为 import 顺序是本缺陷的唯一变量，进程内已经 import
    过的模块无法再"换一种顺序"导入。断言的是两个顺序必须得到同一个结论 ——
    它不依赖 .env 的具体取值（本机 CI 上没有 .env 时同样成立）。
    """

    pytestmark = pytest.mark.unit

    @staticmethod
    def _run(imports, env=None):
        out = subprocess.run(
            [sys.executable, "-c",
             f"{imports}; import query_rewrite as QR; "
             f"print(QR.REWRITE_ENABLED, QR.REWRITE_MODEL)"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr[-500:]
        return out.stdout.strip()

    def test_import_order_does_not_change_effective_config(self):
        via_rag_engine = self._run("import rag_engine")
        via_app = self._run("import app")
        assert via_rag_engine == via_app, (
            f"配置随 import 顺序变化：先 import rag_engine → {via_rag_engine}，"
            f"先 import app → {via_app}。.env 必须由 query_rewrite 自己加载。"
        )

    def test_cli_override_still_wins(self):
        """修法是"自己加载 .env（override=False）"，命令行环境变量必须仍然优先，
        否则 tools/ab_retrieval.py 那套 A/B 用法（只改环境变量）会失效。"""
        env_off = {**os.environ, "RAG_QUERY_REWRITE": "0"}
        env_on = {**os.environ, "RAG_QUERY_REWRITE": "1"}
        def run(env):
            out = subprocess.run(
                [sys.executable, "-c",
                 "import rag_engine, query_rewrite as QR; print(QR.REWRITE_ENABLED)"],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
            )
            assert out.returncode == 0, out.stderr[-500:]
            return out.stdout.strip()
        assert run(env_off) == "False" and run(env_on) == "True"

    def test_env_file_value_actually_reaches_the_module(self):
        """最直接的一条：**.env 里写的值必须就是模块看到的值**。

        上面两条只断言"两种 import 顺序一致"，若 .env 是 1（或本机没有 .env），
        即使回归了也可能侥幸通过。这里把 .env 的实际取值读出来对拍，
        并显式清掉环境变量（否则 shell 里的同名变量会让断言失去意义）。
        """
        env_file = ROOT / ".env"
        if not env_file.exists():
            pytest.skip("本机没有 .env，无从对拍")
        m = re.search(r"^RAG_QUERY_REWRITE=(\S+)", env_file.read_text(encoding="utf-8"), re.M)
        if not m:
            pytest.skip(".env 未显式设置 RAG_QUERY_REWRITE")
        expected = m.group(1).strip().lower() not in ("0", "false", "no", "off")
        env = {k: v for k, v in os.environ.items() if k != "RAG_QUERY_REWRITE"}
        got = self._run("import rag_engine", env=env).split()[0]
        assert got == str(expected), (
            f".env 写着 RAG_QUERY_REWRITE={m.group(1)}（应为 {expected}），"
            f"但 `import rag_engine` 之后模块看到的是 {got} —— .env 没有生效"
        )


# ============================================================================
# D10（HIGH）fix_quotes 把**闭引号**改成了开引号 —— 语料级验收
# ============================================================================
class TestQuoteNormalizationKeepsCorpusBalance:
    """旧实现的硬信号把"直引号左侧紧跟句读"当作必为**开**引号，前提是反的。

    实测 books/红楼梦.txt（2426 个直引号）：
        `："`   0 次   开引号一律写 “
        `？"` 915 次   闭引号
        `！"` 402 次   闭引号
    旧实现把 2094 个直引号判成开引号（其中 1734 个紧跟中文标点），结果是
    `“` 5802→7896、`”` 4202→4534，**未配对开引号从 1600 涨到 3362** ——
    这个函数本想修的引号错位被它做成了一倍，且坏文本已写进 cache_v2/chunks.json
    （能直接搜到 `意欲何往？“那僧笑道` 这样的样本）。

    新实现（单遍二值状态机）在同一份文本上的复算结果：
        `“` 5802→6221、`”` 4202→6209，**未配对 1600 → 12**。

    单元级护栏在 tests/test_chunking.py::TestFixQuotes。这里守住的是**验收线**：
    归一后各本书的不平衡量不得大于原文（原文本身就有错字/漏排，压不到 0）。
    """

    pytestmark = pytest.mark.unit

    @staticmethod
    def _book(name):
        p = ROOT / "books" / f"{name}.txt"
        if not p.exists():
            pytest.skip(f"缺少 {p}，无法做语料级验收")
        return p.read_bytes().decode("utf-8", errors="replace")

    @pytest.mark.parametrize("book", ["红楼梦", "三国演义", "水浒传", "西游记"])
    def test_imbalance_does_not_grow(self, book):
        text = self._book(book)
        raw_gap = abs(text.count("\u201c") - text.count("\u201d"))
        out = RE.fix_quotes(text)
        new_gap = abs(out.count("\u201c") - out.count("\u201d"))
        assert new_gap <= raw_gap, (
            f"{book}: 归一后引号不平衡量 {new_gap} > 原文 {raw_gap} —— "
            f"归一在把引号改坏（旧实现在红楼梦上是 1600 → 3362）"
        )

    def test_no_closing_quote_becomes_an_opening(self):
        """红楼梦里"句末标点后的直引号"必须落在 ”（闭）上，而不是 “。

        旧实现：`？"` → `？“` 1734 次（判成开引号）。
        新实现复算：`？”` 973 次、`？“` 仅 6 次（那 6 次是状态机在上文缺失
        时的边界处理，不是系统性翻转）。
        """
        text = self._book("红楼梦")
        out = RE.fix_quotes(text)
        for punct, floor in (("？", 900), ("！", 390)):
            closed = out.count(punct + "\u201d")
            opened = out.count(punct + "\u201c")
            assert closed >= floor, (
                f"{punct}\" 应当变成 {punct}”，实际只有 {closed} 次"
            )
            assert opened < 50, (
                f"{punct}\" 被改成 {punct}“ 达 {opened} 次 —— 闭引号被系统性翻转"
                f"（旧实现在红楼梦上是 1734 次）"
            )


# ============================================================================
# D11（HIGH）chat.py 把 hybrid_search 的**列表**返回值按二元组解包
# ============================================================================
class TestHybridSearchCallSitesMatchReturnShape:
    """`results, _ = engine.hybrid_search(...)` 在 return_steps 默认 False 时会炸。

    实测：chat.py 就这么写的，而 hybrid_search 默认返回**列表** ——
    条数不为 2 时 `ValueError: too many values to unpack`，恰好为 2 时
    results 变成一个 dict，随后 format_context 会对字符串调 .get。

    这里做的是**静态**检查（AST），因为运行时覆盖需要起 Streamlit：
    凡是把 hybrid_search 的返回值按二元组解包的调用，都必须显式带
    return_steps=True。返回形状变了而调用方没跟，是这类缺陷的通用形态。
    """

    pytestmark = pytest.mark.unit

    SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
                 "models", "qdrant_storage", "cache_v2"}

    def test_tuple_unpacking_requires_return_steps(self):
        import ast

        offenders = []
        for path in sorted(ROOT.rglob("*.py")):
            if any(part in self.SKIP_DIRS for part in path.parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                call = node.value
                if not (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "hybrid_search"):
                    continue
                # 左侧是元组/列表解包（a, b = ... 或 (a, b) = ...）
                if not isinstance(node.targets[0], (ast.Tuple, ast.List)):
                    continue
                kw = {k.arg: k.value for k in call.keywords}
                rs = kw.get("return_steps")
                if not (isinstance(rs, ast.Constant) and rs.value is True):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} —— "
                        f"把它按元组解包了，却没有 return_steps=True")
        assert not offenders, (
            "hybrid_search 的解包与返回形状不一致（return_steps 默认 False 时"
            "只返回列表）：\n  " + "\n  ".join(offenders)
        )


# ============================================================================
# D12（HIGH）rag_engine 的配置依赖 import 顺序（D9 的根因在上一层）
# ============================================================================
class TestRagEngineLoadsEnvItself:
    """`rag_engine` 的常量在 import 时求值，而它自己不 load_dotenv()。

    此前没出事纯属侥幸：文件里 import 了 query_rewrite，而 query_rewrite 自己
    load_dotenv()、且那次 import 恰好排在配置段之前。把 import 挪一下，
    check_health.py / tools/*.py / tests/conftest.py 这些"先 import rag_engine"
    的入口就会**静默**丢掉 .env 里的全部 RAG_* 配置。
    """

    pytestmark = pytest.mark.unit

    def _run(self, imports, env=None):
        out = subprocess.run(
            [sys.executable, "-c",
             f"{imports}; import rag_engine as R; "
             f"print(R.RRF_K, R.ABSTAIN_MEAN_HARD, R.CHILD_MAX_TOKENS)"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr[-500:]
        return out.stdout.strip()

    def test_import_order_does_not_change_effective_config(self):
        via_engine = self._run("import rag_engine")
        via_app = self._run("import app")
        assert via_engine == via_app, (
            f"检索配置随 import 顺序变化：先 import rag_engine → {via_engine}，"
            f"先 import app → {via_app}。rag_engine 必须自己 load_dotenv()。"
        )

    def test_env_file_value_actually_reaches_rag_engine(self):
        """只有"两种顺序一致"是不够的：两者都读到默认值时同样会通过。

        这里挑一个 .env 里**显式写了非默认值**的检索侧变量对拍。
        """
        env_file = ROOT / ".env"
        if not env_file.exists():
            pytest.skip("本机没有 .env，无从对拍")
        text = env_file.read_text(encoding="utf-8")
        pairs = [("RAG_RRF_K", RE.RRF_K), ("RAG_CHILD_MAX_TOKENS", RE.CHILD_MAX_TOKENS),
                 ("RAG_TOP_K", RE.TOP_K)]
        for name, default in pairs:
            m = re.search(rf"^{name}=(\S+)", text, re.M)
            if not m:
                continue
            expected = int(m.group(1))
            if expected == default:
                continue     # .env 写的就是默认值，对拍没有区分度
            env = {k: v for k, v in os.environ.items() if k != name}
            got = self._run("import rag_engine", env=env).split()[0]
            assert got == str(expected), (
                f".env 写着 {name}={expected}，但 `import rag_engine` 之后模块"
                f"看到的是 {got} —— 配置在 import 时就被冻住了"
            )
            return
        pytest.skip(".env 里没有显式非默认的检索侧变量，无从对拍")


# ============================================================================
# D13（MED）生成侧配置与流式实现必须只有一个来源
# ============================================================================
class TestOllamaConfigHasSingleSource:
    """`app._stream_ollama` 曾与 `ollama_client.stream_chat` 是两份同逻辑实现，
    且 MODEL / THINK / NUM_CTX / NUM_PREDICT / OLLAMA_TIMEOUT 在两边各读一遍。

    后果全是缺陷：超时一处硬编码 180s 而另一处读 `OLLAMA_TIMEOUT`（用户调大超时
    只对一个入口生效）；RAG_USE_CONTEXT 只在 app.py 读、别的入口靠转发（chat.py
    就漏过）。现在四个入口一律 `from ollama_client import …`，值不可能分叉。
    """

    pytestmark = pytest.mark.unit

    ENTRY_FILES = ("app.py", "api.py", "chat.py", "bot.py")
    #: 这些环境变量只允许 ollama_client 自己读
    OLLAMA_ENV = ("MODEL", "OLLAMA_BASE_URL", "OLLAMA_TIMEOUT",
                  "RAG_THINK", "RAG_NUM_CTX", "RAG_NUM_PREDICT")

    def test_values_come_from_ollama_client(self):
        """四个入口用的必须是**同一个** stream_chat 对象，且拿到同一份常量。

        入口只导入自己需要的名字（chat.py 不需要 MODEL，app.py 不需要 TIMEOUT），
        所以按 hasattr 逐个校验；`is` 而不是 `==` 才是这里要的语义 ——
        值相等可能只是巧合，对象相同才说明没有第二份定义。
        """
        import ollama_client as OC

        for mod in ("app", "api", "chat", "bot"):
            m = __import__(mod)
            assert m.stream_chat is OC.stream_chat, (
                f"{mod} 用的不是 ollama_client.stream_chat —— 流式实现又被抄了一份"
            )
            for name in ("MODEL", "THINK", "NUM_CTX", "NUM_PREDICT",
                         "OLLAMA_BASE_URL", "TIMEOUT"):
                if hasattr(m, name):
                    assert getattr(m, name) is getattr(OC, name), (
                        f"{mod}.{name} 不是 ollama_client 的那个对象"
                        f" —— 又出现了第二份配置"
                    )

    def test_entry_points_do_not_re_read_ollama_env(self):
        """入口文件里不许再出现对这些环境变量的读取（AST，不靠人眼）。"""
        import ast

        offenders = []
        for fname in self.ENTRY_FILES:
            tree = ast.parse((ROOT / fname).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("getenv", "environ")):
                    continue
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value in self.OLLAMA_ENV:
                        offenders.append(f"{fname}:{node.lineno} 读了 {arg.value}")
        assert not offenders, (
            "生成侧配置只允许 ollama_client 读取，入口不得各读一遍：\n  "
            + "\n  ".join(offenders)
        )


# ============================================================================
# D14（MED）三个入口的生成侧装配必须同口径
# ============================================================================
class TestGenerationPlanIsShared:
    """chat.py 曾漏传 use_context，于是 `RAG_USE_CONTEXT=0` 在它那里静默失效。

    plan_generation 把"传不传开关、判不判硬拒答、资料消息怎么来"收进一个函数，
    这里守住它的三条性质；入口只需要调它，不再各自拼装。
    """

    pytestmark = pytest.mark.unit

    RES = [{"id": "水浒传_0", "book": "水浒传", "chapter_label": "第二十三回",
            "chapter_title": "景阳冈", "parent_text": "武松打虎。",
            "child_text": "武松打虎。", "rerank_score": 1.0}]

    def test_control_mode_sends_nothing_from_retrieval(self):
        plan = RE.plan_generation(self.RES, [], "问题", 8192, 4096,
                                  use_context=False)
        assert plan["system"] == "" and plan["context"] == ""
        assert plan["messages"] == [{"role": "user", "content": "问题"}]

    def test_rag_mode_sends_rules_and_a_separate_context_message(self):
        plan = RE.plan_generation(self.RES, [], "问题", 8192, 4096,
                                  use_context=True)
        assert plan["system"] == RE.SYSTEM_RULES_PROMPT
        roles = [m["role"] for m in plan["messages"]]
        assert roles == ["system", RE.CONTEXT_ROLE, "user"]
        assert "武松打虎" in plan["messages"][1]["content"]

    def test_abstain_only_when_rag_is_on_without_results(self):
        assert RE.plan_generation([], [], "问", 8192, 4096,
                                  use_context=True)["abstain"] is True
        assert RE.plan_generation([], [], "问", 8192, 4096,
                                  use_context=False)["abstain"] is False
        assert RE.plan_generation(self.RES, [], "问", 8192, 4096,
                                  use_context=True)["abstain"] is False

    def test_budget_subtracts_context_and_reports_user_truncation(self):
        huge = "长" * 20000
        plan = RE.plan_generation(self.RES, [], huge, 8192, 4096, use_context=True)
        assert plan["stats"]["truncated_user"] is True
        assert plan["stats"]["budget"] < 8192 - 4096



# ============================================================================
# D15（结构）三份"契约"必须只有一个定义 —— 散弹式修改的护栏
# ============================================================================
class TestContractsHaveSingleSource:
    """同一份知识写在多处，就一定会分叉。这一组守的是**结构**，不是行为。

    实测过的分叉：`compare_ab` 自己抄了一份召回管线（抄漏 with_payload 而崩）、
    `verify_qdrant` 抄了一份抽样检查（抄少了守卫而在缺稀疏时崩、
    又把两通道拼接而不去重）、`chat.py` 抄了一份装配（漏 use_context）。
    这些都不是"写错了"，而是"有两份"。
    """

    pytestmark = pytest.mark.unit

    def test_payload_schema_has_one_definition(self):
        """分块 schema 只有一处声明；派生出来的三处必须与它一致。"""
        spec_fields = set(RE.PAYLOAD_FIELDS)
        assert "chunk_id" in spec_fields and "chapter_title" in spec_fields
        # ① 写入端（build_index）与读取端（chunk_from_payload）共用同一张表：
        #    用一条真实记录往返验证，字段集必须完全一致。
        chunk = {k: (0 if k.endswith("_index") or k == "total_chunks"
                     or k == "parent_chunk_count" else "x")
                 for _f, k, _d in RE._PAYLOAD_SPEC}
        payload = RE.chunk_payload(chunk, "build123", "lex123")
        assert set(payload) == spec_fields | set(RE.PAYLOAD_BOOKKEEPING_FIELDS)
        back = RE.chunk_from_payload(payload, 7)
        assert set(back) == set(RE.CHUNK_FIELDS)

    def test_compare_chunks_field_list_matches_schema(self):
        """compare_chunks 保持零依赖（不 import rag_engine），但它的字段表
        必须与 schema 一致 —— 否则它会对新字段静默给假绿。"""
        import compare_chunks
        # 比对的是 chunks.json 的记录，而 point_id 是建索引时才有的，故对 CHUNK_RECORD_FIELDS
        assert set(compare_chunks.FIELDS) == set(RE.CHUNK_RECORD_FIELDS), (
            f"compare_chunks.FIELDS 与 rag_engine.CHUNK_RECORD_FIELDS 不一致："
            f"{set(compare_chunks.FIELDS) ^ set(RE.CHUNK_RECORD_FIELDS)}"
        )

    def test_health_required_fields_are_a_subset_of_the_schema(self):
        """check_health 列的是"硬要求"策略，不是第二份 schema：
        写错一个字段名会永远为真（那个键根本不存在于任何 payload）。"""
        import check_health
        assert set(check_health.REQUIRED_PAYLOAD_FIELDS) <= set(RE.PAYLOAD_FIELDS)

    def test_steps_has_a_single_writer(self):
        """steps 的写者必须是 rag_engine 里的 hybrid_search（以及 plan_generation）。

        入口只允许写 STEP_INPUT_TOKENS —— 那是生成之后的实测值，检索链不可能知道。
        允许第二个写者就意味着 UI 得知道检索内部的结构（app.py 曾经往里塞
        `上下文装配["messages"]`）。
        """
        import ast

        allowed = {"rag_engine.py": None, "rag_turn.py": None}
        offenders = []
        for path in sorted(ROOT.rglob("*.py")):
            rel = str(path.relative_to(ROOT))
            if rel in allowed or rel.startswith("tests/"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                # 形如 steps[...] = ... 的赋值
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
                    tgt = node.targets[0]
                    if isinstance(tgt.value, ast.Name) and tgt.value.id == "steps":
                        src = ast.unparse(tgt.slice)
                        if "STEP_INPUT_TOKENS" not in src:
                            offenders.append(f"{rel}:{node.lineno} 写了 steps[{src}]")
        assert not offenders, (
            "steps 只允许 rag_engine / rag_turn 写，入口仅可写 STEP_INPUT_TOKENS：\n  "
            + "\n  ".join(offenders)
        )

    def test_entry_points_share_one_turn_sequence(self):
        """四个入口都必须走 rag_turn.run_turn，不许自己拼序列。"""
        import ast

        # bot.py 不做检索，没有"一轮 RAG"可拼，故不在检查范围。
        # api.py 的 /search 是**检索专用端点**（本来就只调 hybrid_search、
        # 不生成），它不构成"一轮"，故那里允许直接调。
        for fname in ("app.py", "api.py", "chat.py"):
            src = (ROOT / fname).read_text(encoding="utf-8")
            calls = {n.func.attr for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            assert "run_turn" in calls, f"{fname} 没有走 rag_turn.run_turn"
        for fname in ("app.py", "chat.py"):
            src = (ROOT / fname).read_text(encoding="utf-8")
            calls = {n.func.attr for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            assert "hybrid_search" not in calls, (
                f"{fname} 自己调了 hybrid_search —— 它应当只消费事件流"
            )

    def test_probe_running_has_one_implementation(self):
        """"跑探针"只有 eval_runner.run_probes_with 一份。"""
        src = (ROOT / "tools" / "ab_retrieval.py").read_text(encoding="utf-8")
        assert "run_probes_with" in src, (
            "tools/ab_retrieval.py 又自己写了一份跑探针的循环"
        )
