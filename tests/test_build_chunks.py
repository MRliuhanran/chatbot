"""build_chunks 的端到端编号回归。

单独成文件的原因：这里跑的是**真实的 build_chunks 主函数**（读目录 → 调分块 →
过滤 → 编号 → 写 json），而不是它的某个子函数。分块本身被 monkeypatch 成固定
输出，"过滤后编号"这段正是出过 bug 的地方，必须用真实调用路径覆盖。

回归背景：旧实现用 enumerate 在**过滤前**编号，被丢掉的短块会留下空洞 ——
实测红楼梦 chunk_index 0..5011 却只有 5010 条、西游记 0..7463 却只有 7461 条，
total_chunks 也随之偏大（5012 / 7464）。

不需要模型权重：只加载 tokenizer 读词表（build_chunks 内部自行加载）。
"""

import json

import pytest

import rag as RE

pytestmark = pytest.mark.needs_models


@pytest.fixture
def tiny_corpus(tmp_path, monkeypatch):
    """把 BOOKS_DIR / CHUNKS_JSON 指向临时目录，避免动到真实产物。"""
    books = tmp_path / "books"
    books.mkdir()
    (books / "测试书.txt").write_text("占位内容，分块被 monkeypatch 接管。", encoding="utf-8")

    out = tmp_path / "chunks.json"
    monkeypatch.setattr(RE, "BOOKS_DIR", str(books))
    monkeypatch.setattr(RE, "CHUNKS_JSON", str(out))
    monkeypatch.setattr(RE, "BM25_CACHE_DIR", str(tmp_path))
    return out


def _run_build_with_chunks(monkeypatch, pairs):
    """让 _hierarchical_split 返回固定的 (child, parent) 序列。"""
    monkeypatch.setattr(RE, "_hierarchical_split", lambda text, tok: list(pairs))


class TestBuildChunksNumbering:
    def test_short_chunks_are_kept_without_numbering_gaps(self, tiny_corpus, monkeypatch):
        """短块必须**保留**，且编号仍然连续无空洞。

        这里断言的是与旧行为相反的事实：旧实现用 `len(child_text) > 5` 过滤，
        丢掉的不是重复内容而是**原文本身**（短块走"父子相同"分支，
        parent_text == child_text）。实测四本书合计丢 14 字，如红楼的
        `宝玉又道：`、西游的 `莫念！`。过滤判据现已改为"有没有正文"，
        只丢纯空白块 —— 那里面没有任何文字，丢了无损。

        "先过滤再编号、编号无空洞"这条不变量仍然成立，故继续断言。
        """
        pairs = [
            ("这是一段足够长的正文内容。", "这是一段足够长的正文内容。"),
            ("短", "短"),                                   # 是正文，必须保留
            ("第二段足够长的正文内容。", "第二段足够长的正文内容。"),
            ("哦。", "哦。"),                                # 是正文，必须保留
            ("第三段足够长的正文内容。", "第三段足够长的正文内容。"),
        ]
        _run_build_with_chunks(monkeypatch, pairs)
        RE.build_chunks()

        data = json.loads(tiny_corpus.read_text(encoding="utf-8"))
        assert len(data) == 5, f"短块是正文，不应被丢弃，实际剩 {len(data)} 条"

        idx = [c["chunk_index"] for c in data]
        assert idx == [0, 1, 2, 3, 4], f"chunk_index 应连续，实际 {idx}"

        totals = set(c["total_chunks"] for c in data)
        assert totals == {5}, f"total_chunks 应为 5，实际 {totals}"

        ids = [c["id"] for c in data]
        assert ids == [f"测试书_{i}" for i in range(5)], f"id 未按编号派生: {ids}"

        texts = [c["child_text"] for c in data]
        assert "短" in texts and "哦。" in texts, f"短块正文被丢了: {texts}"

    def test_whitespace_only_chunks_are_dropped(self, tiny_corpus, monkeypatch):
        """纯空白块（不含任何文字）仍应被丢弃，且不留下编号空洞。"""
        _run_build_with_chunks(monkeypatch, [("   ", "   "), ("\n", "\n")])
        RE.build_chunks()
        data = json.loads(tiny_corpus.read_text(encoding="utf-8"))
        assert data == [], f"纯空白块不含文字，应全部丢弃，实际 {data}"

    def test_fields_written(self, tiny_corpus, monkeypatch):
        _run_build_with_chunks(monkeypatch, [("一段足够长的正文内容。", "父块足够长的正文内容。")])
        RE.build_chunks()
        data = json.loads(tiny_corpus.read_text(encoding="utf-8"))
        assert len(data) == 1
        rec = data[0]
        assert rec["book"] == "测试书"
        assert rec["child_text"] == "一段足够长的正文内容。"
        assert rec["parent_text"] == "父块足够长的正文内容。"
        # 刻意退化：contextual_text 恒等于 child_text
        assert rec["contextual_text"] == rec["child_text"]


class TestBuildChunksErrors:
    def test_missing_books_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(RE, "BOOKS_DIR", str(tmp_path / "不存在"))
        with pytest.raises(FileNotFoundError, match="找不到目录"):
            RE.build_chunks()

    def test_empty_books_dir(self, tmp_path, monkeypatch):
        empty = tmp_path / "books"
        empty.mkdir()
        monkeypatch.setattr(RE, "BOOKS_DIR", str(empty))
        with pytest.raises(FileNotFoundError, match=r"没有 \.txt"):
            RE.build_chunks()
