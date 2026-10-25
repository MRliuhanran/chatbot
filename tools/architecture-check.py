#!/usr/bin/env python3
"""单文件分层架构检查（stdlib only）：把设计约定变成可执行断言。

为什么不是单元测试：项目运行期不装 pytest，而这里检查的是"架构不变量"——
章节顺序、顶层无 streamlit、检索路径不按后端分叉、适配器齐全、指纹未漂移。
每次重构后跑一次；失败即退出码 1。

用法:
    python tools/architecture-check.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "chatbot.py"
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)

    # 1) 章节横幅顺序：基础设施 -> 适配器 -> 检索引擎 -> 单轮编排 -> Streamlit UI
    markers = [
        "── 基础设施",
        "── 适配器层",
        "── 检索引擎",
        "── 单轮编排",
        "── 交互入口：Streamlit",
    ]
    positions = [next((i for i, line in enumerate(lines) if marker in line), -1) for marker in markers]
    check("章节顺序完整", all(p >= 0 for p in positions) and positions == sorted(positions),
          f"positions={positions}")

    # 2) 顶层不得 import streamlit（只能在 UI/函数内惰性导入）
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    bad_imports = []
    for node in top_imports:
        if isinstance(node, ast.ImportFrom) and node.module == "streamlit":
            bad_imports.append(node.lineno)
        if isinstance(node, ast.Import) and any(a.name == "streamlit" for a in node.names):
            bad_imports.append(node.lineno)
    check("顶层不 import streamlit", not bad_imports, f"lines={bad_imports}")

    # 3) 检索主路径不得按后端分叉（适配器化后 hybrid_search 只应有一条编排路径）
    func = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "hybrid_search"), None)
    fsrc = ast.get_source_segment(text, func) if func else ""
    check("hybrid_search 单一编排路径",
          bool(func) and "EMBED_BACKEND" not in fsrc and "RERANK_BACKEND" not in fsrc)

    # 4) 适配器类齐全 + 引擎构造注入
    classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    required = {
        "EmbeddingBackend", "RerankBackend",
        "TorchEmbeddingBackend", "HttpEmbeddingBackend",
        "TorchRerankBackend", "HttpRerankBackend",
        "RuntimeProfile", "RAGEngine",
    }
    check("适配器/值对象类齐全", required <= classes, f"missing={sorted(required - classes)}")
    engine = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RAGEngine"), None)
    init = next((n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None) if engine else None
    init_src = ast.get_source_segment(text, init) if init else ""
    check("引擎构造注入适配器",
          "_embedding_backend" in init_src and "_rerank_backend" in init_src)

    # 5) 指纹未漂移：切分代码指纹必须与已提交产物 meta 一致（改了指纹区 = 需要全量重建）
    sys.path.insert(0, str(ROOT))
    import chatbot  # noqa: E402  （导入会加载 .env/配置；这是检查所需）

    meta_path = ROOT / "artifacts" / "chunks.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = (meta.get("chunking_config") or {}).get("chunking_code")
        actual = chatbot._chunking_code_fingerprint()
        check("切分指纹未漂移", expected == actual, f"meta={expected} current={actual}")
    else:
        check("切分指纹可校验", False, "缺少 artifacts/chunks.meta.json")

    width = max(len(name) for name, _, _ in CHECKS)
    failed = 0
    for name, passed, detail in CHECKS:
        mark = "✅" if passed else "❌"
        suffix = f"  {detail}" if detail and not passed else ""
        print(f"{mark} {name:<{width}}{suffix}")
        failed += 0 if passed else 1
    print(f"\n架构检查: {len(CHECKS) - failed}/{len(CHECKS)} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
