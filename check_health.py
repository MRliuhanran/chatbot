#!/usr/bin/env python3
"""
一键健康检查 —— 确认 RAG 服务各环节是否可用。

检查项（每项互不依赖，单项失败不中断）：
  1. 配置常量与模型/缓存目录
  2. books 源文本齐全
  3. ChromaDB books_v2 非空 + 元数据完整
  4. BM25 缓存哈希一致 + 语料/ids 条数与 ChromaDB 一致
  5. Ollama 后端可达（/api/tags）
  6. 生成模型可用（一次简短 think:false 问答）

用法:
  python check_health.py            # 全部检查
  python check_health.py --offline  # 跳过 Ollama/模型在线检查（快速）
退出码: 0=全部通过 1=存在失败
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_engine as RE  # noqa: E402
import app as A            # noqa: E402  (提供 MODEL / OLLAMA_BASE_URL)

import chromadb  # noqa: E402


def ok(msg):
    print(f"  \u2705 {msg}")


def bad(msg):
    print(f"  \u274c {msg}")
    return 1


def main():
    parser = argparse.ArgumentParser(description="RAG 服务健康检查")
    parser.add_argument("--offline", action="store_true", help="跳过 Ollama / 模型在线检查")
    parser.add_argument("--timeout", type=int, default=30, help="Ollama 问答超时(秒)，默认 30")
    args = parser.parse_args()

    fail = 0

    print("=" * 60)
    print("RAG 健康检查")
    print("=" * 60)

    print("\n[1] 配置")
    if not RE.EMBED_MODEL_PATH or not os.path.isdir(RE.EMBED_MODEL_PATH):
        fail += bad(f"Embedding 模型目录缺失: {RE.EMBED_MODEL_PATH}")
    else:
        ok(f"Embedding 模型: {RE.EMBED_MODEL_PATH}")
    if not os.path.isdir(RE.RERANK_MODEL_PATH):
        fail += bad(f"Reranker 模型目录缺失: {RE.RERANK_MODEL_PATH}")
    else:
        ok(f"Reranker 模型: {RE.RERANK_MODEL_PATH}")
    ok(f"生成模型: {A.MODEL}  @  {A.OLLAMA_BASE_URL}")

    print("\n[2] 源文本")
    txt = [f for f in os.listdir(RE.BOOKS_DIR) if f.endswith(".txt")] if os.path.isdir(RE.BOOKS_DIR) else []
    if len(txt) >= 1:
        ok(f"books/ 含 {len(txt)} 本: {', '.join(txt)}")
    else:
        fail += bad("books/ 中没有 .txt")

    print("\n[3] 向量库 ChromaDB")
    chroma_count = 0
    try:
        col = chromadb.PersistentClient(path=RE.DB_PATH).get_collection("books_v2")
        chroma_count = col.count()
        if chroma_count > 0:
            ok(f"books_v2 记录数: {chroma_count}")
            sample = col.get(limit=1)
            meta = sample["metadatas"][0]
            missing = [k for k in ["book", "chunk_index", "parent_text", "contextual_text"]
                       if k not in meta]
            if missing:
                fail += bad(f"元数据缺字段: {missing}")
            else:
                ok("元数据字段完整")
        else:
            fail += bad("books_v2 为空，请运行: python app.py process && python app.py index")
    except Exception as e:
        fail += bad(f"无法打开 ChromaDB books_v2: {e}")

    print("\n[4] BM25 缓存")
    try:
        hash_file = os.path.join(RE.BM25_CACHE_DIR, "hash.txt")
        cached = open(hash_file).read().strip() if os.path.exists(hash_file) else ""
        computed = RE._get_books_hash()
        if cached == computed:
            ok("hash.txt 与书籍内容一致")
        else:
            fail += bad("hash.txt 过期，请重新运行: python app.py index")
        files_ok = all(os.path.exists(p) for p in [
            RE.BM25_TOKENS_CACHE_FILE, RE.BM25_CACHE_FILE, RE.BM25_IDS_CACHE_FILE])
        if files_ok:
            with open(RE.BM25_TOKENS_CACHE_FILE, "rb") as f:
                n_tokens = len(pickle.load(f))
            with open(RE.BM25_IDS_CACHE_FILE, "rb") as f:
                n_ids = len(pickle.load(f))
            ok(f"BM25 语料条数: {n_tokens}, ids: {n_ids}")
            if chroma_count and (n_tokens != chroma_count or n_ids != chroma_count):
                fail += bad(f"BM25({n_tokens}/{n_ids}) 与 ChromaDB({chroma_count}) 数量不一致，请重建索引")
            else:
                ok("BM25 与 ChromaDB 数量一致")
        else:
            fail += bad("BM25 缓存文件缺失")
    except Exception as e:
        fail += bad(f"BM25 检查异常: {e}")

    if args.offline:
        print("\n(offline 模式：跳过 Ollama 在线检查)")
    else:
        print("\n[5] Ollama 后端")
        import requests
        try:
            r = requests.get(f"{A.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                ok(f"Ollama 可达，已装模型: {models}")
                if A.MODEL not in models:
                    fail += bad(f"MODEL={A.MODEL} 未安装，请: ollama pull {A.MODEL}")
            else:
                fail += bad(f"Ollama /api/tags 返回 HTTP {r.status_code}")
        except Exception as e:
            fail += bad(f"Ollama 不可达: {e}")

        print("\n[6] 生成模型应答（think:false 简短问答）")
        import requests
        payload = {
            "model": A.MODEL,
            "messages": [{"role": "user", "content": "用一句话回答：孙悟空出自哪部小说？"}],
            "stream": False,
            "think": False,
            "options": {"num_predict": 64},
        }
        t0 = time.time()
        try:
            r = requests.post(f"{A.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=args.timeout)
            elapsed = time.time() - t0
            data = r.json()
            reply = (data.get("message", {}) or {}).get("content", "") if isinstance(data, dict) else ""
            ok(f"应答成功 耗时 {elapsed:.1f}s: {reply[:60]!r}")
        except Exception as e:
            fail += bad(f"模型应答失败/超时({args.timeout}s): {e}")

    print("\n" + "=" * 60)
    if fail:
        print(f"健康检查失败: {fail} 项")
        print("=" * 60)
        sys.exit(1)
    else:
        print("健康检查通过 ✓")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
