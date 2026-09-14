#!/usr/bin/env python3
"""
一键健康检查 —— 确认 RAG 服务各环节是否可用。

检查项（每项互不依赖，单项失败不中断）：
  1. 配置常量与模型/缓存目录
  2. books 源文本齐全
  3. Qdrant 集合（rag_engine.COLLECTION_NAME）非空 + 元数据完整
  4. 混合检索：集合同时具备稠密/稀疏空间、点上两种向量都已写入、稀疏单通道可召回
  5. Ollama 后端可达（/api/tags）
  6. 生成模型可用（一次简短 think:false 问答）

用法:
  python check_health.py            # 全部检查
  python check_health.py --offline  # 跳过 Ollama/模型在线检查（快速）
退出码: 0=全部通过 1=存在失败
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_engine as RE  # noqa: E402
import app as A            # noqa: E402  (提供 MODEL / OLLAMA_BASE_URL)

from qdrant_client import QdrantClient  # noqa: E402


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

    print("\n[3] 向量库 Qdrant")
    qdrant_count = 0
    client = None
    try:
        # 连接Docker服务
        client = QdrantClient(host=RE.QDRANT_HOST, port=RE.QDRANT_PORT)
        qdrant_count = client.count(collection_name=RE.COLLECTION_NAME).count
        if qdrant_count > 0:
            ok(f"{RE.COLLECTION_NAME} 记录数: {qdrant_count}")
            # 获取一条样本数据检查元数据
            sample = client.scroll(
                collection_name=RE.COLLECTION_NAME,
                limit=1,
                with_payload=True,
                with_vectors=False
            )[0]
            if sample:
                meta = sample[0].payload
                missing = [k for k in ["book", "chunk_index", "parent_text", "contextual_text"]
                           if k not in meta]
                if missing:
                    fail += bad(f"元数据缺字段: {missing}")
                else:
                    ok("元数据字段完整")
        else:
            fail += bad(f"{RE.COLLECTION_NAME} 为空，请运行: python app.py index")
    except Exception as e:
        fail += bad(f"无法连接 Qdrant Docker服务 ({RE.QDRANT_HOST}:{RE.QDRANT_PORT}): {e}")

    print("\n[4] 混合检索（稠密 + 稀疏）")
    # 旧版这里只做了 hasattr(config) 判断，恒为真，永远打印"已启用"，
    # 因此"稀疏向量其实一条都没写"这个事实被掩盖了很久。改为真检查：
    # 集合配置 + 点上实际向量 + 稀疏单通道能否召回。
    if client is None or qdrant_count == 0:
        print("  ⏭  跳过（向量库不可用）")
    else:
        try:
            info = client.get_collection(collection_name=RE.COLLECTION_NAME)
            sparse_names = list(info.config.params.sparse_vectors.keys())
            dense_names = list(info.config.params.vectors.keys())
            if sparse_names and dense_names:
                ok(f"集合配置: 稠密 {dense_names} + 稀疏 {sparse_names}")
            else:
                fail += bad(f"集合缺少向量空间: 稠密 {dense_names} / 稀疏 {sparse_names}")

            pts, _ = client.scroll(
                collection_name=RE.COLLECTION_NAME,
                limit=50,
                with_payload=True,
                with_vectors=True,
            )
            miss_dense = sum(1 for p in pts if not p.vector.get(RE.DENSE_VECTOR_NAME))
            miss_sparse = sum(1 for p in pts if not p.vector.get(RE.SPARSE_VECTOR_NAME))
            if not miss_dense and not miss_sparse and pts:
                n_terms = len(pts[0].vector[RE.SPARSE_VECTOR_NAME].indices)
                ok(f"抽样 {len(pts)} 条: 稠密/稀疏均已写入（首条稀疏 {n_terms} 个 term）")
            else:
                fail += bad(
                    f"抽样 {len(pts)} 条中: 缺稠密 {miss_dense} 条, 缺稀疏 {miss_sparse} 条"
                    "（稀疏没写 = 混合检索名存实亡）"
                )

            # 功能验证：用样本原文做稀疏单通道检索，必须能召回自己
            probe = pts[0].payload.get("child_text", "")[:80]
            if probe:
                res = client.query_points(
                    collection_name=RE.COLLECTION_NAME,
                    query=RE.sparse_encode(probe),
                    using=RE.SPARSE_VECTOR_NAME,
                    limit=3,
                )
                if res.points:
                    ok(f"稀疏单通道检索可用（{len(res.points)} 条命中，最高分 {res.points[0].score:.2f}）")
                else:
                    fail += bad("稀疏单通道检索返回空，稀疏索引可能未生效")
        except Exception as e:
            fail += bad(f"混合检索检查失败: {e}")

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
            if isinstance(data, dict) and data.get("error"):
                fail += bad(f"模型应答返回错误: {data['error']}")
            else:
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
