#!/usr/bin/env python3
"""
检索质量测试 —— 复用 rag_engine.RAGEngine，与生产链路完全一致。

用法:
  python test_retrieval_quality.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_engine as RE
from rag_engine import RAGEngine, get_device


# 测试用例: query -> 期望命中的书名 + 期望出现在结果中的关键词
TEST_CASES = [
    {"query": "谁是蜀汉的皇帝", "expected_book": "三国演义", "keywords": ["刘备", "玄德", "蜀汉", "昭烈"]},
    {"query": "林黛玉的性格特点", "expected_book": "红楼梦", "keywords": ["林黛玉", "黛玉"]},
    {"query": "孙悟空大闹天宫", "expected_book": "西游记", "keywords": ["孙悟空", "天宫", "大圣"]},
    {"query": "武松打虎", "expected_book": "水浒传", "keywords": ["武松", "虎", "景阳冈"]},
    {"query": "诸葛亮草船借箭", "expected_book": "三国演义", "keywords": ["诸葛亮", "孔明", "箭"]},
    {"query": "贾宝玉和林黛玉的关系", "expected_book": "红楼梦", "keywords": ["宝玉", "黛玉"]},
    {"query": "猪八戒的性格", "expected_book": "西游记", "keywords": ["猪八戒", "八戒"]},
    {"query": "鲁智深倒拔垂杨柳", "expected_book": "水浒传", "keywords": ["鲁智深", "柳"]},
]


def main():
    print("=" * 70)
    print("检索质量测试 (复用 rag_engine.RAGEngine)")
    print("=" * 70)

    # ---- 1. 索引完整性 ----
    print("\n[1] 索引完整性检查")
    import chromadb
    col = chromadb.PersistentClient(path=RE.DB_PATH).get_collection("books_v2")
    count = col.count()
    print(f"    books_v2 记录数: {count}")

    sample = col.get(limit=1)
    meta = sample["metadatas"][0]
    missing = [k for k in ["book", "chunk_index", "parent_text", "contextual_text"] if k not in meta]
    if missing:
        print(f"    ❌ 元数据缺失字段: {missing}")
        return 1
    print("    ✅ 元数据字段完整")

    hash_file = os.path.join(RE.BM25_CACHE_DIR, "hash.txt")
    cached = open(hash_file).read().strip() if os.path.exists(hash_file) else ""
    computed = RE._get_books_hash()
    if cached != computed:
        print(f"    ❌ hash.txt 不一致")
        return 1
    print("    ✅ BM25 hash 一致")

    # ---- 2. 引擎加载 ----
    print("\n[2] 加载 RAGEngine")
    engine = RAGEngine()
    chunks = engine._get_chunks()
    bm25, bm25_ids = engine._get_bm25()
    print(f"    全量 chunks: {len(chunks)} 条")
    print(f"    BM25 语料: {len(bm25_ids) if bm25 is not None else 0} 条")
    if bm25 is not None and len(bm25_ids) != len(chunks):
        print(f"    ❌ BM25({len(bm25_ids)}) 与 ChromaDB({len(chunks)}) 数量不一致")
        return 1
    print("    ✅ BM25 id 清单与 ChromaDB 数量一致")
    print(f"    设备: {get_device()}")

    # ---- 3. 逐用例检索 ----
    print("\n[3] 检索测试")
    passed = failed = 0
    detail = []

    for idx, tc in enumerate(TEST_CASES, 1):
        query = tc["query"]
        exp_book = tc["expected_book"]
        t0 = time.time()

        results = engine.hybrid_search(query, top_k=RE.TOP_K)
        elapsed = time.time() - t0

        ok = False
        top_book = None
        top_score = None
        kw_hit = []
        if results:
            top_book = results[0]["book"]
            top_score = results[0]["rerank_score"]
            books_hit = [r["book"] for r in results]
            merged = " ".join(r["parent_text"] for r in results[:3])
            kw_hit = [kw for kw in tc["keywords"] if kw in merged]
            ok = (exp_book in books_hit) and (len(kw_hit) >= 1)

        status = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"    [{idx}/{len(TEST_CASES)}] {status} {query}")
        print(f"          期望书={exp_book} | Top1书={top_book} | Top5书={[r['book'] for r in results]}")
        if results:
            print(f"          关键词命中={kw_hit} | rerank={top_score:.4f} | 耗时={elapsed:.2f}s")
            print(f"          Top1片段: {results[0]['child_text'][:60]}...")
        else:
            print(f"          ⚠️ 无检索结果 | 耗时={elapsed:.2f}s")
        detail.append({
            "query": query, "expected_book": exp_book, "status": status,
            "top_book": top_book, "top_score": top_score, "elapsed": round(elapsed, 3),
        })

    # ---- 4. 汇总 ----
    print("\n" + "=" * 70)
    print("检索质量汇总")
    print("=" * 70)
    total = len(TEST_CASES)
    rate = passed / total * 100 if total else 0
    print(f"通过: {passed}/{total}  失败: {failed}/{total}  通过率: {rate:.1f}%")
    if detail:
        times = [d["elapsed"] for d in detail]
        times.sort()
        n = len(times)
        print(f"耗时: min={times[0]:.2f}s 中位={times[n // 2]:.2f}s max={times[-1]:.2f}s avg={sum(times) / n:.2f}s")

    report = {"passed": passed, "failed": failed, "rate": rate, "detail": detail}
    with open("test_retrieval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("详细报告: test_retrieval_report.json")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
