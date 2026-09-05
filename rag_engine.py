#!/usr/bin/env python3
"""
RAG 检索引擎 —— 与 UI 框架解耦的纯函数模块。

供 app.py（Streamlit UI / CLI）与 test_retrieval_quality.py 复用，
避免检索逻辑在多处复制粘贴导致漂移。

设计要点：
  - BGE v1.5 官方用法：CLS pooling + query 侧 instruction
  - 模型半精度(fp16)加载，适配 8GB 统一内存的 Apple Silicon
  - Reranker 分批推理 + 限制 max_length，避免大 batch 触发显存换页
  - 向量/BM25 全部用 chunk id 对齐，不再依赖 ChromaDB 返回顺序
"""

import os
import json
import pickle
import hashlib
import warnings

warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than the specified maximum",
)

# ============================================================================
# 配置
# ============================================================================
BOOKS_DIR = "books"

TOP_K = 5                # 最终返回给 LLM 的结果数
RERANK_TOP_K = 30        # 初筛候选数 / 重排序输入数
RERANK_BATCH = 16        # reranker 单批条数（降低峰值显存）
RERANK_MAX_LENGTH = 256  # reranker 最大序列长度
RRF_K = 60               # RRF 融合常数

CHILD_MAX_TOKENS = 128
PARENT_MAX_TOKENS = 512
CHUNK_OVERLAP = 32

DB_PATH = "./chroma_db_v2"
EMBED_MODEL_PATH = "./models/bge-small-zh-v1.5"
RERANK_MODEL_PATH = "./models/bge-reranker-base"

BM25_CACHE_DIR = "./cache_v2"
BM25_CACHE_FILE = os.path.join(BM25_CACHE_DIR, "bm25_index.pkl")
BM25_TOKENS_CACHE_FILE = os.path.join(BM25_CACHE_DIR, "bm25_tokens.pkl")
BM25_IDS_CACHE_FILE = os.path.join(BM25_CACHE_DIR, "bm25_ids.pkl")
CHUNKS_JSON = os.path.join(BM25_CACHE_DIR, "chunks.json")

# BGE v1.5 检索 instruction（仅 query 侧追加，doc 侧不加）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 人物名词表（外置于此，便于扩展；识别到的名字会进入上下文前缀）
NAME_PATTERNS = [
    "刘备", "关羽", "张飞", "诸葛亮", "赵云", "曹操", "孙权", "周瑜",
    "林黛玉", "贾宝玉", "薛宝钗", "王熙凤",
    "孙悟空", "唐僧", "猪八戒", "沙僧",
    "宋江", "武松", "林冲", "鲁智深", "李逵",
]


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _float16_ok(device):
    """fp16 仅在 GPU/MPS 上使用，CPU 回退 fp32。"""
    return device in ("mps", "cuda")


# ============================================================================
# 阶段一：分块
# ============================================================================
def _hierarchical_split(text, tokenizer):
    """分层分块：父 chunk(512 token) 内切子 chunk(128 token) + 重叠。"""
    def _split_sentences(s):
        sentences = []
        current = ""
        for ch in s:
            current += ch
            if ch in "。！？" and current.strip():
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        return sentences

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # 父 chunk
    parent_chunks = []
    cur = []
    cur_tokens = 0
    for sent in sentences:
        n = len(tokenizer.encode(sent, add_special_tokens=False))
        if cur_tokens + n > PARENT_MAX_TOKENS and cur:
            parent_chunks.append(" ".join(cur))
            cur = [sent]
            cur_tokens = n
        else:
            cur.append(sent)
            cur_tokens += n
    if cur:
        parent_chunks.append(" ".join(cur))

    # 子 chunk
    results = []
    for parent_text in parent_chunks:
        if len(tokenizer.encode(parent_text, add_special_tokens=False)) <= CHILD_MAX_TOKENS:
            results.append((parent_text, parent_text))
            continue

        parent_sents = _split_sentences(parent_text)
        child_cur = []
        child_tokens = 0
        for s in parent_sents:
            n = len(tokenizer.encode(s, add_special_tokens=False))
            if child_tokens + n > CHILD_MAX_TOKENS and child_cur:
                results.append((" ".join(child_cur), parent_text))
                # 重叠：从已结束的子块尾部回填 CHUNK_OVERLAP token
                overlap, overlap_tokens = [], 0
                for s2 in reversed(child_cur):
                    ot = len(tokenizer.encode(s2, add_special_tokens=False))
                    if overlap_tokens + ot > CHUNK_OVERLAP:
                        break
                    overlap.insert(0, s2)
                    overlap_tokens += ot
                child_cur = overlap + [s]
                child_tokens = overlap_tokens + n
            else:
                child_cur.append(s)
                child_tokens += n
        if child_cur:
            results.append((" ".join(child_cur), parent_text))

    return results


def build_chunks():
    """读取 books/*.txt → 清洗 → 分块 → 写 chunks.json。返回 chunk 列表。"""
    import re
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)

    if not os.path.isdir(BOOKS_DIR):
        raise FileNotFoundError(f"找不到目录: {BOOKS_DIR}")
    txt_files = sorted(f for f in os.listdir(BOOKS_DIR) if f.endswith(".txt"))
    if not txt_files:
        raise FileNotFoundError(f"{BOOKS_DIR} 中没有 .txt 文件")

    print(f"找到 {len(txt_files)} 本书: {', '.join(txt_files)}")

    all_chunks = []
    for i, filename in enumerate(txt_files):
        book_name = filename[:-4]
        filepath = os.path.join(BOOKS_DIR, filename)
        print(f"[{i + 1}/{len(txt_files)}] 处理: {book_name} ...")

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = _hierarchical_split(text, tokenizer)

        for chunk_idx, (child_text, parent_text) in enumerate(chunks):
            if len(child_text) <= 5:
                continue

            chapter_match = re.search(r"第[一二三四五六七八九十百千\d]+[回章节]", child_text)
            chapter = chapter_match.group(0) if chapter_match else ""

            names = [n for n in NAME_PATTERNS if n in child_text]

            prefix_parts = [f"《{book_name}》"]
            if chapter:
                prefix_parts.append(chapter)
            if names:
                prefix_parts.append(f"涉及: {', '.join(names[:3])}")
            contextual_prefix = " > ".join(prefix_parts) + "\n\n"

            all_chunks.append({
                "child_text": child_text,
                "parent_text": parent_text,
                "book": book_name,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "contextual_text": contextual_prefix + child_text,
                "id": f"{book_name}_{chunk_idx}",
            })

    os.makedirs(BM25_CACHE_DIR, exist_ok=True)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print("分块统计:")
    for b, cnt in sorted(Counter(c["book"] for c in all_chunks).items()):
        print(f"  {b}: {cnt} 条")
    print(f"  总计: {len(all_chunks)} 条")
    print(f"已保存: {CHUNKS_JSON}")
    return all_chunks


# ============================================================================
# 模型加载与 embedding
# ============================================================================
def load_embedding_model(device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = device or get_device()
    # bge-small 仅 ~92MB，用 fp32：避免 MPS 上半精度转换 + 逐批同步的开销（实测 fp16 批量编码反而更慢）
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
    model = AutoModel.from_pretrained(EMBED_MODEL_PATH).to(device)
    model.eval()
    return tokenizer, model, device


def load_reranker_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = get_device()
    dtype = torch.float16 if _float16_ok(device) else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL_PATH, torch_dtype=dtype
    ).to(device)
    model.eval()
    return tokenizer, model, device


def embed(texts, tokenizer, model, device, is_query=False):
    """BGE 标准 embedding：CLS pooling + L2 归一化。query 侧追加 instruction。"""
    import torch

    if is_query:
        texts = [BGE_QUERY_INSTRUCTION + t for t in texts]

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS pooling（bge 官方 pooling_mode_cls_token=true），fp32 下归一化保证数值稳定
    cls = outputs.last_hidden_state[:, 0].float()
    cls = torch.nn.functional.normalize(cls, p=2, dim=1)
    return cls.cpu().numpy()


# ============================================================================
# 阶段二：索引
# ============================================================================
def build_index():
    """chunks.json → 向量化 → ChromaDB + BM25（含 id 对齐清单）。"""
    import time
    import numpy as np
    import jieba
    import chromadb
    from rank_bm25 import BM25Okapi

    if not os.path.exists(CHUNKS_JSON):
        raise FileNotFoundError(f"找不到 {CHUNKS_JSON}，请先运行: python app.py process")

    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    if not all_chunks:
        raise ValueError("chunks.json 为空")

    print(f"加载分块数据: {len(all_chunks)} 条")

    # 索引构建用 CPU：MPS 持续满负荷会触发 GPU 降频（实测 67 条/秒 → 9 条/秒），
    # 且 GPU 与系统 GUI(WindowServer/Chrome) 共享；CPU 8 线程稳定 ~35 条/秒。
    # 查询阶段（单条/小批）仍走 MPS 突发推理，不受影响。
    import torch as _torch
    if get_device() == "mps":
        _torch.set_num_threads(8)
    tokenizer, model, device = load_embedding_model(device="cpu")
    print(f"Embedding 设备: {device}")

    batch_size = 128
    total = len(all_chunks)
    all_embeddings = []
    t0 = time.time()

    # CPU 构建无需间歇冷却（冷却仅针对 MPS 降频场景保留配置，CPU 上 sleep 极短）
    COOL_DOWN_SLEEP = 0.05
    print(f"批量生成 Embedding (batch_size={batch_size}, CPU 8线程)...")
    for i in range(0, total, batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["contextual_text"] for c in batch]
        all_embeddings.append(embed(texts, tokenizer, model, device, is_query=False))
        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        if (i // batch_size) % 5 == 0 or done >= total:
            print(f"  {done / total * 100:5.1f}% ({done}/{total}) {done / elapsed:.0f} 条/秒")
        time.sleep(COOL_DOWN_SLEEP)

    embeddings = np.vstack(all_embeddings).astype("float32")

    # 写 ChromaDB
    print("写入 ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("books_v2")
    except Exception:
        pass
    collection = client.create_collection("books_v2", metadata={"hnsw:space": "cosine"})

    write_batch = 1000
    for i in range(0, total, write_batch):
        batch = all_chunks[i:i + write_batch]
        collection.add(
            documents=[c["child_text"] for c in batch],
            ids=[c["id"] for c in batch],
            metadatas=[{
                "book": c["book"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
                "parent_text": c["parent_text"],
                "contextual_text": c["contextual_text"],
            } for c in batch],
            embeddings=embeddings[i:i + write_batch].tolist(),
        )
        print(f"  写入: {min(i + write_batch, total)}/{total}")
    print(f"ChromaDB 完成: {collection.count()} 条")

    # 释放 embedding 模型内存，再构建 BM25
    del model, tokenizer, embeddings
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 构建 BM25（同时保存 id 对齐清单，彻底消除顺序依赖）
    print("构建 BM25 缓存...")
    jieba.setLogLevel(jieba.logging.INFO)
    tokenized_corpus = [list(jieba.cut(c["child_text"])) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    ids = [c["id"] for c in all_chunks]

    os.makedirs(BM25_CACHE_DIR, exist_ok=True)
    with open(os.path.join(BM25_CACHE_DIR, "hash.txt"), "w") as f:
        f.write(_get_books_hash())
    with open(BM25_TOKENS_CACHE_FILE, "wb") as f:
        pickle.dump(tokenized_corpus, f)
    with open(BM25_CACHE_FILE, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_IDS_CACHE_FILE, "wb") as f:
        pickle.dump(ids, f)

    print("BM25 缓存已保存")
    print(f"索引完成: ChromaDB {collection.count()} 条, BM25 {len(tokenized_corpus)} 条")


def _get_books_hash():
    if not os.path.isdir(BOOKS_DIR):
        return ""
    files = sorted(f for f in os.listdir(BOOKS_DIR) if f.endswith(".txt"))
    content = ""
    for fn in files:
        with open(os.path.join(BOOKS_DIR, fn), "rb") as fh:
            content += hashlib.md5(fh.read()).hexdigest()
    return hashlib.md5(content.encode()).hexdigest()


# ============================================================================
# 检索
# ============================================================================
def _rrf_score(rank, k=RRF_K):
    return 1.0 / (k + rank + 1)


def rrf_fusion(vec_ranked, bm25_ranked):
    """输入 [(id, score)...] 已按分数降序，输出按 RRF 分数降序的 id 列表。"""
    scores = {}
    for rank, (cid, _score) in enumerate(vec_ranked):
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank)
    for rank, (cid, _score) in enumerate(bm25_ranked):
        scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank)
    return sorted(scores, key=lambda c: -scores[c])


def rerank_indices(query, documents, tokenizer, model, device,
                   top_k=TOP_K, batch_size=RERANK_BATCH, max_length=RERANK_MAX_LENGTH):
    """分批 rerank，返回 [(原索引, 分数)...] 按分数降序，取 top_k。"""
    import torch
    import numpy as np

    if not documents:
        return []
    all_scores = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        pairs = [(query, d) for d in batch]
        inputs = tokenizer(pairs, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
        all_scores.extend(logits.float().cpu().numpy().tolist())

    order = np.argsort(all_scores)[::-1][:top_k]
    return [(int(i), float(all_scores[i])) for i in order]


class RAGEngine:
    """持有模型/数据资源的检索引擎，资源懒加载。"""

    def __init__(self):
        self._embed = None
        self._rerank = None
        self._chunks = None
        self._bm25 = None
        self._collection = None

    # ---- 资源加载 ----
    def _get_collection(self):
        import chromadb
        if self._collection is None:
            client = chromadb.PersistentClient(path=DB_PATH)
            try:
                self._collection = client.get_collection("books_v2")
            except Exception:
                self._collection = client.create_collection(
                    "books_v2", metadata={"hnsw:space": "cosine"})
        return self._collection

    def _get_embed(self):
        if self._embed is None:
            self._embed = load_embedding_model()
        return self._embed

    def _get_rerank(self):
        if self._rerank is None:
            self._rerank = load_reranker_model()
        return self._rerank

    def _get_chunks(self):
        """返回 {chunk_id: {...}}，一次全量加载后缓存。"""
        if self._chunks is None:
            col = self._get_collection()
            res = col.get(limit=200000)
            chunks = {}
            for i in range(len(res["ids"])):
                m = res["metadatas"][i]
                chunks[res["ids"][i]] = {
                    "id": res["ids"][i],
                    "child_text": res["documents"][i],
                    "parent_text": m.get("parent_text") or res["documents"][i],
                    "book": m.get("book", ""),
                    "contextual_text": m.get("contextual_text") or res["documents"][i],
                }
            self._chunks = chunks
        return self._chunks

    def _get_bm25(self):
        """返回 (bm25, ids) 或 (None, None)。含 hash 校验。"""
        if self._bm25 is None:
            try:
                if not (os.path.exists(BM25_CACHE_FILE)
                        and os.path.exists(BM25_IDS_CACHE_FILE)):
                    self._bm25 = (None, None)
                    return self._bm25
                hash_file = os.path.join(BM25_CACHE_DIR, "hash.txt")
                if not os.path.exists(hash_file):
                    self._bm25 = (None, None)
                    return self._bm25
                if open(hash_file).read().strip() != _get_books_hash():
                    self._bm25 = (None, None)
                    return self._bm25
                with open(BM25_CACHE_FILE, "rb") as f:
                    bm25 = pickle.load(f)
                with open(BM25_IDS_CACHE_FILE, "rb") as f:
                    ids = pickle.load(f)
                self._bm25 = (bm25, ids)
            except Exception:
                self._bm25 = (None, None)
        return self._bm25

    # ---- 检索主链路 ----
    def hybrid_search(self, query, top_k=TOP_K, expand_queries=None):
        """
        混合检索：向量 + BM25 → RRF → rerank。
        expand_queries: 可选，已扩展的查询列表（未提供则只用原始 query）。
        """
        import numpy as np
        import jieba

        chunks = self._get_chunks()
        if not chunks:
            return []

        col = self._get_collection()
        tokenizer, model, device = self._get_embed()

        queries = expand_queries if expand_queries else [query]

        # 1) 向量检索（多查询取分数最高）
        vec_scores = {}
        for q in queries:
            q_emb = embed([q], tokenizer, model, device, is_query=True)
            res = col.query(query_embeddings=q_emb.tolist(), n_results=RERANK_TOP_K)
            for cid, dist in zip(res["ids"][0], res["distances"][0]):
                score = 1.0 - dist
                if cid not in vec_scores or score > vec_scores[cid]:
                    vec_scores[cid] = score
        vec_ranked = sorted(vec_scores.items(), key=lambda x: -x[1])

        # 2) BM25（用 id 清单对齐）
        bm25_ranked = []
        bm25, bm25_ids = self._get_bm25()
        if bm25 is not None:
            tokens = list(jieba.cut(query))
            scores = bm25.get_scores(tokens)
            top_idx = np.argsort(scores)[::-1][:RERANK_TOP_K]
            bm25_ranked = [(bm25_ids[i], float(scores[i]))
                           for i in top_idx if scores[i] > 0]

        # 3) RRF 融合
        fused_ids = rrf_fusion(vec_ranked, bm25_ranked)[:RERANK_TOP_K]

        # 4) 重排序
        candidates = [chunks[cid] for cid in fused_ids if cid in chunks]
        if not candidates:
            return []

        rr_tok, rr_model, rr_dev = self._get_rerank()
        cand_texts = [c["contextual_text"] for c in candidates]
        reranked = rerank_indices(query, cand_texts, rr_tok, rr_model, rr_dev, top_k=top_k)

        output = []
        for idx, score in reranked:
            c = candidates[idx]
            output.append({
                "id": c["id"],
                "child_text": c["child_text"],
                "parent_text": c["parent_text"],
                "book": c["book"],
                "contextual_text": c["contextual_text"],
                "rerank_score": score,
            })
        return output
