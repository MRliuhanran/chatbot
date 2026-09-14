#!/usr/bin/env python3
"""
RAG 检索引擎 —— 与 UI 框架解耦的纯函数模块。

**本文件是检索逻辑的唯一权威实现**，供 app.py（Streamlit UI / CLI）调用。
仓库里 `bak/chatbot.py` 是旧版 ChromaDB 单体实现，已归档、无人引用，不要参考。

设计要点：
  - 分块：引号归一 → sentencex 分句（单层，不预切段落）→ 超长句 token 兜底 → 父子分层
  - 语义分块用 bge-small 编码句子相似度（分块阶段设备默认 CPU，可用
    SEMANTIC_EMBED_DEVICE 覆盖；实测本机 MPS 吞吐更高，但 GPU 被其它进程
    占用时会阻塞，故默认选确定性）
  - BGE v1.5 官方用法：CLS pooling + query 侧 instruction（doc 侧不加）
  - 稠密 = bge-small（fp32；实测 fp16 在本机反而更慢）；
    reranker = bge-reranker-base（fp16，适配 Apple Silicon 统一内存）
  - 稀疏 = jieba 分词 + Qdrant 内置 BM25 打分（见 sparse_encode）
  - 向量/BM25 全部用 Qdrant point id 对齐；point id 用整数序号
    （Qdrant 只接受无符号整数或 UUID），原始 chunk id 存在 payload.chunk_id
"""

import os
import re
import json
import warnings

warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than the specified maximum",
)

# sentencex 是**硬依赖**（requirements.txt 已固定 sentencex>=1.0.30；MIT、零依赖、Wikimedia 维护）。
# 刻意不做 try/except 降级：内置的字符扫描分句会把闭合引号切给下一句（实测全库 24%~36%
# 的句子以 ” 开头），而且一声不响 —— 静默劣化比启动时报错难排查得多。缺库就让 import 失败。
from sentencex import segment

# ============================================================================
# 配置
# ============================================================================
BOOKS_DIR = "books"

TOP_K = 5                # 最终返回给 LLM 的结果数
RERANK_TOP_K = 20        # 初筛候选数 / 重排序输入数（8GB 机器：30 → 20 砍掉 1/3 峰值）
RERANK_BATCH = 16        # reranker 单批条数（降低峰值显存）
# 384 而非 256：父块 PARENT_MAX_TOKENS=512，256 会把父块砍掉一半，
# 等于抵消"父子分块"的意义。XLM-R 上限 514，384 是安全且不丢上下文的值。
RERANK_MAX_LENGTH = 384  # reranker 最大序列长度
# 融合由 Qdrant 服务端 FusionQuery(RRF) 完成，其 k 值不可在此配置，
# 故不再保留 RRF_K 这类从未生效的常量（旧代码里它一直是死配置）。

CHILD_MAX_TOKENS = 128
PARENT_MAX_TOKENS = 512
CHUNK_OVERLAP = 32

# 语义分块参数
# 相邻句余弦相似度低于 SEMANTIC_THRESHOLD 视为语义边界。
# SEMANTIC_MIN_CHARS 是切分下限：块长度未达该值前不允许切分。
#   取值 500 的依据（水浒传实测，token 中位数）：
#     120 → 父块 151 字，37% 的块父子相同（等于没有父上下文）
#     400 → 父块 388 token
#     500 → 父块 473 token ≈ PARENT_MAX_TOKENS(512)，且 0% 父块被机械切分
#   语义边界因此作用在"父块尺度"上：在每 ~500 字窗口内挑语义最弱处切分，
#   而不是在固定 512 token 处硬切。
SEMANTIC_THRESHOLD = 0.6
SEMANTIC_MIN_CHARS = 500
EMBED_BATCH_SIZE = 64
# 分块阶段句子编码所用设备，可用环境变量覆盖（"mps"/"cpu"）。
# 默认 cpu：MPS 在本机与 CPU 同速，且会因 GPU 争用阻塞（详见 _get_embed_model）。
SEMANTIC_EMBED_DEVICE = os.getenv("SEMANTIC_EMBED_DEVICE", "cpu")
# 建索引阶段（build_index）编码所用设备。空 = 自动（有 MPS 就优先 MPS）。
# 设 INDEX_DEVICE=cpu 可强制 CPU：8GB 机器上 MPS 常因统一内存被挤压而初始化失败
# （见 build_index 里对 "invalid low watermark ratio" 的回退），显式指定可免去试探。
INDEX_DEVICE = os.getenv("INDEX_DEVICE", "")

# Qdrant Docker配置
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "books_v3"
# 换 bge-small(512维) → bge-base(768维)：
#   C-MTEB Retrieval 61.77 → 69.49 (+7.72)，而 large 只再 +0.97 却要 3.2 倍参数。
#   维度变了，旧集合 books_v2 无法增量迁移，故新建 books_v3，回滚零成本。
EMBED_MODEL_PATH = "./models/bge-base-zh-v1.5"
RERANK_MODEL_PATH = "./models/bge-reranker-base"

CHUNKS_JSON = "./cache_v2/chunks.json"
BM25_CACHE_DIR = "./cache_v2"

# 索引建造标识：build_index 每次生成一个 uuid 写进每个 point 的 payload。
# 检索端靠它判断"磁盘上的索引已被重建"，从而自动丢弃进程内的分块缓存 ——
# 否则重建索引后运行中的服务会一直用旧数据（point id 还会错位，读到张冠李戴的正文）。
# 不按点数比对：重建后点数完全可能相同。不写本地文件：payload 是唯一权威来源。
INDEX_BUILD_ID_FIELD = "build_id"

# BGE v1.5 检索 instruction（仅 query 侧追加，doc 侧不加）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# Qdrant 命名向量空间。稠密/稀疏是同一字段的两种表示，
# 文档侧与查询侧必须成对同构，否则不是混合检索而是向量空间错配。
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# 稀疏侧沿用 Qdrant 内置 BM25 方案（服务端 Document 推理 + Modifier.IDF），
# 只把分词这一步换成 jieba：先用 jieba 切词、再用空格连接后交给 Qdrant，
# 它的 word 分词器此时只按我们插入的空格切，等于用 jieba 的分词结果。
#   - word 分词器：保持"按空白切"的默认行为（正是我们要的）
#   - stemmer/stopwords 关闭：Qdrant BM25 默认做英文词干化与英文停用词过滤，
#     对中文语料是纯干扰（实测 "the The THE" 会被整体丢弃）
# 注意：文档侧与查询侧必须使用完全相同的 options，否则召回自毁。
BM25_TEXT_OPTIONS = {
    "tokenizer": "word",
    "stemmer": {"type": "none"},
    "stopwords": {},
    "ascii_folding": True,
}
BM25_MODEL_NAME = "qdrant/bm25"
# jieba 分词后需要丢弃的纯标点 token（word 分词器不认这些，留着只会污染）
_PUNCT_ONLY = set("，。！？、；：\u201c\u201d\u2018\u2019《》…—（）()[]{}<>!?,.;:'\"-·　 \n\t")


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
#
# 分句为什么要用库而不是字符扫描（全量四本书实测，bench_sentence_split.py 可复现）：
#   字符扫描 "。！？．" 有三类系统性缺陷：
#     1. 闭合引号被切给下一句 —— 原文 `……必获恶报。”角拜问姓名。`，
#        切点在 。，于是 ” 落在下一句开头。全库共 26813 句以 ” 开头
#        （西游 10935、三国 9031、红楼 4175、水浒 2672），占全部句子的 24%~36%。
#     2. 段落边界被忽略 —— \n\n 才是真结构（水浒 4195 段），
#        段落末尾常是结构信号（`诗曰：`、`第一回 灵根育孕源流出…`）。
#        现行规则下水浒 50% 的段落末尾没有发生切分。
#     3. 半角 ?!. 与省略号 … 不认 —— 三国 931 个半角 ?，四个文件合计 100+ 处 …。
#
#   sentencex（MIT、零依赖、Wikimedia 维护）的设计假设恰好适配小说：
#   引语是不可切的原子，引号内部不切分。实测引号错位 24%~36% → 0%。
#   对比 pysbd：它只修好了段落边界，引号行为与字符扫描逐字相同，
#   且会在三国乱码段丢字（3 段净丢 32 字符），慢 60~150 倍 —— 故不采用。
# ============================================================================
# 句末终止符。含 ．(U+FF0E 全角句点)：红楼梦以 ． 为主要句读（10043 次，而 。 仅 4550 次）。
# 含半角 ?!. 与省略号 …：三国 931 个半角 ?，四书合计 100+ 处 …。
SENTENCE_TERMINATORS = "。！？．…!?"
# 注：原先还有一个 CLOSING_CHARS（切分点后要吞掉的闭合引号集），随字符扫描分句器
# _split_sentences_rule 一起删除 —— sentencex 把引语当原子，不需要这层手工吞并。

# 引号归一开关。红楼梦原文混用开引号 “ (5802) 与直引号 " (2426)，
# 弯引号 ” 只有 4202 个 —— 引号深度算错会让"引语不可切分"的分句器把大段对话
# 吞进同一句（实测最长句 964 字 → 归一后 705 字，引号不配平 12.3% → 3.5%）。
# 这是对原文的有意修正：chunk 文本里位置恰当的 " 会被改写成 “/”。
NORMALIZE_QUOTES = os.getenv("NORMALIZE_QUOTES", "1") != "0"
# 直引号前紧挨非空白 → 按闭合引号处理；其余 → 按开引号处理。
_STRAIGHT_CLOSE_RE = re.compile(r'(?<=[^\s])"')


def fix_quotes(text):
    """把用错的直引号 " 按位置归一到弯引号（红楼梦专治）。"""
    text = _STRAIGHT_CLOSE_RE.sub("”", text)
    return text.replace('"', "“")


def _split_keep(text, terminators):
    """在 terminators 中每个字符之后切开，终止符保留在左侧。"""
    parts, cur = [], ""
    for ch in text:
        cur += ch
        if ch in terminators:
            parts.append(cur)
            cur = ""
    if cur:
        parts.append(cur)
    return [p for p in parts if p.strip()]


def _split_sentences(s, tokenizer=None, max_tokens=None):
    """分句：sentencex 单层分句 → （可选）token 上限兜底。

    **不再预切 \n\n 段落**：sentencex 自身把 \n\n 与 \r\n\r\n 都当句边界，且从不跨段。
    实测四本书（66079 句）"先按段落预切再逐段分句"与"整本书一次分句"的输出序列
    逐元素完全相同，跨段句（含 \n 的句子）恒为 0 —— 那一层是纯冗余，故删除。
    "句子不跨段"这条性质改由下面的断言守护：它现在是 sentencex 的实现保证，
    而不是本函数的代码保证，所以必须能被测试发现回归。

    tokenizer/max_tokens 给出时，逐句保证不超过 max_tokens —— 语料里有
    整段无标点的文言（西游记"故曰混沌"一段，单句最长 673 字、红楼 705 字），
    纯分句器对它们无能为力，不兜底就会撑破 CHILD_MAX_TOKENS。
    """
    parts = [p.strip() for p in segment("zh", s) if p.strip()]
    # 段落硬边界的守护断言。sentencex 保证引语不可切分、不跨段；一旦这里失败，
    # 说明分句器行为变了（换版本/换库），必须重新评估分块，不能静默继续。
    # 注意只在 token 兜底之前断言：兜底会按 "\n" 等终止符切开长句，可能产生带换行的碎片。
    bad = next((p for p in parts if "\n" in p or "\r" in p), None)
    if bad is not None:
        raise ValueError(
            f"分句器跨段了，段落硬边界保证已失效（sentencex 行为可能已变）：{bad[:60]!r}"
        )
    if tokenizer is not None and max_tokens:
        limited = []
        for p in parts:
            limited.extend(_enforce_token_limit(p, tokenizer, max_tokens))
        parts = limited
    return parts


def _hard_split_by_tokens(text, tokenizer, max_tokens):
    """最后兜底：无任何标点可用时按 token 硬切。"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [
        tokenizer.decode(ids[i:i + max_tokens], skip_special_tokens=True)
        for i in range(0, len(ids), max_tokens)
    ]


def _enforce_token_limit(text, tokenizer, max_tokens, depth=0):
    """保证返回的每个片段都不超过 max_tokens。

    逐级降级：句末标点 → 换行 → 分号 → 逗号/顿号 → 硬切。
    depth 与 len(parts)>1 双重判断是必需的：若唯一终止符恰好落在句尾，
    _split_keep 会切出一个与输入等长的片段，不加判断就会无限自递归。
    """
    # 分块阶段不清理，避免 tokenizer 版本差异带来"空片段"歧义
    if len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
        return [text]
    if depth < 32:
        for terminators in (SENTENCE_TERMINATORS, "\n", "；", "，、"):
            if not any(t in text for t in terminators):
                continue
            parts = _split_keep(text, terminators)
            if len(parts) > 1:
                out = []
                for p in parts:
                    out.extend(_enforce_token_limit(p, tokenizer, max_tokens, depth + 1))
                return out
    return _hard_split_by_tokens(text, tokenizer, max_tokens)


_EMBED_MODEL = None
_EMBED_TOKENIZER = None


def _get_embed_model():
    """惰性加载 embedding 模型，全模块只加载一次。

    旧实现把 from_pretrained 放在每批句子的处理函数里，一本 20 万字的书
    会被重复加载上千次，这是建索引慢的主因之一。

    设备固定为 CPU（见 SEMANTIC_EMBED_DEVICE）：分块阶段要把整本书拆成
    数万句逐批编码，每批一次 .cpu() 都会等一次 Metal 命令缓冲区完成；
    实测在 GPU 被其他应用共用时会永久阻塞在
    MPSStream::copy_and_sync → _MTLCommandBuffer waitUntilCompleted。
    实测本机 MPS 与 CPU 吞吐相同（136 句/秒），故用 CPU 换取确定性。
    """
    global _EMBED_MODEL, _EMBED_TOKENIZER
    if _EMBED_MODEL is None:
        from transformers import AutoTokenizer, AutoModel

        _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
        _EMBED_MODEL = AutoModel.from_pretrained(EMBED_MODEL_PATH)
        _EMBED_MODEL.to(SEMANTIC_EMBED_DEVICE)
        _EMBED_MODEL.eval()
    return _EMBED_TOKENIZER, _EMBED_MODEL


def _encode_sentences(sentences):
    """批量编码句子，返回 L2 归一化后的向量矩阵（点积即余弦相似度）。

    向量先在设备上累积、最后一次搬回 CPU，避免每个 batch 同步一次。
    """
    import numpy as np
    import torch

    tokenizer, model = _get_embed_model()
    device = next(model.parameters()).device
    vecs = []
    for i in range(0, len(sentences), EMBED_BATCH_SIZE):
        batch = sentences[i:i + EMBED_BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0].float()
        vecs.append(torch.nn.functional.normalize(cls, p=2, dim=1))
    if not vecs:
        return np.zeros((0, 0), dtype="float32")
    return torch.cat(vecs).cpu().numpy()


def _semantic_split(text, tokenizer, threshold=SEMANTIC_THRESHOLD):
    """语义分块：在语义边界处切分，并保证每块不小于 SEMANTIC_MIN_CHARS 字符。

    与旧实现的三点差异：
      1. 模型只加载一次（旧实现每 20 句 from_pretrained 一次）；
      2. 取消 20 句硬分批 —— 旧实现会在任意位置强制切断，
         长文本实际被切成一堆 ≤20 句的碎片；
      3. 增加最小长度下限 —— 叙事文本相邻句余弦相似度天然偏低（常低于 0.6），
         只看阈值会把每 1~2 句切成一块（实测平均 30 字）。
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    embeddings = _encode_sentences(sentences)
    # 向量已归一化，逐元素相乘求和即为相邻句余弦相似度
    sims = (embeddings[:-1] * embeddings[1:]).sum(axis=1)

    chunks = []
    current = [sentences[0]]
    current_chars = len(sentences[0])

    for i in range(1, len(sentences)):
        sent = sentences[i]
        is_boundary = sims[i - 1] < threshold
        if is_boundary and current_chars >= SEMANTIC_MIN_CHARS:
            chunks.append(" ".join(current))
            current = [sent]
            current_chars = len(sent)
        else:
            current.append(sent)
            current_chars += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks


def _hierarchical_split(text, tokenizer):
    """语义分块 + 分层组织：语义分块决定边界，父子结构提供上下文。"""
    
    # 第一步：语义分块（决定边界）
    semantic_chunks = _semantic_split(text, tokenizer)
    
    # 第二步：分层组织（子块+父块）
    results = []
    for chunk in semantic_chunks:
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
        
        if chunk_tokens <= CHILD_MAX_TOKENS:
            # 短文本：子块=父块
            results.append((chunk, chunk))
        elif chunk_tokens <= PARENT_MAX_TOKENS:
            # 中等文本：整个作为父块，内部按句子分子块
            child_chunks = _split_into_children(chunk, tokenizer)
            for child in child_chunks:
                results.append((child, chunk))
        else:
            # 长文本：先按PARENT_MAX_TOKENS分割成父块，再分子块
            parent_chunks = _split_by_tokens(chunk, tokenizer, PARENT_MAX_TOKENS)
            for parent_text in parent_chunks:
                child_chunks = _split_into_children(parent_text, tokenizer)
                for child in child_chunks:
                    results.append((child, parent_text))
    
    return results


def _split_by_tokens(text, tokenizer, max_tokens):
    """按token数分割文本，保持句子完整性。

    逐句先做 token 上限兜底：单个超长句（无标点文言段）若不先切开，
    会被原样当成一个父块，直接突破 max_tokens。
    """
    sentences = _split_sentences(text, tokenizer, max_tokens)
    chunks = []
    current = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens
    
    if current:
        chunks.append(" ".join(current))
    
    return chunks


def _split_into_children(parent_text, tokenizer):
    """将父块分割成子块，带重叠。

    逐句先做 CHILD_MAX_TOKENS 兜底：这是"子块不超限"的实际保证点 ——
    语料中最长单句 672 字（西游记无标点文言段），不兜底就会产出
    远超 128 token 的子块，embedding 时被 tokenizer 静默截断，
    检索到的其实是"半句话"。
    """
    sentences = _split_sentences(parent_text, tokenizer, CHILD_MAX_TOKENS)
    children = []
    current = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_tokens + sent_tokens > CHILD_MAX_TOKENS and current:
            children.append(" ".join(current))
            # 重叠：从已结束的子块尾部回填 CHUNK_OVERLAP token
            overlap, overlap_tokens = [], 0
            for s2 in reversed(current):
                ot = len(tokenizer.encode(s2, add_special_tokens=False))
                if overlap_tokens + ot > CHUNK_OVERLAP:
                    break
                overlap.insert(0, s2)
                overlap_tokens += ot
            # 重叠不能把子块顶过上限。原先这里直接 overlap_tokens + sent_tokens，
            # 而兜底后的长句可达 128 token，叠加 32 token 重叠即 160 —— 实测有
            # 254 个子块因此超限（旧实现 85 个）。裁剪时从最旧的重叠句丢起，
            # 保留离新句最近的上下文。
            while overlap and overlap_tokens + sent_tokens > CHILD_MAX_TOKENS:
                oldest = overlap.pop(0)
                overlap_tokens -= len(tokenizer.encode(oldest, add_special_tokens=False))
            current = overlap + [sent]
            current_tokens = overlap_tokens + sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens
    
    if current:
        children.append(" ".join(current))
    
    return children


def read_book_text(filepath):
    """严格解码书籍文本；遇到非法 UTF-8 字节时精确报错，并降级为 U+FFFD。

    原实现用 errors="ignore" 静默丢弃坏字节（水浒传、红楼梦各有 2 个），
    导致损坏位置和内容都无从追查。这里改为显式暴露：
    坏字节被替换成 U+FFFD（可检测、可回溯），而不是无声消失。

    另：原先这里把 CRLF/CR 显式归一为 LF（水浒传含 8436 个 CR），理由是按字节读取
    绕过了文本模式的通用换行转换。现已删除 —— 实测 sentencex 自身就把 \r\n\r\n 当句
    边界，且输出不含 \r，四本书的句子序列与归一前逐元素完全相同（配合已删除的
    \n\n 段落预切，见 _split_sentences）。保留它只会让人以为 CRLF 需要特殊对待。

    最后做引号归一（NORMALIZE_QUOTES）：红楼梦原文开引号用 “、闭引号却混用 ” 与 "，
    直接分句会把大段对话吞进同一句。这一步会改写原文里的直引号，
    关掉只需设 NORMALIZE_QUOTES=0。
    """
    with open(filepath, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print(
            f"    ⚠️  {os.path.basename(filepath)} 非法 UTF-8 字节: "
            f"偏移 {e.start}-{e.end}，字节 {raw[e.start:e.end]!r}（已替换为 U+FFFD）"
        )
        text = raw.decode("utf-8", errors="replace")
    if NORMALIZE_QUOTES:
        text = fix_quotes(text)
    return text


def build_chunks():
    """读取 books/*.txt → 分块 → 写 chunks.json。返回 chunk 列表。

    不做数据清洗：四个文本文件本身就是干净的 UTF-8 典籍，
    实测页码 0 处、HTML 0 处、控制字符几无；而曾接入的清洗会删掉
    全部换行、5.7 万个中文引号和所有阿拉伯数字，属纯损失。
    """
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

        text = read_book_text(filepath)

        chunks = _hierarchical_split(text, tokenizer)

        for chunk_idx, (child_text, parent_text) in enumerate(chunks):
            if len(child_text) <= 5:
                continue

            all_chunks.append({
                "child_text": child_text,
                "parent_text": parent_text,
                "book": book_name,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                # contextual_text == child_text：不再拼接上下文前缀。
                # 旧前缀形如 "《水浒传》 > 第五回 > 涉及: 武松\n\n<正文>"，实测价值极低：
                #   77.6% (16410/21159) 的块除《书名》外没有任何信息，而《书名》对
                #   同书所有块是同一个常量；回目正则整体只命中 1.4%（仅回目 178 条 +
                #   回目&人名 136 条）；人名来源是 21 个硬编码人名表。
                # 字段本身**保留**：稠密索引(build_index)与 rerank 都读它，
                # check_health 也校验 payload 里必须存在该字段。让两者相等等价于
                # "关闭前缀"，行为退化清晰、便于 A/B 对照，不需要动消费链。
                "contextual_text": child_text,
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


def bm25_tokenize(text):
    """jieba 分词 → 空格连接的字符串，供 Qdrant BM25 的 word 分词器接管。

    为什么这么做（而不是让 Qdrant 自己分词）：
      Qdrant BM25 默认的 word 分词器按空白/标点切，中文等于不切词 ——
      "武松喝了三碗酒，上了景阳冈打虎。" 只得到 2 个 term（逗号两侧），
      查询 "武松打虎" 永远对不上。它的 multilingual 分词器虽能切中文，
      但本语料实测召回低于 jieba（见 compare_sparse.py）。
      因此把分词这一层换成 jieba，其余（打分方案、服务端推理接口、
      IDF 修饰符）全部保持 Qdrant 内置 BM25 不变。

    cut_for_search 模式会额外产出细粒度词（如 "景阳冈" 同时给出 "景阳"），
    这是有意的召回扩张；文档侧与查询侧使用同一个函数，保证同构。
    """
    import jieba

    tokens = []
    for token in jieba.cut_for_search(text or ""):
        token = token.strip()
        if not token or all(ch in _PUNCT_ONLY for ch in token):
            continue
        tokens.append(token)
    return " ".join(tokens)


def sparse_encode(text):
    """返回交给 Qdrant BM25 服务端推理的 Document 对象（文档侧/查询侧同一个）。

    Qdrant 收到后会：word 分词 → 小写化 → 计算 BM25 的 TF 与文档长度归一化，
    存入稀疏向量；IDF 由集合的 Modifier.IDF 在查询时施加 —— 这就是 Qdrant
    内置 BM25 的完整打分方案，本函数只替换了其中"分词"这一层。

    注意 Qdrant 不会为 point 自动生成稀疏向量：不显式写入就没有，且不报错。
    """
    from qdrant_client import models

    return models.Document(
        text=bm25_tokenize(text),
        model=BM25_MODEL_NAME,
        options=BM25_TEXT_OPTIONS,
    )


# ============================================================================
# 阶段二：索引（GPU 加速版）
# ============================================================================
def build_index():
    """chunks.json → 向量化 → Qdrant (稠密+稀疏混合)"""
    import time
    import numpy as np
    import torch
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, Distance, PointStruct,
        SparseVectorParams, Modifier
    )

    if not os.path.exists(CHUNKS_JSON):
        raise FileNotFoundError(f"找不到 {CHUNKS_JSON}，请先运行: python app.py process")

    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    if not all_chunks:
        raise ValueError("chunks.json 为空")

    print(f"加载分块数据: {len(all_chunks)} 条")

    # 本次建造标识：写进每个 point，供检索端识别索引换代
    import uuid

    build_id = uuid.uuid4().hex
    print(f"索引建造标识: {build_id}")

    # ---- GPU 优化策略 ----
    device = INDEX_DEVICE or get_device()
    use_gpu = device == "mps"
    tokenizer = model = None

    if use_gpu:
        try:
            # GPU 模式：降低batch避免MPS内存溢出
            tokenizer, model, _ = load_embedding_model(device="mps")
            # 使用 fp16 推理加速（bge 系列模型小，fp16 精度足够）
            model = model.half()
        except Exception as exc:
            # MPS 初始化会因统一内存被挤压而失败，实测报错：
            #   RuntimeError: invalid low watermark ratio 1.4
            # （torch.mps.recommended_max_memory() 被压到阈值以下时水位比越界；
            #  8GB 机器上 Ollama 等进程占用 GPU 统一内存时必现，且时机随机）
            # 这是环境问题而非代码缺陷，但原先会让整个建索引流程直接崩掉 ——
            # 退到 CPU 只是慢一些，向量结果等价，不该失败。
            print(f"    ⚠️  MPS 初始化失败，回退 CPU 建索引（更慢，结果等价）: "
                  f"{type(exc).__name__}: {exc}")
            tokenizer, model, use_gpu = None, None, False

    if use_gpu:
        batch_size = 128
        COOL_DOWN_SLEEP = 0.1  # 增加冷却间隔
        print(f"Embedding 设备: mps (fp16, batch_size={batch_size})")
    else:
        # CPU 模式：多线程 + 适中 batch
        tokenizer, model, device = load_embedding_model(device="cpu")
        # 6 而非 8：M2 是 4P+4E，吃满 8 线程会让前台明显卡顿（8GB 机器上更甚），
        # 留 2 个线程给系统，吞吐损失很小。
        torch.set_num_threads(6)
        batch_size = 128
        COOL_DOWN_SLEEP = 0.05
        print(f"Embedding 设备: cpu (6线程, batch_size={batch_size})")

    total = len(all_chunks)
    all_embeddings = []
    t0 = time.time()

    # ---- 阶段 1: 生成 Embeddings（GPU/CPU）----
    print(f"批量生成 Embedding...")
    for i in range(0, total, batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["contextual_text"] for c in batch]

        if use_gpu:
            # GPU: 混合精度推理
            inputs = tokenizer(texts, padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
            inputs = {k: v.to("mps") for k, v in inputs.items()}
            with torch.no_grad():
                with torch.amp.autocast(device_type="mps", dtype=torch.float16):
                    outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0].float()
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            all_embeddings.append(cls.cpu().numpy())
            # 清理MPS缓存避免内存积累
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            # CPU: 标准推理
            all_embeddings.append(embed(texts, tokenizer, model, device, is_query=False))

        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        if (i // batch_size) % 5 == 0 or done >= total:
            speed = done / elapsed if elapsed > 0 else 0
            print(f"  {done / total * 100:5.1f}% ({done}/{total}) {speed:.0f} 条/秒")
        time.sleep(COOL_DOWN_SLEEP)

    embeddings = np.vstack(all_embeddings).astype("float32")

    # 释放 embedding 模型内存
    del model, tokenizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ---- 阶段 2: 写入 Qdrant (稠密+稀疏) ----
    print("写入 Qdrant (稠密+稀疏混合)...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # 删除旧集合（如果存在）
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
    
    # 创建新集合（支持稠密+稀疏向量）
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=embeddings.shape[1],  # 向量维度 (512)
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                modifier=Modifier.IDF  # 启用BM25 IDF
            )
        }
    )
    
    # 批量写入
    write_batch = 1000
    for i in range(0, total, write_batch):
        batch_chunks = all_chunks[i:i + write_batch]
        batch_embeddings = embeddings[i:i + write_batch]
        
        points = []
        for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
            # 稀疏向量用 child_text（检索命中的正文），不用 contextual_text：
            # 前缀里的《书名》与回目是所有块共有的高频词，进 BM25 只会稀释权重。
            doc_text = chunk.get("child_text", "")
            
            point = PointStruct(
                # Qdrant 的 point id 只接受无符号整数或 UUID。原始 id 形如
                # "三国演义_0"，服务端直接 400 拒绝（这会整批 upsert 失败），
                # 故用全局序号作 id，原始 id 存进 payload 供结果回填。
                id=i + j,
                vector={
                    DENSE_VECTOR_NAME: emb.tolist(),
                    # 稀疏向量必须显式生成（Qdrant 不会替你算）；这里传 Document，
                    # 由 Qdrant 用内置 BM25 模型完成打分，分词则由 jieba 预先做好
                    SPARSE_VECTOR_NAME: sparse_encode(doc_text),
                },
                payload={
                    "chunk_id": chunk["id"],
                    "child_text": chunk["child_text"],
                    "parent_text": chunk["parent_text"],
                    "book": chunk["book"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "contextual_text": chunk["contextual_text"],
                    # 供检索端识别"索引已重建"，见 INDEX_BUILD_ID_FIELD 注释
                    INDEX_BUILD_ID_FIELD: build_id,
                }
            )
            points.append(point)
        
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
        done = min(i + write_batch, total)
        if (i // write_batch) % 5 == 0 or done >= total:
            print(f"  Qdrant 写入进度: {done}/{total}")
    
    info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"  Qdrant 写入完成: {client.count(collection_name=COLLECTION_NAME).count} 条")
    # 打印集合**实际**的稠密维度，而不是本地 embeddings.shape[1]。
    # 旧写法两者混用（键名取自集合配置、size 取自局部变量），维度一旦不一致
    # 日志会显示出一个并不存在的集合配置，排查建索引问题时会直接把人带偏。
    coll_dense_size = next(iter(info.config.params.vectors.values())).size
    assert coll_dense_size == embeddings.shape[1], (
        f"集合稠密维度 {coll_dense_size} 与本次编码维度 {embeddings.shape[1]} 不一致"
        f"（集合名 {COLLECTION_NAME}）"
    )
    print(f"  集合向量空间: 稠密 {list(info.config.params.vectors.keys())} "
          f"(size={coll_dense_size}) + 稀疏 {list(info.config.params.sparse_vectors.keys())}")
    print(f"  索引构建完成: 稠密向量 + 稀疏向量（jieba 分词 + Qdrant 内置 BM25 打分）")
    print(f"  索引建造标识: {build_id}（检索端据此自动丢弃进程内的旧分块缓存）")
    print(f"总耗时: {time.time() - t0:.1f} 秒")


# 注：这里原有一个 _get_books_hash()，用于"书没变就跳过分块"。已删除：
#   1. 它是死代码（定义了、在 app.py 里再导出过，却从未被调用）；
#   2. 它**不该被启用** —— chunks.json 不只依赖 books/ 内容，还依赖分句器与
#      分块参数。换一次 sentencex 就是活例子：书一个字节没变，分块全变了。
#      用书哈希跳过会造成"改了代码却仍在跑旧分块"这种最难排查的错误。
#   将来若要做增量重建，键必须是 (books hash + 分句器/分块配置 hash)。


# ============================================================================
# 检索
# ============================================================================
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
        self._chunks_build_id = None
        self._client = None

    # ---- 资源加载 ----
    def _get_client(self):
        from qdrant_client import QdrantClient
        if self._client is None:
            # 连接Docker服务。
            # 必须显式放宽 timeout：qdrant-client 默认只有 5s，而 _get_chunks 要
            # 一次拉回全部点（当前 2.1 万条、名义 1.4s），负载稍高就会 ReadTimeout
            # —— 表现为"检索偶发失败"，且失败点离真正原因很远，极难排查。
            self._client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                timeout=60,
            )
        return self._client
    
    def count(self):
        """集合内的点数；集合不存在时返回 0。

        旧接口叫 _get_collection()，但返回的其实是 QdrantClient —— 名字与行为
        不符，调用方写成 .count() 会直接 TypeError（missing 'collection_name'），
        app.py 的 UI 因此一打开就崩。这里改成语义明确的 count()，
        并把"集合不存在"归一为 0，让上层能给出"请先构建索引"的提示而不是抛栈。
        """
        client = self._get_client()
        if not client.collection_exists(COLLECTION_NAME):
            return 0
        return client.count(collection_name=COLLECTION_NAME).count

    def _get_embed(self):
        if self._embed is None:
            self._embed = load_embedding_model()
        return self._embed

    def _get_rerank(self):
        if self._rerank is None:
            self._rerank = load_reranker_model()
        return self._rerank

    def _current_build_id(self):
        """单点读取当前索引的建造标识（约 1ms）。

        取不到（集合为空、连不上、或该 point 是加入本机制之前写入的旧数据）返回 None；
        此时不做任何失效判断，行为与加此机制之前一致。
        """
        try:
            pts = self._get_client().retrieve(
                collection_name=COLLECTION_NAME, ids=[0], with_payload=True)
        except Exception:
            return None
        if not pts:
            return None
        return (pts[0].payload or {}).get(INDEX_BUILD_ID_FIELD)

    def _get_chunks(self):
        """返回 {qdrant_point_id: {...}}，一次全量加载后缓存；索引换代时自动重载。

        key 用 Qdrant 的 point id（整数，与检索返回的 point.id 同类型）。
        这里不能吞异常：旧实现把整个 scroll 包在 except 里退化成空字典，
        于是"索引没建成功"和"检索确实无结果"表现完全一样，无从排查。

        缓存失效：每次取用前比对一次索引建造标识。没有这一步，重建索引后
        仍在运行的进程会继续用旧分块 —— 不仅内容过期，point id 与旧内容的
        对应关系也已改变，等于拿错正文喂给 LLM。
        """
        current_id = self._current_build_id()
        if self._chunks is not None:
            if current_id is not None and current_id != self._chunks_build_id:
                print(f"检测到索引已重建（build_id {self._chunks_build_id} → {current_id}），"
                      f"丢弃进程内的旧分块缓存并重新加载")
                self._chunks = None
            else:
                return self._chunks

        client = self._get_client()
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100000,
            with_payload=True,
            with_vectors=False
        )
        chunks = {}
        for point in result[0]:
            payload = point.payload
            chunks[point.id] = {
                "id": payload.get("chunk_id", str(point.id)),
                "point_id": point.id,
                "child_text": payload.get("child_text", ""),
                "parent_text": payload.get("parent_text", ""),
                "book": payload.get("book", ""),
                "contextual_text": payload.get("contextual_text", ""),
            }
        self._chunks = chunks
        self._chunks_build_id = current_id
        if not chunks:
            print(f"警告: 集合 {COLLECTION_NAME} 为空，请先运行: python app.py index")
        return self._chunks

    # ---- 检索主链路 ----
    def hybrid_search(self, query, top_k=TOP_K, return_steps=False):
        """
        混合检索：稠密 + 稀疏(jieba 分词 + Qdrant BM25) → 服务端 RRF → rerank。

        原签名里有个 expand_queries 参数，但它从未被使用（只赋值给一个再也没读过的
        局部变量），查询扩展功能实际不存在。为避免"以为传了就生效"，参数已移除；
        要真正做查询扩展（多路改写 + 结果合并）应作为独立特性加，并配评测验证收益。

        return_steps: 是否返回详细步骤信息。
        """
        import numpy as np
        import time
        from qdrant_client import models

        chunks = self._get_chunks()
        if not chunks:
            return ([], {}) if return_steps else []

        steps = {} if return_steps else None
        t0 = time.time()

        client = self._get_client()
        tokenizer, model, device = self._get_embed()

        # 1) 生成稠密向量
        t1 = time.time()
        q_emb = embed([query], tokenizer, model, device, is_query=True)
        dense_embedding = q_emb[0].tolist()
        if steps is not None:
            steps["生成稠密向量"] = {
                "耗时": f"{time.time() - t1:.2f}s",
                "维度": len(dense_embedding),
            }

        # 2) Qdrant混合搜索（稠密+稀疏 → RRF融合）
        # 稀疏侧必须传编码后的向量；传明文会被服务端 400 拒绝（旧实现即如此，
        # 又被下面的 except 静默吞掉，于是"混合检索"长期退化为纯稠密）。
        t2 = time.time()
        sparse_query = sparse_encode(query)
        n_sparse_terms = len(sparse_query.text.split())
        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    # 稠密向量搜索（语义相似）
                    models.Prefetch(
                        query=dense_embedding,
                        using=DENSE_VECTOR_NAME,
                        limit=RERANK_TOP_K
                    ),
                    # 稀疏向量搜索（词频/BM25 关键词匹配）
                    models.Prefetch(
                        query=sparse_query,
                        using=SPARSE_VECTOR_NAME,
                        limit=RERANK_TOP_K
                    )
                ],
                # RRF融合（Reciprocal Rank Fusion，在 Qdrant 服务端完成）
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=RERANK_TOP_K,
                with_payload=True
            )
            fused_results = results.points
        except Exception as e:
            # 稀疏通道不可用时仍可退化为纯稠密，但必须留下痕迹
            print(f"⚠️  混合搜索失败，已回退到纯稠密检索（召回会下降）: {e}")
            print(f"    稀疏查询分词: {n_sparse_terms} 个 term ({sparse_query.text[:60]})")
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_embedding,
                using=DENSE_VECTOR_NAME,
                limit=RERANK_TOP_K,
                with_payload=True
            )
            fused_results = results.points

        if steps is not None:
            steps["Qdrant混合搜索"] = {
                "耗时": f"{time.time() - t2:.2f}s",
                "候选数": len(fused_results),
                "方法": "稠密+稀疏 RRF融合",
                "稀疏term数": n_sparse_terms,
            }

        # 3) 准备候选集
        candidates = []
        candidate_ids = []
        for point in fused_results:
            cid = point.id
            if cid in chunks:
                candidates.append(chunks[cid])
                candidate_ids.append(cid)

        if not candidates:
            return ([], steps) if return_steps else []

        # 4) Rerank重排
        t3 = time.time()
        rr_tok, rr_model, rr_dev = self._get_rerank()
        cand_texts = [c["contextual_text"] for c in candidates]
        reranked = rerank_indices(query, cand_texts, rr_tok, rr_model, rr_dev, top_k=top_k)

        if steps is not None:
            steps["Rerank重排"] = {
                "耗时": f"{time.time() - t3:.2f}s",
                "输入": len(candidates),
                "输出": len(reranked),
            }
            steps["总耗时"] = f"{time.time() - t0:.2f}s"

        # 5) 构建最终结果
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

        if return_steps:
            return output, steps
        return output
