# ── 基础设施：环境配置与日志 ──
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import datetime
import glob
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import unicodedata
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from sentencex import segment
import requests
try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError("缺少依赖 python-dotenv（配置会静默退到代码默认值），安装: pip install python-dotenv") from exc
# 记录 load_dotenv 之前的环境变量：health 的 env_declared_but_unread 只点名「写进 .env 但代码
# 从不读取」的变量，避免把本机为 Ollama 服务进程准备的 OLLAMA_* 等 shell 变量误报成应用配置。
_ENV_KEYS_BEFORE_DOTENV = frozenset(os.environ)
load_dotenv()

# ── 启动副作用 ──
# 本地 Ollama/Qdrant 不走系统代理（否则 requests 可能被劫持而超时）。
_loopback = ",".join(p for p in (os.environ.get("NO_PROXY", ""), "127.0.0.1", "localhost", "::1") if p)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = _loopback
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum")


def env_int(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"环境变量 {name}={raw!r} 不是合法整数") from None


def env_float(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise ValueError(f"环境变量 {name}={raw!r} 不是合法浮点数") from None


def env_int_min(name, default, minimum):
    val = env_int(name, default)
    if val < minimum:
        raise ValueError(f"环境变量 {name}={val!r} 小于允许的最小值 {minimum}")
    return val


def env_float_min(name, default, minimum):
    val = env_float(name, default)
    if val < minimum:
        raise ValueError(f"环境变量 {name}={val!r} 小于允许的最小值 {minimum}")
    return val


def env_bool(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def env_choice(name, default, allowed):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    val = raw.strip().lower()
    if val not in allowed:
        raise ValueError(f"环境变量 {name}={raw!r} 不在允许取值 {sorted(allowed)} 内")
    return val


_ENV_NAME_RE = re.compile( r"""(?:os\.getenv|os\.environ\.get)\(\s*[\"']([A-Za-z0-9_]+)[\"']""" r"""|env_(?:int|float|bool|choice|int_min|float_min)\(\s*[\"']([A-Za-z0-9_]+)[\"']""" )
_ENV_FAMILIES = ("RAG_", "OLLAMA_", "QDRANT_", "SEMANTIC_", "INDEX_")
_ENV_STANDALONE = ("MODEL", "NORMALIZE_QUOTES")
_ENV_READ_SOURCE = None


def env_declared_but_unread():
    global _ENV_READ_SOURCE
    if _ENV_READ_SOURCE is None:
        import inspect
        try:
            src = inspect.getsource(sys.modules[__name__])
        except (OSError, TypeError, KeyError):
            src = ""
        _ENV_READ_SOURCE = {m[0] or m[1] for m in _ENV_NAME_RE.findall(src)}
    declared = { k for k in os.environ if k not in _ENV_KEYS_BEFORE_DOTENV and (k.startswith(_ENV_FAMILIES) or k in _ENV_STANDALONE) }
    return sorted(declared - _ENV_READ_SOURCE)


class _StdoutHandler(logging.StreamHandler):
    def emit(self, record):
        self.stream = sys.stdout
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)


_LOG_FILE = None


def _file_handler():
    """共享日志文件（RAG_LOG_FILE，默认 rag.log）；打开失败退回仅控制台。"""
    global _LOG_FILE
    if _LOG_FILE is None:
        path = os.getenv("RAG_LOG_FILE", "rag.log")
        try:
            h = logging.FileHandler(path, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        except Exception:
            h = None
        _LOG_FILE = h
    return _LOG_FILE


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_StdoutHandler())
        fh = _file_handler()
        if fh is not None:
            logger.addHandler(fh)
        logger.setLevel(os.getenv("RAG_LOG_LEVEL", "INFO").strip().upper() or "INFO")
        logger.propagate = False
    return logger
console = get_logger("console")
logger = get_logger("rag_engine")


def log(*parts):
    console.info(" ".join(str(part) for part in parts))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_URL = f"{OLLAMA_BASE_URL}/api/chat"
MODEL = os.getenv("MODEL", "qwen3.5:2b-q4_K_M")
# 思维链唯一开关：模型/页面/API/会话共用（默认关）；开启时只当场展示、永不进入生成上下文。
THINK = env_bool("RAG_THINK", False)
# num_ctx / num_predict 与思维链开关解耦：固定预算，不随开关浮动。
NUM_CTX = int(os.getenv("RAG_NUM_CTX", "8192"))
NUM_PREDICT = int(os.getenv("RAG_NUM_PREDICT", "4096"))
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))


# ── 生成客户端：Ollama 流式对话 ──
# 事件词表封闭产出（stream_chat + run_turn）：retrieval → thinking / delta / error / done。
EVENT_RETRIEVAL = "retrieval"
EVENT_THINKING = "thinking"
EVENT_DELTA = "delta"
EVENT_ERROR = "error"
EVENT_DONE = "done"


def stream_chat(messages, temperature=0.0):
    r = None
    dropped_thinking = 0
    try:
        r = requests.post( API_URL, json={ "model": MODEL, "messages": messages, "stream": True, "think": THINK, "options": { "temperature": temperature, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, }, }, stream=True, timeout=TIMEOUT, )
        if r.status_code != 200:
            yield EVENT_ERROR, f"HTTP {r.status_code} {r.text[:300]}"
            return
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                # return 而不是 continue：本函数对外承诺「error 与 done 互斥」，消费侧
                # （_run_assistant_turn 只在 done 时入历史、_handle_ask 只在 done 时收尾）
                # 全靠这条。旧写法 continue 会让同一流后面的 {"done":true} 照常产出 done，
                # 于是「被错误打断的残缺答案」同时满足 done 与非空，被写进会话历史 ——
                # 而页面只渲染了错误，那段答案用户一个字符都没见过。
                yield EVENT_ERROR, f"[Ollama] {chunk['error']}"
                return
            message = chunk.get("message") or {}
            # 生产端唯一闸门：关着时事件流里不存在 EVENT_THINKING。
            if message.get("thinking"):
                if THINK:
                    yield EVENT_THINKING, message["thinking"]
                else:
                    # 开关关着却收到思维链 = 模型侧没听 think=false；按开关丢弃并告警一次。
                    dropped_thinking += 1
                    if dropped_thinking == 1:
                        log("⚠️  RAG_THINK 关闭但模型仍在返回思维链：已按开关丢弃；"
                            "请确认该模型支持 think=false，否则这部分预算白花")
            if message.get("content"):
                yield EVENT_DELTA, message["content"]
            if chunk.get("done"):
                yield EVENT_DONE, { "done_reason": chunk.get("done_reason"), "eval_count": chunk.get("eval_count"), "prompt_eval_count": chunk.get("prompt_eval_count"), }
    except Exception as exc:
        yield EVENT_ERROR, f"{type(exc).__name__}: {exc}"
    finally:
        if r is not None:
            r.close()


# ── 通用文本谓词 ──
def _has_searchable_content(text):
    # 认 Unicode 字母与数字（CJK 的 Lo、数字的 Nd）；标点、符号、零宽等格式字符不算内容。
    # 数字必须算：年份/编号（"1998"、"300 强"、"3.5"）是合法提问，按"仅字母"拦会让它们
    # 零检索，随后无拒答链路照常生成 —— 正是 AGENTS 点名的"检索为空却照常作答"。
    # 不变量是"空查询与纯标点必拦"：纯数字既非空也非标点，不该被这道闸门吃掉。
    if not isinstance(text, str) or not text:
        return False
    return any(unicodedata.category(ch)[0] in ("L", "N") for ch in text)


# ── 领域与检索侧配置 ──
# 全部可配置；默认值面向通用垂直领域，核心代码不假设语料形态、文件名或来源字段。
DATA_DIR = os.getenv("RAG_DATA_DIR") or os.getenv("RAG_BOOKS_DIR") or "corpus"
DOC_GLOB = os.getenv("RAG_DOC_GLOB", "*.txt")
DOMAIN_NAME = os.getenv("RAG_DOMAIN_NAME", "垂直领域知识库")
DOMAIN_DESCRIPTION = os.getenv("RAG_DOMAIN_DESCRIPTION", "")
SOURCE_FIELD = (os.getenv("RAG_SOURCE_FIELD") or "source").strip() or "source"
SOURCE_LABEL = os.getenv("RAG_SOURCE_LABEL", "来源")
HEALTH_PROBE_QUESTION = os.getenv("RAG_HEALTH_PROBE", "请用一句话说明你能回答哪个领域的问题。")
_RESERVED_SOURCE_FIELDS = {
    "chunk_id", "id", "child_text", "parent_text", "chunk_index", "total_chunks",
    "contextual_text", "parent_id", "parent_chunk_count", "build_id", "lexicon_id",
    # 见 INDEX_BUILD_ID_FIELD 一组：这两个字段名也是内部保留键，不能被来源字段占用
    "chunk_fingerprint",
    # 以下不是 payload 字段，而是**对外输出**要用的键：SOURCE_FIELD 原样写进 /search 的
    # _result_json 与 steps.context_sources，同名即就地覆盖 —— score/text/context/n/
    # rerank_score/source_label/point_id 会静默变成来源字符串（响应看着正常，字段全错）。
    # source 是默认值本身，不能列入，否则启动必炸。
    "score", "text", "context", "n", "rerank_score", "source_label", "point_id",
}
if SOURCE_FIELD in _RESERVED_SOURCE_FIELDS:
    raise ValueError(
        f"RAG_SOURCE_FIELD={SOURCE_FIELD!r} 与内部字段冲突，"
        f"不能使用 {sorted(_RESERVED_SOURCE_FIELDS)}"
    )
TOP_K = env_int_min("RAG_TOP_K", 5, 1)
RERANK_TOP_K = env_int_min("RAG_RERANK_TOP_K", 20, 1)
RERANK_BATCH = env_int_min("RAG_RERANK_BATCH", 32, 1)  # padding 隔离下分数与批大小无关；候选<=20 单批一次过
RERANK_MAX_LENGTH = env_int_min("RAG_RERANK_MAX_LENGTH", 384, 1)
RERANK_ON = env_choice("RAG_RERANK_ON", "child", ("child", "parent"))
PRELOAD_MODELS = env_bool("RAG_PRELOAD", True)  # api/serve 启动时后台预热模型与词典，首查询冷加载提前到启动期
CHILD_MAX_TOKENS = env_int("RAG_CHILD_MAX_TOKENS", 128)
PARENT_MAX_TOKENS = env_int("RAG_PARENT_MAX_TOKENS", 512)
CHUNK_OVERLAP = env_int("RAG_CHUNK_OVERLAP", 32)
PARENT_MIN_TOKENS = env_int("RAG_PARENT_MIN_TOKENS", 192)
CHILD_MIN_TOKENS = env_int("RAG_CHILD_MIN_TOKENS", 32)
if CHILD_MAX_TOKENS > PARENT_MAX_TOKENS:
    raise ValueError(f"RAG_CHILD_MAX_TOKENS={CHILD_MAX_TOKENS} 不能大于 RAG_PARENT_MAX_TOKENS={PARENT_MAX_TOKENS}")
SEMANTIC_THRESHOLD = env_float("RAG_SEMANTIC_THRESHOLD", 0.6)
SEMANTIC_MIN_CHARS = env_int("RAG_SEMANTIC_MIN_CHARS", 500)
EMBED_BATCH_SIZE = env_int("RAG_EMBED_BATCH_SIZE", 64)
SEMANTIC_EMBED_DEVICE = os.getenv("SEMANTIC_EMBED_DEVICE", "")  # 空=自动选设备（MPS 优先），见 get_device()
INDEX_DEVICE = os.getenv("INDEX_DEVICE", "")
QDRANT_HOST = os.getenv("QDRANT_HOST") or os.getenv("RAG_QDRANT_HOST") or "localhost"
QDRANT_PORT = int(os.getenv("QDRANT_PORT") or os.getenv("RAG_QDRANT_PORT") or "6333")
# docker-compose.yml 的 QDRANT__SERVICE__API_KEY 取的就是它：不接这条通路，一旦设置了
# key，应用侧每个 Qdrant 请求都会 401，且日志里没有任何地方能填 key（失效原型：静默）。
QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or os.getenv("RAG_QDRANT_API_KEY") or "").strip() or None
COLLECTION_BASE = os.getenv("RAG_COLLECTION_BASE", "rag_documents_v1")
COLLECTION_NAME = COLLECTION_BASE
COLLECTION_ALIAS = os.getenv("RAG_COLLECTION_ALIAS", "rag_current")
USE_COLLECTION_ALIAS = env_bool("RAG_USE_COLLECTION_ALIAS", True)
EMBED_MODEL_PATH = os.getenv("RAG_EMBED_MODEL_PATH", "./models/bge-base-zh-v1.5")
RERANK_MODEL_PATH = os.getenv("RAG_RERANK_MODEL_PATH", "./models/bge-reranker-base")
_ARTIFACTS_DIR = os.getenv("RAG_ARTIFACTS_DIR", "./artifacts")
CHUNKS_JSON = os.path.join(_ARTIFACTS_DIR, "chunks.json")


CHUNKS_META_JSON = os.path.splitext(CHUNKS_JSON)[0] + ".meta.json"
CHUNKS_META_VERSION = 1
EMBED_CACHE_DIR = os.path.join(_ARTIFACTS_DIR, "embeddings")
EMBED_CACHE_ENABLED = env_bool("RAG_EMBED_CACHE", True)
DOC_CHUNK_DIR = os.path.join(_ARTIFACTS_DIR, "docs")  # 单文档分块缓存：只重分块增/改的文档
LEXICON_DIR = os.getenv("RAG_LEXICON_DIR", "./lexicon")
SCROLL_PAGE_SIZE = 1000
INDEX_BUILD_ID_FIELD = "build_id"
LEXICON_ID_FIELD = "lexicon_id"
CHUNK_FINGERPRINT_FIELD = "chunk_fingerprint"
_PAYLOAD_SPEC = ( ("chunk_id", "id", ""), ("child_text", "child_text", ""), ("parent_text", "parent_text", ""), (SOURCE_FIELD, "source", ""), ("chunk_index", "chunk_index", 0), ("total_chunks", "total_chunks", 0), ("contextual_text", "contextual_text", ""), ("parent_id", "parent_id", ""), ("parent_chunk_count", "parent_chunk_count", 0), )


def _source_of(record):
    """读来源字段，兼容配置字段与旧 book 字段。"""
    if not isinstance(record, dict):
        return ""
    for key in (SOURCE_FIELD, "source", "book"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _chunk_value(chunk, key, default=""):
    """_PAYLOAD_SPEC 字段读取；source 兼容旧产物的 book 键。"""
    if key == "source":
        return _source_of(chunk) or default
    return chunk.get(key, default)


def chunk_payload(chunk, build_id, lexicon_id, chunk_fingerprint=None):
    payload = {f: _chunk_value(chunk, k, d) for f, k, d in _PAYLOAD_SPEC}
    payload[INDEX_BUILD_ID_FIELD] = build_id
    payload[LEXICON_ID_FIELD] = lexicon_id
    # 记录本索引是**哪一份分块产物**建起来的。检索侧拿它与当前切分指纹比对，等价于
    # 词表指纹那道闸门：没有它，process 之后的旧 chunks/Qdrant 会被 serve/api 静默
    # 继续服务（stages 报过期，但没人看），用户拿到的是无声的旧知识库。
    payload[CHUNK_FINGERPRINT_FIELD] = chunk_fingerprint
    return payload
RRF_K = env_int_min("RAG_RRF_K", 60, 0)
RRF_DENSE_WEIGHT = env_float_min("RAG_RRF_DENSE_WEIGHT", 1.0, 0.0)
RRF_SPARSE_WEIGHT = env_float_min("RAG_RRF_SPARSE_WEIGHT", 1.0, 0.0)
if not RRF_DENSE_WEIGHT and not RRF_SPARSE_WEIGHT:
    raise ValueError("RAG_RRF_DENSE_WEIGHT 与 RAG_RRF_SPARSE_WEIGHT 不能同时为 0：两条召回通道都退出等于检索恒为空。")
_RECALL_LIMIT_RAW = env_int("RAG_RECALL_LIMIT", 0)
if _RECALL_LIMIT_RAW < 0:
    raise ValueError(f"环境变量 RAG_RECALL_LIMIT={_RECALL_LIMIT_RAW!r} 不能为负数（0 表示跟随 RAG_RERANK_TOP_K，当前 {RERANK_TOP_K}）")
RECALL_LIMIT = _RECALL_LIMIT_RAW or RERANK_TOP_K
DEDUP_BY_PARENT = env_bool("RAG_DEDUP_BY_PARENT", True)
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
BM25_TEXT_OPTIONS = { "tokenizer": "word", "stemmer": {"type": "none"}, "stopwords": {}, "ascii_folding": True, }
BM25_MODEL_NAME = "qdrant/bm25"
_PUNCT_AND_WHITESPACE = set("，。！？、；：\u201c\u201d\u2018\u2019《》…—（）()[]{}<>!?,.;:'\"-·　 \n\t")


def history_to_messages(history):
    """会话 → 模型消息：只取 role/content；页面旁路字段（thinking/trace）永不进入生成上下文。"""
    out = []
    for m in history or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not content or not str(content).strip():
            continue
        out.append({"role": m.get("role") or "user", "content": content})
    return out


def strip_current_turn(history, query):
    msgs = list(history or [])
    if not msgs or not query:
        return msgs, False
    last = msgs[-1]
    if (isinstance(last, dict) and last.get("role") == "user" and (last.get("content") or "").strip() == query.strip()):
        return msgs[:-1], True
    return msgs, False


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _float16_ok(device):
    return device in ("mps", "cuda")


# ── 模型加载与向量化 ──
def _require_model_dir(path, what, hint):
    if not path or not os.path.isdir(path) or not os.listdir(path):
        raise RuntimeError(f"{what}模型目录缺失或为空: {path!r} —— {hint}")
    return path


def load_embedding_model(device=None, model_path=None):
    from transformers import AutoTokenizer, AutoModel
    device = device or get_device()
    model_path = _require_model_dir(
        model_path or EMBED_MODEL_PATH, "嵌入",
        "创空间看运行日志的 [entrypoint] 权重下载行（下载完成后再试）；"
        "本地确认 RAG_EMBED_MODEL_PATH 指向可用的模型目录",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model, device


def load_reranker_model(device=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = device or get_device()
    dtype = torch.float16 if _float16_ok(device) else torch.float32
    model_path = _require_model_dir(
        RERANK_MODEL_PATH, "重排",
        "创空间看运行日志的 [entrypoint] 权重下载行（下载完成后再试）；"
        "本地确认 RAG_RERANK_MODEL_PATH 指向可用的模型目录",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained( model_path, torch_dtype=dtype ).to(device)
    model.eval()
    return tokenizer, model, device


def _embed_batch(texts, tokenizer, model, device, autocast=False):
    import torch
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        if autocast:
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
    cls = outputs.last_hidden_state[:, 0].float()
    return torch.nn.functional.normalize(cls, p=2, dim=1)


def embed(texts, tokenizer, model, device, is_query=False, autocast=False):
    if is_query:
        texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
    return _embed_batch(texts, tokenizer, model, device, autocast=autocast).cpu().numpy()


# ── 分块：归一、分句、父子打包 ──
SENTENCE_TERMINATORS = "。！？．…!?"
# 段落硬边界判据：只认**真正空行**（两个连续换行，允许 \r）。句内单个 \n 是软折行，
# 属合法排版；而"行内只含空格/TAB 的空行"（\n[ \t]*\n）sentencex 并不当边界 ——
# 旧正则把它也算作硬边界，于是"空行带尾随空格、空行后紧跟闭引号（”）"这类合法 txt
# 会在这里直接抛错，把 process 整体打崩（与 D1 同类：报错还把原因误指成 sentencex 变了）。
_PARAGRAPH_GAP_RE = re.compile(r"\r?\n\r?\n")
_SOFT_WRAP_RE = re.compile(r"\s*\r?\n\s*")
NORMALIZE_QUOTES = env_bool("NORMALIZE_QUOTES", True)


def fix_quotes(text):
    """把直引号 " 归一到弯引号：**单遍二值状态机，所有引号字符共同参与状态**。

    判据只有一条：**当前位置是否处在"已开未闭"的引号里** —— 在内则为闭引号，
    在外则为开引号。“ ” 与已判定的 " 一起维护这个状态。

    为什么不能靠"看前一个字符"（旧实现的做法）：旧规则把"直引号左侧紧邻
    句读/空白/括号/右引号"当作**必为开引号**的硬信号（`_OPEN_QUOTE_CTX`）。
    该前提在真实语料上常常是**反的**，实测（某真实语料，2426 个直引号）：
        `："` 出现   0 次   —— 直引号从不作开引号（开引号一律写 “，5777 次）
        `？"` 出现 915 次   —— 句末标点后的直引号是**闭**引号
        `！"` 出现 402 次、`。"` 45 次
    即"紧跟句读"恰恰是**闭引号**的特征。按旧规则，2426 个直引号里有 2094 个
    被判成开引号，其中 1734 个紧跟中文标点 —— 结果把该文本的引号不配平从
    1600 个未配对 “ 推到 3362 个（`“` 7896 / `”` 4534），把该函数本想修的
    引号错位**做成了一倍**（真实语料里的坏样本）。

    为什么不能只靠 `_paired_curly_positions`（只让成对弯引号参与状态）：
    它把 “ 与 ” 配对，**不承认直引号也是一个闭引号**。混合型文本里大量 “ 的闭合者
    就是 "，于是它们被判为"孤儿"、不参与状态 —— 复算显示只有 332 个直引号
    在状态内（即能被判为闭引号），剩下 ~1700 个仍然是错的。

    为什么"孤儿引号"不再是问题：孤儿的统计口径本身就来自旧算法的假设。
    在"所有引号一起维护状态"的模型下，一个 “ 由谁闭合不再需要预先知道 ——
    每个引号字符都只是把状态翻一次，局部结构因此始终成立。

    纯直引号型文本（`他说"你好"`）的行为与旧规则 2（成对交替）完全一致，
    混合型文本也正确；`NORMALIZE_QUOTES=0` 是逃生开关。
    """
    if '"' not in text:
        return text

    out = []
    inside = False
    for ch in text:
        if ch == '"':
            if inside:
                out.append("\u201d")
                inside = False
            else:
                out.append("\u201c")
                inside = True
        else:
            out.append(ch)
            # 弯引号同样参与状态：只承认直引号的"开"、不承认它的"闭"，
            # 正是上一版失效的根源。
            if ch == "\u201c":
                inside = True
            elif ch == "\u201d":
                inside = False
    return "".join(out)


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

    **不预切 \\n\\n 段落**：sentencex 自身把 \\n\\n 与 \\r\\n\\r\\n 都当句边界，且从不跨段。
    实测全量语料 66079 句，"先按段落预切再逐段分句"与"整篇一次分句"的输出序列
    逐元素完全相同，跨段句恒为 0 —— 那一层是纯冗余。
    "句子不跨段"这条性质改由下面的断言守护：它现在是 sentencex 的实现保证，
    而不是本函数的代码保证，所以必须能被测试发现回归。

    tokenizer/max_tokens 给出时，逐句保证不超过 max_tokens —— 语料里有
    整段无标点的长文（单句最长可达 700 余字），
    纯分句器对它们无能为力，不兜底就会撑破 CHILD_MAX_TOKENS。
    """
    parts = [p.strip() for p in segment("zh", s) if p.strip()]

    # 段落硬边界的守护断言。sentencex 保证引语不可切分、不跨段；一旦这里失败，
    # 说明分句器行为变了（换版本/换库），必须重新评估分块，不能静默继续。
    #
    # 判据必须是**空行**（\\n\\n），不能是"含任意 \\n"。旧写法后者恒真于一类
    # 完全合法的输入：按 40/76 列硬折行的 txt（网上 txt 最常见的排版）里，
    # 句内保留单个 \\n 是正常现象，而旧断言会把它判成"分句器跨段"并直接抛
    # ValueError，于是新增这类文档会让 `python chatbot.py process` 整个崩掉，
    # 报错还把原因误指成"sentencex 行为可能已变"。现有语料恰好没有句内换行
    # （实测句内残留换行 = 0），所以这个缺陷一直被掩盖着。
    # 注意只在 token 兜底之前断言：兜底会按 "\n" 等终止符切开长句，可能产生带换行的碎片。
    bad = next((p for p in parts if _PARAGRAPH_GAP_RE.search(p)), None)
    if bad is not None:
        raise ValueError(
            f"分句器跨段了，段落硬边界保证已失效（sentencex 行为可能已变）：{bad[:60]!r}"
        )

    # 句内软换行归一成空格。它不影响语义，但会让"一个句子"在子块文本里分裂成
    # 视觉上的两行，也会让下游按 "\n" 分级的 token 兜底把句子切碎。
    parts = [_SOFT_WRAP_RE.sub(" ", p) for p in parts]

    if tokenizer is not None and max_tokens:
        limited = []
        for p in parts:
            limited.extend(_enforce_token_limit(p, tokenizer, max_tokens))
        parts = limited
    return parts


def _hard_split_by_tokens(text, tokenizer, max_tokens):
    """最后兜底：无任何标点可用时按 token 硬切。

    **必须逐字无损。** 不能用 `tokenizer.decode(ids[a:b])` 还原切片：BERT 式
    分词器的 decode 会在 token 之间插入空格 —— 中文基本一字一 token，于是
    "二尊者即开报"被还原成"二 尊 者 即 开 报"。实测在整段无标点的长文本里
    共 9 个子块、6 个父块中招：文本被改坏会同时污染稠密向量与 BM25 稀疏向量，
    而且肉眼极难察觉（看起来只是"多了些空格"）。

    故改用 offset 映射把 token 边界还原成**原文切片**；末尾用
    `"".join(out) == text` 校验，不满足才退回 decode 兜底。
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text]

    # 优先：offset 映射 → 原文切片（无损，且不引入分词器自带的空格）
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc.get("offset_mapping")
        if offsets is not None and len(offsets) == len(ids):
            out = []
            for i in range(0, len(offsets), max_tokens):
                window = offsets[i:i + max_tokens]
                piece = text[window[0][0]:window[-1][1]]
                if piece:
                    out.append(piece)
            if out and "".join(out) == text:
                return out
    except Exception:
        # 分词器不支持 offset 映射（非 fast tokenizer）时走下面的兜底
        pass

    # 兜底：仍按 token 切片 decode（可能丢空白），但保证不超 token 上限
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
    global _EMBED_MODEL, _EMBED_TOKENIZER
    if _EMBED_MODEL is None:
        # 语义分块嵌入设备自选（MPS 优先），构造统一走 load_embedding_model
        _EMBED_TOKENIZER, _EMBED_MODEL, _ = load_embedding_model(device=SEMANTIC_EMBED_DEVICE)
    return _EMBED_TOKENIZER, _EMBED_MODEL


def _mps_empty_cache():
    """每批后归还 MPS 缓存池；失败只告警（不影响结果）。"""
    import torch
    try:
        torch.mps.empty_cache()
    except Exception as exc:
        log(f"    ⚠️  mps.empty_cache() 失败（忽略，不影响结果）: {type(exc).__name__}: {exc}")


def _encode_sentences(sentences):
    import numpy as np
    import torch
    tokenizer, model = _get_embed_model()
    device = next(model.parameters()).device
    vecs = []
    for i in range(0, len(sentences), EMBED_BATCH_SIZE):
        vecs.append(_embed_batch(sentences[i:i + EMBED_BATCH_SIZE], tokenizer, model, device))
        if device.type == "mps":
            _mps_empty_cache()
    if not vecs:
        return np.zeros((0, 0), dtype="float32")
    return torch.cat(vecs).cpu().numpy()


def _semantic_split(text, threshold=SEMANTIC_THRESHOLD):
    """语义分块：在语义边界处切分，并保证每块不小于 SEMANTIC_MIN_CHARS 字符。

    最小长度下限是必需的：叙事文本相邻句余弦相似度天然偏低（常低于 0.6），
    只看阈值会把每 1~2 句切成一块（实测平均 30 字），等于没有分块。

    不收 tokenizer：语义定界只用句级小批量嵌入（_encode_sentences 内部自取模型），
    逐句 token 上限兜底在下游 _split_by_tokens / _split_into_children 里做。旧签名
    收下它却从不使用，是死参数。
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

    # 小块并入相邻块。SEMANTIC_MIN_CHARS 只在"切分之前"检查**当前**块长度，
    # 末尾剩下的残渣不受它保护 —— 实测语料里每个文档都会出现几十字的尾块。
    # 并入后可能超过 PARENT_MAX_TOKENS，但那由下游 _split_by_tokens 的平衡切分兜住，
    # 不会再退回成短块；而留在这里的小块会变成一个没有上下文的父块。
    return _merge_small_texts(chunks, SEMANTIC_MIN_CHARS)


def _merge_small_texts(chunks, min_chars):
    """把长度低于 min_chars 的块并入相邻块，返回新列表。

    两条不变量：
      * **不丢内容** —— 合并只做拼接，任何输入都不会被丢弃；
      * **不产生更小的块** —— 合并只会让块变长。

    先向前合并（小块并入它的前一块），首块没有前一块则并入后一块；
    最后再兜一次尾块。之所以要"兜多次"：合并本身可能让一个原本达标的
    相邻块变成新的末尾小块，一次遍历处理不干净。
    """
    if len(chunks) <= 1:
        return list(chunks)

    merged = list(chunks)
    i = 0
    while i < len(merged) - 1:
        if len(merged[i]) < min_chars:
            merged[i] = merged[i] + " " + merged[i + 1]
            del merged[i + 1]
        else:
            i += 1
    if len(merged) > 1 and len(merged[-1]) < min_chars:
        merged[-2] = merged[-2] + " " + merged[-1]
        merged.pop()
    return merged


def _hierarchical_split(text, tokenizer):
    """语义分块 + 分层组织：语义分块决定边界，父子结构提供上下文。

    返回 [(child_text, parent_text), ...]。按块长度分三档：
    不超过子块上限时父子相同；不超过父块上限时整块作父块、内部按句切子块；
    再长则先切父块、每个父块内部再切子块。
    """
    results = []
    for chunk in _semantic_split(text):
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))

        if chunk_tokens <= CHILD_MAX_TOKENS:
            results.append((chunk, chunk))
        elif chunk_tokens <= PARENT_MAX_TOKENS:
            for child in _split_into_children(chunk, tokenizer):
                results.append((child, chunk))
        else:
            for parent_text in _split_by_tokens(chunk, tokenizer, PARENT_MAX_TOKENS):
                for child in _split_into_children(parent_text, tokenizer):
                    results.append((child, parent_text))

    return results


def _group_tokens(group, tok):
    """一个句子下标分组的 token 总数。"""
    return sum(tok[i] for i in group)


def _split_two(indices, tok, max_tokens):
    """把一组句子下标尽量均分成两组，**且两组都不超上限**。返回 (前, 后)。

    用于"短块既并不过去、也并不过来"时的重新配平。

    **实现要点（修过一个真 bug）**：旧实现只保证前半不超上限，然后把后半
    原样返回、从不校验。实测可复现：句子长度
    [14, 62, 346, 97, 58, 34]（合计 611，上限 512），按"装到 target=305 就切"
    得到 [346] + [97,58,34]=189，两次配平后最终产出 **[2,3,4,5] = 535 > 512**
    的父块（一次全量重建里共 31 条这样的父块、3 条超 128 的子块，而旧实现是 0 条）。
    后果有两层：L1 的 512/128 上限不变量被破坏；超限子块在 embedding 时被
    **静默截断**，检索到的其实是"半句话"——正是本项目一直在防的那类静默失效。

    因此改为**枚举所有合法切点**（前后都不超上限），再取最接近均分的那一个；
    没有任何合法切点时返回 (indices, [])，调用方据此放弃改写（而不是
    产出一个越界的组）。
    """
    total = _group_tokens(indices, tok)
    if len(indices) < 2:
        return list(indices), []

    best_pos, best_gap = None, None
    head_tokens = 0
    for pos in range(1, len(indices)):
        head_tokens += tok[indices[pos - 1]]
        tail_tokens = total - head_tokens
        if head_tokens > max_tokens or tail_tokens > max_tokens:
            continue
        gap = abs(head_tokens - total / 2)
        if best_gap is None or gap < best_gap:
            best_pos, best_gap = pos, gap
    if best_pos is None:
        return list(indices), []
    return indices[:best_pos], indices[best_pos:]


def _enforce_group_cap(groups, tok, max_tokens):
    """最后一道防线：把任何仍然超上限的分组强行切开。

    为什么必须有它（而不是只修 `_split_two`）：上限是**硬不变量**（embedding
    与 rerank 都按它截断），而下限只是质量偏好。两者会互相拉扯 ——
    "合并小块"可能顶破上限，"遵守上限"可能留下小块。这里的取舍是明确的：
    **上限优先**。小块只是上下文略少，超限则是静默截断，后者更糟。
    分组本身若只有一句且超限，说明它没经过逐句兜底（不该发生），此时原样保留
    而不是死循环 —— 那种情况应当由上游的 `_split_sentences` 兜底负责。
    """
    queue = [list(g) for g in groups]
    out = []
    guard = 0
    while queue:
        g = queue.pop(0)
        guard += 1
        if guard > 10 * len(groups) + 1000:
            # 防御极端输入下的死循环；真发生说明切分逻辑不收敛
            out.extend(queue + [g])
            break
        if _group_tokens(g, tok) <= max_tokens or len(g) <= 1:
            out.append(g)
            continue
        head, tail = _split_two(g, tok, max_tokens)
        if not tail:
            out.append(g)
            continue
        queue = [head, tail] + queue
    return out


def _merge_small_groups(groups, tok, max_tokens, min_tokens):
    """把低于 min_tokens 的分组并入相邻分组；必要时重新配平。

    迭代次数上界 = 分组数：每次成功改写都会让分组数减一，因此必然终止；
    上界只是防住"重新配平后形态没变"这种原地打转。
    """
    if len(groups) <= 1 or min_tokens <= 0:
        return groups
    groups = [list(g) for g in groups]

    for _ in range(len(groups)):
        changed = False
        for i in range(len(groups) - 1, -1, -1):
            cur = _group_tokens(groups[i], tok)
            if cur >= min_tokens:
                continue
            # 优先并入前一组（保持阅读顺序，且尾巴块的问题绝大多数在末尾）
            if i > 0 and _group_tokens(groups[i - 1], tok) + cur <= max_tokens:
                groups[i - 1] = groups[i - 1] + groups[i]
                del groups[i]
                changed = True
                break
            # 其次并入后一组（小块出现在开头时）
            if (i < len(groups) - 1
                    and cur + _group_tokens(groups[i + 1], tok) <= max_tokens):
                groups[i] = groups[i] + groups[i + 1]
                del groups[i + 1]
                changed = True
                break
            # 两边都放不下：与前一组合并后重新均分
            if i > 0:
                merged = groups[i - 1] + groups[i]
                head, tail = _split_two(merged, tok, max_tokens)
                if tail and head != groups[i - 1]:
                    groups[i - 1:i + 1] = [head, tail]
                    changed = True
                    break
        if not changed:
            break

    return groups


def _balanced_pack_groups(tok, max_tokens, min_tokens):
    """把 token 计数序列切成若干分组（返回下标列表的列表）。

    先按总量算出块数 n = ceil(total / max_tokens)，再以 total / n 为目标长度
    逐句装填，最后调用 _merge_small_groups 收拾残块。每块因而天然 ≤ max_tokens。
    """
    n = len(tok)
    if n == 0:
        return []
    total = sum(tok)
    if total <= max_tokens:
        return [list(range(n))]

    parts = max(1, -(-total // max_tokens))  # ceil 除法
    target = total / parts

    groups, cur, cur_tokens = [], [], 0
    for i, t in enumerate(tok):
        if cur and cur_tokens + t > target:
            groups.append(cur)
            cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += t
    if cur:
        groups.append(cur)

    groups = _merge_small_groups(groups, tok, max_tokens, min_tokens)
    # 上限优先于下限：合并小块的过程有可能顶破上限，这里兜住
    return _enforce_group_cap(groups, tok, max_tokens)


def _split_by_tokens(text, tokenizer, max_tokens, min_tokens=None):
    """按 token 数分割文本，保持句子完整性。

    逐句先做 token 上限兜底：单个超长句（无标点的长文本段）若不先切开，
    会被原样当成一个父块，直接突破 max_tokens。
    切分用平衡打包，故不会留下远短于 max_tokens 的尾块（见本节顶部注释）。
    """
    sentences = _split_sentences(text, tokenizer, max_tokens)
    if not sentences:
        return []
    if min_tokens is None:
        min_tokens = PARENT_MIN_TOKENS
    tok = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]
    groups = _balanced_pack_groups(tok, max_tokens, min(max_tokens, min_tokens))
    return [" ".join(sentences[i] for i in group) for group in groups]


def _split_into_children(parent_text, tokenizer):
    """将父块分割成子块，带重叠。

    逐句先做 CHILD_MAX_TOKENS 兜底：这是"子块不超限"的实际保证点 ——
    语料中最长单句约 670 字（整段无标点的长文），不兜底就会产出
    远超 128 token 的子块，embedding 时被 tokenizer 静默截断，
    检索到的其实是"半句话"。

    与旧实现的两点差别（都是有意的）：
      1. 切分同样改为平衡打包，故不会出现"尾子块只有十几个 token"；
      2. 重叠取自**上一个非重叠分组**的尾部，而不是上一个已带重叠的子块。
         旧写法会让重叠逐块累积（第 3 块的重叠里含第 1 块的尾巴），
         既不可预测又白占子块额度。
    """
    sentences = _split_sentences(parent_text, tokenizer, CHILD_MAX_TOKENS)
    if not sentences:
        return []
    tok = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]
    groups = _balanced_pack_groups(
        tok, CHILD_MAX_TOKENS, min(CHILD_MAX_TOKENS, CHILD_MIN_TOKENS)
    )

    children = []
    for gi, group in enumerate(groups):
        if gi == 0:
            current = list(group)
        else:
            group_tokens = _group_tokens(group, tok)
            # 重叠：从上一个非重叠分组尾部回填 CHUNK_OVERLAP token
            overlap, overlap_tokens = [], 0
            for j in reversed(groups[gi - 1]):
                if overlap_tokens + tok[j] > CHUNK_OVERLAP:
                    break
                overlap.insert(0, j)
                overlap_tokens += tok[j]
            # 重叠不能把子块顶过上限：兜底后的长句可达 128 token，叠加 32 token
            # 重叠即 160（实测会有 254 个子块因此超限）。裁剪时从最旧的重叠句
            # 丢起，保留离新句最近的上下文。
            while overlap and overlap_tokens + group_tokens > CHILD_MAX_TOKENS:
                oldest = overlap.pop(0)
                overlap_tokens -= tok[oldest]
            current = overlap + list(group)
        children.append(" ".join(sentences[i] for i in current))

    return children


def read_book_text(filepath):
    """严格解码文档文本；遇到非法 UTF-8 字节时精确报错，并降级为 U+FFFD。

    不用 errors="ignore" 静默丢弃坏字节（实测多份文档各有 2 处），
    否则损坏位置和内容都无从追查。这里坏字节被替换成 U+FFFD（可检测、可回溯）。

    不做 CRLF 归一：实测 sentencex 自身就把 \r\n\r\n 当句边界且输出不含 \r
    （含 8436 个 CR 的文档同样如此），全部文档的句子序列与归一前逐元素完全相同。

    最后做引号归一（NORMALIZE_QUOTES），见 fix_quotes。
    """
    with open(filepath, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        log(
            f"    ⚠️  {os.path.basename(filepath)} 非法 UTF-8 字节: "
            f"偏移 {e.start}-{e.end}，字节 {raw[e.start:e.end]!r}（已替换为 U+FFFD）"
        )
        text = raw.decode("utf-8", errors="replace")
    if NORMALIZE_QUOTES:
        text = fix_quotes(text)
    return text


read_document_text = read_book_text  # 通用别名；保留 read_book_text 以兼容分块指纹
# 语义定界依赖句子嵌入（_encode_sentences→_embed_batch），必须纳入指纹：改嵌入代码会改变
# 切分边界，漏掉就会让"切分产物过期"不被发现（_semantic_split 源码本身不含这些函数的实现）。
_CHUNKING_CODE_UNITS = ( "fix_quotes", "_split_keep", "_split_sentences", "_enforce_token_limit", "_hard_split_by_tokens", "_semantic_split", "_merge_small_texts", "_hierarchical_split", "_balanced_pack_groups", "_group_tokens", "_split_two", "_merge_small_groups", "_enforce_group_cap", "_split_by_tokens", "_split_into_children", "read_book_text", "_encode_sentences", "_embed_batch", "_get_embed_model", )


def _package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def _chunking_code_fingerprint():
    import inspect
    parts = []
    for name in _CHUNKING_CODE_UNITS:
        fn = globals().get(name)
        try:
            parts.append(f"{name}:{inspect.getsource(fn)}")
        except (OSError, TypeError):
            parts.append(f"{name}:<unavailable>")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _chunking_config_fingerprint():
    payload = { "child_max_tokens": CHILD_MAX_TOKENS, "child_min_tokens": CHILD_MIN_TOKENS, "parent_max_tokens": PARENT_MAX_TOKENS, "parent_min_tokens": PARENT_MIN_TOKENS, "chunk_overlap": CHUNK_OVERLAP, "semantic_threshold": SEMANTIC_THRESHOLD, "semantic_min_chars": SEMANTIC_MIN_CHARS, "normalize_quotes": NORMALIZE_QUOTES, "embed_model": EMBED_MODEL_PATH, "sentencex": _package_version("sentencex"), "chunking_code": _chunking_code_fingerprint(), "sentence_terminators": SENTENCE_TERMINATORS, "paragraph_gap_re": _PARAGRAPH_GAP_RE.pattern, "soft_wrap_re": _SOFT_WRAP_RE.pattern, }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], payload


def _sources_fingerprint(paths):
    out = {}
    for path in paths:
        with open(path, "rb") as f:
            key = os.path.relpath(path, DATA_DIR)
            out[key] = hashlib.sha256(f.read()).hexdigest()[:16]
    return out


def _document_paths():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"找不到数据目录: {DATA_DIR}")
    paths = sorted(
        p for p in glob.glob(os.path.join(DATA_DIR, DOC_GLOB), recursive=True)
        if os.path.isfile(p)
    )
    if not paths:
        raise FileNotFoundError(
            f"{DATA_DIR} 中没有匹配 {DOC_GLOB!r} 的文档"
        )
    return paths


def _document_name(path):
    rel = os.path.relpath(path, DATA_DIR)
    stem = os.path.splitext(rel)[0]
    name = stem.replace(os.sep, "__")
    # "__" 既用来编码路径分隔符、也可能原样出现在文件名里（a/b.txt 与 a__b.txt 会映射成
    # 同一个名字，文档缓存与 source 字段互相覆盖）。名字里出现 "__" 就追加相对路径的短
    # 哈希保证唯一；平铺语料不含 "__"，名字保持不变。
    if "__" in name:
        name = f"{name}_{hashlib.sha256(rel.encode('utf-8')).hexdigest()[:16]}"
    return name


def _doc_cache_path(source_name):
    return os.path.join(DOC_CHUNK_DIR, source_name + ".json")


def _load_doc_chunks(source_name, sha, fingerprint):
    """单文档分块缓存：内容 sha 与切分指纹一致才复用；否则重算（增删改增量的关键）。"""
    path = _doc_cache_path(source_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("source") != source_name:
            return None
        if cached.get("sha256") != sha or cached.get("chunking_fingerprint") != fingerprint:
            return None
        chunks = cached.get("chunks")
        if not isinstance(chunks, list):
            return None
        spec = {k for _f, k, _d in _PAYLOAD_SPEC}
        if chunks and set(chunks[0]) != spec:
            return None
        return chunks
    except Exception as exc:
        log(f"    ⚠️  文档缓存不可用，将重算 {source_name}: {type(exc).__name__}: {exc}")
        return None


def _save_doc_chunks(source_name, sha, fingerprint, chunks):
    try:
        os.makedirs(DOC_CHUNK_DIR, exist_ok=True)
        path = _doc_cache_path(source_name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump( { "source": source_name, "sha256": sha, "chunking_fingerprint": fingerprint, "chunks": chunks }, f, ensure_ascii=False )
        os.replace(tmp, path)
    except Exception as exc:
        log(f"    ⚠️  文档缓存写入失败（本次分块不受影响）: {type(exc).__name__}: {exc}")


def _gc_doc_chunks(keep_names):
    """删除已不存在文档的分块缓存（增删改里的「删」）；失败不影响分块。"""
    try:
        if not os.path.isdir(DOC_CHUNK_DIR):
            return
        removed = 0
        for name in os.listdir(DOC_CHUNK_DIR):
            if name.endswith(".json") and name[:-5] not in keep_names:
                os.remove(os.path.join(DOC_CHUNK_DIR, name))
                removed += 1
        if removed:
            log(f"清理 {removed} 个已删除文档的分块缓存")
    except Exception as exc:
        log(f"    ⚠️  文档缓存清理失败（不影响本次分块）: {type(exc).__name__}: {exc}")


# ── 分块产物与门禁 ──
def build_chunks():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
    paths = _document_paths()
    fingerprint, config_snapshot = _chunking_config_fingerprint()
    sources_now = _sources_fingerprint(paths)
    names_now = {_document_name(p) for p in paths}
    log(f"找到 {len(paths)} 个文档: {', '.join(os.path.basename(p) for p in paths)}")
    all_chunks = []
    reused = 0
    for i, filepath in enumerate(paths):
        source_name = _document_name(filepath)
        sha = sources_now[os.path.relpath(filepath, DATA_DIR)]
        cached = _load_doc_chunks(source_name, sha, fingerprint)
        if cached is not None:
            all_chunks.extend(cached)
            reused += 1
            log(f"[{i + 1}/{len(paths)}] 复用未变文档: {source_name}（{len(cached)} 块）")
            continue
        log(f"[{i + 1}/{len(paths)}] 处理: {source_name} ...")
        text = read_document_text(filepath)
        pairs = []
        for child_text, parent_text in _hierarchical_split(text, tokenizer):
            pairs.append((child_text, parent_text))
        kept = [(c, p) for c, p in pairs if c.strip()]
        dropped = len(pairs) - len(kept)
        if dropped:
            log(f"    ⚠️  丢弃 {dropped} 个纯空白块（不含任何文字）")
        if not kept:
            raise ValueError(f"文档 {source_name} 分块后无有效内容（所有块均为空白），请检查源文件或调整分块参数")
        parent_ids = {}
        for _, parent_text in kept:
            if parent_text not in parent_ids:
                parent_ids[parent_text] = f"{source_name}_p{len(parent_ids)}"
        doc_chunks = []
        for chunk_idx, (child_text, parent_text) in enumerate(kept):
            chunk_dict = { "child_text": child_text, "parent_text": parent_text, "chunk_index": chunk_idx, "total_chunks": len(kept), "parent_id": parent_ids[parent_text], "parent_chunk_count": 0, "contextual_text": child_text, "id": f"{source_name}_{chunk_idx}", }
            chunk_dict[SOURCE_FIELD] = source_name
            doc_chunks.append(chunk_dict)
        _save_doc_chunks(source_name, sha, fingerprint, doc_chunks)
        all_chunks.extend(doc_chunks)
    _gc_doc_chunks(names_now)
    if reused:
        log(f"增量：复用 {reused}/{len(paths)} 个未变文档，只重算了 {len(paths) - reused} 个")
    spec_keys = {k for _f, k, _d in _PAYLOAD_SPEC}
    if all_chunks and set(all_chunks[0]) != spec_keys:
        raise RuntimeError(f"分块记录字段与 _PAYLOAD_SPEC 不一致：多 {sorted(set(all_chunks[0]) - spec_keys)}、少 {sorted(spec_keys - set(all_chunks[0]))}，必须同步")
    parent_counts = Counter(c["parent_id"] for c in all_chunks)
    for c in all_chunks:
        c["parent_chunk_count"] = parent_counts[c["parent_id"]]
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    meta = { "meta_version": CHUNKS_META_VERSION, "fingerprint": fingerprint, "chunking_config": config_snapshot, "sources": sources_now, "total_chunks": len(all_chunks), "total_parents": len(parent_counts), "reused_documents": reused, "build_time": datetime.datetime.now().isoformat(timespec="seconds"), }
    with open(CHUNKS_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log("分块统计:")
    for b, cnt in sorted(Counter(c["source"] for c in all_chunks).items()):
        log(f"  {b}: {cnt} 条")
    log(f"  总计: {len(all_chunks)} 条，父块 {len(parent_counts)} 个")
    log(f"已保存: {CHUNKS_JSON}")
    log(f"元信息: {CHUNKS_META_JSON}（指纹 {fingerprint}）")
    return all_chunks


def verify_chunks_freshness():
    if not os.path.exists(CHUNKS_JSON):
        raise FileNotFoundError(f"找不到 {CHUNKS_JSON}，请先运行: python chatbot.py process")
    if not os.path.exists(CHUNKS_META_JSON):
        raise FileNotFoundError(f"找不到 {CHUNKS_META_JSON}，请重跑: python chatbot.py process")
    with open(CHUNKS_META_JSON, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("meta_version") != CHUNKS_META_VERSION:
        raise RuntimeError(f"{CHUNKS_META_JSON} 的 meta_version={meta.get('meta_version')}，当前代码是 {CHUNKS_META_VERSION}，请重跑 process")
    fingerprint, config_snapshot = _chunking_config_fingerprint()
    if meta.get("fingerprint") != fingerprint:
        diff = { k: (meta.get("chunking_config", {}).get(k), v) for k, v in config_snapshot.items() if meta.get("chunking_config", {}).get(k) != v }
        raise RuntimeError(f"分块产物配置指纹 {meta.get('fingerprint')} 与当前 {fingerprint} 不一致，差异（旧→新）: {diff}，请重跑 process")
    paths = _document_paths()
    sources_now = _sources_fingerprint(paths)
    old_sources = meta.get("sources") if "sources" in meta else meta.get("books")
    if old_sources != sources_now:
        changed = sorted( set(old_sources or {}) ^ set(sources_now) ) or [
            k for k in sources_now if (old_sources or {}).get(k) != sources_now[k]
        ]
        raise RuntimeError(f"{DATA_DIR}/ 中匹配 {DOC_GLOB!r} 的文档在分块之后发生了变化: {changed}，请重跑 process")
    return meta


# ── 词表：别名归一与停用词 ──
ALIASES_ENABLED = env_bool("RAG_ALIASES", True)
STOPWORDS_ENABLED = env_bool("RAG_STOPWORDS", True)
ALIASES_FILE = os.getenv("RAG_ALIASES_FILE") or os.path.join(LEXICON_DIR, "aliases.txt")
STOPWORDS_FILE = os.getenv("RAG_STOPWORDS_FILE") or os.path.join(LEXICON_DIR, "stopwords.txt")
if not os.path.exists(STOPWORDS_FILE):
    _legacy_stopwords = os.path.join(LEXICON_DIR, "stopwords_classical.txt")
    if os.path.exists(_legacy_stopwords):
        STOPWORDS_FILE = _legacy_stopwords
_LEXICON = None


def load_lexicon():
    global _LEXICON
    if _LEXICON is not None:
        return _LEXICON
    alias_map, stopwords = {}, set()
    if ALIASES_ENABLED:
        alias_map = _read_aliases(ALIASES_FILE)
    if STOPWORDS_ENABLED:
        stopwords = _read_stopwords(STOPWORDS_FILE)
    if alias_map:
        surfaces = sorted(alias_map, key=len, reverse=True)
        alias_re = re.compile("|".join(re.escape(s) for s in surfaces))
    else:
        alias_re = None
    _LEXICON = (alias_map, alias_re, stopwords)
    return _LEXICON


def parse_aliases(path):
    groups, problems, warnings = [], [], []
    if not os.path.exists(path):
        problems.append("文件不存在：%s" % path)
        return groups, problems, warnings
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            body = raw.split("#", 1)[0]
            if not body.strip():
                continue
            if raw.rstrip("\n").endswith("\t"):
                problems.append("第 %d 行：行尾有多余 TAB（空字段）" % lineno)
            fields = body.rstrip("\n").split("\t")
            fields = [f.strip() for f in fields]
            if any(f == "" for f in fields):
                problems.append("第 %d 行：存在空字段 %r" % (lineno, fields))
                fields = [f for f in fields if f]
            if len(fields) < 2:
                problems.append( "第 %d 行：只有规范名没有别名 %r（不产生任何归一收益）" % (lineno, fields) )
                continue
            canonical, aliases = fields[0], fields[1:]
            if len(set(aliases)) != len(aliases):
                dup = [a for a, n in Counter(aliases).items() if n > 1]
                problems.append("第 %d 行：组内别名重复 %s" % (lineno, dup))
            if canonical in aliases:
                problems.append("第 %d 行：规范名 %s 又出现在别名里" % (lineno, canonical))
            groups.append( {"canonical": canonical, "aliases": aliases, "line": lineno, "note": raw.split("#", 1)[1].strip() if "#" in raw else ""} )
    canon_seen = defaultdict(list)
    surface_owner = defaultdict(list)
    for g in groups:
        canon_seen[g["canonical"]].append(g["line"])
        for s in [g["canonical"]] + g["aliases"]:
            surface_owner[s].append(g["canonical"])
    for canon, lns in canon_seen.items():
        if len(lns) > 1:
            problems.append("规范名 %s 出现在多个组（行 %s）" % (canon, lns))
    for surface, owners in surface_owner.items():
        uniq = sorted(set(owners))
        if len(uniq) > 1:
            problems.append( "词面 %s 同时归属多个规范名 %s —— 归一目标冲突" % (surface, uniq) )
        elif len(owners) > 1:
            problems.append("词面 %s 在同一组内重复出现" % surface)
    return groups, problems, warnings


def parse_stopwords(path):
    words, problems = [], []
    if not os.path.exists(path):
        problems.append("文件不存在：%s" % path)
        return words, problems
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            word = raw.split("#", 1)[0].strip()
            if not word:
                continue
            if any(ch.isspace() for ch in word):
                problems.append("第 %d 行：一个词里出现空白字符 %r" % (lineno, word))
            words.append((word, lineno))
    seen = Counter(w for w, _ in words)
    for w, n in seen.items():
        if n > 1:
            problems.append("停用词重复：%s（%d 次）" % (w, n))
    return words, problems


def _read_aliases(path):
    """运行时词表加载：委托 parse_aliases，与校验器同一套格式。"""
    groups, problems, _warnings = parse_aliases(path)
    for problem in problems:
        log(f"⚠️  别名表 {problem}")
    mapping = {}
    for group in groups:
        for surface in [group["canonical"]] + group["aliases"]:
            mapping.setdefault(surface, group["canonical"])
    if not mapping:
        log(f"⚠️  别名表缺失/为空/无法解析，跳过别名归一: {path}")
        return {}
    log(f"别名表加载完成: {len(mapping)} 个词面 → {len(set(mapping.values()))} 个规范名")
    return mapping


def _read_stopwords(path):
    """运行时停用词加载：委托 parse_stopwords，同 _read_aliases。"""
    entries, problems = parse_stopwords(path)
    for problem in problems:
        log(f"⚠️  停用词表 {problem}")
    words = {word for word, _lineno in entries}
    if not words:
        log(f"⚠️  停用词表缺失或为空，跳过停用词过滤: {path}")
        return set()
    log(f"停用词表加载完成: {len(words)} 个词")
    return words


def lexicon_fingerprint():
    """词表与稀疏编码器指纹；任一变 → 只需 reindex-sparse，不牵连稠密与分块。"""
    parts = []
    for path in (ALIASES_FILE, STOPWORDS_FILE):
        try:
            with open(path, "rb") as f:
                parts.append(hashlib.sha256(f.read()).hexdigest())
        except OSError:
            parts.append("missing")
    # 显式加入解析后的运行时配置值，防止环境变量变更但文件内容相同时指纹不变
    parts.append(f"ALIASES_FILE={ALIASES_FILE}")
    parts.append(f"STOPWORDS_FILE={STOPWORDS_FILE}")
    parts.append(f"aliases={int(ALIASES_ENABLED)},stopwords={int(STOPWORDS_ENABLED)}")
    try:
        from importlib.metadata import version
        parts.append("jieba=" + (version("jieba") or "unknown"))
    except Exception:
        parts.append("jieba=unknown")
    parts.append("bm25=" + json.dumps(BM25_TEXT_OPTIONS, sort_keys=True))
    # BM25 模型名、分词标点集与解析/归一链路源码同样决定稀疏词空间：漏掉会让换编码后
    # 指纹不变、stages 误报 C 新鲜，稀疏索引静默错配。
    parts.append("bm25_model=" + BM25_MODEL_NAME)
    parts.append("punct=" + "".join(sorted(_PUNCT_AND_WHITESPACE)))
    import inspect
    for name in ("parse_aliases", "parse_stopwords", "_read_aliases", "_read_stopwords", "load_lexicon", "normalize_aliases", "bm25_tokenize", "sparse_encode"):
        fn = globals().get(name)
        try:
            parts.append(f"{name}:" + inspect.getsource(fn))
        except (OSError, TypeError):
            parts.append(f"{name}:<unavailable>")
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def normalize_aliases(text):
    if not text:
        return text
    alias_map, alias_re, _ = load_lexicon()
    if not alias_re:
        return text
    return alias_re.sub(lambda m: alias_map[m.group(0)], text)


def bm25_tokenize(text):
    import jieba
    _, _, stopwords = load_lexicon()
    text = normalize_aliases(text or "")
    tokens = []
    for token in jieba.cut_for_search(text):
        token = token.strip()
        if not token or all(ch in _PUNCT_AND_WHITESPACE for ch in token):
            continue
        if stopwords and token in stopwords:
            continue
        tokens.append(token)
    return " ".join(tokens)


_EMPTY_SPARSE_TEXT = "\u200b"  # bm25_tokenize 为空时 sparse_encode 的占位（见下）


def sparse_encode(text):
    from qdrant_client import models
    tokenized = bm25_tokenize(text)
    if not tokenized:
        tokenized = _EMPTY_SPARSE_TEXT  # 零宽空格，避免空文档导致 BM25 静默失败
    return models.Document(text=tokenized, model=BM25_MODEL_NAME, options=BM25_TEXT_OPTIONS)


def sparse_terms_of(text):
    """文本在稀疏通道里的**真实** term 列表（已滤掉停用词与纯标点）。

    为什么不能看 `sparse_encode` 返回的 `Document.text` 判空：空分词会被替换成零宽空格
    哨兵 `_EMPTY_SPARSE_TEXT`，而 Python 的 `str.strip()` 不认为 U+200B 是空白，于是
    哨兵让"是否为空"永远为真 —— 稀疏通道在整句命中停用词时既不产出召回、也不报错、
    也不留任何痕迹（"静默"失效原型）。这里与 sparse_encode 共用 bm25_tokenize，
    使"这个查询在稀疏通道有没有词"只有一个判据。
    """
    return bm25_tokenize(text).split()


def _index_text(rec):
    """**入库文本的唯一来源**：稠密向量、稀疏向量、稠密缓存键都取这一个字段。

    为什么必须是同一份：`contextual_text` 与 `child_text` 是两个字段（`_PAYLOAD_SPEC`
    各存一份），旧代码稠密取前者、稀疏取后者、stages 的缓存键又写成
    `contextual_text or child_text` 的兜底 —— 三份规则。两者今天逐字相同，所以差异
    一直被掩盖；一旦分叉，**同一个点的稠密与稀疏向量就描述了不同文本**，检索静默
    变差且无处可查。这里把规则收敛成一份，缺字段直接报错而不是悄悄换字段。
    """
    text = rec.get("contextual_text")
    if not isinstance(text, str):
        raise RuntimeError(
            f"分块记录缺 contextual_text（id={rec.get('id')!r}）——"
            "稠密/稀疏/缓存键都取它，请重跑 process"
        )
    return text


def _embed_cache_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_EMBED_CODE_FINGERPRINT = None


def _embed_code_fingerprint():
    """嵌入代码指纹：_embed_batch/embed 源码变化必须作废稠密向量缓存（换池化/归一化/截断等）。

    旧实现只按模型路径+精度给缓存起名，改嵌入代码会静默复用旧向量。
    """
    global _EMBED_CODE_FINGERPRINT
    if _EMBED_CODE_FINGERPRINT is None:
        import inspect
        parts = []
        for name in ("_embed_batch", "embed"):
            fn = globals().get(name)
            try:
                parts.append(f"{name}:" + inspect.getsource(fn))
            except (OSError, TypeError):
                parts.append(f"{name}:<unavailable>")
        _EMBED_CODE_FINGERPRINT = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    return _EMBED_CODE_FINGERPRINT


def _embed_cache_file(precision):
    model_key = hashlib.sha256( os.path.abspath(EMBED_MODEL_PATH).encode("utf-8") ).hexdigest()[:12]
    code_key = _embed_code_fingerprint()
    return os.path.join(EMBED_CACHE_DIR, f"{model_key}_{code_key}_{precision}.npz")


def _index_cache_precision():
    """与 build_index 相同的缓存精度选择：INDEX_DEVICE 显式指定优先，否则按设备自动。

    注意 build_index 在 MPS/CUDA 初始化失败时会回退 CPU/fp32，stages 无法预知这一回退，
    只能按常规路径预测。
    """
    device = INDEX_DEVICE or get_device()
    return "fp16" if device in ("mps", "cuda") else "fp32"


def _load_embed_cache(path, expected_dim=None):
    import numpy as np
    if not os.path.exists(path):
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = data["keys"].tolist()
            vecs = data["vectors"]
        if vecs.ndim != 2:
            raise ValueError(f"vectors 不是二维数组: shape={vecs.shape}")
        if len(keys) != vecs.shape[0]:
            raise ValueError( f"keys 与 vectors 行数不一致: {len(keys)} vs {vecs.shape[0]}")
        if vecs.shape[0] == 0:
            raise ValueError("缓存为空数组")
        if expected_dim is not None and vecs.shape[1] != expected_dim:
            raise ValueError(f"缓存向量维度 {vecs.shape[1]} 与模型期望的 {expected_dim} 不一致（多半是换过模型）")
    except Exception as exc:
        log(f"    ⚠️  向量缓存不可用，将全量重算: {type(exc).__name__}: {exc}")
        return {}
    log(f"向量缓存载入: {len(keys)} 条，{vecs.shape[1]} 维")
    return {k: vecs[i] for i, k in enumerate(keys)}


def _save_embed_cache(path, keys, matrix):
    import numpy as np
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, keys=np.array(keys, dtype="U64"), vectors=matrix)
        os.replace(tmp, path)
    except Exception as exc:
        log(f"    ⚠️  向量缓存写入失败（本次索引不受影响）: {type(exc).__name__}: {exc}")


# ── 索引构建 ──
def build_index():
    import time
    import uuid
    import numpy as np
    import torch
    from qdrant_client import QdrantClient
    from qdrant_client.models import ( VectorParams, Distance, PointStruct, SparseVectorParams, Modifier, PayloadSchemaType, CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation, )
    meta = verify_chunks_freshness()
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        all_chunks = json.load(f)
    if not all_chunks:
        raise ValueError("chunks.json 为空")
    log(f"加载分块数据: {len(all_chunks)} 条（{meta.get('total_parents', '?')} 个父块）")
    build_id = uuid.uuid4().hex
    lexicon_id = lexicon_fingerprint()
    # 记进 payload 的是**产物自己的**指纹（verify_chunks_freshness 刚核对过与当前一致），
    # 检索侧拿它与当前指纹比对，process 后忘了重跑 index 就会当场报错而不是静默用旧库。
    chunk_fingerprint = meta.get("fingerprint")
    log(f"索引建造标识: {build_id}")
    log(f"分块指纹: {chunk_fingerprint}（检索侧据此判断分块是否过期）")
    log(f"词表指纹: {lexicon_id}（别名 {ALIASES_ENABLED} / 停用词 {STOPWORDS_ENABLED}）")
    device = INDEX_DEVICE or get_device()
    use_gpu = device in ("mps", "cuda")
    tokenizer = model = None
    if use_gpu:
        try:
            tokenizer, model, _ = load_embedding_model(device=device)
            model = model.half()
        except Exception as exc:
            log(f"    ⚠️  {device.upper()} 初始化失败，回退 CPU 建索引: {type(exc).__name__}: {exc}")
            tokenizer, model, use_gpu = None, None, False
    if use_gpu:
        batch_size = EMBED_BATCH_SIZE
        cool_down_sleep = 0.1
        precision = "fp16"
        log(f"Embedding 设备: {device} (fp16, batch_size={batch_size})")
    else:
        tokenizer, model, device = load_embedding_model(device="cpu")
        torch.set_num_threads(6)
        batch_size = EMBED_BATCH_SIZE
        cool_down_sleep = 0.05
        precision = "fp32"
        log(f"Embedding 设备: cpu (6线程, batch_size={batch_size})")
    total = len(all_chunks)
    t0 = time.time()
    texts = [_index_text(c) for c in all_chunks]
    keys = [_embed_cache_key(t) for t in texts]
    cache_path = _embed_cache_file(precision)
    expected_dim = getattr(getattr(model, "config", None), "hidden_size", None)
    if expected_dim is None:
        # 回退：用一次前向传播推断维度，避免缓存维度校验被绕过
        test_vec = embed([texts[0]], tokenizer, model, device, is_query=False, autocast=use_gpu)
        expected_dim = test_vec.shape[-1]
        log(f"  推断嵌入维度: {expected_dim}")
    cache = (_load_embed_cache(cache_path, expected_dim=expected_dim) if EMBED_CACHE_ENABLED else {})
    missing = [i for i, k in enumerate(keys) if k not in cache]
    log(f"批量生成 Embedding... 命中缓存 {total - len(missing)}/{total} 条，需计算 {len(missing)} 条")
    embeddings = [cache.get(k) for k in keys] if cache else [None] * total
    if missing:
        batches = [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]
        for bi, idxs in enumerate(batches):
            batch_texts = [texts[i] for i in idxs]
            vecs = embed(batch_texts, tokenizer, model, device, is_query=False, autocast=use_gpu)
            if device == "mps":
                _mps_empty_cache()
            for row, i in enumerate(idxs):
                embeddings[i] = vecs[row]
            done = min((bi + 1) * batch_size, len(missing))
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            log(f"  计算进度 {done / len(missing) * 100:5.1f}% ({done}/{len(missing)}) {speed:.0f} 条/秒")
            time.sleep(cool_down_sleep)
    holes = [i for i, e in enumerate(embeddings) if e is None]
    if holes:
        raise RuntimeError(f"{len(holes)} 个槽位没有被填充（例如 {holes[:5]}）：缓存命中数与待计算集合不一致")
    dims = {int(np.asarray(e).shape[-1]) for e in embeddings}
    if len(dims) != 1:
        raise RuntimeError(f"向量维度不一致: {sorted(dims)} —— 通常意味着缓存里混入了别的模型的向量，请删掉 {cache_path} 后重跑或设 RAG_EMBED_CACHE=0")
    embeddings = np.vstack([np.asarray(e, dtype="float32") for e in embeddings])
    assert embeddings.shape[0] == total, f"向量条数 {embeddings.shape[0]} 与分块数 {total} 不一致"
    if EMBED_CACHE_ENABLED:
        _save_embed_cache(cache_path, keys, embeddings)
        log(f"向量缓存已更新: {cache_path}")
    del model, tokenizer
    if device == "mps":
        _mps_empty_cache()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    alias_ok = USE_COLLECTION_ALIAS
    if alias_ok:
        try:
            client.get_aliases()
        except Exception as exc:
            log(f"    ⚠️  Qdrant 不支持集合别名，回退为 delete+create 全量重建: {exc}")
            alias_ok = False
    old_collection = None
    target_collection = f"{COLLECTION_NAME}__{build_id[:8]}" if alias_ok else COLLECTION_NAME
    if not alias_ok:
        try:
            client.delete_collection(collection_name=target_collection)
        except Exception:
            pass
    log(f"写入 Qdrant (稠密+稀疏混合) → 集合 {target_collection}")
    client.create_collection( collection_name=target_collection, vectors_config={ DENSE_VECTOR_NAME: VectorParams( size=embeddings.shape[1], distance=Distance.COSINE, ) }, sparse_vectors_config={ SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF) }, )
    for field, schema in ( (SOURCE_FIELD, PayloadSchemaType.KEYWORD), ("parent_id", PayloadSchemaType.KEYWORD), ):
        try:
            client.create_payload_index( collection_name=target_collection, field_name=field, field_schema=schema )
        except Exception as exc:
            log(f"    ⚠️  payload 索引 {field} 创建失败（过滤会退化为扫描）: {exc}")
    write_batch = 1000
    for i in range(0, total, write_batch):
        batch_chunks = all_chunks[i:i + write_batch]
        batch_embeddings = embeddings[i:i + write_batch]
        points = []
        for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
            # 稀疏与稠密取同一份文本（_index_text），不让两条通道描述不同内容
            points.append(PointStruct( id=i + j, vector={ DENSE_VECTOR_NAME: emb.tolist(), SPARSE_VECTOR_NAME: sparse_encode(_index_text(chunk)), }, payload=chunk_payload(chunk, build_id, lexicon_id, chunk_fingerprint), ))
        client.upsert(collection_name=target_collection, points=points)
        done = min(i + write_batch, total)
        if (i // write_batch) % 5 == 0 or done >= total:
            log(f"  Qdrant 写入进度: {done}/{total}")
    info = client.get_collection(collection_name=target_collection)
    # 不变量「索引点数 = 产物记录数」在这里当场把关，不只靠 stages/health 事后发现：
    # 非别名路径下 delete_collection 被 except: pass 吞掉时，旧集合点数多于本次 total
    # 会留下 id >= total 的残留点（写入只覆盖 0..total-1），而它们永远不会被检索命中。
    written = client.count(collection_name=target_collection, exact=True).count
    log(f"  Qdrant 写入完成: {written} 条")
    if written != total:
        raise RuntimeError(
            f"集合 {target_collection} 点数 {written} != 分块记录数 {total}："
            "非别名模式下旧集合可能未被删净（残留 id 检索不到）；"
            "请删掉该集合后重跑 index，或启用 RAG_USE_COLLECTION_ALIAS"
        )
    coll_dense_size = next(iter(info.config.params.vectors.values())).size
    assert coll_dense_size == embeddings.shape[1], f"集合稠密维度 {coll_dense_size} 与本次编码维度 {embeddings.shape[1]} 不一致（集合名 {target_collection}）"
    log(f"  集合向量空间: 稠密 {list(info.config.params.vectors.keys())} (size={coll_dense_size}) + 稀疏 {list(info.config.params.sparse_vectors.keys())}")
    log("  索引构建完成: 稠密向量 + 稀疏向量（jieba 分词 + Qdrant 内置 BM25 打分）")
    if alias_ok:
        # 别名切换存在竞态窗口：get_aliases 到 update_collection_aliases 间其他进程可能修改别名。
        # 采用重试 + 验证策略：最多重试 3 次，每次切换后验证别名确实指向新集合。
        max_retries = 3
        for attempt in range(max_retries):
            try:
                old_collection = next(
                    (a.collection_name for a in client.get_aliases().aliases if a.alias_name == COLLECTION_ALIAS),
                    None,
                )
                # 空别名冷启动不能对不存在的 alias 提交 Delete（部分 Qdrant 版本整批失败）；有旧别名才删，Create 恒提交。
                ops = []
                if old_collection is not None:
                    ops.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=COLLECTION_ALIAS)))
                ops.append(CreateAliasOperation(create_alias=CreateAlias(collection_name=target_collection, alias_name=COLLECTION_ALIAS)))
                client.update_collection_aliases(change_aliases_operations=ops)
                # 验证别名切换成功
                aliases_after = client.get_aliases().aliases
                new_target = next((a.collection_name for a in aliases_after if a.alias_name == COLLECTION_ALIAS), None)
                if new_target == target_collection:
                    log(f"  别名已切换: {COLLECTION_ALIAS} → {target_collection}" + (f"（原 {old_collection}）" if old_collection else "（冷启动，无旧别名）"))
                    break
                else:
                    log(f"  ⚠️  别名切换验证失败（尝试 {attempt + 1}/{max_retries}）：别名指向 {new_target}，期望 {target_collection}，重试中...")
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"别名切换失败：重试 {max_retries} 次后别名仍指向 {new_target}，期望 {target_collection}")
            except Exception as exc:
                log(f"  ⚠️  别名切换异常（尝试 {attempt + 1}/{max_retries}）：{exc}，重试中...")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"别名切换失败：新集合 {target_collection} 已建好但检索端仍指向旧集合，请检查 Qdrant 别名支持: {exc}") from exc
                time.sleep(0.5 * (attempt + 1))
        if old_collection and old_collection != target_collection:
            try:
                client.delete_collection(collection_name=old_collection)
                log(f"  旧集合已清理: {old_collection}")
            except Exception as exc:
                log(f"    ⚠️  旧集合 {old_collection} 删除失败（仅占磁盘，检索不受影响）: {exc}")
        try:
            stragglers = [ x.name for x in client.get_collections().collections if x.name != target_collection and x.name.split("__")[0] == COLLECTION_NAME ]
        except Exception:
            stragglers = []
        if stragglers:
            log(f"  ⚠️  检测到 {len(stragglers)} 个同名前缀历史集合（检索已不读）: {stragglers}")
            log("     确认新索引可用后可回收磁盘（保留仅占空间）：")
            for name in stragglers:
                log(f"       curl -X DELETE {QDRANT_HOST}:{QDRANT_PORT}/collections/{name}")
    else:
        log(f"  ⚠️  未使用别名：检索端读 {COLLECTION_NAME}，重建期间该集合会短暂不可用")
    log(f"  索引建造标识: {build_id}")
    log(f"总耗时: {time.time() - t0:.1f} 秒")


# ── 检索原语 ──
def rerank_indices(query, documents, tokenizer, model, device, top_k=TOP_K, batch_size=RERANK_BATCH, max_length=RERANK_MAX_LENGTH):
    import torch
    import numpy as np
    if not documents:
        return []
    all_scores = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        pairs = [(query, d) for d in batch]
        inputs = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
        all_scores.extend(logits.float().cpu().numpy().tolist())
    order = np.argsort(all_scores)[::-1]
    ranked = [(int(i), float(all_scores[i])) for i in order]
    return ranked[:top_k]


def weighted_rrf(rankings, k=RRF_K, limit=None):
    scores, resources, first_seen = {}, {}, {}
    seq = 0
    for name, ids, weight in rankings:
        if not weight:
            continue
        for rank, doc_id in enumerate(ids, 1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
                resources[doc_id] = []
                first_seen[doc_id] = seq
                seq += 1
            scores[doc_id] += weight / (k + rank)
            resources[doc_id].append(f"{name}#{rank}")
    items = sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    if limit is not None:
        items = items[:limit]
    return [(doc_id, score, resources[doc_id]) for doc_id, score in items]


def dual_channel_recall(client, collection, dense_query, sparse_query, limit, query_filter=None, with_payload=False, tolerate_sparse_error=False, use_dense=True, use_sparse=True):
    dense_hits = []
    if use_dense:
        dense_hits = client.query_points( collection_name=collection, query=dense_query, using=DENSE_VECTOR_NAME, limit=limit, with_payload=with_payload, query_filter=query_filter, ).points
    if not use_sparse:
        return dense_hits, [], ""
    sparse_hits = []
    try:
        sparse_hits = client.query_points( collection_name=collection, query=sparse_query, using=SPARSE_VECTOR_NAME, limit=limit, with_payload=with_payload, query_filter=query_filter, ).points
    except Exception as exc:
        if not tolerate_sparse_error:
            raise
        term_count = len((sparse_query.text or '').split())
        note = f"稀疏通道召回失败，本次退化为纯稠密检索: {type(exc).__name__}: {exc}；分词 {term_count} 个 term"
        return dense_hits, [], note
    return dense_hits, sparse_hits, ""


def _degraded(steps, note):
    log(f"⚠️  {note}")
    if steps is not None:
        steps.setdefault(STEP_DEGRADE, []).append(note)


def channel_rankings(dense_hits, sparse_hits, suffix=""):
    return [ (name, ids, weight) for name, ids, weight in ( (f"dense{suffix}", [p.id for p in dense_hits], RRF_DENSE_WEIGHT), (f"sparse{suffix}", [p.id for p in sparse_hits], RRF_SPARSE_WEIGHT), ) if weight ]


def rerank_text_for(candidate):
    if RERANK_ON == "parent":
        return candidate.get("parent_text") or candidate.get("child_text", "")
    return candidate.get("contextual_text") or candidate.get("child_text", "")


def dedup_candidates_by_parent(candidates):
    kept, folded, seen = [], [], set()
    for c in candidates:
        key = c.get("parent_id") or c.get("parent_text") or c.get("id")
        if key in seen:
            folded.append(c)
            continue
        seen.add(key)
        kept.append(c)
    return kept, folded


def confidence_signal(scores, sources=None):
    """检索分数概览：只统计、不判定（不产生拒答）。"""
    vals = [float(s) for s in scores]
    n_sources = len({s for s in (sources or []) if s}) if sources is not None else None
    if not vals:
        return {"top": None, "median_rest": None, "spread": None, "ratio": None, "mean": None, "min": None, "n": 0, "n_sources": n_sources}
    top = max(vals)
    rest = sorted(vals)[:-1]
    if rest:
        mid = len(rest) // 2
        median_rest = (rest[mid] if len(rest) % 2 else (rest[mid - 1] + rest[mid]) / 2)
    else:
        median_rest = top
    spread = top - median_rest
    mean = sum(vals) / len(vals)
    return { "top": top, "median_rest": median_rest, "spread": spread, "ratio": spread / (abs(top) + 1.0), "mean": mean, "min": min(vals), "n": len(vals), "n_sources": n_sources }


def results_confidence(results):
    return confidence_signal( [r.get("rerank_score") for r in results], [_source_of(r) for r in results], )


def resolve_collection_name(client):
    if not USE_COLLECTION_ALIAS:
        return COLLECTION_NAME
    try:
        aliases = client.get_aliases().aliases
    except Exception as exc:
        log(f"⚠️  读取集合别名失败，回退直接访问 {COLLECTION_NAME}: {exc}")
        return COLLECTION_NAME
    names = {a.alias_name for a in aliases}
    if COLLECTION_ALIAS in names:
        return COLLECTION_ALIAS
    return COLLECTION_NAME


# ── 生成装配：上下文、预算、裁剪 ──
def format_context(results):
    """送进模型的资料正文：只拼检索到的 parent_text 原文，无编号、无来源标签、无说明文字。"""
    return "\n\n".join((r.get("parent_text") or "") for r in results)


def context_sources(results):
    out = []
    for i, r in enumerate(results, 1):
        source = _source_of(r)
        row = { "n": i, "id": r.get("id", ""), "rerank_score": r.get("rerank_score"), "source": source, "source_label": SOURCE_LABEL }
        if SOURCE_FIELD != "source":
            row[SOURCE_FIELD] = source
        out.append(row)
    return out


# 本部署刻意不给 system 提示词：发给模型的消息只有检索资料与本轮提问，全部是参数化内容。
# _DOMAIN_LINE 仍按领域配置计算，恢复规则时把它（+规则行）赋给 SYSTEM_RULES_PROMPT 即可。
_DOMAIN_LINE = f"你是一个面向「{DOMAIN_NAME}」的知识库助手。"
if DOMAIN_DESCRIPTION:
    _DOMAIN_LINE += f"领域说明：{DOMAIN_DESCRIPTION}。"
SYSTEM_RULES_PROMPT = ""
CONTEXT_ROLE = env_choice("RAG_CONTEXT_ROLE", "user", ("user", "system"))  # 见 build_generation_messages：system 档并入首条 system；tool 档无 tool_call_id 配对，恒为孤儿消息，故不接受
RAG_USE_CONTEXT = env_bool("RAG_USE_CONTEXT", True)


def build_system_prompt(use_context=True):
    """system 只有一条路径：开上下文注入给规则，关则不给（检索为空也照常生成，无拒答分支）。"""
    return SYSTEM_RULES_PROMPT if use_context else ""


def build_context_message(results):
    """资料消息：没有结果就不发；正文是纯检索原文。"""
    return format_context(results) if results else ""
GEN_INPUT_RESERVE = env_int("RAG_INPUT_RESERVE", 256)
# 生成历史的硬上限（token）：把"历史最多塞多少"从窗口余量里独立出来。
# 0=不设上限（行为与不带该配置完全一致，只受 history_budget 的窗口余量约束）。
# 生效形式是 min(窗口余量, 本上限)：本上限只做第二道、更严的约束，永不越过窗口安全边界。
GEN_HISTORY_MAX_TOKENS = env_int_min("RAG_GEN_HISTORY_MAX_TOKENS", 0, 0)
_TRUNCATION_MARK = "……（已按上下文预算截断）"


def estimate_tokens(text):
    if not text:
        return 0
    cjk = sum(1 for ch in text if ord(ch) >= 0x2E80)
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def message_tokens(messages):
    return sum(estimate_tokens(m.get("content") or "") for m in messages)


def history_budget(num_ctx, num_predict, system="", reserve=None, extra=""):
    reserve = GEN_INPUT_RESERVE if reserve is None else reserve
    left = (int(num_ctx) - int(num_predict) - estimate_tokens(system or "") - estimate_tokens(extra or "") - int(reserve))
    return max(0, left)


def _truncate_history_to_budget(msgs, budget):
    """历史超预算：从最旧一侧直接按 token 截断到预算内，保留尽可能多的近期内容。

    只做整条丢弃 + 最旧一条按剩余额度截断正文；截断/丢弃后若首条是 assistant
    （其对应提问已被整条丢弃），继续整条丢弃，保证历史首条恒为 user —— 否则模型
    会收到一条没有对应提问的"孤儿回答"。若截断后仅剩截断标记，视为无效并整条丢弃。
    就地修改 msgs（调用方已持有副本）。
    返回 (丢弃条数, 是否截断, 被截断的是否 user)。
    """
    total = message_tokens(msgs)
    dropped, truncated, truncated_user = 0, False, False
    while msgs and total > budget:
        first = msgs[0]
        cost = estimate_tokens(first.get("content") or "")
        excess = total - budget
        if cost <= excess:
            total -= cost
            msgs.pop(0)
            dropped += 1
            continue
        first["content"] = _cut_to_tokens(first.get("content") or "", cost - excess)
        truncated, truncated_user = True, first.get("role") != "assistant"
        total = message_tokens(msgs)
    # 截断后若首条仅含截断标记（或为空），视为无效内容并整条丢弃
    while msgs and msgs[0].get("role") != "user":
        msgs.pop(0)
        dropped += 1
    while msgs and (not msgs[0].get("content") or msgs[0]["content"] == _TRUNCATION_MARK):
        msgs.pop(0)
        dropped += 1
        # 继续确保首条为 user
        while msgs and msgs[0].get("role") != "user":
            msgs.pop(0)
            dropped += 1
    return dropped, truncated, truncated_user


def build_generation_messages(system, history, query, max_history_tokens, context=None, context_role=None):
    budget = max(0, int(max_history_tokens))
    role = context_role or CONTEXT_ROLE
    msgs = [dict(m) for m in history_to_messages(history)]
    # 调用方可能把本轮提问也塞进 history（漏了 [:-1]），甚至连续塞了多份。必须剥干净：
    # 否则模型会收到同一问题两份。本轮提问只在末尾追加这一次。
    while True:
        msgs, dropped = strip_current_turn(msgs, query)
        if not dropped:
            break
    # 超限整条丢弃 + 最旧一条按预算截断；截断后首条历史恒为 user。
    dropped_messages, truncated, truncated_user = _truncate_history_to_budget(msgs, budget)
    context = context or ""
    ctx_tokens = estimate_tokens(context) if context else 0
    out = []
    # 资料走 system 档时必须落在首条 system：chat template 普遍只认开头的 system，
    # 排在历史之后的 system 消息会被静默丢弃（项目最忌的静默失效）。合并而非新增一条，
    # 避免同一角色出现两次。历史与本轮提问在两个分支里都原样保留。
    if context and role == "system":
        out.append({"role": "system", "content": "\n\n".join(p for p in (system, context) if p)})
    elif system:
        out.append({"role": "system", "content": system})
    out += msgs
    if context and role != "system":
        out.append({"role": role, "content": context})
    out.append({"role": "user", "content": query})
    return { "messages": out, "dropped_messages": dropped_messages, "used_tokens": message_tokens(msgs), "context_tokens": ctx_tokens, "budget": budget, "truncated": truncated, "truncated_user": truncated_user, }


def _cut_to_tokens(text, max_tokens, mark=_TRUNCATION_MARK):
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    budget = max_tokens - estimate_tokens(mark)
    if budget <= 0:
        budget, mark = max_tokens, ""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + mark


def plan_generation(results, history, query, num_ctx, num_predict, use_context=True, context_role=None, trace=True):
    system = build_system_prompt(use_context=use_context)
    context = (build_context_message(results) if use_context else "")
    # 本轮提问的 token 必须占位：短问句（≤ RAG_INPUT_RESERVE）预算逐字不变，长问句历史让位。
    q_tokens = estimate_tokens(query)
    reserve_eff = max(GEN_INPUT_RESERVE, q_tokens)
    budget = history_budget(num_ctx, num_predict, system, extra=context, reserve=reserve_eff)
    # 精确算式（不看 budget<0）：历史归零也装不下才报警，避免"归零但实际没超"的假阳性。
    over_window = (estimate_tokens(system) + estimate_tokens(context) + q_tokens > int(num_ctx) - int(num_predict))
    # 配置只做第二道上限：更严时收紧历史预算，0 或比窗口余量宽时不改变行为。
    if GEN_HISTORY_MAX_TOKENS > 0:
        budget = min(budget, GEN_HISTORY_MAX_TOKENS)
    gen = build_generation_messages( system, history, query, context=context, max_history_tokens=budget, context_role=context_role, )
    gen["query_tokens"], gen["over_window"] = q_tokens, over_window
    if over_window:
        log("⚠️  本轮提问+资料超出可用窗口（历史已全部让位仍不足）：本轮提问未截断，超出部分可能被服务端截断")
    return { "system": system, "context": context, "messages": gen["messages"], "stats": gen, "trace": (plan_trace(results, gen, use_context=use_context) if trace else {}), }


STEP_BLANK_QUERY = "空查询"
STEP_SPARSE_TERMS = "稀疏编码 (jieba + BM25)"
STEP_RECALL = "单通道召回"
STEP_RRF = "RRF 融合"
STEP_CANDIDATES = "候选落地分块表"
STEP_DEDUP = "父块去重"
STEP_RERANK = "Rerank 重排"
STEP_RESULTS = "检索汇总"
STEP_RESULT_COUNT = "检索汇总条数"
STEP_CONFIDENCE = "置信度"
STEP_CONTEXT = "上下文装配"
STEP_HISTORY_TRIM = "历史裁剪"
STEP_DEGRADE = "降级"
TRACE_NOTES = {
    "去重": "同一父块下的兄弟子块只保留融合名次最好的一个；上下文装配用的是 parent_text，不去重会让同一段父文本重复出现",
    "条数": "返回条数少于 top_k 只可能是『去重后不同父块本身不够』；不补同父子块，见 hybrid_search 末尾的说明",
}


def plan_trace(results, gen, use_context=True):
    # use_context 关闭时不许报告资料正文：steps 是对外诊断（/search 直接把它返回给
    # 客户端），报一份从未发出的原文等于让诊断与实际请求相反，且与同一条响应里的
    # context_tokens=0 自相矛盾。
    trace = { STEP_CONTEXT: { "use_context": bool(use_context), "context": (format_context(results) if use_context else ""), "sources": context_sources(results), "messages": [{"role": m["role"], "chars": len(m["content"])} for m in gen["messages"]], } }
    if gen["dropped_messages"] or gen["truncated"] or gen.get("over_window"):
        note = "历史超出上下文预算，已从最旧一侧整条丢弃/截断，截断后首条历史恒为 user；system、检索资料与本轮提问始终保留"
        if gen.get("over_window"):
            note += "。本轮提问+资料超出可用窗口：历史让位为 0 仍不足，本轮提问未截断，超出部分可能被服务端截断"
        trace[STEP_HISTORY_TRIM] = { "丢弃条数": gen["dropped_messages"], "预算 token": gen["budget"], "实用 token": gen["used_tokens"], "资料 token": gen["context_tokens"], "截断": gen["truncated"], "被截断的是 user": gen["truncated_user"], "本轮提问 token": gen.get("query_tokens"), "超出可用窗口": bool(gen.get("over_window")), "说明": note, }
    return trace


def sample_vector_coverage(client, collection, limit=100):
    pts, _ = client.scroll(collection_name=collection, limit=limit, with_payload=False, with_vectors=True)
    first_sparse = pts[0].vector.get(SPARSE_VECTOR_NAME) if pts else None
    return { "n": len(pts), "miss_dense": sum(1 for p in pts if not p.vector.get(DENSE_VECTOR_NAME)), "miss_sparse": sum(1 for p in pts if not p.vector.get(SPARSE_VECTOR_NAME)), "first_sparse_terms": len(first_sparse.indices) if first_sparse is not None else 0, }


def chunk_from_payload(payload, point_id):
    payload = payload or {}
    out = {k: payload.get(f, d) for f, k, d in _PAYLOAD_SPEC}
    # source 兼容旧索引的 book 字段（payload 里只有 book 时 _PAYLOAD_SPEC 取不到值）
    out["source"] = _source_of(payload)
    out["id"] = payload.get("chunk_id") or str(point_id)
    out["point_id"] = point_id
    return out


def _empty_search_result(steps):
    """空结果出口：只写置信度（不判拒答），按 return_steps 形态返回。"""
    if steps is not None:
        steps[STEP_CONFIDENCE] = results_confidence([])
    return ([], steps) if steps is not None else []


# ── 检索引擎 ──
class RAGEngine:
    def __init__(self, client=None, embed=None, rerank=None):
        self._embed = embed
        self._rerank = rerank
        self._client = client
        self._collection_name = COLLECTION_NAME
        self._lexicon_checked = False
        self._chunk_fp_checked = False
        self._empty_warned = False
        self._model_lock = threading.Lock()
    @property
    def collection_name(self):
        return self._collection_name
    def _get_client(self):
        from qdrant_client import QdrantClient
        if self._client is None:
            self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=60, )
        return self._client
    def _resolve_collection(self):
        self._collection_name = resolve_collection_name(self._get_client())
        return self._collection_name
    def count(self):
        client = self._get_client()
        self._resolve_collection()
        if not client.collection_exists(self._collection_name):
            return 0
        return client.count(collection_name=self._collection_name).count
    def _get_embed(self):
        with self._model_lock:
            if self._embed is None:
                self._embed = load_embedding_model()
            return self._embed
    def _get_rerank(self):
        with self._model_lock:
            if self._rerank is None:
                self._rerank = load_reranker_model()
            return self._rerank
    def _inference_lock(self):
        return self._model_lock
    def _check_chunk_freshness(self, index_fp):
        """分块新鲜度闸门（与词表闸门对称）：process 之后未重跑 index 时，检索会静默
        继续吃旧分块——stages 报过期但没人看，用户拿到的是无声的旧知识库。"""
        if index_fp is None or self._chunk_fp_checked:
            return
        self._chunk_fp_checked = True  # 只查一次：产物不在运行期热加载
        try:
            current, _ = _chunking_config_fingerprint()
        except Exception as exc:
            log(f"⚠️  无法计算当前切分指纹，跳过分块新鲜度检查: {type(exc).__name__}: {exc}")
            return
        if index_fp == current:
            return
        logger.error(
            "分块产物与索引不一致：索引 %s / 当前 %s。检索在用过期分块，请运行 `python chatbot.py process && python chatbot.py index`。",
            index_fp, current,
        )
    def _check_lexicon_consistency(self, index_lexicon_id):
        if index_lexicon_id is None or self._lexicon_checked:
            return
        self._lexicon_checked = True  # 一致与否都只查一次：词表文件不做运行期热加载
        current = lexicon_fingerprint()
        if index_lexicon_id == current:
            return
        logger.error( "词表与索引不一致：索引 %s / 当前 %s。稀疏通道词空间已错配，召回会静默下降。请运行 `python chatbot.py reindex-sparse`。", index_lexicon_id, current, )
    def _fetch_chunks(self, point_ids, collection=None):
        """按命中 id 逐次回取 payload（按 id 精确检索，不全量滚动 2 万条）。

        每次调用都向 Qdrant 取，进程内不留分块表：既不占常驻内存，也不存在"索引重建后
        旧缓存失步"这个状态点。按 id retrieve 结构性做不到部分加载——缺失的 id 由调用方
        显式计数告警，完整性由 Qdrant 本身保证。

        collection 必须由调用方传入（本轮召回用的那个集合）。这里再解析一次别名，可能撞上
        index 的原子切换而换成另一个集合：点 id 是位置整数（0..N-1），另一集合**同样存在**
        这些 id，于是 retrieve 不缺、不报错，却取回完全不相干的正文（静默错配，判据恒真）。
        """
        if not point_ids:
            return {}
        if collection is None:
            self._resolve_collection()
            collection = self._collection_name
        points = self._get_client().retrieve( collection_name=collection, ids=list(point_ids), with_payload=True, with_vectors=False, )
        if not points:
            # 命中 id 回取为空：集合为空或索引正在重建。不再用 id=0 探针判断——id 0 缺失时
            # 旧写法会把非空集合误判为空并静默丢掉全部命中。
            if not self._empty_warned:
                log(f"⚠️  集合 {collection} 按命中 id 回取为空，请确认已运行: python chatbot.py index")
                self._empty_warned = True
            return {}
        self._empty_warned = False
        self._check_lexicon_consistency((points[0].payload or {}).get(LEXICON_ID_FIELD))
        self._check_chunk_freshness((points[0].payload or {}).get(CHUNK_FINGERPRINT_FIELD))
        return {p.id: chunk_from_payload(p.payload, p.id) for p in points}
    def _source_filter(self, source):
        if not source:
            return None
        from qdrant_client import models
        return models.Filter(must=[ models.FieldCondition(key=SOURCE_FIELD, match=models.MatchValue(value=source)) ])
    def hybrid_search(self, query, top_k=TOP_K, return_steps=False, source=None, book=None):
        source = source if source is not None else book
        steps = {} if return_steps else None
        if not _has_searchable_content(query):
            if steps is not None:
                steps[STEP_BLANK_QUERY] = f"查询 {query!r} 不含可检索内容，直接返回空结果"
            return _empty_search_result(steps)
        self._resolve_collection()
        client = self._get_client()
        collection = self._collection_name
        query_filter = self._source_filter(source)
        # 稠密查询向量只在稠密通道权重非零时才计算：权重为 0 时通道整体退出（health 的
        # 口径），若这里仍无条件加载并前向嵌入模型，模型目录缺失/损坏会让本可由稀疏通道
        # 独立服务的检索整体失败，还会白白付一次冷加载。
        dense_query = None
        if RRF_DENSE_WEIGHT:
            tokenizer, model, device = self._get_embed()
            # 查询与入库同精度（MPS 上 fp16）：量纲一致、计算与显存减半
            q_emb = embed([query], tokenizer, model, device, is_query=True, autocast=_float16_ok(device))
            dense_query = q_emb[0].tolist()
        sparse_query = sparse_encode(query)
        # 判空走 sparse_terms_of：sparse_query.text 里的零宽空格是占位哨兵，不是词
        _sparse_terms = sparse_terms_of(query)
        if steps is not None:
            steps[STEP_SPARSE_TERMS] = _sparse_terms
        # 稀疏通道"存在但不产出"原来完全静默：query 全是停用词/纯标点时
        # bm25_tokenize 返回空串，dual_channel_recall 既不抛异常也不召回 → 没有降级标记，
        # 分数概览也不体现，结果看起来是正常双通道融合，实际是纯稠密。必须点名。
        if bool(RRF_SPARSE_WEIGHT) and not _sparse_terms:
            _degraded(steps, "稀疏通道：查询分词后为空（整句命中停用词或纯标点），本次退化为纯稠密检索")
        # 两种检索：稠密 + 稀疏各召回一次，随后统一加权 RRF 融合。
        dense_hits, sparse_hits, note = dual_channel_recall( client, collection, dense_query, sparse_query, RECALL_LIMIT, query_filter=query_filter, tolerate_sparse_error=True, use_dense=bool(RRF_DENSE_WEIGHT), use_sparse=bool(RRF_SPARSE_WEIGHT), )
        if note:
            _degraded(steps, note)
        rankings = channel_rankings(dense_hits, sparse_hits)
        if steps is not None:
            steps[STEP_RECALL] = [ {"channel": ch, "rank": r, "point_id": pid} for ch, pids, _w in rankings for r, pid in enumerate(pids, 1) ]
        n_channels = max(1, len(rankings))
        fuse_limit = RERANK_TOP_K * n_channels if DEDUP_BY_PARENT else RERANK_TOP_K
        fused = weighted_rrf(rankings, k=RRF_K, limit=fuse_limit)
        chunks = self._fetch_chunks([point_id for point_id, _score, _res in fused], collection) if fused else {}
        if steps is not None:
            steps[STEP_RRF] = [ { "resource": resources, "score": float(score), "id": (chunks.get(point_id) or {}).get("id", str(point_id)), "text": (chunks.get(point_id) or {}).get("child_text", ""), } for point_id, score, resources in fused ]
        candidates = []
        candidate_fused_rank = {}
        missing = 0
        for fused_rank, (point_id, _score, _res) in enumerate(fused, 1):
            if point_id in chunks:
                candidates.append(chunks[point_id])
                candidate_fused_rank[chunks[point_id]["id"]] = fused_rank
            else:
                missing += 1
        if missing:
            _degraded(steps, f"{missing} 条检索命中不在分块表中（索引可能正在重建），已跳过")
        if not candidates:
            return _empty_search_result(steps)
        if steps is not None:
            steps[STEP_CANDIDATES] = [[c["point_id"], c["id"]] for c in candidates]
        folded = []
        pre_dedup = len(candidates)
        if DEDUP_BY_PARENT:
            candidates, folded = dedup_candidates_by_parent(candidates)
        # 候选池上限的截断与父块折叠分开计数：旧写法把两者合成一个 folded，被池上限
        # 吃掉的候选既不在 kept 也不在 folded，诊断上无法区分"折叠"与"截断"。
        pool_trimmed = 0
        if len(candidates) > RERANK_TOP_K:
            pool_trimmed = len(candidates) - RERANK_TOP_K
            candidates = candidates[:RERANK_TOP_K]
        if steps is not None:
            steps[STEP_DEDUP] = { "enabled": DEDUP_BY_PARENT, "去重前候选": pre_dedup, "kept": len(candidates), "folded": len(folded) if DEDUP_BY_PARENT else 0, "folded_ids": [c["id"] for c in folded], "被候选池上限截断": pool_trimmed, "说明": (TRACE_NOTES["去重"] if DEDUP_BY_PARENT else "RAG_DEDUP_BY_PARENT=0：未做父块去重，同父兄弟子块会原样进入重排"), }
        if not candidates:
            return _empty_search_result(steps)
        if top_k > RERANK_TOP_K:
            log(f"⚠️  top_k={top_k} 超过候选池上限 {RERANK_TOP_K}，最多返回 {RERANK_TOP_K} 条（可调大 RAG_RERANK_TOP_K）")
        rr_tok, rr_model, rr_dev = self._get_rerank()
        cand_texts = [rerank_text_for(c) for c in candidates]
        rr_k = min(top_k, len(candidates))
        with self._inference_lock():
            reranked = rerank_indices(query, cand_texts, rr_tok, rr_model, rr_dev, top_k=rr_k)
        if steps is not None:
            rr_query_tokens = len(rr_tok.encode(query))
            # 使用 tokenizer 实际的特殊 token 数，而非硬编码
            special_tokens = rr_tok.num_special_tokens_to_add(pair=True)
            rr_doc_budget = max(0, RERANK_MAX_LENGTH - rr_query_tokens - special_tokens)
            rr_rows = []
            for idx, score in reranked:
                text = cand_texts[idx]
                full_tokens = len(rr_tok.encode(text))
                # text 必须是**送进 rerank 的那段**（cand_texts[idx]），不是 child_text：
                # RERANK_ON=parent 时两者不同，旧写法展示 child_text 却用 parent_text 算
                # tokens/被截断，对外 steps 自相矛盾。scored_field 说明这段来自哪个字段。
                row = { "id": candidates[idx]["id"], "fused_rank": candidate_fused_rank.get(candidates[idx]["id"]), "score": float(score), "tokens": full_tokens, "text": text, "scored_field": ("parent_text" if RERANK_ON == "parent" else "contextual_text") }
                row["送入上限"] = rr_doc_budget
                row["被截断"] = full_tokens > rr_doc_budget
                rr_rows.append(row)
            steps[STEP_RERANK] = rr_rows
        output = []
        for idx, score in reranked:
            c = candidates[idx]
            src = _source_of(c)
            item = { "id": c["id"], "child_text": c["child_text"], "parent_text": c["parent_text"], "source": src, "contextual_text": c["contextual_text"], "rerank_score": score, "parent_id": c.get("parent_id", ""), }
            if SOURCE_FIELD != "source":
                item[SOURCE_FIELD] = src
            output.append(item)
        if len(output) < top_k:
            log(f"⚠️  去重后只有 {len(output)} 个不同父块，少于 top_k={top_k}；按实返回（不补同父子块）")
        confidence = results_confidence(output)
        if steps is not None:
            steps[STEP_RESULTS] = output
            steps[STEP_RESULT_COUNT] = { "返回": len(output), "请求 top_k": top_k, "候选池": len(candidates), "说明": TRACE_NOTES["条数"], }
            steps[STEP_CONFIDENCE] = confidence
        if return_steps:
            return output, steps
        return output


# ── 单轮编排 ──
def drain_events(events, on_thinking=None, on_delta=None, on_error=None):
    """消费事件流，返回 (thinking, answer, error, done)；回调参数为 (本段, 通道累计全文)。
    词表外的 kind 直接抛错——兜底会把「词表分叉」伪装成正常结尾。"""
    acc = {EVENT_THINKING: [], EVENT_DELTA: [], EVENT_ERROR: []}
    sink = {EVENT_THINKING: on_thinking, EVENT_DELTA: on_delta, EVENT_ERROR: on_error}
    done = {}
    for kind, payload in events:
        if kind == EVENT_DONE:
            done = payload
            continue
        bucket = acc.get(kind)
        if bucket is None:
            # 词表外的 kind（含 RETRIEVAL，首事件须已被 next() 取走）= 事件契约被绕过
            raise RuntimeError(f"drain_events 收到词表外事件: {kind!r}（首事件须由消费方 next() 消费）")
        bucket.append(payload)
        callback = sink[kind]
        if callback:
            callback(payload, "".join(bucket))
    return "".join(acc[EVENT_THINKING]), "".join(acc[EVENT_DELTA]), "".join(acc[EVENT_ERROR]), done


def run_turn(engine, query, history, *, top_k=None, source=None, book=None, use_context=None, num_ctx=None, num_predict=None, trace=True, chat=None):
    source = source if source is not None else book
    chat = chat or stream_chat
    top_k = TOP_K if top_k is None else top_k
    use_context = RAG_USE_CONTEXT if use_context is None else use_context
    num_ctx = NUM_CTX if num_ctx is None else num_ctx
    num_predict = NUM_PREDICT if num_predict is None else num_predict
    started = time.time()
    if trace:
        results, steps = engine.hybrid_search( query, top_k=top_k, return_steps=True, source=source )
        confidence = steps.get(STEP_CONFIDENCE) or {}
    else:
        results = engine.hybrid_search( query, top_k=top_k, source=source )
        steps = {}
        confidence = results_confidence(results)
    elapsed = time.time() - started
    plan = plan_generation( results, history, query, num_ctx, num_predict, use_context=use_context, trace=trace, )
    steps.update(plan["trace"])
    # 唯一事件契约：首事件恒为 RETRIEVAL，之后 thinking/delta/error/done；检索为空也照常生成。
    yield EVENT_RETRIEVAL, { "query": query, "results": results, "steps": steps, "confidence": confidence, "messages": plan["messages"], "gen": plan["stats"], "elapsed": elapsed, }
    answer_chars = 0
    for kind, delta in chat(plan["messages"]):
        if kind == EVENT_DELTA:
            answer_chars += len(delta)
        elif kind == EVENT_DONE:
            delta = { **delta, "answer_chars": answer_chars, "elapsed": elapsed, "steps": steps, }
        yield kind, delta


def cmd_process():
    log("阶段一: 处理数据 (读取 + 分块)")
    build_chunks()


def cmd_index():
    log("阶段二: 向量化入库")
    build_index()


def _port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _streamlit_pids():
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-f", r"streamlit run .*chatbot\.py"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(p) for p in out.split() if p.isdigit() and int(p) != os.getpid()]


def _qdrant_gate():
    """serve/api 共用的启动门禁：连不上或集合为空则报错退出，返回 (client, coll, count)。"""
    from qdrant_client import QdrantClient
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    coll = resolve_collection_name(client)
    try:
        count = 0 if not client.collection_exists(coll) else client.count(collection_name=coll).count
    except Exception as exc:
        log(f"错误: 连不上向量数据库 {QDRANT_HOST}:{QDRANT_PORT}")
        log(f"  原因: {type(exc).__name__}: {exc}")
        log("  请先确认服务已启动: docker compose up -d")
        sys.exit(1)
    if count == 0:
        log("错误: 向量数据库为空")
        log("请先运行:")
        log("  python chatbot.py process")
        log("  python chatbot.py index")
        sys.exit(1)
    return client, coll, count


def cmd_serve():
    _client, coll, count = _qdrant_gate()
    log(f"向量数据库: {coll} {count} 条记录")
    log(f"Qdrant服务: {QDRANT_HOST}:{QDRANT_PORT}")
    if _port_in_use(8501):
        pids = _streamlit_pids()
        if not pids:
            log("错误: 端口 8501 已被其它进程占用，但没找到本项目启动的 Streamlit。")
            log("  请先确认占用者: lsof -nP -iTCP:8501 -sTCP:LISTEN")
            sys.exit(1)
        log("Streamlit 服务已在运行: http://localhost:8501")
        log("改过代码务必重启：进程里跑的是启动时导入的旧模块。")
        log(r"  重启: pkill -f 'streamlit run .*chatbot\.py' && python chatbot.py serve")
        log(f"  当前 PID: {', '.join(map(str, pids))}")
        sys.exit(0)
    log("启动 Streamlit 服务...")
    log("服务地址: http://localhost:8501  （按 Ctrl+C 停止）")
    log("")
    # 必须与 .streamlit/config.toml 的 address = "127.0.0.1" 一致。CLI 优先级高于该文件，
    # 传 "::" 会把只监听回环的配置静默改写成监听全部网卡（含公网 v6），与配置文件意图相反。
    os.execv(sys.executable, [ sys.executable, "-m", "streamlit", "run", __file__, "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true", ])


# ── 交互入口：Streamlit 主界面 ──
def _render_stored_message(msg):
    """渲染输入、输出、思维链与错误（兼容旧会话里带 error 字段的历史条目）。"""
    import streamlit as st
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and THINK and msg.get("thinking"):
            with st.expander("🧠", expanded=False):
                st.markdown(msg["thinking"])
        if msg.get("content"):
            st.markdown(msg["content"])
        elif msg["role"] == "assistant" and msg.get("error"):
            st.error(msg["error"])


def _gen_failure_hint(gen_error):
    """按错误文本给下一步动作：本地与创空间共用同一套判据，只说该错误本身指向的事实。

    顺序即优先级：具体判据在前、泛化判据在后。旧顺序把 "connection" 排在 "http 500"
    之前，含 connection 字样的 500 会被误报成"连不上生成服务"；同时整表漏掉了 404 ——
    而 entrypoint 故意把模型拉取放后台，"模型不在本地"是冷启动期最可能的一类错误。
    """
    low = (gen_error or "").lower()
    if "http 404" in low or ("not found" in low and "model" in low):
        return "生成模型不在 Ollama 里：看运行日志的 [entrypoint] 拉取行（后台拉取期间问答都会失败，拉完自动恢复）"
    if "signal: killed" in low or "out of memory" in low:
        return "生成进程被系统杀掉（内存超限）：看运行日志的 [entrypoint] [mem] 观测行，必要时调小 RAG_NUM_CTX"
    if "http 500" in low:
        return "生成服务返回 500：模型未就绪或进程已被杀，看运行日志的 [entrypoint] 与 /tmp/ollama.log"
    if "econnrefused" in low or "connection" in low or "timed out" in low:
        return "连不上生成服务：确认 ollama serve 已启动、OLLAMA_BASE_URL 可达"
    return None


def _stream_reply(events):
    """流式渲染本轮答案：生成失败与 num_predict 截断必须可见（静默失效违背项目不变量）。
    返回 (thinking, answer, gen_error, done)；done 为空即生成未正常结束（与 API 侧同判据）。"""
    import streamlit as st
    think_status, think_ph = None, None
    if THINK:
        think_status = st.status("🧠", expanded=True)
        with think_status:
            think_ph = st.empty()
    answer_ph = st.empty()
    thinking, answer, gen_error, done = drain_events(
        events,
        on_thinking=(lambda _chunk, total: think_ph.markdown(total)) if think_ph is not None else None,
        on_delta=lambda _chunk, total: answer_ph.markdown(total),
    )
    if gen_error:
        answer_ph.error(gen_error)
        hint = _gen_failure_hint(gen_error)
        if hint:
            st.caption(hint)
    elif not answer:
        answer_ph.markdown("⚠️")
    if think_status is not None:
        if thinking:
            think_ph.markdown(thinking)
        think_status.update(state="complete", expanded=False)
    if done and done.get("done_reason") == "length":
        st.warning("⚠️ 生成被 num_predict 截断（done_reason=length）：答案没写完，可调大 RAG_NUM_PREDICT。")
    return thinking, answer, gen_error, done


def _run_assistant_turn(engine, prompt, history):
    """跑一轮检索与生成并渲染；只有生成正常结束且答案非空才把助手消息写进会话历史。"""
    import streamlit as st
    try:
        events = run_turn(engine, prompt, history, top_k=TOP_K, trace=False)
        next(events)  # 首事件恒为 retrieval，不在页面上渲染
        thinking, answer, gen_error, done = _stream_reply(events)
    except Exception as exc:
        # 只打「类型: 消息」无法定位（stat(None) 这类消息看不出是哪一行）：完整调用栈必须落盘，
        # 页面上同时给出最后一帧（file:line），部署环境读不到 rag.log 时也能直接定位。
        log(f"⚠️  本轮执行异常: {type(exc).__name__}: {exc}")
        log(traceback.format_exc().rstrip())
        frames = traceback.extract_tb(exc.__traceback__)
        own = [f for f in frames if os.path.basename(f.filename) == os.path.basename(__file__)]
        hit = own[-1] if own else (frames[-1] if frames else None)
        where = f"{hit.filename}:{hit.lineno} {hit.name}" if hit else "未知"
        st.error(
            f"本轮执行异常: {type(exc).__name__}: {exc}\n"
            f"出错位置: {where}\n"
            f"完整调用栈: {os.getenv('RAG_LOG_FILE', 'rag.log')}"
        )
        return
    # 生成失败或空答案：不入历史。gen_error 也进判据（不只依赖"error 必不产出 done"
    # 这个上游契约）——一旦生产侧违约，残缺答案也不会被当成完整回复留下。
    if gen_error or not done or not answer or not answer.strip():
        if gen_error:
            log(f"⚠️  生成失败，本轮回复未写入历史: {gen_error[:300]}")
        elif not done:
            log("⚠️  生成未正常结束（无 done 事件），本轮回复未写入历史")
        else:
            log("⚠️  生成为空答案，本轮回复未写入历史")
        return
    entry = {"role": "assistant", "content": answer}
    if THINK and thinking:
        entry["thinking"] = thinking
    st.session_state.messages.append(entry)


def _st_engine():
    import streamlit as st
    return st.cache_resource(show_spinner=False)(get_engine)()


def run_streamlit():
    import streamlit as st
    st.set_page_config(layout="wide")
    engine = _st_engine()
    try:
        empty = engine.count() == 0
    except Exception as exc:
        st.error(f"无法访问向量库: {type(exc).__name__}: {exc}")
        st.stop()
    if empty:
        st.error("向量库为空，先跑：python chatbot.py process && python chatbot.py index")
        st.stop()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 复制消息列表避免迭代时修改导致的竞态（Streamlit 重跑机制下虽偶然正确，但显式复制更稳健）
    for msg in list(st.session_state.messages):
        _render_stored_message(msg)
    if prompt := st.chat_input(""):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            _run_assistant_turn(engine, prompt, st.session_state.messages[:-1])


# ── 交互入口：HTTP API ──
_ENGINE = None
_ENGINE_LOCK = threading.RLock()
_REQUEST_LOCK = threading.Lock()
MAX_BODY_BYTES = int(os.getenv("RAG_API_MAX_BODY", str(64 * 1024)))


def _warm_engine(engine):
    """后台预热：jieba 词典、嵌入/重排模型、生成模型常驻 Ollama；失败仅告警（首查询冷加载兜底）。"""
    def _warm():
        try:
            import jieba
            jieba.initialize()
            load_lexicon()
            engine._get_embed()
            engine._get_rerank()
            # 必须看状态码：对未拉取的模型 Ollama 返回 404 而 requests 不抛异常，
            # 旧写法丢弃返回值、直接打印"预热完成"。而 entrypoint 正是把生成模型拉取
            # 放在后台的，冷启动窗口内这次预热必然落在 404 上 —— 唯一的"已预热"信号
            # 成了假信号，首个真故障要等用户提问时的 500 才暴露。
            r = requests.post( API_URL, json={ "model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": False, "think": THINK, "options": {"num_predict": 1}, }, timeout=180, )
            if r.status_code != 200:
                raise RuntimeError(f"生成模型预热失败 HTTP {r.status_code}: {r.text[:200]}")
            log("模型预热完成（嵌入/重排/生成）")
        except Exception as exc:
            log(f"⚠️  预热未完成（不影响服务，首次查询冷加载）: {type(exc).__name__}: {exc}")
    threading.Thread(target=_warm, daemon=True).start()


def get_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = RAGEngine()
            if PRELOAD_MODELS:
                _warm_engine(_ENGINE)
        return _ENGINE


def _json_response(handler, status, payload, close=False):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    if close:
        handler.send_header("Connection", "close")
        handler.close_connection = True
    handler.end_headers()
    handler.wfile.write(body)


def _drain_body(handler, limit=None):
    if limit is None:
        limit = MAX_BODY_BYTES
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        length = 0
    if 0 < length <= limit:
        try:
            handler.rfile.read(length)
        except Exception:
            pass


def _read_json(handler):
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        if handler.headers.get("Transfer-Encoding"):
            raise ValueError( "不支持 Transfer-Encoding（chunked）请求体：请带 Content-Length")
        return {}
    try:
        length = int(raw_length)
    except ValueError:
        raise ValueError(f"Content-Length 非法: {raw_length!r}") from None
    if length <= 0:
        return {}
    if length > MAX_BODY_BYTES:
        raise ValueError(f"请求体过大: {length} > {MAX_BODY_BYTES} 字节")
    raw = handler.rfile.read(length)
    handler._body_consumed = True  # 读满后 body 已排空；后续失败不得二次 _drain_body
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"请求体不是合法 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return data


def _search_params(data):
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("参数 query 必填，且必须是非空字符串")
    # data.get(key, 默认) 只在**键缺失**时用默认；键存在且值为 JSON null 时拿到 None。
    # top_k 与 history/source 一样是可选字段，显式 null 应当等同"不传"（与下面三处
    # 对 None 的处理保持一致），否则 {"query":"x","top_k":null} 会莫名 400。
    top_k = data.get("top_k")
    if top_k is None:
        top_k = TOP_K
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("参数 top_k 必须是正整数")
    source = data.get("source")
    if source is None:
        source = data.get("book")  # 兼容旧客户端
    if source is not None and not isinstance(source, str):
        raise ValueError(f"参数 source 必须是字符串（兼容旧名 book；当前来源字段 {SOURCE_FIELD!r}）")
    history = data.get("history")
    if history is not None:
        if not isinstance(history, list):
            raise ValueError("参数 history 必须是消息数组")
        for i, m in enumerate(history):
            if not isinstance(m, dict):
                raise ValueError(f"history[{i}] 必须是对象（含 role 与 content）")
            role = m.get("role")
            if role not in ("user", "assistant"):
                raise ValueError( f"history[{i}].role 必须是 user 或 assistant，收到 {role!r}")
            content = m.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError( f"history[{i}].content 必须是字符串，收到 {type(content).__name__}")
    return query, top_k, source, history


def _result_json(r):
    source = _source_of(r)
    out = { "id": r.get("id", ""), "source": source, "score": r.get("rerank_score"), "text": r.get("child_text", ""), "context": r.get("parent_text", "") }
    if SOURCE_FIELD != "source":
        out[SOURCE_FIELD] = source
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "RAGKnowledgeBase/1.0"
    protocol_version = "HTTP/1.1"
    # socketserver 会把 timeout 设到 socket 上，**读和写都受管**。/ask 是 NDJSON 流式
    # 回答，单轮生成最长可到 OLLAMA_TIMEOUT（默认 180s），写端若沿用 60s 会在客户端
    # 读得慢（浏览器后台标签）时中途 socket.timeout 把流截断。故按生成超时留余量。
    timeout = int(TIMEOUT) + 120
    _body_consumed = False  # _read_json 成功排空 Content-Length 后置真，_reject 不得再读

    def handle_one_request(self):
        # handle() 在 keep-alive 连接上用**同一个实例**反复调用本方法
        # （BaseHTTPRequestHandler 的既定行为），所以每请求状态必须在这里复位：
        # 第一个成功 POST 置起的 _body_consumed 会一直为真，此后该连接上所有 _reject
        # 都跳过 _drain_body，残留 body 留在缓冲区，被下一个请求当成请求行读走。
        self._body_consumed = False
        super().handle_one_request()

    def log_message(self, fmt, *args):
        logger.info("API %s - %s", self.address_string(), fmt % args)
    def _reject(self, status, message):
        if not self._body_consumed:
            _drain_body(self)
        _json_response(self, status, {"error": message}, close=True)
    def do_GET(self):
        if self.path.split("?")[0] != "/health":
            self._reject(404, f"未知路径: {self.path}")
            return
        try:
            engine = get_engine()
            with _REQUEST_LOCK:
                count = engine.count()
                collection = engine.collection_name
        except Exception as exc:
            _json_response(self, 503, { "status": "error", "error": f"无法访问向量库: {type(exc).__name__}: {exc}", }, close=True)
            return
        _json_response(self, 200, { "status": "ok" if count else "empty", "chunks": count, "collection": collection, "model": MODEL, "ollama": OLLAMA_BASE_URL, "use_context": RAG_USE_CONTEXT, "think": THINK, })
    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/search", "/ask"):
            self._reject(404, f"未知路径: {self.path}")
            return
        try:
            data = _read_json(self)
            query, top_k, source, history = _search_params(data)
        except ValueError as exc:
            self._reject(400, str(exc))
            return
        history, history_stripped = strip_current_turn(history, query)
        if path == "/search":
            self._handle_search(query, top_k, source, history_stripped)
        else:
            self._handle_ask(query, top_k, source, history, history_stripped)
    def _handle_search(self, query, top_k, source, history_stripped=False):
        try:
            with _REQUEST_LOCK:
                engine = get_engine()
                t0 = time.time()
                results, steps = engine.hybrid_search( query, top_k=top_k, return_steps=True, source=source )
        except Exception as exc:
            _json_response(self, 500, {"error": f"检索失败: {type(exc).__name__}: {exc}"})
            return
        _json_response(self, 200, { "query": query, "source": source, "source_field": SOURCE_FIELD, "elapsed": round(time.time() - t0, 3), "confidence": steps.get(STEP_CONFIDENCE), "history_stripped": history_stripped, "results": [_result_json(r) for r in results], "steps": steps, })
    def _handle_ask(self, query, top_k, source, history, history_stripped=False):
        headers_sent = False
        t0 = time.time()
        try:
            with _REQUEST_LOCK:
                engine = get_engine()
                events = run_turn(engine, query, history, top_k=top_k, source=source, trace=False)
                _kind, payload = next(events)   # 首事件恒为 retrieval，没有第二种起始契约
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            headers_sent = True
            def emit(obj):
                self.wfile.write( (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            emit({ "type": "retrieval", "query": query, "elapsed": round(payload["elapsed"], 3), "confidence": payload["confidence"], "source": source, "source_field": SOURCE_FIELD, "sources": context_sources(payload["results"]), "history_stripped": history_stripped, "results": [_result_json(r) for r in payload["results"]], })
            gen = payload["gen"]
            if gen["dropped_messages"] or gen["truncated"] or gen.get("over_window"):
                emit({ "type": "history_trimmed", "dropped_messages": gen["dropped_messages"], "budget": gen["budget"], "used_tokens": gen["used_tokens"], "context_tokens": gen["context_tokens"], "truncated": gen["truncated"], "truncated_user": gen["truncated_user"], "query_tokens": gen.get("query_tokens"), "over_window": bool(gen.get("over_window")), })
            # 无拒答分支：首个事件恒为 retrieval，之后一律走 drain_events；检索为空也照常生成。
            _thinking, _answer, _error, done = drain_events(
                events,
                on_thinking=(lambda chunk, _total: emit({"type": "thinking", "text": chunk})) if THINK else None,
                on_delta=lambda chunk, _total: emit({"type": "delta", "text": chunk}),
                on_error=lambda chunk, _total: emit({"type": "error", "error": chunk}),
            )
            # 生成失败时 stream_chat 只产出 error、不产出 done，故必须判空
            if done:
                emit({ "type": "done", "done_reason": done.get("done_reason"), "eval_count": done.get("eval_count"), "prompt_eval_count": done.get("prompt_eval_count"), "answer_chars": done.get("answer_chars"), "search_s": round(done["elapsed"], 3), "gen_s": round(time.time() - t0 - done["elapsed"], 3), })
            else:
                # 流没走到 done：显式补一个终止 error。旧写法只发 error 就结束连接，NDJSON
                # 客户端只能靠"连接被关"判断本轮结束，无法与"正常收尾"区分。事件词表不新增
                # 类型，error 本就是异常终止的既有词；terminal 字段标明这是收尾而非增量。
                emit({ "type": "error", "terminal": True, "error": _error or "生成未正常结束（无 done 事件）" })
        except Exception as exc:
            # 头已发出后的失败（Ollama 中途断流、客户端断开、wfile broken pipe、socket 超时）
            # 必须落盘：旧写法只往一个已经坏掉的 socket 写再 pass，rag.log 里什么都没有，
            # "流断了一半"与"服务端崩了"在日志上完全无法区分。
            log(f"⚠️  /ask 流中断（响应头已发出）: {type(exc).__name__}: {exc}")
            log(traceback.format_exc().rstrip())
            if not headers_sent:
                if not self._body_consumed:
                    _drain_body(self)
                _json_response(self, 500, { "error": f"检索/装配失败: {type(exc).__name__}: {exc}"}, close=True)
                return
            try:
                if not self._body_consumed:
                    _drain_body(self)
                self.wfile.write((json.dumps( {"type": "error", "terminal": True, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


def cmd_api():
    parser = argparse.ArgumentParser(description="RAG 知识库 HTTP API")
    parser.add_argument("--host", default=os.getenv("RAG_API_HOST", "::"), help="监听地址（默认 :: 双栈，IPv4/IPv6 localhost 均可访问）")
    parser.add_argument("--port", type=int, default=int(os.getenv("RAG_API_PORT", "8000")))
    args = parser.parse_args()
    _client, _coll, count = _qdrant_gate()
    import socket as _socket
    server_cls = ThreadingHTTPServer
    if ":" in args.host:
        server_cls = type("DualStackHTTPServer", (ThreadingHTTPServer,), {"address_family": _socket.AF_INET6})
    server = server_cls((args.host, args.port), Handler)
    _shown = f"[{args.host}]" if ":" in args.host else args.host
    log(f"API 就绪: http://{_shown}:{args.port}  （{count} 条，模型 {MODEL}）")
    log("  GET  /health")
    log("  POST /search  {\"query\": \"<你的问题>\", \"top_k\": 5}")
    log("  POST /ask     {\"query\": \"<你的问题>\"}  → NDJSON 流")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# 元数据必填字段；来源字段名可配置，兼容旧索引的 book / source 写法。
REQUIRED_PAYLOAD_FIELDS = ("chunk_index", "parent_text", "contextual_text")
SOURCE_PAYLOAD_FIELDS = tuple(dict.fromkeys((SOURCE_FIELD, "source", "book")))


# ── 工具：健康检查 ──
def ok(msg):
    log(f"  \u2705 {msg}")


def bad(msg):
    log(f"  \u274c {msg}")
    return 1


def cmd_health():
    from qdrant_client import QdrantClient
    parser = argparse.ArgumentParser(description="RAG 服务健康检查")
    parser.add_argument("--offline", action="store_true", help="跳过 Ollama / 模型在线检查")
    parser.add_argument("--timeout", type=int, default=120, help="Ollama 问答超时(秒)，默认 120（冷启动需加载模型）")
    args = parser.parse_args()
    fail = 0
    log("RAG 健康检查")
    log("\n[1] 配置")
    # 模型目录判据只有一份：直接问 _require_model_dir（load_embedding_model /
    # load_reranker_model 用的同一个），避免"health 说 ✅、查询却抛 RuntimeError"。
    for _path, _what, _hint in (
        (EMBED_MODEL_PATH, "嵌入", "确认 RAG_EMBED_MODEL_PATH 指向可用的模型目录"),
        (RERANK_MODEL_PATH, "重排", "确认 RAG_RERANK_MODEL_PATH 指向可用的模型目录"),
    ):
        try:
            _require_model_dir(_path, _what, _hint)
            ok(f"{_what}模型: {_path}")
        except RuntimeError as exc:
            fail += bad(str(exc))
    # MPS 水印由 torch 读取（不是应用配置，故不进 env_declared_but_unread），但它们是
    # 本机唯一防 MPS 冻结的安全阀：只设 HIGH 不设 LOW 会让"任何触碰 MPS 的调用"抛
    # invalid low watermark ratio。删掉 .env 里任意一行都必须在这里被点名。
    _mps_hi = os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    _mps_lo = os.getenv("PYTORCH_MPS_LOW_WATERMARK_RATIO")
    if _mps_hi is not None and _mps_lo is None:
        log("  ⚠️  设了 PYTORCH_MPS_HIGH_WATERMARK_RATIO 却没有 PYTORCH_MPS_LOW_WATERMARK_RATIO："
            "PyTorch 的 LOW 默认 1.4 > HIGH，任何触碰 MPS 的调用都会抛 "
            "invalid low watermark ratio（连 torch.mps.empty_cache() 也会炸）")
    else:
        ok(f"MPS 水印: HIGH={_mps_hi or '默认'} LOW={_mps_lo or '默认'}（torch 读取，非应用配置）")
    ok(f"生成模型: {MODEL}  @  {OLLAMA_BASE_URL}")
    # 思维链开关的生效值回读：模型侧与页面/API 侧共用一个值，这里只报这一个值。
    if THINK:
        log(f"  ⚠️  思维链: 开启（RAG_THINK） —— think=true，页面/API 同步展示；num_ctx={NUM_CTX} num_predict={NUM_PREDICT} 共用")
    else:
        log(f"  ℹ️  思维链: 关闭（RAG_THINK） —— think=false，页面与 /ask 不产生思维链输出；num_ctx={NUM_CTX} num_predict={NUM_PREDICT} 固定")
    if not RAG_USE_CONTEXT:
        log("  \u26a0\ufe0f  上下文注入: 关闭（RAG_USE_CONTEXT=0） —— 不发资料消息，只发本轮提问")
    else:
        sys_note = f"规则 {len(SYSTEM_RULES_PROMPT)} 字" if SYSTEM_RULES_PROMPT else "空"
        ok(f"上下文注入: 纯资料消息(role={CONTEXT_ROLE}，只含检索 parent_text) + 本轮提问；system={sys_note}")
    if RERANK_ON == "parent" and RERANK_MAX_LENGTH < PARENT_MAX_TOKENS:
        log(f"  \u26a0\ufe0f  RERANK_ON=parent 但 RERANK_MAX_LENGTH={RERANK_MAX_LENGTH} < 父块上限 {PARENT_MAX_TOKENS}：父块尾部会被静默截掉")
    # 校验 rerank 模型实际最大长度
    try:
        from transformers import AutoTokenizer
        rr_tok = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
        model_max_len = getattr(rr_tok, "model_max_length", None)
        if model_max_len and model_max_len < 1000000:  # 忽略超大默认值（如 10^9）
            effective_max = min(RERANK_MAX_LENGTH, model_max_len)
            if RERANK_ON == "parent" and effective_max < PARENT_MAX_TOKENS:
                log(f"  \u26a0\ufe0f  Rerank 模型最大长度 {model_max_len} 导致有效上限 {effective_max} < 父块上限 {PARENT_MAX_TOKENS}：父块尾部会被静默截掉")
            elif RERANK_ON == "child" and effective_max < CHILD_MAX_TOKENS:
                log(f"  \u26a0\ufe0f  Rerank 模型最大长度 {model_max_len} 导致有效上限 {effective_max} < 子块上限 {CHILD_MAX_TOKENS}：子块可能被静默截掉")
            elif RERANK_MAX_LENGTH > model_max_len:
                log(f"  \u2139\ufe0f  RERANK_MAX_LENGTH={RERANK_MAX_LENGTH} 超过模型最大长度 {model_max_len}，实际将被截断至 {model_max_len}")
    except Exception as exc:
        log(f"  \u26a0\ufe0f  无法校验 rerank 模型最大长度: {type(exc).__name__}: {exc}")
    if not RRF_DENSE_WEIGHT:
        log("  \u2139\ufe0f  RRF_DENSE_WEIGHT=0：稠密通道整体退出")
    if not RRF_SPARSE_WEIGHT:
        log("  \u2139\ufe0f  RRF_SPARSE_WEIGHT=0：稀疏通道整体退出")
    ok(f"检索参数: top_k={TOP_K} / 候选池 {RERANK_TOP_K} / 每通道召回 {RECALL_LIMIT} / RRF k={RRF_K} 权重 稠密 {RRF_DENSE_WEIGHT} 稀疏 {RRF_SPARSE_WEIGHT} / rerank {RERANK_ON}（max_length {RERANK_MAX_LENGTH}）")
    unread = env_declared_but_unread()
    if unread:
        log(f"  \u26a0\ufe0f  以下 {len(unread)} 个 .env 变量本应用不读取（写了对本应用无效）: {unread}")
    else:
        ok("配置项检查: .env 里声明的检索/生成变量全部被代码读取")
    log("\n[2] 源文档")
    try:
        doc_paths = _document_paths()
    except Exception as exc:
        doc_paths = []
        fail += bad(f"数据目录检查失败: {type(exc).__name__}: {exc}")
    if doc_paths:
        names = [os.path.basename(p) for p in doc_paths]
        ok(f"{DATA_DIR}/ 匹配 {DOC_GLOB!r}: {len(names)} 个（{', '.join(names)}）")
    log("\n[3] 向量库 Qdrant")
    qdrant_count = 0
    client = None
    coll = None
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=60)
        coll = resolve_collection_name(client)
        if not client.collection_exists(coll):
            fail += bad( f"集合 {coll} 不存在（索引未构建），请运行: python chatbot.py index" )
        else:
            qdrant_count = client.count(collection_name=coll, exact=True).count
            if qdrant_count > 0:
                ok(f"{coll} 记录数: {qdrant_count}")
                if os.path.exists(CHUNKS_JSON):
                    try:
                        import json as _json
                        with open(CHUNKS_JSON, encoding="utf-8") as f:
                            n_chunks = len(_json.load(f))
                    except Exception as exc:
                        fail += bad(f"分块产物 {CHUNKS_JSON} 读取失败: {type(exc).__name__}: {exc} —— 请重跑 process")
                    else:
                        if n_chunks == qdrant_count:
                            ok(f"数量一致: chunks.json {n_chunks} 条 == 集合 {qdrant_count} 条")
                        else:
                            fail += bad(f"数量不一致: chunks.json {n_chunks} 条 != 集合 {qdrant_count} 条（请重跑 index）")
                else:
                    log(f"  ⏭  跳过数量一致性（找不到 {CHUNKS_JSON}）")
                sample = client.scroll( collection_name=coll, limit=1, with_payload=True, with_vectors=False )[0]
                if sample:
                    meta = sample[0].payload
                    missing = [k for k in REQUIRED_PAYLOAD_FIELDS if k not in meta]
                    if not any(f in meta for f in SOURCE_PAYLOAD_FIELDS):
                        missing.append(SOURCE_FIELD)
                    if missing:
                        fail += bad(f"元数据缺字段: {missing}")
                    else:
                        ok(f"元数据字段完整（来源字段 {SOURCE_FIELD}）")
                try:
                    pts, _ = client.scroll(collection_name=coll, limit=1, with_payload=True, with_vectors=False)
                    idx_lex = ((pts[0].payload or {}).get(LEXICON_ID_FIELD) if pts else None)
                    cur_lex = lexicon_fingerprint()
                    if idx_lex is None:
                        ok("词表指纹: 索引未记录（旧索引），跳过一致性检查")
                    elif idx_lex == cur_lex:
                        ok(f"词表指纹一致: {cur_lex}")
                    else:
                        fail += bad(f"词表与索引不一致（索引 {idx_lex} / 当前 {cur_lex}）—— 请运行: python chatbot.py reindex-sparse")
                except Exception as e:
                    fail += bad(f"词表一致性检查失败: {e}")
                # 分块新鲜度：与 stages 的 A 阶段同一判据，在检索侧提前暴露
                try:
                    pts, _ = client.scroll(collection_name=coll, limit=1, with_payload=True, with_vectors=False)
                    idx_fp = ((pts[0].payload or {}).get(CHUNK_FINGERPRINT_FIELD) if pts else None)
                    cur_fp, _ = _chunking_config_fingerprint()
                    if idx_fp is None:
                        log("  ⏭  索引未记录分块指纹（旧索引），跳过分块新鲜度检查")
                    elif idx_fp == cur_fp:
                        ok(f"分块指纹一致: {cur_fp}")
                    else:
                        fail += bad(f"分块产物与索引不一致（索引 {idx_fp} / 当前 {cur_fp}）—— 检索在用过期分块，请运行: python chatbot.py process && python chatbot.py index")
                except Exception as e:
                    fail += bad(f"分块新鲜度检查失败: {e}")
            else:
                fail += bad(f"{coll} 为空，请运行: python chatbot.py index")
    except Exception as e:
        fail += bad(f"无法连接 Qdrant Docker服务 ({QDRANT_HOST}:{QDRANT_PORT}): {e}")
    log("\n[4] 混合检索（稠密 + 稀疏）")
    if client is None or qdrant_count == 0:
        log("  ⏭  跳过（向量库不可用）")
    else:
        try:
            info = client.get_collection(collection_name=coll)
            sparse_names = list(info.config.params.sparse_vectors.keys())
            dense_names = list(info.config.params.vectors.keys())
            if sparse_names and dense_names:
                ok(f"集合配置: 稠密 {dense_names} + 稀疏 {sparse_names}")
            else:
                fail += bad(f"集合缺少向量空间: 稠密 {dense_names} / 稀疏 {sparse_names}")
            cov = sample_vector_coverage(client, coll, limit=50)
            if not cov["miss_dense"] and not cov["miss_sparse"] and cov["n"]:
                ok(f"抽样 {cov['n']} 条: 稠密/稀疏均已写入（首条稀疏 {cov['first_sparse_terms']} 个 term）")
            else:
                fail += bad(f"抽样 {cov['n']} 条中: 缺稠密 {cov['miss_dense']} 条, 缺稀疏 {cov['miss_sparse']} 条（混合检索名存实亡）")
            pts, _ = client.scroll(collection_name=coll, limit=1, with_payload=True)
            probe = (pts[0].payload.get("child_text", "")[:80] if pts else "")
            if probe:
                res = client.query_points( collection_name=coll, query=sparse_encode(probe), using=SPARSE_VECTOR_NAME, limit=3, )
                if res.points:
                    ok(f"稀疏单通道检索可用（{len(res.points)} 条命中，最高分 {res.points[0].score:.2f}）")
                else:
                    fail += bad("稀疏单通道检索返回空，稀疏索引可能未生效")
        except Exception as e:
            fail += bad(f"混合检索检查失败: {e}")
    if args.offline:
        log("\n(offline 模式：跳过 Ollama 在线检查)")
    else:
        log("\n[5] Ollama 后端")
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                ok(f"Ollama 可达，已装模型: {models}")
                # Ollama 的 /api/tags 恒返回 name:tag（无 tag 补 latest），而 MODEL 允许写成
                # 不带 tag（llama3）。末段含 ":" 才是显式 tag；否则按前缀补 latest 比对，
                # 否则明明已装也报"未安装"，把 health 引到一次根本不需要的 pull 上。
                _installed = MODEL in models
                if ":" not in MODEL.rsplit("/", 1)[-1]:
                    _installed = _installed or any(m.rsplit(":", 1)[0] == MODEL for m in models)
                if not _installed:
                    fail += bad(f"MODEL={MODEL} 未安装，请: ollama pull {MODEL}")
            else:
                fail += bad(f"Ollama /api/tags 返回 HTTP {r.status_code}")
        except Exception as e:
            fail += bad(f"Ollama 不可达: {e}")
        log(f"\n[6] 生成模型应答（think:{str(THINK).lower()} 简短问答）")
        payload = { "model": MODEL, "messages": [{"role": "user", "content": HEALTH_PROBE_QUESTION}], "stream": False, "think": THINK, "options": {"num_predict": 64}, }
        t0 = time.time()
        try:
            r = requests.post(API_URL, json=payload, timeout=args.timeout)
            elapsed = time.time() - t0
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                fail += bad(f"模型应答返回错误: {data['error']}")
            else:
                reply = (data.get("message", {}) or {}).get("content", "") if isinstance(data, dict) else ""
                ok(f"应答成功 耗时 {elapsed:.1f}s: {reply[:60]!r}")
        except requests.exceptions.Timeout:
            fail += bad(f"模型应答超时({args.timeout}s)——多半是模型冷启动（权重加载慢）：先预热或调大超时，例如:\n      curl -s {API_URL} -d '{{\"model\":\"{MODEL}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"stream\":false,\"think\":false}}' >/dev/null\n      python chatbot.py health --timeout 300")
        except Exception as e:
            fail += bad(f"模型应答失败({args.timeout}s): {type(e).__name__}: {e}")
    log("\n" + "=" * 60)
    if fail:
        log(f"健康检查失败: {fail} 项")
        sys.exit(1)
    else:
        log("健康检查通过 ✓")
        sys.exit(0)


# ── 工具：分块 A/B 比较 ──
DEFAULT_B = CHUNKS_JSON
# 字段名以产物实际结构为准；source 是来源字段，旧产物里叫 book，按 source 归一比较。
FIELDS = [ "child_text", "parent_text", "source", "chunk_index", "total_chunks", "contextual_text", "id", "parent_id", "parent_chunk_count", ]
_LEGACY_CHUNK_FIELDS = {"book"}


def load(path):
    if not os.path.exists(path):
        sys.exit(f"❌ 找不到文件: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"❌ {path} 不是 JSON 列表（chunks.json 应为记录数组）")
    return data


def show(text, limit=90):
    text = text.replace("\n", "\\n")
    return text[:limit] + ("…" if len(text) > limit else "")


def check_schema(a, b, path_a, path_b):
    seen = set()
    for rows in (a, b):
        for r in rows[:50]:
            if isinstance(r, dict):
                seen |= set(r)
    unknown = sorted(seen - set(FIELDS) - _LEGACY_CHUNK_FIELDS)
    if unknown:
        sys.exit(f"❌ 分块产物出现未登记字段 {unknown}（{path_a} / {path_b}）：比较器只遍历 FIELDS，未登记字段会被静默忽略。请先登记进 FIELDS。")


def compare_chunks_diff(a, b, allow, path_a, path_b):
    check_schema(a, b, path_a, path_b)
    log("分块结果比较")
    log(f"  A: {path_a}")
    log(f"  B: {path_b}")
    log(f"  允许不同的字段: {sorted(allow) if allow else '（无，要求逐字节相同）'}")
    ok = True
    if len(a) == len(b):
        log(f"[1] 总量: A={len(a)} 条  B={len(b)} 条  ✅")
    else:
        log(f"[1] 总量: A={len(a)} 条  B={len(b)} 条  ❌ 相差 {len(b) - len(a):+d} 条")
        ok = False
    ca, cb = Counter(_source_of(c) or "?" for c in a), Counter(_source_of(c) or "?" for c in b)
    for src in sorted(set(ca) | set(cb)):
        na, nb = ca.get(src, 0), cb.get(src, 0)
        flag = "✅" if na == nb else f"❌ 相差 {nb - na:+d}"
        log(f"      {src}: A={na} B={nb} {flag}")
        if na != nb:
            ok = False
    n = min(len(a), len(b))
    diff = {f: [] for f in FIELDS}
    for i in range(n):
        ra, rb = a[i], b[i]
        for f in FIELDS:
            va = _source_of(ra) if f == "source" else ra.get(f)
            vb = _source_of(rb) if f == "source" else rb.get(f)
            if va != vb:
                diff[f].append((i, va, vb))
    log("\n[2] 字段差异（按列表位置逐条比对）")
    for f in FIELDS:
        cnt = len(diff[f])
        if cnt == 0:
            log(f"      {f:16s} ✅ 完全一致")
            continue
        allowed = f in allow
        mark = "🟡 在允许范围内" if allowed else "❌ 不允许的改动"
        if not allowed:
            ok = False
        log(f"      {f:16s} {mark}  差异 {cnt} 条 ({cnt / n * 100:.2f}%)")
        for i, va, vb in diff[f][:3]:
            log(f"          位置 {i}: A={show(str(va))}")
            log(f"           {' ' * len(str(i))}      B={show(str(vb))}")
    if diff["child_text"]:
        pos = [x[0] for x in diff["child_text"]]
        log(f"\n[3] ⚠️  正文出现差异，首个位置 = {pos[0]}，最后 = {pos[-1]}（共 {len(pos)} 条）")
        log("     若差异连续延伸到列表末尾，说明是分块边界漂移而非局部文本差异，不应接受该改动。")
    log("\n" + "=" * 78)
    if ok:
        log("✅ 比较通过：差异全部落在允许范围内")
    else:
        log("❌ 比较不通过：出现了不允许的差异")
    return 0 if ok else 1


def cmd_compare_chunks():
    ap = argparse.ArgumentParser(description="分块结果 A/B 比较")
    ap.add_argument("a", help="基线文件（改动前存档的分块产物）")
    ap.add_argument("b", nargs="?", default=DEFAULT_B, help=f"对照文件（默认 {DEFAULT_B}）")
    ap.add_argument("--allow", action="append", default=[], help="允许不同的字段名，可重复指定，例如 --allow contextual_text")
    args = ap.parse_args()
    unknown = [f for f in args.allow if f not in FIELDS]
    if unknown:
        sys.exit(f"❌ --allow 出现未知字段: {unknown}（可用: {FIELDS}）")
    a, b = load(args.a), load(args.b)
    sys.exit(compare_chunks_diff(a, b, set(args.allow), args.a, args.b))


# ── 工具：稀疏向量重建 ──
BATCH = 1000


def cmd_reindex_sparse():
    from qdrant_client import QdrantClient, models
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=120)
    COLL = resolve_collection_name(client)
    info = client.get_collection(COLL)
    total = info.points_count
    log(f"集合 {COLL}: {total} 条（Qdrant 内置 BM25 + jieba 分词，options={BM25_TEXT_OPTIONS}）")
    if not total:
        log(f"❌ 集合 {COLL} 为空，无事可做；请先运行: python chatbot.py process && python chatbot.py index")
        return 1
    t0 = time.time()
    done = 0
    lexicon_id = lexicon_fingerprint()
    offset = None
    while True:
        pts, offset = client.scroll( COLL, limit=BATCH, offset=offset, with_payload=True, with_vectors=False, timeout=60 )
        if not pts:
            break
        ids = [p.id for p in pts]
        # 与 build_index 同一份文本（_index_text），否则重算的稀疏向量与稠密侧错配
        points = [ models.PointVectors( id=p.id, vector={SPARSE_VECTOR_NAME: sparse_encode(_index_text(p.payload or {}))}, ) for p in pts ]
        client.update_vectors(COLL, points=points)
        client.set_payload(collection_name=COLL, payload={LEXICON_ID_FIELD: lexicon_id}, points=ids)
        done += len(pts)
        log(f"  {done}/{total}  ({time.time() - t0:.1f}s)")
        if offset is None:
            break
    log(f"词表指纹已回写: {lexicon_id}（{done} 条）")
    if done != total:
        log(f"❌ 只更新了 {done}/{total} 条，请重跑")
        return 1
    sample, _ = client.scroll(COLL, limit=5, with_vectors=True, timeout=30)
    for p in sample:
        sp = p.vector.get(SPARSE_VECTOR_NAME)
        log(f"  校验 point {p.id}: 稠密 {len(p.vector.get(DENSE_VECTOR_NAME, []))} 维, 稀疏 {len(sp.indices) if sp else 0} terms")
    log(f"完成: {done} 条，耗时 {time.time() - t0:.1f}s")
    return 0


# ── 工具：词表校验 ──
# 数据目录与词表路径全部走全局配置，不写死文档名或文件名。
ALIAS_PATH = ALIASES_FILE
STOP_PATH = STOPWORDS_FILE
READ_KWARGS = dict(encoding="utf-8", errors="replace")
FORBIDDEN_NEGATION = set("不无未莫非没别勿弗毋否亡沒無別罔靡")
FORBIDDEN_DEGREE = set("很太最更极極甚颇頗稍略愈越挺蛮太殊煞")
NOISE_CHARS = set( "的了着过们这那一二三四五六七八九十百千万见说道问答叫唤令教使与和在" "是有将把被大小老好众位个名字等来去出入上下前后里外中而又也都便就却" "只遂因故然若且乃其之于以为所曰我你他她它相再从向往到至及并同连皆亦" "复可会要想知听看望走行坐立笑哭怒喜心手头身口眼声气人家儿子们兮乎者" "也矣焉哉尔汝卿彼此谁每各另别样般些点儿" )


def die(msg: str) -> None:
    log("错误：" + msg)


def load_documents():
    docs = []
    for p in _document_paths():
        with open(p, **READ_KWARGS) as fh:
            text = fh.read()
        text = text.replace("\ufffd", "")
        docs.append((_document_name(p), text))
    return docs


def raw_byte_report():
    lines = []
    for p in _document_paths():
        name = _document_name(p)
        with open(p, "rb") as fh:
            data = fh.read()
        try:
            data.decode("utf-8")
            lines.append((name, len(data), "OK"))
        except UnicodeDecodeError as exc:
            lines.append( (name, len(data), "非法字节 @ %d-%d" % (exc.start, exc.end - 1)) )
    return lines


def count_in_documents(docs, terms):
    out = {}
    for t in terms:
        out[t] = {name: text.count(t) for name, text in docs}
    return out


def longest_first_scan(docs, surfaces):
    by_first = defaultdict(list)
    for s in surfaces:
        by_first[s[0]].append(s)
    for ch in by_first:
        by_first[ch].sort(key=len, reverse=True)
    matched = Counter()
    crossing = defaultdict(Counter)
    for _name, text in docs:
        i, n = 0, len(text)
        while i < n:
            cands = by_first.get(text[i])
            hit = None
            if cands:
                for s in cands:
                    if text.startswith(s, i):
                        hit = s
                        break
            if hit is None:
                i += 1
                continue
            end = i + len(hit)
            for j in range(i + 1, min(end, n)):
                for a in by_first.get(text[j], ()):
                    if j + len(a) > end and text.startswith(a, j):
                        crossing[a][hit] += 1
                        break
            matched[hit] += 1
            i = end
    return matched, crossing


def competitor_windows(docs, surface, surface_set, limit=4, min_count=2):
    found = Counter()
    L = len(surface)
    for _name, text in docs:
        start = 0
        while True:
            idx = text.find(surface, start)
            if idx < 0:
                break
            start = idx + 1
            for ext in (1, 2):
                s = idx - ext
                if s >= 0:
                    w = text[s:idx + L]
                    extra = w[:ext]
                    if len(w) == ext + L and all( "\u4e00" <= c <= "\u9fff" for c in w ) and all(c not in NOISE_CHARS for c in extra):
                        if w not in surface_set:
                            found[w] += 1
            r = idx + L
            if r < len(text):
                w = text[idx:r + 1]
                if len(w) == L + 1 and "\u4e00" <= text[r] <= "\u9fff" and text[r] not in NOISE_CHARS and w not in surface_set:
                    found[w] += 1
    return [(w, n) for w, n in found.most_common() if n >= min_count][:limit]


def cmd_verify_lexicon():
    # main 分发统一 fn() 零参调用，并已把 rest 拼回 sys.argv = ["chatbot.py lexicon", *rest]
    argv = sys.argv[1:]
    quiet = "-q" in argv or "--quiet" in argv
    pos = [a for a in argv if not a.startswith("-")]
    alias_path = pos[0] if len(pos) > 0 else ALIAS_PATH
    stop_path = pos[1] if len(pos) > 1 else STOP_PATH
    def out(*a):
        if not quiet:
            log(*a)
    problems, warnings = [], []
    log("RAG 词表校验（%s + %s）" % (alias_path, stop_path))
    docs = load_documents()
    total_chars = sum(len(t) for _, t in docs)
    out("数据：%d 个文档，合计 %d 字符" % (len(docs), total_chars))
    for name, nbytes, status in raw_byte_report():
        out("    %-8s %9d 字节   UTF-8: %s" % (name, nbytes, status))
    out("    读取方式：utf-8 + errors='replace'（坏字节替换为 U+FFFD）")
    out("")
    out("-" * 78)
    out("[1] 别名表: %s" % alias_path)
    out("-" * 78)
    groups, ap, aw = parse_aliases(alias_path)
    problems += ap
    warnings += aw
    if not groups:
        die("别名表为空或无法解析")
        return 1
    surfaces = []
    for g in groups:
        surfaces.append(g["canonical"])
        surfaces.extend(g["aliases"])
    surface_set = set(surfaces)
    counts = count_in_documents(docs, surfaces)
    matched, crossing = longest_first_scan(docs, surfaces)
    out("别名组数：%d；词面总数：%d" % (len(groups), len(surfaces)))
    out("最长优先扫描：命中 %d 次" % sum(matched.values()))
    missing = [s for s in surfaces if sum(counts[s].values()) == 0]
    for s in missing:
        problems.append( "词面 %r 在 %s/ 中出现 0 次 —— 违反「规范名与别名都必须真实出现」" % (s, DATA_DIR) )
    out("")
    out("    规范名                次数   别名（次数）")
    out("    " + "-" * 72)
    for g in groups:
        c = sum(counts[g["canonical"]].values())
        al = "  ".join( "%s(%d)" % (a, sum(counts[a].values())) for a in g["aliases"] )
        out("    %-10s %8d   %s" % (g["canonical"], c, al))
    out("")
    out("-" * 78)
    out("[2] 子串冲突 / 遮蔽检测（对称替换 + 最长优先）")
    out("-" * 78)
    out("    独立=按最长优先扫描吃掉的次数；包含遮蔽=被更长条目吃掉的次数（同目标=安全）")
    out("    左重叠保护=更长的词面先命中，把短别名挡掉（保护性）")
    out("")
    owner_of = {}
    for g in groups:
        for s in [g["canonical"]] + g["aliases"]:
            owner_of[s] = g["canonical"]
    shadow_rows = []
    for g in groups:
        for a in g["aliases"]:
            tot = sum(counts[a].values())
            ind = matched[a]
            shadow = tot - ind
            if shadow > 0:
                contains = sorted( s for s in surface_set if s != a and a in s and len(s) > len(a) )
                overlaps = [s for s, _c in crossing[a].most_common()]
                same = all(owner_of[s] == g["canonical"] for s in contains)
                shadow_rows.append( (a, tot, shadow, ind, contains, overlaps, same, g["canonical"]) )
    if shadow_rows:
        for a, tot, shadow, ind, contains, overlaps, same, canon in shadow_rows:
            flag = "安全(同归一目标)" if same else "注意(归入别人!)"
            cross_n = sum(crossing[a].values())
            contain_n = max(shadow - cross_n, 0)
            csrc = "%d:%s" % (contain_n, "/".join(contains)) if contains else "0:-"
            osrc = ("%d:%s" % (cross_n, "/".join(overlaps))) if overlaps else "0:-"
            out("    %-8s 总%5d 独立%5d 遮蔽%5d | 包含 %-22s | 左重叠保护 %-16s %s" % (a, tot, ind, shadow, csrc, osrc, flag))
            if not same:
                warnings.append( "%s 的 %d 次出现被更长的条目 %s 先匹配走（分属 %s），不会再落到 %s，请确认更长条目归属" % (a, contain_n, "/".join(contains), "/".join(sorted({owner_of[s] for s in contains})), canon) )
    else:
        out("    （无：本表不含「别名是另一词面子串」的情况）")
    out("")
    out("    竞争窗口（不在词典、但包含某别名的更长汉字串，>=2 次才列出；启发式人工复核）")
    out("          判定原则：更长串与短别名同指=安全，异指则需剔除或加保护。")
    any_comp = False
    for g in groups:
        for a in g["aliases"]:
            wins = competitor_windows(docs, a, surface_set)
            if wins:
                any_comp = True
                out("      %-8s -> %s" % (a, "  ".join("%s(%d)" % w for w in wins)))
    if not any_comp:
        out("      （无）")
    out("")
    out("-" * 78)
    out("[3] 归一自检：最长优先 + 规范名恒等映射（调用方加载逻辑的参考实现）")
    out("-" * 78)
    flat = {}
    for g in groups:
        for s in [g["canonical"]] + g["aliases"]:
            flat[s] = g["canonical"]
    first_idx = defaultdict(list)
    for s in flat:
        first_idx[s[0]].append(s)
    for ch in first_idx:
        first_idx[ch].sort(key=len, reverse=True)
    def normalize(text, identity_canonical=True):
        out_chars, i, n = [], 0, len(text)
        while i < n:
            hit = None
            for s in first_idx.get(text[i], ()):
                if not identity_canonical and s in canon_set:
                    continue
                if text.startswith(s, i):
                    hit = s
                    break
            if hit:
                out_chars.append(flat[hit])
                i += len(hit)
            else:
                out_chars.append(text[i])
                i += 1
        return "".join(out_chars)
    canon_set = {g["canonical"] for g in groups}
    sample = []
    for _name, text in docs:
        step = max(1, len(text) // 120000)
        sample.append(text[::step])
    idem_bad, checked = [], 0
    for s in sample:
        once = normalize(s)
        twice = normalize(once)
        checked += len(s)
        if once != twice:
            bad = next( (i for i in range(min(len(once), len(twice))) if once[i] != twice[i]), min(len(once), len(twice)), )
            idem_bad.append((once[max(0, bad - 12):bad + 12], twice[max(0, bad - 12):bad + 12]))
    out("抽样 %d 字符做归一，规范名是否参与匹配：是" % checked)
    if idem_bad:
        for a, b in idem_bad[:3]:
            out("    非幂等样例：%r -> %r" % (a, b))
        problems.append( "归一不是幂等的（规范名未按恒等映射参与匹配，短别名会二次替换成叠加串）" )
    else:
        out("幂等性：通过 —— norm(norm(x)) == norm(x)，不会出现二次替换叠加")
    # 反例样例从当前词表动态构造：找「别名是规范名真子串」的一组
    demo = next(
        (g["canonical"] for g in groups
         if any(a and a in g["canonical"] and len(a) < len(g["canonical"]) for a in g["aliases"])),
        groups[0]["canonical"],
    )
    out("    反例演示（若加载方忘记把规范名映射到自身）：")
    out("      正确：%s" % normalize(demo))
    out("      错误：%s" % normalize(demo, identity_canonical=False))
    if normalize(demo) == normalize(demo, identity_canonical=False):
        warnings.append("参考实现的正/反例输出相同，说明该样例没覆盖到子串陷阱")
    # 对拍只在"校验的正是线上加载的那张表"时才有意义：normalize_aliases 读的是模块级
    # ALIASES_FILE/_LEXICON，传入自定义路径时线上实现根本没换表，硬比只会报假"分叉"。
    same_table = os.path.abspath(alias_path) == os.path.abspath(ALIASES_FILE)
    if not same_table or not ALIASES_ENABLED:
        out("与线上实现对拍：跳过（校验 %s；线上加载 %s，别名开关 %s）" % (
            alias_path, ALIASES_FILE, "开" if ALIASES_ENABLED else "关"))
    else:
        mismatch = []
        for s in sample[:2]:
            got, want = normalize(s), normalize_aliases(s)
            if got != want:
                i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), min(len(got), len(want)))
                mismatch.append((got[max(0, i - 12):i + 12], want[max(0, i - 12):i + 12]))
        if mismatch:
            for a, b in mismatch[:2]:
                out("    与线上实现不一致：%r (本文件参照实现) vs %r (normalize_aliases)" % (a, b))
            problems.append( "参照实现与 normalize_aliases 输出不一致 —— 两份实现已分叉，自检结论不可信")
        else:
            out("与线上实现对拍：通过 —— normalize_aliases 输出逐字相同")
    out("")
    out("-" * 78)
    out("[4] 停用词表: %s" % stop_path)
    out("-" * 78)
    swords, sp = parse_stopwords(stop_path)
    problems += sp
    words = [w for w, _ in swords]
    out("停用词数量：%d" % len(words))
    danger = []
    for w in words:
        bad_neg = sorted(set(w) & FORBIDDEN_NEGATION)
        bad_deg = sorted(set(w) & FORBIDDEN_DEGREE)
        if bad_neg or bad_deg:
            danger.append((w, bad_neg, bad_deg))
    if danger:
        for w, bn, bd in danger:
            problems.append( "停用词 %r 含禁止字：否定%s 程度%s —— 过滤后会破坏语义（「不读书」与「读书」会变成同一查询）" % (w, bn or "-", bd or "-") )
    else:
        out("危险性检查：通过 —— 无否定词（不/无/未/莫/非/没/别…）、无程度词（很/太/最/更/极/甚…）")
    name_conflict = sorted(set(words) & surface_set)
    for w in name_conflict:
        problems.append("停用词 %r 同时是别名词典里的词面（会把实体名过滤掉）" % w)
    if not name_conflict:
        out("交叉检查：停用词与别名词典无交集")
    zero_sw = [w for w in words if sum(count_in_documents(docs, [w])[w].values()) == 0]
    for w in zero_sw:
        warnings.append("停用词 %r 在语料中出现 0 次（留着无害，可删）" % w)
    out("")
    out("    停用词              次数    停用词              次数")
    out("    " + "-" * 60)
    sw_counts = count_in_documents(docs, words)
    pairs = [(w, sum(sw_counts[w].values())) for w in words]
    half = (len(pairs) + 1) // 2
    for i in range(half):
        left = "%-8s %8d" % pairs[i]
        right = "%-8s %8d" % pairs[i + half] if i + half < len(pairs) else ""
        out("    %s      %s" % (left, right))
    out("")
    out("=" * 78)
    if problems:
        log("发现问题 %d 处：" % len(problems))
        for p in problems:
            log("  [错误] " + p)
        return 1
    if warnings:
        log("提示 %d 处（不阻断）：" % len(warnings))
        for w in warnings:
            log("  [提示] " + w)
    log("全部通过：别名组 %d 组 / 词面 %d 个 / 停用词 %d 个，均已数据接地，无禁止类停用词。" % (len(groups), len(surfaces), len(words)))
    return 0


# ── 工具：阶段状态与最小重跑指引 ──
def cmd_stages():
    """逐阶段比对指纹，给出最小重跑命令。
    A 分块（语料/切分变 → process） B 稠密（换模型/新内容 → index 增量）
    C 稀疏（词表/BM25/编码变 → reindex-sparse） D 入库（集合缺失/点数不符 → index）"""
    stale = []
    cd_unknown = False

    log("[A] 分块产物")
    if not os.path.exists(CHUNKS_META_JSON):
        stale.append("A"); log("  ❌ 缺少元信息指纹 -> 需要: python chatbot.py process")
    else:
        with open(CHUNKS_META_JSON, encoding="utf-8") as f:
            meta = json.load(f)
        fingerprint, _ = _chunking_config_fingerprint()
        if meta.get("fingerprint") != fingerprint:
            stale.append("A"); log(f"  ❌ 切分代码/参数指纹不一致（{meta.get('fingerprint')} -> {fingerprint}）-> 需要: process")
        else:
            try:
                paths = _document_paths()
                old_sources = meta.get("sources") if "sources" in meta else meta.get("books")
                if old_sources != _sources_fingerprint(paths):
                    stale.append("A"); log("  ❌ 语料内容已变化 -> 需要: process")
                else:
                    log(f"  ✅ 新鲜（{len(paths)} 个文档，指纹 {fingerprint}）")
            except Exception as exc:
                stale.append("A"); log(f"  ❌ 语料检查失败: {type(exc).__name__}: {exc}")

    log("[B] 稠密向量缓存")
    if "A" in stale:
        log("  ⏭  A 过期，先重跑 process 再谈 B")
    elif not EMBED_CACHE_ENABLED:
        log("  ⏭  RAG_EMBED_CACHE=0：无缓存，index 每次全量重算")
    elif not os.path.exists(CHUNKS_JSON):
        log("  ⏭  缺 chunks.json，先跑 process")
    else:
        precision = _index_cache_precision()
        preferred = _embed_cache_file(precision)
        if not os.path.exists(preferred):
            stale.append("B")
            log(f"  ❌ 缺少当前设备精度（{precision}）的缓存文件 {os.path.basename(preferred)} -> index 会全量重算稠密向量")
        else:
            try:
                cache = _load_embed_cache(preferred)
                with open(CHUNKS_JSON, encoding="utf-8") as f:
                    chunks = json.load(f)
                need = sum(1 for c in chunks if _embed_cache_key(_index_text(c)) not in cache)
                if need:
                    # 缓存里缺条目 = index 仍要重算 need 条：不标 B 会让 A/C/D 全新鲜时
                    # 打出"无需重跑"，与上面这行自相矛盾（同一轮日志两个结论）。
                    stale.append("B")
                    log(f"  ⚠️  缓存 {os.path.basename(preferred)}（{precision}）：{len(cache)} 条，缺 {need}/{len(chunks)} 条（增量）→ 需要: python chatbot.py index")
                else:
                    log(f"  ✅ 缓存 {os.path.basename(preferred)}（{precision}）：{len(cache)} 条；当前产物 0/{len(chunks)} 条需重算")
            except Exception as exc:
                log(f"  ⚠️  缓存检查失败（不影响 A/C/D 判定）: {type(exc).__name__}: {exc}")

    log("[C] 稀疏向量（词表/BM25）与 [D] 入库（Qdrant）")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=30)
        coll = resolve_collection_name(client)
        if not client.collection_exists(coll):
            stale.append("D"); log("  [D] ❌ 集合不存在 -> 需要: python chatbot.py index")
        else:
            qd_count = client.count(collection_name=coll, exact=True).count
            pts, _ = client.scroll(collection_name=coll, limit=1, with_payload=True) if qd_count else ([], None)
            idx_lex = (pts[0].payload or {}).get(LEXICON_ID_FIELD) if pts else None
            lex_now = lexicon_fingerprint()
            if idx_lex is None:
                stale.append("C"); log("  [C] ❌ 索引未记录词表指纹（旧索引）-> 需要: reindex-sparse")
            elif idx_lex != lex_now:
                stale.append("C"); log(f"  [C] ❌ 词表/稀疏编码指纹不一致（索引 {idx_lex} / 当前 {lex_now}）-> 需要: reindex-sparse")
            else:
                log("  [C] ✅ 新鲜（稠密向量不受影响）")
            n_chunks = None
            if os.path.exists(CHUNKS_JSON):
                with open(CHUNKS_JSON, encoding="utf-8") as f:
                    n_chunks = len(json.load(f))
            if n_chunks is not None and qd_count != n_chunks:
                stale.append("D"); log(f"  [D] ❌ 点数不一致（集合 {qd_count} / 产物 {n_chunks}）-> 需要: index")
            else:
                log(f"  [D] ✅ 集合 {coll} 共 {qd_count} 条")
    except Exception as exc:
        log(f"  ⏭  连不上 Qdrant（{type(exc).__name__}: {exc}）→ C/D 未验证（不算新鲜）")
        cd_unknown = True

    log("")
    if not stale and not cd_unknown:
        log("全部阶段新鲜，无需重跑。")
    elif "A" in stale:
        log("最小重跑: python chatbot.py process && python chatbot.py index")
    elif "B" in stale or "D" in stale:
        log("最小重跑: python chatbot.py index（稠密走缓存增量）")
    elif "C" in stale:
        log("最小重跑: python chatbot.py reindex-sparse")
    else:
        # stale 为空但 Qdrant 没连上：C/D 没验过，绝不能落到上面的"全部新鲜"。
        log("C/D 未能验证（Qdrant 不可达）→ 恢复后重跑: python chatbot.py stages")
    return 0


# ── 命令分发 ──
COMMANDS = { "process": cmd_process, "index": cmd_index, "serve": cmd_serve, "api": cmd_api, "health": cmd_health, "reindex-sparse": cmd_reindex_sparse, "compare-chunks": cmd_compare_chunks, "lexicon": cmd_verify_lexicon, "stages": cmd_stages, }


def usage():
    log("用法: python chatbot.py <命令> [参数]")
    log("  " + " ".join(COMMANDS))
    log("Streamlit 界面: streamlit run chatbot.py")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        usage()
        return 0 if argv else 1
    name, rest = argv[0], argv[1:]
    fn = COMMANDS.get(name)
    if fn is None:
        log(f"未知命令: {name}")
        usage()
        return 1
    sys.argv = [f"chatbot.py {name}"] + rest
    result = fn()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _in_streamlit = get_script_run_ctx() is not None
    except Exception:
        _in_streamlit = False
    if _in_streamlit:
        run_streamlit()
    else:
        sys.exit(main())
