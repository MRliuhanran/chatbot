#!/usr/bin/env python3
"""RAG 服务的 HTTP API —— 让检索能力脱离 Streamlit 被复用。

## 为什么用标准库而不是 FastAPI

本项目的一贯取舍是"单机可跑、不引入非必要依赖"（sentencex 零依赖被专门写进
requirements 注释、pydantic/scikit-learn 因零引用被删除）。这里只有 3 个端点，
标准库的 ThreadingHTTPServer 足够，且不必为它装一个 web 框架 + ASGI 栈。
若将来要加鉴权、限流、OpenAPI 文档，再引入框架是合理的 —— 届时这层薄封装
可以整体替换，因为业务逻辑都在 rag_engine 里。

## 端点

    GET  /health            → 服务与索引状态
    POST /search            → 只检索，返回结果 + 推理链 + 置信度
    POST /ask               → 检索 + 生成，NDJSON 流式返回（含 thinking/answer）

## 并发

检索要跑深度学习模型，且 RAGEngine 内部有懒加载缓存（模型、分块表），
因此这里用一把**全局锁把请求串行化**。理由：
  * 8GB 统一内存的机器上，并发推理不只是变慢，而是会触发换页甚至 MPS 卡死；
  * 引擎的缓存不是为并发写设计的（两个线程同时发现 _chunks 为 None 会重复拉全表）。
并发扩展的正确做法是加进程池 + 每进程一份模型，而不是在单进程里放开线程 ——
那属于部署层的事，不该由这个文件假装解决。
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# .env 必须先加载：本模块的 argparse 默认值读 RAG_API_HOST / RAG_API_PORT。
# （rag_engine 也会加载，但显式写出依赖，读代码的人不必去追调用链。）
import bootstrap  # noqa: E402,F401

import rag_engine as RE    # noqa: E402
import rag_turn as RT      # noqa: E402
from ollama_client import (  # noqa: E402
    MODEL, OLLAMA_BASE_URL, THINK,
)

# 引擎单例锁：**只保护 _ENGINE 的创建**，不要在它内部再调用 get_engine()。
# 实测教训：最初的 do_GET 写成 `with _ENGINE_LOCK: engine = get_engine()`，
# 而 get_engine() 内部也要拿同一把普通 Lock —— 非重入锁自锁，/health 直接
# 挂死（curl 超时、没有任何报错，只有"服务不响应"这一现象）。
# 这里用 RLock 是第二道防线：即使将来又有人在持锁状态下调用 get_engine()，
# 也只是嵌套而非死锁。
_ENGINE = None
_ENGINE_LOCK = threading.RLock()
# 请求串行锁：把"一次检索/生成"整体串起来。与 _ENGINE_LOCK 分工不同，
# 不要混用 —— 混用正是上面那个死锁的成因。
_REQUEST_LOCK = threading.Lock()

MAX_BODY_BYTES = int(os.getenv("RAG_API_MAX_BODY", str(64 * 1024)))
DEFAULT_TOP_K = RE.TOP_K


def get_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = RE.RAGEngine()
        return _ENGINE


def _json_response(handler, status, payload, close=False):
    """发一个 JSON 响应。

    close=True：显式声明连接用完即关（错误路径一律用它）。
    为什么错误路径必须关连接：HTTP/1.1 下 stdlib 默认保持长连接，
    `close_connection` 只在 `send_header('Connection', …)` 里被复位 ——
    若我们在**没读完请求体**的情况下回错误，残留的 body 会被下一条请求
    当成自己的请求行读走，于是客户端发出的**合法新请求**收到 400 并断连
    （实测路径：POST 到未知路径、或 body 超过 MAX_BODY_BYTES）。
    """
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
    """按 Content-Length 读掉并丢弃请求体（有界）。

    拒绝一个请求之前必须先把它声明要发的 body 读完，否则连接会失步
    （见 _json_response 的说明）。**只排空能排空的**：超过上限时不能照读
    （那会把服务拖死），此时要靠 `Connection: close` 兜住。
    """
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
    """读并解析请求体。超长/非法 JSON 都返回可诊断的错误而不是栈。

    每一条错误路径都负责**先把 body 排空、再关连接**：不排空会失步，
    不关连接会在超长 body 上继续失步（那时排空本身不可行）。
    """
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        if handler.headers.get("Transfer-Encoding"):
            raise ValueError(
                "不支持 Transfer-Encoding（chunked）请求体：请带 Content-Length")
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
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"请求体不是合法 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return data


def _search_params(data):
    """从请求体取出检索参数，并校验类型。

    校验存在的意义：query=123 这类错误若不拦，会在 embed() 里以
    "expected str" 的形式炸在下游，排查时完全看不出是入参问题。

    history 的**元素**同样要校验，不能只判"是不是 list"：一个
    `{"role":"user","content":123}` 会让 strip_current_turn 抛 AttributeError，
    表现为 500"检索失败"而不是 400"参数错了"—— 报错层级被抬高了一级，
    排查方向也就跟着偏了。
    """
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("参数 query 必填，且必须是非空字符串")
    top_k = data.get("top_k", DEFAULT_TOP_K)
    # bool 是 int 的子类：不单独挡掉的话 true 会被静默当成 top_k=1
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("参数 top_k 必须是正整数")
    book = data.get("book")
    if book is not None and not isinstance(book, str):
        raise ValueError("参数 book 必须是字符串（书名，如 '红楼梦'）")
    history = data.get("history")
    if history is not None:
        if not isinstance(history, list):
            raise ValueError("参数 history 必须是消息数组")
        for i, m in enumerate(history):
            if not isinstance(m, dict):
                raise ValueError(f"history[{i}] 必须是对象（含 role 与 content）")
            role = m.get("role")
            if role not in ("user", "assistant"):
                raise ValueError(
                    f"history[{i}].role 必须是 user 或 assistant，收到 {role!r}")
            content = m.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError(
                    f"history[{i}].content 必须是字符串，收到 {type(content).__name__}")
    return query, top_k, book, history


def _result_json(r):
    """把检索结果裁成 API 形状。

    只暴露调用方真正需要的字段：child_text 是命中片段（用于高亮），
    parent_text 是喂给 LLM 的完整上下文，chapter_* 用于出处标注。
    point_id / contextual_text 这类内部字段不外露 —— 它们会随实现变化。
    """
    return {
        "id": r.get("id", ""),
        "book": r.get("book", ""),
        "chapter": r.get("chapter_label", ""),
        "chapter_title": r.get("chapter_title", ""),
        "score": r.get("rerank_score"),
        "text": r.get("child_text", ""),
        "context": r.get("parent_text", ""),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "RAGKnowledgeBase/1.0"
    protocol_version = "HTTP/1.1"
    #: 客户端不读响应时的写阻塞上限。默认 None = **永不超时**：一个只连不读的
    #: 客户端能让 emit 的 wfile.write 永久阻塞，而 /ask 若仍持锁就会拖垮全服务。
    #: 这里给 socket 一个上限，让"坏客户端"变成一次可诊断的连接超时。
    timeout = 60

    # 默认的 log_message 把每个请求打到 stderr，格式与本项目的 logging 不一致。
    # 用 logging 输出，便于与 rag_engine 的日志统一收集。
    def log_message(self, fmt, *args):
        RE.logger.info("API %s - %s", self.address_string(), fmt % args)

    def _reject(self, status, message):
        """统一的拒绝路径：先排空 body，再回错误并关连接（防 keep-alive 失步）。"""
        _drain_body(self)
        _json_response(self, status, {"error": message}, close=True)

    def do_GET(self):
        if self.path.split("?")[0] != "/health":
            self._reject(404, f"未知路径: {self.path}")
            return
        try:
            # 不要在这里加 `with _ENGINE_LOCK`：get_engine() 内部已经拿它了
            # （见文件顶部 _ENGINE_LOCK 的注释）。
            # 但 count() 会改写 engine._collection_name，与并发的 hybrid_search
            # 争同一份状态，故仍走 _REQUEST_LOCK（检索 0.5~1s，/health 等得起）。
            engine = get_engine()
            with _REQUEST_LOCK:
                count = engine.count()
                collection = engine.collection_name
        except Exception as exc:
            # 连不上 Qdrant 与"索引为空"必须区分：前者是环境问题，后者要重跑索引
            _json_response(self, 503, {
                "status": "error",
                "error": f"无法访问向量库: {type(exc).__name__}: {exc}",
            }, close=True)
            return
        _json_response(self, 200, {
            "status": "ok" if count else "empty",
            "chunks": count,
            "collection": collection,
            "model": MODEL,
            "ollama": OLLAMA_BASE_URL,
            "use_context": RE.RAG_USE_CONTEXT,
            "think": THINK,
            "query_rewrite": __import__("query_rewrite").REWRITE_ENABLED,
        })

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/search", "/ask"):
            self._reject(404, f"未知路径: {self.path}")
            return
        try:
            data = _read_json(self)
            query, top_k, book, history = _search_params(data)
        except ValueError as exc:
            self._reject(400, str(exc))
            return

        # 剥掉"末尾即本轮提问"：两个 UI 入口都传 session_state.messages[:-1]，
        # 而 API 的 history 由调用方给 —— 客户端照 UI 习惯提交完整会话时，
        # 模型会收到 [历史…, 资料, 本轮问句, 本轮问句]：问句重复、历史预算被
        # 白吃，而且"同一个会话在两个入口发给模型不同"正是最难发现的那类不一致。
        # strip_current_turn 只在末尾 user 与 query 逐字相同时才剥，
        # 不会误伤"用户真的连问了两遍同一句话"。
        history, history_stripped = RE.strip_current_turn(history, query)

        if path == "/search":
            self._handle_search(query, top_k, book, history, history_stripped)
        else:
            self._handle_ask(query, top_k, book, history, data, history_stripped)

    # ---- /search --------------------------------------------------------
    def _handle_search(self, query, top_k, book, history, history_stripped=False):
        try:
            with _REQUEST_LOCK:
                engine = get_engine()
                t0 = time.time()
                results, steps = engine.hybrid_search(
                    query, top_k=top_k, return_steps=True, book=book, history=history,
                )
        except Exception as exc:
            _json_response(self, 500, {"error": f"检索失败: {type(exc).__name__}: {exc}"})
            return
        # rewritten 保持旧语义（未改写时为 null）；三态另给一个字段 ——
        # "开关关了""Ollama 挂了""模型没输出"在旧口径下都是 null，
        # 调用方无从分辨，与 UI 侧要的是同一件事。
        rewrite = steps.get(RE.STEP_REWRITE) or {}
        _json_response(self, 200, {
            "query": query,
            "rewritten": rewrite.get("query") if rewrite.get("applied") else None,
            "rewrite_detail": rewrite,
            "book": book,
            "elapsed": round(time.time() - t0, 3),
            "confidence": steps.get(RE.STEP_CONFIDENCE),
            "history_stripped": history_stripped,
            "results": [_result_json(r) for r in results],
            # steps 原样返回：它是一个扁平 JSON（键=步骤名，值=该步产出），
            # 调用方不需要认识任何步骤名就能把它整个展示或丢弃。
            "steps": steps,
        })

    # ---- /ask -----------------------------------------------------------
    def _handle_ask(self, query, top_k, book, history, data, history_stripped=False):
        """检索 + 生成，NDJSON 流式返回。

        每行一个 JSON 对象，type 取值：
            retrieval        检索完成（含 results / confidence / sources）
            history_trimmed  历史被按预算裁剪（静默裁剪=回答看着正常却没了依据）
            thinking         思维链增量
            delta            答案增量
            error            生成失败（**独立类型**，不会混进 delta 被当成答案）
            done             结束（含统计）
        用 NDJSON 而不是 SSE：不需要额外解析分帧协议，逐行 json.loads 即可。

        一轮的**序列**由 rag_turn.run_turn 定义（检索 → 置信度 → 装配 → 拒答 →
        流式生成），本方法只把这些事件翻译成 NDJSON。此前这里与 app.py 各写一遍
        序列，而"同一会话在两个入口行为不同"正是最难发现的那类不一致。

        **检索在第一次 next() 时发生**，所以请求锁只罩住那一次 next ——
        生成（think 打开时实测 ~135s）不占锁，否则一个 /ask 会把并发的 /search
        全部堵死（ThreadingHTTPServer 给人的"能并发"就是假象了）。

        **错误分两种**（`headers_sent`）：
          * 响应头还没发出 → 这是一个完全正常的 HTTP 错误，回 500 JSON。
            旧实现在这种情况下直接往 wfile 写一行 NDJSON，于是客户端收到的是
            没有状态行、没有 header 的"响应"—— 既读不到错误也解析不了帧，
            而同一个故障在 /search 里却是干净的 500 JSON（同一故障两种行为）。
          * 响应头已发出 → 流式协议中途无法改状态码，只能写一行 error 事件并
            关连接。
        """
        headers_sent = False
        t0 = time.time()
        try:
            with _REQUEST_LOCK:
                engine = get_engine()
                events = RT.run_turn(engine, query, history, top_k=top_k, book=book)
                kind, payload = next(events)     # ← 检索 + 装配在这里，之后出锁

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            # **必须显式声明连接用完即关**。流式响应没有 Content-Length，
            # 而 protocol_version="HTTP/1.1" 默认保持长连接 —— 客户端因此
            # 无从判断响应到哪里结束，会一直挂在读上直到自己超时。
            # 实测（端到端跑 /ask）：NDJSON 里 done 事件早已发出、答案完整，
            # 而 curl 仍以退出码 28（超时）收场，看起来像"服务没回完"。
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            headers_sent = True

            def emit(obj):
                self.wfile.write(
                    (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()

            emit({
                "type": "retrieval",
                "query": query,
                "elapsed": round(payload["elapsed"], 3),
                "confidence": payload["confidence"],
                "sources": RE.context_sources(payload["results"]),
                "history_stripped": history_stripped,
                "results": [_result_json(r) for r in payload["results"]],
            })

            gen = payload["gen"]
            if gen["dropped_turns"] or gen["truncated"]:
                emit({
                    "type": "history_trimmed",
                    "dropped_turns": gen["dropped_turns"],
                    "budget": gen["budget"],
                    "used_tokens": gen["used_tokens"],
                    "context_tokens": gen["context_tokens"],
                    "truncated": gen["truncated"],
                    "truncated_user": gen["truncated_user"],
                })

            if kind == RT.EVENT_ABSTAIN:
                # 检索为空时不让模型自由发挥（那等于把幻觉当答案返回）。
                emit({"type": "delta", "text": payload["answer"]})
                emit({"type": "done", "done_reason": "no_context",
                      "eval_count": 0, "search_s": round(payload["elapsed"], 3)})
                return

            for kind, payload in events:
                if kind == RT.EVENT_THINKING:
                    emit({"type": "thinking", "text": payload})
                elif kind == RT.EVENT_DELTA:
                    emit({"type": "delta", "text": payload})
                elif kind == RT.EVENT_ERROR:
                    emit({"type": "error", "error": payload})
                elif kind == RT.EVENT_DONE:
                    emit({
                        "type": "done",
                        "done_reason": payload.get("done_reason"),
                        "eval_count": payload.get("eval_count"),
                        "prompt_eval_count": payload.get("prompt_eval_count"),
                        "thinking_chars": payload.get("thinking_chars"),
                        "answer_chars": payload.get("answer_chars"),
                        "search_s": round(payload["elapsed"], 3),
                        "gen_s": round(time.time() - t0 - payload["elapsed"], 3),
                    })
        except Exception as exc:
            if not headers_sent:
                # 还没发过响应头 —— 这是一个完全正常的 HTTP 错误，
                # 必须按 HTTP 错误回，而不是把 NDJSON 裸写进 wfile
                # （那会产出没有状态行/header 的"响应"，客户端既读不到错误
                #   也解析不了帧；同一故障在 /search 里却是干净的 500）。
                _json_response(self, 500, {
                    "error": f"检索/装配失败: {type(exc).__name__}: {exc}"}, close=True)
                return
            # 已经发过响应头就只能中断连接：NDJSON 是流式协议，
            # 中途无法再改状态码，硬写一个 error 行至少让客户端知道发生了什么。
            try:
                self.wfile.write((json.dumps(
                    {"type": "error", "error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="RAG 知识库 HTTP API")
    parser.add_argument("--host", default=os.getenv("RAG_API_HOST", "127.0.0.1"),
                        help="监听地址（默认 127.0.0.1，仅本机可访问）")
    parser.add_argument("--port", type=int, default=int(os.getenv("RAG_API_PORT", "8000")))
    args = parser.parse_args()

    # 启动路径的错误处理必须与运行路径对等：/health 在连不上向量库时回可诊断的
    # 503，而这里此前会抛裸栈 —— 同一个环境问题，两条路径两种表现。
    try:
        count = get_engine().count()
    except Exception as exc:
        print(f"错误: 无法访问向量库 {RE.QDRANT_HOST}:{RE.QDRANT_PORT}")
        print(f"  原因: {type(exc).__name__}: {exc}")
        print("  请先确认服务已启动: docker compose up -d")
        sys.exit(1)
    if count == 0:
        print("错误: 向量数据库为空，请先运行: python app.py process && python app.py index")
        sys.exit(1)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    RE.log(f"API 就绪: http://{args.host}:{args.port}  （{count} 条，模型 {MODEL}）")
    RE.log("  GET  /health")
    RE.log("  POST /search  {\"query\": \"武松打虎\", \"top_k\": 5}")
    RE.log("  POST /ask     {\"query\": \"武松打虎\"}  → NDJSON 流")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        RE.log("收到中断，正在关闭…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
