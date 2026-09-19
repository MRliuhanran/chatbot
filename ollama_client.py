#!/usr/bin/env python3
"""Ollama 流式对话客户端 —— 不依赖 rag_engine，供通用聊天机器人使用。

存在的理由：通用机器人不该为了发一个 HTTP 请求就把 BGE 模型和 Qdrant 拽进来。
本模块同时是**全仓库 Ollama 配置与流式实现的唯一来源**（app.py / api.py /
chat.py / bot.py 都从这里取），环境变量名与 .env 保持一致。
"""

import json
import os

import requests

# .env 必须在读任何环境变量之前加载（单源，见 bootstrap）。
import bootstrap  # noqa: E402,F401
from bootstrap import env_bool  # noqa: E402

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_URL = f"{OLLAMA_BASE_URL}/api/chat"
MODEL = os.getenv("MODEL", "qwen3.5:2b-q4_K_M")

# 通用机器人的人设。留空 = 不发 system 消息，模型用自己的默认行为。
# ⚠️ 已按用户要求**刻意置空**（2026-09-18）：常量本身写成空串，不再读 BOT_SYSTEM ——
# 即"人设"这条链路整体作废，bot.py 不再发 system 消息。
# 置空前它读的是 `os.getenv("BOT_SYSTEM", "").strip()`，且 .env 里从未设过该变量，
# 所以**行为与之前完全一致**，差别只在于不再有环境变量能把它打开。
SYSTEM_PROMPT = ""

# think 与两项预算必须联动：num_predict 是「思考 + 答案」的总预算，
# 只给 512 会让模型把预算全烧在思考上、答案长度为 0（done_reason=length）。
THINK = env_bool("RAG_THINK", True)
NUM_CTX = int(os.getenv("RAG_NUM_CTX", "8192" if THINK else "4096"))
NUM_PREDICT = int(os.getenv("RAG_NUM_PREDICT", "4096" if THINK else "512"))

# 首字节超时：模型冷启动首次加载可能超过 180s
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))


def stream_chat(messages, temperature=0.0):
    """流式产出 (kind, delta)；kind ∈ {"thinking", "content", "error", "done"}。

    全部 Ollama 的 HTTP 细节只在这里实现一次：app.py（UI/CLI）、api.py（HTTP）、
    chat.py、bot.py 四个入口共用它。**这不是为了少写几行** —— 此前 app.py 另有一份
    同逻辑的 `_stream_ollama(payload)`，两份的差异全是缺陷：超时一处硬编码 180s
    而另一处读 `OLLAMA_TIMEOUT`（用户调大超时只对一个入口生效），模型名/think/预算
    也在四处各读一遍环境变量。合并之后这些差异在结构上不可能发生。

    thinking 与 content 分开送，调用方才能把「过程」和「答案」放进两个区域。

    **错误必须是独立的 kind**：借 "content" 送出去的话，报错文本会被当成模型
    输出写进会话历史，下一轮又作为「助手说过的话」回灌给模型。实测触发路径：
    模型名写错（HTTP 404）、Ollama 没启动（ConnectionError）、冷启动超时。
    """
    r = None
    try:
        r = requests.post(
            API_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": True,
                "think": THINK,
                "options": {
                    "temperature": temperature,
                    "num_ctx": NUM_CTX,
                    "num_predict": NUM_PREDICT,
                },
            },
            stream=True,
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            yield "error", f"HTTP {r.status_code} {r.text[:300]}"
            return
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                yield "error", f"[Ollama] {chunk['error']}"
                continue
            message = chunk.get("message") or {}
            if message.get("thinking"):
                yield "thinking", message["thinking"]
            if message.get("content"):
                yield "content", message["content"]
            if chunk.get("done"):
                yield "done", {
                    "done_reason": chunk.get("done_reason"),
                    "eval_count": chunk.get("eval_count"),
                    "prompt_eval_count": chunk.get("prompt_eval_count"),
                }
    except Exception as exc:
        yield "error", f"{type(exc).__name__}: {exc}"
    finally:
        # 调用方提前中断生成器时也要释放连接，否则流式响应一直挂着
        if r is not None:
            r.close()
