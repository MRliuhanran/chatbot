#!/usr/bin/env python3
"""把「一轮 RAG」跑成事件流 —— 四个入口只负责渲染。

## 为什么需要这个模块

一轮对话的序列是固定的：

    检索 → 取置信度 → 生成侧装配 → （检索为空？硬拒答）→ 流式生成 → 统计

此前这段序列在 app.py（Streamlit 渲染）与 api.py（NDJSON 渲染）里**各写一遍**，
chat.py 是第三份简化版。三个入口各自知道正确顺序，而顺序错了**没有任何地方会
报错** —— 已经因此出过两次同类缺陷：chat.py 漏传 `use_context`（对照模式静默
失效）、漏掉"检索为空硬拒答"。这与本项目在"工具各抄一份检索管线"上踩的坑同源：
**同一个知识写在多处，就一定会分叉。**

现在序列只在这里实现一次，产出 `(kind, payload)` 事件；入口退化成"把这些事件
画出来"。要改序列（加一步拒答、换装配方式）只改这一处。

## 为什么不放进 rag_engine.py

`rag_engine` 的定位是"与 UI / 生成框架解耦的检索模块"，把生成编排塞进去会让
它同时知道检索、提示词、Ollama 与事件协议 —— 那是用低耦合换来的高耦合。
反过来，本模块依赖 rag_engine（检索）、ollama_client（生成），但**不被它们依赖**，
依赖方向是干净的。

## 事件契约

    EVENT_RETRIEVAL  {query, results, steps, confidence, messages, gen, elapsed}
                     一次检索 + 装配完成。steps 里已经含生成侧两步
                     （上下文装配 / 历史裁剪），因为那两步由 plan_generation 产出。
    EVENT_ABSTAIN    {answer, steps}
                     检索为空 → 硬拒答，**不进模型**。事件流到此结束。
    EVENT_THINKING   str   思维链增量
    EVENT_DELTA      str   答案增量
    EVENT_ERROR      str   生成失败（与 DELTA 分开，绝不能混进答案）
    EVENT_DONE       {done_reason, eval_count, prompt_eval_count,
                      thinking_chars, answer_chars, elapsed, steps}

`abstain` 与 `error` 是独立事件而不是"带标记的 delta"：它们的区别在**调用方要不要
把它当模型说过的话**。此前 error 借 content 送出去，结果被写进会话历史、下一轮
当"助手说过的话"回灌给模型。
"""

import time

import ollama_client as OC
import rag_engine as RE

EVENT_RETRIEVAL = "retrieval"
EVENT_ABSTAIN = "abstain"
EVENT_THINKING = "thinking"
EVENT_DELTA = "delta"
EVENT_ERROR = "error"
EVENT_DONE = "done"


def run_turn(engine, query, history, *, top_k=None, book=None,
             use_context=None, num_ctx=None, num_predict=None):
    """跑一轮 RAG，产出 (kind, payload)。

    参数缺省时取线上配置（rag_engine / ollama_client 的模块常量），
    因此入口不必逐个转发 —— 那正是"某入口漏传一个开关"的成因。

    **检索发生在第一次迭代时**（生成器语义）。调用方若要给检索加超时/锁/进度条，
    把 `next()` 包在自己的临界区里即可（api.py 就是这么把请求锁只罩住检索的）。
    """
    top_k = RE.TOP_K if top_k is None else top_k
    use_context = RE.RAG_USE_CONTEXT if use_context is None else use_context
    num_ctx = OC.NUM_CTX if num_ctx is None else num_ctx
    num_predict = OC.NUM_PREDICT if num_predict is None else num_predict

    started = time.time()
    results, steps = engine.hybrid_search(
        query, top_k=top_k, return_steps=True, book=book, history=history,
    )
    elapsed = time.time() - started

    # 置信度直接用 hybrid_search 算好的那份：它用的就是结果的 rerank_score 与
    # book（见 confidence_signal 的调用点）。api.py 曾自己再算一遍 —— 同一判据
    # 两处计算，改一处漏一处就会让"UI 判拒答、API 不判"。
    confidence = steps.get(RE.STEP_CONFIDENCE) or {}
    plan = RE.plan_generation(
        results, history, query, num_ctx, num_predict,
        use_context=use_context, low_evidence=bool(confidence.get("refuse")),
    )
    steps.update(plan["trace"])          # 生成侧两步（上下文装配 / 历史裁剪）

    yield EVENT_RETRIEVAL, {
        "query": query,
        "results": results,
        "steps": steps,
        "confidence": confidence,
        "messages": plan["messages"],
        "gen": plan["stats"],
        "elapsed": elapsed,
    }

    if plan["abstain"]:
        yield EVENT_ABSTAIN, {"answer": RE.NO_CONTEXT_ANSWER, "steps": steps}
        return

    thinking_chars = answer_chars = 0
    for kind, delta in OC.stream_chat(plan["messages"]):
        if kind == "thinking":
            thinking_chars += len(delta)
            yield EVENT_THINKING, delta
        elif kind == "content":
            answer_chars += len(delta)
            yield EVENT_DELTA, delta
        elif kind == "error":
            yield EVENT_ERROR, delta
        else:
            yield EVENT_DONE, {
                **delta,
                "thinking_chars": thinking_chars,
                "answer_chars": answer_chars,
                "elapsed": elapsed,
                "steps": steps,
            }
