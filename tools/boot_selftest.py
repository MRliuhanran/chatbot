#!/usr/bin/env python3
"""创空间冷启动自检：容器内跑**真实一轮**检索 + 生成，把结果与内存现场写进运行日志。

为什么必须放在构建侧（2026-10-16 实测）：
  - 8G 档下 llama-server 被 cgroup OOM 杀（`signal: killed`）时，业务侧只表现为一次 500；
    运行日志里既没有 KV 分配规模、也没有峰值内存，只能靠事后猜——而根因（缺少
    OLLAMA_KV_CACHE_TYPE / FLASH_ATTENTION / NUM_PARALLEL 三个服务端变量）恰恰不在
    业务代码里。
  - 容器外无法驱动验收：Streamlit 的脚本只在浏览器 websocket 会话建立后才运行，curl 只
    拿到静态壳（python 常驻 128M = 从未跑过脚本），"页面 200"不等于"能问答"。

实现要点：
  - 检索走 `get_engine().hybrid_search`，生成走 `plan_generation` + `stream_chat`，与
    界面/API **同一条**代码路径，不另写探针；
  - 生成探针刻意 think=false、num_predict=8：与 `_warm_engine` 同一判据（验链路能出字，
    不验思维链），否则 think 会吃光预算、正文恒空，把"通"报成"不通"；
  - **num_ctx 取业务生效值**（RAG_NUM_CTX=65536），这才是 KV 能否分配、会不会 OOM 的
    真实条件；
  - 失败非致命（恒退出 0）：自检失败只告警，业务侧仍可服务（首查询冷加载兜底）。
"""
import os
import subprocess
import sys
import time

QUERY = os.environ.get("RAG_BOOT_SELFTEST_QUERY", "毛泽东思想活的灵魂是什么？")


def _mem_available_mb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def _max_rss_mb(proc_name):
    """取指定进程名最大 RSS（MB）。procps 已在镜像里装（见 Dockerfile 的 apt 层）。"""
    try:
        out = subprocess.run(["ps", "-C", proc_name, "-o", "rss="],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    vals = [int(x) for x in out.split() if x.strip().isdigit()]
    return max(vals) // 1024 if vals else None


def _snapshot():
    return _mem_available_mb(), _max_rss_mb("llama-server")


def main():
    # 容器内应用根是 /app；本地自测时用本文件上一级目录（tools/ 的父目录）。
    root = "/app" if os.path.isdir("/app/chatbot.py") else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    log = print
    mem0 = _mem_available_mb()
    log(f"[自检] 开始：query={QUERY!r} MemAvailable={mem0}MB")
    t0 = time.time()
    try:
        import chatbot

        engine = chatbot.get_engine()
        results = engine.hybrid_search(QUERY, top_k=chatbot.TOP_K)
        log(f"[自检] 检索完成：命中 {len(results)} 个父块，{time.time() - t0:.1f}s")
        for item in results[:1]:
            log(f"[自检] Top1 来源={item.get('source')!r} rerank={item.get('rerank_score')}")

        if chatbot.EXCLUSIVE_MODELS:
            engine.release_models()

        plan = chatbot.plan_generation(results, [], QUERY, chatbot.NUM_CTX)
        stats = plan.get("stats") or {}
        log(f"[自检] 装配：context_docs_kept={stats.get('context_docs_kept')} "
            f"context_tokens={stats.get('context_tokens')} num_ctx={chatbot.NUM_CTX}")
        thinking, answer, err, done = chatbot.drain_events(
            chatbot.stream_chat(plan["messages"], num_ctx=chatbot.NUM_CTX,
                                num_predict=8, think=False)
        )
        mem1, rss = _snapshot()
        ok = (not err) and bool(done) and bool(answer.strip())
        log(f"[自检] 生成：正文 {len(answer)} 字 error={err!r} "
            f"done_reason={done.get('done_reason')!r} eval_count={done.get('eval_count')!r}")
        log(f"[自检] 答案前 60 字：{answer.strip()[:60]!r}")
        log(f"[自检] 现场：MemAvailable={mem1}MB（Δ{(mem0 - mem1) if (mem0 and mem1) else '?'}MB）"
            f" llama-server RSS={rss}MB 总耗时 {time.time() - t0:.1f}s")
        log("[自检] " + ("✅ 通过：检索 + 生成全链路通畅" if ok else "❌ 未通过：链路有失败，见上"))
    except Exception as exc:
        mem1, rss = _snapshot()
        log(f"[自检] ❌ 异常：{type(exc).__name__}: {exc}")
        log(f"[自检] 现场：MemAvailable={mem1}MB llama-server RSS={rss}MB 耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
