#!/usr/bin/env python3
"""创空间冷启动自检：容器内验证**生成链路**（默认）或**完整检索+生成**（可选），把结果与
内存现场写进运行日志。

为什么必须放在构建侧（2026-10-16 实测）：
  - 8G 档下 llama-server 被 cgroup OOM 杀（`signal: killed`）时，业务侧只表现为一次 500；
    运行日志里既无 KV 规模也无峰值内存，只能靠事后猜——而根因（缺少 OLLAMA_KV_CACHE_TYPE /
    FLASH_ATTENTION / NUM_PARALLEL 三个服务端变量）恰恰不在业务代码里。
  - 容器外无法驱动验收：Streamlit 的脚本只在浏览器 websocket 会话建立后才运行，curl 只拿
    到静态壳（python 常驻 128M = 从未跑过脚本），"页面 200"不等于"能问答"。

两段式（**默认只跑 A 段**，理由见下）：
  A. 生成探针（默认）：直接打 Ollama 的 /api/chat，num_ctx 取业务生效值（RAG_NUM_CTX），
     think=false + num_predict=8，要求正文非空。**不导入 chatbot、不碰 torch**，进程只有
     几十 MB → 与真实查询重叠也不会把 8G 撑爆（实测 llama-server 常驻 ~5.4G，含视觉塔）。
     这正是当初 OOM 的那一步（KV 在 num_ctx=65536 上分配）。
  B. 完整一轮检索+生成（RAG_BOOT_SELFTEST_FULL=1 时才跑）：走 get_engine().hybrid_search
     → plan_generation → stream_chat，与界面/API 同一条路径。它会在本进程加载嵌入/重排
     权重（峰值 ~2G），只在"需要验收检索侧"时开。

**运行方式由 entrypoint 决定：后台 + 延迟启动**。早期版本在启动期同步跑（跑完才 exec
Streamlit），结果自检占住 7860 端口 85~240 秒，平台健康探针等不到监听 → 容器反复重启
（2026-10-16 实测 2 分钟内 6 次冷启动）。延迟启动后端口秒级可用，自检在服务已就绪的
空档里跑。

退出码恒 0：自检失败只是告警，服务照常提供（首查询冷加载兜底）。
"""
import os
import subprocess
import sys
import time
import json
import urllib.request

ENV_PATH = "/app/.env" if os.path.exists("/app/.env") else \
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
QUERY = os.environ.get("RAG_BOOT_SELFTEST_QUERY", "毛泽东思想活的灵魂是什么？")


def _env_get(key, default=""):
    """从 .env 里取键值（与 entrypoint/_load_dotenv 同源文件）。只做最小解析：
    跳过注释与空行，容忍 `export ` 前缀与行内引号。"""
    try:
        with open(ENV_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip().removeprefix("export ").strip() == key:
                    return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return default


def _mem_available_mb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def _max_rss_mb(match):
    """匹配命令行里含 match 的进程，取最大 RSS（MB）。用 args 而非 comm：
    ollama 的 runner 进程命令行里带 llama-server，但 comm 未必叫这个名字。"""
    try:
        out = subprocess.run(["ps", "-eo", "rss=,args="],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    best = 0
    for line in out.splitlines():
        line = line.strip()
        if not line or match not in line:
            continue
        head = line.split(None, 1)[0]
        if head.isdigit():
            best = max(best, int(head))
    return best // 1024 if best else None


def _log(msg):
    print(f"[自检] {msg}", flush=True)


def stage_generation(model, num_ctx, base_url):
    """A 段：最小生成探针（不导入 chatbot）。返回是否通过。"""
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "自检：请回复 ok"}],
        "stream": False,
        "think": False,
        "options": {"num_ctx": num_ctx, "num_predict": 8},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode() or "{}")
            code = resp.status
    except Exception as exc:  # HTTP 错误体里才有 ollama 的原因（如 signal: killed）
        body, code = {"__error__": f"{type(exc).__name__}: {exc}"}, None
        try:
            body["__error__"] += f" / {exc.read().decode()[:200]}"
        except Exception:
            pass
    content = ((body.get("message") or {}).get("content") or "") if isinstance(body, dict) else ""
    ok = code == 200 and bool(content.strip())
    _log(f"A 生成探针：HTTP={code} 正文={len(content)} 字 "
         f"done_reason={body.get('done_reason')!r} eval_count={body.get('eval_count')!r} "
         f"耗时={time.time() - t0:.1f}s" + ("" if ok else f" 详情={body}"))
    if content.strip():
        _log(f"A 正文前 60 字：{content.strip()[:60]!r}")
    return ok


def stage_full():
    """B 段：完整一轮检索 + 生成（与界面/API 同一代码路径）。返回是否通过。"""
    root = "/app" if os.path.isdir("/app/chatbot.py") else \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import chatbot

    t0 = time.time()
    engine = chatbot.get_engine()
    results = engine.hybrid_search(QUERY, top_k=chatbot.TOP_K)
    _log(f"B 检索：命中 {len(results)} 个父块，{time.time() - t0:.1f}s")
    for item in results[:1]:
        _log(f"B Top1 来源={item.get('source')!r} rerank={item.get('rerank_score')}")
    if chatbot.EXCLUSIVE_MODELS:
        engine.release_models()
    plan = chatbot.plan_generation(results, [], QUERY, chatbot.NUM_CTX)
    stats = plan.get("stats") or {}
    _log(f"B 装配：context_docs_kept={stats.get('context_docs_kept')} "
         f"context_tokens={stats.get('context_tokens')} num_ctx={chatbot.NUM_CTX}")
    thinking, answer, err, done = chatbot.drain_events(
        chatbot.stream_chat(plan["messages"], num_ctx=chatbot.NUM_CTX,
                            num_predict=8, think=False))
    ok = (not err) and bool(done) and bool(answer.strip())
    _log(f"B 生成：正文 {len(answer)} 字 error={err!r} "
         f"done_reason={done.get('done_reason')!r} eval_count={done.get('eval_count')!r}")
    if answer.strip():
        _log(f"B 正文前 60 字：{answer.strip()[:60]!r}")
    return ok, time.time() - t0


def main():
    model = os.environ.get("RAG_BOOT_SELFTEST_MODEL") or _env_get("MODEL") or "qwen3.5:4b-q4_K_M"
    base_url = os.environ.get("OLLAMA_BASE_URL") or _env_get("OLLAMA_BASE_URL") or \
        "http://127.0.0.1:11434"
    try:
        num_ctx = int(os.environ.get("RAG_NUM_CTX") or _env_get("RAG_NUM_CTX") or 8192)
    except ValueError:
        num_ctx = 8192
    full = os.environ.get("RAG_BOOT_SELFTEST_FULL", "0") != "0"

    mem0 = _mem_available_mb()
    _log(f"开始：model={model} num_ctx={num_ctx} 模式={'检索+生成' if full else '仅生成探针'} "
         f"KV={os.environ.get('OLLAMA_KV_CACHE_TYPE', '<未设=f16>')} "
         f"FA={os.environ.get('OLLAMA_FLASH_ATTENTION', '<未设>')} "
         f"PARALLEL={os.environ.get('OLLAMA_NUM_PARALLEL', '<未设>')} MemAvailable={mem0}MB")
    t0 = time.time()
    passed = False
    try:
        passed = stage_generation(model, num_ctx, base_url)
        if full:
            ok_b, _ = stage_full()
            passed = passed and ok_b
    except Exception as exc:
        _log(f"❌ 异常：{type(exc).__name__}: {exc}")
    mem1, rss = _mem_available_mb(), _max_rss_mb("llama-server")
    _log(f"现场：MemAvailable={mem1}MB（Δ{(mem0 - mem1) if (mem0 and mem1) else '?'}MB）"
         f" llama-server RSS={rss}MB 总耗时 {time.time() - t0:.1f}s")
    _log("✅ 通过：生成链路在 num_ctx 上可分配 KV 且能出字" if passed
         else "❌ 未通过：见上（服务仍可启动，失败会在首次查询暴露）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
