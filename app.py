#!/usr/bin/env python3
"""
四大名著知识库 - 统一入口（处理 / 索引 / 服务）

用法:
  python app.py process   处理数据(清洗+分块)
  python app.py index     向量化入库
  python app.py           启动查询服务

检索引擎实现在 rag_engine.py，本文件只负责 CLI 编排与 Streamlit 展示。
"""

import os
import sys
import json

# 可选加载 .env（不强制依赖 python-dotenv）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import rag_engine as RE

# 兼容再导出：历史代码/测试仍可能 `from app import ...`，实际实现已移至 rag_engine
from rag_engine import (  # noqa: F401
    BOOKS_DIR,
    TOP_K,
    RERANK_TOP_K,
    CHILD_MAX_TOKENS,
    PARENT_MAX_TOKENS,
    CHUNK_OVERLAP,
    DB_PATH,
    EMBED_MODEL_PATH,
    RERANK_MODEL_PATH,
    BM25_CACHE_DIR,
    BM25_CACHE_FILE,
    BM25_TOKENS_CACHE_FILE,
    BM25_IDS_CACHE_FILE,
    CHUNKS_JSON,
    _hierarchical_split,
    _get_books_hash,
    get_device,
)

# 遗留常量：历史版本中定义但从未参与检索路径，保留以兼容旧引用
RECALL_K = RE.RERANK_TOP_K

# ============================================================================
# 生成侧配置（Ollama）
# ============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "qwen3.5:2b-q4_K_M")
API_URL = f"{OLLAMA_BASE_URL}/api/chat"


# ============================================================================
# CLI：process / index / serve
# ============================================================================
def cmd_process():
    print("=" * 60)
    print("阶段一: 处理数据 (清洗 + 分块)")
    print("=" * 60)
    RE.build_chunks()


def cmd_index():
    print("=" * 60)
    print("阶段二: 向量化入库")
    print("=" * 60)
    RE.build_index()


def _port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def cmd_serve():
    import chromadb

    client = chromadb.PersistentClient(path=RE.DB_PATH)
    try:
        count = client.get_collection("books_v2").count()
    except Exception:
        count = 0

    if count == 0:
        print("错误: 向量数据库为空")
        print("请先运行:")
        print("  python app.py process")
        print("  python app.py index")
        sys.exit(1)

    print(f"向量数据库: {count} 条记录")

    if _port_in_use(8501):
        print("Streamlit 服务已在运行: http://localhost:8501")
        print("如需重启，请先停止旧进程（可用: pkill -f 'streamlit run app.py'）")
        sys.exit(0)

    print("启动 Streamlit 服务...")
    print("服务地址: http://localhost:8501  （按 Ctrl+C 停止）")
    print(flush=True)

    try:
        with open("streamlit.pid", "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    # exec 替换当前进程为 streamlit：避免残留父进程
    os.execv(sys.executable, [
        sys.executable, "-m", "streamlit", "run", __file__,
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
        "--server.headless", "true",
    ])


# ============================================================================
# Streamlit UI
# ============================================================================
def run_streamlit():
    import streamlit as st
    import requests

    st.set_page_config(page_title="四大名著知识库", layout="wide")
    st.title("📚 四大名著知识库")

    # 引擎单例（st.cache_resource 跨 rerun 持久化，模型只加载一次）
    @st.cache_resource(show_spinner="加载检索引擎...")
    def get_engine():
        return RE.RAGEngine()

    engine = get_engine()

    # ---- 索引检查 ----
    if engine._get_collection().count() == 0:
        st.error("向量数据库为空！请先构建索引:")
        st.code("python app.py process\npython app.py index", language="bash")
        st.stop()

    # ---- 会话历史 ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- 用户输入 ----
    if prompt := st.chat_input("请输入关于四大名著的问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 1) RAG 检索
        with st.spinner("检索中..."):
            results = engine.hybrid_search(prompt, top_k=RE.TOP_K)

        context_parts = []
        sources = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result['parent_text']}")
            sources.append({
                "book": result["book"],
                "score": result["rerank_score"],
                "text": result["child_text"],
            })
        context = "\n\n".join(context_parts)
        sources_text = "\n".join(f"[{i + 1}] {s['book']}" for i, s in enumerate(sources))

        # 2) 构造 LLM 请求（带历史上下文）
        system_prompt = "你是一个知识库助手，专门回答关于四大名著的问题。"
        if context:
            system_prompt += (
                f"\n\n规则：\n1. 回答时标注引用来源，格式为 [数字]\n"
                f"2. 如果上下文中没有相关信息，请说明\n"
                f"3. 结合多个来源综合回答\n\n上下文：\n{context}\n\n来源：\n{sources_text}"
            )

        ollama_messages = [{"role": "system", "content": system_prompt}]
        for m in st.session_state.messages[:-1]:
            ollama_messages.append({"role": m["role"], "content": m["content"]})
        ollama_messages.append({"role": "user", "content": prompt})

        payload = {
            "model": MODEL,
            "messages": ollama_messages,
            "stream": True,
            "think": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
        }

        # 3) 流式输出
        with st.chat_message("assistant"):
            response = st.write_stream(_stream_ollama(payload))

        st.session_state.messages.append({"role": "assistant", "content": response})

        # 4) 展示引用来源
        if sources:
            with st.expander("📖 引用来源"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {source['book']}** (相关度: {source['score']:.4f})")
                    st.text(source["text"][:150] + "..." if len(source["text"]) > 150 else source["text"])
                    st.divider()


def _stream_ollama(payload):
    import requests
    try:
        r = requests.post(API_URL, json=payload, stream=True, timeout=180)
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                yield delta
    except Exception as e:
        yield f"请求失败: {e}"


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    import streamlit as st
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _is_streamlit = get_script_run_ctx() is not None
    except Exception:
        _is_streamlit = False

    if _is_streamlit:
        run_streamlit()
    elif len(sys.argv) < 2:
        cmd_serve()
    else:
        command = sys.argv[1].lower()
        if command == "process":
            cmd_process()
        elif command == "index":
            cmd_index()
        elif command == "serve":
            cmd_serve()
        else:
            print(f"未知命令: {command}")
            print("用法:")
            print("  python app.py process   处理数据")
            print("  python app.py index     向量化入库")
            print("  python app.py           启动查询服务")
            sys.exit(1)
