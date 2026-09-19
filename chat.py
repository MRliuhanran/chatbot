#!/usr/bin/env python3
"""最简对话机器人：只有一问一答，没有任何诊断界面。

    streamlit run chat.py

一轮 RAG 的序列（检索 → 装配 → 拒答 → 流式生成）由 rag_turn.run_turn 定义，
与 app.py / api.py 是同一份；本页只把 delta 画出来，其余事件一概丢弃。
要看推理链、思维链、token 统计，用 `python app.py`。
"""

import streamlit as st

# 生成侧常量与 RAG 开关都由 rag_turn / rag_engine 的缺省值提供，本页**一个都不用
# 读** —— 此前它要从 app.py 转手，于是漏过 RAG_USE_CONTEXT：.env 置 0 时别的入口
# 走对照模式、这里照样注入检索正文。
import rag_engine as RE
import rag_turn as RT

st.set_page_config(page_title="对话", layout="centered")


@st.cache_resource(show_spinner=False)
def get_engine():
    return RE.RAGEngine()


engine = get_engine()

if engine.count() == 0:
    st.error("向量库为空，先跑：python app.py process && python app.py index")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])
    if m.get("error"):
        st.error(f"生成失败：{m['error']}")

if question := st.chat_input("问点什么"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        # 检索与生成的**序列**在 rag_turn.run_turn 里定义（与 app.py / api.py
        # 同一份），本页只把事件画出来 —— 这正是"一个字都不显示"的极简入口
        # 该有的样子：它不再需要知道 use_context / 硬拒答 / 历史预算这些细节。
        engine = get_engine()
        events = RT.run_turn(engine, question, st.session_state.messages[:-1])
        kind, payload = next(events)        # ← 检索在这里发生

        box = {"error": ""}
        if kind == RT.EVENT_ABSTAIN:
            text = payload["answer"]
            st.markdown(text)
        else:
            # 只把答案喂给 write_stream；thinking 直接丢掉。
            # 报错走独立事件，绝不混进答案 —— 它会作为"助手说过的话"写进历史，
            # 下一轮回灌给模型（见 app.py 的 D8 说明）。
            def answer():
                for kind, payload in events:
                    if kind == RT.EVENT_DELTA:
                        yield payload
                    elif kind == RT.EVENT_ERROR:
                        box["error"] += payload

            text = st.write_stream(answer())
            if box["error"]:
                st.error(f"生成失败：{box['error']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": "" if box["error"] else (text or ""),
        # error 必须**存进历史**：只 st.error 的话，任何一次 rerun（下一次提问、
        # 页面刷新）之后这一回合就只剩一个没有提示的空气泡，分不清"模型没写"
        # 还是"请求失败"。app.py / bot.py 都存了，这里此前漏了。
        "error": box["error"],
    })
