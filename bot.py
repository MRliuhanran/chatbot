#!/usr/bin/env python3
"""通用聊天机器人 —— 界面上只有三样东西：输入、输出、过程。

    streamlit run bot.py

刻意不做的事：没有标题、没有设置面板、没有会话管理、没有引用出处、
没有 token 统计、不挂任何知识库。想换模型/人设改 .env 即可。
"""

import streamlit as st

from ollama_client import SYSTEM_PROMPT, THINK, stream_chat
# 只有两个纯函数：历史裁剪与空回合过滤。本文件不做检索，但**不能因此不裁历史**
# —— 超窗时 Ollama 不报错、保留 system、其余整条丢（本项目实测：一条 16000 字
# 的消息让 prompt_eval_count 从 18743 字塌到 40 token），长会话下回答照样通顺、
# 只是凭空少掉上下文，客户端零信号。"界面只有三样东西"限制的是 UI，不是上下文管理。
import rag_engine as RE

st.set_page_config(page_title="chat", layout="centered")

# 连 Streamlit 自己的外框也去掉：右上角菜单、顶部工具条、页脚。
# 默认它们会占掉一截留白，而这里要求整页只有三样东西。
st.markdown(
    """
    <style>
      #MainMenu, header, footer, [data-testid="stToolbar"] {display: none;}
      .block-container {padding-top: 2.5rem; padding-bottom: 6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_history(msg):
    """历史回放：过程默认收起，答案常显。"""
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.status(f"过程 · {len(msg['thinking'])} 字",
                           state="complete", expanded=False):
                st.markdown(msg["thinking"])
        if msg.get("error"):
            st.error(f"生成失败：{msg['error']}")
        else:
            st.markdown(msg["content"])


if "messages" not in st.session_state:
    st.session_state.messages = []

for _msg in st.session_state.messages:
    render_history(_msg)


if question := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages]          # 已含本轮提问

        # 与另外三个入口同一套装配：丢空回合（上一轮生成失败会留下
        # content="" 的助手消息，回灌给模型等于让它以为"我上一轮什么都没说"）、
        # 按 token 预算整轮裁历史、system 与本轮提问永不丢弃。
        # num_ctx/num_predict 必须先扣掉 system 与本轮提问，否则预算算不对。
        from ollama_client import NUM_CTX, NUM_PREDICT
        budget = RE.history_budget(NUM_CTX, NUM_PREDICT, SYSTEM_PROMPT,
                                   extra=question)
        gen = RE.build_generation_messages(
            SYSTEM_PROMPT, history[:-1], question,
            max_history_tokens=budget,
        )
        messages = gen["messages"]
        if gen["dropped_turns"] or gen["truncated"]:
            # 裁剪必须可见：静默丢历史是"回答看着正常却没了依据"的成因。
            st.caption(
                f"已按上下文预算丢弃 {gen['dropped_turns']} 轮历史"
                + ("（本轮文本被截断）" if gen["truncated"] else "")
            )

        # ---- 过程区：思维链逐字上屏，答完自动收起（不消失，可点开回看）----
        process = st.status("过程", expanded=True) if THINK else None
        process_ph = process.empty() if process else None

        # ---- 输出区 ----
        answer_ph = st.empty()

        thinking, answer, error = "", "", ""
        for kind, delta in stream_chat(messages):
            if kind == "thinking":
                thinking += delta
                # THINK=0 时没有过程容器；此处必须判空 —— app.py 的同一位置
                # 有这层保护，这里此前漏了（think 关闭却仍收到 thinking 时崩）。
                if process_ph is not None:
                    process_ph.markdown(thinking)
            elif kind == "content":
                answer += delta
                answer_ph.markdown(answer)
            elif kind == "error":
                error += delta          # 独立通道，绝不混进 answer

        if process:
            process.update(label=f"过程 · {len(thinking)} 字",
                           state="complete", expanded=False)
        if error:
            answer_ph.error(f"生成失败：{error}")
        elif not answer:
            answer_ph.caption("本轮没有产出内容（多为 num_predict 预算被思考耗尽）")

    st.session_state.messages.append({
        "role": "assistant",
        # 生成失败时存空串：报错一旦写进历史，下一轮会被当成模型说过的话回灌
        "content": "" if error else answer,
        "thinking": thinking,
        "error": error,
    })
