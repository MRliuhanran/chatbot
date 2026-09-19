#!/usr/bin/env python3
"""
四大名著知识库 - 统一入口（处理 / 索引 / 服务）

用法:
  python app.py process   处理数据(读取+分块)
  python app.py index     向量化入库
  python app.py           启动查询服务

检索引擎实现在 rag_engine.py，本文件只负责 CLI 编排与 Streamlit 展示。
"""

import os
import sys

# .env 必须在读任何环境变量之前加载（单源，见 bootstrap）。
import bootstrap  # noqa: E402,F401

import rag_engine as RE  # noqa: E402
import rag_turn as RT  # noqa: E402

# 检索侧常量从 rag_engine 取（RE.XXX），生成侧（Ollama）常量从 ollama_client 取。
# 本模块**不再自己读任何环境变量**：此前 MODEL / THINK / NUM_CTX / NUM_PREDICT /
# OLLAMA_TIMEOUT 在 app.py 与 ollama_client.py 里各读一遍，同一份配置出现两个值
# （典型后果：`OLLAMA_TIMEOUT` 只对 bot.py 生效，app/ui/api 三处仍是硬编码 180s）；
# RAG_USE_CONTEXT 则只在 app.py 读、其它入口靠转发（chat.py 就漏过）。
#
# ---- 思维链（think）与配套预算 --------------------------------------------
# qwen3 系列支持 think=true：Ollama 把推理过程放在 message.thinking，答案放在
# message.content，两者分开返回，所以前端可以分别落进两个区域而不混淆。
#
# 代价是实打实的，且实测数字很硬（qwen3.5:4b-q4_K_M，本机，一条 RAG 问题）：
#   * 思考约 5400 字 / 2000 token，整轮 ~135s；
#   * num_predict 是「思考 + 答案」的**总预算**。设成旧的 512 或 2048 时，
#     模型会把预算全部烧在思考上、还没开始写答案就被切断 ——
#     done_reason=length，答案长度为 0（真实复现过，不是理论风险）。
#     实测一次完整思考需要 2008 token，故默认给 4096，留一倍余量。
# 所以 num_predict / num_ctx 是跟 think 绑定的，不能只改一个。
# 想关掉思维链换速度：在 .env 里设 RAG_THINK=0，两项预算自动回落到 512/4096。
from ollama_client import NUM_CTX, THINK  # noqa: E402

# 是否把检索结果真正注入生成（0 = 对照模式）。取值来自 rag_engine，本模块只转发。
RAG_USE_CONTEXT = RE.RAG_USE_CONTEXT


# ============================================================================
# CLI：process / index / serve
# ============================================================================
def cmd_process():
    print("=" * 60)
    print("阶段一: 处理数据 (读取 + 分块)")
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


def _streamlit_pids():
    """正在跑的 streamlit 服务进程 PID（排除自己）。"""
    import subprocess

    try:
        out = subprocess.run(["pgrep", "-f", r"streamlit run .*app\.py"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(p) for p in out.split() if p.isdigit() and int(p) != os.getpid()]


def cmd_serve():
    from qdrant_client import QdrantClient

    # 连接Docker服务
    client = QdrantClient(host=RE.QDRANT_HOST, port=RE.QDRANT_PORT)
    # 连接失败与"集合为空"必须分开报。旧实现用 `except Exception: count = 0`
    # 把两者吞成同一个分支，于是 Docker 没起时用户看到的是"向量数据库为空，
    # 请先 process + index" —— 排查方向被彻底带偏（他会去重跑 70 分钟的索引，
    # 而真正的问题是 Qdrant 没启动）。check_health.py 里专门写注释防过这个坑，
    # 这里恰好是它的镜像错误。
    # 集合名运行时解析（别名优先），否则启用别名后这里会去查一个已被删除的旧名，
    # 把"索引好好的"误报成"向量数据库为空"
    coll = RE.default_collection_name(client)
    try:
        if not client.collection_exists(coll):
            count = 0
        else:
            count = client.count(collection_name=coll).count
    except Exception as exc:
        print(f"错误: 连不上向量数据库 {RE.QDRANT_HOST}:{RE.QDRANT_PORT}")
        print(f"  原因: {type(exc).__name__}: {exc}")
        print("  请先确认服务已启动: docker compose up -d")
        sys.exit(1)

    if count == 0:
        print("错误: 向量数据库为空")
        print("请先运行:")
        print("  python app.py process")
        print("  python app.py index")
        sys.exit(1)

    print(f"向量数据库: {coll} {count} 条记录")
    print(f"Qdrant服务: {RE.QDRANT_HOST}:{RE.QDRANT_PORT}")

    if _port_in_use(8501):
        # 端口有监听者 ≠ Streamlit 服务在运行：别的程序占用 8501 时，
        # 旧实现会打印"已在运行"并 exit 0 —— 用户以为服务起着，其实没起。
        pids = _streamlit_pids()
        if not pids:
            print(f"错误: 端口 8501 已被其它进程占用，但没找到本项目启动的 Streamlit。")
            print("  请先确认占用者: lsof -nP -iTCP:8501 -sTCP:LISTEN")
            sys.exit(1)
        print("Streamlit 服务已在运行: http://localhost:8501")
        # 旧提示写的是 pkill -f 'streamlit run app.py'，**匹配不到任何进程**：
        # 真实命令行里 app.py 是绝对路径（-m streamlit run /Users/…/app.py）。
        # 照做的人会以为已经重启，实际旧进程还活着、继续跑启动时导入的旧模块 ——
        # 表现为改完代码后报 "unexpected keyword argument" 这类错：界面（主脚本，
        # 会被文件监听重跑）已经是新的，rag_engine 还是进程启动那天的那份。
        # 所以这里给的是真能匹配的模式，并把 PID 一并报出来。
        print("改过代码务必重启：进程里跑的是启动时导入的旧模块。")
        print(r"  重启: pkill -f 'streamlit run .*app\.py' && python app.py")
        print(f"  当前 PID: {', '.join(map(str, pids))}")
        sys.exit(0)

    print("启动 Streamlit 服务...")
    print("服务地址: http://localhost:8501  （按 Ctrl+C 停止）")
    print(flush=True)

    # exec 替换当前进程为 streamlit：避免残留父进程
    os.execv(sys.executable, [
        sys.executable, "-m", "streamlit", "run", __file__,
        # 只监听回环地址：仅本机可访问，局域网/外网访问不到。
        # 原先绑 0.0.0.0 会把服务暴露给同网段的所有机器。
        "--server.address", "127.0.0.1",
        "--server.port", "8501",
        "--server.headless", "true",
    ])


# ============================================================================
# Streamlit UI
# ============================================================================
# 页面上只有四样东西：输入、输出、RAG 过程、思维链。
#
# 不放任何说明性文字 —— 没有标题、没有分区名称、没有耗时与 token 统计、
# 没有截断提示、没有低证据提示。过程区与思维链的标题各自只有一个图标：
# 运行中展开、结束后自动收起，要回看就点一下。
#
# 两处例外，都跟"界面不许说谎"有关，不是装饰：
#   * 空库 / 生成失败必须把原因原样说出来 —— 否则页面只剩一片空白，
#     分不清是没反应还是坏了（D7、D8 就是这么栽的）；
#   * 截断与低证据改挂到图标上（🔍 ⚠️ / 🧠 ⚠️），信息还在但不占版面。
#
# 宽度不靠 CSS：`client.showSidebarNavigation = false` 在 `.streamlit/config.toml`
# 里把左侧栏关掉，主区与底部输入容器就都拿到 100% 宽度（这是 Streamlit 官方
# 支持项，不是样式 hack）。见该文件里的注释。


# ---- 推理链里不展示通道召回 ------------------------------------------------
# 「单通道召回」这一步（[{channel, rank, point_id}, …]，dense/sparse 两路各
# RECALL_LIMIT 条）整步去掉；
# 「RRF 融合」候选里的 resource（"dense#1" / "sparse#7"）也一并去掉 —— 那是
# 同一份"谁召回的、在各自通道排第几"的信息，留着等于没删。
#
# 只改展示、不改数据：steps 本身仍然完整（api.py 的 /search 拿到的还是原样），
# 需要时随时能查。写进 session_state 之前另走 _slim_trace（见那里的说明）。
_HIDDEN_STEPS = (RE.STEP_RECALL,)
_HIDDEN_CANDIDATE_KEYS = ("resource",)
_RRF_STEP = RE.STEP_RRF


def _visible_trace(steps):
    """按展示规则过滤推理链，返回新 dict（不改动传入的 steps）。"""
    visible = {}
    for name, value in steps.items():
        if name in _HIDDEN_STEPS:
            continue
        if name == _RRF_STEP and isinstance(value, list):
            value = [
                {k: v for k, v in item.items() if k not in _HIDDEN_CANDIDATE_KEYS}
                if isinstance(item, dict) else item
                for item in value
            ]
        visible[name] = value
    return visible


def _slim_trace(steps):
    """写进 session_state 之前把推理链瘦身（不改动传入的 steps）。

    为什么必须瘦身：`steps` 一轮的体量约 20~40KB —— 里面同时存着
    `检索汇总`（5 条 parent_text，每条 ≤512 token）与 `RRF 融合`（20 条
    child_text，与前者大量重叠）。整份塞进 session_state 之后，**每一轮 rerun
    都要对全部历史回合重新构造与序列化一遍**（st.status 的折叠只是视觉，
    不省这层开销），随会话线性变慢。

    保留的是"事后还能判断发生了什么"的最小集合：每步的**计数与名次**，
    丢掉的只是正文全文（正文在 `检索汇总` 的 id/章节里仍可追回）。

    ⚠️ 不能删 `上下文装配 → messages`：那是"资料到底进没进 prompt"的唯一
    可断言证据（见 AGENTS.md「别把给模型的上下文寄生在 system 上」）。
    """
    slim = {}
    for name, value in steps.items():
        if name == RE.STEP_RESULTS and isinstance(value, list):
            slim[name] = [
                {
                    "id": r.get("id", ""),
                    "book": r.get("book", ""),
                    "chapter": r.get("chapter_label", ""),
                    "score": r.get("rerank_score"),
                    "chars": len(r.get("parent_text", "") or ""),
                }
                for r in value
            ]
        elif name == _RRF_STEP and isinstance(value, list):
            slim[name] = [
                {"id": item.get("id", ""), "score": item.get("score")}
                if isinstance(item, dict) else item
                for item in value
            ]
        elif name == RE.STEP_RERANK and isinstance(value, list):
            slim[name] = [
                {"id": item.get("id", ""), "fused_rank": item.get("fused_rank"),
                 "score": item.get("score"), "tokens": item.get("tokens")}
                if isinstance(item, dict) else item
                for item in value
            ]
        elif name == RE.STEP_CONTEXT and isinstance(value, dict):
            # context 是**整份资料正文**，与「检索汇总」的 parent_text 完全重叠；
            # 只留长度与 sources。⚠️ messages 必须原样保留：它是"资料到底进没进
            # prompt"的唯一可断言证据（见 AGENTS.md 的同名坑）。
            slim[name] = {k: v for k, v in value.items() if k != "context"}
            if "context" in value:
                slim[name]["context_chars"] = len(value.get("context") or "")
        else:
            slim[name] = value
    return slim


def _render_trace(steps):
    """推理链原样渲染：键是步骤名，值是该步端到端跑出来的产出。

    steps 是有序 dict，形如 {"稀疏编码 (jieba + BM25)": [...], "RRF 融合": [命中…]}，
    值就是产出本身，中间不再包一层 {"耗时": …, "输出": …}，也没有任何描述性字段。
    所以直接 st.json：不排版、不解析、不硬编码步骤名 —— 检索链路加减步骤时
    前端不用跟着改（唯一的例外是上面那份"隐藏通道召回"清单）。

    expanded=1：只展开到「步骤名 → 产出」这一层，768 维向量、几十条候选这些
    长数组默认收起，需要时点开看，避免整页一次性铺开。
    """
    import streamlit as st

    st.json(_visible_trace(steps), expanded=1)


def _render_assistant_extras(msg):
    """历史回合：过程与思维链一律收起，页面上只留答案。"""
    import streamlit as st

    if msg.get("trace"):
        with st.status("🔍", state="complete", expanded=False):
            _render_trace(msg["trace"])

    if msg.get("thinking"):
        with st.status("🧠", state="complete", expanded=False):
            st.markdown(msg["thinking"])


def run_streamlit():
    import streamlit as st

    # 宽度全靠官方配置，不写一行样式：
    #   layout="wide"          主区与底部输入容器不再钉在 ~736px
    #   .streamlit/config.toml 里 client.showSidebarNavigation=false 关掉侧栏
    # 两者都是 Streamlit 官方支持项，改版不会让它们失效（CSS 选择器会）。
    st.set_page_config(page_title="chat", layout="wide")

    # 引擎单例（st.cache_resource 跨 rerun 持久化，模型只加载一次）
    @st.cache_resource(show_spinner=False)
    def get_engine():
        return RE.RAGEngine()

    engine = get_engine()

    # ---- 索引检查 ----
    if engine.count() == 0:
        st.error("向量库为空，先跑：python app.py process && python app.py index")
        st.stop()

    # ---- 会话历史 ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_extras(msg)
            if msg["content"]:
                st.markdown(msg["content"])
            elif msg["role"] == "assistant" and msg.get("error"):
                st.error(msg["error"])
            elif msg["role"] == "assistant":
                # 空回合必须留个记号：否则历史里只剩一个空气泡，
                # 与"界面卡住了"分不出来。用图标，不写字。
                st.markdown("⚠️")

    # ---- 输入 ----
    if prompt := st.chat_input(""):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            rag_status = st.status("🔍", expanded=True)

            # 一轮 RAG 的序列在 rag_turn.run_turn 里定义（检索 → 置信度 → 装配 →
            # 拒答判定 → 流式生成），本入口只把事件画出来。
            #
            # **检索发生在第一次迭代时**（生成器语义）：所以这里显式 next() 一次
            # 来拿"检索完成"事件，之后才进渲染 —— 顺序与"先检索、再画推理链、
            # 再流式输出"完全一致。
            #
            # 传 history 才会触发多轮查询改写（指代消解）。不传的话
            # "他最后结局如何"这类问句会以字面检索，实测召回到
            # 西游记/红楼梦/红楼梦/红楼梦/水浒传 —— 全无关。
            # st.session_state.messages 此刻已含本轮 user 提问，故取 [:-1]。
            events = RT.run_turn(
                engine, prompt, st.session_state.messages[:-1], top_k=RE.TOP_K,
            )
            kind, payload = next(events)
            results, steps = payload["results"], payload["steps"]
            confidence, gen = payload["confidence"], payload["gen"]
            ollama_messages = payload["messages"]

            with rag_status:
                _render_trace(steps)
            rag_status.update(
                label=("🔍 ⚠️" if (RAG_USE_CONTEXT and confidence.get("refuse")
                                   and results) else "🔍"),
                state="complete",
                expanded=False,
            )

            # 检索为空时不让模型自由发挥 —— 判据由 run_turn 给出（三入口同一条）。
            if kind == RT.EVENT_ABSTAIN:
                st.markdown(payload["answer"])
                st.session_state.messages.append({
                    "role": "assistant", "content": payload["answer"],
                    "thinking": "", "error": "", "trace": _slim_trace(steps),
                })
                st.stop()

            if gen["truncated_user"]:
                # 用户提问本身被截断是**例外中的例外**（见 plan_generation 的
                # 说明）：不截就会超出 num_ctx，服务端会整条丢消息。这里必须
                # 显式告知，否则用户会以为自己问的话模型都收到了。
                st.warning(
                    f"⚠️ 本轮提问过长（{RE.estimate_tokens(prompt)} token），"
                    f"已按上下文预算截断后发送；模型没有看到完整提问。"
                    f"如需完整处理，请调大 RAG_NUM_CTX 或分次提问。"
                )

            # ================================================================
            # 2) 思维链 + 输出
            #
            # think=true 时 Ollama 把 message.thinking 与 message.content 分成
            # 两路返回，所以这里不能用 st.write_stream（它只吃一路文本），
            # 改为按事件类型分别写入两个占位符，仍然逐 token 流式。
            # RAG_THINK=0 时连这个容器都不建，页面少一个死元素。
            # ================================================================
            think_status, think_ph = None, None
            if THINK:
                think_status = st.status("🧠", expanded=True)
                with think_status:
                    think_ph = st.empty()
            answer_ph = st.empty()

            thinking, answer, stats = "", "", {}
            gen_error = ""
            for kind, payload in events:
                if kind == RT.EVENT_THINKING:
                    thinking += payload
                    if think_ph is not None:
                        think_ph.markdown(thinking)
                elif kind == RT.EVENT_DELTA:
                    answer += payload
                    answer_ph.markdown(answer)
                elif kind == RT.EVENT_ERROR:
                    gen_error += payload
                elif kind == RT.EVENT_DONE:
                    stats = payload

            if gen_error:
                answer_ph.error(gen_error)
            elif not answer:
                # 空答案必须留记号：否则聊天框一片空白，看不出是模型没写、
                # 还是请求根本没通。实测绝大多数情况是 num_predict 预算
                # 被思考烧光了（见文件顶部 THINK 注释）。
                answer_ph.markdown("⚠️")

            # 截断判定必须**在 if think_status 之外**：关掉思维链时
            # think_status 为 None，旧写法让整块（含 done_reason=length 的告警）
            # 一起不执行 —— 而"答案被 num_predict 截断"是关掉 think 之后
            # 唯一可达的截断信号，它消失意味着答案看起来正常、其实被砍了半句。
            truncated = stats.get("done_reason") == "length"
            if think_status is not None:
                if thinking:
                    think_ph.markdown(thinking)
                think_status.update(
                    label="🧠 ⚠️" if truncated else "🧠",
                    state="error" if truncated else "complete",
                    expanded=False,
                )
            if truncated:
                st.warning(
                    "⚠️ 生成被 num_predict 截断（done_reason=length）："
                    "思考与答案共用这一个预算，答案可能没写完。"
                    "可调大 RAG_NUM_PREDICT 或关闭 RAG_THINK。"
                )

            # ---- 用真实 token 数反查估算，并检查有没有撞上窗口 ----
            # prompt_eval_count 是 Ollama 实测的输入 token 数，而历史预算靠
            # estimate_tokens 估算（刻意取上界）。两者一比就知道预算这一层
            # 有没有失准：
            #   * 实测逼近 num_ctx → 窗口吃紧，服务端会开始整条丢消息
            #     （实测不报错、保留 system、其余随便丢），必须报警；
            #   * 实测远小于估算 → 估算过于保守，白丢了几轮历史。
            # 两种情况在界面上都是"回答看着正常"，只能靠这两个数发现。
            #
            # 这是**唯一由入口写入 steps 的键**：它来自生成之后的实测值，
            # 检索链不可能知道；其余步骤的写权都在 rag_engine。
            prompt_tokens = stats.get("prompt_eval_count")
            est_tokens = RE.message_tokens(ollama_messages)
            if prompt_tokens:
                steps[RE.STEP_INPUT_TOKENS] = {
                    "Ollama 实测": prompt_tokens,
                    "本地估算": est_tokens,
                    "num_ctx": NUM_CTX,
                    "余量": NUM_CTX - prompt_tokens,
                }
                if prompt_tokens >= NUM_CTX - RE.GEN_INPUT_RESERVE:
                    st.warning(
                        f"⚠️ 本轮输入 {prompt_tokens} token，已逼近 num_ctx={NUM_CTX}："
                        f"服务端会开始**整条丢弃**消息（实测不报错、system 保留、"
                        f"其余随便丢），历史会静默少掉几轮。请调大 RAG_NUM_CTX。"
                    )
                if est_tokens > prompt_tokens * 1.5:
                    st.info(
                        f"ℹ️ 本地估算 {est_tokens} token、实测 {prompt_tokens}："
                        f"估算偏保守（中文实测约 0.77 字/token），"
                        f"历史预算可以调大一些。"
                    )

            st.session_state.messages.append({
                "role": "assistant",
                # 生成失败时不把报错写成"助手说过的话"：它会被下一轮拼进
                # ollama_messages 当成模型历史回灌。宁可存空串，让历史保持"无回答"。
                "content": "" if gen_error else answer,
                "thinking": thinking,
                "error": gen_error,
                # 存**瘦身后**的 trace：完整 steps 里同一批正文存了两份
                # （检索汇总的 parent_text + RRF 融合的 child_text），约
                # 20~40KB/轮，而每轮 rerun 都会对全部历史重新构造一遍。
                "trace": _slim_trace(steps),
            })

# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
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
