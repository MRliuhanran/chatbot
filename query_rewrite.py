#!/usr/bin/env python3
"""多轮查询改写 —— 把指代性问句补全成可独立检索的问句。

## 为什么需要它

检索此前直接吃用户的原始输入，对话历史只喂给生成模型。于是多轮场景下
指代性问句完全失效。实测（集合 books_v3，原始链路）：

    Q: "他最后结局如何"
    → 召回 西游记 / 红楼梦 / 红楼梦 / 红楼梦 / 水浒传
    → rerank 分数 [0.834, 0.693, 0.656, 0.497, 0.157]（全线低分，即全部无关）

"他"没有先行词，稠密向量只能退化成"询问某人结局"这个泛化语义，
四本书里随便挑。而检索侧拿到失败信号后**没有任何补救手段** ——
生成侧再强也救不回没召回的正文。

## 设计取舍

* **用本地 Ollama，不引入新依赖**：改写是一次很短的前向（num_predict=64，
  think=false），比走外部 API 更可控，也不会让"检索"变成需要联网的操作。
* **失败必须退化为原查询**：改写是**增益**手段，不是必需品。Ollama 没起、
  超时、返回垃圾，都必须退回原始问句继续检索，绝不能把检索本身搞挂。
  这条是硬约束，因此所有异常都在这里被吞掉并留下警告日志。
* **产物要能单测**：提示词构造（build_rewrite_prompt）与清洗（clean_rewritten）
  是纯函数，可上 L0；真正发 HTTP 的 _ollama_chat 通过 call 参数注入，
  测试不需要 Ollama 在跑。
* **阈值护栏**：改写结果若为空、过长（>MAX_REWRITE_CHARS）、或与原文相同，
  一律退回原查询。模型偶尔会输出"改写后的问句：…"这类解释性文字，
  由 clean_rewritten 剥掉。
* **三态必须可分辨**：`rewrite_query_detailed` 除了问句还返回
  `applied` 与 `reason`（见下面的 REASON_* 常量）。原因是这三种情况
  ——「开关关了」「Ollama 调用失败」「模型输出不可用」—— 以前在界面上
  完全一样（steps 里没有"查询改写"键），排查时只能猜，A/B 时甚至分不清
  "改写没用"和"改写没跑"。`rewrite_query` 保留原签名，内部委托给它。
"""

import os
import re

# ---------------------------------------------------------------------------
# .env 与日志都交给 bootstrap（单源）。
#
# 本模块在 **import 时**就把开关与模型名读成模块常量，而 rag_engine 顶层就
# `from query_rewrite import ...`。此前这里自己 load_dotenv()，而 rag_engine 也
# 各写了一份 —— "谁先被 import"决定 .env 生不生效（D9），于是 check_health.py
# 把"已关闭"报成"开启"，tools/ab_retrieval.py 与 record_baseline 里 .env 的 0
# 一律失效（"关掉改写做 A/B"实际上根本没关）。
# 现在两处都改为 import bootstrap：加载只发生一次，顺序不再重要。
# ---------------------------------------------------------------------------
import bootstrap  # noqa: E402,F401  (必须在读取任何环境变量之前)
from bootstrap import env_bool, get_logger  # noqa: E402

# 与 rag_engine 共用同一个 logger：日志顺序输出一份，便于按时间对齐
logger = get_logger("rag_engine")


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# 改写用模型：默认与生成侧同一个（.env 的 MODEL），避免多占一份显存 ——
# 8GB 统一内存上同时驻留两个模型是实打实的风险。
REWRITE_MODEL = os.getenv("RAG_REWRITE_MODEL") or os.getenv("MODEL", "qwen3.5:4b-q4_K_M")
REWRITE_ENABLED = env_bool("RAG_QUERY_REWRITE", True)
REWRITE_TIMEOUT = int(os.getenv("RAG_QUERY_REWRITE_TIMEOUT", "20"))
# 历史窗口：**默认 0 = 不限制**，整段对话历史都进改写 prompt。
#
# 为什么改成不限制（2026-09-19）：按轮数硬砍有两个说不清的地方 ——
#   ① "最近 4 轮"是拿**消息条数**切（[-max_turns*2:]），历史一出现奇数长度
#      （末尾是 user、或出错留下的空气泡被 history_to_messages 丢掉一条）
#      窗口就落到半轮上，"轮"这个单位本身不成立；
#   ② 它砍掉的恰恰是 LLM 档**独有**的能力 —— 触达两轮以前提到的实体。
#      拼接档只取最近一条用户问句（还截断 60 字），够不着更早的轮次，
#      所以"砍到 1 轮"没有任何替代品能补回来。
# 现在的语义是：**能看到多少，由模型的上下文窗口决定**（见 REWRITE_NUM_CTX），
# 超出窗口时由模型侧自然遗忘，而不是在这里按轮数提前丢掉。
#
# 这个环境变量保留下来只为 A/B 旧行为：>0 时退化成"只取最近 N 轮"。
REWRITE_MAX_TURNS = int(os.getenv("RAG_QUERY_REWRITE_MAX_TURNS", "0"))
MAX_REWRITE_CHARS = int(os.getenv("RAG_QUERY_REWRITE_MAX_CHARS", "120"))
#: 改写请求的上下文窗口（**显式**发给 Ollama）。
#: 不显式指定就用服务端默认 —— 那意味着"窗口多大"不可知，"超窗时丢什么"也不可知；
#: 显式指定后，"记不住"这件事至少发生在可预期的地方。
#: 默认跟随生成侧的 RAG_NUM_CTX，避免同一次会话里两个窗口不一致。
REWRITE_NUM_CTX = int(os.getenv("RAG_QUERY_REWRITE_NUM_CTX")
                      or os.getenv("RAG_NUM_CTX", "8192"))

#: 改写用的 system prompt。
#:
#: ⚠️ 这里曾被**刻意置空**（2026-09-18，按用户要求）。2026-09-19 的 P0 四态实测
#: 证明置空正是 LLM 改写档崩塌的根因，故恢复原文。同一台机器、同一模型、
#: 同样 21 条多轮探针，**只改这一处**：
#:
#:     置空   book@1 85.7%   kw@k 76.2%   topic@k 66.7%   （且 21 条里有 1 次 20s 超时）
#:     恢复   book@1 100%    kw@k 76.2%   topic@k 95.2%
#:
#: 行为证据比数字更直白 —— 置空后模型开始"回答问题"而不是改写：
#:   "他头上的箍儿是谁给他戴上的" → "给孙悟空戴上紧箍儿的是如来佛祖。"（直接作答）
#:   "他最后是被谁杀的"           → "请问这两位英雄分别是死于谁的刀下？"（把两个实体合并）
#:   "她哥哥叫什么名字"           → "她的哥哥叫什么名字？"（原句照抄，未消解）
#:
#: 复现脚本与完整四态表见 MULTITURN_PLAN.md §0.1。
#: 想再做"置空 vs 恢复"的 A/B：赋空串即可，但**置空与 RAG_QUERY_REWRITE=1
#: 不可同时成立** —— 那是四态里最差的一档（topic@k 66.7%）。
_SYSTEM_PROMPT = (
    "你是一个检索查询改写器。根据对话历史，把用户当前的问句改写成一个"
    "不依赖上下文、可独立检索的完整问句。\n"
    "规则：\n"
    "1. 只做指代消解与省略补全，不要回答问题，不要解释，不要添加原文没有的信息；\n"
    "2. 保留人名、书名、地名等专有名词的原写法；\n"
    "3. 若当前问句本身已完整（不含代词、不需要上文），原样输出；\n"
    "4. 只输出改写后的问句本身，不要任何前缀、引号或说明。"
)

# 模型常见的解释性前缀，例如 "改写后的问句：xxx"。剥掉它们，
# 否则前缀会作为噪声词进入 BM25 与稠密编码。
_PREFIX_RE = re.compile(r"^\s*(改写后(的)?(问句|查询)?|检索(问句|查询)|问句|答案)\s*[:：]\s*")

#: 明显的"元话语行"（不是问句本身）。只收最无歧义的一批：模型要写前言，
#: 用的几乎总是这几种开场。刻意不做穷举 —— 穷举必然漏，而漏的后果是
#: 把前言当检索问句（见 clean_rewritten_detailed）。
_META_LINE_RE = re.compile(
    r"^\s*(好的|好，|当然|可以|没问题|明白|收到|以下是|下面是|我来|让我|根据|"
    r"改写后|改写如下|简化为|整理后)")


def build_rewrite_prompt(query, history, max_turns=REWRITE_MAX_TURNS):
    """构造改写请求的 messages（纯函数，可单测）。

    **默认不限制轮数**（`max_turns<=0`）：整段历史都进 prompt，只看得到多少
    由模型上下文窗口决定，超窗由模型侧自然遗忘。`max_turns>0` 时退化为
    "只取最近 N 轮"，仅供 A/B 旧行为，生产默认不走这条路。

    历史里只保留 role/content —— 会话历史还挂着 thinking / trace / error 等大字段，
    原样塞进去会让 prompt 无谓地膨胀。
    """
    turns = list(history or [])
    if max_turns and max_turns > 0:
        turns = turns[-max_turns * 2:]
    lines = []
    for m in turns:
        # 非 dict 项必须跳过而不是崩：history 由调用方提供，API 只校验了它是
        # list（元素可以是 str/int），而本函数在 rewrite_prompt_stats 里被
        # **无条件**调用（连 RAG_QUERY_REWRITE=0 也照样调）—— 一个字符串元素
        # 就会让整条检索链抛 AttributeError。last_turn_text 一直有这层 isinstance
        # 守卫，此处是漏写。
        if not isinstance(m, dict):
            continue
        role = "用户" if m.get("role") == "user" else "助手"
        # content 可能是任意类型（API 不校验元素类型）：非 str 一律当空处理，
        # 不能让它把 `.strip()` 炸掉。
        raw_content = m.get("content")
        content = raw_content.strip() if isinstance(raw_content, str) else ""
        if not content:
            continue
        lines.append(f"{role}：{content}")
    dialogue = "\n".join(lines) if lines else "（无历史）"

    user = (
        f"对话历史：\n{dialogue}\n\n"
        f"用户当前问句：{query}\n\n"
        f"改写后的问句："
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# 改写结果的三态原因码
#
# 此前"开关关了""Ollama 挂了""模型没按格式输出"三种情况在调用方眼里**完全
# 一样**：steps 里干脆没有"查询改写"这个键。于是排查时只能猜，A/B 时也无法
# 分辨"改写没用"与"改写根本没跑"。原因码就是为这件事存在的，必须落进
# steps["查询改写"].reason 并展示到界面上。
# ---------------------------------------------------------------------------
REASON_DISABLED = "disabled"            # RAG_QUERY_REWRITE=0
REASON_NO_HISTORY = "no_history"        # 首轮 / 调用方没传历史
REASON_BLANK_QUERY = "blank_query"      # 本轮提问是空的
REASON_CALL_ERROR = "call_error"        # 调用失败（后面接 :异常类名）
REASON_EMPTY_OUTPUT = "empty_output"    # 模型没输出可用问句（含只输出前缀）
REASON_TOO_LONG = "too_long"            # 超过 MAX_REWRITE_CHARS，多半是没按要求输出
REASON_IDENTICAL = "identical"          # 与原文逐字相同 = 本就不需要改写
REASON_APPLIED = "applied"              # 真的改写了
#: 上面几种都不可用时，退到"拼接上一轮对话"这一档（见 build_retrieval_query）。
#: 它**不是失败**：实测 topic@k 从 42.9%（字面检索）提到 95.2%，且零延迟零失败面。
REASON_CONCAT = "concat_fallback"

#: 三态判定用：只有 applied 才会改变检索用问句。
REASONS = (
    REASON_DISABLED, REASON_NO_HISTORY, REASON_BLANK_QUERY, REASON_CALL_ERROR,
    REASON_EMPTY_OUTPUT, REASON_TOO_LONG, REASON_IDENTICAL, REASON_APPLIED,
    REASON_CONCAT,
)

# ---------------------------------------------------------------------------
# 降级链的中间档：拼接上一轮对话
#
# 现状是**两档**：LLM 改写成功 → 用改写句；失败/关闭 → 用字面原句。第二档
# 实测等于没有这个功能（21 条多轮探针：book@1 81.0%、topic@k 42.9%）。
# 中间这一档免费、不联网、不会失败，实测（同一套探针、同一评测器、只换检索用问句）：
#
#     检索用问句                    book@1   kw@k    topic@k
#     字面原句（旧降级落点）          81.0%   38.1%   42.9%
#     上轮用户 + 本轮                 100%    71.4%   95.2%   ← 本模块选它
#     上轮用户+助手 + 本轮            100%    57.1%   100%
#     LLM 改写                       100%    76.2%   95.2%
#
# 为什么选"只拼上轮用户"而不是再带上助手那句：topic@k 上两者打平（95.2 vs 100 只差
# 一条探针），但 kw@k 差了 14 个百分点 —— 助手回答用的是**自己的措辞**，把它拼进
# 查询会把语义平均掉，召回落到与回答相近、而不是与书中原文关键词相近的段落。
# 反过来，只拼用户轮的短板是"实体只出现在助手那一轮"（实测栽 1/5 条），
# 那种情况本来就该由 LLM 改写兜住（它能看到助手轮：实测 assistant_only 100%）。
# 两档的失手点恰好互补，所以是**降级链**而不是二选一。
# ---------------------------------------------------------------------------
CONCAT_ENABLED = env_bool("RAG_QUERY_REWRITE_CONCAT", True)
#: 哪些 reason 才允许落到拼接档。**白名单而非黑名单**：新增原因码时
#: 默认行为是"不拼接"，逼着人显式判断它该不该拼，而不是默默开始拼。
CONCAT_FROM_REASONS = (
    REASON_DISABLED,        # 开关关了：这是最常见的路径
    REASON_CALL_ERROR,      # Ollama 没起 / 超时
    REASON_EMPTY_OUTPUT,    # 模型没按格式输出
    REASON_TOO_LONG,        # 输出被判废
)
#: 拼接时只取上一轮用户问句的前若干字：真实对话里上一轮可能是长问句，
#: 全量拼进去会把当前问句的权重稀释掉（探针里的问句都很短，测不出这一点，
#: 所以这个上限是**保守设定**，不是实测最优值）。
CONCAT_MAX_CHARS = int(os.getenv("RAG_QUERY_REWRITE_CONCAT_CHARS", "60"))


def clean_rewritten_detailed(text):
    """清洗模型输出，并给出**为什么**不可用（纯函数，可单测）。

    返回 (问句, reason)：问句为空串时 reason 说明原因，否则 reason 为 ""。
    拆出原因是为了能区分"模型没输出"与"输出太长被判废"—— 两者的处置
    完全不同（前者要查模型/提示词，后者要查 MAX_REWRITE_CHARS 是否过紧），
    而它们的现象都是"退回原查询"。
    """
    if not text:
        return "", REASON_EMPTY_OUTPUT
    if not isinstance(text, str):
        # 注入的 call（测试/A-B）可能返回非 str；旧实现在 try 之外调用本函数，
        # 于是 `text.strip()` 的 AttributeError 会穿透"绝不抛异常"的契约。
        return "", REASON_EMPTY_OUTPUT
    out = text.strip()
    # 逐行找"真正的问句"：模型常先写一句元话语再给问句，而旧实现只取首行，
    # 于是 `好的，我来改写：` 会被当成检索问句，且 applied=True / source="llm"，
    # 推理链上显示"改写成功"——指标与界面同时被骗。
    # 判据不是穷举前缀（形态是开放的），而是"这一行是不是明显在说别的事"：
    #   ① 以冒号结尾（"好的，我来改写："）→ 后面那句才是正文；
    #   ② 去掉可能的前缀后仍以元话语开头 → 跳过。
    out = ""
    for line in text.strip().splitlines():
        line = _PREFIX_RE.sub("", line.strip()).strip()
        line = line.strip("“”\"'《》").strip()
        if not line:
            continue
        if line.endswith("：") or line.endswith(":"):
            continue          # 元话语行，真正的问句在下一行
        if _META_LINE_RE.match(line):
            continue
        out = line
        break
    if not out:
        return "", REASON_EMPTY_OUTPUT
    if len(out) > MAX_REWRITE_CHARS:
        return "", REASON_TOO_LONG
    return out, ""


def clean_rewritten(text):
    """清洗模型输出：去掉前缀、引号、换行，限制长度（纯函数，可单测）。

    返回空串表示"这次改写不可用"，调用方应退回原查询。
    需要知道**原因**时用 clean_rewritten_detailed。
    """
    return clean_rewritten_detailed(text)[0]


def _ollama_chat(payload):
    """向 Ollama 发一次非流式请求，返回 message.content。"""
    import requests

    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload,
                      timeout=REWRITE_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    return ((data.get("message") or {}).get("content") or "")


def _build_payload(query, history):
    """改写请求的完整 payload（纯函数）。

    抽出来的理由：详细版与兼容版必须发**一模一样**的请求，
    否则两条路径的 A/B 结果不可比。
    """
    return {
        "model": REWRITE_MODEL,
        "messages": build_rewrite_prompt(query, history),
        "stream": False,
        # 改写是抽取式任务，不需要推理链。开着 think 会让这一次额外前向
        # 从"零点几秒"涨到"几十秒"（实测生成侧 think 一轮 ~135s），
        # 那是用户完全无法接受的检索延迟。
        "think": False,
        # num_ctx 显式给出：历史不再按轮数截断后，"模型能看多少"必须是个已知数，
        # 否则超窗时服务端会静默丢内容，而丢的是什么没人知道。
        "options": {"temperature": 0, "num_predict": 64, "num_ctx": REWRITE_NUM_CTX},
    }


def rewrite_prompt_stats(query, history, max_turns=REWRITE_MAX_TURNS):
    """改写 prompt 的体量统计（纯函数）。

    存在的理由：历史窗口不再按轮数截断（默认全量），"这次 prompt 有多大、
    会不会超出模型窗口"就只能靠数字看，不能靠读代码猜。
    `over_window` 用字符数对 token 数做**保守粗估**（中文≈1 字/token，英文更省），
    它是"该报警了"的信号，不是精确的 token 计数 —— 这里刻意不引 tokenizer：
    改写发生在检索前，为了一次日志去加载 BGE 分词器不划算。
    """
    text = build_rewrite_prompt(query, history, max_turns=max_turns)[-1]["content"]
    # 历史行以 "用户："/"助手：" 开头；当前问句那行是 "用户当前问句："，不会被误计。
    # 多行 content 会被拼进同一行（见 build_rewrite_prompt），故这里统计的是
    # "消息条数"，不是"行数"—— 名字与语义一致，避免读数的人误以为按行数算。
    hist_msgs = sum(1 for ln in text.splitlines()
                    if ln.startswith("用户：") or ln.startswith("助手："))
    # system prompt 也是输入的一部分：旧实现只算 user 消息，于是 over_window
    # 在"历史恰好塞满、system 把窗口顶破"时漏报 —— 而 system 有 150 字左右。
    system_chars = len(_SYSTEM_PROMPT)
    total_chars = len(text) + system_chars
    return {
        "prompt_chars": total_chars,
        "history_msgs": hist_msgs,
        "num_ctx": REWRITE_NUM_CTX,
        "over_window": total_chars > REWRITE_NUM_CTX,
    }


def rewrite_query_detailed(query, history, call=None, enabled=None):
    """同 rewrite_query，但**把结果与原因一起返回**（绝不抛异常）。

    返回 {"query": 实际用于检索的问句, "applied": bool, "reason": str}。

    硬约束与 rewrite_query 完全一致：任何失败路径上 query 都原样退回，
    检索绝不会因为改写失败而失败。区别只在于 applied/reason 让调用方
    （以及界面）能分辨"关闭 / 没历史 / 调用失败 / 输出不可用 / 与原文相同"，
    而不是统统表现为"没有这个步骤"。
    """
    if enabled is None:
        enabled = REWRITE_ENABLED

    def no(reason):
        return {"query": query, "applied": False, "reason": reason}

    if not enabled:
        return no(REASON_DISABLED)
    if not history:
        return no(REASON_NO_HISTORY)
    if not (query or "").strip():
        return no(REASON_BLANK_QUERY)

    try:
        raw = (call or _ollama_chat)(_build_payload(query, history))
    except Exception as exc:
        # 改写是增益手段，失败绝不能连带检索一起失败
        logger.warning(f"⚠️  查询改写失败（交给降级链处理，检索不受影响）: "
                       f"{type(exc).__name__}: {exc}")
        return no(f"{REASON_CALL_ERROR}:{type(exc).__name__}")

    cleaned, why = clean_rewritten_detailed(raw)
    if not cleaned:
        return no(why)
    if cleaned == query.strip():
        return no(REASON_IDENTICAL)
    return {"query": cleaned, "applied": True, "reason": REASON_APPLIED}


def last_turn_text(history, role):
    """取历史里**最后一条**该角色消息的文本（纯函数）。

    刻意只取最后一条而不是全部：拼接的目的是补上"指代对象的先行词"，
    而先行词基本就在最近一轮；把整段历史拼进去会稀释当前问句
    （实测把助手那句也拼上，kw@k 从 71.4% 掉到 57.1%）。
    """
    for m in reversed(list(history or [])):
        if isinstance(m, dict) and m.get("role") == role:
            text = (m.get("content") or "").strip()
            if text:
                return text
    return ""


def build_concat_query(query, history, max_chars=CONCAT_MAX_CHARS):
    """把上一轮用户问句与当前问句拼成一个检索用问句（纯函数）。

    返回空串表示"这一档不适用"（没有上一轮用户消息），调用方应继续降级。
    """
    prev = last_turn_text(history, "user")
    if not prev or not (query or "").strip():
        return ""
    if prev.strip() == query.strip():
        # 同一句话又问一遍：拼起来是重复文本，只会给 BM25 平白加权重
        return ""
    # max_chars <= 0 = **不截断**（与本仓库"0 表示不限制"的惯例一致，见
    # RAG_RECALL_LIMIT / RAG_QUERY_REWRITE_MAX_TURNS）。旧写法 `prev[:0]` 会得到
    # 空串，拼出 `" 问句"` —— 它仍然 != query，于是 applied=True / source="concat"，
    # 推理链上"拼接档生效了"，实际一个字都没拼上。负数更糟：会砍掉末尾字符。
    head = prev if max_chars <= 0 else prev[:max_chars]
    return f"{head} {query.strip()}"


def build_retrieval_query(query, history, call=None, enabled=None,
                          concat_enabled=None, allow_concat=True):
    """决定这次检索到底用什么问句（纯函数外壳 + 可注入的 LLM 调用）。

    降级链（**每一档都记进 reason，绝无静默替换**）：
        LLM 改写 → 拼接上一轮用户问句 → 字面原句

    返回 {"query": 检索用问句, "applied": bool, "reason": str,
          "source": "llm" | "concat" | "literal"}
      * applied=True 仅当问句与入参 query 不同（concat 档也算，因为确实换了问句）；
      * source 是给界面/A-B 看的粗粒度标签，reason 是细粒度原因。
      另附 rewrite_prompt_stats 的四个键（prompt_chars / history_msgs /
      num_ctx / over_window）：历史不再按轮数截断后，界面与日志必须能看到
      "这次到底喂了多少历史、会不会超出窗口"，否则那就是一处静默。

    allow_concat=False 时跳过拼接档（调用方判定"本轮提问没有可检索内容"时用）。
    这个判据由 rag_engine 掌着（`_has_searchable_content`），**不在这里再抄一份** ——
    否则"？"该不该拼会变成两处各有一套答案。

    **本函数整体是"永不抛异常"的**：模块头承诺的"任何失败都退回原句"由这里的
    try 兜住，而不是靠调用方守规矩。此前 `rewrite_prompt_stats` 在 try 之外被
    无条件调用（连开关关掉时也调），一个非 dict 的历史元素就能让整条检索链抛
    AttributeError —— 契约由代码保证，不由注释保证。
    """
    try:
        return _build_retrieval_query_inner(
            query, history, call=call, enabled=enabled,
            concat_enabled=concat_enabled, allow_concat=allow_concat)
    except Exception as exc:
        logger.warning(f"⚠️  检索问句决策失败，退回字面原句（检索不受影响）: "
                       f"{type(exc).__name__}: {exc}")
        return {"query": query, "applied": False,
                "reason": f"{REASON_CALL_ERROR}:{type(exc).__name__}",
                "source": "literal"}


def _build_retrieval_query_inner(query, history, call=None, enabled=None,
                                 concat_enabled=None, allow_concat=True):
    stats = rewrite_prompt_stats(query, history)
    outcome = rewrite_query_detailed(query, history, call=call, enabled=enabled)
    if outcome["applied"]:
        return {**outcome, "source": "llm", **stats}

    if concat_enabled is None:
        concat_enabled = CONCAT_ENABLED
    # 只在"本该改写却没改成"的情况下拼 —— 这两个 reason 是例外，拼了只会更差：
    #   identical   模型判定当前问句已自足 → 拼接等于往一个完整问句里掺上轮噪音
    #   blank_query 没有可检索内容 → 让空查询守护照常返回空结果，
    #               而不是悄悄拿上一轮的话题去检索（那是另一个功能）
    # 原因码可能是 "call_error:TimeoutError" 这种带后缀的形式，按前缀判定
    reason_code = str(outcome["reason"]).split(":", 1)[0]
    if concat_enabled and allow_concat and reason_code in CONCAT_FROM_REASONS:
        concat = build_concat_query(query, history)
        if concat:
            return {"query": concat, "applied": True,
                    "reason": f"{REASON_CONCAT}<={outcome['reason']}",
                    "source": "concat", **stats}

    return {**outcome, "source": "literal", **stats}


def build_retrieval_routes(query, history, call=None, enabled=None,
                           concat_enabled=None, allow_concat=True,
                           fusion=False):
    """在降级链之上给出**召回路由计划**（纯函数外壳 + 可注入的 LLM 调用）。

    返回::

        {
          "primary": {query, applied, reason, source},   # 与 build_retrieval_query 逐字相同
          "queries": [{"query": str, "source": str}, …], # 实际要各召回一次的问句，primary 在首位
          "extra":   [str, …],                           # 比单路多出来的检索问句（供 steps 展示）
        }

    为什么要它（而不是直接改 build_retrieval_query）：单路那套是**已被三级降级
    链保证过**的行为，不该为了加融合去动它 —— `primary` 就是原函数的返回值，
    融合关闭时调用方只看 primary，行为与加这个函数之前完全一致。

    fusion=True 时**只**追加"拼接档"这一路，且只在 LLM 改写真的成功时追加。
    依据是多轮探针上的**独家命中**统计（见 rag_engine.QUERY_FUSION 注释）：
      * 字面原句一路没有任何独家命中 → 不追加（白付召回，还给候选池加噪声）；
      * LLM 失败时 primary 已经是拼接/字面档，没有第二档可加 → 不追加；
      * 两个不同的问句相同则去重，绝不把同一句召回两遍（那等于给某一路加倍权重）。

    ⚠️ `allow_concat` 要在追加时**再判一次**（不只是传给降级链）：调用方判定
    "本轮提问没有可检索内容"（如"？？？"）时会传 False，而 LLM 有时仍会为这种
    输入吐出一句像模像样的改写 —— 那时 primary 是 llm 档，若不判 allow_concat，
    融合就会额外拼上**上一轮的话题**去检索，正是 allow_concat 要挡住的语义
    （"悄悄拿上一轮的话题去检索"是另一个功能）。此 bug 已实测复现并修掉。

    ⚠️ `concat_enabled` 同样要在追加时**再判一次**：关闭的开关必须真的关闭。
    此前只判了 allow_concat，于是 `RAG_QUERY_FUSION=1` + `RAG_QUERY_REWRITE_CONCAT=0`
    时仍会追加拼接路，而 check_health 报的是"只剩 LLM → 字面" —— 配置报告与
    实际行为不一致。降级链里已经判过（见 _build_retrieval_query_inner），
    这里漏判就会让融合档绕过它。
    """
    if concat_enabled is None:
        concat_enabled = CONCAT_ENABLED

    primary = build_retrieval_query(
        query, history, call=call, enabled=enabled,
        concat_enabled=concat_enabled, allow_concat=allow_concat)

    queries = [{"query": primary["query"], "source": primary["source"]}]
    if fusion and allow_concat and concat_enabled and primary.get("source") == "llm":
        # primary 是 LLM 档时，拼接档通常也在链上（它是更低的一档）；
        # build_concat_query 返回空串只说明没有上一轮用户消息。
        alt = build_concat_query(query, history)
        if alt and alt != primary["query"]:
            queries.append({"query": alt, "source": "concat"})

    return {
        "primary": primary,
        "queries": queries,
        "extra": [q["query"] for q in queries[1:]],
    }


def rewrite_query(query, history, call=None, enabled=None):
    """按对话历史改写 query。任何失败都退回原 query（绝不抛异常）。

    call: 可注入的 LLM 调用（payload → 文本）。默认走 Ollama。
          注入点存在的意义：测试不需要 Ollama 在跑，且可以在不联网的情况下
          覆盖"返回垃圾""抛异常""超时"这些分支。
    enabled: 显式覆盖开关（None 则读环境变量 RAG_QUERY_REWRITE）。

    本函数只回问句（保持旧调用方兼容）；要看原因用 rewrite_query_detailed。
    """
    return rewrite_query_detailed(query, history, call=call, enabled=enabled)["query"]
