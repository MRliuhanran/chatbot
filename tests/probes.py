"""检索评测探针 —— **唯一来源**。

此前这份探针在 `compare_ab.py`（12 条，带期望书目+关键词）和
`verify_qdrant.py`（10 条，只有关键词）里各写了一份，且同一个"武松打虎"
一处写 `景阳冈`、一处写 `武松`。两份副本会各自漂移，导致两次评测的结果
不可比 —— 而"可比"正是 A/B 评测的全部意义。

本模块用 NamedTuple 定义，因此：
  * 属性访问：probe.query / probe.book / probe.keywords
  * 元组解包：for query, book, kws in PROBES   （兼容旧调用方）
两种写法都成立。

每个探针带两个独立标注，分别衡量不同能力：
  book     期望书目  → 排序质量（能否把正确的书排上来）
  keywords 期望关键词 → 召回质量（正文里到底有没有那段内容）
必须两个都看：书目命中高但关键词全错，说明只是"书名那一层"在起作用。

============================================================================
四类探针：为什么只有短查询不够
============================================================================
原有 12 条全是 3~7 字的**短关键词查询**（"武松打虎"）。实测这个集合漏掉了
三类真实缺陷，所以本模块把它们各自扩成一类探针：

  A. SHORT_PROBES（原 12 条，逐字保留）—— 短查询基线，用于与历史基线对拍。
  B. LONG_QUERY_PROBES —— 自然语言长问句。缺陷：**同一父块下的多个子块会
     同时挤进 top_k**，白占候选位（top_k=5 时 2 条同源就等于只看到 4 处内容）。
     本机实测 24 条正向里 5 条命中该缺陷：长问句 3/12（"曹操为什么要杀杨修"
     最严重，5 条里 3 条同源），短查询 2/12（"火烧赤壁""空城计"各 2 条同源）。
     ——注意：长问句更频繁、更严重，但**并非长问句独有**，所以 A/B 两类都要留。
  C. MULTITURN_PROBES —— 多轮 + 指代（"他最后结局如何"）。缺陷：把指代性
     问句直接丢给检索器会连书名都捞错 —— 实测该问句 top_k =
     西游记/红楼梦/红楼梦/红楼梦/水浒传，首位与期望书目不符、关键词全落空。
     单轮短查询根本测不到这条链路。
  D. NEGATIVE_PROBES —— 应拒答的负样本。缺陷：此前完全没有"不知道"的度量，
     检索器永远返回 top_k，无从判断它是否该闭嘴。

分类指标由 `aggregate_all` 汇总 —— 一个总数（例如 book@1=70%）会把
"短查询很好、多轮很差"这种结构性缺陷平均掉，必须分类看。

============================================================================
本机实测基线（books_v3，top_k=5，bge-base + bge-reranker-base）
============================================================================
  positive   n=24  book@1 = 95.8%  book@k = 100%  kw@k = 87.5%
  multiturn  n=5   book@1 = 60.0%  book@k = 100%  kw@k = 40.0%  topic@k = 60.0%
  negative   n=10  refuse = 0.0%   mean_rel_gap = 0.72   mean_n_books = 2.10
                   （正向的 mean_n_books = 1.08，对比见 evaluate_negative 的说明）

这张表就是"为什么要分类"的证据：合起来看 book@1 是 93%，看不出多轮只有 60%；
多轮的 kw@k 40% 与 topic@k 60% 更是完全被正向的 87.5% 盖住。
三条 kw@k 未命中的正向探针（宝玉不喜读书 / 贾府衰败 / 林冲逼上梁山）都是
**单点证据**型（关键词全书仅 1~2 处），是真实的召回缺口，不是探针写错 ——
它们 book@1/book@k 都是命中的。

============================================================================
接地（grounding）约定
============================================================================
正向/多轮探针的每个关键词都**逐条在 books/ 里数过命中次数**（写在各自注释里），
确保该探针真实可能命中；负样本则反过来，用 absent_terms / never_together
证明其主题在本语料里确实无据可依。校验逻辑见 tests/test_probes.py。

语料是"残本"这一点必须记住：《水浒传》只有 23 回（到王婆贪贿说风情为止，
所以正文里根本没有"潘金莲"这个名字，'金莲' 仅 1 次），《红楼梦》只有 64 回
（没有抄家）。选词时踩过这个坑，故 D 类里有专门的 corpus_gap 子类。
"""

from typing import NamedTuple

# ============================================================================
# A. 短关键词查询 —— 原有 12 条，逐字保留
# ============================================================================
# 拆成两个列表再拼（SHORT_PROBES + LONG_QUERY_PROBES = PROBES），是为了让
# "PROBES 的前 12 条 == 历史基线的 12 条"这件事在代码里看得见：
# append 之后顺序与内容都不变，老基线仍然可按位置对拍。
# 注意追加会让 L3 的零容差基线失效（条目数变了），这是预期行为，需重录。


class Probe(NamedTuple):
    query: str
    book: str
    keywords: tuple  # 任一命中即算命中（同一个概念的多种写法）


SHORT_PROBES = [
    Probe("武松打虎", "水浒传", ("武松", "景阳冈")),
    Probe("黛玉葬花", "红楼梦", ("葬花", "黛玉")),
    Probe("桃园结义", "三国演义", ("桃园", "结义")),
    Probe("火烧赤壁", "三国演义", ("赤壁",)),
    Probe("倒拔垂杨柳", "水浒传", ("垂杨", "鲁智深")),
    Probe("大闹天宫", "西游记", ("天宫", "大圣")),
    Probe("刘姥姥进大观园", "红楼梦", ("刘姥姥",)),
    Probe("空城计", "三国演义", ("空城",)),
    Probe("三打白骨精", "西游记", ("白骨", "悟空")),
    Probe("鲁智深拳打镇关西", "水浒传", ("镇关西", "鲁智深")),
    Probe("草船借箭", "三国演义", ("草船", "孔明", "诸葛亮")),
    Probe("宝玉挨打", "红楼梦", ("贾政", "宝玉")),
]


# ============================================================================
# B. 长问句正向探针
# ============================================================================
# 写法刻意统一成"实体+为什么/是什么/怎么样"的自然问句，因为要测的正是
# **长问句**这个形态本身：短查询的稀疏通道能整串精确命中，长问句被切成一堆
# term 后信噪比骤降、只能更多依赖语义通道，候选分布也更发散。
#
# 括号里是该关键词在**对应书籍**里的出现次数（errors="replace" 逐字统计）：
LONG_QUERY_PROBES = [
    # 红楼梦：《不喜读书》2、《禄蠹》1、《仕途经济》1 —— 三处都在贾政/湘云劝
    # 宝玉读书的同一段情节里，正是"宝玉为什么不爱读书"的标准答案所在。
    Probe("为什么宝玉不喜欢读书", "红楼梦", ("不喜读书", "禄蠹", "仕途经济")),
    # 红楼梦：《小性》5、《猜忌》1、《多心》24 —— "素习猜忌，好弄小性儿"是
    # 书里对黛玉性格最直接的定评（同句同时含"猜忌"与"小性"）。
    Probe("林黛玉的性格有什么特点", "红楼梦", ("小性", "猜忌", "多心")),
    # 红楼梦：《树倒猢狲散》1、《月满则亏》1 —— 都出自可卿托梦凤姐那段，
    # 全书唯一一次正面预言贾府结局；命中次数极少，是典型"单点证据"探针。
    Probe("贾府最终为什么会衰败", "红楼梦", ("树倒猢狲散", "月满则亏")),
    # 水浒传：《高俅》62、《野猪林》4、《沧州》38 —— 林冲被陷害→刺配→
    # 野猪林获救→上梁山，是前半部的主线，三个词分属三个环节。
    Probe("林冲为什么会被逼上梁山", "水浒传", ("高俅", "野猪林", "沧州")),
    # 水浒传：《生辰纲》26、《智取》4 —— "智取"含回目（第十五回　杨志押送
    # 金银担　吴用智取生辰纲）；注意本语料只到 23 回，但这一段完整在内。
    Probe("吴用是怎么智取生辰纲的", "水浒传", ("生辰纲", "智取")),
    # 水浒传：《五台山》20、《剃度》5、《智真长老》6 —— 鲁智深"大闹五台山"
    # 本身只出现 1 次（回目），故关键词取三个更稳的同情节词。
    Probe("鲁智深为什么大闹五台山", "水浒传", ("五台山", "剃度", "智真长老")),
    # 三国演义：《孟获》157、《南蛮》13 —— 七擒孟获整段完整。
    Probe("诸葛亮为什么要七擒孟获", "三国演义", ("孟获", "南蛮")),
    # 三国演义：《三顾》10、《茅庐》12 —— 两词在整个语料里只出现在《三国演义》。
    Probe("刘备为什么要三顾茅庐请诸葛亮", "三国演义", ("三顾", "茅庐")),
    # 三国演义：《杨修》15、《鸡肋》7 —— 杀杨修的导火索就是"鸡肋"口令。
    Probe("曹操为什么要杀杨修", "三国演义", ("杨修", "鸡肋")),
    # 三国演义：《华容》12 —— 华容道放曹操，关羽义气那一段。
    Probe("关羽为什么在华容道放走曹操", "三国演义", ("华容",)),
    # 西游记：《五行山》11 —— 只出现在西游记。
    Probe("孙悟空为什么被压在五行山下", "西游记", ("五行山",)),
    # 西游记：《紧箍》34 —— 三个箍儿是如来交给观音、再由观音转授唐僧的，
    # "谁教的"正好落在这一段的语义上。
    Probe("紧箍咒是谁教给唐僧的", "西游记", ("紧箍",)),
]


# ============================================================================
# 唯一入口：PROBES = 短查询 + 长问句
# ============================================================================
# 旧调用方（eval_runner / test_retrieval_quality / compare_ab / verify_qdrant）
# 都直接遍历 PROBES，所以必须"追加"而不是新建一个集合：那样才只需要重录一次
# 基线，且长问句天然进入所有既有评测通道。
PROBES = SHORT_PROBES + LONG_QUERY_PROBES


def keyword_hit(probe, text):
    """关键词命中判定：任一关键字出现在 text 中即为命中。"""
    return any(k in text for k in probe.keywords)


def evaluate_probe(probe, results):
    """对单条探针的检索结果做判定，返回 {book@1, book@k, kw@k}。

    显式传入 probe（而不是读模块级全局），保证函数纯、可单测。

    results: [{"book":..., "child_text":..., "parent_text":...}, ...] 按名次排列
    """
    if not results:
        return {"book@1": False, "book@k": False, "kw@k": False}
    books = [r.get("book", "") for r in results]
    body = " ".join(
        (r.get("child_text", "") or "") + (r.get("parent_text", "") or "")
        for r in results
    )
    return {
        "book@1": books[0] == probe.book,
        "book@k": probe.book in books,
        "kw@k": keyword_hit(probe, body),
    }


# ============================================================================
# C. 多轮 + 指代探针
# ============================================================================
# 多轮探针的**指代类型**。分类不是为了好看：不同失效原因需要不同修法，
# 而合在一起只看 book@1 会把"代词没消解""实体切错了""省略没补全"平均成一个数。
#
#   pronoun        上轮用户句里就有实体，当前只说"他/她" —— 最基础的一类
#   assistant_only 实体**只出现在助手那一轮**。只把用户历史拼进改写提示词的
#                  实现会在这类上立刻露馅（那正是最初的实现缺陷）
#   entity_switch  历史里先后出现过**两个不同实体**，当前问句指向更靠后的那个。
#                  把整段历史无差别拼接的实现会把两个实体搅在一起
#   ellipsis       省略主语/宾语（"后来是怎么处置的"），没有代词可抓
#   temporal       时间指代（"后来呢""再之后"）—— 代词表覆盖不到，只能靠语义
#   recall_detail  指代对象**不是人名，而是上一轮回答里的一个内容词**
#                  （"你刚说的那个官名"）。它考的是"上一轮到底说了什么"，
#                  而答案原文并不会留在下一轮上下文里（见 rag_engine 的说明）
VALID_MULTITURN_KINDS = (
    "pronoun", "assistant_only", "entity_switch",
    "ellipsis", "temporal", "recall_detail",
)


class MultiturnProbe(NamedTuple):
    """多轮对话里的一句指代性问句。

    history       list[dict]，形如 [{"role":"user","content":...},
                  {"role":"assistant","content":...}]，即 OpenAI 消息格式。
                  指代对象**可能只在 assistant 的一轮里出现过**（P3 就是），
                  故意这么设计：只把用户历史拼进查询的实现会立刻露馅。
    query         当前问句，本身**不含实体名**（"他最后结局如何"）——
                  单独看它无法检索，必须先结合 history 消解指代。
    book          期望书目（指代对象所属的书）。
    keywords      期望关键词（消解正确后应当被召回的证据）。
    resolved      正确消解后的查询。仅用于诊断/打印：把"检索错了"与
                  "查询改写错了"这两件事分开 —— 没有它就只能看到 book@1=False，
                  看不出到底是改写没做还是检索没跟上。
    topic_terms   指代所指的实体词，用于 topic@k（见 evaluate_multiturn）。
    kind          指代类型，取值见 VALID_MULTITURN_KINDS。默认空串是为了
                  向后兼容（NamedTuple 加字段会改变元组长度，旧调用方若按
                  位置解包会立刻破 —— 本模块的既有约定是只往后加默认值）。
    """

    history: list
    query: str
    book: str
    keywords: tuple
    resolved: str
    topic_terms: tuple
    kind: str = ""


MULTITURN_PROBES = [
    MultiturnProbe(
        history=[
            {"role": "user", "content": "孙悟空大闹天宫之后怎么样了？"},
            {"role": "assistant", "content": "他被如来佛祖压在五行山下，一压就是五百年。"},
        ],
        query="他最后结局如何",
        book="西游记",
        # 西游记：《成佛》12、《真经》50、《灵山》70
        keywords=("成佛", "真经", "灵山"),
        resolved="孙悟空最后结局如何",
        topic_terms=("悟空", "大圣", "行者"),
        kind="pronoun",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "林黛玉小时候家里是什么情况？"},
            {"role": "assistant", "content": "她母亲贾敏早逝，父亲林如海把她送进贾府寄养。"},
        ],
        query="她为什么总是爱哭",
        book="红楼梦",
        # 红楼梦：《还泪》1（绛珠仙草以泪还债，正是"爱哭"的书中解释）、
        # 《眼泪》18。第一个词只有 1 处，是本集合里最"单点"的证据。
        keywords=("还泪", "眼泪"),
        resolved="林黛玉为什么总是爱哭",
        topic_terms=("黛玉",),
        kind="pronoun",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "六出祁山讲的是谁？"},
            {"role": "assistant", "content": "讲的是诸葛亮北伐曹魏，屡次从祁山出兵。"},
        ],
        query="他最后死在哪里",
        book="三国演义",
        # 三国演义：《五丈原》11 —— "星落秋风五丈原"即其殁地。
        keywords=("五丈原",),
        resolved="诸葛亮最后死在哪里",
        topic_terms=("孔明", "诸葛亮"),
        # 用户句里只有"谁"，实体只在助手那一轮 —— 即 assistant_only。
        kind="assistant_only",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "林冲是怎么被高太尉陷害的？"},
            {"role": "assistant", "content": "高俅设计让他带刀误入白虎堂，借此问罪。"},
        ],
        query="他后来为什么被发配到沧州",
        book="水浒传",
        # 水浒传：《高俅》62、《白虎堂》1、《沧州》38
        keywords=("高俅", "白虎堂", "沧州"),
        resolved="林冲后来为什么被发配到沧州",
        topic_terms=("林冲",),
        kind="pronoun",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "武松在景阳冈上做了什么？"},
            {"role": "assistant", "content": "他醉后上冈，赤手空拳打死了那只吊睛白额大虫。"},
        ],
        query="他因此做了什么官",
        book="水浒传",
        # 水浒传：《都头》63、《阳谷县》10 —— 打虎后被阳谷县知县参做都头。
        keywords=("都头", "阳谷县"),
        resolved="武松因此做了什么官",
        topic_terms=("武松",),
        kind="pronoun",
    ),
    # ------------------------------------------------------------------
    # 以下 16 条按 kind 补齐覆盖。原有 5 条**逐字保留在前面**：历史基线
    # 与逐条对拍依赖顺序，追加只能往后加（同 PROBES 的约定）。
    # 每条的关键词都在 tests/test_probes.py 里做语料接地校验。
    # ------------------------------------------------------------------

    # ---- assistant_only：实体只出现在助手那一轮 ------------------------
    MultiturnProbe(
        history=[
            {"role": "user", "content": "那个被压在山下五百年的是谁？"},
            {"role": "assistant", "content": "是孙悟空，他大闹天宫后被如来佛祖压在五行山下。"},
        ],
        query="他头上的箍儿是谁给他戴上的",
        book="西游记",
        # 西游记：《紧箍》34、《金箍》161
        keywords=("紧箍", "金箍"),
        resolved="孙悟空头上的紧箍儿是谁给他戴上的",
        topic_terms=("悟空", "行者", "大圣"),
        kind="assistant_only",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "那个用眼泪还债的仙子是谁？"},
            {"role": "assistant", "content": "是林黛玉，她前世是绛珠仙草，下凡以泪还债。"},
        ],
        query="她是在谁家长大的",
        book="红楼梦",
        # 红楼梦：《贾母》841、《外祖母》8 —— 贾母即其外祖母
        keywords=("贾母", "外祖母"),
        resolved="林黛玉是在谁家长大的",
        topic_terms=("黛玉",),
        kind="assistant_only",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "七擒孟获讲的是谁？"},
            {"role": "assistant", "content": "是诸葛亮南征时七擒七纵孟获的故事。"},
        ],
        query="他给后主写的那篇表叫什么",
        book="三国演义",
        # 三国演义：《出师表》3
        keywords=("出师表",),
        resolved="诸葛亮给后主写的那篇表叫什么",
        topic_terms=("孔明", "诸葛亮"),
        kind="assistant_only",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "那个在景阳冈上打死老虎的是谁？"},
            {"role": "assistant", "content": "是武松，他醉后上冈打死了吊睛白额大虫。"},
        ],
        query="他哥哥的绰号是什么",
        book="水浒传",
        # 水浒传：《武大》76、《三寸丁》3、《谷树皮》3
        keywords=("武大", "三寸丁", "谷树皮"),
        resolved="武松的哥哥的绰号是什么",
        topic_terms=("武松",),
        kind="assistant_only",
    ),

    # ---- entity_switch：历史里前后两个实体，当前指向更靠后的那个 --------
    MultiturnProbe(
        history=[
            {"role": "user", "content": "孙悟空大闹天宫之后被压在哪里？"},
            {"role": "assistant", "content": "他被如来压在五行山下。"},
            {"role": "user", "content": "那红孩儿最后被谁收服了？"},
            {"role": "assistant", "content": "被观音菩萨用金箍儿收作善财童子。"},
        ],
        query="他的父母是谁",
        book="西游记",
        # 西游记：《牛魔王》39、《铁扇》15。若改写被上一轮的悟空粘住，
        # 关键词会全落空而书仍然对 —— 这正是 topic@k 抓的形态。
        keywords=("牛魔王", "铁扇"),
        resolved="红孩儿的父母是谁",
        topic_terms=("红孩儿", "圣婴"),
        kind="entity_switch",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "林黛玉为什么总是爱哭？"},
            {"role": "assistant", "content": "因为她前世是绛珠仙草，要以泪还债。"},
            {"role": "user", "content": "那薛宝钗身上有什么特别的？"},
            {"role": "assistant", "content": "她戴着一个金锁，据说是癞头和尚给的。"},
        ],
        query="她哥哥叫什么名字",
        book="红楼梦",
        # 红楼梦：《薛蟠》164、《呆霸王》2
        keywords=("薛蟠", "呆霸王"),
        resolved="薛宝钗的哥哥叫什么名字",
        topic_terms=("宝钗",),
        kind="entity_switch",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "关羽最后是怎么死的？"},
            {"role": "assistant", "content": "他败走麦城，被孙权所擒杀。"},
            {"role": "user", "content": "那张飞呢，他后来怎么了？"},
            {"role": "assistant", "content": "他被自己帐下的部将所害。"},
        ],
        query="他最后是被谁杀的",
        book="三国演义",
        # 三国演义：《范疆》8、《张达》8（原著作"范疆"，非"范强"）
        keywords=("范疆", "张达"),
        resolved="张飞最后是被谁杀的",
        topic_terms=("张飞",),
        kind="entity_switch",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "林冲是怎么被逼上梁山的？"},
            {"role": "assistant", "content": "高俅设计让他误入白虎堂，又火烧草料场，最后被逼上梁山。"},
            {"role": "user", "content": "那鲁智深呢，他为什么出家？"},
            {"role": "assistant", "content": "他打死人后为避祸，出家做了和尚。"},
        ],
        query="他打死的那个卖肉的叫什么",
        book="水浒传",
        # 水浒传：《镇关西》7、《郑屠》34
        keywords=("镇关西", "郑屠"),
        resolved="鲁智深打死的那个卖肉的叫什么",
        topic_terms=("鲁智深",),
        kind="entity_switch",
    ),

    # ---- ellipsis：连代词都没有，只有省略 ------------------------------
    MultiturnProbe(
        history=[
            {"role": "user", "content": "孙悟空在五庄观偷吃了什么？"},
            {"role": "assistant", "content": "他偷吃了人参果，还推倒了果树。"},
        ],
        query="后来是怎么把树救活的",
        book="西游记",
        # 西游记：《甘露》15、《净瓶》33 —— 观音以净瓶甘露救活果树
        keywords=("甘露", "净瓶"),
        resolved="五庄观的人参果树后来是怎么救活的",
        # 省略型没有代词，指代对象是"那棵树"：topic 只能取历史里出现过的实体词
        topic_terms=("人参果", "五庄观"),
        kind="ellipsis",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "刘姥姥进大观园时闹了什么笑话？"},
            {"role": "assistant", "content": "她在宴席上说'老刘老刘食量大如牛'，被众人取笑。"},
        ],
        query="为什么后来给她那么多银子",
        book="红楼梦",
        # 红楼梦：《接济》2、《二十两》19
        keywords=("接济", "二十两"),
        resolved="为什么后来给刘姥姥那么多银子",
        topic_terms=("刘姥姥",),
        kind="ellipsis",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "诸葛亮北伐时在街亭用了谁？"},
            {"role": "assistant", "content": "他用了马谡，结果街亭失守。"},
        ],
        query="后来是怎么处置的",
        book="三国演义",
        # 三国演义：《马谡》48、《街亭》44 —— 失街亭后挥泪斩马谡
        keywords=("马谡", "街亭"),
        resolved="诸葛亮后来是怎么处置马谡的",
        topic_terms=("孔明", "诸葛亮"),
        kind="ellipsis",
    ),

    # ---- temporal：时间指代，代词表覆盖不到 ----------------------------
    MultiturnProbe(
        history=[
            {"role": "user", "content": "唐僧师徒是怎么过火焰山的？"},
            {"role": "assistant", "content": "孙悟空向铁扇公主借了芭蕉扇，扇灭了火焰。"},
        ],
        query="后来那把扇子还回去了吗",
        book="西游记",
        # 西游记：《芭蕉》62、《铁扇》15
        keywords=("芭蕉", "铁扇"),
        resolved="芭蕉扇后来还回去了吗",
        topic_terms=("铁扇", "芭蕉"),
        kind="temporal",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "赤壁之战是怎么打的？"},
            {"role": "assistant", "content": "周瑜与诸葛亮用火攻，把曹操的战船烧了个干净。"},
        ],
        query="后来他是从哪里逃走的",
        book="三国演义",
        # 三国演义：《华容》12、《华容道》7
        keywords=("华容", "华容道"),
        resolved="曹操后来是从哪里逃走的",
        topic_terms=("曹操",),
        kind="temporal",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "元妃省亲是怎么一回事？"},
            {"role": "assistant", "content": "贾元春被封为妃后回家探亲，贾府为此修了大观园。"},
        ],
        query="后来那园子给谁住了",
        book="红楼梦",
        # 红楼梦：《大观园》28 —— 省亲后宝玉与众姐妹奉元妃命住进去
        keywords=("大观园",),
        resolved="大观园后来给谁住了",
        topic_terms=("大观园",),
        kind="temporal",
    ),

    # ---- recall_detail：指代对象不是人名，而是上轮回答里的内容词 -------
    MultiturnProbe(
        history=[
            {"role": "user", "content": "武松打完虎之后做了什么？"},
            {"role": "assistant", "content": "他被阳谷县知县参做了都头。"},
        ],
        query="你刚说的那个官名，是管什么的",
        book="水浒传",
        # 水浒传：《都头》63、都头是县里的捕役头目
        keywords=("都头", "阳谷县"),
        resolved="武松当的都头是管什么的",
        topic_terms=("武松",),
        kind="recall_detail",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "孙悟空从东海龙宫拿走了什么？"},
            {"role": "assistant", "content": "他拿走了那根如意金箍棒。"},
        ],
        query="你刚说的那根棒子上刻着什么字",
        book="西游记",
        # 西游记：《如意金箍棒》9、《一万三千五百斤》4
        keywords=("如意金箍棒", "一万三千五百斤"),
        resolved="如意金箍棒上刻着什么字",
        topic_terms=("金箍", "如意"),
        kind="recall_detail",
    ),

    # ------------------------------------------------------------------
    # 2026-09-19 补充：每类补齐到 5 条。原 21 条里 recall_detail 只有 2 条、
    # ellipsis / temporal 各 3 条 —— 那种规模下"这一类比那类差"的结论完全
    # 由单条探针左右。背景见 MULTITURN_PLAN.md §0.1：
    # 实测 `concat` 与 `LLM 改写` 两路的失手点**互补**，而看出互补靠的是
    # "逐条独家命中"，n 太小会被聚合比率抹平。
    # 所有 keywords / topic_terms 仍由 tests/test_probes.py 做语料接地校验。
    # ------------------------------------------------------------------
    MultiturnProbe(
        history=[
            {"role": "user", "content": "猪八戒原本是什么身份？"},
            {"role": "assistant", "content": "他本是天蓬元帅，因醉酒调戏嫦娥被贬下凡，又错投了猪胎。"},
        ],
        query="他取经之后被封为什么",
        book="西游记",
        # 西游记：《净坛使者》4 —— 取经功成后如来封他的正果
        keywords=("净坛使者",),
        resolved="猪八戒取经之后被封为什么",
        topic_terms=("八戒",),
        kind="pronoun",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "林黛玉刚进贾府时是什么情形？"},
            {"role": "assistant", "content": "她母亲早逝，父亲林如海把她送进贾府寄养，从此住在荣国府。"},
            {"role": "user", "content": "那薛宝钗呢，她家是做什么的？"},
            {"role": "assistant", "content": "薛家是皇商，宝钗随母亲和哥哥一起进京，借住在贾府。"},
        ],
        query="她哥哥是个什么样的人",
        book="红楼梦",
        # 红楼梦：《薛蟠》164 —— 宝钗之兄，人称"呆霸王"
        keywords=("薛蟠",),
        resolved="薛宝钗的哥哥是个什么样的人",
        topic_terms=("宝钗",),
        # 历史里先出现黛玉、后出现宝钗，当前问句只带"她"。
        # 把整段历史无差别拼接的实现会把两个实体搅在一起。
        kind="entity_switch",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "曹操败走华容道时被谁拦住了？"},
            {"role": "assistant", "content": "被关羽拦住，但关羽念及旧日恩情，最终放他走了。"},
        ],
        query="后来这件事是怎么了结的",
        book="三国演义",
        # 三国演义：《军令状》11 —— 关羽临行前立下军令状，回营后险些被斩
        keywords=("军令状",),
        resolved="关羽华容道放走曹操这件事后来是怎么了结的",
        # ⚠️ topic_terms 必须用**书里真会这么写**的称呼。这条最初只写了"关羽"，
        # 而《三国演义》里"关羽"仅出现 9 次、"云长"443 次、"关公"519 次 ——
        # 于是它在三种配置下全部 topic@k=N，看起来像"检索系统性失手"，
        # 实际是**度量假象**（捞对了章节，但正文写的是别名）。
        # 这类错误的通用护栏见 tests/test_probes.py::TestTopicTermsAreReachable。
        topic_terms=("关羽", "云长", "关公"),
        kind="ellipsis",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "唐僧师徒在通天河遇到了什么妖怪？"},
            {"role": "assistant", "content": "河里有个灵感大王，每年要吃一对童男童女，最后被观音菩萨收走了。"},
        ],
        query="后来是怎么过去的",
        book="西游记",
        # 西游记：《老鼋》23 —— 灵感大王既去，老鼋驮师徒过河
        keywords=("老鼋",),
        resolved="唐僧师徒后来是怎么过通天河的",
        topic_terms=("唐僧",),
        kind="ellipsis",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "杨志是怎么丢掉生辰纲的？"},
            {"role": "assistant", "content": "他在黄泥冈被晁盖等人用蒙汗药麻翻，生辰纲被劫走，醒来后不敢回去复命。"},
        ],
        query="后来他去哪里落草了",
        book="水浒传",
        # 水浒传：《二龙山》7、《宝珠寺》4 —— 杨志后来与鲁智深同上二龙山
        keywords=("二龙山",),
        resolved="杨志后来去哪里落草了",
        topic_terms=("杨志",),
        kind="temporal",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "王熙凤是怎么弄权铁槛寺的？"},
            {"role": "assistant", "content": "她收了三千两银子，假托贾琏之名，逼得一对未婚夫妻双双自尽。"},
        ],
        query="后来她还做过哪些贪财的事",
        book="红楼梦",
        # 红楼梦：《铁槛寺》17 —— 弄权一节即在此处
        keywords=("铁槛寺",),
        resolved="王熙凤后来还做过哪些贪财的事",
        # 取"熙凤"而不是"凤姐"：历史里写的是"王熙凤"，而 topic_terms
        # 必须真的在历史里出现（接地校验会查这一条）。
        topic_terms=("熙凤",),
        kind="temporal",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "武松的哥哥是谁？"},
            {"role": "assistant", "content": "他哥哥叫武大郎，是个挑担子卖炊饼的。"},
        ],
        query="你刚说的那种吃食是怎么做的",
        book="水浒传",
        # 水浒传：《炊饼》10 —— 武大郎所卖
        keywords=("炊饼",),
        resolved="武大郎卖的炊饼是怎么做的",
        # 指代对象是上轮回答里的**内容词**（炊饼），不是人名。
        topic_terms=("炊饼",),
        kind="recall_detail",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "唐僧在哪里收了猪八戒？"},
            {"role": "assistant", "content": "在高老庄，收他做了二徒弟，法名悟能。"},
        ],
        query="你刚说的那个地方还住了谁",
        book="西游记",
        # 西游记：《高老庄》12、《高太公》4 —— 庄主高太公一家
        keywords=("高太公",),
        resolved="高老庄里还住了谁",
        topic_terms=("高老庄",),
        kind="recall_detail",
    ),
    MultiturnProbe(
        history=[
            {"role": "user", "content": "赤壁之战中谁献了连环计？"},
            {"role": "assistant", "content": "庞统献了连环计，让曹操把战船用铁环首尾相连。"},
        ],
        query="你刚说的那种计策是谁想出来的",
        book="三国演义",
        # 三国演义：《庞统》83、《连环计》6
        keywords=("庞统",),
        resolved="赤壁之战里的连环计是谁想出来的",
        topic_terms=("连环计",),
        kind="recall_detail",
    ),
]


def evaluate_multiturn(probe, results, normalize=None):
    """多轮探针判定：在单轮三指标之外，追加一个 topic@k。

    前三个指标直接复用 evaluate_probe（MultiturnProbe 同样有
    query/book/keywords 三个属性），所以两类探针的指标口径完全一致，
    可以直接横向比 —— 这正是"多轮比单轮差多少"要回答的问题。

    topic@k：指代对象（topic_terms）是否出现在召回的正文里。
    为什么需要它：book@k 只说"捞对了书"，而指代失效的典型表现恰恰是
    **书对了、内容却与指代对象无关**（比如"他最后结局如何"在西游记里
    召回了不相干的段落）。book@k 会把这种情况判成通过，topic@k 不会。

    normalize: 可注入的"词面归一"函数（`rag_engine.normalize_aliases`）。
    **为什么必须归一**：同一个实体在一本书里有多种写法，只看字面会让判据
    **假失败** —— 实测踩过：新探针"后来这件事是怎么了结的"只写了
    topic_terms=("关羽",)，而《三国演义》里"关羽"9 次 / "云长"443 次 /
    "关公"519 次，于是它在三种配置下全部 topic@k=N，看起来像"检索系统性
    失手"，实际是**度量假象**（章节捞对了，正文写的是别名）。
    检索侧的稀疏通道早就在做同样的事（jieba → 别名归一 → BM25），
    度量侧不做就是两套口径。

    这里**不 import rag_engine**：本模块刻意保持零依赖（只有 typing），
    以便 L0 纯函数层直接用它。归一化由调用方（`tests/eval_runner.py`）注入。
    """
    verdict = evaluate_probe(probe, results)
    body = " ".join(
        (r.get("child_text", "") or "") + (r.get("parent_text", "") or "")
        for r in results
    )
    terms = probe.topic_terms
    if normalize is not None:
        body = normalize(body)
        terms = tuple(normalize(t) for t in terms)
    verdict["topic@k"] = any(t in body for t in terms)
    return verdict


# ============================================================================
# D. 负样本（应拒答）探针
# ============================================================================
class NegativeProbe(NamedTuple):
    """一条**正确答案是"我不知道"**的探针。

    query          问题本身。
    reason         为什么应当拒答 —— 必须具体到"哪本书缺什么"，便于复盘。
    kind           三类，判定口径相同但修法不同：
                     out_of_domain  域外：问题根本不属于这四本书的领域
                     corpus_gap     语料缺失：名著里有，但本语料这一版没有
                                    （水浒只有 23 回、红楼只有 64 回）
                     not_in_canon   原著也不存在：跨书拉郎配 / 前提为假
    absent_terms   必须"0 次出现"的关键词，用来证明无据可依。
                    默认在**整个语料**里查；kind=corpus_gap 时只在 gap_book
                    那一本里查（该词别的书可能有，正是要抓的误导源）。
    gap_book       kind=corpus_gap 时，缺内容属于哪本书；否则为 ""。
    never_together 二元组 (a, b)：二者**从未出现在同一本书**里 ——
                    跨书问题用它证明"这个问题在单书语料里不可能有答案"。
    """

    query: str
    reason: str
    kind: str
    absent_terms: tuple = ()
    gap_book: str = ""
    never_together: tuple = ()


NEGATIVE_PROBES = [
    # ---- 域外：问题本身不属于四大名著的领域 ----
    NegativeProbe(
        "今天北京的天气怎么样",
        "实时天气属于域外问题，四本书里没有任何可依据的内容",
        "out_of_domain",
        # 《西游记》里有"气温"2 次，所以不能用它当证据词。
        ("温度", "天气预报", "摄氏度"),
    ),
    NegativeProbe(
        "怎么做红烧肉",
        "烹饪菜谱属于域外问题，与古典小说语料无关",
        "out_of_domain",
        ("红烧肉", "五花肉", "食材", "菜谱"),
    ),
    NegativeProbe(
        "帮我写一段Python排序代码",
        "编程任务属于域外问题，语料中不含任何代码或算法内容",
        "out_of_domain",
        ("Python", "代码", "编程", "排序算法"),
    ),
    NegativeProbe(
        "最近流感疫苗在哪里可以打",
        "现代医疗与公共卫生服务属于域外问题",
        "out_of_domain",
        ("疫苗", "流感", "接种"),
    ),
    # ---- 语料缺失：名著里有，本语料这一版没有 ----
    NegativeProbe(
        "宋江最后接受招安了吗",
        "本语料《水浒传》只到第二十三回（王婆贪贿说风情），招安情节在全书后半部，"
        "《水浒传》正文里'招安'0 次；而'招安'在三国/西游里有 22/11 次，"
        "检索器极可能拿别书的同名词来充数 —— 这正是必须拒答的场景",
        "corpus_gap",
        ("招安",),
        gap_book="水浒传",
    ),
    NegativeProbe(
        "潘金莲最后是什么结局",
        "《水浒传》正文里从未出现'潘金莲'三字（仅第二十三回'小名唤做金莲'出现"
        "'金莲'1 次），她的结局不在本语料范围内",
        "corpus_gap",
        ("潘金莲",),
        gap_book="水浒传",
    ),
    NegativeProbe(
        "贾府最后被抄家了吗",
        "'抄家'在整个语料 0 次；本语料《红楼梦》只到第六十四回，"
        "抄家发生在后四十回，书中只有'树倒猢狲散'式的预言",
        "corpus_gap",
        ("抄家",),
        gap_book="红楼梦",
    ),
    # ---- 原著也不存在：跨书问题 / 假前提 ----
    NegativeProbe(
        "武松和林黛玉是什么关系",
        "二人分属《水浒传》《红楼梦》，原著中不存在任何关系"
        "（'武松'只在《水浒传》出现 247 次，'林黛玉'只在《红楼梦》出现 231 次）",
        "not_in_canon",
        never_together=("武松", "林黛玉"),
    ),
    NegativeProbe(
        "孙悟空和关羽谁更厉害",
        "二人分属《西游记》《三国演义》，原著中没有任何交手或交集，"
        "任何比较都是编造",
        "not_in_canon",
        never_together=("孙悟空", "关羽"),
    ),
    NegativeProbe(
        "诸葛亮和孙悟空谁的法术更高强",
        "'诸葛亮'在《三国演义》156 次、《水浒传》1 次，'孙悟空'只在《西游记》"
        "126 次，两书之间无交集；且'法术'的强弱在原著中并无标准",
        "not_in_canon",
        never_together=("诸葛亮", "孙悟空"),
    ),
]


# ----------------------------------------------------------------------------
# 拒答判定：为什么不能用绝对分数阈值，以及相对口径怎么设计
# ----------------------------------------------------------------------------
# 实测事实（本机 bge-reranker-base，books_v3 集合，top_k=5）：
#   "三打白骨精"     首位 rerank 分数 = -0.11，却是**正确答案**
#   "刘姥姥进大观园" 首位 rerank 分数 =  5.34，同样是正确答案
#   → 同一个模型、同一次评测，正确回答的首位分数可以相差 5.4 分且跨零。
#     bge-reranker-base 输出的是**未过 sigmoid 的 logit**，它的零点由训练时的
#     正负样本先验决定，没有"大于 0 就算相关"的语义，**跨查询不可比**。
#     所以任何形如 `if top1_score < 2.0: 拒答` 的绝对阈值都是错的：
#     它会把 -0.11 那条正确的"三打白骨精"判成拒答，同时放行得分 2.88 的
#     域外问题（实测"孙悟空的师父是不是诸葛亮的老师"首位 2.88）——
#     分数只表示"文本与查询在字面/风格上有多像"，不表示"语料里真有答案"。
#
# 可用的信号只有一个：**同一次查询内部**的分数形态。同一批候选出自同一次
# 前向计算，量纲相同，互相可比。于是定义：
#
#     相对首位优势 rel_gap = (s1 - mean(s2..sk)) / (s1 - min(s1..sk))
#
#   分子：首位比"其余候选的平均水平"高出多少 —— 有突出候选才为正。
#   分母：首位到全体的落差，**用同一个查询自己的尺度去归一化**，
#         因此结果是无量纲比值，不受跨查询尺度漂移影响。
#   取值范围 [0,1]：s1 是最大值（结果已按名次排序），故 0 <= s1-mean(其余)
#         <= s1-min <= 分母。
#   全体分数重合（分母≈0）时定义为 0.0 —— "没有任何候选突出"正是
#   应当拒答的形态。
#
# 判定：rel_gap < REFUSE_REL_GAP  →  判定为"分数层面无可靠依据"，应当拒答。
# 阈值是比例而不是分数，所以换集合/换语料时只需重校准这一个数，
# 不会像绝对阈值那样随模型或语料整体漂移而全面失效。
#
# ---------------------------------------------------------------------------
# 【重要】本机实测的校准结果：rel_gap 在本栈上**没有区分度**，别被它骗了
# ---------------------------------------------------------------------------
# 拿 24 条正向探针（12 短 + 12 长）与 10 条负样本实跑，rel_gap 的分布是：
#     正向：min 0.43  max 0.97  mean 0.72
#     负向：min 0.60  max 0.95  mean 0.72      ← 与正向完全重合（均值一模一样）
# 也就是说：域外问题同样会产出一个"看起来挺突出"的首位候选。本函数给出的
# refuse=False **不代表系统有把握**，只代表"分数形态没给出反对证据"。
# 因此在本栈上 refuse_rate 会接近 0 —— 这是对**引擎现状**的真实描述：
# 引擎**有**拒答信号了（rag_engine.confidence_signal 的 refuse，实测校准：误拒 0/24、拒答召回 6/10），但它是**分层**的 —— 检索侧只给信号，最终由生成侧结合上下文裁定，且不硬拦用户。因此"检索仍返回了 top_k 条"并不等于拒答失败。
# 早先这里写的是"压根没有拒答通路"，那是加入 confidence_signal 之前的实情，
# 不是本模块判定错了。
#
# 阈值取 0.30（低于实测正向最低值 0.43）：宁可漏拒、不可错拒 —— 把一个
# 正确的"三打白骨精"（rel_gap 0.60）误判成拒答，比漏掉一条域外问题更糟，
# 因为前者会直接损害正常问答体验。
#
# 唯一有区分度的**相对**信号是"书目分散度"（见 evaluate_negative 的 n_books）：
# 正向 24 条里 22 条的 top_k 全落在同一本书（均值 1.08 本），负样本 10 条里
# 9 条跨了 2~4 本书（均值 2.10 本）。它是"领域归属"信号而非"相关度"信号，
# 与分数的绝对值无关，但也没有强到可以直接当判定用（"贾府最后被抄家了吗"
# 就只命中《红楼梦》一本书），故只作诊断字段暴露，不并入 refuse。
#
# 结论（给后续维护者）：真正的拒答能力需要 rerank 分数之外的证据 ——
# 例如用"同一查询在下界对照语料上的分数分布"做归一化后再定阈值，或引入
# 生成阶段的"无法回答"判定。本模块只负责**度量**并把阈值做成一个旋钮。
REFUSE_REL_GAP = 0.30

# 只用于避免除零的数值护栏，不是判定阈值。
_EPS = 1e-9


def score_profile(results):
    """取出结果里的 rerank 分数序列（缺失则为 None），保持名次顺序。"""
    return [r.get("rerank_score") for r in (results or [])]


def rel_gap(results):
    """相对首位优势（无量纲，0~1）。见上方长注释里对口径的推导。

    对入参的要求很宽松，便于纯函数单测：
      * 少于 2 条候选 → 无法比较，返回 0.0（无依据）；
      * 分数缺失（None）→ 该条被剔除；剔除后不足 2 条同样返回 0.0。
    """
    scores = [float(s) for s in score_profile(results) if s is not None]
    if len(scores) < 2:
        return 0.0
    head = scores[0]
    rest = scores[1:]
    span = head - min(scores)
    if span <= _EPS:
        # 全体分数重合：候选之间没有任何区分度 → 没有可依赖的证据。
        return 0.0
    raw = (head - (sum(rest) / len(rest))) / span
    # 截断到 [0,1]：理论上不会越界，显式截断是为了让浮点误差/异常输入
    # （比如调用方传了未排序的结果）不至于产出 >1 的值而污染聚合。
    return max(0.0, min(1.0, raw))


def evaluate_negative(probe, results):
    """负样本判定：系统是否"意识到自己无据可依"。

    返回 {"refuse", "rel_gap", "head", "n_books", "n", "empty"}：
      refuse    True = 判定为应拒答（**这是期望结果**），即 rel_gap 低于阈值；
                False = 系统仍会把 top_k 当作答案喂给 LLM（幻觉风险）。
                results 为空时也记 True：检索器什么都没给，是无据可依的
                一种（虽然是最粗暴的一种，故单列 empty 便于区分）。
      rel_gap   相对首位优势（连续值）—— 报告里给出均值/分布，
                避免只看布尔值丢失"差多少"的信息。
      head      首位 rerank 分数，仅作记录。**不要**拿它做任何判定：
                实测正向最低 -0.11、负样本最高 2.88，两者区间重叠，
                任何绝对阈值都会同时误伤正确回答和放行域外问题。
      n_books   top_k 里出现了**几本书**（书目分散度）。这是本模块里唯一
                有实测区分度的相对信号（正向 22/24 集中在 1 本，负样本 6/7
                跨 2~3 本），但它衡量的是"领域归属"而不是"相关度"：
                跨书提问（"孙悟空和关羽谁更厉害"）也可能全部落在同一本书里。
                故只作诊断暴露，**不并入 refuse** —— 否则会让 refuse_rate
                看起来"有拒答能力"，而当时的引擎确实一条拒答通路都没有。
                （引擎**有**拒答信号了（rag_engine.confidence_signal 的 refuse，实测校准：误拒 0/24、拒答召回 6/10），但它是**分层**的 —— 检索侧只给信号，最终由生成侧结合上下文裁定，且不硬拦用户。因此"检索仍返回了 top_k 条"并不等于拒答失败。）
      n         参与判定的候选条数。
      empty     结果是否为空。
    """
    scores = [float(s) for s in score_profile(results) if s is not None]
    if not results:
        return {"refuse": True, "rel_gap": 0.0, "head": None,
                "n_books": 0, "n": 0, "empty": True}
    gap = rel_gap(results)
    return {
        "refuse": gap < REFUSE_REL_GAP,
        "rel_gap": gap,
        "head": scores[0] if scores else None,
        "n_books": len({r.get("book", "") for r in results}),
        "n": len(scores),
        "empty": False,
    }


# ============================================================================
# 分类聚合
# ============================================================================
# 为什么不给一个"总分"：三类探针测的是三种不同能力，合成一个数以后
# "短查询 100%、多轮 0%" 会显示成 50 分，看不出问题在哪。
def _rate(per_probe, keys):
    """把逐条判定折算成比率。keys 之外的键（如 books/resolved）自动忽略。"""
    n = len(per_probe) or 1
    out = {"n": len(per_probe)}
    for k in keys:
        out[k] = sum(1 for v in per_probe.values() if v.get(k)) / n
    return out


POSITIVE_METRICS = ("book@1", "book@k", "kw@k")
MULTITURN_METRICS = ("book@1", "book@k", "kw@k", "topic@k")


def aggregate_positive(per_probe):
    """短查询 + 长问句的 {book@1, book@k, kw@k} 比率（含 n）。"""
    return _rate(per_probe, POSITIVE_METRICS)


def aggregate_multiturn(per_probe):
    """多轮探针比率：三指标口径同单轮，外加 topic@k。"""
    return _rate(per_probe, MULTITURN_METRICS)


def aggregate_multiturn_by_kind(per_probe, probes=None):
    """按 kind 分组的多轮比率（含每类各自的 n）。

    为什么必须分组：合起来只有 book@1 一个数，而"代词没消解""实体切错了"
    "省略没补全"的修法完全不同 —— 实测不同 kind 的失手率本来就不一样
    （代词类靠改写就能救，recall_detail 类要的是上下文保持，改写无能为力）。
    只报总数会把后者的问题记在前者账上。未出现在 per_probe 里的 kind 不返回，
    避免给出一堆 n=0 的 0% 被误读成"全错"。
    """
    probes = MULTITURN_PROBES if probes is None else probes
    out = {}
    for kind in VALID_MULTITURN_KINDS:
        sub = {p.query: per_probe[p.query] for p in probes
               if p.kind == kind and p.query in per_probe}
        if sub:
            out[kind] = aggregate_multiturn(sub)
    return out


def aggregate_negative(per_probe):
    """负样本比率。

    refuse_rate      正确拒答率（越高越好）—— 主要的拒答能力指标。
                     注意：实测本栈上它接近 0，原因见 REFUSE_REL_GAP 上方
                     的校准说明，不要把它当"指标失灵"。
    answer_rate      仍然作答的比例（越低越好），等于 1-refuse_rate，
                     单列是因为"没拒答"才是幻觉风险的直接来源。
    empty_rate       靠"检索结果为空"实现的拒答占比 —— 把这种和"靠分数
                     形态判断出无依据"区分开：前者是检索失败，不是能力。
    mean_rel_gap     相对首位优势均值（连续值，便于观察趋势而不是只看阈值）。
    mean_n_books     top_k 的书目分散度均值（1.0 = 全部集中在同一本书）。
                     实测：正向 1.08 本 vs 负样本 2.14 本 —— 这是本模块里
                     唯一有区分度的相对信号，但它不是判定，只是观察窗。
    """
    n = len(per_probe) or 1
    vals = list(per_probe.values())
    return {
        "n": len(per_probe),
        "refuse_rate": sum(1 for v in vals if v.get("refuse")) / n,
        "answer_rate": sum(1 for v in vals if not v.get("refuse")) / n,
        "empty_rate": sum(1 for v in vals if v.get("empty")) / n,
        "mean_rel_gap": sum(float(v.get("rel_gap", 0.0)) for v in vals) / n,
        "mean_n_books": sum(float(v.get("n_books", 0)) for v in vals) / n,
    }


def aggregate_all(positive=None, negative=None, multiturn=None):
    """三类指标一次性汇总，只收传入的类别。

    每类的"率"必须按各自的 n 算，不能合成一个总数：
      * positive 的 n = 12 短 + 12 长，混在一起会掩盖"长问句更差"；
      * negative 的 n 与正向无关，分母不同，本来就不能相加。

    用法（调用方自己跑检索，本模块只负责判定与折算）：
        agg = aggregate_all(
            positive={p.query: evaluate_probe(p, run(p)) for p in PROBES},
            negative={p.query: evaluate_negative(p, run(p)) for p in NEGATIVE_PROBES},
            multiturn={p.query: evaluate_multiturn(p, run(p)) for p in MULTITURN_PROBES},
        )
    返回 {"n_total":…, "positive":…, "negative":…, "multiturn":…}，
    未传入的类别不出现在结果里（而不是给一堆 0，避免被误读成"全错"）。
    """
    out = {}
    n_total = 0
    if positive is not None:
        out["positive"] = aggregate_positive(positive)
        n_total += len(positive)
    if negative is not None:
        out["negative"] = aggregate_negative(negative)
        n_total += len(negative)
    if multiturn is not None:
        out["multiturn"] = aggregate_multiturn(multiturn)
        n_total += len(multiturn)
    out["n_total"] = n_total
    return out
