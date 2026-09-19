"""L3：检索质量回归 —— 需要 Qdrant + 嵌入/重排模型（分钟级）。

与 `compare_ab.py` 的分工：
    compare_ab.py   两个集合谁更好（相对比较，换整套方案时用）
    本文件          这次改动有没有让检索变差（与存档基线比，改动分块/参数/模型时用）

基线由 `python -m tests.record_baseline` 录制到
`tests/golden/retrieval_baseline.json`。

检索是确定性的（嵌入无随机性、reranker 只做前向、无采样），所以
判定用**零容差**：任何下降都是真实回归。若本层开始出现不稳定，
那本身就是需要查明的缺陷，而不是把容差放宽了事。

运行: pytest -m slow
"""

import json
import os

import pytest

import rag as RE
from tests.eval_runner import (
    aggregate,
    multiturn_degrade_state,
    multiturn_fusion_state,
    run_multiturn,
    run_probes,
    short_only,
)
from tests.probes import PROBES

# 只标 slow：L3 用 `pytest -m slow` 选中，这样 `-m needs_qdrant` 精确对应 L2。
# （本层同样依赖 Qdrant，但由 engine fixture 在不可达时跳过，不必额外标记。）
pytestmark = [pytest.mark.slow, pytest.mark.needs_models]

BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "golden", "retrieval_baseline.json"
)


@pytest.fixture(scope="module")
def baseline():
    if not os.path.exists(BASELINE_PATH):
        pytest.skip(
            f"缺少黄金基线 {BASELINE_PATH}，请先运行: python -m tests.record_baseline"
        )
    with open(BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def current(engine):
    return run_probes(engine, top_k=RE.TOP_K)


@pytest.fixture(scope="module")
def current_multiturn(engine):
    """多轮探针 —— 带 history 跑，因此会真的触发查询改写。"""
    return run_multiturn(engine, top_k=RE.TOP_K)


class TestProbeSanity:
    def test_probes_are_wellformed(self):
        for p in PROBES:
            assert p.query and p.book and p.keywords

    def test_probe_keywords_exist_in_corpus(self, chunks):
        """探针关键词必须真的出现在语料里。

        否则这条探针永远不可能命中，指标会被它稀释成一个无意义的下界
        （verify_qdrant.py 里已有"语料中无此词则跳过"的处理，这里前置成断言）。
        """
        for p in PROBES:
            hit = sum(
                1 for c in chunks
                if c["book"] == p.book and any(k in c["child_text"] for k in p.keywords)
            )
            assert hit > 0, (
                f"探针 {p.query!r} 的关键词 {p.keywords} 在《{p.book}》语料中一次都没出现，"
                f"该探针无法命中"
            )


class TestRetrievalRegression:
    def test_aggregate_not_worse_than_baseline(self, baseline, current):
        """三个指标都不得低于基线。"""
        base = baseline["aggregate"]
        now = aggregate(current)
        regressions = []
        for key in ("book@1", "book@k", "kw@k"):
            if now[key] < base[key]:
                regressions.append(f"{key}: {base[key]:.1%} → {now[key]:.1%}")
        assert not regressions, (
            "检索质量出现回归（零容差判定）: " + "; ".join(regressions)
            + "\n若确认是可接受的权衡，请重跑 python -m tests.record_baseline 更新基线。"
        )

    def test_no_per_probe_regression(self, baseline, current):
        """逐条探针也不得从命中掉成未命中（聚合指标会掩盖个别退化）。"""
        bad = []
        for q, v in current.items():
            b = baseline["per_probe"].get(q)
            if b is None:
                continue
            for key in ("book@1", "book@k", "kw@k"):
                if b[key] and not v[key]:
                    bad.append(f"{q}.{key}")
        assert not bad, f"以下探针由命中退化为未命中: {bad}"

    def test_baseline_is_comparable(self, baseline, current):
        """基线必须与当前探针集**规模一致**，否则比的是两个不可比的量。

        这条是补上一个隐患：`PROBES` 从 12 条扩到 24 条（12 短 + 12 长）之后，
        旧写法会拿"24 条的聚合比率"去比"12 条的存档比率"，键名相同所以不会报错，
        但那个比较没有任何意义（长问句天然更难，数字会莫名其妙变低）。
        与其靠人记得重录，不如把"不可比"变成显式失败。
        """
        base_n = baseline.get("aggregate", {}).get("n")
        now = aggregate(current)
        assert base_n == now["n"], (
            f"基线记的是 n={base_n} 条探针，当前 n={now['n']} 条 —— "
            f"两者的比率不可比。请重录: python -m tests.record_baseline"
        )

    def test_collection_matches_baseline(self, baseline, engine):
        """基线必须是对当前这个集合录的，否则比对无意义。"""
        assert baseline.get("collection") == RE.COLLECTION_NAME, (
            f"基线录自集合 {baseline.get('collection')}，当前是 {RE.COLLECTION_NAME}"
        )
        assert baseline.get("chunks") == engine.count(), (
            f"基线录制时 {baseline.get('chunks')} 条，当前 {engine.count()} 条 —— "
            f"索引已变，基线不可比，请重新录制"
        )

    def test_rerank_input_unchanged(self, baseline):
        """重排输入规模与截断长度也是会影响质量的有效配置，变了就该重新记录。"""
        assert baseline.get("rerank_top_k") == RE.RERANK_TOP_K
        assert baseline.get("rerank_max_length") == RE.RERANK_MAX_LENGTH

    def test_reports_metrics(self, current, capsys):
        """非断言：打印本次指标（-s 时可见）。"""
        now = aggregate(current)
        short = short_only(current)
        with capsys.disabled():
            print(
                f"\n  正向全部 n={now['n']}: book@1={now['book@1']:.1%}  "
                f"book@k={now['book@k']:.1%}  关键词@k={now['kw@k']:.1%}"
                f"\n  仅短探针 n={short['n']}: book@1={short['book@1']:.1%}  "
                f"book@k={short['book@k']:.1%}  关键词@k={short['kw@k']:.1%}"
                f"   ← 与旧基线可比的那组"
            )


class TestShortProbeRegression:
    """冻结的 12 条短探针单独回归。

    为什么要单独一组：旧基线只有这 12 条，而 PROBES 现在是 12 短 + 12 长。
    把长问句混进聚合比率后数字会天然变低（长问句更难），"是不是退化了"
    这个判断会被稀释。短探针组是唯一能与历史数字**按位置对拍**的一组。
    """

    def test_short_subset_not_worse(self, baseline, current):
        base = baseline.get("aggregate_short") or baseline["aggregate"]
        now = short_only(current)
        regressions = [
            f"{k}: {base[k]:.1%} → {now[k]:.1%}"
            for k in ("book@1", "book@k", "kw@k")
            if k in base and now[k] < base[k]
        ]
        assert not regressions, "短探针组出现回归: " + "; ".join(regressions)


class TestMultiturnRegression:
    """多轮探针回归 —— 衡量查询改写是否真的起作用。

    指标里有 `topic@k`（指代对象是否出现在召回正文里），它抓的是一种
    book@k 抓不到的失效：书对了但内容与指代对象无关。实测"他最后死在哪里"
    就是 book@1=Y 而 topic@k=N。
    """

    def test_multiturn_not_worse(self, baseline, current_multiturn):
        base = (baseline.get("aggregate_all") or {}).get("multiturn")
        if not base:
            pytest.skip("基线未包含多轮指标，请重新录制: python -m tests.record_baseline")
        # 判据是「基线录于哪一档 vs 当前跑在哪一档」，**不是"改写开着没"**。
        # 曾经这里写的是 `if not REWRITE_ENABLED: skip`，理由是"改写关了多轮不可比" ——
        # 那个推论是错的：RAG_QUERY_REWRITE=0 只关掉 LLM 那一档，拼接档
        # （RAG_QUERY_REWRITE_CONCAT，默认 1）照样替换检索问句，多轮指标照样在动、
        # 照样可比。后果是**在默认配置下这条护栏整组失效**（.env 就是 0），
        # 而"多轮现在什么水平"恰恰是默认配置下最该被守住的东西。
        # 基线没记 state 时（旧格式）退回单看 rewrite_enabled，只为了不让老基线
        # 静默通过 —— 那种情况会明确 skip 并提示重录。
        now_state = multiturn_degrade_state()
        base_state = baseline.get("multiturn_degrade_state")
        if base_state is None:
            base_state = "llm" if baseline.get("rewrite_enabled") else "literal"
            pytest.skip(
                f"基线未记录多轮降级档（旧格式，按 rewrite_enabled={baseline.get('rewrite_enabled')} "
                f"推断为 {base_state}），而当前跑在 {now_state} 档，不可比。"
                f"请重录基线: python -m tests.record_baseline"
            )
        if base_state != now_state:
            pytest.skip(
                f"多轮降级档已变更（基线 {base_state} → 当前 {now_state}），检索问句口径不同，"
                f"不可比。要么用基线那一档重跑，要么重录基线: python -m tests.record_baseline"
            )
        # 融合档同理：开/关 RAG_QUERY_FUSION 会改变"召回几路、用哪句重排"，
        # 只比降级档会把两组不同口径的数字当成一组。旧基线没有这个字段时
        # 按 "False:primary" 兜底 —— 那正是它录制时的配置（融合默认关）。
        now_fusion = multiturn_fusion_state()
        base_fusion = baseline.get("multiturn_fusion_state", "False:primary")
        if base_fusion != now_fusion:
            pytest.skip(
                f"多路召回配置已变更（基线 {base_fusion} → 当前 {now_fusion}），"
                f"多轮指标口径不同，不可比。请重录基线: python -m tests.record_baseline"
            )
        # 必须用多轮自己的聚合器：它比单轮多一个 topic@k。复用 aggregate()
        # 会让 topic@k 静默变成 nan（打印出 "topic@k=nan%"），
        # 而 topic@k 恰恰是抓"书对了但内容与指代对象无关"的唯一指标。
        from tests.probes import aggregate_multiturn as _agg

        now = _agg(current_multiturn)
        if base.get("n") != now["n"]:
            # 分母不同不能比：基线 n=5、当前 n=21 时任何"回归/改善"都只是探针集换了。
            # 刻意 skip 而不是自动放宽或静默比较 —— 后者会让这条护栏彻底失效。
            pytest.skip(
                f"多轮探针集已变更（基线 n={base.get('n')}，当前 n={now['n']}），"
                f"分母不同不可比。启用改写并重录基线后再跑："
                f"python -m tests.record_baseline"
            )
        regressions = [
            f"{k}: {base[k]:.1%} → {now[k]:.1%}"
            for k in ("book@1", "book@k", "kw@k", "topic@k")
            if k in base and k in now and now[k] < base[k]
        ]
        assert not regressions, "多轮指标出现回归: " + "; ".join(regressions)

    def test_reports_multiturn(self, current_multiturn, capsys):
        from tests.probes import aggregate_multiturn as _agg

        now = _agg(current_multiturn)
        with capsys.disabled():
            print(f"\n  多轮 n={now['n']}: book@1={now['book@1']:.1%}  "
                  f"book@k={now['book@k']:.1%}  kw@k={now['kw@k']:.1%}  "
                  f"topic@k={now.get('topic@k', float('nan')):.1%}")


class TestNegativeProbesAreRecorded:
    """负样本只**记录现状、不设断言**。

    当前引擎的拒答是"分层"的：检索侧给一个实测校准的信号（误拒 0/24、
    拒答召回 6/10），最终由生成侧结合上下文裁定。因此"检索返回了 top_k 条"
    本身并不等于"拒答失败" —— 硬断言 refuse_rate 会逼着实现去迎合一个
    并不正确的口径（实测过：负样本里有 4 条属于"语料部分相关"，
    连"平均分 + 书目分散"都判不出来，见 PROJECT_DOC 的校准记录）。
    所以这里只在缺失时提醒，不 fail。
    """

    def test_baseline_contains_negative_section(self, baseline, capsys):
        neg = (baseline.get("aggregate_all") or {}).get("negative")
        with capsys.disabled():
            if neg:
                print(f"\n  负样本 n={neg['n']}: 拒答率={neg['refuse_rate']:.1%}  "
                      f"平均分散={neg.get('mean_n_books', float('nan')):.2f} 本"
                      f"（校验用，非断言）")
            else:
                print("\n  基线未包含负样本指标，建议重录: python -m tests.record_baseline")
