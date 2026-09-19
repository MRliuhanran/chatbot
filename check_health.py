#!/usr/bin/env python3
"""
一键健康检查 —— 确认 RAG 服务各环节是否可用。

检查项（每项互不依赖，单项失败不中断）：
  1. 配置常量与模型/缓存目录
  2. books 源文本齐全
  3. Qdrant 集合（rag_engine.COLLECTION_NAME）非空 + 元数据完整
  4. 混合检索：集合同时具备稠密/稀疏空间、点上两种向量都已写入、稀疏单通道可召回
  5. Ollama 后端可达（/api/tags）
  6. 生成模型可用（一次简短 think:false 问答）

用法:
  python check_health.py            # 全部检查
  python check_health.py --offline  # 跳过 Ollama/模型在线检查（快速）
退出码: 0=全部通过 1=存在失败
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag_engine as RE  # noqa: E402
#: 元数据检查里"缺了就判失败"的字段。schema 本身在 rag_engine._PAYLOAD_SPEC，
#: 这里只是挑了其中几个当硬门槛（旧索引可能没有结构化字段，那些只降级不报错）。
REQUIRED_PAYLOAD_FIELDS = ("book", "chunk_index", "parent_text", "contextual_text")
from ollama_client import MODEL, OLLAMA_BASE_URL  # noqa: E402

from qdrant_client import QdrantClient  # noqa: E402


def ok(msg):
    print(f"  \u2705 {msg}")


def bad(msg):
    print(f"  \u274c {msg}")
    return 1


def main():
    parser = argparse.ArgumentParser(description="RAG 服务健康检查")
    parser.add_argument("--offline", action="store_true", help="跳过 Ollama / 模型在线检查")
    # 默认 120s 而非 30s：这一步跑的是**非流式**请求，而模型冷启动时要先把
    # 4B 权重读进内存（8GB 机器上实测首字节常 >=30s）。30s 会把"模型没预热"
    # 报成"健康检查失败"，与 D7（把"连不上"误报成"库为空"）是同一类误诊 ——
    # 都是让使用者朝错误的方向排查。
    parser.add_argument("--timeout", type=int, default=120,
                        help="Ollama 问答超时(秒)，默认 120（冷启动需加载模型）")
    args = parser.parse_args()

    fail = 0

    print("=" * 60)
    print("RAG 健康检查")
    print("=" * 60)

    print("\n[1] 配置")
    if not RE.EMBED_MODEL_PATH or not os.path.isdir(RE.EMBED_MODEL_PATH):
        fail += bad(f"Embedding 模型目录缺失: {RE.EMBED_MODEL_PATH}")
    else:
        ok(f"Embedding 模型: {RE.EMBED_MODEL_PATH}")
    if not os.path.isdir(RE.RERANK_MODEL_PATH):
        fail += bad(f"Reranker 模型目录缺失: {RE.RERANK_MODEL_PATH}")
    else:
        ok(f"Reranker 模型: {RE.RERANK_MODEL_PATH}")
    ok(f"生成模型: {MODEL}  @  {OLLAMA_BASE_URL}")

    # 生成侧上下文注入：**报出实际的装配形态**，同样不照抄文档。
    # 这一段的存在理由就是 2026-09-18 那次静默失效 —— 三段提示词被置空后
    # UI/API 一切正常，召回的正文一条也没进模型，只有读代码才发现。
    # 现在形态是可验证的：规则在 system、正文是独立消息，两者分开报。
    if not RE.RAG_USE_CONTEXT:
        print("  \u26a0\ufe0f  上下文注入: **关闭**（RAG_USE_CONTEXT=0 对照模式）"
              " —— system 为空且不发资料消息，模型只靠参数化记忆作答")
    elif RE.SYSTEM_RULES_PROMPT and RE.CONTEXT_MESSAGE_HEADER:
        ok(f"上下文注入: system=规则({len(RE.SYSTEM_RULES_PROMPT)} 字) + "
           f"独立资料消息(role={RE.CONTEXT_ROLE})")
    else:
        fail += bad("上下文注入: RAG_USE_CONTEXT=1 但规则/资料头为空串 —— "
                    "RAG 会静默退化成凭记忆作答，检查 rag_engine 的三段提示词")

    # 多轮改写开关：**报出实际生效值**，而不是照抄文档。
    # 本仓库已踩过两次"文档里写了就当已生效"（RAG_QDRANT_HOST、RAG_ABSTAIN_RATIO），
    # RAG_QUERY_REWRITE 是第三次（AGENTS.md 写 1、.env 写 0）。关掉后失败还是
    # **静默**的：生成侧照传历史，模型凭记忆也能把指代听懂、答得像模像样，
    # 只是上下文里全是错书的正文，从回答上分辨不出来。
    import query_rewrite as QR

    if QR.REWRITE_ENABLED:
        # 历史窗口：max_turns=0 表示**不再按轮数截断**（整段历史进 prompt，
        # 由模型窗口决定看得到多少）。照字面打"只取最近 0 轮"是错的读数 ——
        # 0 在这里是"不限"，不是"一轮都不给"。
        if QR.REWRITE_MAX_TURNS and QR.REWRITE_MAX_TURNS > 0:
            window = f"只取最近 {QR.REWRITE_MAX_TURNS} 轮"
        else:
            window = (f"历史全量（不按轮数截断；粗略窗口 "
                      f"{QR.REWRITE_NUM_CTX} 字，超出只报警不裁剪）")
        ok(f"多轮查询改写: 开启（模型 {QR.REWRITE_MODEL}，{window}，失败退回原查询）")
    else:
        print("  \u26a0\ufe0f  多轮查询改写: **关闭**（RAG_QUERY_REWRITE=0）"
              " —— 指代性问句会以字面检索，实测\"他最后结局如何\""
              " 召回 4/5 条是错书。恢复：.env 置 1 或 RAG_QUERY_REWRITE=1 覆盖")

    # 多轮降级链与融合：同样报**实际生效值**。
    # 这两项在真实配置里出过同一个坑 —— 判据/文档说的和实际跑的不是一回事
    # （.env=0 却把"改写开着"报出来；融合开着但重排仍按主路问句打分）。
    # 一条命中的问句 + 每档的 state 一眼可见，就不必再去读代码推断。
    if QR.REWRITE_ENABLED and QR.CONCAT_ENABLED:
        ok(f"多轮降级链: LLM 改写 → 拼接上轮用户句 → 字面原句"
           f"（拼接档截断 {QR.CONCAT_MAX_CHARS} 字）")
    elif QR.REWRITE_ENABLED:
        print("  \u26a0\ufe0f  多轮降级链: 只剩 LLM 改写 → 字面原句"
              "（RAG_QUERY_REWRITE_CONCAT=0，实测 topic@k 会掉到 42.9%）")
    elif QR.CONCAT_ENABLED:
        ok("多轮降级链: 拼接上轮用户句 → 字面原句（LLM 档关闭）")
    else:
        print("  \u26a0\ufe0f  多轮降级链: 两档都关，多轮等于字面检索"
              "（实测 book@1 100%→81%、topic@k 95.2%→42.9%）")

    if RE.QUERY_FUSION:
        ok(f"多路召回融合: 开启（LLM 档成功时额外召回拼接档；"
           f"rerank 打分口径 {RE.RERANK_QUERY_MODE}）")
        if RE.RERANK_QUERY_MODE != "primary":
            print(f"  \u2139\ufe0f  RAG_RERANK_QUERY={RE.RERANK_QUERY_MODE} 是实测未定论项"
                  "（新口径下与 primary 持平），改默认前请先扩探针")
    else:
        print("  \u2139\ufe0f  多路召回融合: 关闭（RAG_QUERY_FUSION=0）"
              " —— 修正 topic@k 别名口径后实测 3 改善 / 1 退化"
              "（kw@k 70%→80%，但一条 entity_switch 的 topic@k 掉成 N），"
              "撑不起改默认值；两轮数字见 .env 注释")

    print("\n[2] 源文本")
    txt = [f for f in os.listdir(RE.BOOKS_DIR) if f.endswith(".txt")] if os.path.isdir(RE.BOOKS_DIR) else []
    if len(txt) >= 1:
        ok(f"books/ 含 {len(txt)} 本: {', '.join(txt)}")
    else:
        fail += bad("books/ 中没有 .txt")

    print("\n[3] 向量库 Qdrant")
    qdrant_count = 0
    client = None
    coll = None
    try:
        # 这个 try 只应覆盖**与 Qdrant 通信**的部分。任何本地动作（读分块产物、
        # 算词表指纹）都必须自己在内部兜异常 —— 否则它们的失败会被统一报成
        # "无法连接 Qdrant Docker服务"，把人支向 docker compose 而不是真正的原因。
        # timeout 与 RAGEngine._get_client() 对齐（默认 5s 太短，负载稍高就会误报）
        client = QdrantClient(host=RE.QDRANT_HOST, port=RE.QDRANT_PORT, timeout=60)

        # 集合名运行时解析（别名优先）。启用别名后具体集合名是
        # books_v3__<build_id前8位>，写死 coll 会把
        # "索引好好的" 误报成 "集合不存在（索引未构建）"。
        coll = RE.default_collection_name(client)

        # 必须先判断集合是否存在。直接 count() 在集合不存在时抛 404，
        # 会被下面的 except 归为"无法连接 Qdrant"—— 把"索引没建"误报成
        # "Docker 挂了"，排查方向完全错误。这里与 RAGEngine.count() 保持一致。
        if not client.collection_exists(coll):
            fail += bad(
                f"集合 {coll} 不存在（索引未构建），请运行: python app.py index"
            )
        else:
            qdrant_count = client.count(collection_name=coll, exact=True).count
            if qdrant_count > 0:
                ok(f"{coll} 记录数: {qdrant_count}")

                # 数量一致性：集合点数必须等于 chunks.json 条数
                # （否则索引与分块源已脱节，检索会漏召回且无从察觉）
                #
                # ⚠️ 这段是**本地文件读取**，必须自己兜异常：外面的 except 把
                # 任何异常都报成"无法连接 Qdrant Docker服务"，于是一个损坏的
                # chunks.json 会被误诊成 Docker 没起 —— 正是 D7 要消灭的那类
                # 误导（它会把人支去 docker compose up 而不是重跑 process）。
                if os.path.exists(RE.CHUNKS_JSON):
                    try:
                        import json as _json
                        with open(RE.CHUNKS_JSON, encoding="utf-8") as f:
                            n_chunks = len(_json.load(f))
                    except Exception as exc:
                        fail += bad(
                            f"分块产物 {RE.CHUNKS_JSON} 读取失败（与 Qdrant 无关）: "
                            f"{type(exc).__name__}: {exc} —— 请重跑: python app.py process"
                        )
                    else:
                        if n_chunks == qdrant_count:
                            ok(f"数量一致: chunks.json {n_chunks} 条 == 集合 {qdrant_count} 条")
                        else:
                            fail += bad(
                                f"数量不一致: chunks.json {n_chunks} 条 != 集合 {qdrant_count} 条"
                                f"（索引与分块源脱节，请重跑: python app.py index）"
                            )
                else:
                    print(f"  ⏭  跳过数量一致性（找不到 {RE.CHUNKS_JSON}）")

                # 获取一条样本数据检查元数据
                sample = client.scroll(
                    collection_name=coll,
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )[0]
                if sample:
                    meta = sample[0].payload
                    # 这里是"哪些字段算硬要求"的**策略**，不是第二份 schema ——
                    # 字段名必须存在于 rag_engine.PAYLOAD_FIELDS（有测试断言这条子集关系，
                    # 免得这里写出一个 schema 里根本没有的名字而永远不报错）。
                    missing = [k for k in REQUIRED_PAYLOAD_FIELDS if k not in meta]
                    if missing:
                        fail += bad(f"元数据缺字段: {missing}")
                    else:
                        ok("元数据字段完整")

                # 词表指纹：改变稀疏词空间的东西，不一致会让稀疏通道静默错配
                try:
                    pts, _ = client.scroll(collection_name=coll, limit=1,
                                           with_payload=True, with_vectors=False)
                    idx_lex = ((pts[0].payload or {}).get(RE.LEXICON_ID_FIELD)
                               if pts else None)
                    cur_lex = RE.lexicon_fingerprint()
                    if idx_lex is None:
                        ok("词表指纹: 索引未记录（旧索引），跳过一致性检查")
                    elif idx_lex == cur_lex:
                        ok(f"词表指纹一致: {cur_lex}")
                    else:
                        fail += bad(
                            f"词表与索引不一致（索引 {idx_lex} / 当前 {cur_lex}）——"
                            f"稀疏通道词空间已错配，召回会静默下降。"
                            f"请运行: python reindex_sparse.py"
                        )
                except Exception as e:
                    fail += bad(f"词表一致性检查失败: {e}")
            else:
                fail += bad(f"{coll} 为空，请运行: python app.py index")
    except Exception as e:
        fail += bad(f"无法连接 Qdrant Docker服务 ({RE.QDRANT_HOST}:{RE.QDRANT_PORT}): {e}")

    print("\n[4] 混合检索（稠密 + 稀疏）")
    # 旧版这里只做了 hasattr(config) 判断，恒为真，永远打印"已启用"，
    # 因此"稀疏向量其实一条都没写"这个事实被掩盖了很久。改为真检查：
    # 集合配置 + 点上实际向量 + 稀疏单通道能否召回。
    if client is None or qdrant_count == 0:
        print("  ⏭  跳过（向量库不可用）")
    else:
        try:
            info = client.get_collection(collection_name=coll)
            sparse_names = list(info.config.params.sparse_vectors.keys())
            dense_names = list(info.config.params.vectors.keys())
            if sparse_names and dense_names:
                ok(f"集合配置: 稠密 {dense_names} + 稀疏 {sparse_names}")
            else:
                fail += bad(f"集合缺少向量空间: 稠密 {dense_names} / 稀疏 {sparse_names}")

            # 抽样覆盖检查走 RE.sample_vector_coverage（与 verify_qdrant 同一实现）
            cov = RE.sample_vector_coverage(client, coll, limit=50)
            if not cov["miss_dense"] and not cov["miss_sparse"] and cov["n"]:
                ok(f"抽样 {cov['n']} 条: 稠密/稀疏均已写入"
                   f"（首条稀疏 {cov['first_sparse_terms']} 个 term）")
            else:
                fail += bad(
                    f"抽样 {cov['n']} 条中: 缺稠密 {cov['miss_dense']} 条, "
                    f"缺稀疏 {cov['miss_sparse']} 条（稀疏没写 = 混合检索名存实亡）"
                )

            # 功能验证：用样本原文做稀疏单通道检索，必须能召回自己
            pts, _ = client.scroll(collection_name=coll, limit=1, with_payload=True)
            probe = (pts[0].payload.get("child_text", "")[:80] if pts else "")
            if probe:
                res = client.query_points(
                    collection_name=coll,
                    query=RE.sparse_encode(probe),
                    using=RE.SPARSE_VECTOR_NAME,
                    limit=3,
                )
                if res.points:
                    ok(f"稀疏单通道检索可用（{len(res.points)} 条命中，最高分 {res.points[0].score:.2f}）")
                else:
                    fail += bad("稀疏单通道检索返回空，稀疏索引可能未生效")
        except Exception as e:
            fail += bad(f"混合检索检查失败: {e}")

    if args.offline:
        print("\n(offline 模式：跳过 Ollama 在线检查)")
    else:
        print("\n[5] Ollama 后端")
        import requests
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                ok(f"Ollama 可达，已装模型: {models}")
                if MODEL not in models:
                    fail += bad(f"MODEL={MODEL} 未安装，请: ollama pull {MODEL}")
            else:
                fail += bad(f"Ollama /api/tags 返回 HTTP {r.status_code}")
        except Exception as e:
            fail += bad(f"Ollama 不可达: {e}")

        print("\n[6] 生成模型应答（think:false 简短问答）")
        import requests
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "用一句话回答：孙悟空出自哪部小说？"}],
            "stream": False,
            "think": False,
            "options": {"num_predict": 64},
        }
        t0 = time.time()
        try:
            r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=args.timeout)
            elapsed = time.time() - t0
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                fail += bad(f"模型应答返回错误: {data['error']}")
            else:
                reply = (data.get("message", {}) or {}).get("content", "") if isinstance(data, dict) else ""
                ok(f"应答成功 耗时 {elapsed:.1f}s: {reply[:60]!r}")
        except requests.exceptions.Timeout:
            # 超时必须与"模型坏了"分开报：绝大多数情况只是冷启动需要加载权重
            fail += bad(
                f"模型应答超时({args.timeout}s)。这**多半不是故障**：非流式请求会在"
                f"模型冷启动时等待权重加载（8GB 机器实测常超过 30s）。\n"
                f"    先手动预热一次再重跑，或直接调大超时，例如:\n"
                f"      curl -s {OLLAMA_BASE_URL}/api/chat -d "
                f"'{{\"model\":\"{MODEL}\",\"messages\":[{{\"role\":\"user\","
                f"\"content\":\"hi\"}}],\"stream\":false,\"think\":false}}' >/dev/null\n"
                f"      python check_health.py --timeout 300"
            )
        except Exception as e:
            fail += bad(f"模型应答失败({args.timeout}s): {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    if fail:
        print(f"健康检查失败: {fail} 项")
        print("=" * 60)
        sys.exit(1)
    else:
        print("健康检查通过 ✓")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
