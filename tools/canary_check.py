# ── 金丝雀对拍/重捕：8 条查询 hybrid_search top-10 与基线逐位比（2026-11-12 F6 重写）──
# 为什么有它：检索链路任何一行改动（融合/重排/词表/闸门）都可能让排序静默漂移，
# evaluate 的统计读数（recall/mrr）对小扰动不敏感 —— 金丝雀要的是**逐位**一致：
# id/source/chunk_index/parent_id/score(6dp) 五元组完全相等才算绿。
# 三个历史坑（本次全部根治）：
#   1. ROOT 曾硬编码 "/Users/liuhranran/..." —— 换机器/换克隆即失效；改为
#      Path(__file__).resolve().parents[1]（tools/ 的上一级 = 仓库根，与 start.sh
#      的 ROOT 推导同一逻辑，可移植且不依赖 cwd）。
#   2. 基线曾放 artifacts/baseline/（git 未跟踪）—— 本地一份"真相"没进版本库，
#      换机器连比对对象都没有；迁 evalsets/canary.baseline.json 随仓库走。
#   3. 结果受 RAG_SERVER_FUSION / RAG_QUERY_CACHE_TTL 漂移：对拍在"融合路径 A"
#      下捕获、又在"路径 B"下复跑，分数进位差会让逐位比对假红（两路径语义都
#      正确，只是 RRF/重排输入序不同）。金丝雀钉死**规范性服务状态**再跑：
#      RAG_SERVER_FUSION=0（客户端双通道融合，读数与 D1 落地前口径连续）+
#      RAG_QUERY_CACHE_TTL=0（每查必直查，缓存命中/写入绝不进比对）。
#      必须在 import chatbot 之前设 —— 常量在 import 期读 env；chatbot 内部
#      load_dotenv(override=False) 保证 ambient 环境优先，钉住不被 .env 反转。
# --capture 需 --force：重捕 = 声明"源状态合法换代"，静默覆盖基线会把漂移合法化
# （基线漂移必须先解剖归因）。裸 --capture 直接拒绝并给指引。
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

BASELINE = ROOT / "evalsets" / "canary.baseline.json"
# 历史备份仍落 artifacts/baseline/（未跟踪的本机留档，与既有 .bak-* 约定一致），
# evalsets/ 只放"当前真相" —— 版本库里不堆日期备份。
BAK_DIR = ROOT / "artifacts" / "baseline"

ap = argparse.ArgumentParser(description="金丝雀：8 条查询 top-10 与基线逐位比（--capture --force 重捕）")
ap.add_argument("--capture", action="store_true", help="重捕基线（须与 --force 同用）")
ap.add_argument("--force", action="store_true", help="确认源状态已合法换代，允许覆盖基线")
args = ap.parse_args()

# 规范性服务状态（见文件头第 3 坑）：必须在 import chatbot 前设。
os.environ["RAG_SERVER_FUSION"] = "0"
os.environ["RAG_QUERY_CACHE_TTL"] = "0"

import chatbot   # 本仓 lint 面不含 E402（ruff.toml 未选 E 族），import 位置无需压制

if not BASELINE.exists():
    print(f"❌ 基线缺失: {BASELINE} —— 首次建基线: python tools/canary_check.py --capture --force")
    sys.exit(2)
old = json.loads(BASELINE.read_text(encoding="utf-8"))
queries = [cq["q"] for cq in old["queries"]]
engine = chatbot.RAGEngine()


def hit(c):
    # 五元组快照：id/source/chunk_index 定位，parent_id 看层级，score 6dp 进位敏感
    return {
        "id": c["id"],
        "source": chatbot._source_of(c),
        "chunk_index": c.get("chunk_index"),
        "parent_id": c.get("parent_id", ""),
        "score": round(float(c["rerank_score"]), 6),
    }


results = []
for q in queries:
    out = engine.hybrid_search(q, top_k=10)
    results.append({"q": q, "count": len(out), "hits": [hit(c) for c in out]})

if args.capture:
    if not args.force:
        print("❌ 重捕被拒绝：--capture 必须与 --force 同用。")
        print("   重捕 = 声明\"源状态合法换代\"；对拍先红时先解剖归因（是检索真变了，")
        print("   还是 serving 状态漂了），不许用重捕把漂移合法化。")
        sys.exit(2)
    BAK_DIR.mkdir(parents=True, exist_ok=True)
    backup = BAK_DIR / f"canary.bak-{datetime.date.today().isoformat()}"
    if BASELINE.exists() and not backup.exists():
        shutil.copy2(BASELINE, backup)
        print(f"旧基线已备份: {backup}")
    new = {
        "captured": datetime.datetime.now().isoformat(timespec="seconds"),
        "top_k": 10,
        "engine": "hybrid_search(RAG_SERVER_FUSION=0+RAG_QUERY_CACHE_TTL=0 钉死：http嵌入+客户端双通道融合+http重排+指纹闸门)",
        "queries": results,
    }
    tmp = str(BASELINE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(new, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, BASELINE)
    print(f"✅ 基线已重捕: {BASELINE}（captured={new['captured']}）")
    sys.exit(0)


def hitkey(h):
    return f"{h['id']}|{h.get('source')}|{h.get('chunk_index')}|{h.get('parent_id')}|{h.get('score')}"


total_bad = 0
for cq, gotq in zip(old["queries"], results):
    want = [hitkey(h) for h in cq["hits"]]
    got = [hitkey(g) for g in gotq["hits"]]
    if got == want:
        print(f"✅ {gotq['q']!r}  top-10 逐位一致")
    else:
        total_bad += 1
        print(f"❌ {gotq['q']!r}  期望 {len(want)} 条 / 实得 {len(got)} 条")
        for i, (w, g) in enumerate(zip(want, got)):
            if w != g:
                print(f"   首个分叉 @{i}: want={w}\n              got ={g}")
                break

print("RESULT:", "ALL-BIT-EXACT(8/8)" if total_bad == 0 else f"MISMATCH x{total_bad}")
sys.exit(1 if total_bad else 0)
