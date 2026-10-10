#!/usr/bin/env bash
# 发布前闸门：把「能不能推/能不能上」收敛成一次本地检查，失败即退出非 0。
#
# 为什么不是 CI：本项目的质效支点是「指纹闸门 + 原子切换」，而判据需要 Qdrant 在跑
# （stages 的 C/D 段）。这个脚本就是把那道判据固化下来，避免「凭印象以为是新鲜的」。
#
# 闸门顺序（前一步不过就不跑后面的，省时间也省误导性输出）：
#   1. lexicon  —— 词表自检（纯本地，快）
#   2. Qdrant   —— 连不上就不必谈 C/D；给出启动提示
#   3. stages   —— A/B/C/D 四阶段指纹比对，退出码 0 才算全新鲜
#   4. 冒烟      —— 仅当 API 已在跑时才查（启动 API 需加载嵌入/重排模型，不该塞进这个快闸门）
#
# 用法：tools/deploy-check.sh
# 环境：PYTHON（默认 python）  SMOKE_URL（默认 http://127.0.0.1:8000，含协议）  SMOKE_QUERY（冒烟查询）
set -u

# 定位到仓库根：脚本可能从任意 cwd 被调用，而下面所有命令都按根目录相对路径。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1

PY="${PYTHON:-python}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
SMOKE_URL="${SMOKE_URL:-http://127.0.0.1:8000}"
SMOKE_QUERY="${SMOKE_QUERY:-红楼梦}"

_fail=0
_say() { printf '\033[1m[deploy-check]\033[0m %s\n' "$*"; }
_ok() { printf '  ✅ %s\n' "$*"; }
_bad() { printf '  ❌ %s\n' "$*"; _fail=1; }
_skip() { printf '  ⏭  %s\n' "$*"; }

# ── 1. 词表自检 ─────────────────────────────────────────────────────────────
_say "1/4 词表自检（lexicon）"
if "${PY}" chatbot.py lexicon -q; then
    _ok "词表可用"
else
    _bad "词表自检未通过 -> 修 lexicon 或先跑 lexicon 看明细"
    # 词表坏了会污染稀疏通道；后面三步都建立在它之上，没必要继续。
    exit 1
fi

# ── 2. Qdrant 可达 ──────────────────────────────────────────────────────────
_say "2/4 Qdrant 可达性（${QDRANT_URL}）"
if curl -fsS "${QDRANT_URL}/healthz" >/dev/null 2>&1; then
    _ok "Qdrant 就绪"
else
    # 连不上时 stages 会把 C/D 判成「未验证」并退出码 1 —— 那是真实结论，不是脚本故障。
    # 与其让用户拿到一个含糊的 stages 失败，不如在这里把根因和恢复动作说清楚。
    _bad "Qdrant 不可达（${QDRANT_URL}/healthz）；C/D 段无法验证"
    _say "  恢复：docker compose up -d qdrant   # 然后重跑本脚本"
    exit 1
fi

# ── 3. 四阶段指纹 ────────────────────────────────────────────────────────────
_say "3/4 产物新鲜度（stages：A 分块 / B 稠密 / C 稀疏 / D 入库）"
if "${PY}" chatbot.py stages; then
    _ok "A/B/C/D 全新鲜，无需重跑"
else
    _bad "存在过期或未验证阶段（见上方 stages 输出与最小重跑命令）"
    exit 1
fi

# ── 4. 冒烟（可选：仅当 API 已在跑）────────────────────────────────────────
# 两段判据分开：curl 失败=API 没起（跳过，不是闸门失败）；curl 成功但 results 空=检索
# 链路异常（真失败）。旧写法 grep '"results"' 会把空结果数组也判成通过（假通过）。
_say "4/4 检索冒烟（/search）"
_resp="$(curl -fsS -X POST "${SMOKE_URL}/search" \
        -H 'Content-Type: application/json' \
        --data "{\"query\": \"${SMOKE_QUERY}\", \"top_k\": 3}" 2>/dev/null)" || _resp=""
if [ -z "${_resp}" ]; then
    # API 没起不是闸门失败：起服务要加载嵌入/重排模型，属运行期的事，不该拖进这个快闸门。
    # 但必须留下明确后续动作，否则「没冒烟」与「冒烟通过」在结果上无从区分。
    _skip "API 未在 ${SMOKE_URL} 运行；冒烟未执行。发布后请确认：curl -X POST ${SMOKE_URL}/search -H 'Content-Type: application/json' -d '{\"query\":\"${SMOKE_QUERY}\"}'"
elif printf '%s' "${_resp}" | "${PY}" -c 'import sys,json; sys.exit(0 if json.load(sys.stdin).get("results") else 1)'; then
    _ok "/search 返回非空 results"
else
    _bad "/search 返回空 results（查询=${SMOKE_QUERY}）—— 检索链路可能异常，排查后重跑"
fi

echo
if [ "${_fail}" -eq 0 ]; then
    _say "全部闸门通过（冒烟见上）→ 可以发布"
    exit 0
else
    _say "存在未通过项（见上方 ❌）→ 不要发布"
    exit 1
fi
