#!/usr/bin/env bash
# ── 一键发布：preflight → push（魔搭必需 + GitHub 尽力）→ 触发构建 → 轮询状态 → 运行日志验收 ──
# 用法:
#   bash tools/ms_release.sh                  # 完整发布（preflight+push+deploy+轮询+日志验收）
#   bash tools/ms_release.sh --verify-only    # 不 push 不 deploy，仅验收当前运行日志
#   bash tools/ms_release.sh --no-deploy      # preflight + 只 push，不触发构建
#   bash tools/ms_release.sh --allow-dirty    # 允许带未提交改动发布（默认拒绝）
#   bash tools/ms_release.sh --strict-github  # GitHub 推送失败也视为失败（默认警告放行）
#   bash tools/ms_release.sh --skip-preflight # 应急：跳过 preflight 全部闸门（verify-build /
#                                             # Qdrant 未停 / git 追踪）；EXPECTED_FP 仍尽量从清单取
#
# 发布前置闸门（preflight，默认开；每条对应一次实证事故）:
#   ①）python chatbot.py verify-build：manifest 的 chunk_fingerprint 必须等于本地现算指纹。
#      —— 2026-10-31 远程事故：索引=923236fe7e4acc7c / 当前=3502968fe6f7039e，数据落后于
#         应用 → 检索闸门 fail-closed。本地先拦，比上线后被拒服务便宜得多。
#   ②）本机 Qdrant 仍在跑 → 拒绝发布。运行中的 Qdrant 持续写 storage（WAL/segment churn），
#      而 .dockerignore 排除 wal/open-*，打出来的是半写快照——与 10-31 事故同类：数据不可信。
#      先 ./start.sh stop（让它刷盘）再发布。
#   ③）从 artifacts/index_manifest.json 取 chunk_fingerprint 作 EXPECTED_FP；取不到=拒绝：
#      没有清单就无法证明镜像里的索引与语料同源，验收也没有比对基准。
#   ④）git 追踪完整性：chunk 清单必须已跟踪；storage/ 在 .gitignore 里，**新索引文件不在
#      git 索引里就不会随 push 进构建上下文**（镜像仍是旧 storage = 923/350 事故重演）。
#
# 验收判据（新契约，2026-10-31 起；与部署事故复盘同源）:
#   1) 启动期必须有 🔧 分块指纹: 索引=… / 当前=…（cmd_all 启动即打）。**到点未见 = FAIL**，
#      不再按「未见=ABSENT 放行」——旧契约要等用户先发一次检索才打指纹行，10-31 事故正是
#      这样漏过了 923/350。且两侧必须相等、索引侧必须等于 EXPECTED_FP。
#   2) 模型装载/落位行（📦 [模型装载] 全部就绪（零下载命中） 或 [模型落位] ×≥2）。
#   3) 运行日志无 RuntimeError、无 ❌ [all]（启动编排失败）；infinity 就绪行尽力检查。
#   4) 🧮 词表指纹 启动行只记录不阻断（稀疏侧仍由检索期词表闸门兜底）。
set -euo pipefail

OWNER="${MS_OWNER:-liuhanran}"
STUDIO="${MS_STUDIO:-chatbot}"
BRANCH="${MS_BRANCH:-master}"
API="https://modelscope.cn/openapi/v1/studios/${OWNER}/${STUDIO}"
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${TARGET_DIR}/artifacts/index_manifest.json"

NO_DEPLOY=0; VERIFY_ONLY=0; ALLOW_DIRTY=0; STRICT_GITHUB=0; SKIP_PREFLIGHT=0
BUILD_TIMEOUT="${BUILD_TIMEOUT:-1500}"     # 首窗口上限（秒）：须覆盖真构建最坏情形（调度6min+构建16min）；
                                           # 2026-10-27 23:37 实证：卡在 runner 排队的第 30 分钟由超时重触发解围
POLL_INTERVAL="${POLL_INTERVAL:-20}"
VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-300}"   # 等「启动期分块指纹行」的上限（秒）；数据层变大后启动变慢可调高
VERIFY_INTERVAL="${VERIFY_INTERVAL:-10}"  # 轮询运行日志的间隔（秒）
EXPECTED_FP=""                             # 来自 artifacts/index_manifest.json 的 chunk_fingerprint

for arg in "$@"; do
  case "${arg}" in
    --no-deploy) NO_DEPLOY=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --strict-github) STRICT_GITHUB=1 ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    *) echo "未知参数: ${arg}"; exit 2 ;;
  esac
done

cd "${TARGET_DIR}"

_pick_python() {  # 优先显式 PYTHON，其次仓库 .venv（本地依赖最全），最后系统 python3
  if [ -n "${PYTHON:-}" ]; then printf '%s' "${PYTHON}"; return 0; fi
  if [ -x "${TARGET_DIR}/.venv/bin/python" ]; then printf '%s' "${TARGET_DIR}/.venv/bin/python"; return 0; fi
  printf '%s' python3
}
PY_BIN="$(_pick_python)"

_ms_token() {
  ( security find-internet-password -s www.modelscope.cn -w 2>/dev/null \
      || echo "${MODELSCOPE_API_TOKEN:-}" ) | head -1
}
TOKEN="$(_ms_token)"
[ -n "${TOKEN}" ] || { echo "❌ 拿不到魔搭 token（keychain www.modelscope.cn 或环境变量 MODELSCOPE_API_TOKEN）"; exit 1; }

_api() { curl -sS -H "Authorization: Bearer ${TOKEN}" "$@"; }

_buildlog_lastts() {  # 构建日志最后一行的时间戳（yyyy-mm-dd hh:mm:ss，runner 本地 CST≈本机时间）；空=无 executionStep/无日志
  _api "${API}/logs/build?page=1&page_size=20" | python3 -c '
import json,sys,re
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
rows=(d.get("data") or {}).get("logs") or (d.get("Data") or [])
txt="".join(str(r) for r in rows)
lines=[l for l in txt.split("\n") if l.strip()]
for l in reversed(lines):
    m=re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", l)
    if m: print(m.group(1)); break' 2>/dev/null
}
_tsparse() { date -d "$1" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s 2>/dev/null || echo 0; }

_studio_state() { _api "${API}" | python3 -c '
import json,sys
d=json.load(sys.stdin).get("data") or {}
print(d.get("status") or "?")' 2>/dev/null || echo "?"; }

_run_logs() {  # 运行日志按行摊平输出（接口把超长行压成一条需 split；data/Data 双形状兜底同 _buildlog_lastts）
  # page_size=100 而非 800：平台对每条日志行做 ~2KB 截断，page_size 越大每行被切得越碎、
  # 甚至丢行（2026-10-31 实证：同一时刻 800 只还原 161 行文本、100 还原 210 行，且
  # "🧮 词表指纹"行在 800 下消失 —— 启动期自检明明已打）。验收只关心最近窗口，用小页。
  _api "${API}/logs/run?page=1&page_size=100" | python3 -c '
import json,sys
d=json.load(sys.stdin); rows=((d.get("data") or {}).get("logs")) or (d.get("Data") or [])
for r in rows:
    for line in str(r).splitlines(): print(line)' 2>/dev/null
}

_manifest_fp() {  # 读 artifacts/index_manifest.json 的 chunk_fingerprint；文件缺失=空串，JSON 坏了=非 0
  local f="${MANIFEST}"
  [ -f "${f}" ] || return 0
  "${PY_BIN}" -c '
import json,sys
try:
    with open(sys.argv[1],encoding="utf-8") as fh:
        d=json.load(fh)
except Exception:
    sys.exit(3)
if not isinstance(d,dict):
    sys.exit(4)
fp=d.get("chunk_fingerprint") or d.get("chunkFingerprint") or ""
print(str(fp).strip())' "${f}"
}

_load_expected_fp() {  # 硬闸门：发布/验收都要求 EXPECTED_FP 可取
  if ! EXPECTED_FP="$(_manifest_fp)"; then
    echo "❌ 解析 ${MANIFEST} 失败（JSON 损坏或形状不对）；修复或重跑 index 后重试"
    exit 1
  fi
  if [ -z "${EXPECTED_FP}" ]; then
    echo "❌ 读不到 ${MANIFEST} 的 chunk_fingerprint（文件缺失或键为空）"
    echo "   → 先 ./start.sh index（index 成功收尾会写该清单）；没有清单无法证明镜像数据来源，"
    echo "     验收也没有 EXPECTED_FP 可比，按 fail-closed 拒绝发布/验收。"
    exit 1
  fi
  echo "   ✅ EXPECTED_FP=${EXPECTED_FP}（来自 artifacts/index_manifest.json）"
}

_try_expected_fp() {  # --skip-preflight 时仍尽量取基准；取不到只警告，验收环节会 FAIL
  if ! EXPECTED_FP="$(_manifest_fp)"; then
    echo "⚠️ 解析 ${MANIFEST} 失败；EXPECTED_FP 置空（验收将因无法比对而 FAIL）"
    EXPECTED_FP=""
  elif [ -z "${EXPECTED_FP}" ]; then
    echo "⚠️ ${MANIFEST} 缺失或 chunk_fingerprint 为空；EXPECTED_FP 置空（验收将因无法比对而 FAIL）"
  else
    echo "   ✅ EXPECTED_FP=${EXPECTED_FP}（来自 artifacts/index_manifest.json）"
  fi
}

_preflight() {
  echo "── preflight：发布前防呆（① Qdrant 已停 ② verify-build ③ EXPECTED_FP ④ git 追踪完整性）──"

  # ① Qdrant 必须已停：先查再跑 verify-build——别一边被写一边被读。
  local pids
  pids="$({ pgrep -x qdrant 2>/dev/null || true; pgrep -f '[/]qdrant' 2>/dev/null || true; } | sort -u | tr '\n' ' ')"
  if [ -n "${pids// /}" ]; then
    echo "❌ preflight①: 本机 qdrant 仍在运行（pid: ${pids% }）"
    echo "   运行中的 Qdrant 会持续写 storage；.dockerignore 又排除 wal/open-*，镜像会拿到半写快照。"
    echo "   发布前先停干净：./start.sh stop（用 ./start.sh status 确认 Qdrant 已停）再重跑本脚本。"
    echo "   （2026-10-31 事故启示：不可信的数据不会自己变正确，只会在上线后被闸门惩罚。）"
    exit 1
  fi
  echo "   ✅ preflight①: 本机 Qdrant 未在运行（storage 是静置快照）"

  # ② verify-build：本地数据/清单的指纹一致性；与 Dockerfile 构建期同一命令。
  echo "── preflight②: ${PY_BIN} chatbot.py verify-build ──"
  local vb_rc=0 vb_out=""
  set +e
  vb_out="$("${PY_BIN}" chatbot.py verify-build 2>&1)"
  vb_rc=$?
  set -e
  printf '%s\n' "${vb_out}" | tail -n 20
  if [ "${vb_rc}" -ne 0 ]; then
    echo "❌ preflight②: verify-build 失败（rc=${vb_rc}）——本地数据与清单指纹不一致，禁止发布。"
    echo "   按 verify-build 提示重跑 ./start.sh process / ./start.sh index（index 成功收尾会重写清单），"
    echo "   再 ./start.sh verify-build 直到通过。"
    echo "   （若上面输出是命令不存在/依赖缺失而非指纹不一致：先同步代码与依赖；应急可 --skip-preflight，后果自担。）"
    exit 1
  fi
  echo "   ✅ preflight②: verify-build 通过（清单指纹 == 本地现算指纹）"

  # ③ EXPECTED_FP
  _load_expected_fp

  # ④ git 追踪完整性（git push 只送已跟踪文件）：
  #    a) 发布清单/同代 meta 必须已跟踪 —— 否则远端 Dockerfile COPY 直接缺文件；
  #    b) storage/、corpus/ 被 .gitignore 忽略 → **新增文件**普通 git add 收集不到，漏了
  #       镜像拿到的仍是旧数据（检索期闸门以指纹不一致拒服务，但那是构建之后了）；
  #    c) 已跟踪的数据文件被**就地改写**（commit 后又跑 sparse/index upsert）→ 磁盘状态与
  #       push 出去的内容不一致，本地 verify-build 仍绿，远端才炸。git status 抓这一类。
  local f
  for f in artifacts/index_manifest.json artifacts/chunks.meta.json; do
    if ! git ls-files --error-unmatch "${f}" >/dev/null 2>&1; then
      echo "❌ preflight④: ${f} 未入 git（push 不会带上它，Dockerfile COPY 会失败）"
      echo "   先 git add ${f} 并提交。"
      exit 1
    fi
  done
  local tracked_dirty
  tracked_dirty="$(git status --porcelain -- storage corpus lexicon qdrant_config 2>/dev/null | head -n 5 || true)"
  if [ -n "${tracked_dirty}" ]; then
    echo "❌ preflight④: 已跟踪的数据文件有未提交改动（磁盘 ≠ push 内容）"
    printf '%s\n' "${tracked_dirty}" | sed 's/^/   /'
    echo "   典型原因：commit 之后又跑了 sparse/index（原地 upsert）或 qdrant 还在写。"
    echo "   发布前：./start.sh stop → 重新 process/index（或 sparse）→ 提交数据改动再发。"
    exit 1
  fi
  local untracked_data
  untracked_data="$(git ls-files --others --ignored --exclude-standard -- storage corpus lexicon qdrant_config models/bge-m3/1_Pooling 2>/dev/null | grep -vE '^storage/[.]deleted/|/wal/' | head -n 5 || true)"
  if [ -n "${untracked_data}" ]; then
    echo "❌ preflight④: 可变数据里有未入 git 的新文件（被 .gitignore 忽略，普通 git add 收集不到）"
    printf '%s\n' "${untracked_data}" | sed 's/^/   /'
    echo "   这些文件不会随 git push 进构建上下文 → 镜像里的数据仍是旧快照（923/350 事故重演）。"
    echo "   发布前：git add -f storage corpus（先 ./start.sh stop，别把运行中 churn 一起提交）。"
    exit 1
  fi
  echo "   ✅ preflight④: 清单/同代 meta 已跟踪、数据无未提交改动、无遗漏新文件"
}

if [ "${VERIFY_ONLY}" != "1" ]; then
  BR="$(git branch --show-current)"
  [ "${BR}" = "main" ] || { echo "❌ 当前分支 ${BR} ≠ main —— 发布只从 main 发起"; exit 1; }

  if ! git diff --quiet --ignore-submodules -- ':(exclude)storage' || ! git diff --cached --quiet -- ':(exclude)storage'; then
    if [ "${ALLOW_DIRTY}" = "1" ]; then
      echo "⚠️ 工作区有未提交改动（--allow-dirty 放行）："; git status --short | head -12
    else
      echo "❌ 工作区有未提交改动（storage 运行时 churn 除外）。先提交，或 --allow-dirty 强发："
      git status --short -- ':(exclude)storage' | head -12; exit 1
    fi
  fi

  if [ "${SKIP_PREFLIGHT}" = "1" ]; then
    echo "⚠️ --skip-preflight：跳过 verify-build 与「Qdrant 已停」闸门（应急路径，后果自担）"
    _try_expected_fp
  else
    _preflight
  fi

  HEAD_SHA="$(git rev-parse --short HEAD)"
  MS_HEAD="$(git ls-remote modelscope "refs/heads/${BRANCH}" 2>/dev/null | awk '{print $1}')"
  if [ -n "${MS_HEAD}" ] && [ "${MS_HEAD}" = "$(git rev-parse HEAD)" ]; then
    echo "ℹ️ 魔搭 ${BRANCH} 已在 ${HEAD_SHA}，跳过 push"
  else
    echo "── push 魔搭 ${HEAD_SHA} → ${BRANCH} ──"
    git push modelscope HEAD:"refs/heads/${BRANCH}"
  fi

  echo "── push GitHub main（尽力，失败仅警告）──"
  if ! git push origin main; then
    echo "⚠️ GitHub 推送失败（部署不受影响——魔搭才是构建源）"
    [ "${STRICT_GITHUB}" = "1" ] && { echo "（--strict-github 视为失败）"; exit 1; }
  fi

  if [ "${NO_DEPLOY}" = "1" ]; then
    echo "✅ push 完成，--no-deploy 不触发构建"
    exit 0
  fi

  STATE="$(_studio_state)"
  if [ "${STATE}" = "Building" ]; then
    echo "⚠️ 上一场构建仍在 Building —— 新部署请求将排队/覆写，等待其翻车再发是更稳的选择（10-27 22:04 场已 40min+）"
  fi

  echo "── 触发构建（pipeline_type=build_image, branch=${BRANCH}）──"
  RESP="$(_api -X POST "${API}/deploy" -H 'Content-Type: application/json' -d "{\"branch\":\"${BRANCH}\",\"pipeline_type\":\"build_image\"}")"
  echo "${RESP}" | head -c 400; echo

  echo "── 轮询构建状态（每 ${POLL_INTERVAL}s，上限 ${BUILD_TIMEOUT}s）──"
  # ── 轮询（活量感知）：死线到点不盲目重触发——先看构建日志最后行年龄：还在推进就续期
  #    （2026-10-28 01:11→01:27 实证：烤镜像 4.9G 推仓库要 16min+，fixed 死线会误杀真进度），
  #    日志停滞（>阈值）才重触发；两次停滞直接退出归因。
  _deadline=$(( $(date +%s) + BUILD_TIMEOUT ))
  _retrigged=0; _live_extend=0
  while : ; do
    sleep "${POLL_INTERVAL}"
    STATE="$(_studio_state)"
    _t=$(( $(date +%s) ))
    echo "  [$(date +%H:%M:%S)] state=${STATE}"
    case "${STATE}" in
      Building) ;;
      *Fail*|*fail*|*Stop*|*Error*) echo "❌ 构建失败（state=${STATE}）"; exit 1 ;;
      Running)
        echo "✅ 构建+调度完成 → 进入运行日志验收"; break ;;
      *)
        echo "  （未知状态，继续观察）" ;;
    esac
    if [ "${STATE}" != "Building" ] || [ "${_t}" -le "${_deadline}" ]; then continue; fi
    _lastts="$(_buildlog_lastts)"
    if [ -n "${_lastts}" ]; then
      _age=$(( _t - $(_tsparse "${_lastts}") ))
    else
      _age=99999   # 无 executionStep/无日志 = runner 排队卡死（23:07/00:44 两场实证）
    fi
    if [ "${_age}" -le "${BUILD_LIVE_THRESHOLD:-420}" ] && [ "${_live_extend}" -lt "${BUILD_MAX_LIVE_EXTEND:-6}" ]; then
      _live_extend=$((_live_extend+1))
      echo "  🔎 构建日志仍在推进（最后行 ${_age}s 前，续期 ${_live_extend}/${BUILD_MAX_LIVE_EXTEND:-6}）——不重触发，宽限 10min"
      _deadline=$(( _t + 600 ))
    elif [ "${_retrigged}" = "0" ]; then
      _retrigged=1
      echo "⚠️ 构建停滞（最后行 ${_age}s 前）——自动重触发一次（runner 排队/卡死的最常见解法）"
      _api -X POST "${API}/deploy" -H 'Content-Type: application/json' \
        -d "{\"branch\":\"${BRANCH}\",\"pipeline_type\":\"build_image\"}" | head -c 200; echo
      _deadline=$(( _t + 1800 ))
    else
      echo "❌ 重触发后仍停滞（最后行 ${_age}s 前）—— 稍后 --verify-only 复核（不排除仍在收尾）；"
      echo "   或查魔搭控制台 /logs/build 人工归因"
      exit 1
    fi
  done
else
  # --verify-only：不 push；但仍需要 EXPECTED_FP 才能判定「索引是否就是仓库那次 index」
  _load_expected_fp
fi

# ─── 运行日志验收 ───────────────────────────────────────────────
echo "── 运行日志验收 ──"
FAIL=0
say() { echo "$*"; }

_wait_startup_fp() {  # 轮询运行日志直到出现启动期分块指纹行（或超时）
  local deadline=$(( $(date +%s) + VERIFY_TIMEOUT ))
  while : ; do
    LOGS="$(_run_logs)"
    if [ -n "${LOGS}" ] && printf '%s\n' "${LOGS}" | grep -q '🔧 分块指纹'; then return 0; fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then return 1; fi
    echo "  [$(date +%H:%M:%S)] 未见启动期分块指纹行，${VERIFY_INTERVAL}s 后再取（上限 ${VERIFY_TIMEOUT}s）"
    sleep "${VERIFY_INTERVAL}"
  done
}

if [ "${VERIFY_ONLY}" = "1" ]; then
  echo "  （--verify-only：直接读当前运行日志，不等待启动）"
  LOGS="$(_run_logs)"
elif _wait_startup_fp; then
  echo "  ✅ 已见启动期分块指纹行：$(printf '%s\n' "${LOGS}" | grep -m1 '🔧 分块指纹' | sed 's/^ *//')"
else
  echo "  ❌ 启动后 ${VERIFY_TIMEOUT}s 内未见 '🔧 分块指纹: 索引=… / 当前=…' 行"
  echo "     新版本 cmd_all 启动期必打这一行；未见 = 应用没走完启动（或日志接口没拉到）。"
  echo "     不再按旧契约「未见=ABSENT 放行」——那正是 2026-10-31 923/350 事故漏检的缺口。"
  FAIL=1
fi

if [ -z "${LOGS}" ]; then echo "❌ 运行日志为空（状态未翻 Running 或日志接口延迟）；稍后可用 --verify-only 复跑"; exit 1; fi

# 分块指纹判定：两侧相等 + 索引侧 == EXPECTED_FP。到点未见/不等都是 FAIL（无 ABSENT 放行）。
_fp_verdict="$(printf '%s\n' "${LOGS}" | "${PY_BIN}" -c '
import re,sys
expected=(sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
pat=re.compile(r"分块指纹: *索引=([0-9a-fA-F]+) */ *当前=([0-9a-fA-F]+)")
pairs=[]
for line in sys.stdin.read().splitlines():
    m=pat.search(line)
    if m: pairs.append((m.group(1).lower(), m.group(2).lower()))
if not pairs:
    print("ABSENT|"); raise SystemExit(0)
bad=[p for p in pairs if p[0] != p[1]]
idx=sorted({p[0] for p in pairs})
if bad:
    print("MISMATCH|"+" ".join(i+"/"+c for i,c in pairs)); raise SystemExit(0)
if not expected:
    print("NO_EXPECTED|"+" ".join(idx)); raise SystemExit(0)
if expected not in idx:
    print("EXPECTED_MISMATCH|"+" ".join(idx)); raise SystemExit(0)
print("PASS|"+" ".join(idx))' "${EXPECTED_FP}")"
case "${_fp_verdict}" in
  PASS*)
    say "  ✅ 分块指纹两侧一致且 == EXPECTED_FP（索引=${_fp_verdict#*|}）" ;;
  ABSENT*)
    say "  ❌ 运行日志里没有 '🔧 分块指纹: 索引=… / 当前=…' 行（启动行缺失或已被日志窗口滚掉）"; FAIL=1 ;;
  MISMATCH*)
    say "  ❌ 分块指纹不一致（索引/当前）：${_fp_verdict#*|}"
    say "     垃圾桶在吃过期分块——镜像里的 storage 与应用层数据不同源"; FAIL=1 ;;
  EXPECTED_MISMATCH*)
    say "  ❌ 索引侧指纹 ${_fp_verdict#*|} ≠ EXPECTED_FP=${EXPECTED_FP}"
    say "     镜像里的索引不是仓库清单那一次 index 的产物（storage 与 manifest 不同源）"; FAIL=1 ;;
  NO_EXPECTED*)
    say "  ❌ EXPECTED_FP 为空（--skip-preflight 且清单不可读）→ 无法证明索引与仓库数据同源"; FAIL=1 ;;
  *)
    say "  ❌ 分块指纹判定异常：${_fp_verdict}"; FAIL=1 ;;
esac

# 词表指纹启动行：两边相等 → PASS；不等 → FAIL（稀疏通道已错配，召回会静默下降）。
# 未见行不阻断：平台日志行会被 ~2KB 截断/丢行（page_size=100 已尽量补全），且稀疏侧
# 一致性仍由检索期词表闸门长期兜底；「分块指纹行」才是必见判据。
_lex_verdict="$(printf '%s\n' "${LOGS}" | "${PY_BIN}" -c '
import re,sys
pat=re.compile(r"词表指纹: *索引=([0-9a-fA-F]+) */ *当前=([0-9a-fA-F]+)")
pairs=[]
for line in sys.stdin.read().splitlines():
    m=pat.search(line)
    if m: pairs.append((m.group(1).lower(), m.group(2).lower()))
if not pairs:
    print("ABSENT|"); raise SystemExit(0)
bad=[p for p in pairs if p[0] != p[1]]
print(("FAIL|" if bad else "PASS|") + " ".join(i+"/"+c for i,c in pairs))')"
case "${_lex_verdict}" in
  PASS*)
    say "  ✅ 词表指纹两侧一致（${_lex_verdict#*|}）" ;;
  FAIL*)
    say "  ❌ 词表指纹不一致（索引/当前）：${_lex_verdict#*|}"
    say "     稀疏通道词空间已错配，检索会静默下降——应在同一份词表/Qdrant 版本下重跑 sparse 后重新发布" ; FAIL=1 ;;
  *)
    say "  ℹ️ 未见启动期 🧮 词表指纹行（平台日志截断/尚未滚动完，不阻断；稀疏侧仍由检索期闸门兜底）" ;;
esac

printf '%s\n' "${LOGS}" | grep -q "📦 \[模型装载\]" \
  && say "  ✅ 模型装载行: $(printf '%s\n' "${LOGS}" | grep -m1 '📦 \[模型装载\]' | sed 's/^ *//')" \
  || say "  ⚠️ 未见到模型装载行"
printf '%s\n' "${LOGS}" | grep -cE '\[模型落位\]|检索模型落位' >/dev/null \
  && say "  ✅ 有模型落位行 ×$(printf '%s\n' "${LOGS}" | grep -cE '\[模型落位\]|检索模型落位')" \
  || say " （无模型落位行）"

if printf '%s\n' "${LOGS}" | grep -q "RuntimeError"; then
  say "  ❌ 运行日志出现 RuntimeError:"; printf '%s\n' "${LOGS}" | grep -n "RuntimeError" | head -3; FAIL=1
else say "  ✅ 无 RuntimeError"; fi
# 启动编排失败（❌ [all] 是 cmd_all 退出容器交平台重启的那类行）必须算失败——不然 app
# 没起来、日志干净没错误的"空跑"会被误判成验收 PASS。
if printf '%s\n' "${LOGS}" | grep -q '❌ \[all\]'; then
  say "  ❌ 运行日志出现 ❌ [all]（启动编排失败）:"; printf '%s\n' "${LOGS}" | grep '❌ \[all\]' | head -3; FAIL=1
else say "  ✅ 无 ❌ [all]"; fi
printf '%s\n' "${LOGS}" | grep -q '\[all\] infinity 就绪' \
  && say "  ✅ infinity: $(printf '%s\n' "${LOGS}" | grep -m1 '\[all\] infinity 就绪' | sed 's/^ *//')" \
  || say "  ⚠️ 未见到 infinity 就绪行（torch/http 后端不编排或尚未就绪）"
if printf '%s\n' "${LOGS}" | grep -q "❌"; then
  say "  ⚠️ 有 ❌ 行（人工过目是否有致命项）:"; printf '%s\n' "${LOGS}" | grep "❌" | tail -5
fi

if [ "${FAIL}" = "0" ]; then echo "🎯 验收 PASS"; else echo "🛑 验收 FAIL"; exit 1; fi
