#!/usr/bin/env bash
# ── 一键发布：push（魔搭必需 + GitHub 尽力）→ 触发构建 → 轮询状态 → 运行日志验收 ──
# 用法:
#   bash tools/ms_release.sh                 # 完整发布（push+deploy+轮询+日志验收）
#   bash tools/ms_release.sh --verify-only   # 不 push 不 deploy，仅验收当前运行日志
#   bash tools/ms_release.sh --no-deploy     # 只 push，不触发构建
#   bash tools/ms_release.sh --allow-dirty   # 允许带未提交改动发布（默认拒绝）
#   bash tools/ms_release.sh --strict-github # GitHub 推送失败也视为失败（默认警告放行）
# 验收判据（与部署事故复盘同源的运行日志关键行）:
#   1) 🔧 分块指纹: 索引=… / 当前=…   —— 两侧一致才 PASS；**未见行=warn 不阻断**
#      （指纹行要在用户实际发起一次检索后才出现，刚 Running 时没有是正常的）
#   2) 模型装载/落位行（📦 [模型装载] 全部就绪（零下载命中） 或 [模型落位] ×≥2）
#   3) 运行日志无 ❌ / RuntimeError / 无新增异常 traceback
set -euo pipefail

OWNER="${MS_OWNER:-liuhanran}"
STUDIO="${MS_STUDIO:-chatbot}"
BRANCH="${MS_BRANCH:-master}"
API="https://modelscope.cn/openapi/v1/studios/${OWNER}/${STUDIO}"
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NO_DEPLOY=0; VERIFY_ONLY=0; ALLOW_DIRTY=0; STRICT_GITHUB=0
BUILD_TIMEOUT="${BUILD_TIMEOUT:-1500}"     # 首窗口上限（秒）：须覆盖真构建最坏情形（调度6min+构建16min）；
                                           # 2026-10-27 23:37 实证：卡在 runner 排队的第 30 分钟由超时重触发解围
POLL_INTERVAL="${POLL_INTERVAL:-20}"

for arg in "$@"; do
  case "${arg}" in
    --no-deploy) NO_DEPLOY=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --strict-github) STRICT_GITHUB=1 ;;
    *) echo "未知参数: ${arg}"; exit 2 ;;
  esac
done

_ms_token() {
  ( security find-internet-password -s www.modelscope.cn -w 2>/dev/null \
      || echo "${MODELSCOPE_API_TOKEN:-}" ) | head -1
}
TOKEN="$(_ms_token)"
[ -n "${TOKEN}" ] || { echo "❌ 拿不到魔搭 token（keychain www.modelscope.cn 或环境变量 MODELSCOPE_API_TOKEN）"; exit 1; }

_api() { curl -sS -H "Authorization: Bearer ${TOKEN}" "$@"; }

_studio_state() { _api "${API}" | python3 -c '
import json,sys
d=json.load(sys.stdin).get("data") or {}
print(d.get("status") or "?")' 2>/dev/null || echo "?"; }

_run_logs() {  # 运行日志按行摊平输出（接口把超长行压成一条需 split）
  _api "${API}/logs/run?page=1&page_size=800" | python3 -c '
import json,sys
d=json.load(sys.stdin); rows=d.get("Data") or []
for r in rows:
    for line in str(r).splitlines(): print(line)' 2>/dev/null
}

if [ "${VERIFY_ONLY}" != "1" ]; then
  cd "${TARGET_DIR}"

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
  _deadline=$(( $(date +%s) + BUILD_TIMEOUT ))
  _retrigged=0
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
    if [ "${_t}" -gt "${_deadline}" ]; then
      if [ "${_retrigged}" = "0" ]; then
        _retrigged=1
        echo "⚠️ 构建超时（>${BUILD_TIMEOUT}s）——自动重触发一次（平台调度/卡死最常见解法）"
        _api -X POST "${API}/deploy" -H 'Content-Type: application/json' \
          -d "{\"branch\":\"${BRANCH}\",\"pipeline_type\":\"build_image\"}" | head -c 200; echo
        _deadline=$(( $(date +%s) + 1800 ))   # 二次窗口更长：重触发的新构建要走完整 build+push+调度
      else
        echo "❌ 重触发后仍超时 —— 不排除仍在收尾：稍后 --verify-only 复核；"
        echo "   或查 /logs/build 接口看 executionStep 是否在推进（23:37 场实证：HK runner 排队卡住时新起一场 14min 就建完）"
        exit 1
      fi
    fi
  done
  sleep 12   # 等运行日志刚起步的首屏落库
fi

# ─── 运行日志验收 ───────────────────────────────────────────────
echo "── 运行日志验收 ──"
LOGS="$(_run_logs)"
if [ -z "${LOGS}" ]; then echo "⚠️ 运行日志为空（状态未翻 Running 或日志接口延迟）；稍后可用 --verify-only 复跑"; exit 1; fi

FAIL=0
say() { echo "$*"; }

_fp_ok="$(printf '%s\n' "${LOGS}" | python3 -c '
import sys,re
ok=seen=0
for line in sys.stdin:
    m=re.search(r"分块指纹: *索引=([0-9a-f]+) */ *当前=([0-9a-f]+)", line)
    if m:
        seen+=1
        if m.group(1)==m.group(2): ok+=1
print("PASS" if seen and ok==seen else ("FAIL" if seen else "ABSENT"))')"
case "${_fp_ok}" in
  PASS) say "  ✅ 分块指纹两侧一致" ;;
  FAIL) say "  ❌ 分块指纹不一致（垃圾桶在吃过期分块）"; FAIL=1 ;;
  ABSENT) say "  ⚠️ 未见到分块指纹行（尚未有检索触发指纹闸门，属正常；发一次检索后 --verify-only 复核）——不阻断" ;;
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
if printf '%s\n' "${LOGS}" | grep -q "❌"; then
  say "  ⚠️ 有 ❌ 行（人工过目是否有致命项）:"; printf '%s\n' "${LOGS}" | grep "❌" | tail -5
fi

if [ "${FAIL}" = "0" ]; then echo "🎯 验收 PASS"; else echo "🛑 验收 FAIL"; exit 1; fi
