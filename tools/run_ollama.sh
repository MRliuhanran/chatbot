#!/usr/bin/env bash
# 本地（纯原生，不用 Docker）拉起 Ollama 服务端的**唯一机制**，与容器逐字同源：
#
#     .env  →  tools/dotenv.sh 的 _load_dotenv  →  本进程环境  →  exec ollama serve
#
# 容器侧对应物是 entrypoint.sh：它 source 同一个 tools/dotenv.sh，且在 _ollama_start
# 之前完成导出。两边因此**同源、同导出、同启动**，不存在第二套注入路径。
#
# 为什么需要它：OLLAMA_KV_CACHE_TYPE / OLLAMA_FLASH_ATTENTION / OLLAMA_NUM_PARALLEL /
# OLLAMA_NUM_THREADS 是 **Ollama 服务端**变量 —— 客户端 .env 对**已经在跑**的服务毫无
# 作用（本仓 .env 与 README 都写过这一点）。此前容器靠 Dockerfile ENV 注入、本地靠
# launchctl setenv，效果相同而来源不同，改一处必漏另一处。现在两边都从 .env 读。
#
# Ollama.app（GUI）无法继承 shell 环境 —— 这是 macOS 的进程模型，不是本项目的选择。
# 继续用 GUI 的人走 --apply-to-launchd，**然后重启 Ollama.app** 才生效。
#
# 用法：
#   tools/run_ollama.sh                      # 前台起 ollama serve（Ctrl-C 结束）
#   tools/run_ollama.sh --show               # 只打印 .env 解析后的生效值，不起服务
#   tools/run_ollama.sh --apply-to-launchd   # 把值写进 launchd gui 域（给 Ollama.app 用）
#   tools/run_ollama.sh --help
#
# 幂等：--apply-to-launchd 重复执行只是把同样的值再写一次；--show 无副作用。
# 不改变 Ollama 的其它行为：除这四项外不导出、不设置任何东西，尤其**不设 OLLAMA_MODELS**
# （本地仍用 ~/.ollama/models，与容器里 /app/ollama_models 的差异是**目录布局**，不是参数）。
# 兼容 bash 3.2（macOS /bin/bash）：不用关联数组 / mapfile / ${var^^}。
set -u

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "${_ROOT}" || exit 1

if [ ! -f tools/dotenv.sh ]; then
    echo "❌ 缺少 tools/dotenv.sh（.env 解析的唯一实现）—— 拒绝在"读不到 .env"的状态下起服务" >&2
    exit 1
fi
# shellcheck source=tools/dotenv.sh
. tools/dotenv.sh
_load_dotenv .env

# .env 里定义的**服务端**四项。此清单与 .env 中那段注释、Dockerfile 末尾的"容器专有"段
# 一一对应；增删任何一项都要三处同步（这是本仓库"同一知识只留一处"的已知例外：
# 变量名清单无法从 .env 自动发现，只能显式列出）。
_KEYS="OLLAMA_KV_CACHE_TYPE OLLAMA_FLASH_ATTENTION OLLAMA_NUM_PARALLEL OLLAMA_NUM_THREADS"

# 取值不用 eval / 间接展开：bash 3.2 的 ${!name} 在 set -u 下有坑，且显式 case 更好审。
_val_of() {
    case "$1" in
        OLLAMA_KV_CACHE_TYPE)   printf '%s' "${OLLAMA_KV_CACHE_TYPE:-}" ;;
        OLLAMA_FLASH_ATTENTION) printf '%s' "${OLLAMA_FLASH_ATTENTION:-}" ;;
        OLLAMA_NUM_PARALLEL)    printf '%s' "${OLLAMA_NUM_PARALLEL:-}" ;;
        OLLAMA_NUM_THREADS)     printf '%s' "${OLLAMA_NUM_THREADS:-}" ;;
        *) printf '%s' "" ;;
    esac
}

_usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'
}

_show() {
    printf '%-26s %-10s %s\n' "变量" "值" "来源"
    for _k in ${_KEYS}; do
        _v="$(_val_of "${_k}")"
        if [ -n "${_v}" ]; then
            printf '%-26s %-10s %s\n' "${_k}" "${_v}" ".env（或调用时已在进程环境里，进程环境优先）"
        else
            printf '%-26s %-10s %s\n' "${_k}" "<空>" "未设 —— 见下方告警"
        fi
    done
}

# Ollama.app 用：把 .env 的值写进 launchd gui 域。只写不改（不 unset 任何键）。
_apply_to_launchd() {
    _rc=0
    for _k in ${_KEYS}; do
        _v="$(_val_of "${_k}")"
        if [ -z "${_v}" ]; then
            echo "⚠️  ${_k} 在 .env 里没有值 → 跳过（launchd 保持原样，不做破坏性动作）" >&2
            continue
        fi
        if launchctl setenv "${_k}" "${_v}"; then
            echo "launchctl setenv ${_k} ${_v}"
        else
            echo "❌ launchctl setenv ${_k} 失败（当前域不允许？）" >&2
            _rc=1
        fi
    done
    if [ "${_rc}" = 0 ]; then
        echo
        echo "已写入 launchd gui 域。⚠️ 必须**重启 Ollama.app** 才生效（GUI 进程只在启动时读一次）。"
        echo "核查：launchctl print gui/$(id -u) | grep -i OLLAMA_"
    fi
    return "${_rc}"
}

case "${1:-}" in
    --help|-h)
        _usage
        exit 0
        ;;
    --show)
        _show
        exit 0
        ;;
    --apply-to-launchd)
        _apply_to_launchd
        exit $?
        ;;
    ""|--serve)
        ;;
    *)
        echo "未知参数: $1" >&2
        _usage
        exit 2
        ;;
esac

# ── fail-closed：与 entrypoint.sh 逐条同构，宁可起不来也不静默退回 f16 KV ──────────
# KV 退回 f16 时 65536 窗口的体积翻倍；本机 Metal 工作集上限 5.33GB（.env 里记过实测），
# 后果是整机换页卡死而不是报错。所以这里与容器一样直接拒绝启动。
if [ -z "${OLLAMA_KV_CACHE_TYPE:-}" ]; then
    echo "❌ OLLAMA_KV_CACHE_TYPE 为空：KV 会退回 f16，65536 窗口下体积翻倍，本机必被拖死。" >&2
    echo "   唯一来源是 .env（仓库根目录）：请确认它存在且含 OLLAMA_KV_CACHE_TYPE=q8_0。" >&2
    exit 1
fi
[ -n "${OLLAMA_FLASH_ATTENTION:-}" ] || echo "⚠️  OLLAMA_FLASH_ATTENTION 为空：KV 量化可能不生效（.env 未生效？）" >&2
[ -n "${OLLAMA_NUM_PARALLEL:-}" ]    || echo "⚠️  OLLAMA_NUM_PARALLEL 为空：退回多槽位，65536 窗口按并发数翻倍吃内存（.env 未生效？）" >&2
[ -n "${OLLAMA_NUM_THREADS:-}" ]     || echo "⚠️  OLLAMA_NUM_THREADS 为空：按 vCPU 全开线程（.env 未生效？）" >&2

if ! command -v ollama >/dev/null 2>&1; then
    echo "❌ PATH 里没有 ollama 可执行文件。" >&2
    exit 1
fi

echo "[run_ollama] 服务端参数（来源 .env，与容器同一解析器）：KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE} FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION:-<未设>} NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-<未设>} NUM_THREADS=${OLLAMA_NUM_THREADS:-<未设>}" >&2
echo "[run_ollama] 若你用的是 Ollama.app（GUI 不继承 shell 环境）：先 tools/run_ollama.sh --apply-to-launchd 再重启它。" >&2

exec ollama serve
