#!/usr/bin/env bash
# 硬件适配参数的**薄封装**：机器事实的唯一实现是 tools/hw_profile.py（Python，可 import）。
# 本文件只做三件事：eval 它的 --sh 输出、打印 [hw] 审计行、核对契约确实**导出**了。
#
# 为什么收敛成 Python：此前 shell 与离线侧（chatbot.py 的 cgroup 自探测）各有一份探测，
# 两者算出的值一致，但"这台机器有几个核"成了**两处实现** —— 本仓库为"同一知识多处实现"
# 已踩过两次坑。现在探测与派生只在 hw_profile.py 里，本文件与离线侧都读它的结论。
# （shell 侧刻意保留：eval 必须在**当前 shell** 生效才能把变量传给 ollama/torch 子进程。）
#
# 用法（**必须先进程加载 .env，再调用本函数** —— 显式覆盖优先，与 dotenv.sh 同一语义）：
#     . tools/hw_profile.sh && _load_hw_profile
#
# 接口（**已冻结**，详见 hw_profile.py 头注释）：
#   _load_hw_profile  幂等；导出 HW_CORES / HW_MEM_MB / HW_THREADS
#                     与 OMP/MKL/OPENBLAS/OLLAMA_NUM_THREADS（glibc 命中时另有 MALLOC_ARENA_MAX），
#                     并打印 [hw] 审计行（取值 + 探测来源 + 导出契约）。
#   辅助（下划线开头 = 非契约）：_HW_GLIBC / _HW_SRC_CORES / _HW_SRC_MEM
#
# 已接入的入口：entrypoint.sh（容器）、tools/run_local.sh、tools/run_ollama.sh。
# 它们的**子进程**（ollama serve / streamlit / torch）全部继承，无需再做别的。
# ⚠️ 裸跑 python chatbot.py <process|index|sparse|…> **不会**自动带上契约 —— 那条路不经过
#    上面三个脚本。要么先经 tools/run_local.sh 起服务，要么在使用前补上上面那行 source+调用。
# ⚠️ 读不到 HW_CORES 时**不要**在调用方再探测一遍兜底（那正是两处实现的分叉点）：
#    **读不到 = 没加载，补上那一行即可。** 本函数末尾有一道契约自检，漏 export 会直接喊出来。
#
# 兼容 bash 3.2（macOS /bin/bash）：不用关联数组 / mapfile / 大小写转换展开。
# source 时**零副作用**：路径懒求值，探测与导出都在 _load_hw_profile 里。
# 故意不设 set -u：sourced 库改调用方的 shell 选项是意外副作用；展开处都做了空值防护。

# 本文件所在目录（懒求值：source 时不产生副作用）
_HW_SH_DIR=""
# 幂等标记：**必须在这里初始化**。调用方（entrypoint.sh、run_ollama.sh）都开了 set -u，
# 少了这一行，_load_hw_profile 里的 "${_HW_LOADED}" 会在**第一次**调用时就报
# "unbound variable" 直接退出 —— 契约根本加载不上（2026-10-16 由 set -u 用例抓出）。
_HW_LOADED=""

_load_hw_profile() {
    [ "${_HW_LOADED}" = "1" ] && return 0

    if [ -z "${_HW_SH_DIR}" ]; then
        _HW_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
    fi
    local _hw_py="${_HW_SH_DIR}/hw_profile.py"
    if [ ! -f "${_hw_py}" ]; then
        printf '[hw] ❌ 缺少 %s（机器事实的唯一实现）—— 线程数/arena 会静默偏离，拒绝继续\n' "${_hw_py}" >&2
        return 1
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        printf '[hw] ❌ 需要 python3 才能加载硬件契约（实现是 %s）\n' "${_hw_py}" >&2
        return 1
    fi

    # --sh 的 stdout **只有赋值语句**（诊断走它的 stderr），故可直接 eval；
    # eval 在当前 shell 生效 —— 命令替换的子 shell 不会吃掉导出。
    local _hw_out
    _hw_out="$(python3 "${_hw_py}" --sh)" || {
        printf '[hw] ❌ 加载失败：python3 %s --sh 返回非 0\n' "${_hw_py}" >&2
        return 1
    }
    eval "${_hw_out}"
    _HW_LOADED=1

    # ── 契约自检：HW_* 必须**真的在导出环境里** ──────────────────────────────
    # 漏掉 export 时本 shell 读得到、**子进程读不到**，下游于是自己再探测一遍（两处实现），
    # 而且不报任何错。export -p 只列**已导出**的变量，未导出的同名 shell 变量不在其中。
    # 覆盖范围：只挡"忘了 export"；调用方若把加载放进子 shell / 管道，导出会随子 shell 消失，
    # 这里看不见 —— 所以加载语句必须在脚本顶层，不要包在 $( ) 里。
    local _hw_exports _hw_unexported _k
    _hw_exports="$(export -p 2>/dev/null)"
    _hw_unexported=""
    for _k in HW_CORES HW_MEM_MB HW_THREADS; do
        case "${_hw_exports}" in
            *" ${_k}="*) : ;;
            *) _hw_unexported="${_hw_unexported} ${_k}" ;;
        esac
    done
    [ -z "${_hw_unexported}" ] || printf '[hw] ⚠️  契约变量未导出:%s —— 子进程会读到空值并自行探测（两处实现），请检查 hw_profile.py 的 --sh 输出\n' "${_hw_unexported}"

    # ── 审计行（取值 + 探测来源 + 导出契约；必须能一眼看出"为什么是 2 而不是 8"）──
    # 打印的是**生效值**（可能来自显式覆盖）；被覆盖时带 (显式) 标记。
    printf '[hw] cores=%s mem=%sMB glibc=%s threads=%s%s arena=%s%s 来源：%s / %s\n' \
        "${HW_CORES}" "${HW_MEM_MB}" "${_HW_GLIBC}" "${OLLAMA_NUM_THREADS:-?}" "${_HW_THR_MARK}" \
        "${_HW_ARENA_SHOW}" "${_HW_ARENA_MARK}" "${_HW_SRC_CORES}" "${_HW_SRC_MEM}"
    printf '[hw] 导出契约：HW_CORES=%s HW_MEM_MB=%s HW_THREADS=%s（OMP/MKL/OPENBLAS/OLLAMA_NUM_THREADS 同值）\n' \
        "${HW_CORES}" "${HW_MEM_MB}" "${HW_THREADS}"
    [ "${_HW_CORES_RAW:-${HW_CORES}}" != "${HW_CORES}" ] && \
        printf '[hw] 熔断前原始核数=%s（仅记录；对外契约一律用 HW_CORES=%s）\n' "${_HW_CORES_RAW}" "${HW_CORES}"
    [ -n "${_HW_WARN}" ] && printf '[hw] ⚠️  %s\n' "${_HW_WARN}"
    [ -n "${_HW_MARK}" ] && printf '[hw] 注：%s\n' "${_HW_MARK}"
    return 0
}
