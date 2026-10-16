#!/usr/bin/env bash
# 硬件适配参数的**唯一实现**：把"这台机器有多少核 / 多少内存"翻译成线程数与内存旋钮。
# 本地原生路径（tools/run_local.sh、tools/run_ollama.sh）与容器 entrypoint.sh **source 同一份**。
#
# ┌─ 接口（**已冻结**，2026-10-16）─────────────────────────────────────────────┐
# │ _load_hw_profile  加载探测 + 派生 + export（幂等；调用方先 _load_dotenv 再调它）│
# │   HW_CORES    实际可用核数（容器= cgroup 配额，非宿主核数）                    │
# │   HW_MEM_MB   实际可用内存 MB（容器= cgroup 限额）                            │
# │   HW_THREADS  线程数 = HW_CORES（**不封顶**，只受 256 安全熔断）              │
# │ 下游派生（同一条规则的 export，调用方不必自己算）：                            │
# │   OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS / OLLAMA_NUM_THREADS │
# │   MALLOC_ARENA_MAX（仅 glibc 且内存条件命中时才设）                            │
# │ 辅助（下划线开头 = 非契约，仅供日志/排查）：                                    │
# │   _HW_GLIBC   1=Linux(glibc) 0=其它                                          │
# │   _HW_SRC_CORES / _HW_SRC_MEM   探测来源字符串                                │
# └──────────────────────────────────────────────────────────────────────────┘
# 其它 agent 若要按核数并行（如离线 process/index），**读 HW_CORES / HW_MEM_MB 即可**，
# 不要另写一套探测：容器里 nproc 报的是宿主核数，自己探测必然与这里分叉。
#
# 为什么派生而不是写死：同一份 .env 要同时服务多核 Mac 与 2vCPU 容器。写死必然有一边是错的
# —— 写 2 则本地白丢大半算力（本地因为容器化而变慢，违反"只是把本地容器化"的根本约束）；
# 写 8 则 2 vCPU 容器上开 8 个线程，既慢又按线程吃内存。**探测到多少核就用多少核**，
# 两边各自吃满自己的配额。
#
# ★ 容器感知是重点（本模块最容易出错、后果最严重的一处）★
# 容器里 nproc / /proc/meminfo 报的是**宿主的**核数与内存（FC 实例上可能是几十核、上百 G）。
# 只看它们会让 OpenMP/BLAS 与 llama.cpp 都按宿主规模去抢那两个 vCPU：上下文切换把吞吐打下去，
# 每线程的栈与 glibc arena 又把内存吃掉。故核数与内存都**优先读 cgroup 的 quota/limit**
# （那才是容器真实的配额），nproc/meminfo/sysctl 只是没有 cgroup 时的回退。
#
# 派生规则（是**策略**，不是值；改策略只改这一处）：
#   CORES  = ceil(quota/period)   cgroup v2 /sys/fs/cgroup/cpu.max（"max" = 不限制）
#                                 cgroup v1 cpu/cpu.cfs_quota_us + cpu.cfs_period_us（-1 = 不限制）
#            → 回退 nproc（Linux）/ sysctl hw.ncpu→hw.physicalcpu→hw.perflevel0.physicalcpu（macOS）
#            → 安全熔断：> 256 视为探测异常，压到 256 并告警（防"按宿主 1000 核开线程"）
#   MEM_MB = cgroup v2 memory.max / v1 memory/memory.limit_in_bytes（"max"/天文数字 = 不限制）
#            → 回退 /proc/meminfo MemTotal / sysctl hw.memsize
#   THREADS = CORES                                    # **不封顶**：探测到多少就用多少
#   OMP/MKL/OPENBLAS_NUM_THREADS = THREADS             # torch 的 OpenMP/BLAS
#   OLLAMA_NUM_THREADS           = THREADS             # llama.cpp 生成线程
#   MALLOC_ARENA_MAX = 2                     glibc 且 MEM_MB <= 8192（内存紧张，压 per-thread arena 碎片）
#                    = min(8, max(2, CORES)) glibc 且内存宽裕
#                    = 不设                    非 glibc（macOS 的 malloc 没有 arena 概念）
#                      ↑ 它与核数**不是同一维度**：arena 数只与"内存紧不紧 + 是不是 glibc"有关，
#                        故保留取值上限；线程数不设上限。
#   探测失败 → **安全回退**：THREADS = min(能拿到的核数, 4)，内存按"紧张"一支处理，并告警
#
# 显式覆盖优先（与 tools/dotenv.sh 的"进程环境优先"同一语义）：
#   调用方必须**先** _load_dotenv、**再** _load_hw_profile。这样 .env 或进程环境里显式写了的
#   键会被尊重，派生只作为默认值。反过来（先派生后 dotenv）会让 .env 永远失效 —— 正是本仓库
#   点名过的静默原型"配置写了却不生效"。
#
# 审计：每次加载打印一行 [hw]，列出**实际生效值**与探测来源。启动日志里必须能一眼看出
#   "为什么是 2 而不是 8" —— 否则下次有人看到线程数变小只会以为坏了。
#
# 测试钩子：环境变量 _HW_CGROUP_ROOT 可把 cgroup 根指到夹具目录（默认 /sys/fs/cgroup），
#   便于在 macOS 上验证容器分支：造 根/cpu.max = "200000 100000" 与
#   根/memory.max = 8589934592，应算出 HW_CORES=2 HW_THREADS=2 arena=2。
#
# 兼容 bash 3.2（macOS /bin/bash）：不用关联数组 / mapfile / 大小写转换展开。
# 本文件只定义函数与常量，source 时**不产生任何副作用**（探测与导出都在 _load_hw_profile 里）。
# 故意不设 set -u：sourced 库改调用方的 shell 选项是意外副作用；但下面所有展开都做了空值防护。

# cgroup 根目录：测试钩子可覆盖。末尾不留斜杠。
_HW_CGROUP_ROOT="${_HW_CGROUP_ROOT:-/sys/fs/cgroup}"
# 本次进程是否已加载（幂等）
_HW_LOADED=""
# 安全熔断：探测值大于它按"探测异常"处理（不是性能封顶 —— 正常机器永远碰不到）
_HW_MAX_CORES=256

# _hw_cores_cgroup：容器真实 CPU 配额 → 核数（向上取整）。拿不到返回非 0。
_hw_cores_cgroup() {
    local _max="" _period="" _q=""
    # cgroup v2：cpu.max = "<quota|max> <period>"
    if [ -r "${_HW_CGROUP_ROOT}/cpu.max" ]; then
        read -r _max _period < "${_HW_CGROUP_ROOT}/cpu.max" || true
        case "${_max}" in
            ""|max) : ;;
            *)
                if [ "${_period:-0}" -gt 0 ] 2>/dev/null; then
                    printf '%s' "$(( (_max + _period - 1) / _period ))"
                    return 0
                fi
                ;;
        esac
    fi
    # cgroup v1
    if [ -r "${_HW_CGROUP_ROOT}/cpu/cpu.cfs_quota_us" ] && [ -r "${_HW_CGROUP_ROOT}/cpu/cpu.cfs_period_us" ]; then
        _q="$(cat "${_HW_CGROUP_ROOT}/cpu/cpu.cfs_quota_us" 2>/dev/null)"
        _period="$(cat "${_HW_CGROUP_ROOT}/cpu/cpu.cfs_period_us" 2>/dev/null)"
        if [ "${_q:-0}" -gt 0 ] 2>/dev/null && [ "${_period:-0}" -gt 0 ] 2>/dev/null; then
            printf '%s' "$(( (_q + _period - 1) / _period ))"
            return 0
        fi
    fi
    return 1
}

# _hw_mem_cgroup：容器真实内存上限 → MB。拿不到 / 无限制返回非 0。
_hw_mem_cgroup() {
    local _v=""
    if [ -r "${_HW_CGROUP_ROOT}/memory.max" ]; then
        _v="$(cat "${_HW_CGROUP_ROOT}/memory.max" 2>/dev/null)"
        case "${_v}" in
            ""|max) : ;;
            *)
                # 天文数字 = 未限制（有的运行时写成 2^63-1 一类）
                if [ "${_v}" -gt 0 ] 2>/dev/null && [ "${_v}" -lt 9000000000000000000 ] 2>/dev/null; then
                    printf '%s' "$(( _v / 1048576 ))"
                    return 0
                fi
                ;;
        esac
    fi
    if [ -r "${_HW_CGROUP_ROOT}/memory/memory.limit_in_bytes" ]; then
        _v="$(cat "${_HW_CGROUP_ROOT}/memory/memory.limit_in_bytes" 2>/dev/null)"
        if [ "${_v:-0}" -gt 0 ] 2>/dev/null && [ "${_v}" -lt 9000000000000000000 ] 2>/dev/null; then
            printf '%s' "$(( _v / 1048576 ))"
            return 0
        fi
    fi
    return 1
}

# _hw_cores_system：非容器回退。Linux 用 nproc；macOS 依次试 hw.ncpu / hw.physicalcpu /
# hw.perflevel0.physicalcpu（**探测到多少用多少**，不再按 8 封顶）。hw.ncpu 优先是有意的：
# 改造前本地 ollama 未设 OLLAMA_NUM_THREADS，Ollama 默认取逻辑核数（本机 8）；若改用
# perflevel0（P 核 = 4）会让本地因为这次改造而变慢。要按 P 核跑请显式覆盖：
#   OLLAMA_NUM_THREADS=4 tools/run_ollama.sh
_hw_cores_system() {
    local _v="" _k=""
    if command -v nproc >/dev/null 2>&1; then
        _v="$(nproc 2>/dev/null)"
        if [ "${_v:-0}" -gt 0 ] 2>/dev/null; then printf '%s' "${_v}"; return 0; fi
    fi
    if command -v sysctl >/dev/null 2>&1; then
        for _k in hw.ncpu hw.physicalcpu hw.perflevel0.physicalcpu; do
            _v="$(sysctl -n "${_k}" 2>/dev/null)"
            if [ "${_v:-0}" -gt 0 ] 2>/dev/null; then printf '%s' "${_v}"; return 0; fi
        done
    fi
    return 1
}

# _hw_mem_system：非容器回退。/proc/meminfo MemTotal(kB) → MB；macOS 用 hw.memsize(bytes)。
_hw_mem_system() {
    local _v=""
    if [ -r /proc/meminfo ]; then
        _v="$(awk '/^MemTotal:/{printf "%d", $2/1024; exit}' /proc/meminfo 2>/dev/null)"
        if [ "${_v:-0}" -gt 0 ] 2>/dev/null; then printf '%s' "${_v}"; return 0; fi
    fi
    if command -v sysctl >/dev/null 2>&1; then
        _v="$(sysctl -n hw.memsize 2>/dev/null)"
        if [ "${_v:-0}" -gt 0 ] 2>/dev/null; then printf '%s' "$(( _v / 1048576 ))"; return 0; fi
    fi
    return 1
}

# _load_hw_profile：探测 + 派生 + 导出 + 打印审计行。幂等（同一进程重复调用只生效一次）。
_load_hw_profile() {
    [ "${_HW_LOADED}" = "1" ] && return 0

    local _cores="" _mem="" _warn="" _HW_CORES_RAW=""
    local _threads="" _arena="" _arena_show="" _glibc=0 _v=""
    local _mem_known=1 _mem_tight=0 _mark="" _arena_mark="" _thr_mark=""

    # ── 1. 探测（cgroup 优先）────────────────────────────────────────────────
    if _v="$(_hw_cores_cgroup)"; then
        _cores="${_v}"
        if [ -r "${_HW_CGROUP_ROOT}/cpu.max" ]; then
            _HW_SRC_CORES="cgroup v2 cpu.max"
        else
            _HW_SRC_CORES="cgroup v1 cpu.cfs_quota_us"
        fi
        case "${_HW_CGROUP_ROOT}" in
            /sys/fs/cgroup) : ;;
            *) _HW_SRC_CORES="${_HW_SRC_CORES}（夹具 ${_HW_CGROUP_ROOT}）" ;;
        esac
    elif _v="$(_hw_cores_system)"; then
        _cores="${_v}"
        if command -v nproc >/dev/null 2>&1 && [ "$(nproc 2>/dev/null)" = "${_v}" ]; then
            _HW_SRC_CORES="nproc"
        else
            _HW_SRC_CORES="sysctl hw.*"
        fi
    else
        _cores=""
        _warn="${_warn}核数探测失败（按 4 线程安全回退）；"
        _HW_SRC_CORES="探测失败"
    fi

    if _v="$(_hw_mem_cgroup)"; then
        _mem="${_v}"
        if [ -r "${_HW_CGROUP_ROOT}/memory.max" ]; then
            _HW_SRC_MEM="cgroup v2 memory.max"
        else
            _HW_SRC_MEM="cgroup v1 memory.limit_in_bytes"
        fi
        case "${_HW_CGROUP_ROOT}" in
            /sys/fs/cgroup) : ;;
            *) _HW_SRC_MEM="${_HW_SRC_MEM}（夹具 ${_HW_CGROUP_ROOT}）" ;;
        esac
    elif _v="$(_hw_mem_system)"; then
        _mem="${_v}"
        if [ -r /proc/meminfo ]; then _HW_SRC_MEM="/proc/meminfo MemTotal"; else _HW_SRC_MEM="sysctl hw.memsize"; fi
    else
        _mem=""
        _warn="${_warn}内存探测失败（按内存紧张一支回退）；"
        _HW_SRC_MEM="探测失败"
    fi

    case "$(uname -s 2>/dev/null)" in
        Linux) _glibc=1 ;;
        *)     _glibc=0 ;;
    esac

    # ── 2. 派生 ──────────────────────────────────────────────────────────────
    # 线程数 = 核数，**不封顶**；只做安全熔断（防探测异常把宿主的上千核当成本容器可用）。
    if [ -n "${_cores}" ]; then
        # 熔断**直接作用在 HW_CORES 上**：契约要求 HW_THREADS == HW_CORES，若只压线程数、
        # 让 HW_CORES 留着宿主的上千核，读契约做并行的调用方仍会按 2560 开池子 —— 熔断就白做了。
        _HW_CORES_RAW="${_cores}"
        if [ "${_cores}" -gt "${_HW_MAX_CORES}" ] 2>/dev/null; then
            _warn="${_warn}核数探测值 ${_cores} > 安全熔断 ${_HW_MAX_CORES}，按 ${_HW_MAX_CORES} 处理（探测异常？原始值见下方 [hw] 行）；"
            _cores="${_HW_MAX_CORES}"
        fi
        _threads="${_cores}"
    else
        _threads="$(nproc 2>/dev/null || printf '4')"
        [ "${_threads}" -gt 4 ] && _threads=4
    fi
    [ "${_threads}" -lt 1 ] && _threads=1

    # 内存未知 → 按"紧张"处理（保守：arena 压到 2），与需求的安全回退一致
    [ -n "${_mem}" ] || _mem_known=0
    if [ "${_mem_known}" = 0 ] || [ "${_mem:-0}" -le 8192 ]; then _mem_tight=1; fi

    # arena 与核数不是同一维度：只与"是不是 glibc + 内存紧不紧"有关，故保留上限
    if [ "${_glibc}" = 1 ]; then
        if [ "${_mem_tight}" = 1 ]; then
            _arena=2
        else
            _arena="${_threads}"
            [ "${_arena}" -lt 2 ] && _arena=2
            [ "${_arena}" -gt 8 ] && _arena=8
        fi
    fi

    # ── 3. 导出（显式覆盖优先：非空则不覆盖）────────────────────────────────
    if [ -n "${OMP_NUM_THREADS:-}${MKL_NUM_THREADS:-}${OPENBLAS_NUM_THREADS:-}${OLLAMA_NUM_THREADS:-}" ]; then
        _mark="（线程族有显式值，派生只填未设的项）"
    fi
    [ -n "${OLLAMA_NUM_THREADS:-}" ] && _thr_mark="(显式)"
    [ -n "${OMP_NUM_THREADS:-}" ]      || export OMP_NUM_THREADS="${_threads}"
    [ -n "${MKL_NUM_THREADS:-}" ]      || export MKL_NUM_THREADS="${_threads}"
    [ -n "${OPENBLAS_NUM_THREADS:-}" ] || export OPENBLAS_NUM_THREADS="${_threads}"
    [ -n "${OLLAMA_NUM_THREADS:-}" ]   || export OLLAMA_NUM_THREADS="${_threads}"

    if [ "${_glibc}" != 1 ]; then
        _arena_show="不适用(非glibc)"
    elif [ -n "${MALLOC_ARENA_MAX:-}" ]; then
        _arena_show="${MALLOC_ARENA_MAX}"; _arena_mark="(显式)"
    else
        export MALLOC_ARENA_MAX="${_arena}"
        _arena_show="${_arena}"
    fi

    # ── 4. 冻结契约：导出 HW_CORES / HW_MEM_MB / HW_THREADS ──────────────────
    # 其它 agent 的并行化直接读这三个，不要自己探测（容器里 nproc 是宿主核数）。
    export HW_CORES="${_cores:-?}"
    export HW_MEM_MB="${_mem:-?}"
    export HW_THREADS="${_threads}"
    _HW_GLIBC="${_glibc}"
    _HW_LOADED=1

    # ── 5. 审计行（必须能一眼看出"为什么是 2 而不是 8"）─────────────────────
    # 打印的是**生效值**（可能来自显式覆盖）；被覆盖时加 (显式) 标记。
    printf '[hw] cores=%s mem=%sMB glibc=%s threads=%s%s arena=%s%s 来源：%s / %s\n' \
        "${HW_CORES}" "${HW_MEM_MB}" "${_glibc}" "${OLLAMA_NUM_THREADS:-?}" "${_thr_mark}" \
        "${_arena_show}" "${_arena_mark}" "${_HW_SRC_CORES}" "${_HW_SRC_MEM}"
    printf '[hw] 导出契约：HW_CORES=%s HW_MEM_MB=%s HW_THREADS=%s（OMP/MKL/OPENBLAS/OLLAMA_NUM_THREADS 同值）\n' \
        "${HW_CORES}" "${HW_MEM_MB}" "${HW_THREADS}"
    [ "${_HW_CORES_RAW:-${HW_CORES}}" != "${HW_CORES}" ] && \
        printf '[hw] 熔断前原始核数=%s（仅记录；对外契约一律用 HW_CORES=%s）\n' "${_HW_CORES_RAW}" "${HW_CORES}"
    [ -n "${_warn}" ] && printf '[hw] ⚠️  %s\n' "${_warn}"
    [ -n "${_mark}" ] && printf '[hw] 注：%s\n' "${_mark}"
    return 0
}
