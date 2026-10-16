#!/usr/bin/env python3
"""硬件适配参数的**唯一实现**（机器事实只此一处）。

本地原生路径（tools/run_local.sh、tools/run_ollama.sh）与容器 entrypoint.sh 都通过
tools/hw_profile.sh 这个薄封装调用本模块；离线侧（chatbot.py 等）可以直接
"from hw_profile import detect, plan"，不必再写第二套 cgroup 探测。

为什么要有这个文件：同一份 .env 要同时服务多核 Mac 与 2vCPU 容器，写死必然有一边是错的
（写 2 则本地白丢大半算力 —— 本地因为容器化而变慢；写 8 则 2 vCPU 容器上开 8 个线程，
既慢又按线程吃内存）。**探测到多少核就用多少核**，两边各自吃满自己的配额。

★ 容器感知是重点（最容易出错、后果最严重的一处）★
容器里 nproc / /proc/meminfo 报的是**宿主的**核数与内存（FC 实例上可能是几十核、上百 G）。
只看它们会让 OpenMP/BLAS 与 llama.cpp 都按宿主规模去抢那两个 vCPU：上下文切换把吞吐打下去，
每线程的栈与 glibc arena 又把内存吃掉。故核数与内存都**优先读 cgroup 的 quota/limit**。

派生规则（是**策略**，不是值；改策略只改这一处）：
  CORES  = ceil(quota/period)   cgroup v2 <root>/cpu.max（"max" = 不限制）
                                cgroup v1 <root>/cpu/cpu.cfs_quota_us + cpu.cfs_period_us（-1 = 不限制）
           → 回退 CPU 计数（Linux，与 nproc 同口径）/ sysctl hw.ncpu→hw.physicalcpu→hw.perflevel0
           → 安全熔断：> 256 视为探测异常，压到 256（**不是**性能封顶，正常机器碰不到）
  MEM_MB = cgroup v2 <root>/memory.max / v1 <root>/memory/memory.limit_in_bytes（"max"/天文数字 = 不限制）
           → 回退 /proc/meminfo MemTotal / sysctl hw.memsize
  THREADS = CORES                                    # **不封顶**
  OMP/MKL/OPENBLAS_NUM_THREADS = THREADS             # torch 的 OpenMP/BLAS
  OLLAMA_NUM_THREADS           = THREADS             # llama.cpp 生成线程
  MALLOC_ARENA_MAX = 2                     glibc 且 MEM_MB <= 8192（内存紧张，压 per-thread arena 碎片）
                   = min(8, max(2, THREADS)) glibc 且内存宽裕
                   = 不设                    非 glibc（macOS 的 malloc 没有 arena 概念）
                     ↑ 它与核数**不是同一维度**：只与"内存紧不紧 + 是不是 glibc"有关，故保留上限。
  探测失败 → 安全回退：THREADS = min(CPU 计数, 4)，内存按"紧张"一支处理，并告警。

显式覆盖优先（与 tools/dotenv.sh 的"进程环境优先"同一语义）：调用方**先**加载 .env、**再**调用
plan()；进程环境里已非空的那几项一律尊重，派生只做默认值。

测试钩子（仅供夹具验证，生产环境不会设）：
  _HW_CGROUP_ROOT  把 cgroup 根指到夹具目录（默认 /sys/fs/cgroup）
  _HW_OS           强制平台名（Linux / Darwin）—— 用来在 macOS 上验证 glibc 分支；
                   Python 的 os.uname() 走系统调用，伪装 uname 可执行文件对它无效

命令行：
  python3 tools/hw_profile.py --sh     # 打印 export/赋值语句供 shell eval（stdout 只有赋值）
  python3 tools/hw_profile.py --json   # 打印完整计划（排查用）
"""

import json
import os
import shlex
import subprocess
import sys

# ── 测试钩子 ────────────────────────────────────────────────────────────────
CGROUP_ROOT = os.environ.get("_HW_CGROUP_ROOT") or "/sys/fs/cgroup"
OS_OVERRIDE = os.environ.get("_HW_OS") or ""

# ── 策略常量 ────────────────────────────────────────────────────────────────
MAX_CORES = 256            # 安全熔断（防"容器里读到宿主上千核"），不是性能封顶
MEM_TIGHT_MB = 8192        # <= 它按"内存紧张"一支派生
ARENA_MAX = 8
UNLIMITED = 9000000000000000000   # cgroup 用它表示"未限制"的天文数字下界
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OLLAMA_NUM_THREADS")


def platform_name():
    """Linux / Darwin / …；_HW_OS 可强制（测试钩子）。"""
    if OS_OVERRIDE:
        return OS_OVERRIDE
    try:
        return os.uname().sysname
    except AttributeError:      # 非 Unix（理论上不会走到）
        return ""


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _sysctl(key):
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _ceil_div(a, b):
    return -(-a // b)


# ── 探测：cgroup v2 → v1 → 系统 ──────────────────────────────────────────────

def cores_from_cgroup():
    """容器真实 CPU 配额 → 核数（向上取整）。拿不到返回 None。"""
    raw = _read_text(os.path.join(CGROUP_ROOT, "cpu.max"))      # v2: "<quota|max> <period>"
    if raw:
        parts = raw.split()
        if len(parts) >= 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if period > 0:
                    return _ceil_div(quota, period)
            except ValueError:
                pass
    try:                                                        # v1
        quota = int(_read_text(os.path.join(CGROUP_ROOT, "cpu/cpu.cfs_quota_us")))
        period = int(_read_text(os.path.join(CGROUP_ROOT, "cpu/cpu.cfs_period_us")))
        if quota > 0 and period > 0:
            return _ceil_div(quota, period)
    except ValueError:
        pass
    return None


def mem_from_cgroup():
    """容器真实内存上限 → MB。拿不到 / 无限制返回 None。"""
    for path in (os.path.join(CGROUP_ROOT, "memory.max"),
                 os.path.join(CGROUP_ROOT, "memory/memory.limit_in_bytes")):
        raw = _read_text(path)
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if 0 < value < UNLIMITED:
            return value // 1048576
    return None


def cores_from_system():
    """非容器回退 → (核数, 来源标签)。Linux 用 CPU 计数（与 nproc 同口径：尊重 affinity）。"""
    if platform_name() == "Linux":
        try:
            count = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            count = os.cpu_count() or 0
        if count > 0:
            return count, "nproc"
        return None, None
    for key in ("hw.ncpu", "hw.physicalcpu", "hw.perflevel0.physicalcpu"):
        try:
            count = int(_sysctl(key))
        except ValueError:
            continue
        if count > 0:
            return count, "sysctl hw.*"
    return None, None


def mem_from_system():
    """非容器回退 → (MB, 来源标签)。"""
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024, "/proc/meminfo MemTotal"
        except (OSError, ValueError, IndexError):
            pass
    try:
        size = int(_sysctl("hw.memsize"))
    except ValueError:
        size = 0
    if size > 0:
        return size // 1048576, "sysctl hw.memsize"
    return None, None


def detect():
    """纯机器事实（无派生、无覆盖）。可被离线侧直接 import。"""
    cores = cores_from_cgroup()
    if cores is not None:
        src_cores = "cgroup v2 cpu.max" if os.path.exists(os.path.join(CGROUP_ROOT, "cpu.max")) \
            else "cgroup v1 cpu.cfs_quota_us"
        if CGROUP_ROOT != "/sys/fs/cgroup":
            src_cores += "（夹具 %s）" % CGROUP_ROOT
    else:
        cores, src_cores = cores_from_system()
        if cores is None:
            src_cores = "探测失败"

    mem = mem_from_cgroup()
    if mem is not None:
        src_mem = "cgroup v2 memory.max" if os.path.exists(os.path.join(CGROUP_ROOT, "memory.max")) \
            else "cgroup v1 memory.limit_in_bytes"
        if CGROUP_ROOT != "/sys/fs/cgroup":
            src_mem += "（夹具 %s）" % CGROUP_ROOT
    else:
        mem, src_mem = mem_from_system()
        if mem is None:
            src_mem = "探测失败"

    return {
        "cores": cores,
        "mem_mb": mem,
        "glibc": 1 if platform_name() == "Linux" else 0,
        "src_cores": src_cores or "探测失败",
        "src_mem": src_mem or "探测失败",
        "cgroup_root": CGROUP_ROOT,
    }


def plan(env=None):
    """探测 + 派生 + 显式覆盖 → 完整计划（导出值、辅助值、审计元数据）。

    env 默认取 os.environ；显式传入一个映射即可做单元测试。
    """
    env = os.environ if env is None else env
    facts = detect()

    cores = facts["cores"]
    mem = facts["mem_mb"]
    glibc = facts["glibc"]
    warn = ""
    cores_raw = ""

    # 核数：不封顶，只做安全熔断（熔断直接作用在 cores 上，保证 HW_THREADS == HW_CORES）
    if cores is not None:
        cores_raw = cores
        if cores > MAX_CORES:
            warn += "核数探测值 %d > 安全熔断 %d，按 %d 处理（探测异常？原始值见下方 [hw] 行）；" % (
                cores, MAX_CORES, MAX_CORES)
            cores = MAX_CORES
        threads = cores
    else:
        warn += "核数探测失败（按 4 线程安全回退）；"
        try:
            threads = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            threads = os.cpu_count() or 4
        if threads > 4:
            threads = 4
    if threads < 1:
        threads = 1

    # 内存未知 → 按"紧张"处理（保守），与 shell 版同一回退
    if mem is None:
        warn += "内存探测失败（按内存紧张一支回退）；"
    tight = (mem is None) or (mem <= MEM_TIGHT_MB)

    # arena 与核数不是同一维度：只与"是不是 glibc + 内存紧不紧"有关，故保留上限
    arena = None
    if glibc:
        if tight:
            arena = 2
        else:
            arena = min(ARENA_MAX, max(2, threads))

    # ── 显式覆盖优先 ──
    exports = {}
    for name in THREAD_VARS:
        if not env.get(name):
            exports[name] = str(threads)
    thread_mark = "(显式)" if env.get("OLLAMA_NUM_THREADS") else ""

    if not glibc:
        arena_show, arena_mark = "不适用(非glibc)", ""
    elif env.get("MALLOC_ARENA_MAX"):
        arena_show, arena_mark = str(env["MALLOC_ARENA_MAX"]), "(显式)"
    else:
        exports["MALLOC_ARENA_MAX"] = str(arena)
        arena_show, arena_mark = str(arena), ""

    exports["HW_CORES"] = "?" if cores is None else str(cores)
    exports["HW_MEM_MB"] = "?" if mem is None else str(mem)
    exports["HW_THREADS"] = str(threads)

    if any(env.get(n) for n in THREAD_VARS):
        mark = "（线程族有显式值，派生只填未设的项）"
    else:
        mark = ""

    aux = {
        "_HW_SRC_CORES": facts["src_cores"],
        "_HW_SRC_MEM": facts["src_mem"],
        "_HW_GLIBC": str(glibc),
        "_HW_CORES_RAW": "" if cores_raw == "" else str(cores_raw),
        "_HW_THR_MARK": thread_mark,
        "_HW_ARENA_SHOW": arena_show,
        "_HW_ARENA_MARK": arena_mark,
        "_HW_WARN": warn,
        "_HW_MARK": mark,
    }
    return {"facts": facts, "exports": exports, "aux": aux,
            "cores": exports["HW_CORES"], "mem_mb": exports["HW_MEM_MB"],
            "threads": exports["HW_THREADS"]}


def main(argv):
    mode = argv[1] if len(argv) > 1 else "--sh"
    result = plan()
    if mode == "--sh":
        # stdout **只放赋值语句**（供 eval）；诊断一律走 stderr，否则 eval 会炸
        for name, value in result["exports"].items():
            sys.stdout.write("export %s=%s\n" % (name, shlex.quote(value)))
        for name, value in result["aux"].items():
            sys.stdout.write("%s=%s\n" % (name, shlex.quote(value)))
        return 0
    if mode == "--json":
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return 0
    if mode in ("-h", "--help"):
        sys.stdout.write(__doc__ + "\n")
        return 0
    sys.stderr.write("用法: hw_profile.py [--sh|--json|--help]\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
