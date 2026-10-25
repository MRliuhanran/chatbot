# ── 基础设施：环境配置与日志 ──
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
import unicodedata
import warnings
from collections import Counter, defaultdict
from logging.handlers import RotatingFileHandler
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError("缺少依赖 python-dotenv（配置会静默退到代码默认值），安装: pip install python-dotenv") from exc
load_dotenv()

# ── 启动副作用 ──
# 本地 Ollama/Qdrant 不走系统代理（否则 requests 可能被劫持而超时）。
# 幂等：重复 import（reload/测试）时把历史已写入的 NO_PROXY 一并拼进来去重，
# 不会因为再跑一次就把旧值再包一层。
_proxy_parts = [p.strip() for p in os.environ.get("NO_PROXY", "").replace(",", " ").split()]
_loopback = ",".join(dict.fromkeys(p for p in _proxy_parts + ["127.0.0.1", "localhost", "::1"] if p))
os.environ["NO_PROXY"] = os.environ["no_proxy"] = _loopback
del _proxy_parts, _loopback
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum")


def env_int(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"环境变量 {name}={raw!r} 不是合法整数") from None


def env_float(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise ValueError(f"环境变量 {name}={raw!r} 不是合法浮点数") from None


def env_int_min(name, default, minimum):
    val = env_int(name, default)
    if val < minimum:
        raise ValueError(f"环境变量 {name}={val!r} 小于允许的最小值 {minimum}")
    return val


def env_float_min(name, default, minimum):
    val = env_float(name, default)
    if val < minimum:
        raise ValueError(f"环境变量 {name}={val!r} 小于允许的最小值 {minimum}")
    return val


_ENV_TRUE = ("1", "true", "yes", "on")
_ENV_FALSE = ("0", "false", "no", "off")


def env_bool(name, default):
    """布尔配置：只认显式真/假集合，其余取值**报错**。

    旧写法把 0/false/no/off 之外的一切都当真：RAG_EMBED_CACHE=flase（拼错）、
    =2、=禁用 都会静默取真，缓存、别名这些开关于是"配了等于没配"，
    且日志里看不出任何异常（失效原型：静默）。与 env_int/env_choice 一样具名报错。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    val = raw.strip().lower()
    if val in _ENV_TRUE:
        return True
    if val in _ENV_FALSE:
        return False
    raise ValueError(
        f"环境变量 {name}={raw!r} 不是合法布尔值（真: {'/'.join(_ENV_TRUE)}，假: {'/'.join(_ENV_FALSE)}）"
    )


def env_choice(name, default, allowed):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    val = raw.strip().lower()
    if val not in allowed:
        raise ValueError(f"环境变量 {name}={raw!r} 不在允许取值 {sorted(allowed)} 内")
    return val


def env_int_any(default, *names):
    """按顺序取第一个已设置且非空的名字（等价旧代码的 os.getenv(a) or os.getenv(b) 链），
    但解析失败时报出**具体那个**名字 —— 裸 int() 只抛 "invalid literal"，指不到变量。"""
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return env_int(name, default)
    return default


# ── 机器规格探测：离线负载能用的核数与内存 ──
# tools/hw_profile.sh 是 shell 侧的**唯一实现**，但它派生的是线程族（OMP/MKL/Ollama/torch）并把
# THREADS 封顶 8 —— 那是在线推理的合理策略（再多线程只是互相争抢）；离线批处理（process 的
# 分块池、index 的写批、sparse 的页更新）要的是**真实核数**：16 核机器上只开 8 路就是白丢
# 一半算力。另外 chatbot.py 也可能被直接调用（不经 run_local.sh / entrypoint.sh），那时 shell
# 侧的派生根本没跑过。故 Python 侧再探一次，规则与 hw_profile.sh 同源：
#   CORES ：HW_CORES（契约值，不封顶）→ cgroup v2 cpu.max → cgroup v1 cpu.cfs_quota_us → nproc/sysctl
#   MEM_MB：HW_MEM_MB → cgroup v2 memory.max → cgroup v1 memory.limit_in_bytes → sysconf/meminfo
# 显式覆盖优先的语义一致：环境里写了就用，探测只当默认值。探测失败 → 安全回退（核数 4、内存未知）。
_HW_PROBE: dict = {}


def _probe_positive_int(name):
    """环境变量的正整数读法；未设/空/非法都返回 None（探测失败不抛，由调用方回退）。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        val = int(raw.strip())
    except ValueError:
        return None
    return val if val > 0 else None


def _hw_profile_module():
    """按文件路径 import tools/hw_profile.py（不依赖包结构与 CWD）；失败返回 None。

    **机器事实只此一处实现**：容器 entrypoint 与 tools/run_local.sh 走它导出的 HW_*；
    裸跑 `python chatbot.py process|index|sparse`（不经过那两个入口）时环境里没有 HW_*，
    这里直接取同一份 detect() —— 两条路径同一份实现，不会各探一套而分叉。
    """
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "hw_profile.py")
    try:
        spec = importlib.util.spec_from_file_location("_rag_hw_profile", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _hw_facts():
    """(核数, 内存MB, 来源串)。环境变量优先 → tools/hw_profile.detect() → 安全回退。"""
    cached = _HW_PROBE.get("facts")
    if cached:
        return cached
    cores = _probe_positive_int("HW_CORES")
    mem = _probe_positive_int("HW_MEM_MB")
    src = []
    if cores:
        src.append(f"HW_CORES={cores}")
    if mem:
        src.append(f"HW_MEM_MB={mem}")
    if not cores or not mem:
        facts: dict = {}
        mod = _hw_profile_module()
        if mod is not None:
            try:
                facts = mod.detect() or {}
            except Exception:
                facts = {}
        if not cores and facts.get("cores"):
            cores = int(facts["cores"])
            src.append(f"hw_profile.detect()←{facts.get('src_cores', '?')}")
        if not mem and facts.get("mem_mb"):
            mem = int(facts["mem_mb"])
            src.append(f"hw_profile.detect()←{facts.get('src_mem', '?')}")
    if not cores:
        # 最后兜底：线程族也是"这台机器能用几个核"的另一份结论，直接沿用
        for name in ("HW_THREADS", "TORCH_NUM_THREADS", "OMP_NUM_THREADS"):
            cores = _probe_positive_int(name)
            if cores:
                src.append(f"回退 {name}={cores}")
                break
    facts_out = (max(1, int(cores)) if cores else 4, mem if mem else None, " + ".join(src) or "探测失败→安全回退(4 核/内存按紧张)")
    _HW_PROBE["facts"] = facts_out
    return facts_out


def hw_cores():
    """离线负载可用核数（**不封顶**）。"""
    return _hw_facts()[0]


def hw_mem_mb():
    """可用内存（MB）；未知返回 None（调用方按"内存紧张"处理，与 hw_profile 同一回退）。"""
    return _hw_facts()[1]


def hw_report():
    """审计行：为什么是 N 路（与 hw_profile 的 [hw] 行同一意图，日志里要能一眼归因）。"""
    cores, mem, src = _hw_facts()
    return f"cores={cores} mem={str(mem) + 'MB' if mem else '未知(按内存紧张回退)'} 来源={src}"


def _mem_available_mb():
    """当前可开内存 MB，**取保守口径（宁可报小、绝不报大）**。

    · 坑：Docker 默认不隔离 /proc/meminfo —— 容器里读到的常是**宿主机**的数（宿主可能
      剩 12G，cgroup 限额 8G 却已被本容器用满）。单信它会误判"有余量"→ 水位豁免放行
      → 检索与生成同驻 → cgroup OOM kill。因此这里同时读 cgroup 头寸
      （memory.max - memory.current；v1 用 usage/limit），与宿主 MemAvailable 取 min。
      报小的失真方向是安全的：current 把可回收的 file 页也算进占用 → 头寸偏小 → gate
      倾向照卸，不会导向"误判有余量"的死亡方向。
    · 探测不到（macOS 无 /proc 无 cgroup v2/v1 文件）返回 None —— 调用方必须按
      "闸门不可用"跳过，而不是拿 None 当无穷大。
    """
    readings = []
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    readings.append(int(line.split()[1]) // 1024)
                    break
    except Exception:
        pass
    for _cur, _lim in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                       ("/sys/fs/cgroup/memory.usage_in_bytes", "/sys/fs/cgroup/memory.limit_in_bytes")):
        try:
            with open(_cur, encoding="utf-8") as f1:
                _u = int(f1.read().strip()) // 1024
            with open(_lim, encoding="utf-8") as f2:
                _raw = f2.read().strip()
            if _raw == "max":
                continue
            _lim_kb = int(_raw) // 1024
            if _lim_kb > (1 << 40):      # v1"未设限额"是一个天文数——按无限处理，不产头寸
                continue
            readings.append(max(0, _lim_kb - _u))
        except Exception:
            continue
    return min(readings) if readings else None


# 单路离线 worker 的常驻内存量级（解释器 + 库 + 一份工作集）；内存未知 → 只开 1 路（与
# hw_profile.sh「内存未知按紧张处理」同一回退）。核数封顶、内存封底，两个都是机器事实。
_OFFLINE_WORKER_MEM_MB = 768
# 每一路写批在内存里驻留的点批（WRITE_BATCH 条稠密向量 + 序列化副本）量级
_WRITE_WORKER_MEM_MB = 512


def offline_workers(mem_mb=None, unit_mb=_OFFLINE_WORKER_MEM_MB):
    """离线池并行度 = min(核数, 可用内存 / 单路常驻)。"""
    mem = hw_mem_mb() if mem_mb is None else mem_mb
    by_mem = max(1, mem // unit_mb) if mem else 1
    return max(1, min(hw_cores(), by_mem))


def embed_batch_size(device, mem_mb=None):
    """嵌入批大小：**按设备与可用内存派生**（显式 RAG_EMBED_BATCH_SIZE 时不调用本函数）。

    实测梯度（真语料；MPS = 本机 M2/8G/2048 块，CPU = 2 线程模拟 2vCPU 容器/128 块）：

      MPS fp16 : bs   8    16    32    64    128
                 块/秒 13.0  12.9  11.8  12.2  10.9
                 峰值MB 2067  2067  2067  3091  3093
      CPU fp32 : bs   8    16    32    64    128
                 块/秒  4.7   4.7   5.2   5.0   3.0
                 峰值MB 2174  2118  2245  2594  2100

    结论：**两边都不是越大越好**。MPS 上 16 与 8 并列最快、比 64 快 6% 且少占 1GB 显存；
    CPU 上 32 最快（每批的 python/padding 开销摊薄），而 128 在 8G 机器上因内存压力塌到
    3.0 块/秒（fp32 激活随批线性涨）。故取各设备**实测最优点**，再按"内存不足退一档"收口。
    """
    mem = hw_mem_mb() if mem_mb is None else mem_mb
    if device == "mps":
        batch = 16
    elif device == "cuda":
        # 本机无 CUDA 设备可实测：按**显存**派生（fp16 权重 ~1.2GB、每样本激活 ~40MB 量级，
        # 取显存 1/8 作批预算），显存读不到退回 64；仍可被 RAG_EMBED_BATCH_SIZE 覆盖。
        try:
            import torch
            vram_mb = int(torch.cuda.get_device_properties(0).total_memory) // 1048576
        except Exception:
            vram_mb = 0
        batch = max(8, min(128, (vram_mb // 8) // 40)) if vram_mb else 64
    else:
        batch = 32
    if mem is not None and mem < 4096:
        batch = max(8, batch // 2)
    return batch


# ── Qdrant 写批并发度：Little 定律 + 实测平台，不照抄核数 ──
# 并发 = 目标吞吐 × 单批往返时延（RTT），两个数都是**实测**的（本机 Qdrant，21255 点 /
# 22 批 × 1000 点，稠密 1024 维 + 稀疏 Document，与 index 写入路径同一套 points）：
#   1 路：写批墙钟 37.0 / 42.1s（505~574 点/秒），单批 = 构造 0.10s + 往返 1.57s
#   4 路：17.1 / 27.5 / 32.3s（658~1246 点/秒）
#   8 路：21.4 / 21.6 / 29.1s（730~982 点/秒）      ← 平台
#  12 路：21.5s（988 点/秒）                        ← 不再涨
# 故 Little 下限（1000 点/秒 × 1.5s ÷ 1000 点/批 = 1.5 → 2 路）只是"把往返等待藏住"的
# 下限；曲线要到 4~8 路才平，因为**客户端构造 points（emb.tolist + JSON）在 GIL 下也会
# 排队**，需要额外线程互相搭桥。取 max(4, 2×Little) 为平台起点，再用核数/内存/批数收口。
QDRANT_BATCH_RTT_S = 1.5        # 1 路并发时单批（1000 点）实测墙钟（秒）——原 1.0 是猜的
QDRANT_TARGET_PPS = 1000        # 实测平台吞吐（点/秒）


def qdrant_write_workers(batch_count, mem_mb=None):
    """Qdrant 写批/页并发度：Little 下限 × GIL 搭桥系数，再按核数、内存、批数收口。"""
    mem = hw_mem_mb() if mem_mb is None else mem_mb
    by_little = (QDRANT_TARGET_PPS * QDRANT_BATCH_RTT_S + WRITE_BATCH - 1) // WRITE_BATCH
    by_mem = max(1, mem // _WRITE_WORKER_MEM_MB) if mem else 1
    return max(1, min(hw_cores(), by_mem, max(1, batch_count), max(4, by_little * 2)))


def embed_batch_note(device):
    """批大小的来源（日志要能一眼看出"值 + 依据"，不是新的黑盒）。"""
    if EMBED_BATCH_SIZE:
        return f"显式 RAG_EMBED_BATCH_SIZE={EMBED_BATCH_SIZE}"
    return f"派生 device={device} {hw_report()}"


class _StdoutHandler(logging.StreamHandler):
    def emit(self, record):
        self.stream = sys.stdout
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)


class _LockedRotatingFileHandler(RotatingFileHandler):
    """多进程安全的轮转：flock 把「查大小→轮转→写入」整段串起来。

    CLI（process/index/…）与常驻服务（api/serve）可能同时写同一份 rag.log：两个进程
    同时进 doRollover 会互相覆盖 *.log.N，日志丢行甚至把轮转中的文件写坏。fcntl 不可用
    的平台退回基类行为（单进程安全，多进程轮转尽力而为，不让日志把主流程带崩）。

    flock 之外还要修第二个失效源（stale fd）：进程 A 轮转 rename 后，进程 B 手里
    持有的 fd 仍指向已被改名的 rag.log.N —— 只凭 fd 偏移继续写，日志会落进旧编号
    文件直到 B 自己触发下一次轮转。emit 里按「当前路径的 inode/大小」复核：
    fd 已脱离现行文件就重开，当前路径真实大小超限就按它轮转。
    """

    def shouldRollover(self, record, msg):
        """禁用基类 fd 位置轮转：BaseRotatingHandler.emit 的默认流程还会走一遍
        (stream.tell() + len(msg)) >= maxBytes —— fd 陈旧时这是**旧文件的偏移**，结论不可
        靠（实测：每条 emit 轮转两次、备份文件只剩目标交替）。轮转的唯一触发点交给
        _rotate_if_current_is_full（按当前路径真实大小），本钩子恒 False。"""
        return False

    def emit(self, record):
        try:
            import fcntl
        except ImportError:
            super().emit(record)
            return
        try:
            fd = os.open(self.baseFilename + ".lock", os.O_CREAT | os.O_WRONLY, 0o644)
        except OSError:
            super().emit(record)
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._resync_borrowed_fd()
            if self.stream is None:
                self.stream = self._open()
            self._rotate_if_current_is_full()
            logging.FileHandler.emit(self, record)
        finally:
            os.close(fd)  # close 即释放 flock

    def _resync_borrowed_fd(self):
        """fd 指向的文件若已不是当前 backing 文件（他人轮转 rename 走过），重开到现行文件。"""
        try:
            st_path = os.stat(self.baseFilename)
        except OSError:
            return
        if self.stream is None:
            return
        try:
            st_fd = os.fstat(self.stream.fileno())
        except OSError:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = self._open()
            return
        if (st_fd.st_dev, st_fd.st_ino) != (st_path.st_dev, st_path.st_ino):
            self.stream.close()
            self.stream = self._open()

    def _rotate_if_current_is_full(self):
        """按**当前路径真实大小**判定轮转：stream.tell() 在 fd 陈旧时是旧文件的偏移，不可信。"""
        if not self.maxBytes:
            return
        try:
            if os.path.getsize(self.baseFilename) < self.maxBytes:
                return
        except OSError:
            return
        try:
            self.doRollover()
        except Exception:
            pass  # 轮转失败按原文件继续写（保守：不丢日志，占盘风险交给下一次 emit 重试）


_LOG_FILE = None
# 日志文件默认名的单一定义处：文件 handler 与 Streamlit 报错提示（指向 rag.log）共用，
# 避免两处各写一个默认字符串，以后改默认名只改这一行。
DEFAULT_LOG_FILE = "rag.log"


def _file_handler():
    """共享日志文件（RAG_LOG_FILE，默认 rag.log）；打开失败退回仅控制台。

    必须轮转：固定 FileHandler 会无限增长，长跑几个月吃满容器可写层，届时"Qdrant 写不进"
    的根因被磁盘满掩盖，而运维是被指到这里来看根因的。轮转产生的 rag.log.N 不匹配
    .gitignore 的 *.log，故 .gitignore 里另有 *.log.* 一条。
    """
    global _LOG_FILE
    if _LOG_FILE is None:
        path = os.getenv("RAG_LOG_FILE", DEFAULT_LOG_FILE)
        try:
            h = _LockedRotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        except Exception:
            h = None
        _LOG_FILE = h
    return _LOG_FILE


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_StdoutHandler())
        fh = _file_handler()
        if fh is not None:
            logger.addHandler(fh)
        _lvl = os.getenv("RAG_LOG_LEVEL", "INFO").strip().upper() or "INFO"
        if _lvl not in logging._nameToLevel:
            # 一个笔误不该让所有命令起不来（import 期就崩）：告警 + 回退 INFO。
            logging.getLogger().warning("未知 RAG_LOG_LEVEL=%r，回退到 INFO（可用: DEBUG/INFO/WARNING/ERROR/CRITICAL）", _lvl)
            _lvl = "INFO"
        logger.setLevel(_lvl)
        logger.propagate = False
    return logger
console = get_logger("console")
logger = get_logger("rag_engine")


def log(*parts):
    console.info(" ".join(str(part) for part in parts))


# ── 运行态单例（模块级缓存/锁的统一定义处）──
# tokenizer、词表、别名解析结果都属于"每个进程只应有一份、且懒加载要加锁"的状态。
# Ollama 的 requests.Session **不在**这里：requests.Session 不保证线程安全，进程级
# 单例会让预热线程与多个会话线程并发共用同一个连接池。它按线程各持一份（_OLLAMA_SESSION_TLS，
# 线程结束随线程回收，keep-alive 收益不减），集合进 _RuntimeState 反而制造跨线程竞争。
# 集中放一个 _RuntimeState：多线程（预热线程/UI 脚本线程）与 Streamlit 多次 rerun 之间不会
# 出现"同一载荷两份缓存"的分叉；__init__ 里析构即可。属性全部在模块级初始化，保证任何
# 线程首个访问者拿到的都只这一份（_state_ref 双检锁）。
class _RuntimeState:
    def __init__(self):
        self.lexicon = None
        self.lexicon_lock = threading.Lock()
        self.gen_tokenizer = None
        self.gen_tokenizer_tried = False
        self.gen_tokenizer_lock = threading.Lock()
        self.alias_resolved = {}
        self.alias_degraded = False


_STATE = None
_STATE_LOCK = threading.Lock()


def _state_ref():
    """模块级运行态单例（双检锁）：任何线程首访问只构造一份，之后全部复用。"""
    global _STATE
    if _STATE is None:
        with _STATE_LOCK:
            if _STATE is None:
                _STATE = _RuntimeState()
    return _STATE


def _streamlit_runtime_state():
    """Streamlit 侧取同一份运行态：cache_resource 保证会话内复用，指向的就是模块单例。
    必须在 run_streamlit 里先于 engine 构建调用，让预热/首查询线程与脚本线程共用一份。"""
    import streamlit as st
    return st.cache_resource(show_spinner=False)(_state_ref)()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_URL = f"{OLLAMA_BASE_URL}/api/chat"
MODEL = os.getenv("MODEL", "qwen3.5:2b-q4_K_M")
# 思维链唯一开关：模型/页面/API/会话共用（默认关）；开启时只展示（历史与正文同栏）、永不进入生成上下文。
THINK = env_bool("RAG_THINK", False)
# num_ctx / num_predict 与思维链开关解耦：固定预算，不随开关浮动。
# 解析必须走 env_int（具名报错）；下限在 GEN_INPUT_RESERVE 定义处与它交叉校验。
NUM_CTX = env_int("RAG_NUM_CTX", 8192)
NUM_PREDICT = env_int("RAG_NUM_PREDICT", 4096)
TIMEOUT = env_float_min("OLLAMA_TIMEOUT", 180, 1.0)
# 生成流的读空闲超时（无任何新数据即判超时）与整轮硬上限：
# 先前的"无时间闸"让 Ollama/llama-server 僵死时前端只能永久空白；这两个闸只负责
# 把"无进展"变成显式错误，不限制正常生成长度（正常流持续有 chunk 不会触发）。
STREAM_READ_TIMEOUT = env_float_min("RAG_STREAM_READ_TIMEOUT", 300, 10.0)
TURN_TIMEOUT = env_float_min("RAG_TURN_TIMEOUT", 1800, 60.0)
# 采样温度：0=贪心（问答类检索场景的默认）；改它走 RAG_TEMPERATURE，不改代码。
TEMPERATURE = env_float("RAG_TEMPERATURE", 0.0)
# 惩罚回看窗口：repeat/presence/frequency 三类惩罚共用同一历史窗口（llama.cpp 采样链）。
# 64=Ollama 内置默认；128 拉长一倍去挡间隔更大的原文复读（思考链循环式复核）。
REPEAT_LAST_N = env_int_min("RAG_REPEAT_LAST_N", 64, 1)
# 输入窗口上限 = RAG_NUM_CTX - RAG_INPUT_RESERVE（token 度量）：窗口**全部给输入，不做输出
# 预留**。生成空间由服务端上下文移位腾出 —— 生成超窗时从最前面按 token 遗忘（先丢输入前文
# /历史，见 plan_generation）。RAG_NUM_PREDICT ≤ 0 时请求发 num_predict=-1，输出不限长。
# ── 采样/惩罚（Ollama options；单一来源见 _chat_options）──
# 截断类：temperature=0（贪心）时 top_p/top_k/min_p 全都不参与 argmax，设了也不影响输出；
# 保留它们，是为了温度>0 做 A/B 时有可复现的截断参数。`RAG_TOP_K` 已被「检索返回条数」
# 占用，采样侧统一加 gen_ 前缀。0 表示该截断不生效。
GEN_TOP_K = env_int_min("RAG_GEN_TOP_K", 20, 0)
GEN_TOP_P = env_float_min("RAG_GEN_TOP_P", 0.9, 0.0)
GEN_MIN_P = env_float_min("RAG_GEN_MIN_P", 0.0, 0.0)
# 惩罚类：直接改 logits，贪心下依然生效。模型自带 presence_penalty=1.5（`ollama show`），
# 对资料/提问里出现过的每个字词都减 1.5，会压制人名、术语的**正常复现** —— 默认显式归零
# 覆盖模型默认；要复现旧行为把 RAG_GEN_PRESENCE_PENALTY 写回 1.5。
GEN_PRESENCE_PENALTY = env_float("RAG_GEN_PRESENCE_PENALTY", 0.0)
GEN_FREQUENCY_PENALTY = env_float("RAG_GEN_FREQUENCY_PENALTY", 0.0)
GEN_REPEAT_PENALTY = env_float_min("RAG_GEN_REPEAT_PENALTY", 1.1, 0.0)
# 随机种子：-1=每次随机（temperature=0 下无影响）；固定正整数用于可复现采样。
GEN_SEED = env_int("RAG_GEN_SEED", -1)
# 生成侧**不设任何长度/时间闸、也不重试**：窗口（RAG_NUM_CTX，token 度量）是唯一上限；超窗
# 由服务端上下文移位从最前面按 token 丢续写；空答/截断/报错原样交给上层。


_OLLAMA_SESSION_TLS = threading.local()
_MODEL_HTTP_TLS = threading.local()


def _model_http_session():
    """模型服务 HTTP 连接复用：线程本地 requests.Session。

    与 _ollama_session 同一策略：不自己实现连接池，用 requests 自带适配器；
    线程本地避免 requests.Session 跨线程共享内部状态；keep-alive 由库维护。
    """
    session = getattr(_MODEL_HTTP_TLS, "session", None)
    if session is None:
        import requests
        from requests.adapters import HTTPAdapter
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=32)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _MODEL_HTTP_TLS.session = session
    return session


def _ollama_session():
    """Ollama HTTP 连接复用：**线程本地** Session 走 keep-alive，省每请求的 TCP 握手。

    旧写法是进程级单例 Session：requests.Session 官方不保证线程安全，预热线程与多个
    会话线程并发复用同一连接池/适配器会竞争内部状态，且一份会话的连接故障会传染到所有
    会话。线程本地各持一份后 keep-alive 收益不减，跨线程竞争消失。
    """
    session = getattr(_OLLAMA_SESSION_TLS, "session", None)
    if session is None:
        import requests
        session = requests.Session()
        _OLLAMA_SESSION_TLS.session = session
    return session



# ── 生成客户端：Ollama 流式对话 ──
# 事件词表封闭产出（stream_chat + run_turn）：retrieval → thinking / delta / error / done。
EVENT_RETRIEVAL = "retrieval"
EVENT_THINKING = "thinking"
EVENT_DELTA = "delta"
EVENT_ERROR = "error"
EVENT_DONE = "done"


def _chat_options(temperature=None, num_ctx=None, num_predict=None):
    """Ollama 生成参数的单一定义处：生成请求与预热都从这里取。

    只显式列出应用要**钉死**的项；没列出的才交给模型自带默认。模型默认里
    `presence_penalty=1.5` 对本 RAG 场景偏强，已在此显式覆盖，避免"静默继承一个
    没人注意到的高惩罚"。
    """
    _predict = int(NUM_PREDICT if num_predict is None else num_predict)
    opts = {
        "temperature": TEMPERATURE if temperature is None else temperature,
        "num_ctx": int(NUM_CTX if num_ctx is None else num_ctx),
        # ≤0 = 不限长，映射到 Ollama 的 -1（默认即无限）；正数才是硬上限。
        "num_predict": _predict if _predict > 0 else -1,
        "top_k": int(GEN_TOP_K),
        "top_p": float(GEN_TOP_P),
        "min_p": float(GEN_MIN_P),
        "repeat_last_n": int(REPEAT_LAST_N),
        "repeat_penalty": float(GEN_REPEAT_PENALTY),
        "presence_penalty": float(GEN_PRESENCE_PENALTY),
        "frequency_penalty": float(GEN_FREQUENCY_PENALTY),
        "seed": int(GEN_SEED),
    }
    # num_gpu 只在这里组装（**必须在 options 里**，顶层同名字段 Ollama 不认）：
    # 取"生效值"而不是原始模式串 —— auto 模式下生效值是 999，回落之后是 None。
    _ng = _gen_num_gpu_effective()
    if _ng is not None:
        opts["num_gpu"] = _ng
    return opts


# GPU 优先/回落的进程内状态（RAG_GEN_NUM_GPU=auto）。回落是**一次性**的：一旦确认这台机器
# 起不来 GPU，本进程内所有后续请求都不再尝试，避免每次问答都白等一次失败。
GEN_AUTO_NUM_GPU = 999
_GEN_GPU_LOCK = threading.Lock()
_GEN_GPU_STATE = {"fallback": False}


def _gen_num_gpu_effective():
    """本次请求要发的 num_gpu；None = 不发该字段（交给 llama.cpp，纯 CPU 机上即全 CPU）。"""
    if GEN_NUM_GPU == "auto":
        with _GEN_GPU_LOCK:
            if _GEN_GPU_STATE["fallback"]:
                return None
        return GEN_AUTO_NUM_GPU
    return GEN_NUM_GPU


def _gen_gpu_fallback(reason):
    """auto 模式：把 num_gpu 永久降级为"不发该字段"。返回 True 表示这次真的落了一次
    （调用方据此决定是否重发）；非 auto 或已经落过则返回 False。"""
    if GEN_NUM_GPU != "auto":
        return False
    # ★ 只认**设备类**失败：旧判据是"任何非 200"，于是容器冷启动时模型还在后台拉取的那个
    #   404、或一次 503，就能把 GPU 机器此后所有请求永久降级为自动分层（.env 自述 65536 档
    #   要求 RAG_GEN_NUM_GPU=999，自动分层实测 47% 的层掉 CPU），日志还把它归因成
    #   "生成侧 GPU 不可用"——归因错误 + 跨轮次持久污染（无复位路径）。
    _DEVICE_HINTS = ("cuda", "out of memory", "oom", "device", "gpu", "metal", "mps", "vram", "insufficient memory")
    if not any(_k in reason.lower() for _k in _DEVICE_HINTS):
        log(f"⚠️  生成失败但不像设备问题（{reason}）：保持 num_gpu 不变，不做 GPU 回落")
        return False
    with _GEN_GPU_LOCK:
        if _GEN_GPU_STATE["fallback"]:
            return False
        _GEN_GPU_STATE["fallback"] = True
    log(f"⚠️  生成侧 GPU 不可用（{reason}）→ RAG_GEN_NUM_GPU=auto 回落为 CPU，本进程内不再尝试 GPU")
    return True


def stream_chat(messages, *, temperature=None, num_ctx=None, num_predict=None, think=None):
    """temperature/num_ctx/num_predict 缺省取全局生效值；显式传入即覆盖。

    run_turn 的同名参数就是经此生效的 —— 若它只把参数传给 plan_generation（算预算）
    而请求仍用全局值，就会出现"预算按 A 算、请求按 B 发"的静默分叉。
    调用方自定义的 chat 可调用对象必须接受 run_turn 实际传入的关键字参数
    （num_ctx / num_predict；temperature 未传时取全局生效值）。

    think=None 取全局 RAG_THINK；显式传入即覆盖。
    本函数**不设输出长度闸、也不重试**：输出与思考都不限长，窗口（num_ctx）是唯一上限，
    超窗由服务端上下文移位从最前面按 token 遗忘续写；空答/截断/报错原样交给上层。
    但设**读空闲超时**（RAG_STREAM_READ_TIMEOUT，默认 300s 无新数据 → EVENT_ERROR）与
    **整轮硬上限**（RAG_TURN_TIMEOUT，默认 1800s）：只把卡死变显式错误，不影响正常长回答。
    """
    options = _chat_options(temperature, num_ctx, num_predict)
    use_think = THINK if think is None else bool(think)
    r = None
    dropped_thinking = 0
    try:
        # try 内发起请求：requests 缺失/连接失败要走 EVENT_ERROR，而不是从生成器里
        # 抛出去砸消费侧（requests 由 _ollama_session 惰性 import，失败同样落在这层）。
        # 不发请求级 keep_alive：驻留/卸载交给 Ollama 服务端自身策略（上游默认 5m），
        # 应用层不编排生成模型内存（2026-10-24 定稿：不做互斥/自适应）。
        payload = { "model": MODEL, "messages": messages, "stream": True, "think": use_think, "options": options, }
        # (connect, read) 分开：连接 10s；读空闲 STREAM_READ_TIMEOUT（默认 300s）。
        # 模型加载/首 token 可能慢，但不应无限等待；超时由下面的 except 转成 EVENT_ERROR。
        r = _ollama_session().post( API_URL, json=payload, stream=True, timeout=(10.0, STREAM_READ_TIMEOUT), )
        if r.status_code != 200 and _gen_gpu_fallback(f"HTTP {r.status_code} {r.text[:160]!r}"):
            # auto 模式 + 本进程首次：去掉 num_gpu 重发一次。此刻**尚未产出任何 token**
            # （流都还没开始读），所以不构成"生成重试"，没有任何已生成内容被丢弃。
            # options 重新组装（而不是就地 pop）：回落状态已翻转，_chat_options 自然不再放 num_gpu。
            r.close()
            payload["options"] = _chat_options(temperature, num_ctx, num_predict)
            r = _ollama_session().post( API_URL, json=payload, stream=True, timeout=(10.0, STREAM_READ_TIMEOUT), )
        if r.status_code != 200:
            yield EVENT_ERROR, f"HTTP {r.status_code} {r.text[:300]}"
            return
        _gen_started = time.monotonic()
        for line in r.iter_lines():
            if not line:
                continue
            if TURN_TIMEOUT and (time.monotonic() - _gen_started) > TURN_TIMEOUT:
                yield EVENT_ERROR, f"生成超时：本轮超过 {TURN_TIMEOUT:.0f}s（RAG_TURN_TIMEOUT）"
                return
            chunk = json.loads(line)
            if chunk.get("error"):
                # return 而不是 continue：本函数对外承诺「error 与 done 互斥」，消费侧
                # （唯一的消费方 _run_assistant_turn 只在 done 时入历史）
                # 全靠这条。旧写法 continue 会让同一流后面的 {"done":true} 照常产出 done，
                # 于是「被错误打断的残缺答案」同时满足 done 与非空，被写进会话历史 ——
                # 而页面只渲染了错误，那段答案用户一个字符都没见过。
                yield EVENT_ERROR, f"[Ollama] {chunk['error']}"
                return
            message = chunk.get("message") or {}
            # 生产端唯一闸门：关着时事件流里不存在 EVENT_THINKING。
            if message.get("thinking"):
                if use_think:
                    yield EVENT_THINKING, message["thinking"]
                else:
                    # 开关关着却收到思维链 = 模型侧没听 think=false；按开关丢弃并告警一次。
                    dropped_thinking += 1
                    if dropped_thinking == 1:
                        log("⚠️  RAG_THINK 关闭但模型仍在返回思维链：已按开关丢弃；"
                            "请确认该模型支持 think=false，否则这部分预算白花")
            if message.get("content"):
                yield EVENT_DELTA, message["content"]
            if chunk.get("done"):
                yield EVENT_DONE, { "done_reason": chunk.get("done_reason"), "eval_count": chunk.get("eval_count"), "prompt_eval_count": chunk.get("prompt_eval_count"), }
    except Exception as exc:
        _ename = type(exc).__name__
        if _ename in ("ReadTimeout", "Timeout", "ConnectionError", "ChunkedEncodingError"):
            yield EVENT_ERROR, (f"生成超时/连接中断：等待超过 {STREAM_READ_TIMEOUT:.0f}s 无新数据"
                               f"（{_ename}；可调 RAG_STREAM_READ_TIMEOUT / 看 ollama 日志）")
        else:
            yield EVENT_ERROR, f"{_ename}: {exc}"
    finally:
        if r is not None:
            r.close()


# 内存策略（2026-10-24 定稿）：应用层不做互斥驻留 / 自适应 / 压力闸。
# 每个进程按自身策略使用内存：Ollama 由服务端 keep_alive（上游默认 5m）自行卸载；
# 检索模型首次用到后在进程内常驻；Qdrant 自管。OOM 等进程异常由容器 entrypoint 的
# 进程级巡检就地重启（不重新部署），详见 README §11.4。


# ── 通用文本谓词 ──
def _has_searchable_content(text):
    # 认 Unicode 字母与数字（CJK 的 Lo、数字的 Nd）；标点、符号、零宽等格式字符不算内容。
    # 数字必须算：年份/编号（"1998"、"300 强"、"3.5"）是合法提问，按"仅字母"拦会让它们
    # 零检索，随后无拒答链路照常生成 —— 正是 README §10 点名的"检索为空却照常作答"。
    # 不变量是"空查询与纯标点必拦"：纯数字既非空也非标点，不该被这道闸门吃掉。
    if not isinstance(text, str) or not text:
        return False
    return any(unicodedata.category(ch)[0] in ("L", "N") for ch in text)


# ── 领域与检索侧配置 ──
# 全部可配置；默认值面向通用垂直领域，核心代码不假设语料形态、文件名或来源字段。
# RAG_BOOKS_DIR 是 RAG_DATA_DIR 的遗留别名（仍生效，老部署迁移用；新配置只写 RAG_DATA_DIR）
DATA_DIR = os.getenv("RAG_DATA_DIR") or os.getenv("RAG_BOOKS_DIR") or "corpus"
DOC_GLOB = os.getenv("RAG_DOC_GLOB", "*.txt")
SOURCE_FIELD = (os.getenv("RAG_SOURCE_FIELD") or "source").strip() or "source"
SOURCE_LABEL = os.getenv("RAG_SOURCE_LABEL", "来源")
_RESERVED_SOURCE_FIELDS = {
    "chunk_id", "id", "child_text", "parent_text", "chunk_index", "total_chunks",
    "contextual_text", "parent_id", "parent_chunk_count", "build_id", "lexicon_id",
    # 见 INDEX_BUILD_ID_FIELD 一组：这两个字段名也是内部保留键，不能被来源字段占用
    "chunk_fingerprint",
    # 以下不是 payload 字段，而是**对外输出**要用的键：SOURCE_FIELD 原样写进
    # steps.context_sources 等诊断输出，同名即就地覆盖 —— score/text/context/n/
    # rerank_score/source_label/point_id 会静默变成来源字符串（响应看着正常，字段全错）。
    # source 是默认值本身，不能列入，否则启动必炸。
    "score", "text", "context", "n", "rerank_score", "source_label", "point_id",
}
if SOURCE_FIELD in _RESERVED_SOURCE_FIELDS:
    raise ValueError(
        f"RAG_SOURCE_FIELD={SOURCE_FIELD!r} 与内部字段冲突，"
        f"不能使用 {sorted(_RESERVED_SOURCE_FIELDS)}"
    )
TOP_K = env_int_min("RAG_TOP_K", 5, 1)
RERANK_TOP_K = env_int_min("RAG_RERANK_TOP_K", 20, 1)
# 重排候选池**恒等于** RAG_RERANK_TOP_K（默认 20）—— 本地与容器一字不差。
# ★ "延迟换精度"是**用户决策，不是系统自动决策**：自动按预算收缩会让同一次提问在本地与容器
#   得到不同质量的答案（容器 20→9 = 直接砍召回），而且没有任何告警。故这里只做两件事：
#     ① 观测：实测单候选耗时 × 池大小打进日志（超线时给出**显式调小**的建议，由人来决定）；
#     ② 执行用户的显式决定：只有 RAG_RERANK_BUDGET_MS > 0（默认 0 = 不启用）才按预算收缩。
# 实测（bge-reranker-v2-m3，候选=child 文本 ≈100 token，与线上 RERANK_ON=child 同口径）：
#   MPS fp16 41.9 ms/候选（20 条 ≈ 0.84s）；CPU 2 线程 fp32 872 ms/候选（20 条 ≈ 17.4s）。
RERANK_BUDGET_MS = env_int_min("RAG_RERANK_BUDGET_MS", 0, 0)       # 0 = 只观测、不收缩（默认）
RERANK_OBSERVE_MS = env_int_min("RAG_RERANK_OBSERVE_MS", 8000, 1)  # 只用于提示文案的观测线
_RERANK_MS_PER_CAND = {"mps": 41.9, "cuda": 15.0, "cpu": 872.4}
_RERANK_POOL_NOTED = False


def rerank_pool_size(pool_max=None):
    """返回 (候选池大小, 依据串)。**默认恒等于上限**，只有显式预算才收缩。

    观测行每进程只打一次（不刷屏），且只给建议、不改池大小 —— 池大小的唯一决定权在
    RAG_RERANK_TOP_K（质量/用户意图）与显式 RAG_RERANK_BUDGET_MS（用户显式要求降延迟）。
    """
    global _RERANK_POOL_NOTED
    top = RERANK_TOP_K if pool_max is None else int(pool_max)
    if RERANK_BACKEND != "torch":
        return top, f"http 重排后端：服务端算力不可知，恒保持 {top}（HTTP 超时兜底）"
    device = get_device()
    per = _RERANK_MS_PER_CAND.get(device, _RERANK_MS_PER_CAND["cpu"])
    est_ms = per * top
    if RERANK_BUDGET_MS <= 0:
        if not _RERANK_POOL_NOTED:
            _RERANK_POOL_NOTED = True
            if est_ms > RERANK_OBSERVE_MS:
                suggest = max(1, int(RERANK_OBSERVE_MS // per))
                log(
                    f"[rerank] 单候选 {per}ms × {top} = {est_ms / 1000:.1f}s，超出 {RERANK_OBSERVE_MS / 1000:.0f}s 观测线；"
                    f"若需降延迟可显式设 RAG_RERANK_TOP_K={suggest}（会降低召回，需用户决定）"
                )
            else:
                log(
                    f"[rerank] 单候选 {per}ms × {top} = {est_ms / 1000:.1f}s（device={device}），"
                    f"在 {RERANK_OBSERVE_MS / 1000:.0f}s 观测线内；候选池保持 {top}（本地与容器一致）"
                )
        return top, f"观测：{per}ms × {top} ≈ {est_ms / 1000:.1f}s（device={device}），未启用预算收缩"
    pool = max(1, min(top, int(RERANK_BUDGET_MS // per) or 1))
    return pool, f"**显式**预算 {RERANK_BUDGET_MS}ms ÷ 实测单候选 {per}ms（device={device}）"
RERANK_BATCH = env_int_min("RAG_RERANK_BATCH", 32, 1)  # padding 隔离下分数与批大小无关；候选<=20 单批一次过
RERANK_MAX_LENGTH = env_int_min("RAG_RERANK_MAX_LENGTH", 384, 1)
RERANK_ON = env_choice("RAG_RERANK_ON", "child", ("child", "parent"))
PRELOAD_MODELS = env_bool("RAG_PRELOAD", True)  # api/serve 启动时后台预热模型与词典，首查询冷加载提前到启动期
# 检索侧嵌入后端：torch=进程内（默认）；http=现成的 OpenAI 兼容嵌入服务（compose 用
# infinity 容器起 bge-base-zh-v1.5，而**不**是自建微服务）。http 只作用于检索查询侧与
# 预热：process 的分块语义定界与 index 入库固定进程内 torch（换后端不改切分/嵌入代码
# 指纹，索引无需重建）。**重排没有再拆分**：bge-reranker 没有能直接用的现成服务
# （infinity × XLM-R 有 upstream bug），恒与 Streamlit 同进程（torch）。非法取值启动即报错。
EMBED_BACKEND = env_choice("RAG_EMBED_BACKEND", "torch", ("torch", "http"))
# OpenAI 兼容 base_url：以 /v1 结尾（OpenAI SDK 惯例），客户端拼 "{url}/embeddings"。
# infinity 的 --url-prefix /v1 与之对应；若服务端无前缀则把 RAG_EMBED_URL 设为
# http://host:port（端点变 /embeddings）。RAG_EMBED_URL 的 /v1 就是契约的一部分。
EMBED_URL = (os.getenv("RAG_EMBED_URL") or "http://127.0.0.1:8001/v1").rstrip("/")
# http 端按 OpenAI /v1/embeddings 契约通信（infinity 等现成服务都实现它）；model 字段填
# 服务端注册名（infinity 的 --served-model-name，默认即模型目录 basename）。
EMBED_MODEL_NAME = (os.getenv("RAG_EMBED_MODEL_NAME") or "bge-base-zh-v1.5").strip() or "bge-base-zh-v1.5"
# 嵌入服务 HTTP 超时（秒）：指向 embed 容器。首请求含服务端冷加载，预热线程已把它提前到
# 启动期，这里取中档超时就够（冷启动失败有预热告警兜底）。
MODEL_HTTP_TIMEOUT = env_int_min("RAG_MODEL_HTTP_TIMEOUT", 60, 1)
# 重排后端与嵌入对称：torch=进程内（默认）；http=现成的 Cohere/Jina 风格 /rerank 服务，
# 与嵌入共用同一个 infinity 进程（一个进程托管两个模型、共享一份 torch 运行时；infinity
# 的 /rerank 就是"本地版 Cohere rerank"）。**重排的 http 契约比嵌入严**：响应必须一次
# 给回**全部**候选的分数、服务端不得裁剪 —— 少给一条就直接报错，因为按子集排序会静默
# 破坏"对全体候选统一打分"这条不变量。
RERANK_BACKEND = env_choice("RAG_RERANK_BACKEND", "torch", ("torch", "http"))
RERANK_URL = (os.getenv("RAG_RERANK_URL") or EMBED_URL).rstrip("/")
RERANK_MODEL_NAME = (os.getenv("RAG_RERANK_MODEL_NAME") or "bge-reranker-v2-m3").strip() or "bge-reranker-v2-m3"
# ── 驻留策略：不做应用层互斥/自适应（2026-10-24 定稿）──────────────────────────
# 应用不再编排检索/生成模型的内存装卸：没有 gate（每轮强制卸载确认）、没有水位豁免、
# 没有压力保险闸、请求也不带 keep_alive。每个进程按自身策略使用内存：
#   · Ollama：按服务端 OLLAMA_KEEP_ALIVE 自身策略（上游默认 5m）决定生成模型驻留/卸载；
#   · 检索：懒加载，首次用到后在进程内常驻（不再每轮 release）；
#   · Qdrant：自身配置管理。
# 内存不足的现实后果 = 容器内某进程被 OOM kill → entrypoint 进程级巡检发现并就地重启
# （只重启进程，不重新部署）。这是明确选择的行为；不要在本文件里再加回互斥/自适应/闸门
# （需求变更先改 README §11.4）。
# RAG_PRELOAD=0 仍表示启动不预热（懒加载，进程自身行为，与编排无关）。
# 生成模型的 GPU 层数（llama.cpp 的 -ngl，Ollama 暴露为 options.num_gpu）：
#   空    = 不发该字段，由 llama.cpp 按**当时**可用显存自动分层（改造前的默认行为）
#   0     = 全 CPU
#   999   = 全 GPU（固定档，不回落）
#   auto  = **GPU 优先、起不来回落 CPU**：先按 999 发；若首个请求在**产出任何 token 之前**
#           失败（HTTP 非 200），去掉该字段重发一次，并在**本进程内**记住、之后不再尝试 GPU。
#           这是"设备不可用"的启动级回落，不是"生成失败重试"——它发生在零 token 产出之前，
#           不会丢弃任何已生成内容（生成仍然无长度/时间闸、不重试）。同一份 .env 因此在
#           本地（Metal）与纯 CPU 容器（魔搭）都能用。
# **只改"谁来算"，不改数值**；但换分层 = 换 kernel（Metal vs CPU BLAS），浮点累加顺序不同，
# 贪心解码的长答案可能不再逐字可复现 —— 要严格复现就把档位固定住、不要每轮变。
_gpu_raw = (os.getenv("RAG_GEN_NUM_GPU") or "").strip()
GEN_NUM_GPU = None
if _gpu_raw:
    if _gpu_raw.lower() == "auto":
        GEN_NUM_GPU = "auto"
    else:
        try:
            GEN_NUM_GPU = int(_gpu_raw)
        except ValueError:
            raise ValueError(f"RAG_GEN_NUM_GPU 必须是整数或 auto（空=自动分层、0=全 CPU、999=全 GPU、auto=GPU 优先失败回落），收到 {_gpu_raw!r}") from None
        if GEN_NUM_GPU < -1:
            raise ValueError(f"RAG_GEN_NUM_GPU 不能小于 -1（-1=自动），收到 {GEN_NUM_GPU}")
CHILD_MAX_TOKENS = env_int_min("RAG_CHILD_MAX_TOKENS", 128, 1)
PARENT_MAX_TOKENS = env_int_min("RAG_PARENT_MAX_TOKENS", 512, 1)
# 以下三项旧写法是 env_int（无下限）：负的重叠/最小值能一路进到 splitter，
# 每个文档都抛 ValueError 让 process 全量失败（报错点还落在组件内部，指不到配置）。
CHUNK_OVERLAP = env_int_min("RAG_CHUNK_OVERLAP", 32, 0)
PARENT_MIN_TOKENS = env_int_min("RAG_PARENT_MIN_TOKENS", 192, 0)
# 死配置：子块最小长度由 overlap 与句子边界决定，当前**没有任何消费点**，且已从切分指纹
# payload 移除（留在指纹里只会让"改它"触发一次无意义的全量重建）。保留解析只为对 .env 里
# 的历史写入做具名校验；要真正接进子块合并逻辑时，必须同步把它放回指纹。
CHILD_MIN_TOKENS = env_int_min("RAG_CHILD_MIN_TOKENS", 32, 0)
if CHILD_MAX_TOKENS > PARENT_MAX_TOKENS:
    raise ValueError(f"RAG_CHILD_MAX_TOKENS={CHILD_MAX_TOKENS} 不能大于 RAG_PARENT_MAX_TOKENS={PARENT_MAX_TOKENS}")
if CHUNK_OVERLAP >= CHILD_MAX_TOKENS:
    raise ValueError(
        f"RAG_CHUNK_OVERLAP={CHUNK_OVERLAP} 必须小于 RAG_CHILD_MAX_TOKENS={CHILD_MAX_TOKENS}："
        f"子块切分要求 overlap < capacity，否则每篇文档都在 splitter 里抛 ValueError（process 全量失败）"
    )
# 嵌入批大小：**显式值优先**（>0）；未设/留空 = 按设备与可用内存派生（embed_batch_size()，
# 规则来自实测梯度，见该函数文档）。旧写法默认 64 —— 那是一个与设备、内存都无关的常数：
# MPS 上 64 比 16 慢 6% 且多占 1GB，CPU 上 128 会因内存压力从 5.2 掉到 3.0 块/秒。
_BATCH_RAW = (os.getenv("RAG_EMBED_BATCH_SIZE") or "").strip()
EMBED_BATCH_SIZE = env_int_min("RAG_EMBED_BATCH_SIZE", 1, 1) if _BATCH_RAW else 0
# （已删除）SEMANTIC_EMBED_DEVICE：组件化后不再有"分块期嵌入设备"这一路，全仓无引用；
# 索引/检索的设备选型仍由 INDEX_DEVICE（空=自动，见 get_device）与各后端决定。
INDEX_DEVICE = os.getenv("INDEX_DEVICE", "")
# index/sparse 的可开内存闸门（默认 0 = 不启用；魔塔 8G 档按 .env 显式开启）。阈值语义：
# 嵌入阶段的实测峰值 ~2.2-3.0G（fp16 batch 16-32），若同机还有 serve/生成模型驻留就会
# 顶穿 8G —— "重建与服务隔离"这条运维规则在这里用**可开内存**表达，而不是另起进程探测。
# fail-closed 与 index 单例锁同款：门槛设了、读数低于它 = 明确报错退出；macOS 读不到
# MemAvailable 时闸门跳过并显式说明（不静默放行成"检查过了"）。
INDEX_MIN_FREE_MB = env_int_min("RAG_INDEX_MIN_FREE_MB", 0, 0)
QDRANT_HOST = os.getenv("QDRANT_HOST") or os.getenv("RAG_QDRANT_HOST") or "localhost"
QDRANT_PORT = env_int_any(6333, "QDRANT_PORT", "RAG_QDRANT_PORT")
# docker-compose.yml 的 QDRANT__SERVICE__API_KEY 取的就是它：不接这条通路，一旦设置了
# key，应用侧每个 Qdrant 请求都会 401，且日志里没有任何地方能填 key（失效原型：静默）。
QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or os.getenv("RAG_QDRANT_API_KEY") or "").strip() or None
# Qdrant 客户端超时（秒）：普通查询走它，批量重建（sparse）在调用点给 2 倍。
# 0 之前由各调用点各自写死 60/30/120 —— 有的默认、有的没设，故障表现既不透明也不一致。
QDRANT_TIMEOUT = env_int_min("RAG_QDRANT_TIMEOUT", 60, 1)
# Qdrant 写批并发度（index）：0/未设 = 派生（见 qdrant_write_workers，规则来自实测 RTT +
# Little 定律）；显式值只用来压测/限流，正常路径不该填。sparse 的页并发共用同一个派生。
INDEX_WRITE_WORKERS = env_int_min("RAG_INDEX_WRITE_WORKERS", 0, 0)
# Streamlit 会话内 count 缓存时长（秒）：rerun 期间点数不会变，TTL 内直接复用缓存值，
# 省每次 rerun 一趟网络。index 重建生效后最迟 TTL 走完就取新值；0 = 禁用缓存。
COLLECTION_COUNT_TTL_S = env_int_min("RAG_COLLECTION_COUNT_TTL_S", 300, 0)
COLLECTION_BASE = os.getenv("RAG_COLLECTION_BASE", "rag_documents_v1")
COLLECTION_NAME = COLLECTION_BASE
COLLECTION_ALIAS = os.getenv("RAG_COLLECTION_ALIAS", "rag_current")
USE_COLLECTION_ALIAS = env_bool("RAG_USE_COLLECTION_ALIAS", True)
EMBED_MODEL_PATH = os.getenv("RAG_EMBED_MODEL_PATH", "./models/bge-base-zh-v1.5")
RERANK_MODEL_PATH = os.getenv("RAG_RERANK_MODEL_PATH", "./models/bge-reranker-base")
# 创空间 entrypoint 把嵌入/重排权重下载放后台、并先启动服务：首查询会撞上下载窗口，
# 旧行为直接抛"模型目录缺失或为空"。entrypoint 在真实下载时导出 RAG_WEIGHTS_BACKGROUND=1，
# 加载侧据此轮询等待（RAG_MODEL_WAIT_SECS，默认 900s），下载明确失败才提前放弃；
# 本地/离线命令不置位，缺模型依旧立即报错，不长等。
WEIGHTS_BACKGROUND = env_bool("RAG_WEIGHTS_BACKGROUND", False)
MODEL_WAIT_SECS = env_int_min("RAG_MODEL_WAIT_SECS", 900, 0)
_ARTIFACTS_DIR = os.getenv("RAG_ARTIFACTS_DIR", "./artifacts")
CHUNKS_JSON = os.path.join(_ARTIFACTS_DIR, "chunks.json")


CHUNKS_META_JSON = os.path.splitext(CHUNKS_JSON)[0] + ".meta.json"
CHUNKS_META_VERSION = 1
EMBED_CACHE_DIR = os.path.join(_ARTIFACTS_DIR, "embeddings")
EMBED_CACHE_ENABLED = env_bool("RAG_EMBED_CACHE", True)
DOC_CHUNK_DIR = os.path.join(_ARTIFACTS_DIR, "docs")  # 单文档分块缓存：只重分块增/改的文档
LEXICON_DIR = os.getenv("RAG_LEXICON_DIR", "./lexicon")
WRITE_BATCH = 1000
INDEX_BUILD_ID_FIELD = "build_id"
LEXICON_ID_FIELD = "lexicon_id"
CHUNK_FINGERPRINT_FIELD = "chunk_fingerprint"
_PAYLOAD_SPEC = ( ("chunk_id", "id", ""), ("child_text", "child_text", ""), ("parent_text", "parent_text", ""), (SOURCE_FIELD, "source", ""), ("chunk_index", "chunk_index", 0), ("total_chunks", "total_chunks", 0), ("contextual_text", "contextual_text", ""), ("parent_id", "parent_id", ""), ("parent_chunk_count", "parent_chunk_count", 0), )
# chunks.json 的记录字段：除来源字段用 payload 键（=RAG_SOURCE_FIELD）外，其余都用规范键 k。
# 不能直接用 _PAYLOAD_SPEC 的 f：chunk_id 的 payload 键是 "chunk_id"、记录键却是 "id"。
_RECORD_CORE_FIELDS = {k for _f, k, _d in _PAYLOAD_SPEC if k != "source"}
_RECORD_FIELDS = _RECORD_CORE_FIELDS | {SOURCE_FIELD}
# ★ 来源字段的**历史别名键**只有这一份定义：读取（_source_of）、检索过滤
# （_source_filter）、比较器放行（_ALLOWED_CHUNK_FIELDS/_SOURCE_ALIASES）全部派生自它。
# 旧写法是三份互不派生的字面量 —— 未来改一处漏两处，来源过滤会恒空命中且不报错（静默）。
SOURCE_PAYLOAD_FIELDS = tuple(dict.fromkeys((SOURCE_FIELD, "source", "book")))

# ── 单轮效益台账（可选诊断，默认关闭）──────────────────────────────────────
# 每轮一行 JSON（固定字段、追加式），摘要工具 tools/turn_stats.py。默认不开：项目不做
# 基准测量；需要排查某次/某档性能时临时把 RAG_TURN_STATS=1，看完即关。
#   · prompt_tokens_est = 装配侧估算的整条 prompt token（message_tokens）
#   · prompt_eval_count = 服务端实际 prefill 的 token 规模（Ollama done 事件给值）
# 两者之比 ≈1 → 只有新增段被 prefill；≫1 → 整段重算。
TURN_STATS_ENABLED = env_bool("RAG_TURN_STATS", False)
TURN_STATS_FILE = (os.getenv("RAG_TURN_STATS_FILE") or "").strip() or os.path.join(_ARTIFACTS_DIR, "turns.jsonl")


def record_turn_stats(fields: dict):
    """追加一条单轮台账。写失败只告警不抛 —— 故障不能把台账升级成故障源（诊断路径
    不许崩，与日志 handler 的兜底同一原则）。"""
    if not TURN_STATS_ENABLED:
        return
    try:
        parent = os.path.dirname(TURN_STATS_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        row = {"ts": round(time.time(), 3)}
        row.update(fields)
        with open(TURN_STATS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:
        log(f"⚠️  单轮台账写入失败（不影响本轮）: {type(exc).__name__}: {exc}")


def _source_of(record):
    """读来源字段，兼容配置字段与旧 book 字段。"""
    if not isinstance(record, dict):
        return ""
    for key in SOURCE_PAYLOAD_FIELDS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _chunk_value(chunk, key, default=""):
    """_PAYLOAD_SPEC 字段读取；source 兼容旧产物的 book 键。"""
    if key == "source":
        return _source_of(chunk) or default
    return chunk.get(key, default)


def chunk_payload(chunk, build_id, lexicon_id, chunk_fingerprint):
    payload = {f: _chunk_value(chunk, k, d) for f, k, d in _PAYLOAD_SPEC}
    payload[INDEX_BUILD_ID_FIELD] = build_id
    payload[LEXICON_ID_FIELD] = lexicon_id
    # 记录本索引是**哪一份分块产物**建起来的。检索侧拿它与当前切分指纹比对，等价于
    # 词表指纹那道闸门：没有它，process 之后的旧 chunks/Qdrant 会被 serve/api 静默
    # 继续服务（stages 报过期，但没人看），用户拿到的是无声的旧知识库。必填：不填 =
    # 新档没建指纹却能静默入库（失效原型：静默）。唯一调用点 build_index 已传。
    payload[CHUNK_FINGERPRINT_FIELD] = chunk_fingerprint
    return payload
RRF_K = env_int_min("RAG_RRF_K", 60, 0)
RRF_DENSE_WEIGHT = env_float_min("RAG_RRF_DENSE_WEIGHT", 1.0, 0.0)
RRF_SPARSE_WEIGHT = env_float_min("RAG_RRF_SPARSE_WEIGHT", 1.0, 0.0)
if not RRF_DENSE_WEIGHT and not RRF_SPARSE_WEIGHT:
    raise ValueError("RAG_RRF_DENSE_WEIGHT 与 RAG_RRF_SPARSE_WEIGHT 不能同时为 0：两条召回通道都退出等于检索恒为空。")
_RECALL_LIMIT_RAW = env_int("RAG_RECALL_LIMIT", 0)
if _RECALL_LIMIT_RAW < 0:
    raise ValueError(f"环境变量 RAG_RECALL_LIMIT={_RECALL_LIMIT_RAW!r} 不能为负数（0 表示跟随 RAG_RERANK_TOP_K，当前 {RERANK_TOP_K}）")
RECALL_LIMIT = _RECALL_LIMIT_RAW or RERANK_TOP_K
DEDUP_BY_PARENT = env_bool("RAG_DEDUP_BY_PARENT", True)
# 原值"为这个句子生成表示以用于检索相关文章："（bge 官方查询指令）；按全裸 prompt 配置置空走裸查询嵌入。
BGE_QUERY_INSTRUCTION = os.getenv("RAG_EMBED_QUERY_INSTRUCTION", "")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
BM25_MODEL_NAME = "qdrant/bm25"


def bm25_options():
    """稀疏通道的 BM25 配置。**分词与停用词都交给 Qdrant**（2026-10-14 组件化）。

    · tokenizer 由 "word" 改为 "multilingual"。实测（同一批文档 + 查询"中国革命的对象是谁"）：
      "word" / "whitespace" / "prefix" 对中文整句**零召回**（没空格 → 整句 1 个 token），
      "multilingual" 正确命中（相关 5.069 vs 无关 0.221）。故 Python 侧的 jieba 预分词整层删除。
    · stopwords 用 Qdrant 内置中文表（"chinese"）；关掉时用 {} —— 字符串 "none" 不是合法值
      （实测 400 Bad Request），"没有停用词"要用空对象表达。
    """
    return {
        "tokenizer": "multilingual",
        "stemmer": {"type": "none"},
        "stopwords": "chinese" if STOPWORDS_ENABLED else {},
        "ascii_folding": True,
    }


def history_to_messages(history):
    """会话 → 模型消息：只取 role/content；页面旁路字段（thinking/trace）永不进入生成上下文。"""
    out = []
    for m in history or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not content or not str(content).strip():
            continue
        # 非字符串统一 str()：原样 append 会把 int/对象等非 str 内容直接送进 JSON 请求体，
        # Ollama 侧要么 400 要么静默 ToString，两种都与"发出去的是这段原文"的预期不符。
        out.append({"role": m.get("role") or "user", "content": str(content)})
    return out


def strip_current_turn(history, query):
    msgs = list(history or [])
    if not msgs or not query:
        return msgs, False
    last = msgs[-1]
    if (isinstance(last, dict) and last.get("role") == "user" and (last.get("content") or "").strip() == query.strip()):
        return msgs[:-1], True
    return msgs, False


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _float16_ok(device):
    return device in ("mps", "cuda")


# ── 模型加载与向量化 ──
def _model_dir_ready(path):
    return bool(path) and os.path.isdir(path) and bool(os.listdir(path))


def _wait_for_weights(path):
    """后台下载权重时轮询等待目录就绪；返回 True=可用。entrypoint 用 目录.part → 清单
    齐备后原子改名，故最终目录一出现即完整；下载明确失败会写 /tmp/model-download.status
    =failed，据此提前放弃，不白等到超时。"""
    _deadline = time.monotonic() + MODEL_WAIT_SECS
    _told = False
    while True:
        if _model_dir_ready(path):
            return True
        try:
            with open("/tmp/model-download.status", encoding="utf-8") as f:
                if f.read().strip() == "failed":
                    return False
        except OSError:
            pass
        if time.monotonic() >= _deadline:
            return False
        if not _told:
            log(f"⏳ 权重尚未就绪（{path}），等待 entrypoint 后台下载完成（最多 {MODEL_WAIT_SECS}s）…")
            _told = True
        time.sleep(2)


def _require_model_dir(path, what, hint):
    if _model_dir_ready(path):
        return path
    if WEIGHTS_BACKGROUND and MODEL_WAIT_SECS > 0 and _wait_for_weights(path):
        return path
    raise RuntimeError(f"{what}模型目录缺失或为空: {path!r} —— {hint}")


def load_embedding_model(device=None, model_path=None):
    """嵌入模型加载：**权重精度按设备派生**（与 load_reranker_model 完全同款）。

    GPU（mps/cuda）= fp16 权重：内存 2166MiB → 1083MiB，吞吐实测 28.2 块/s vs 纯 autocast
    4.63 块/s（32 块探针，量化对拍 agent 实测；余弦 0.999998、top-10 近邻 recall 0.998）。
    CPU = fp32（fp16 在 CPU 上不加速反而更慢），故容器走 CPU 时精度与语义同改造前。

    注意：本函数在 _embed_code_fingerprint 的源码清单里 —— 改它会让稠密缓存文件名里的
    code_key 变化，旧向量**自动作废、不会静默复用**（这是有意的：换精度必须全量重算）。
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = device or get_device()
    dtype = torch.float16 if _float16_ok(device) else torch.float32
    model_path = _require_model_dir(
        model_path or EMBED_MODEL_PATH, "嵌入",
        "创空间看运行日志的 [entrypoint] 权重下载行（下载完成后再试）；"
        "本地确认 RAG_EMBED_MODEL_PATH 指向可用的模型目录",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()
    return tokenizer, model, device


def load_reranker_model(device=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = device or get_device()
    dtype = torch.float16 if _float16_ok(device) else torch.float32
    model_path = _require_model_dir(
        RERANK_MODEL_PATH, "重排",
        "创空间看运行日志的 [entrypoint] 权重下载行（下载完成后再试）；"
        "本地确认 RAG_RERANK_MODEL_PATH 指向可用的模型目录",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained( model_path, torch_dtype=dtype ).to(device)
    model.eval()
    return tokenizer, model, device


def load_rerank_tokenizer_only():
    """http 重排后端下只加载 tokenizer（几十 MB），不加载权重。

    steps 的 token 预算列需要它；拿不到就返回 None，由调用点把这几列留空 —— 不编造数字，
    也不因为"算不出 trace 的辅助列"而让检索失败。
    """
    try:
        from transformers import AutoTokenizer
        if not _model_dir_ready(RERANK_MODEL_PATH):
            return None
        return AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
    except Exception as exc:
        log(f"⚠️  http 重排：本地重排 tokenizer 加载失败（steps 的 token 预算列将留空）: {type(exc).__name__}: {exc}")
        return None


def _embed_batch(texts, tokenizer, model, device, autocast=False):
    import torch
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        if autocast:
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
    cls = outputs.last_hidden_state[:, 0].float()
    return torch.nn.functional.normalize(cls, p=2, dim=1)


def embed(texts, tokenizer, model, device, is_query=False, autocast=False):
    if is_query:
        texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
    return _embed_batch(texts, tokenizer, model, device, autocast=autocast).cpu().numpy()


# ── 分块：归一、分句、父子打包 ──
# 段落硬边界判据：只认**真正空行**（两个连续换行，允许 \r）。句内单个 \n 是软折行，
# 属合法排版；而"行内只含空格/TAB 的空行"（\n[ \t]*\n）sentencex 并不当边界 ——
# 旧正则把它也算作硬边界，于是"空行带尾随空格、空行后紧跟闭引号（”）"这类合法 txt
# 会在这里直接抛错，把 process 整体打崩（与 D1 同类：报错还把原因误指成 sentencex 变了）。
NORMALIZE_QUOTES = env_bool("NORMALIZE_QUOTES", True)


# 文本归一化组件（2026-10-14 组件化）：
#   · ftfy —— 修**编码层**的坏：mojibake（"ï¿½"这类）、错编码的引号/连字。
#     只用它修编码，其余全部关掉：ftfy 默认还会 uncurl_quotes（把弯引号拉直，直接
#     与 fix_quotes 打架）、fix_character_width（改全角宽度）、fix_line_breaks（动换行，
#     与"不归一 CRLF"的决定冲突）。判据是"一个功能只能有一个实现"。
#   · OpenCC —— 繁简/异体归一（t2s）。简化字语料上是恒等变换；作用是让**同一实体的
#     两种写法**在稀疏与稠密两侧不再分叉。置空则跳过转换。
FTFY_FIX = { "fix_encoding": True, "unescape_html": False, "remove_terminal_escapes": False, "fix_latin_ligatures": False, "fix_character_width": False, "uncurl_quotes": False, "fix_line_breaks": False, "fix_surrogates": False, "remove_control_chars": False, }
TEXT_OPENCC = (os.getenv("RAG_TEXT_OPENCC") or "t2s").strip()
_OPENCC_INSTANCE = None
_OPENCC_LOCK = threading.Lock()


def _opencc():
    """惰性取 OpenCC 转换器（首次调用才 import，符合"第三方依赖全部惰性"）。

    双检锁：预热线程与首个查询并发归一时，旧写法会各建一份转换器（一份白占内存一份
    被 GC，词典加载白付一趟）。与 lexicon 单例双检锁同一纪律。
    """
    global _OPENCC_INSTANCE
    instance = _OPENCC_INSTANCE
    if instance is None:
        with _OPENCC_LOCK:
            if _OPENCC_INSTANCE is None:
                from opencc import OpenCC
                _OPENCC_INSTANCE = OpenCC(TEXT_OPENCC)
        instance = _OPENCC_INSTANCE
    return instance


def normalize_text(text):
    """归一化组件入口：ftfy 修编码 → OpenCC 繁简归一。"""
    import ftfy
    text = ftfy.fix_text(text, **FTFY_FIX)
    if TEXT_OPENCC:
        text = _opencc().convert(text)
    return text


def fix_quotes(text):
    """把直引号 " 归一到弯引号：**单遍二值状态机，所有引号字符共同参与状态**。

    判据只有一条：**当前位置是否处在"已开未闭"的引号里** —— 在内则为闭引号，
    在外则为开引号。“ ” 与已判定的 " 一起维护这个状态。

    为什么不能靠"看前一个字符"（旧实现的做法）：旧规则把"直引号左侧紧邻
    句读/空白/括号/右引号"当作**必为开引号**的硬信号（`_OPEN_QUOTE_CTX`）。
    该前提在真实语料上常常是**反的**，实测（某真实语料，2426 个直引号）：
        `："` 出现   0 次   —— 直引号从不作开引号（开引号一律写 “，5777 次）
        `？"` 出现 915 次   —— 句末标点后的直引号是**闭**引号
        `！"` 出现 402 次、`。"` 45 次
    即"紧跟句读"恰恰是**闭引号**的特征。按旧规则，2426 个直引号里有 2094 个
    被判成开引号，其中 1734 个紧跟中文标点 —— 结果把该文本的引号不配平从
    1600 个未配对 “ 推到 3362 个（`“` 7896 / `”` 4534），把该函数本想修的
    引号错位**做成了一倍**（真实语料里的坏样本）。

    为什么不能只靠 `_paired_curly_positions`（只让成对弯引号参与状态）：
    它把 “ 与 ” 配对，**不承认直引号也是一个闭引号**。混合型文本里大量 “ 的闭合者
    就是 "，于是它们被判为"孤儿"、不参与状态 —— 复算显示只有 332 个直引号
    在状态内（即能被判为闭引号），剩下 ~1700 个仍然是错的。

    为什么"孤儿引号"不再是问题：孤儿的统计口径本身就来自旧算法的假设。
    在"所有引号一起维护状态"的模型下，一个 “ 由谁闭合不再需要预先知道 ——
    每个引号字符都只是把状态翻一次，局部结构因此始终成立。

    纯直引号型文本（`他说"你好"`）的行为与旧规则 2（成对交替）完全一致，
    混合型文本也正确；`NORMALIZE_QUOTES=0` 是逃生开关。
    """
    if '"' not in text:
        return text

    out = []
    inside = False
    for ch in text:
        if ch == '"':
            if inside:
                out.append("\u201d")
                inside = False
            else:
                out.append("\u201c")
                inside = True
        else:
            out.append(ch)
            # 弯引号同样参与状态：只承认直引号的"开"、不承认它的"闭"，
            # 正是上一版失效的根源。
            if ch == "\u201c":
                inside = True
            elif ch == "\u201d":
                inside = False
    return "".join(out)


# （已删除）分块期嵌入单例 _get_embed_model/_EMBED_MODEL/_EMBED_TOKENIZER：2026-10-14
# 组件化后语义定界改由 semantic-text-splitter 按结构递归切分，process 不再做逐句嵌入，
# 这三个名字全仓已无任何调用点；留着只会让人以为"分块还要加载嵌入模型"。


def _mps_empty_cache():
    """每批后归还 MPS 缓存池；失败只告警（不影响结果）。"""
    import torch
    try:
        torch.mps.empty_cache()
    except Exception as exc:
        log(f"    ⚠️  mps.empty_cache() 失败（忽略，不影响结果）: {type(exc).__name__}: {exc}")


def _token_counter(tokenizer):
    """容量回调：把"块 ≤ N"表达成 **token** 上限（与改造前同一把尺子）。"""
    def _count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    return _count


def _merge_tiny_parents(parents, tokenizer):
    """把过短的父块并入相邻父块 —— 组件不做合并，这是旧 _merge_small_texts 的等价物。

    只做**拼接**，且只在"合并后仍 <= PARENT_MAX_TOKENS"时合并：上限是硬不变量，块太短只是
    质量问题（短父块没有上下文可给，9.8% 的父块曾短于 40 token）。trim=False 的块自带边界
    空白，直接相加即逐字无损。
    """
    out = list(parents)
    if not out:
        return out
    count = _token_counter(tokenizer)
    i = 0
    while i < len(out) - 1:
        if count(out[i]) < PARENT_MIN_TOKENS and count(out[i] + out[i + 1]) <= PARENT_MAX_TOKENS:
            out[i] = out[i] + out[i + 1]
            del out[i + 1]
        else:
            i += 1
    return out


def _hierarchical_split(text, tokenizer):
    """父子分块：**机械部分交给 semantic-text-splitter**，父子契约不变。

    2026-10-14 组件化。改造前这里是"逐句嵌入 + 余弦阈值定界（_semantic_split）+ 平衡打包
    + 小块合并"的一大套自研逻辑（process 的耗时大头也在那批逐句嵌入上）。现在：

      · 父块：capacity=PARENT_MAX_TOKENS，按段落/句子结构递归切分
      · 子块：在每个父块内部再切，capacity=CHILD_MAX_TOKENS、overlap=CHUNK_OVERLAP
      · 容量用**本仓库的 tokenizer** 度量（from_callback），故"父≤512 / 子≤128 token"
        与改造前同源同义；tokenizer 参数因此仍是必需的（旧实现只把它往下游传）
      · trim=False —— 实测这是"**切分逐字无损**"不变量成立的必要条件：
        trim=True 会吃掉块边界空白，join(chunks) != text

    返回 [(child_text, parent_text), ...]。子块恒为父块子串（组件在父块**内部**再切，
    天然成立）；纯空白块照旧原样产出，由调用方（build_chunks）统一丢弃并留痕。
    """
    from semantic_text_splitter import TextSplitter
    count = _token_counter(tokenizer)
    # **只给上限**：实测（40 篇）区间容量在这个组件里不是"合并小块"的意思 —— 父块区间
    # (192,512) 会把父块从 411 涨到 672、短父块从 30 涨到 110，子块区间还会让子块翻倍
    # （4240 vs 2014）；子块区间更会直接撞 ValueError(overlap >= 容量下界) 让 process 全量失败。
    parent_splitter = TextSplitter.from_callback(count, PARENT_MAX_TOKENS, trim=False)
    child_splitter = TextSplitter.from_callback(count, CHILD_MAX_TOKENS, overlap=CHUNK_OVERLAP, trim=False)
    for parent_text in _merge_tiny_parents(parent_splitter.chunks(text), tokenizer):
        for child_text in child_splitter.chunks(parent_text):
            yield (child_text, parent_text)


def read_book_text(filepath):
    """严格解码文档文本；遇到非法 UTF-8 字节时精确报错，并降级为 U+FFFD。

    不用 errors="ignore" 静默丢弃坏字节（实测多份文档各有 2 处），
    否则损坏位置和内容都无从追查。这里坏字节被替换成 U+FFFD（可检测、可回溯）。

    不做 CRLF 归一：实测 sentencex 自身就把 \r\n\r\n 当句边界且输出不含 \r
    （含 8436 个 CR 的文档同样如此），全部文档的句子序列与归一前逐元素完全相同。

    最后做引号归一（NORMALIZE_QUOTES），见 fix_quotes。
    """
    with open(filepath, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        log(
            f"    ⚠️  {os.path.basename(filepath)} 非法 UTF-8 字节: "
            f"偏移 {e.start}-{e.end}，字节 {raw[e.start:e.end]!r}（已替换为 U+FFFD）"
        )
        text = raw.decode("utf-8", errors="replace")
    text = normalize_text(text)
    if NORMALIZE_QUOTES:
        text = fix_quotes(text)
    return text


read_document_text = read_book_text  # 通用别名；保留 read_book_text 以兼容分块指纹
# 指纹单元 = 真正决定切分结果的自研代码。引号归一、文档读取（含 ftfy+OpenCC 归一化）、
# 父子分块（semantic-text-splitter 的调用参数与容量回调）、token 计数器与短父块合并。
# 分块组件的**版本**另在 config 指纹里（splitter 项）。
# normalize_text/_opencc 必须在这里：read_book_text 只是**调用**它们，改归一化实现
# （换清洗项、换 OpenCC 配置语义）不改 read_book_text 的源码 —— 漏掉就是"改了归一化、
# 指纹不动、旧分块静默沿用"（fix_quotes 早已在列，两者是同一层的实现）。
_CHUNKING_CODE_UNITS = (
    "fix_quotes",
    "read_book_text",
    "normalize_text",
    "_opencc",
    "_hierarchical_split",
    "_token_counter",
    "_merge_tiny_parents",
)


def _package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def _chunking_code_fingerprint():
    import inspect
    parts = []
    for name in _CHUNKING_CODE_UNITS:
        fn = globals().get(name)
        try:
            parts.append(f"{name}:{inspect.getsource(fn)}")  # type: ignore[arg-type]
        except (OSError, TypeError):
            parts.append(f"{name}:<unavailable>")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _chunking_config_fingerprint():
    # embed_model_content：同路径换权重必须触发重分块/重建缓存；目录不可用时它是哨兵，
    # 这时**单独报"嵌入模型目录缺失"**，而不是让指纹比对去打"请重跑 process+index"
    # （那条路正需要这个目录，报错会把排查方向指反）。
    _model_content = _model_content_fingerprint()
    if _model_content == _MODEL_CONTENT_UNKNOWN:
        raise RuntimeError(
            f"嵌入模型目录缺失或为空: {EMBED_MODEL_PATH!r} —— 无法计算切分指纹，"
            f"也就无法判断分块/索引是否新鲜（重跑 process 与 index 都需要这个目录，"
            f"先恢复权重：本地看 RAG_EMBED_MODEL_PATH，创空间看运行日志的权重下载行）"
        )
    payload = { "child_max_tokens": CHILD_MAX_TOKENS, "parent_max_tokens": PARENT_MAX_TOKENS, "parent_min_tokens": PARENT_MIN_TOKENS, "chunk_overlap": CHUNK_OVERLAP, "normalize_quotes": NORMALIZE_QUOTES, "text_opencc": TEXT_OPENCC, "ftfy_fix": json.dumps(FTFY_FIX, sort_keys=True), "embed_model": EMBED_MODEL_PATH, "embed_model_content": _model_content,         # 分块组件版本：升级它会改变切分边界，漏掉就会让"产物过期"不被发现（与 sentencex 同款理由）
        "splitter": _package_version("semantic-text-splitter"), "chunking_code": _chunking_code_fingerprint(), }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], payload


def _chunking_fingerprint(sources, config_fp=None):
    """**存盘与比对共用的唯一切分指纹** = 配置指纹 + 源文本 sha 指纹。

    为什么必须把源文本并进来：索引 payload 里的 chunk_fingerprint 是检索侧 fail-closed
    闸门的唯一判据（_check_chunk_freshness）。只算配置时，"改了语料 + 只跑 process"会让
    索引里的旧指纹与"当前"仍然相等 —— 闸门恒等放行，serve 静默继续服务旧知识库，只有
    stages 能发现（失效原型：静默 + 判据恒真）。两侧都调这一个函数，口径不可能分叉。
    """
    if config_fp is None:
        config_fp = _chunking_config_fingerprint()[0]
    blob = f"{config_fp}|{json.dumps(sources, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _sources_fingerprint(paths):
    """每份源文档的 sha 指纹（1MB 流式读：源文件很大时不必一次性载入内存）。

    只供"语料有没有变"判断，语义不变：流式分块哈希与整文件一次哈希结果一致。
    """
    out = {}
    for path in paths:
        with open(path, "rb") as f:
            h = hashlib.sha256()
            while True:
                block = f.read(1024 * 1024)
                if not block:
                    break
                h.update(block)
            out[os.path.relpath(path, DATA_DIR)] = h.hexdigest()[:16]
    return out


def _document_paths():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"找不到数据目录: {DATA_DIR}")
    paths = sorted(
        p for p in glob.glob(os.path.join(DATA_DIR, DOC_GLOB), recursive=True)
        if os.path.isfile(p)
    )
    if not paths and not glob.has_magic(DOC_GLOB):
        # 区分两种"一个都没匹配"：glob 里**没有通配符**等于写死了文件名，匹配不到就是配置
        # 指错了（保留 raise）；目录存在、通配符没命中则是**合法边界** —— 语料被清空，
        # 必须返回空列表走 0 文档分支，否则空语料让 process/index 双双失败、知识库永远
        # 清不掉（旧写法一律 raise：把"清空"这个正常操作变成不可达状态）。
        raise FileNotFoundError(
            f"{DATA_DIR} 中没有匹配 {DOC_GLOB!r} 的文档"
            f"（该 glob 不含通配符，等于指定了单个文件名 —— 检查 RAG_DOC_GLOB 与 RAG_DATA_DIR）"
        )
    return paths


_DOC_NAME_CACHE: dict = {}


def _document_names(paths):
    """{绝对路径: source 名}（唯一化表）：**完整相对路径全量参与，只在冲突时加短哈希**。

    候选名 = 相对路径去扩展名 + 路径分隔符编码成 "__"，与旧产物逐字相同 —— 未冲突时
    source 名、point id、文档缓存名都不变（换名会让重建后的检索基线不可比）。旧实现只对
    "名字里含 __"加哈希，于是**同 stem 不同扩展**（a.txt / a.md，放宽 RAG_DOC_GLOB 后很
    常见）派生同一个 source/id/parent_id：两篇文档在 payload 与文档缓存里互相覆盖，而且
    因名字相同、判据恒真，每次 process 都要重算这两篇（失效原型：多份 + 静默覆盖）。
    冲突组内一律追加相对路径短哈希；含 "__" 的候选（路径分隔符编码与文件名撞名）照旧加。
    """
    key = (DATA_DIR, DOC_GLOB, tuple(paths))
    names = _DOC_NAME_CACHE.get(key)
    if names is None:
        rels = {p: os.path.relpath(p, DATA_DIR) for p in paths}
        groups: dict = defaultdict(list)
        for p, rel in rels.items():
            groups[os.path.splitext(rel)[0].replace(os.sep, "__")].append(p)
        names = {}
        for candidate, members in groups.items():
            if len(members) == 1 and "__" not in candidate:
                names[members[0]] = candidate
                continue
            for p in members:
                names[p] = f"{candidate}_{hashlib.sha256(rels[p].encode('utf-8')).hexdigest()[:16]}"
        _DOC_NAME_CACHE.clear()   # 表很小；只留最近一次（换 DATA_DIR/GLOB 时不残留旧映射）
        _DOC_NAME_CACHE[key] = names
    return names


def _doc_cache_path(source_name):
    return os.path.join(DOC_CHUNK_DIR, source_name + ".json")


def _load_doc_chunks(source_name, sha, fingerprint):
    """单文档分块缓存：内容 sha 与切分指纹一致才复用；否则重算（增删改增量的关键）。"""
    path = _doc_cache_path(source_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("source") != source_name:
            return None
        if cached.get("sha256") != sha or cached.get("chunking_fingerprint") != fingerprint:
            return None
        chunks = cached.get("chunks")
        if not isinstance(chunks, list):
            return None
        if chunks and set(chunks[0]) != _RECORD_FIELDS:
            return None
        return chunks
    except Exception as exc:
        log(f"    ⚠️  文档缓存不可用，将重算 {source_name}: {type(exc).__name__}: {exc}")
        return None


def _save_doc_chunks(source_name, sha, fingerprint, chunks):
    try:
        os.makedirs(DOC_CHUNK_DIR, exist_ok=True)
        path = _doc_cache_path(source_name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump( { "source": source_name, "sha256": sha, "chunking_fingerprint": fingerprint, "chunks": chunks }, f, ensure_ascii=False )
        os.replace(tmp, path)
    except Exception as exc:
        log(f"    ⚠️  文档缓存写入失败（本次分块不受影响）: {type(exc).__name__}: {exc}")
        try:
            os.remove(tmp)  # 失败留下的半截 tmp 由 _gc_doc_chunks 兜底，这里先就地清
        except (OSError, UnboundLocalError):
            pass


def _gc_doc_chunks(keep_names):
    """删除已不存在文档的分块缓存（增删改里的「删」）；顺带清掉中断遗留的 .tmp。
    失败不影响分块。"""
    try:
        if not os.path.isdir(DOC_CHUNK_DIR):
            return
        removed = 0
        tmp_removed = 0
        for name in os.listdir(DOC_CHUNK_DIR):
            path = os.path.join(DOC_CHUNK_DIR, name)
            if name.endswith(".json.tmp"):
                # 写入中断遗留的半截文件：不属于任何文档缓存，旧逻辑匹配不到后缀永不清理
                os.remove(path)
                tmp_removed += 1
            elif name.endswith(".json") and name[:-5] not in keep_names:
                os.remove(path)
                removed += 1
        if removed:
            log(f"清理 {removed} 个已删除文档的分块缓存")
        if tmp_removed:
            log(f"清理 {tmp_removed} 个中断遗留的分块缓存临时文件")
    except Exception as exc:
        log(f"    ⚠️  文档缓存清理失败（不影响本次分块）: {type(exc).__name__}: {exc}")


# ── 分块产物与门禁 ──
# 分块并行：**按文档分发、按原顺序收集**。每篇文档互相独立（读自己的文件、切自己的文本、
# 写自己的 artifacts/docs/<name>.json 缓存），串行的唯一理由只是历史上只有一条 for 循环。
# 主进程只做三件事：分发、按下标归位、按原顺序汇总 —— 故产物与串行逐字节相同。
_CHUNK_WORKER: dict = {}
_ORIGINAL_LOG = log  # 分块期 log 会被临时接管，这里留一份原件（接管函数用它兜底）


def _chunk_capture_log(*parts):
    """分块期的 log 落进缓冲区。worker 直接往 stdout/rag.log 写会让并行日志交错、顺序与
    串行不一致；统一回主进程按文档顺序打印（顺序、内容与串行逐行相同）。"""
    sink = _CHUNK_WORKER.get("messages")
    if sink is None:
        _ORIGINAL_LOG(*parts)
    else:
        sink.append(" ".join(str(part) for part in parts))


def _chunk_worker_tokenizer():
    """worker 进程内的 tokenizer 单例（懒加载）：全部命中缓存时一个 token 都不用分。"""
    tok = _CHUNK_WORKER.get("tokenizer")
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
        _CHUNK_WORKER["tokenizer"] = tok
    return tok


def _chunk_one_document(job):
    """单文档分块（串行/并行共用**同一份实现**）：命中缓存即返回缓存块，缺失才分块并
    **自己写自己那份缓存**（各自写各自的文件，主进程只汇总）。

    返回 (文档下标, 块列表或 None, 是否命中缓存, 错误串或 None, 日志行列表)。
    """
    global log
    i, filepath, source_name, sha, fingerprint, total = job
    messages: list = []
    prev_log = log
    log = _chunk_capture_log
    _CHUNK_WORKER["messages"] = messages
    try:
        cached = _load_doc_chunks(source_name, sha, fingerprint)
        if cached is not None:
            messages.append(f"[{i + 1}/{total}] 复用未变文档: {source_name}（{len(cached)} 块）")
            return i, cached, True, None, messages
        messages.append(f"[{i + 1}/{total}] 处理: {source_name} ...")
        text = read_document_text(filepath)
        pairs = []
        for child_text, parent_text in _hierarchical_split(text, _chunk_worker_tokenizer()):
            pairs.append((child_text, parent_text))
        kept = [(c, p) for c, p in pairs if c.strip()]
        dropped = len(pairs) - len(kept)
        if dropped:
            messages.append(f"    ⚠️  丢弃 {dropped} 个纯空白块（不含任何文字）")
        if not kept:
            # 空文档（0 字节/纯空白）是**合法边界**，不是本轮失败：按 0 块正常结算并留痕。
            # 旧写法在此 raise -> failures 短路 -> 整轮拒绝写 chunks.json：一篇空文件就能让
            # 其余文档的增改全都不入库，且 process 永远退出 1（卡死式静默）。
            messages.append(f"    ⚠️  文档 {source_name} 分块后无有效内容（0 块）：按空文档结算，不影响其余文档")
            _save_doc_chunks(source_name, sha, fingerprint, [])
            return i, [], False, None, messages
        # pairs 是整篇的 (child, parent) 文本对，kept 是它的子集 —— 两者并存就是
        # 这篇文档的文本在峰值里翻倍。kept 起不再需要 pairs。
        del pairs
        parent_ids: dict = {}
        for _, parent_text in kept:
            if parent_text not in parent_ids:
                parent_ids[parent_text] = f"{source_name}_p{len(parent_ids)}"
        doc_chunks = []
        for chunk_idx, (child_text, parent_text) in enumerate(kept):
            chunk_dict = { "child_text": child_text, "parent_text": parent_text, "chunk_index": chunk_idx, "total_chunks": len(kept), "parent_id": parent_ids[parent_text], "parent_chunk_count": 0, "contextual_text": child_text, "id": f"{source_name}_{chunk_idx}", }
            chunk_dict[SOURCE_FIELD] = source_name
            doc_chunks.append(chunk_dict)
        _save_doc_chunks(source_name, sha, fingerprint, doc_chunks)
        return i, doc_chunks, False, None, messages
    except Exception as exc:
        # 逐文档兜底：单篇坏数据/异常不再拖垮整个 process（与串行逐句一致）
        return i, None, False, f"{type(exc).__name__}: {exc}", messages
    finally:
        _CHUNK_WORKER["messages"] = None
        log = prev_log


def build_chunks():
    # 上次 process 中断留下的半截 tmp（原子写靠 os.replace 收尾）：进到这里说明本轮
    # 没有别的 writer 在跑（process 单进程串行），直接清掉，不让孤儿 tmp 永久占盘。
    for _stale_tmp in (CHUNKS_JSON + ".tmp", CHUNKS_META_JSON + ".tmp"):
        try:
            os.remove(_stale_tmp)
        except OSError:
            pass
    # tokenizer 不在这里预加载：分块 worker（串行路径也用同一个函数）按需在自己的进程里
    # 加载一次；全部命中缓存时一个 token 都不用分（旧写法无条件加载，warm 路径白花几秒）。
    t_start = time.time()
    paths = _document_paths()
    config_fp, config_snapshot = _chunking_config_fingerprint()
    sources_now = _sources_fingerprint(paths)
    # ★ 存盘指纹 = 配置指纹 + 源文本 sha（见 _chunking_fingerprint）。旧写法只把配置算进
    #   fingerprint，源文件 sha 单独躺在 meta['sources'] 里：改了语料只跑 process 不跑 index
    #   时，索引 payload 里的 chunk_fingerprint 与"当前"仍然相等，serve 侧闸门恒等放行
    #   → 用户静默拿到旧知识库（只有 stages 能发现）。
    fingerprint = _chunking_fingerprint(sources_now, config_fp)
    name_of = _document_names(paths)
    names_now = set(name_of.values())
    if paths:
        log(f"找到 {len(paths)} 个文档: {', '.join(os.path.basename(p) for p in paths)}")
    else:
        # 目录存在、通配符一个都没命中 = 语料被清空（合法边界，见 _document_paths）：
        # 按 0 文档结算，写出空的 chunks.json + 空 sources，知识库才有"清空"这条路。
        log(f"⚠️  {DATA_DIR} 中没有匹配 {DOC_GLOB!r} 的文档：按空语料结算（chunks.json 写成 0 条）")
    all_chunks = []
    reused = 0
    failures: list = []
    total = len(paths)
    # 逐文档兜底语义不变：单篇坏数据/异常只跳过自己，末尾因 failures 短路返回、不覆盖
    # chunks.json —— 残缺产物一旦入库，索引就会静默少文档。
    jobs = [
        (i, filepath, name_of[filepath], sources_now[os.path.relpath(filepath, DATA_DIR)], fingerprint, total)
        for i, filepath in enumerate(paths)
    ]
    workers = min(total, offline_workers()) if total else 1
    log(f"分块并行度: {workers} 路（{hw_report()}）；按文档分发、按原顺序收集，产物与串行逐字节一致")
    done: list = [None] * total
    cursor = 0

    def _absorb_ready():
        """把已完成的**前缀**按原顺序并入产物：日志顺序、all_chunks 顺序都与串行完全一致
        （并行只改变"谁先算完"，不改变"按什么顺序汇总"）。"""
        nonlocal cursor, reused
        while cursor < total and done[cursor] is not None:
            idx, chunks, was_reused, err, messages = done[cursor]
            for line in messages:
                log(line)
            source_name = jobs[idx][2]
            if err is not None:
                log(f"[{idx + 1}/{total}] ❌ {source_name} 分块失败，继续下一篇: {err}")
                failures.append((source_name, err))
            else:
                all_chunks.extend(chunks)
                if was_reused:
                    reused += 1
            done[cursor] = None  # 已汇总：立刻放掉这条结果（块本身已并入 all_chunks）
            cursor += 1

    if workers > 1:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        try:
            pool = ProcessPoolExecutor(max_workers=workers)
        except Exception as exc:
            log(f"    ⚠️  无法创建分块进程池（{type(exc).__name__}: {exc}），回退串行分块")
            workers = 1
    if workers > 1:
        # 窗口化提交（在飞 ≤ 2×并行度）：结果不必全量驻留，Ctrl-C 也只需等在飞的那几篇，
        # 不会把 997 篇一次性提交后再空等收尾。
        with pool:
            pending: dict = {}
            next_job = 0
            while next_job < total or pending:
                while next_job < total and len(pending) < workers * 2:
                    job = jobs[next_job]
                    next_job += 1
                    pending[pool.submit(_chunk_one_document, job)] = job[0]
                finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for fut in finished:
                    idx = pending.pop(fut)
                    try:
                        done[idx] = fut.result()
                    except Exception as exc:
                        # worker 崩溃/结果不可 pickle：按单篇失败结算，其余文档照算
                        done[idx] = (idx, None, False, f"{type(exc).__name__}: {exc}", [])
                    _absorb_ready()
    else:
        for job in jobs:
            done[job[0]] = _chunk_one_document(job)
            _absorb_ready()
    _gc_doc_chunks(names_now)
    if reused:
        log(f"增量：复用 {reused}/{len(paths)} 个未变文档，只重算了 {len(paths) - reused} 个")
    if failures:
        log("")
        log(f"❌ {len(failures)}/{len(paths)} 个文档分块失败（其余已处理）:")
        for fail_name, fail_err in failures:
            log(f"   - {fail_name}: {fail_err}")
        log("chunks.json 与 meta 保持上一版（残缺产物不入库），失败文档未写缓存；修复后重跑 process 即可增量续算。")
        log(f"process 墙钟: {time.time() - t_start:.1f}s（复用 {reused}/{total}，失败 {len(failures)}；未写产物）")
        return 1
    if all_chunks and set(all_chunks[0]) != _RECORD_FIELDS:
        raise RuntimeError(f"分块记录字段与 _PAYLOAD_SPEC 不一致：多 {sorted(set(all_chunks[0]) - _RECORD_FIELDS)}、少 {sorted(_RECORD_FIELDS - set(all_chunks[0]))}，必须同步")
    parent_counts = Counter(c["parent_id"] for c in all_chunks)
    for c in all_chunks:
        c["parent_chunk_count"] = parent_counts[c["parent_id"]]
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    # 原子写（tmp + rename，与文档缓存/向量缓存同款）：直写时进程被杀会留下半截
    # chunks.json，下次 json.load 当场崩且无法自愈。先写齐两份 tmp 再依次 rename：
    # 任一步崩溃都只剩「新 chunks + 旧 meta」或整份旧产物，verify_chunks_freshness
    # 的指纹/语料比对都能拦下重跑，不会出现能通过校验的撕裂产物。
    chunks_tmp = CHUNKS_JSON + ".tmp"
    meta_tmp = CHUNKS_META_JSON + ".tmp"
    with open(chunks_tmp, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    meta = { "meta_version": CHUNKS_META_VERSION, "fingerprint": fingerprint, "chunking_config": config_snapshot, "sources": sources_now, "total_chunks": len(all_chunks), "total_parents": len(parent_counts), "reused_documents": reused, "build_time": datetime.datetime.now().isoformat(timespec="seconds"), }
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(chunks_tmp, CHUNKS_JSON)
    os.replace(meta_tmp, CHUNKS_META_JSON)
    log("分块统计:")
    for b, cnt in sorted(Counter(_source_of(c) or "?" for c in all_chunks).items()):
        log(f"  {b}: {cnt} 条")
    log(f"  总计: {len(all_chunks)} 条，父块 {len(parent_counts)} 个")
    log(f"已保存: {CHUNKS_JSON}")
    log(f"元信息: {CHUNKS_META_JSON}（指纹 {fingerprint}）")
    # 缓存命中 / 耗时可视化：一眼看出"这轮到底做了多少活、花了多久、卡在哪一段"。
    log(
        f"process 墙钟: {time.time() - t_start:.1f}s"
        f"（复用 {reused}/{total} 篇，重算 {total - reused} 篇，并行 {workers} 路；{hw_report()}）"
    )
    return 0


def verify_chunks_freshness():
    if not os.path.exists(CHUNKS_JSON):
        raise FileNotFoundError(f"找不到 {CHUNKS_JSON}，请先运行: python chatbot.py process")
    if not os.path.exists(CHUNKS_META_JSON):
        raise FileNotFoundError(f"找不到 {CHUNKS_META_JSON}，请重跑: python chatbot.py process")
    with open(CHUNKS_META_JSON, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("meta_version") != CHUNKS_META_VERSION:
        raise RuntimeError(f"{CHUNKS_META_JSON} 的 meta_version={meta.get('meta_version')}，当前代码是 {CHUNKS_META_VERSION}，请重跑 process")
    config_fp, config_snapshot = _chunking_config_fingerprint()
    paths = _document_paths()
    sources_now = _sources_fingerprint(paths)
    # 与 build_chunks **同一个函数**算存盘指纹：配置 + 源文本一起进哈希，两侧不可能分叉。
    fingerprint = _chunking_fingerprint(sources_now, config_fp)
    old_sources = meta.get("sources") if "sources" in meta else meta.get("books")
    if meta.get("fingerprint") != fingerprint:
        if old_sources != sources_now:
            changed = sorted( set(old_sources or {}) ^ set(sources_now) ) or [
                k for k in sources_now if (old_sources or {}).get(k) != sources_now[k]
            ]
            raise RuntimeError(f"{DATA_DIR}/ 中匹配 {DOC_GLOB!r} 的文档在分块之后发生了变化: {changed}，请重跑 process")
        diff = { k: (meta.get("chunking_config", {}).get(k), v) for k, v in config_snapshot.items() if meta.get("chunking_config", {}).get(k) != v }
        raise RuntimeError(f"分块产物的切分指纹 {meta.get('fingerprint')} 与当前 {fingerprint} 不一致，配置差异（旧→新）: {diff}，请重跑 process")
    return meta


# ── 词表：别名归一与停用词 ──
ALIASES_ENABLED = env_bool("RAG_ALIASES", True)
STOPWORDS_ENABLED = env_bool("RAG_STOPWORDS", True)
ALIASES_FILE = os.getenv("RAG_ALIASES_FILE") or os.path.join(LEXICON_DIR, "aliases.txt")
STOPWORDS_FILE = os.getenv("RAG_STOPWORDS_FILE") or os.path.join(LEXICON_DIR, "stopwords.txt")
if not os.path.exists(STOPWORDS_FILE):
    _legacy_stopwords = os.path.join(LEXICON_DIR, "stopwords_classical.txt")
    if os.path.exists(_legacy_stopwords):
        log("⚠️  %s 不存在，回退使用旧文件名 stopwords_classical.txt"
            % STOPWORDS_FILE)
        STOPWORDS_FILE = _legacy_stopwords
def load_lexicon():
    """词表（别名归一 + 停用词）单例：懒加载 + 加锁。重复并发（预热线程/首查询）
    只构造一份；同一进程内保持同一份解析结果，避免"一次走 A 词表一次走 B 词表"。"""
    state = _state_ref()
    with state.lexicon_lock:
        if state.lexicon is not None:
            return state.lexicon
        alias_map, stopwords = {}, set()
        if ALIASES_ENABLED:
            alias_map = _read_aliases(ALIASES_FILE)
        if STOPWORDS_ENABLED:
            stopwords = _read_stopwords(STOPWORDS_FILE)
        if alias_map:
            surfaces = sorted(alias_map, key=len, reverse=True)
            from ahocorasick import Automaton
            alias_re = Automaton()
            for _surface in sorted(set(surfaces)):
                alias_re.add_word(_surface, _surface)
            alias_re.make_automaton()
        else:
            alias_re = None
        state.lexicon = (alias_map, alias_re, stopwords)
        return state.lexicon


def parse_aliases(path):
    groups, problems = [], []
    if not os.path.exists(path):
        problems.append("文件不存在：%s" % path)
        return groups, problems
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            body = raw.split("#", 1)[0]
            if not body.strip():
                continue
            if raw.rstrip("\n").endswith("\t"):
                problems.append("第 %d 行：行尾有多余 TAB（空字段）" % lineno)
            fields = body.rstrip("\n").split("\t")
            fields = [f.strip() for f in fields]
            if any(f == "" for f in fields):
                problems.append("第 %d 行：存在空字段 %r" % (lineno, fields))
                fields = [f for f in fields if f]
            if len(fields) < 2:
                problems.append( "第 %d 行：只有规范名没有别名 %r（不产生任何归一收益）" % (lineno, fields) )
                continue
            canonical, aliases = fields[0], fields[1:]
            if len(set(aliases)) != len(aliases):
                dup = [a for a, n in Counter(aliases).items() if n > 1]
                problems.append("第 %d 行：组内别名重复 %s" % (lineno, dup))
            if canonical in aliases:
                problems.append("第 %d 行：规范名 %s 又出现在别名里" % (lineno, canonical))
            groups.append( {"canonical": canonical, "aliases": aliases, "line": lineno, "note": raw.split("#", 1)[1].strip() if "#" in raw else ""} )
    canon_seen = defaultdict(list)
    surface_owner = defaultdict(list)
    for g in groups:
        canon_seen[g["canonical"]].append(g["line"])
        for s in [g["canonical"]] + g["aliases"]:
            surface_owner[s].append(g["canonical"])
    for canon, lns in canon_seen.items():
        if len(lns) > 1:
            problems.append("规范名 %s 出现在多个组（行 %s）" % (canon, lns))
    for surface, owners in surface_owner.items():
        uniq = sorted(set(owners))
        if len(uniq) > 1:
            problems.append( "词面 %s 同时归属多个规范名 %s —— 归一目标冲突" % (surface, uniq) )
        elif len(owners) > 1:
            problems.append("词面 %s 在同一组内重复出现" % surface)
    return groups, problems


def parse_stopwords(path):
    words: list = []
    problems: list = []
    if not os.path.exists(path):
        problems.append("文件不存在：%s" % path)
        return words, problems
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            word = raw.split("#", 1)[0].strip()
            if not word:
                continue
            if any(ch.isspace() for ch in word):
                problems.append("第 %d 行：一个词里出现空白字符 %r" % (lineno, word))
            words.append((word, lineno))
    seen = Counter(w for w, _ in words)
    for w, n in seen.items():
        if n > 1:
            problems.append("停用词重复：%s（%d 次）" % (w, n))
    return words, problems


def _read_aliases(path):
    """运行时词表加载：委托 parse_aliases，与校验器同一套格式。"""
    groups, problems = parse_aliases(path)
    for problem in problems:
        log(f"⚠️  别名表 {problem}")
    mapping: dict = {}
    for group in groups:
        for surface in [group["canonical"]] + group["aliases"]:
            mapping.setdefault(surface, group["canonical"])
    if not mapping:
        log(f"⚠️  别名表缺失/为空/无法解析，跳过别名归一: {path}")
        return {}
    log(f"别名表加载完成: {len(mapping)} 个词面 → {len(set(mapping.values()))} 个规范名")
    return mapping


def _read_stopwords(path):
    """运行时停用词加载：委托 parse_stopwords，同 _read_aliases。"""
    entries, problems = parse_stopwords(path)
    for problem in problems:
        log(f"⚠️  停用词表 {problem}")
    words = {word for word, _lineno in entries}
    if not words:
        log(f"⚠️  停用词表缺失或为空，跳过停用词过滤: {path}")
        return set()
    log(f"停用词表加载完成: {len(words)} 个词")
    return words


def _distinct_lexicon_ids(client, collection, page_size=1000):
    """滚动全量收集集合里出现过的 lexicon_id（只带这一个 payload 键），返回
    [(指纹, 点数), ...]，集合为空返回 []；点数用于混合态时把残留规模打出来。

    必须全量而非抽首点：sparse 逐页原地更新，中断/跳页留下"已更新前缀 +
    旧指纹后缀"，而 scroll 首点恒属已更新前缀 —— 抽 1 个点必过，旧稀疏向量就
    静默参与融合（判据恒真的抽样形态）。单次进程只取这一个键，23 页滚动开销可忽略。
    """
    seen: dict = {}
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=collection, limit=page_size, offset=offset,
            with_payload=[LEXICON_ID_FIELD], with_vectors=False,
        )
        for p in pts:
            lid = (p.payload or {}).get(LEXICON_ID_FIELD)
            seen[lid] = seen.get(lid, 0) + 1
        if offset is None:
            break
    return sorted(seen.items(), key=lambda kv: str(kv[0]))


def qdrant_server_version(client=None):
    """Qdrant 服务端版本；取不到就**抛错**（不让调用方拿到一个可能恒等的常量）。

    真正的稀疏编码器在服务端：BM25 的分词器、停用词表、IDF 都由它实现。升级 Qdrant
    就可能换词空间，而词表指纹是稀疏侧唯一的 fail-closed 判据 —— 版本不进指纹时，
    升级后 stages 照报 C 阶段新鲜，稀疏向量静默错配（失效原型：静默）。
    """
    created = client is None
    if created:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
    try:
        info = client.info()
    finally:
        if created:
            try:
                client.close()
            except Exception:
                pass
    version = getattr(info, "version", None)
    if not version:
        raise RuntimeError(
            "无法从 Qdrant 读取服务端版本（info().version 为空）：稀疏编码器在服务端，"
            "版本未知则词表指纹无从判断（Qdrant 不可达时请先把它起起来）"
        )
    return str(version)


def lexicon_fingerprint(client=None):
    """词表与稀疏编码器指纹；任一变 → 只需 sparse，不牵连稠密与分块。

    ★ 只哈希**真正参与运行期归一**的文件（ALIASES_FILE，load_lexicon 会读它）。
      lexicon/stopwords.txt 运行期从不被使用 —— 停用词走 Qdrant 内置 chinese 表
      （见 bm25_options），RAG_STOPWORDS_FILE 只服务于 python chatbot.py lexicon 自检。
      旧写法无条件把它哈希进指纹：改这个文件检索行为一点不变，却会让闸门报"词表与索引
      不一致"、拦住全部检索并强制跑一次 sparse（失效原型：判据与实现不同源）。
      若将来把该文件真正下发到 bm25_options()["stopwords"]，必须把它（的内容）重新纳入
      指纹，否则改停用词不动指纹 = 稀疏词空间静默错配。
    """
    parts = []
    try:
        with open(ALIASES_FILE, "rb") as f:
            parts.append(hashlib.sha256(f.read()).hexdigest())
    except OSError:
        parts.append("missing")
    # 显式加入解析后的运行时配置值，防止环境变量变更但文件内容相同时指纹不变
    parts.append(f"ALIASES_FILE={ALIASES_FILE}")
    parts.append(f"aliases={int(ALIASES_ENABLED)},stopwords={int(STOPWORDS_ENABLED)}")
    # 分词/停用词已交给 Qdrant（bm25_options 的 tokenizer 与 stopwords 就是词空间的定义），
    # 故不再有 jieba 版本项与标点集项；选项本身仍必须进指纹（改 tokenizer 就是改词空间）。
    parts.append("bm25=" + json.dumps(bm25_options(), sort_keys=True))
    # Qdrant 服务端版本：稀疏编码器在服务端，升级它就可能换词空间（见 qdrant_server_version）。
    parts.append("qdrant_server=" + qdrant_server_version(client))
    # BM25 模型名与解析/归一链路源码同样决定稀疏词空间：漏掉会让换编码后指纹不变、
    # stages 误报 C 新鲜，稀疏索引静默错配。
    parts.append("bm25_model=" + BM25_MODEL_NAME)
    import inspect
    # parse_stopwords/_read_stopwords 不在列表里：它们的产物只进 lexicon 自检，
    # 不影响稀疏词空间（同"与实现同源"的理由）。
    for name in ("parse_aliases", "_read_aliases", "load_lexicon", "normalize_aliases", "sparse_text", "sparse_encode"):
        fn = globals().get(name)
        try:
            parts.append(f"{name}:" + inspect.getsource(fn))  # type: ignore[arg-type]
        except (OSError, TypeError):
            parts.append(f"{name}:<unavailable>")
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def normalize_aliases(text):
    """别名归一：Aho-Corasick **最长优先、从左到右不重叠**（与旧正则交替换逐字等价）。

    2026-10-14 组件化：匹配引擎换成 pyahocorasick（C 实现）。等价性由 lexicon 自检的
    「与线上实现对拍」守着 —— 那条判据拿参照实现与这里逐字比对，不一致就报错。
    """
    if not text:
        return text
    alias_map, automaton, _ = load_lexicon()
    if automaton is None:
        return text
    best = {}
    for end, surface in automaton.iter(text):
        start = end - len(surface) + 1
        cur = best.get(start)
        if cur is None or len(surface) > len(cur):
            best[start] = surface
    if not best:
        return text
    out, i, n = [], 0, len(text)
    while i < n:
        surface = best.get(i)
        if surface is not None:
            out.append(alias_map[surface])
            i += len(surface)
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def sparse_text(text):
    """稀疏通道的文本准备：**只做领域别名归一**，分词与停用词都交给 Qdrant。

    改造前这里是 jieba 的 cut_for_search 分词 + 标点过滤 + 停用词过滤，并为此维护一把串行锁
    （jieba 分词非线程安全，并发会产出被截断的分词 → 稀疏 Document 变空 → Qdrant 才回 422，
    属"静默错配"）。现在切词在服务端（tokenizer=multilingual），Python 侧只剩"别名归一"
    这一件真领域知识，分词器与锁一并删除。
    """
    normalized = normalize_aliases(text or "")
    # 纯标点/空白：服务端切不出任何 term，空文档会让 BM25 静默失败 → 用哨兵占位
    return normalized if any(ch.isalnum() for ch in normalized) else _EMPTY_SPARSE_TEXT


_EMPTY_SPARSE_TEXT = "\u200b"  # sparse_text 判空时的哨兵（零宽空格）


def sparse_encode(text):
    from qdrant_client import models
    return models.Document(text=sparse_text(text), model=BM25_MODEL_NAME, options=bm25_options())


def sparse_terms_of(text):
    """稀疏通道**有没有可检索内容**（不再是"有哪些词"）。

    分词在服务端，Python 拿不到 term 列表，所以判据退一步：别名归一后还剩字母/数字就有
    内容，纯标点/纯空白就没有。调用方（hybrid_search）据此既不融合也不召回 —— 与改造前
    "分词为空即退化为纯稠密"是同一条判据口径，只是粒度从"逐词"变成"有没有"。
    返回列表是为了保持调用方的形状不变（判真值 + 记进 steps）。
    """
    prepared = sparse_text(text)
    return [] if prepared == _EMPTY_SPARSE_TEXT else [prepared]


def _index_text(rec):
    """**入库文本的唯一来源**：稠密向量、稀疏向量、稠密缓存键都取这一个字段。

    为什么必须是同一份：`contextual_text` 与 `child_text` 是两个字段（`_PAYLOAD_SPEC`
    各存一份），旧代码稠密取前者、稀疏取后者、stages 的缓存键又写成
    `contextual_text or child_text` 的兜底 —— 三份规则。两者今天逐字相同，所以差异
    一直被掩盖；一旦分叉，**同一个点的稠密与稀疏向量就描述了不同文本**，检索静默
    变差且无处可查。这里把规则收敛成一份，缺字段直接报错而不是悄悄换字段。
    """
    text = rec.get("contextual_text")
    if not isinstance(text, str):
        raise RuntimeError(
            f"分块记录缺 contextual_text（id={rec.get('id')!r}）——"
            "稠密/稀疏/缓存键都取它，请重跑 process"
        )
    return text


def _embed_cache_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_EMBED_CODE_FINGERPRINT = None


def _embed_code_fingerprint():
    """嵌入代码指纹：_embed_batch/embed 源码变化必须作废稠密向量缓存（换池化/归一化/截断等）。

    旧实现只按模型路径+精度给缓存起名，改嵌入代码会静默复用旧向量。
    """
    global _EMBED_CODE_FINGERPRINT
    if _EMBED_CODE_FINGERPRINT is None:
        import inspect
        parts = []
        for name in ("_embed_batch", "embed", "load_embedding_model"):
            fn = globals().get(name)
            try:
                parts.append(f"{name}:" + inspect.getsource(fn))  # type: ignore[arg-type]
            except (OSError, TypeError):
                parts.append(f"{name}:<unavailable>")
        _EMBED_CODE_FINGERPRINT = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    return _EMBED_CODE_FINGERPRINT


# 模型目录不可用时的哨兵：表示"**无从判断**"，而不是某个哈希值。它进切分指纹后必然与
# 任何真实产物都不相等（fail-closed），并由 _chunking_config_fingerprint 单独报"嵌入模型
# 目录缺失"，而不是报"重跑 process+index" —— 那条路正需要这个目录（归因不能指反）。
_MODEL_CONTENT_UNKNOWN = "unknown"
_MODEL_FULL_HASH_MAX = 4 * 1024 * 1024   # 小文件（config/tokenizer/池化头/词表）全量哈希
_MODEL_HEAD_BYTES = 4096
_MODEL_TAIL_BYTES = 65536


def _model_content_fingerprint():
    """嵌入模型目录内容指纹（相对路径 + 大小 + 内容特征）。

    原实现用 mtime_ns：容器启动时下载模型会更新 mtime，导致切分指纹漂移，
    进而触发「分块产物与索引不一致」闸门。改用内容特征，同一份模型内容不变即指纹不变，
    容器重建/重新下载不再漂移。

    · **跳过隐藏文件/目录**（HF 的下载记账目录、.part 暂存）：下载器元数据与模型内容
      无关，有无会让指纹凭空分叉。清单文件（config/tokenizer/权重）才是内容身份。
    · **目录不可用返回哨兵**（不存在/为空）：旧实现 except OSError 后对空列表求哈希，
      静默返回常量 e3b0c44298fc —— "模型目录没了"因此看起来像一个正常指纹，一路走到
      "请重跑 process+index"，而那条路正需要这个目录（失效原型：静默 + 归因错）。
    · **小文件全量、大文件头 + 尾**：旧实现只读每个文件前 4096 字节，同大小、4KB 之后
      的改动（换权重、截断、同尺寸不同版本）完全检测不到，指纹不变 → 稠密缓存与分块
      指纹被静默复用。
    """
    if not _model_dir_ready(EMBED_MODEL_PATH):
        return _MODEL_CONTENT_UNKNOWN
    parts = []
    for root, dirs, files in os.walk(EMBED_MODEL_PATH):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, EMBED_MODEL_PATH)
            try:
                st = os.stat(path)
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    if st.st_size <= _MODEL_FULL_HASH_MAX:
                        h.update(f.read())          # 小文件全量：一个字节都不放过
                    else:
                        h.update(f.read(_MODEL_HEAD_BYTES))
                        if st.st_size > _MODEL_TAIL_BYTES:
                            f.seek(st.st_size - _MODEL_TAIL_BYTES)
                            h.update(f.read(_MODEL_TAIL_BYTES))
                        # 长度也入哈希：头尾恰好相同、只有中段长度不同也能区分
                        h.update(str(st.st_size).encode("ascii"))
                parts.append(f"{rel}:{st.st_size}:{h.hexdigest()[:8]}")
            except OSError:
                parts.append(f"{rel}:missing")
    if not parts:
        return _MODEL_CONTENT_UNKNOWN
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def _embed_cache_file(precision):
    model_key = hashlib.sha256( os.path.abspath(EMBED_MODEL_PATH).encode("utf-8") ).hexdigest()[:12]
    content_key = _model_content_fingerprint()
    code_key = _embed_code_fingerprint()
    return os.path.join(EMBED_CACHE_DIR, f"{model_key}_{content_key}_{code_key}_{precision}.npz")


def _expected_embed_dim():
    """嵌入维度的无模型读法：直接读模型目录的 config.json。

    只服务"缓存命中后不必加载模型"的路径 —— 那条路拿不到 model.config.hidden_size，
    而维度交叉校验不能因此消失：同路径换了个不同维度的模型时，少了这道校验就会静默
    复用旧向量。正常加载路径仍以已加载模型的 config 为准；这里读不到就返回 None，
    由 _load_embed_cache 跳过该项校验（config.json 都缺的模型目录本身已是异常）。
    """
    try:
        with open(os.path.join(EMBED_MODEL_PATH, "config.json"), encoding="utf-8") as f:
            dim = json.load(f).get("hidden_size")
        return int(dim) if dim else None
    except Exception:
        return None


# 固定优先级，与设备无关：完整命中时向量全部来自同一份文件，精度只决定"读哪份"，
# 不混用就不出错。顺序写死成常量，避免两处各自排序导致跨次运行读到不同那份。
_EMBED_CACHE_PRECEDENCE = ("fp16", "fp32")


def _inspect_embed_caches(keys):
    """逐份检查候选稠密缓存，返回 [(precision, path, cache, missing), ...]。

    只含磁盘上存在且读得出来的那份，按 _EMBED_CACHE_PRECEDENCE 排序；没有任何候选
    返回空列表。**这是 index 与 stages 共用的唯一一份缓存选择规则** —— 两处各写一份
    必然分叉，正是"stages 说新鲜 / index 却重算"这类同轮日志两个结论的来源。

    维度校验走 _expected_embed_dim()（不加载模型），missing 是相对传入 keys 的缺口。
    """
    out = []
    for precision in _EMBED_CACHE_PRECEDENCE:
        path = _embed_cache_file(precision)
        if not os.path.exists(path):
            continue
        cache = _load_embed_cache(path, expected_dim=_expected_embed_dim())
        if not cache:
            continue
        missing = [i for i, k in enumerate(keys) if k not in cache]
        out.append((precision, path, cache, missing))
        # 优先级排第一的已完整覆盖：后面的候选不可能再提供任何东西了（只会把缺失列表
        # 排得更长），提前收工省一次全量字典遍历。
        if not missing:
            break
    return out


def _full_embed_cache(cands):
    """候选里第一个**完整覆盖 keys** 的（按固定优先级），没有则 None。

    完整命中 = 一条向量都不用算 = 不需要知道设备、不需要加载 torch 与嵌入模型。
    """
    return next((c for c in cands if not c[3]), None)


def _load_embed_cache(path, expected_dim=None):
    import numpy as np
    if not os.path.exists(path):
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = data["keys"].tolist()
            vecs = data["vectors"]
        if vecs.ndim != 2:
            raise ValueError(f"vectors 不是二维数组: shape={vecs.shape}")
        if len(keys) != vecs.shape[0]:
            raise ValueError( f"keys 与 vectors 行数不一致: {len(keys)} vs {vecs.shape[0]}")
        if vecs.shape[0] == 0:
            raise ValueError("缓存为空数组")
        if expected_dim is not None and vecs.shape[1] != expected_dim:
            raise ValueError(f"缓存向量维度 {vecs.shape[1]} 与模型期望的 {expected_dim} 不一致（多半是换过模型）")
    except Exception as exc:
        log(f"    ⚠️  向量缓存不可用，将全量重算: {type(exc).__name__}: {exc}")
        return {}
    log(f"向量缓存载入: {len(keys)} 条，{vecs.shape[1]} 维")
    return {k: vecs[i] for i, k in enumerate(keys)}


def _save_embed_cache(path, keys, matrix):
    import numpy as np
    tmp = path + ".tmp.npz"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(tmp, keys=np.array(keys, dtype="U64"), vectors=matrix)
        os.replace(tmp, path)
    except Exception as exc:
        log(f"    ⚠️  向量缓存写入失败（本次索引不受影响）: {type(exc).__name__}: {exc}")
        try:
            os.remove(tmp)  # 失败留下的半截 tmp 别占盘
        except OSError:
            pass


def _gc_embed_cache_tmp():
    """清掉写入中断遗留的 *.tmp.npz（正常缓存名恒以 _fp16.npz/_fp32.npz 结尾，
    .tmp.npz 只可能来自被杀掉的那次 index）。"""
    try:
        if not os.path.isdir(EMBED_CACHE_DIR):
            return
        removed = 0
        for name in os.listdir(EMBED_CACHE_DIR):
            if name.endswith(".tmp.npz"):
                os.remove(os.path.join(EMBED_CACHE_DIR, name))
                removed += 1
        if removed:
            log(f"清理 {removed} 个中断遗留的向量缓存临时文件")
    except Exception as exc:
        log(f"    ⚠️  向量缓存临时文件清理失败（不影响本次索引）: {type(exc).__name__}: {exc}")


# ── 索引构建 ──
# qdrant-client 的本地推理器是**每客户端一份、且内部没有锁**的：
# qdrant_client/embed/model_embedder.py 用 self._embed_storage[model_name] 暂存待发 Document，
# 同一客户端上两个线程同时走同名 Document 就会互相 drain，实测
# IndexError: pop from empty list（另一批表现为 422 "dense vector must not be empty"）。
# 触发点是任何带 Document 的请求。
# 这里只解决"多个线程共用同一个 client"的场景（检索侧双通道的稀疏分支，见
# dual_channel_recall._sparse）；写批走 per-worker client（每个线程一份推理器），
# 因此**不**在这把锁里 —— 否则 HTTP 往返会被串行化，4 路写批退化成 1 路。
_QDRANT_INFERENCE_LOCK = threading.Lock()


def _upsert_batch_with_retry(client: Any, collection_name: str, points: list, retries: int = 1) -> str | None:
    """单批 upsert：瞬时故障立即重试 retries 次；仍失败返回错误串，由调用方记账后继续写其余批。
    继续写只为更快暴露完整失败面 —— 末尾 write_failures 非空一律 raise、不切别名，旧集合继续服务。
    重试不睡眠：写批已并发发出，等待只会占住 worker 槽；瞬时故障立即重发反而更快。

    **不持 _QDRANT_INFERENCE_LOCK**：调用方（build_index 的写批）给每个 worker 线程一个
    自己的 client，Document 的本地拆包天然不会跨线程；在这里持锁会把 HTTP 往返也串行化，
    4 路写批退化成 1 路（注释宣称的并发不存在）。持锁版留给共用同一 client 的调用点。"""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return None
        except Exception as exc:
            last = exc
            if attempt < retries:
                log(f"    ⚠️  upsert 失败，立即重试（{attempt + 1}/{retries}）: {type(exc).__name__}: {exc}")
    return f"{type(last).__name__}: {last}"


def _current_alias_target(client, alias=COLLECTION_ALIAS):
    """别名当前**实际**指向的集合；读不到返回 None。

    失败路径的文案与恢复动作都以此为准：旧写法在 catch 里凭空断言"检索端仍指向旧集合"，
    而 DeleteAlias 成功、CreateAlias 失败时别名其实已经被删空 —— 报错把排查方向指反。
    """
    try:
        return next((a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias), None)
    except Exception as exc:
        log(f"    ⚠️  读取别名 {alias} 失败，无法判定其实际指向: {type(exc).__name__}: {exc}")
        return None


def build_index():
    import time
    import uuid

    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        CreateAlias,
        CreateAliasOperation,
        DeleteAlias,
        DeleteAliasOperation,
        Distance,
        Modifier,
        PayloadSchemaType,
        PointStruct,
        SparseVectorParams,
        VectorParams,
    )
    meta = verify_chunks_freshness()
    # 内存闸门（RAG_INDEX_MIN_FREE_MB > 0 时启用）：嵌入阶段实测峰值 ~2.2-3.0G，与
    # serve（进程内检索模型 ~2-3G）或生成模型（131072 档 KV 预留 ≈3G + 权重 1.6G）
    # 同机并存就会顶穿 8G —— "重建与服务的隔离规则"在这里用可开内存表达，不检测进程。
    # macOS 探测不到 MemAvailable：闸门跳过但显式说明（不冒充"检查过了"）。
    _free_mb = _mem_available_mb()
    if INDEX_MIN_FREE_MB > 0:
        if _free_mb is None:
            log(f"⚠️  index 内存闸门：本机读不到 MemAvailable（macOS/容器外），跳过检查（门槛 {INDEX_MIN_FREE_MB}MB 未生效）")
        elif _free_mb < INDEX_MIN_FREE_MB:
            raise RuntimeError(
                f"可开内存 {_free_mb}MB < 门槛 {INDEX_MIN_FREE_MB}MB（RAG_INDEX_MIN_FREE_MB）："
                f"大概率有 serve/生成模型同机驻留。按 README §11.4 的隔离规则："
                f"先停生成/检索侧负载再跑 index；确要强行重建请临时把 "
                f"RAG_INDEX_MIN_FREE_MB 调低后重试（后果自担）"
            )
        else:
            log(f"index 内存闸门: 可开 {_free_mb}MB ≥ 门槛 {INDEX_MIN_FREE_MB}MB")
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        all_chunks = json.load(f)
        # ★ 记录键集必须与当前契约一致：改 RAG_SOURCE_FIELD 后只跑 index（没跑 process）时，
        # 切分指纹与源 sha 都不变、stages 四段全绿，而旧键读不到 -> payload 来源全空、
        # 来源过滤恒零命中（静默）。这里 fail-closed 并指名要重跑 process。
        if all_chunks and set(all_chunks[0]) != _RECORD_FIELDS:
            _miss = sorted(_RECORD_FIELDS - set(all_chunks[0]))
            _extra = sorted(set(all_chunks[0]) - _RECORD_FIELDS)
            raise RuntimeError(f"{CHUNKS_JSON} 记录键集与当前契约不一致（多 {_extra}、少 {_miss}）——通常是改了 RAG_SOURCE_FIELD 等字段配置，请重跑 process 后再 index")
    log(f"加载分块数据: {len(all_chunks)} 条（{meta.get('total_parents', '?')} 个父块）")
    build_id = uuid.uuid4().hex
    lexicon_id = lexicon_fingerprint()
    # 记进 payload 的是**产物自己的**指纹（verify_chunks_freshness 刚核对过与当前一致），
    # 检索侧拿它与当前指纹比对，process 后忘了重跑 index 就会当场报错而不是静默用旧库。
    chunk_fingerprint = meta.get("fingerprint")
    log(f"索引建造标识: {build_id}")
    log(f"分块指纹: {chunk_fingerprint}（检索侧据此判断分块是否过期）")
    log(f"词表指纹: {lexicon_id}（别名 {ALIASES_ENABLED} / 停用词 {STOPWORDS_ENABLED}）")
    total = len(all_chunks)
    t0 = time.time()
    # 不保留全量 texts：那是一份 ~35MB 的中间 list，而它的两次用途（算 key、逐批取文）
    # 都能按需重取 _index_text（纯读字段+校验）。命中路径下调用次数不变，缺失路径每条
    # 多算一次 —— 换峰值里少背 35MB。
    keys = [_embed_cache_key(_index_text(c)) for c in all_chunks]
    _gc_embed_cache_tmp()  # 写入端顺手清中断遗留（*.tmp.npz 不匹配任何正常缓存名）
    # 先按磁盘事实看有没有一份缓存能完整覆盖 keys：完整命中就一条向量都不用算，于是
    # 不必知道设备、不必加载 torch 与嵌入模型（约 600MB 峰值，原本只为猜"读哪个精度
    # 的文件名"和拿 hidden_size 而付）。cands 同时供下面的增量路径复用，同一份文件
    # 不会读两遍。
    cands = _inspect_embed_caches(keys) if (EMBED_CACHE_ENABLED and total) else []
    hit = _full_embed_cache(cands)
    if not total:
        # 空产物 =「已清空」的合法边界（corpus 被清空，见 _document_paths / build_chunks）：
        # 与正常重建同法写入 0 点新集合并切别名 —— 写出就是一个**已清空**的知识库，
        # 而不是在此 raise 把"清空"变成不可达状态（serve/stages 永远指向死路）。
        # 清空集合同样需要一个稠密维度：嵌入模型目录缺失在 verify_chunks_freshness 的
        # 切分指纹里已先行报错，这里 config.json 缺 hidden_size 是最后防线。
        dense_dim = _expected_embed_dim()
        if not dense_dim:
            raise RuntimeError(
                f"嵌入模型目录读不出稠密维度（config.json 缺失或 hidden_size 为空，{EMBED_MODEL_PATH!r}）："
                "空集合同样需要维度，先恢复模型目录再重跑 index"
            )
        dense_dim = int(dense_dim)
        precision, device, cache_path = None, None, None
        cache, missing, cache_entries = {}, [], 0
        tokenizer = model = None
        log("⚠️  chunks.json 为空（corpus 已清空）：跳过嵌入阶段，按清空语义写入 0 点集合并切换别名")
    elif hit is not None:
        precision, cache_path, cache, missing = hit
        device = None
        use_gpu = False
        tokenizer = model = None
        log(f"稠密缓存完整命中: {os.path.basename(cache_path)}（{precision}）{len(keys)} 条 → 跳过设备选择与模型加载")
    else:
        device = INDEX_DEVICE or get_device()
        use_gpu = device in ("mps", "cuda")
        tokenizer = model = None
        if use_gpu:
            try:
                tokenizer, model, _ = load_embedding_model(device=device)
                model = model.half()
            except Exception as exc:
                log(f"    ⚠️  {device.upper()} 初始化失败，回退 CPU 建索引: {type(exc).__name__}: {exc}")
                tokenizer, model, use_gpu = None, None, False
        if use_gpu:
            batch_size = EMBED_BATCH_SIZE or embed_batch_size(device)
            precision = "fp16"
            log(f"Embedding 设备: {device} (fp16, batch_size={batch_size}；{embed_batch_note(device)})")
        else:
            import torch
            tokenizer, model, device = load_embedding_model(device="cpu")
            # 线程数 = **真实核数**（HW_CORES 优先，缺失才自探测，见 hw_cores）：旧写法在这里
            # 写死 6 —— 8 核机器白丢 2 个核、16 核机器白丢 10 个，而且日志里的"6线程"还会
            # 把"为什么是 6"包装成一个像是刻意的选择。
            _threads = hw_cores()
            torch.set_num_threads(_threads)
            batch_size = EMBED_BATCH_SIZE or embed_batch_size(device)
            precision = "fp32"
            log(f"Embedding 设备: cpu ({_threads} 线程, batch_size={batch_size}；{embed_batch_note(device)})")
        cache_path = _embed_cache_file(precision)
        expected_dim = getattr(getattr(model, "config", None), "hidden_size", None)
        if expected_dim is None:
            # 回退：用一次前向传播推断维度，避免缓存维度校验被绕过
            test_vec = embed([_index_text(all_chunks[0])], tokenizer, model, device, is_query=False, autocast=use_gpu)
            expected_dim = test_vec.shape[-1]
            log(f"  推断嵌入维度: {expected_dim}")
        # 该精度的缓存刚被 _inspect_embed_caches 读过（含维度校验），复用它的结论；
        # 没这份文件或它读坏了 → 全量待算，与旧写法 cache={} 等价。
        reuse = next((c for c in cands if c[0] == precision), None)
        cache, missing = (reuse[2], reuse[3]) if reuse else ({}, list(range(total)))
    # 先记条数再 del cache：后面"要不要重写缓存"的判据要用它，而缓存 dict 本身
    # （22019 个条目 ≈ 12MB）在 embeddings 取完值之后就不该再占着峰值。
    cache_entries = len(cache)
    log(f"批量生成 Embedding... 命中缓存 {total - len(missing)}/{total} 条，需计算 {len(missing)} 条")
    embeddings = [cache.get(k) for k in keys] if cache else [None] * total
    del cache
    if missing:
        batches = [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]
        for bi, idxs in enumerate(batches):
            batch_texts = [_index_text(all_chunks[i]) for i in idxs]
            vecs = embed(batch_texts, tokenizer, model, device, is_query=False, autocast=use_gpu)
            if device == "mps":
                _mps_empty_cache()
            for row, i in enumerate(idxs):
                embeddings[i] = vecs[row]
            done = min((bi + 1) * batch_size, len(missing))
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            log(f"  计算进度 {done / len(missing) * 100:5.1f}% ({done}/{len(missing)}) {speed:.0f} 条/秒")
    if total:
        holes = [i for i, e in enumerate(embeddings) if e is None]
        if holes:
            raise RuntimeError(f"{len(holes)} 个槽位没有被填充（例如 {holes[:5]}）：缓存命中数与待计算集合不一致")
        dims = {int(np.asarray(e).shape[-1]) for e in embeddings}
        if len(dims) != 1:
            raise RuntimeError(f"向量维度不一致: {sorted(dims)} —— 通常意味着缓存里混入了别的模型的向量，请删掉 {cache_path} 后重跑或设 RAG_EMBED_CACHE=0")
        # 1710: 上方 embeddings 是 list[ndarray|None]（槽位填充），从这里起是整块矩阵 —— 同名两义正是类型报错根源，矩阵一侧改名 matrix。
        matrix = np.vstack([np.asarray(e, dtype="float32") for e in embeddings])
        # 显式 raise 而非 assert：`python -O` 会剥离 assert，不变量静默消失
        if matrix.shape[0] != total:
            raise RuntimeError(f"向量条数 {matrix.shape[0]} 与分块数 {total} 不一致")
        # 维度下游（create_collection / 一致性校验）一律引用 dense_dim：它与矩阵生命周期
        # 解耦，清空重建（total==0）_branch 上方从模型目录读出，两条路都有一份。
        dense_dim = int(matrix.shape[1])
        if EMBED_CACHE_ENABLED:
            # 命中且无冗余时整份重写是纯浪费：np.savez 在峰值里再摊一份 zip 缓冲，
            # 磁盘上 73MB 也原地重写一遍。判据要两条一起看才严 —— missing==0 保证每个
            # key 都在缓存里，条数相等才排除"缓存里残留已删文档的旧条目"（那种仍需重写清理）。
            # ★ 两侧口径必须都去重：cache 是 dict（键天然唯一），keys 是逐块列表 —— 只要两条
            #   分块正文逐字相同（重复段落/重复标题很常见），len(keys) 就恒大于 len(cache)，
            #   判据恒真，index 每次都整份重写 npz（判据恒真式浪费，日志却写"全部命中"）。
            if missing or cache_entries != len(set(keys)):
                _save_embed_cache(cache_path, keys, matrix)
                log(f"向量缓存已更新: {cache_path}（重算 {len(missing)} 条）")
            else:
                log(f"向量缓存无需更新（{len(keys)} 条全部命中且无冗余）: {cache_path}")
    del keys, model, tokenizer
    if device == "mps":
        _mps_empty_cache()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
    alias_ok = USE_COLLECTION_ALIAS
    if alias_ok:
        try:
            client.get_aliases()
        except Exception as exc:
            log(f"    ⚠️  Qdrant 不支持集合别名，回退为 delete+create 全量重建: {exc}")
            alias_ok = False
    # 旧集合名不再在这里占位：别名分支用 prev_alias_target（本轮之前的目标），
    # 非别名分支的目标集合就是 COLLECTION_NAME 本身。
    target_collection = f"{COLLECTION_NAME}__{build_id[:8]}" if alias_ok else COLLECTION_NAME
    if not alias_ok:
        try:
            client.delete_collection(collection_name=target_collection)
        except Exception:
            pass
    log(f"写入 Qdrant (稠密+稀疏混合) → 集合 {target_collection}")
    client.create_collection( collection_name=target_collection, vectors_config={ DENSE_VECTOR_NAME: VectorParams( size=dense_dim, distance=Distance.COSINE, ) }, sparse_vectors_config={ SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF) }, )
    for field, schema in ( (SOURCE_FIELD, PayloadSchemaType.KEYWORD), ("parent_id", PayloadSchemaType.KEYWORD), ):
        try:
            client.create_payload_index( collection_name=target_collection, field_name=field, field_schema=schema )
        except Exception as exc:
            log(f"    ⚠️  payload 索引 {field} 创建失败（过滤会退化为扫描）: {exc}")
    write_failures: list = []
    batch_starts = list(range(0, total, WRITE_BATCH))
    batch_sizes = {i: min(i + WRITE_BATCH, total) - i for i in batch_starts}

    # 线程本地客户端：ModelEmbedder 每客户端一份，per-worker 客户端 = per-worker 推理器，
    # 于是本地推理不需要跨线程互斥（见上方注释）。只用公开 API，不碰 qdrant-client 私有成员；
    # 客户端随 worker 线程存活到本次 index 结束（进程退出即回收）。
    _worker_state = threading.local()

    def _worker_client():
        c = getattr(_worker_state, "client", None)
        if c is None:
            c = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
            _worker_state.client = c
        return c

    _write_times: list = []   # 每批的 (构造, 往返) 秒数：并发度就是按这两个数派生的（见下）

    def _write_batch(i):
        # 批次内任何异常（点构造 / 稀疏 Document 拆包 / MemoryError / KeyboardInterrupt…）
        # 都必须转成错误串交回调用方：从 fut.result() 抛出会**跳过整段 write_failures 收尾**
        # （不删孤儿集合、不报"谁在服务"、连集合名都不留 —— 失效原型：自崩）。
        try:
            _t_construct = time.perf_counter()
            batch_chunks = all_chunks[i:i + WRITE_BATCH]
            batch_embeddings = embeddings[i:i + WRITE_BATCH]
            points = []
            for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
                # 稀疏与稠密取同一份文本（_index_text），不让两条通道描述不同内容
                points.append(PointStruct( id=i + j, vector={ DENSE_VECTOR_NAME: emb.tolist(), SPARSE_VECTOR_NAME: sparse_encode(_index_text(chunk)), }, payload=chunk_payload(chunk, build_id, lexicon_id, chunk_fingerprint), ))  # type: ignore[union-attr]
            _t_http = time.perf_counter()
            _err = _upsert_batch_with_retry(_worker_client(), target_collection, points)
            _write_times.append((_t_http - _t_construct, time.perf_counter() - _t_http))
            return i, _err
        except BaseException as exc:  # noqa: BLE001 - 见上：必须转成错误串，不能越过收尾
            return i, f"{type(exc).__name__}: {exc}"

    # 各批彼此独立（point id 由 batch 起始下标决定，与写入先后无关），故并发发往 Qdrant：
    # 墙钟 ≈ 批次往返 × 批数 ÷ 并发度，而非逐批之和。**每 worker 一个客户端**是这里的
    # 并发前提：qdrant-client 的 ModelEmbedder（Document 的本地拆包器）是每客户端一份且
    # 内部无锁的，共用一份客户端时多线程会互相 drain _embed_storage（IndexError / 422），
    # 旧修法用一把模块锁把整次 upsert（含 HTTP 往返）串行化 —— 4 路写批于是退化成 1 路，
    # 与"并发发往 Qdrant"这句注释相反。分开客户端后每份推理器只被自己的线程使用，
    # 本地推理天然串行，HTTP 往返真正并发。
    max_workers = INDEX_WRITE_WORKERS or qdrant_write_workers(len(batch_starts))
    log(
        f"  Qdrant 写批并发: {max_workers} 路（{len(batch_starts)} 批 × {WRITE_BATCH} 点；"
        f"Little: 目标 {QDRANT_TARGET_PPS} 点/秒 × 实测单批 {QDRANT_BATCH_RTT_S}s ÷ {WRITE_BATCH} 点/批；{hw_report()}）"
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _write_t0 = time.time()
    done_points = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_write_batch, i) for i in batch_starts]
        for fut in as_completed(futures):
            i, err = fut.result()
            completed += 1
            done_points += batch_sizes[i]
            if err:
                write_failures.append((i, err))
                log(f"    ⚠️  写入批次 [{i}, {min(i + WRITE_BATCH, total)}) 失败，其余批继续: {err}")
            if completed % 5 == 0 or completed == len(batch_starts):
                log(f"  Qdrant 写入进度: {min(done_points, total)}/{total}")
    _write_wall = time.time() - _write_t0
    if done_points:
        log(f"  写批墙钟: {_write_wall:.1f}s / {done_points} 点（{done_points / max(_write_wall, 1e-9):.0f} 点/秒，{max_workers} 路并发）")
    if _write_times:
        # 分解：构造 = 纯 Python（emb.tolist + PointStruct + 服务端 Document 包装），往返 = HTTP。
        # 两者之比正是"要几路并发"的依据 —— 构造被 GIL 串行，只有往返能真正重叠。
        _c_avg = sum(c for c, _h in _write_times) / len(_write_times)
        _h_avg = sum(h for _c2, h in _write_times) / len(_write_times)
        log(f"  写批分解: 单批构造 {_c_avg:.2f}s + 往返 {_h_avg:.2f}s（并发 {max_workers} 路）")
    write_failures.sort()
    if write_failures:
        # 半截集合是本次失败留下的孤儿：别名没切、检索不读它，而下一次 index 会用新的
        # build_id 另起集合 —— 不清理就永久占盘，且当次失败日志里连名字都不留（直到某次
        # 成功运行的 stragglers 扫描才被发现）。能删就删，删不掉给出手动回收命令。
        orphan_hint = ""
        if alias_ok:
            try:
                client.delete_collection(collection_name=target_collection)
                log(f"  已清理未完成的集合 {target_collection}（未切别名，重跑 index 会新建）")
            except Exception as cleanup_exc:
                orphan_hint = f"；孤儿集合 {target_collection} 需手动回收: curl -X DELETE {QDRANT_HOST}:{QDRANT_PORT}/collections/{target_collection}（{type(cleanup_exc).__name__}: {cleanup_exc}）"
        # 两种模式下"谁在服务"是相反的：别名模式旧集合还在并继续服务；非别名模式
        # 上面已把旧集合删掉，残缺的目标集合就是当前服务集合 —— 报错必须说清。
        serving = (
            f"集合 {target_collection} 未切别名，旧集合继续服务"
            if alias_ok
            else f"集合 {target_collection} 是非别名模式的服务集合（旧集合已在写入前删除），当前点数残缺，检索会吃到不完整结果"
        )
        raise RuntimeError(
            f"{len(write_failures)} 批写入失败（批次起始 {[fi for fi, _ in write_failures][:5]}）："
            f"{serving}；"
            f"失败示例 {write_failures[0][1]}{orphan_hint}。网络/Qdrant 恢复后重跑 index（稠密走缓存增量）。"
        )
    info = client.get_collection(collection_name=target_collection)
    # 不变量「索引点数 = 产物记录数」在这里当场把关，不只靠 stages 事后发现。
    # 非别名路径下旧集合若没删掉（delete_collection 失败/集合不存在时报错被吞），写入只覆盖
    # id 0..total-1，旧点会以 id >= total 残留。**它们不是"检索不到"**：检索是做向量搜索、
    # 没有任何 id 过滤，残留点会被正常召回并返回陈旧正文（旧文案把方向说反了）。
    # 所以这里不是报错了事，而是**按实现清理**：先删 id >= total 的残留点，再复核点数。
    written = client.count(collection_name=target_collection, exact=True).count
    log(f"  Qdrant 写入完成: {written} 条")
    if written != total:
        residual = []
        offset = None
        while True:
            pts, offset = client.scroll( collection_name=target_collection, limit=WRITE_BATCH, offset=offset, with_payload=False, with_vectors=False )
            residual.extend(p.id for p in pts if isinstance(p.id, int) and p.id >= total)
            if offset is None:
                break
        if residual:
            client.delete(collection_name=target_collection, points_selector=residual)
            written = client.count(collection_name=target_collection, exact=True).count
            log(f"  已清理 {len(residual)} 个 id >= {total} 的残留点（旧集合未删净），复核点数: {written}")
    if written != total:
        raise RuntimeError(
            f"集合 {target_collection} 点数 {written} != 分块记录数 {total}："
            "残留点（id >= 分块记录数）会被检索正常召回并返回陈旧正文（检索不做 id 过滤）；"
            "清理后仍不一致，请删掉该集合后重跑 index，或启用 RAG_USE_COLLECTION_ALIAS"
        )
    coll_dense_size = next(iter(info.config.params.vectors.values())).size  # type: ignore[union-attr]
    # 显式 raise 而非 assert（-O 会剥离 assert，维度错配就静默放行了）
    if coll_dense_size != dense_dim:
        raise RuntimeError(f"集合稠密维度 {coll_dense_size} 与本次编码维度 {dense_dim} 不一致（集合名 {target_collection}）")
    log(f"  集合向量空间: 稠密 {list(info.config.params.vectors.keys())} (size={coll_dense_size}) + 稀疏 {list(info.config.params.sparse_vectors.keys())}")  # type: ignore[union-attr]
    log("  索引构建完成: 稠密向量 + 稀疏向量（Qdrant 内置 BM25，分词在服务端 multilingual）")
    if alias_ok:
        # 别名切换存在竞态窗口：get_aliases 到 update_collection_aliases 间其他进程可能修改别名。
        # 采用重试 + 验证策略：只重试一次（共 2 次尝试），每次切换后验证别名确实指向新集合。
        # 但 DeleteAlias + CreateAlias 是**两个非事务操作**：Delete 成功而 Create 失败（重试
        # 耗尽）会把别名删空 —— 检索端下一个请求全线挂，而旧文案写的是"检索端仍指向旧集合"，
        # 方向正好相反（失效原型：状态 + 报错指错方向）。故：①失败分支按"本轮之前的目标"
        # 先发一次恢复性 CreateAlias（失败仅告警）；②文案按 client.get_aliases() 的**实际**
        # 结果组织；③成功路径用"本轮之前的目标"删旧集合 —— 重试后循环内变量可能已变成
        # 本次的 target（切成功但读回旧缓存），那时真正的旧集合就永远不会被删。
        prev_alias_target = None   # 本轮之前别名指向的集合，只在第一次尝试前读一次
        switched = False
        max_retries = 2
        for attempt in range(max_retries):
            try:
                current_target = next(
                    (a.collection_name for a in client.get_aliases().aliases if a.alias_name == COLLECTION_ALIAS),
                    None,
                )
                if attempt == 0:
                    prev_alias_target = current_target
                # 空别名冷启动不能对不存在的 alias 提交 Delete（部分 Qdrant 版本整批失败）；有旧别名才删，Create 恒提交。
                ops: list = []
                if current_target is not None:
                    ops.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=COLLECTION_ALIAS)))
                ops.append(CreateAliasOperation(create_alias=CreateAlias(collection_name=target_collection, alias_name=COLLECTION_ALIAS)))
                client.update_collection_aliases(change_aliases_operations=ops)
                # 验证别名切换成功
                aliases_after = client.get_aliases().aliases
                new_target = next((a.collection_name for a in aliases_after if a.alias_name == COLLECTION_ALIAS), None)
                if new_target == target_collection:
                    switched = True
                    log(f"  别名已切换: {COLLECTION_ALIAS} → {target_collection}" + (f"（原 {prev_alias_target}）" if prev_alias_target else "（冷启动，无旧别名）"))
                    break
                log(f"  ⚠️  别名切换验证失败（尝试 {attempt + 1}/{max_retries}）：别名指向 {new_target}，期望 {target_collection}")
            except Exception as exc:
                log(f"  ⚠️  别名切换异常（尝试 {attempt + 1}/{max_retries}）：{exc}")
        if not switched:
            actual = _current_alias_target(client)
            if actual is None and prev_alias_target:
                # 别名此刻可能是被本轮 DeleteAlias 删掉的：先把它指回旧集合，别让检索裸奔。
                try:
                    client.update_collection_aliases(change_aliases_operations=[CreateAliasOperation(create_alias=CreateAlias(collection_name=prev_alias_target, alias_name=COLLECTION_ALIAS))])
                    actual = prev_alias_target
                    log(f"  ♻️  别名已恢复到 {prev_alias_target}（本轮切换失败，先保证检索端有可用集合）")
                except Exception as exc:
                    log(f"  ⚠️  恢复别名到 {prev_alias_target} 失败: {type(exc).__name__}: {exc}")
                    actual = _current_alias_target(client)
            raise RuntimeError(
                f"别名切换失败：新集合 {target_collection} 已建好；别名 {COLLECTION_ALIAS} 实际指向 "
                f"{actual if actual else '（不存在 —— 检索端当前不可用，需人工恢复）'}。"
                f"修好后重跑 index（稠密走缓存增量），或直接用 Qdrant 别名 API 指回可用集合。"
            )
        if prev_alias_target and prev_alias_target != target_collection:
            try:
                client.delete_collection(collection_name=prev_alias_target)
                log(f"  旧集合已清理: {prev_alias_target}")
            except Exception as exc:
                log(f"    ⚠️  旧集合 {prev_alias_target} 删除失败（仅占磁盘，检索不受影响）: {exc}")
        try:
            stragglers = [ x.name for x in client.get_collections().collections if x.name != target_collection and x.name.split("__")[0] == COLLECTION_NAME ]
        except Exception:
            stragglers = []
        if stragglers:
            log(f"  ⚠️  检测到 {len(stragglers)} 个同名前缀历史集合（检索已不读）: {stragglers}")
            log("     确认新索引可用后可回收磁盘（保留仅占空间）：")
            for name in stragglers:
                log(f"       curl -X DELETE {QDRANT_HOST}:{QDRANT_PORT}/collections/{name}")
    else:
        log(f"  ⚠️  未使用别名：检索端读 {COLLECTION_NAME}，重建期间该集合会短暂不可用")
    log(f"  索引建造标识: {build_id}")
    log(f"总耗时: {time.time() - t0:.1f} 秒")


# ── 检索原语 ──
def rerank_indices(query, documents, tokenizer, model, device, top_k=TOP_K, batch_size=RERANK_BATCH, max_length=RERANK_MAX_LENGTH):
    import numpy as np
    import torch
    if not documents:
        return []
    all_scores = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        pairs = [(query, d) for d in batch]
        inputs = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
        all_scores.extend(logits.float().cpu().numpy().tolist())
    order = np.argsort(all_scores)[::-1]
    ranked = [(int(i), float(all_scores[i])) for i in order]
    return ranked[:top_k]


def weighted_rrf(rankings, k=RRF_K, limit=None):
    scores: dict = {}
    resources: dict = {}
    first_seen: dict = {}
    seq = 0
    for name, ids, weight in rankings:
        if not weight:
            continue
        for rank, doc_id in enumerate(ids, 1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
                resources[doc_id] = []
                first_seen[doc_id] = seq
                seq += 1
            scores[doc_id] += weight / (k + rank)
            resources[doc_id].append(f"{name}#{rank}")
    items = sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    if limit is not None:
        items = items[:limit]
    return [(doc_id, score, resources[doc_id]) for doc_id, score in items]


def _qdrant_dense_hits(client, collection, dense_query, limit, query_filter=None, with_payload=False):
    return client.query_points( collection_name=collection, query=dense_query, using=DENSE_VECTOR_NAME, limit=limit, with_payload=with_payload, query_filter=query_filter, ).points


def _qdrant_sparse_hits(client, collection, sparse_query, limit, query_filter=None, with_payload=False):
    # 本地推理器（qdrant-client 的 ModelEmbedder）每客户端一份且内部无锁：带 Document 的
    # 请求必须串行，否则并发时 pop(0) 竞争 -> IndexError，或两份 Document 互换（稀疏查询
    # 拿另一路提问的文本去打分）。写入口早有这把锁，检索侧这条一直裸跑；失败又被
    # tolerate_sparse_error 吞成"退化为纯稠密"、页面无提示（静默丢召回）。
    with _QDRANT_INFERENCE_LOCK:
        return client.query_points( collection_name=collection, query=sparse_query, using=SPARSE_VECTOR_NAME, limit=limit, with_payload=with_payload, query_filter=query_filter, ).points


def _sparse_recall_note(exc, sparse_query):
    term_count = len((sparse_query.text or '').split())
    return f"稀疏通道召回失败，本次退化为纯稠密检索: {type(exc).__name__}: {exc}；分词 {term_count} 个 term"


def dual_channel_recall(client, collection, dense_query, sparse_query, limit, query_filter=None, with_payload=False, tolerate_sparse_error=False, use_dense=True, use_sparse=True):
    from concurrent.futures import ThreadPoolExecutor
    dense_hits = []

    if not use_sparse:
        return (_qdrant_dense_hits(client, collection, dense_query, limit, query_filter, with_payload) if use_dense else []), [], ""
    sparse_hits = []

    # 双通道并发：两个 Qdrant 往返并行发，墙钟 = max(dense, sparse) 而非和。
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_dense = pool.submit(_qdrant_dense_hits, client, collection, dense_query, limit, query_filter, with_payload) if use_dense else None
        future_sparse = pool.submit(_qdrant_sparse_hits, client, collection, sparse_query, limit, query_filter, with_payload)
        if future_dense is not None:
            dense_hits = future_dense.result()
        try:
            sparse_hits = future_sparse.result()
        except Exception as exc:
            if not tolerate_sparse_error:
                raise
            return dense_hits, [], _sparse_recall_note(exc, sparse_query)
    return dense_hits, sparse_hits, ""


def _degraded(steps, note):
    log(f"⚠️  {note}")
    if steps is not None:
        steps.setdefault(STEP_DEGRADE, []).append(note)


def channel_rankings(dense_hits, sparse_hits, suffix=""):
    return [ (name, ids, weight) for name, ids, weight in ( (f"dense{suffix}", [p.id for p in dense_hits], RRF_DENSE_WEIGHT), (f"sparse{suffix}", [p.id for p in sparse_hits], RRF_SPARSE_WEIGHT), ) if weight ]


def rerank_text_for(candidate):
    if RERANK_ON == "parent":
        return candidate.get("parent_text") or candidate.get("child_text", "")
    return candidate.get("contextual_text") or candidate.get("child_text", "")


def dedup_candidates_by_parent(candidates):
    kept, folded, seen = [], [], set()
    for c in candidates:
        key = c.get("parent_id") or c.get("parent_text") or c.get("id")
        if key in seen:
            folded.append(c)
            continue
        seen.add(key)
        kept.append(c)
    return kept, folded


def confidence_signal(scores, sources=None):
    """检索分数概览：只统计、不判定（不产生拒答）。"""
    vals = [float(s) for s in scores]
    n_sources = len({s for s in (sources or []) if s}) if sources is not None else None
    if not vals:
        return {"top": None, "median_rest": None, "spread": None, "ratio": None, "mean": None, "min": None, "n": 0, "n_sources": n_sources}
    top = max(vals)
    rest = sorted(vals)[:-1]
    if rest:
        mid = len(rest) // 2
        median_rest = (rest[mid] if len(rest) % 2 else (rest[mid - 1] + rest[mid]) / 2)
    else:
        median_rest = top
    spread = top - median_rest
    mean = sum(vals) / len(vals)
    return { "top": top, "median_rest": median_rest, "spread": spread, "ratio": spread / (abs(top) + 1.0), "mean": mean, "min": min(vals), "n": len(vals), "n_sources": n_sources }


def results_confidence(results):
    return confidence_signal( [r.get("rerank_score") for r in results], [_source_of(r) for r in results], )


def resolve_collection_name(client):
    if not USE_COLLECTION_ALIAS:
        return COLLECTION_NAME
    state = _state_ref()
    # 正缓存：别名解析成功后记住结果（缓存的是别名字符串本身，Qdrant 查询时再路由到
    # 当前集合 —— reindex 切别名自动跟随，无旧集合坑）。只缓存成功，失败每次重试，
    # 且把"本次处于降级"记到 state：调用方（_qdrant_gate/cmd_stages）据它对外告警，
    # 避免"看起来正常、实际在直读旧集合"的静默状态。
    if state.alias_resolved.get("name"):
        return state.alias_resolved["name"]
    try:
        aliases = client.get_aliases().aliases
    except Exception as exc:
        state.alias_degraded = True
        log(f"⚠️  读取集合别名失败，回退直接访问 {COLLECTION_NAME}: {exc}")
        return COLLECTION_NAME
    names = {a.alias_name for a in aliases}
    if COLLECTION_ALIAS in names:
        state.alias_resolved["name"] = COLLECTION_ALIAS
        state.alias_degraded = False
        return COLLECTION_ALIAS
    state.alias_degraded = True
    return COLLECTION_NAME


# ── 生成装配：上下文、预算、裁剪 ──
# 资料是否带编号与来源：带编号时模型能按 [n] 引用，客户端能把 [n] 回指到来源清单
# （context_sources[].n 与这里的 n 一一对应）；来源名还让模型能判断这条资料与提问是否
# 同域。关掉即回到"纯 parent_text"旧行为。
CONTEXT_LABELS = env_bool("RAG_CONTEXT_LABELS", True)
# 资料边界标签（与编号/来源正交）：每条资料包一层 <document>…</document>。资料消息与
# 本轮提问同为 user 角色，全裸模式（LABELS=0 且不发 system）下资料正文里的祈使句
# （"请听我说"之类）会被模型当成对它的指令——标签是"这是数据不是指令"的唯一防线。
# 要连标签也裸掉（回到纯拼接）设 RAG_CONTEXT_TAGS=0。
CONTEXT_TAGS = env_bool("RAG_CONTEXT_TAGS", True)
# 资料元数据：以**结构化标签**给出（模型最易解析的形式）——
#   <document index="1" source="书一" id="a1" chunk="3/12" score="0.91">
#   正文……
#   </document>
# 属性名固定英文、值一律转义，缺字段即省略该属性。与 RAG_CONTEXT_LABELS / RAG_CONTEXT_TAGS
# 的关系：META=1 时以这个「带属性的 <document>」输出并**接管包裹**（不再看 TAGS/LABELS）；
# META=0 时才回到 LABELS（[n]（来源）标头）与 TAGS（裸 <document> 包裹）的旧行为。
CONTEXT_META = env_bool("RAG_CONTEXT_META", True)


def _context_meta_open_tag(i, r, source):
    """单条资料的结构化开始标签：<document index=.. source=.. id=.. chunk=.. score=..>。

    属性值走 quoteattr（正确转义引号/&/<），来源名或 id 里含特殊字符也不会破坏结构。
    缺字段省略对应属性；index 恒有（资料清单/引用回指靠它）。
    """
    from xml.sax.saxutils import quoteattr
    attrs = [("index", str(i))]
    if source:
        attrs.append(("source", str(source)))
    if r.get("id"):
        attrs.append(("id", str(r.get("id"))))
    ci, ct = r.get("chunk_index"), r.get("total_chunks")
    if ct:
        attrs.append(("chunk", f"{int(ci or 0) + 1}/{int(ct)}"))
    score = r.get("rerank_score")
    if score is not None:
        attrs.append(("score", f"{float(score):.2f}"))
    return "<document " + " ".join(f"{k}={quoteattr(v)}" for k, v in attrs) + ">"


def context_blocks(results):
    """按检索顺序返回每一条资料的独立块（与资料装配的拼接单元一致）。

    独立成块是为了让生成侧按**整条资料**从尾部让位：截掉哪几条，sources 就只报哪几条，
    诊断不会比模型实际看到的多。
    """
    if not results:
        return []
    if not CONTEXT_LABELS and not CONTEXT_TAGS and not CONTEXT_META:
        return [(r.get("parent_text") or "") for r in results]
    blocks = []
    for i, r in enumerate(results, 1):
        source = _source_of(r)
        text = r.get("parent_text") or ""
        if CONTEXT_META:
            # 结构化档：一条资料 = 一个带属性 <document> 块，元数据与正文同处一个元素。
            block = f"{_context_meta_open_tag(i, r, source)}\n{text}\n</document>"
        else:
            head = ""
            if CONTEXT_LABELS:
                head = f"[{i}]" + (f"（{SOURCE_LABEL}：{source}）" if source else "")
            block = "\n".join(part for part in (head, text) if part)
            if CONTEXT_TAGS:
                block = f"<document>\n{block}\n</document>"
        blocks.append(block)
    return blocks


def context_sources(results):
    out = []
    for i, r in enumerate(results, 1):
        source = _source_of(r)
        row = { "n": i, "id": r.get("id", ""), "rerank_score": r.get("rerank_score"), "source": source, "source_label": SOURCE_LABEL }
        if SOURCE_FIELD != "source":
            row[SOURCE_FIELD] = source
        out.append(row)
    return out


# 不发 system：system prompt 已整体移除，恒为空字符串（原回答规则与领域身份句不再可配）。
# 检索资料**恒以 user 角色**随本轮提问发出：RAG_CONTEXT_ROLE 已删除，不再有"资料寄生成
# system"的入口（README「核心不变量：资料不寄生 system」）。
RAG_USE_CONTEXT = env_bool("RAG_USE_CONTEXT", True)


# 下限 0：负值会让输入上界 I = NUM_CTX - RESERVE **超过真实窗口**（下面的交叉校验只拦
# I ≤ 0，拦不住"上界比窗口还大"），装配于是把超长 prompt 发出去，由服务端移位悄悄丢前文。
GEN_INPUT_RESERVE = env_int_min("RAG_INPUT_RESERVE", 256, 0)
# 交叉校验（启动即报错）：输入上界 I = NUM_CTX - RESERVE。I ≤ 0 时装配会把提问截成
# 空串，最后一条 user 消息内容为空 —— 模型收到空提问，而启动时没有任何人报错。
if NUM_CTX <= GEN_INPUT_RESERVE:
    raise ValueError(
        f"RAG_NUM_CTX={NUM_CTX} 必须大于 RAG_INPUT_RESERVE={GEN_INPUT_RESERVE}："
        f"否则输入上界 ≤ 0，装配会截出空提问（请调大 RAG_NUM_CTX 或调小 RAG_INPUT_RESERVE）"
    )
# 生成历史的硬上限（token）：把"历史最多塞多少"从窗口余量里独立出来。
# 0=不设上限（行为与不带该配置完全一致，只受窗口余量约束）。
# 生效形式是 min(窗口余量, 本上限)：本上限只做第二道、更严的约束，永不越过窗口安全边界。
GEN_HISTORY_MAX_TOKENS = env_int_min("RAG_GEN_HISTORY_MAX_TOKENS", 0, 0)
# 原值"……（已按上下文预算截断）"；按全裸 prompt 配置置空：截断无痕，空内容条照旧被整条丢弃。
_TRUNCATION_MARK = ""
# 生成侧精确计数。Ollama 不暴露分词接口（/api/tokenize 恒 404），因此直接用模型自带的
# tokenizer 词表——这是"对应模型自己的方式"，与生成端同源、无需照抄任何切分规则：
#   1) RAG_GEN_TOKENIZER_PATH 指定的 tokenizer.json / HF 目录（推荐；本项目 tokenizer.json
#      由 tools/build_gen_tokenizer.py 从模型 GGUF 重建，词表/merges/切分规则均来自模型自身）；
#   2) 兜底：按 MODEL 的 Ollama manifest 定位 GGUF，用 llama-cpp-python vocab_only 分词；
#   3) 保守估算：1 字=1 token 偏高、不越界但少用窗口。
# 任一级失败只降级，不中断生成；日志如实说明用了哪一级。
GEN_TOKENIZER_PATH = os.getenv("RAG_GEN_TOKENIZER_PATH", "")
# 显式指定生成模型的 GGUF；留空则按 MODEL 自动定位。
GEN_GGUF_PATH = os.getenv("RAG_GEN_GGUF_PATH", "")
# Ollama 模型目录（manifests/ + blobs/）。容器里由镜像 OLLAMA_MODELS 指到烘焙好的
# /app/ollama_models，本地原生默认 ~/.ollama/models。
OLLAMA_MODELS_DIR = os.getenv("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models")
# 懒加载必须上锁：无锁时并发首请求会一个走"加载中=估算"、一个等完拿到真 tokenizer，
# 同一次生成装配里前后两段计数各按一套口径（预算裁剪与硬上界分叉）。缓存放 _RuntimeState：
# 预热线程/API 请求/Streamlit rerun 之间共用一个进程内的同一份。
def _find_ollama_gguf(model, models_dir=OLLAMA_MODELS_DIR):
    """按 Ollama manifest 定位 model 层的 GGUF blob。

    manifest 布局对自 Ollama registry：<models_dir>/manifests/registry.ollama.ai/library/
    <name>/<tag>；取 mediaType 以 image.model 结尾的层 digest，拼成 blobs/sha256-<digest>。
    定位失败抛异常，由调用方降级（不吞错，日志里能看到具体原因）。
    """
    name, tag = model.split(":", 1) if ":" in model else (model, "latest")
    manifest = os.path.join(models_dir, "manifests", "registry.ollama.ai", "library", name, tag)
    with open(manifest, encoding="utf-8") as fh:
        layers = json.load(fh)["layers"]
    digest = next(layer["digest"].split(":", 1)[1] for layer in layers
                  if layer["mediaType"].endswith("image.model"))
    return os.path.join(models_dir, "blobs", "sha256-" + digest)


class _LlamaCppTokenizer:
    """把 Ollama 背后的 llama.cpp 直接当 tokenizer：零切分规则、与生成端逐字节一致。

    vocab_only=True 只读词表、不加载权重张量；encode 走 llama_tokenize（与 Ollama 运行期
    同一份实现，故计数精确）。对外只暴露 encode()（与 tokenizers / transformers 对齐），
    _count_tokens 沿用同一入口。首次构造有一次性词表加载开销，故由 caller 懒加载。
    """

    def __init__(self, gguf_path):
        from llama_cpp import Llama
        self._llm = Llama(model_path=gguf_path, vocab_only=True, verbose=False)

    def encode(self, text, add_special_tokens=False):
        # add_bos=False 与 tokenizers 路径的 add_special_tokens=False 对齐：本函数按"条"计数，
        # BOS 属于整条 prompt、由服务端模板加一次，不在这里重复计。
        return self._llm.tokenize(text.encode("utf-8"), add_bos=False)


def _load_gen_tokenizer(path):
    """加载生成模型 tokenizer（官方标准件优先，所有模型通用）：

    - 文件（tokenizer.json）→ tokenizers.Tokenizer.from_file；
    - 目录：先读目录里的 tokenizer.json（tokenizers，最轻、不依赖 transformers），
      再退 transformers.AutoTokenizer（兼容 SentencePiece 等没有 tokenizer.json 的模型）。

    两者都只读模型官方的分词文件，不做任何 GGUF/词表重建；与 transformers 的完整依赖解耦。
    """
    if os.path.isfile(path):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(path)
    _tj = os.path.join(path, "tokenizer.json")
    if os.path.isfile(_tj):
        try:
            from tokenizers import Tokenizer
            return Tokenizer.from_file(_tj)
        except Exception:
            pass  # 旧版 tokenizers 读不了新格式时，先试兼容副本，再走 AutoTokenizer
    _tj_legacy = os.path.join(path, "tokenizer.legacy.json")
    if os.path.isfile(_tj_legacy):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(_tj_legacy)
            log("ℹ️  官方 tokenizer 与当前 tokenizers 版本不兼容，使用兼容派生副本（同词表，计数差 ~0.006%）")
            return tok
        except Exception:
            pass
    last = None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path)
    except Exception as exc:
        last = exc
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(_tj)
    except Exception:
        raise last if last is not None else RuntimeError("tokenizer 加载失败")


def _count_tokens(tok, text):
    """统一两种返回：transformers 的 encode → list[int]；tokenizers 的 encode → Encoding。"""
    ids = tok.encode(text, add_special_tokens=False)
    if isinstance(ids, (list, tuple)):
        return len(ids)
    return len(ids.ids)


def _load_gen_tokenizer_default():
    """按优先级取生成侧 tokenizer：模型自带 tokenizer.json → GGUF llama.cpp → None（估算）。

    每一级都"加载即验证"（编一次冒烟串）：编不了就当该级失败、降级到下一级，绝不让坏
    tokenizer 挂上去导致每次调用静默出错。优先 tokenizer.json 是刻意的：它是从模型自身
    GGUF 重建的标准分词件（tools/build_gen_tokenizer.py，可重放），与生成端同源，也不依赖
    llama.cpp wheel 对新模型的支持程度。
    """
    if GEN_TOKENIZER_PATH:
        # 来源一致性校验（2026-10-24）：tokenizer.json 是派生数据；换模型后若忘记重跑
        # tools/build_gen_tokenizer.py，这里仍会指向旧模型的词表 → token 预算错算。
        # 用 meta 里记录的 GGUF sha256 与当前 MODEL 的 GGUF blob 比对（文件名即 digest，
        # 不重算 1.5GB 文件），不一致就**不用**旧 tokenizer，退回 GGUF/估算（保守安全）。
        _tok_ok = True
        if os.path.isdir(GEN_TOKENIZER_PATH):
            _meta_path = os.path.join(GEN_TOKENIZER_PATH, "tokenizer.meta.json")
            _tok_file = os.path.join(GEN_TOKENIZER_PATH, "tokenizer.json")
        else:
            _meta_path = os.path.splitext(GEN_TOKENIZER_PATH)[0] + ".meta.json"
            _tok_file = GEN_TOKENIZER_PATH
        if os.path.isfile(_tok_file) and os.path.isfile(_meta_path):
            try:
                with open(_meta_path, encoding="utf-8") as fh:
                    _tmeta = json.load(fh)
                _want = str(_tmeta.get("gguf_sha256") or "")
                _meta_model = str(_tmeta.get("model") or "").strip().lower()
                _cur_model = str(MODEL or "").strip().lower()
                _cur_gguf = GEN_GGUF_PATH
                if not _cur_gguf:
                    try:
                        _cur_gguf = _find_ollama_gguf(MODEL)
                    except Exception:
                        _cur_gguf = ""
                _cur = os.path.basename(_cur_gguf or "")
                _cur = _cur[len("sha256-"):] if _cur.startswith("sha256-") else ""
                _bad_reason = ""
                if _want and _cur and _want != _cur:
                    _bad_reason = f"GGUF 不一致（meta={_want[:12]}… 当前={_cur[:12]}…）"
                elif _meta_model and _cur_model and _meta_model != _cur_model:
                    _bad_reason = f"模型名不一致（meta={_meta_model!r} 当前={_cur_model!r}）"
                if _bad_reason:
                    _tok_ok = False
                    log(f"⚠️  tokenizer 与当前 MODEL 不匹配：{_bad_reason}；"
                        f"为避免 token 预算错算，改用 GGUF/估算兜底；请更新 artifacts/gen_tokenizer/ 的官方 tokenizer")
                _want_sha = str(_tmeta.get("tokenizer_sha256") or "")
                if _tok_ok and _want_sha:
                    import hashlib
                    with open(_tok_file, "rb") as _fh:
                        _got_sha = hashlib.sha256(_fh.read()).hexdigest()
                    if _want_sha != _got_sha:
                        _tok_ok = False
                        log(f"⚠️  tokenizer.json 校验失败（meta={_want_sha[:12]}… 实际={_got_sha[:12]}…），改用 GGUF/估算兜底")
            except Exception as exc:
                log(f"ℹ️  tokenizer 来源 meta 读取失败（{_meta_path}），跳过一致性校验: {type(exc).__name__}: {exc}")
        if _tok_ok:
            try:
                tok = _load_gen_tokenizer(GEN_TOKENIZER_PATH)
                _count_tokens(tok, "冒烟")
                log(f"✅ 生成 tokenizer：{GEN_TOKENIZER_PATH}（模型自带）")
                return tok
            except Exception as exc:
                log(f"⚠️  tokenizer.json 加载失败（{GEN_TOKENIZER_PATH}），尝试 GGUF 兜底: {type(exc).__name__}: {exc}")
                log("   提示：官方 tokenizer.json 需要 tokenizers>=0.21.2（requirements.txt 已锁）；"
                    "本地环境过旧时请先 pip install -r requirements.txt")
    gguf = GEN_GGUF_PATH
    if not gguf:
        try:
            gguf = _find_ollama_gguf(MODEL)
        except Exception:
            gguf = None
    if gguf and os.path.isfile(gguf):
        try:
            tok = _LlamaCppTokenizer(gguf)
            _count_tokens(tok, "冒烟")
            log(f"✅ 生成 tokenizer：llama.cpp（GGUF={gguf}）")
            return tok
        except Exception as exc:
            # 兜底失败是预期分支（wheel 对新 pre-tokenizer 支持滞后）：只降级、不打 ⚠️。
            log(f"ℹ️  GGUF 分词兜底不可用（{type(exc).__name__}: {exc}），改用估算计数")
    return None


def _get_gen_tokenizer():
    state = _state_ref()
    # 快路径无锁读引用：CPython 下属性读是原子的，tokenizer 一旦挂好就是常量，后续
    # estimate_tokens 紧循环（_cut_to_tokens 二分、message_tokens 逐条）不再每步取锁。
    # 锁只保护"首次加载"，失败（gen_tokenizer=None）仍退 lock 路径、不重试。
    tok = state.gen_tokenizer
    if tok is not None:
        return tok
    with state.gen_tokenizer_lock:
        if state.gen_tokenizer_tried:
            return state.gen_tokenizer
        state.gen_tokenizer_tried = True
        state.gen_tokenizer = _load_gen_tokenizer_default()
        return state.gen_tokenizer


def estimate_tokens(text):
    """生成侧 token 计数：配了生成模型 tokenizer 就用真值，否则用保守估算。

    所有生成装配的预算/裁剪/硬上界都只经过这一个函数，换 tokenizer 不会出现"预算按 A
    算、裁剪按 B 算"的分叉。真 tokenizer 对纯空白可能返回 0，无影响：空内容条本就被
    history_to_messages 过滤。
    """
    if not text:
        return 0
    tok = _get_gen_tokenizer()
    if tok is not None:
        return _count_tokens(tok, text)
    cjk = sum(1 for ch in text if ord(ch) >= 0x2E80)
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def message_tokens(messages):
    return sum(estimate_tokens(m.get("content") or "") for m in messages)


def _truncate_history_to_budget(msgs, budget):
    """历史超预算：按**整轮**淘汰 —— 从最旧的 user 起，到下一个 user 之前，整轮**全部**丢。

    一轮 = 一条 user + 其后的所有连续 assistant（通常一条）。轮是语义最小单位：半个提问 +
    半个回答只会给模型看残句，所以历史**不做 token 级部分截断**——要么整轮留、要么整轮走。
    队首恒为 user（孤儿 assistant 单独清掉）。就地修改 msgs（调用方已持有副本）。
    返回 (丢弃条数, 是否截断, 被截断的是否 user) —— 后两者恒为 False，只为统计兼容。
    """
    total = message_tokens(msgs)
    dropped = 0
    while msgs and total > budget:
        end = 1
        while end < len(msgs) and msgs[end].get("role") != "user":
            end += 1
        for m in msgs[:end]:
            total -= estimate_tokens(m.get("content") or "")
        del msgs[:end]
        dropped += end
    # 队首恒为 user（整轮淘汰已保证；防御孤儿残留）
    while msgs and msgs[0].get("role") != "user":
        msgs.pop(0)
        dropped += 1
    return dropped, False, False


def build_generation_messages(system, history, query, max_history_tokens, context=None, input_cap=None, context_blocks=None):
    budget = max(0, int(max_history_tokens))
    msgs = [dict(m) for m in history_to_messages(history)]
    # 调用方可能把本轮提问也塞进 history（漏了 [:-1]），甚至连续塞了多份。必须剥干净：
    # 否则模型会收到同一问题两份。本轮提问只在末尾追加这一次。
    while True:
        msgs, dropped = strip_current_turn(msgs, query)
        if not dropped:
            break
    # 超限整条丢弃；截断后首条历史恒为 user。
    dropped_messages, truncated, truncated_user = _truncate_history_to_budget(msgs, budget)
    context = context or ""
    ctx_tokens = estimate_tokens(context) if context else 0
    # ── 整条 prompt 的硬上界：估算 token 恒 ≤ input_cap ──
    # 历史按整轮淘汰（见 _truncate_history_to_budget）；锚点（system→资料→提问）才做 token
    # 级截取。永不抛异常或整体放弃；提问只截前缀、绝不整条删掉，保证最后一条 user 消息存在。
    cap = None if input_cap is None else max(0, int(input_cap))
    cap_trimmed = frame_truncated = context_truncated = query_truncated = context_partial = False
    # 供 sources 对齐：模型实际收到几条资料；None=调用方没给块信息，无法对齐
    context_docs_kept = len(context_blocks) if context_blocks is not None else None
    # 历史 token 无条件先计一次：返回汇总 used_tokens 要用，且下面 cap 分支复用同一
    # 口径。只在 cap 分支内赋值会让 input_cap=None 的默认参数路径在汇总处
    # UnboundLocalError —— 默认路径一用就崩的潜在缺陷。
    h_tok = message_tokens(msgs)
    if cap is not None:
        sys_tok = estimate_tokens(system or "")
        q_tok = estimate_tokens(query)
        # 各段计数只算一次：预算裁剪与硬上界逐段共用同一口径，不会"这段按 A 算、那段按 B 算"。
        # 历史是真正的滑动队列：先按"上界扣掉 system+资料+提问"的剩余额度再 FIFO 一次。
        # 这样硬上界覆盖整条 prompt（不只是历史）——即使调用方给的 max_history_tokens
        # 比窗口还宽，也不会越界。
        body_cap = cap - sys_tok - ctx_tokens - q_tok
        if h_tok > body_cap:
            cap_trimmed = True
            extra_dropped, extra_trunc, extra_user = _truncate_history_to_budget(msgs, max(0, body_cap))
            h_tok = message_tokens(msgs)
            dropped_messages += extra_dropped
            truncated = truncated or extra_trunc
            truncated_user = truncated_user or extra_user
        # 历史让位后锚点（system+资料+提问）自身仍超上界：继续按 FIFO（system→资料→提问）
        # 截取到 token 粒度。提问只截前缀，保证"有输入"。
        over = (sys_tok + ctx_tokens + h_tok + q_tok) - cap
        if over > 0:
            cap_trimmed = True
            if system:
                new_sys = _cut_to_tokens(system, max(0, sys_tok - over))
                over -= sys_tok - estimate_tokens(new_sys)
                frame_truncated = True
                system, sys_tok = new_sys, estimate_tokens(new_sys)
            if over > 0 and context:
                context_truncated = True
                # 先按整条资料从尾部丢：sources 与模型实际看到的一一对应。只有单条资料
                # 自身仍超上界时才退回 token 级前缀截取（保留 [n] 标头）。
                kept = list(context_blocks) if context_blocks else []
                if kept:
                    # 逐块 token 预计算：去掉最后一块时同步扣掉它自己的 token 与间距
                    # 分隔符，O(1) 内联递减，避免每步重复 join+估算整条。
                    # ★ 判据必须用**真实拼接**的计数：estimate_tokens 不可加（BPE 会跨边界合并，
                    #   估算档还逐块各自取整），所以 sum(块)+n*sep 与整条拼接的 token 数并不相等
                    #   （实测本仓库 tokenizer：k=2 为 646/645、k=5 为 1731/1727）。旧写法用加性
                    #   "运行计数"当判据、再把它与实际拼接比对后 raise —— 只要有 ≥2 条资料且触发裁剪，
                    #   这条路径必然自崩，把整轮打成 RuntimeError（本质是记账自检先崩）。
                    #   候选 ≤ RERANK_TOP_K(20) 条，每步重算一次 join+估算的成本可忽略。
                    k = len(kept)
                    while k > 1:
                        k -= 1
                        if sys_tok + h_tok + q_tok + estimate_tokens("\n\n".join(kept[:k])) <= cap:
                            break
                    kept = kept[:k]
                    context = "\n\n".join(kept)
                    ctx_tokens = estimate_tokens(context)
                    allow = max(0, cap - sys_tok - h_tok - q_tok)
                    if ctx_tokens > allow:
                        context = _cut_to_tokens(context, allow)
                        ctx_tokens = estimate_tokens(context)
                        context_partial = True
                    context_docs_kept = len(kept) if context else 0
                else:
                    ctx_tokens = estimate_tokens(context)
                    context = _cut_to_tokens(context, max(0, ctx_tokens - over))
                    ctx_tokens = estimate_tokens(context)
                over = (sys_tok + ctx_tokens + h_tok + q_tok) - cap
            if over > 0 and query:
                query = _cut_to_tokens(query, max(0, q_tok - over))
                query_truncated = True
    out = []
    # 资料恒以 user 角色承载（RAG_CONTEXT_ROLE 已删除）：不与历史/本轮提问分叉出别的
    # 角色，也就没有"排在历史之后的 system 消息被 chat template 静默丢弃"这条失效路径。
    # system 恒为空（本仓库不发 system prompt），故这里实际不会产出 system 消息。
    if system:
        out.append({"role": "system", "content": system})
    out += msgs
    if context:
        out.append({"role": "user", "content": context})
    out.append({"role": "user", "content": query})
    # 硬上界不变量：整条 prompt 估算 token 恒 ≤ 输入上界（README §10 明文）。
    # raise 而非 assert：-O 下不失效；估算口径与装配同源，正常路径恒成立，越界即装配 bug。
    if cap is not None and message_tokens(out) > cap:
        raise RuntimeError(f"整条 prompt 估算 {message_tokens(out)} tok 超过输入上界 {cap}")
    return { "messages": out, "system_text": system, "context_text": context, "dropped_messages": dropped_messages, "used_tokens": h_tok, "context_tokens": ctx_tokens, "budget": budget, "truncated": truncated, "truncated_user": truncated_user, "input_cap": cap, "cap_trimmed": cap_trimmed, "frame_truncated": frame_truncated, "context_truncated": context_truncated, "context_partial": context_partial, "context_docs_kept": context_docs_kept, "query_truncated": query_truncated, }


def _cut_to_tokens(text, max_tokens, mark=_TRUNCATION_MARK):
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    budget = max_tokens - estimate_tokens(mark)
    if budget <= 0:
        budget, mark = max_tokens, ""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + mark


def plan_generation(results, history, query, num_ctx, use_context=True):
    system = ""  # 不发 system prompt（恒空字符串）；use_context 只决定检索资料是否注入
    # 资料正文与逐条对齐共用同一份块列表：一次构造。截到哪几条，诊断（sources/context_docs_kept）
    # 就只报哪几条 —— 不再有第二条拼资料的路径。
    ctx_blocks = (context_blocks(results) if use_context else [])
    context = "\n\n".join(ctx_blocks) if ctx_blocks else ""
    q_tokens = estimate_tokens(query)
    # 输入窗口上限 = 模型上下文长度（token 度量）减 chat template 特殊 token 的余量。
    # 不做输出预留：生成超窗时由服务端上下文移位从最前面按 token 遗忘（先丢输入前文/历史）；
    # 输出不限长（num_predict ≤ 0 → 请求 -1）。
    input_cap = max(0, int(num_ctx) - GEN_INPUT_RESERVE)
    # 与启动期 NUM_CTX 交叉校验对称：外部调用方可以传进任意 num_ctx，这里同样拦。
    # 不拦的后果是"看似正常发出、实则空提问"（失效原型：静默）。
    if input_cap <= 0:
        raise ValueError(
            f"num_ctx={num_ctx} 扣除 RAG_INPUT_RESERVE={GEN_INPUT_RESERVE} 后输入上界 ≤ 0："
            f"装配不出有效提问（调大 num_ctx 或调小 RAG_INPUT_RESERVE）"
        )
    # 历史预算单点出处：输入上界扣掉 system、资料、本轮提问后剩下的额给历史。与
    # build_generation_messages 内的 body_cap 同式，预算先算好、装配时按同一算式兜底硬上界。
    # 配置只做第二道上限：更严才收紧，0 或比窗口余量宽时不改变行为。
    context_tokens = estimate_tokens(context) if context else 0
    budget = max(0, input_cap - estimate_tokens(system or "") - context_tokens - q_tokens)
    if GEN_HISTORY_MAX_TOKENS > 0:
        budget = min(budget, GEN_HISTORY_MAX_TOKENS)
    # 精确算式（不看 budget<0）：归零仍装不下才报警，避免"归零但实际没超"的假阳性。
    over_window = (estimate_tokens(system or "") + context_tokens + q_tokens > input_cap)
    gen = build_generation_messages( system, history, query, context=context, context_blocks=ctx_blocks, max_history_tokens=budget, input_cap=input_cap, )
    gen["query_tokens"], gen["over_window"] = q_tokens, bool(over_window or gen["cap_trimmed"])
    if over_window:
        log("⚠️  本轮提问+资料超出输入窗口（历史已全部让位仍不足）：按 system→资料→提问 截到窗口内，提问只截前缀")
    # prompt 的 token 估算在这里**只算一次**并存进 stats：run_turn 的台账直接取用
    # （prompt_tokens_est ↔ 服务端 prompt_eval_count，两者比值≈1 说明只有本轮增量被
    # prefill；≈≫1 说明整段重算——README §11.4 的重载税台账就靠这一对）。
    gen["prompt_tokens_est"] = message_tokens(gen["messages"])
    if gen["prompt_tokens_est"] > int(num_ctx) * 0.9:
        log("ℹ️  输入占用窗口超过 90%：生成空间靠服务端上下文移位腾出（从最前面按 token 遗忘，先丢历史/资料前文）")
    return { "system": gen["system_text"], "context": gen["context_text"], "messages": gen["messages"], "stats": gen }


STEP_BLANK_QUERY = "空查询"
STEP_SPARSE_TERMS = "稀疏查询文本 (服务端分词)"
STEP_RECALL = "单通道召回"
STEP_RRF = "RRF 融合"
STEP_CANDIDATES = "候选落地分块表"
STEP_DEDUP = "父块去重"
STEP_RERANK = "Rerank 重排"
STEP_RESULTS = "检索汇总"
STEP_RESULT_COUNT = "检索汇总条数"
STEP_CONFIDENCE = "置信度"
STEP_DEGRADE = "降级"
TRACE_NOTES = {
    "去重": "同一父块下的兄弟子块只保留融合名次最好的一个；上下文装配用的是 parent_text，不去重会让同一段父文本重复出现",
    "条数": "返回条数少于 top_k 只可能是『去重后不同父块本身不够』；不补同父子块，见 hybrid_search 末尾的说明",
}


def chunk_from_payload(payload, point_id):
    payload = payload or {}
    out = {k: payload.get(f, d) for f, k, d in _PAYLOAD_SPEC}
    # source 兼容旧索引的 book 字段（payload 里只有 book 时 _PAYLOAD_SPEC 取不到值）
    out["source"] = _source_of(payload)
    out["id"] = payload.get("chunk_id") or str(point_id)
    out["point_id"] = point_id
    return out


def _empty_search_result(steps):
    """空结果出口：只写置信度（不判拒答），按 return_steps 形态返回。"""
    if steps is not None:
        steps[STEP_CONFIDENCE] = results_confidence([])
    return ([], steps) if steps is not None else []


# ── 检索引擎 ──
class RAGEngine:
    def __init__(self, client=None, embed=None, rerank=None):
        self._embed = embed
        self._rerank = rerank
        self._rerank_tok = None          # http 重排下只留 tokenizer（steps 用），不加载权重
        self._rerank_tok_failed = False
        self._client = client
        self._collection_name = COLLECTION_NAME
        self._lexicon_checked = False
        self._chunk_fp_checked = False
        self._chunk_fp_logged = False  # 首次指纹比对留痕一次（部署后从日志即可判"产物是否同源"）
        self._empty_warned = {}  # 集合名 -> 已告警过：空回取告警按集合记，别名切新集合后不会被旧告警状态压掉
        # 加载（懒初始化）与推理分开两把锁：加载慢、推理短，共用一把会拖慢并发查询。
        # 查询侧前向/rerank 用推理锁串行（MPS 单模型不做真并行），加载持初始化锁。
        self._model_init_lock = threading.Lock()
        self._model_infer_lock = threading.Lock()
    @property
    def collection_name(self):
        return self._collection_name
    def _get_client(self):
        # 本类唯一没加锁的懒加载单例（检查与赋值之间是双建窗口）：两个线程各建一个
        # QdrantClient，各自带一份连接池与**一份本地推理器**（ModelEmbedder 是每客户端
        # 一份、内部无锁），稀疏 Document 推理会跨实例错配。与另两个单例同款双检。
        if self._client is None:
            with self._model_init_lock:
                if self._client is None:
                    from qdrant_client import QdrantClient
                    self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT, )
        return self._client
    def _resolve_collection(self):
        self._collection_name = resolve_collection_name(self._get_client())
        return self._collection_name
    def count(self):
        client = self._get_client()
        self._resolve_collection()
        if not client.collection_exists(self._collection_name):
            return 0
        return client.count(collection_name=self._collection_name).count
    def _await_weights_outside_lock(self, path):
        """把「等待权重就绪」（最长 RAG_MODEL_WAIT_SECS=900s）挪出 _model_init_lock。

        持初始化锁等后台下载，会让**所有**并发回合整轮阻塞在一把只该保护"是否已加载"
        的锁上（预热线程持锁等待时，首查询线程也一起等满 900s）。这里只做前置等待，
        目录最终不可用时仍由锁内的 _require_model_dir 报同一条具名错误（错误文案单点）。
        **模型加载本身必须留在锁内**：那是"同一模型两份驻留"的唯一防线，不能为并发让步。
        """
        if not WEIGHTS_BACKGROUND or MODEL_WAIT_SECS <= 0 or _model_dir_ready(path):
            return
        _wait_for_weights(path)

    def _get_embed(self):
        self._await_weights_outside_lock(EMBED_MODEL_PATH)
        with self._model_init_lock:
            if self._embed is None:
                self._embed = load_embedding_model()
            return self._embed
    def _get_rerank(self):
        self._await_weights_outside_lock(RERANK_MODEL_PATH)
        with self._model_init_lock:
            if self._rerank is None:
                self._rerank = load_reranker_model()
            return self._rerank
    def _get_rerank_tokenizer_only(self):
        """http 重排后端下只取 tokenizer（几十 MB）：steps 的 token 预算列要用它算，
        但没有理由为它加载 2 G 的权重。拿不到就返回 None，调用点整列留空——不编造数字。"""
        with self._model_init_lock:
            if self._rerank_tok is None and not self._rerank_tok_failed:
                self._rerank_tok = load_rerank_tokenizer_only()
                self._rerank_tok_failed = self._rerank_tok is None
            return self._rerank_tok
    def _http_rerank(self, query, documents, top_k):
        """Cohere 风格 /rerank（infinity 等）：客户端做排序与截断，服务端只出分数。

        不变量：**对全体候选统一打分**。故请求 top_n = 候选总数，并强制校验返回条数与
        index 齐全 —— 服务端少给一条、重复一条都直接报错，绝不在子集上排序。
        """
        if not documents:
            return []
        data = self._http_json( RERANK_URL + "/rerank", { "model": RERANK_MODEL_NAME, "query": query, "documents": documents, "top_n": len(documents), }, )
        rows = data.get("results")
        if not isinstance(rows, list) or len(rows) != len(documents):
            _got = len(rows) if isinstance(rows, list) else "非列表"
            raise RuntimeError(
                f"重排服务 {RERANK_URL}/rerank 返回 {_got} 条 ≠ 候选 {len(documents)} 条"
                "（重排必须对全体候选统一打分，服务端不得裁剪）"
            )
        scores: list = [None] * len(documents)
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(f"重排服务 {RERANK_URL}/rerank 返回了非对象条目: {row!r}")
            idx = row.get("index")
            score = row.get("relevance_score", row.get("score"))
            if not isinstance(idx, int) or not (0 <= idx < len(documents)) or score is None:
                raise RuntimeError(f"重排服务 {RERANK_URL}/rerank 条目缺 index/score: {row!r}")
            if scores[idx] is not None:
                raise RuntimeError(f"重排服务 {RERANK_URL}/rerank 重复返回候选下标 {idx}")
            scores[idx] = float(score)
        missing = [i for i, s in enumerate(scores) if s is None]
        if missing:
            raise RuntimeError(f"重排服务 {RERANK_URL}/rerank 缺少候选下标 {missing[:5]} 的分数（共 {len(missing)} 条）")
        ranked = sorted(((i, s) for i, s in enumerate(scores) if s is not None), key=lambda p: (-p[1], p[0]))
        return ranked[:top_k]
    def _http_json(self, url, payload):
        t0 = time.time()
        r = _model_http_session().post(url, json=payload, timeout=MODEL_HTTP_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"模型服务 {url} 返回 HTTP {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError(f"模型服务 {url} 响应不是合法 JSON: {r.text[:200]}") from None
        if not isinstance(data, dict) or data.get("error"):
            raise RuntimeError(f"模型服务 {url} 响应异常: {str(data)[:200]}")
        log(f"  模型服务 {url} {round(time.time() - t0, 3)}s")
        return data
    def _http_embed(self, texts, is_query=False):
        if is_query:
            texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
        data = self._http_json(EMBED_URL + "/embeddings", {"input": texts, "model": EMBED_MODEL_NAME})
        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            off = EMBED_URL + "/embeddings"
            raise RuntimeError(
                f"模型服务 {off} 返回 {len(rows) if isinstance(rows, list) else '非列表'} 条向量 ≠ 输入 {len(texts)} 条"
            )
        vectors = []
        for row in rows:
            v = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(v, list):
                raise RuntimeError(f"模型服务 {off} 返回向量字段缺失")
            vectors.append(v)
        return vectors
    def _inference_lock(self):
        return self._model_infer_lock
    def _check_chunk_freshness(self, index_fp):
        """分块新鲜度闸门（与词表闸门对称）：process 之后未重跑 index 时，检索会静默
        继续吃旧分块——stages 报过期但没人看，用户拿到的是无声的旧知识库。fail-closed：
        对不上或无从判断就显式抛错，而不是"看似正常"地服务过期分块。只查一次、且**通过
        （一致）才置位**：失败/抛错都不置位，后续每轮检索都会重试，直到重建对齐为止。"""
        if self._chunk_fp_checked:
            return
        if index_fp is None:
            # 旧索引没记指纹：不是"一致"，是**无从判断**。静默放行 = 过期索引假新鲜。
            raise RuntimeError(
                "索引未记录分块指纹（旧索引），无法与当前切分指纹比对："
                "可能在服务过期分块，请运行 `python chatbot.py index` 重建后再检索。"
            )
        # fail-closed：本地路径只涉及文件读取与哈希（不碰网络），失败就显式报错，
        # 不允许"暂缓核对"——旧写法吞掉异常继续检索，等于把"结果未知"当"一致"放行，
        # 坏指纹缓存过期索引能谎称新鲜整整一个进程生命周期。
        # 与 build_chunks / verify_chunks_freshness 用**同一个**函数：配置 + 源文本 sha。
        # 只算配置时，"改语料 + 只跑 process"会让索引里的旧指纹与"当前"仍然相等，
        # 闸门恒等放行 —— 用户静默吃到旧知识库（失效原型：静默 + 判据恒真）。
        paths = _document_paths()
        current = _chunking_fingerprint(_sources_fingerprint(paths))
        if not self._chunk_fp_logged:  # 每进程只留痕一次：部署后从运行日志判"产物是否同源"
            self._chunk_fp_logged = True
            log(f"🔧 分块指纹: 索引={index_fp} / 当前={current}（一致才放行检索）")
        if index_fp == current:
            self._chunk_fp_checked = True
            return
        raise RuntimeError(
            f"分块产物与索引不一致（索引 {index_fp} / 当前 {current}）：检索会吃到过期分块，"
            f"请运行 `python chatbot.py process && python chatbot.py index` 重建后再检索。"
        )
    def _check_lexicon_consistency(self, index_lexicon_id):
        if self._lexicon_checked:
            return
        if index_lexicon_id is None:
            # 与分块指纹同理：无记录 ≠ 一致，旧索引的稀疏词空间无从判断。
            raise RuntimeError(
                "索引未记录词表指纹（旧索引）：稀疏通道新旧无从判断，"
                "请运行 `python chatbot.py sparse` 重建后再检索。"
            )
        # fail-closed：与分块闸门同理由。词表指纹要读 Qdrant 里的词表点，失败≠暂缓——
        # 计算失败（词表损坏/点缺失）意味着稀疏通道状态未知，安静放行就是静默错配。
        # 唯一的分寸在网络抖动（Qdrant 瞬断/超时）：状态未知≠已知不一致，拦截本轮再
        # 等一轮比 503 报"词表不一致"更符合下游预期——与网络类暂缓口径保持一致。
        try:
            current = lexicon_fingerprint(self._get_client())
        except Exception as exc:
            _msg = str(exc).lower()
            _net = (
                isinstance(exc, (ConnectionError, TimeoutError))
                or "connect" in _msg
                or "timed out" in _msg
                or "timeout" in _msg
            )
            if _net:
                log(f"⚠️  词表指纹网络读取失败，暂缓一致性核对（下轮重试）：{type(exc).__name__}: {exc}")
                return
            raise
        if index_lexicon_id == current:
            self._lexicon_checked = True
            return
        raise RuntimeError(
            f"词表与索引不一致（索引 {index_lexicon_id} / 当前 {current}）："
            f"稀疏通道词空间已错配，召回会静默下降，"
            f"请运行 `python chatbot.py sparse` 重建后再检索。"
        )
    def _gate_check_collection_fingerprint(self, client, collection):
        """检索开跑前的指纹闸门：分块指纹取一条 payload 核对，词表指纹全量滚动核对。
        fail-closed——对不上或无从判断就抛错（见 _check_*）；未入库/读不到时不拦截
        （交由检索主路径报"尚未入库"），避免把 503 误报成"指纹不一致"。"""
        if self._chunk_fp_checked and self._lexicon_checked:
            return
        try:
            pts, _ = client.scroll( collection_name=collection, limit=1, with_payload=True, with_vectors=False )
        except Exception as exc:
            log(f"⚠️  指纹闸门读集合 {collection} 失败，暂缓核对: {type(exc).__name__}: {exc}")
            return
        if not pts:
            return
        payload = pts[0].payload or {}
        # 分块指纹抽首点成立：一个集合只由一次 index 写成（新集合 + 原子切别名），
        # 集合内恒一。词表指纹不成立 —— sparse 原地更新会留下混合态，
        # 首点恒属已更新前缀（见 _distinct_lexicon_ids），必须全量滚动。
        self._check_chunk_freshness(payload.get(CHUNK_FINGERPRINT_FIELD))
        try:
            lex_ids = _distinct_lexicon_ids(client, collection)
        except Exception as exc:
            log(f"⚠️  指纹闸门全量读取词表指纹失败，暂缓核对: {type(exc).__name__}: {exc}")
            return
        if not lex_ids:
            return
        if len(lex_ids) > 1:
            # 混合态 = 判定不一致（读取本身成功）：fail-closed，且不置位 —— 重试到
            # sparse 重建对齐为止，与 _check_* 的"通过才置位"同规则。
            raise RuntimeError(
                f"集合 {collection} 内词表指纹混合（{dict(lex_ids)}），sparse 中断/跳页残留，"
                f"新旧稀疏向量正在混同参与融合，请运行 `python chatbot.py sparse` 重建后再检索。"
            )
        self._check_lexicon_consistency(lex_ids[0][0])
    def _fetch_chunks(self, point_ids, collection=None):
        """按命中 id 逐次回取 payload（按 id 精确检索，不全量滚动 2 万条）。

        每次调用都向 Qdrant 取，进程内不留分块表：既不占常驻内存，也不存在"索引重建后
        旧缓存失步"这个状态点。按 id retrieve 结构性做不到部分加载——缺失的 id 由调用方
        显式计数告警，完整性由 Qdrant 本身保证。

        collection 必须由调用方传入（本轮召回用的那个集合）。这里再解析一次别名，可能撞上
        index 的原子切换而换成另一个集合：点 id 是位置整数（0..N-1），另一集合**同样存在**
        这些 id，于是 retrieve 不缺、不报错，却取回完全不相干的正文（静默错配，判据恒真）。
        """
        if not point_ids:
            return {}
        if collection is None:
            self._resolve_collection()
            collection = self._collection_name
        points = self._get_client().retrieve( collection_name=collection, ids=list(point_ids), with_payload=True, with_vectors=False, )
        if not points:
            # 命中 id 回取为空：集合为空或索引正在重建。不再用 id=0 探针判断——id 0 缺失时
            # 旧写法会把非空集合误判为空并静默丢掉全部命中。
            if not self._empty_warned.get(collection):
                log(f"⚠️  集合 {collection} 按命中 id 回取为空，请确认已运行: python chatbot.py index")
                self._empty_warned[collection] = True
            return {}
        self._empty_warned.pop(collection, None)
        # 分块/词表指纹核对由 hybrid_search 开跑前的 _gate_check_collection_fingerprint 承担
        # （fail-closed 且只查一次）；这里只做按 id 精确回取，不再重复取号。
        return {p.id: chunk_from_payload(p.payload, p.id) for p in points}
    def _source_filter(self, source):
        if not source:
            return None
        from qdrant_client import models
        # 过滤键必须与读取侧 _source_of 同一套兼容面（SOURCE_PAYLOAD_FIELDS 已是这三键的
        # 唯一来源，别再自拼一份）：旧索引 payload 里只有 book/source，只按 SOURCE_FIELD
        # 过滤 = 命中恒空且不报错（静默空结果）。should=任一键命中即可；未建索引的旧键
        # 走扫描，22k 点规模可接受。
        return models.Filter(should=[ models.FieldCondition(key=k, match=models.MatchValue(value=source)) for k in SOURCE_PAYLOAD_FIELDS ])
    def hybrid_search(self, query, top_k=TOP_K, return_steps=False, source=None, book=None):
        source = source if source is not None else book
        steps: dict | None = {} if return_steps else None
        if not _has_searchable_content(query):
            if steps is not None:
                steps[STEP_BLANK_QUERY] = f"查询 {query!r} 不含可检索内容，直接返回空结果"
            return _empty_search_result(steps)
        self._resolve_collection()
        client = self._get_client()
        collection = self._collection_name
        # 指纹闸门（fail-closed）：每轮检索开跑前核对分块/词表指纹，通过后只核对一次；
        # 失败不置位、每轮重试（见 _check_chunk_freshness/_check_lexicon_consistency）。
        self._gate_check_collection_fingerprint(client, collection)
        query_filter = self._source_filter(source)
        # 稀疏查询先编码：HTTP 嵌入是网络等待，稀疏 Qdrant 召回可在同一时间并发跑，
        # 墙钟从"嵌入 + 双路召回"降到 max(嵌入+稠密, 稀疏)。torch 后端保持原路径。
        sparse_query = sparse_encode(query)
        _sparse_terms = sparse_terms_of(query)
        if steps is not None:
            steps[STEP_SPARSE_TERMS] = _sparse_terms
        if bool(RRF_SPARSE_WEIGHT) and not _sparse_terms:
            if RRF_DENSE_WEIGHT:
                _degraded(steps, "稀疏通道：查询无可检索内容（纯标点/空白），本次退化为纯稠密检索")
            else:
                _degraded(steps, "稀疏通道：查询无可检索内容（纯标点/空白），且稠密通道权重为 0，本次无可用召回")
        _use_sparse = bool(RRF_SPARSE_WEIGHT) and bool(_sparse_terms)
        dense_query = None
        dense_hits = []
        sparse_hits = []
        note = ""
        if bool(RRF_DENSE_WEIGHT) and EMBED_BACKEND == "http" and _use_sparse:
            # 并发：稀疏 Qdrant 查询与稠密嵌入（HTTP）同时进行；随后再查稠密。
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as pool:
                sparse_future = pool.submit(
                    _qdrant_sparse_hits, client, collection, sparse_query, RECALL_LIMIT, query_filter,
                )
                dense_query = self._http_embed([query], is_query=True)[0]
                dense_hits = _qdrant_dense_hits(client, collection, dense_query, RECALL_LIMIT, query_filter)
                try:
                    sparse_hits = sparse_future.result()
                except Exception as exc:
                    note = _sparse_recall_note(exc, sparse_query)
        else:
            if RRF_DENSE_WEIGHT:
                if EMBED_BACKEND == "http":
                    dense_query = self._http_embed([query], is_query=True)[0]
                else:
                    tokenizer, model, device = self._get_embed()
                    with self._inference_lock():
                        if self._embed is None or self._embed[1] is not model:
                            tokenizer, model, device = self._get_embed()
                        q_emb = embed([query], tokenizer, model, device, is_query=True, autocast=_float16_ok(device))
                    dense_query = q_emb[0].tolist()
            dense_hits, sparse_hits, note = dual_channel_recall( client, collection, dense_query, sparse_query, RECALL_LIMIT, query_filter=query_filter, tolerate_sparse_error=True, use_dense=bool(RRF_DENSE_WEIGHT), use_sparse=_use_sparse, )
        if note:
            _degraded(steps, note)
        rankings = channel_rankings(dense_hits, sparse_hits)
        if steps is not None:
            steps[STEP_RECALL] = [ {"channel": ch, "rank": r, "point_id": pid} for ch, pids, _w in rankings for r, pid in enumerate(pids, 1) ]
        n_channels = max(1, len(rankings))
        # 候选池默认**恒等于** RAG_RERANK_TOP_K（本地与容器质量一致）；只有用户显式设了
        # RAG_RERANK_BUDGET_MS 才收缩 —— 收缩时每次都点名，杜绝"静默降精度"。
        _pool, _pool_note = rerank_pool_size()
        if _pool != RERANK_TOP_K:
            log(f"  重排候选池: {_pool}/{RERANK_TOP_K}（{_pool_note}）—— 会降低召回，来自显式配置")
        fuse_limit = _pool * n_channels if DEDUP_BY_PARENT else _pool
        fused = weighted_rrf(rankings, k=RRF_K, limit=fuse_limit)
        chunks = self._fetch_chunks([point_id for point_id, _score, _res in fused], collection) if fused else {}
        if steps is not None:
            steps[STEP_RRF] = [ { "resource": resources, "score": float(score), "id": (chunks.get(point_id) or {}).get("id", str(point_id)), "text": (chunks.get(point_id) or {}).get("child_text", ""), } for point_id, score, resources in fused ]
        candidates = []
        candidate_fused_rank = {}
        missing = 0
        for fused_rank, (point_id, _score, _res) in enumerate(fused, 1):
            if point_id in chunks:
                candidates.append(chunks[point_id])
                candidate_fused_rank[chunks[point_id]["id"]] = fused_rank
            else:
                missing += 1
        if missing:
            _degraded(steps, f"{missing} 条检索命中不在分块表中（索引可能正在重建），已跳过")
        if not candidates:
            return _empty_search_result(steps)
        if steps is not None:
            steps[STEP_CANDIDATES] = [[c["point_id"], c["id"]] for c in candidates]
        folded = []
        pre_dedup = len(candidates)
        if DEDUP_BY_PARENT:
            candidates, folded = dedup_candidates_by_parent(candidates)
        # 候选池上限的截断与父块折叠分开计数：旧写法把两者合成一个 folded，被池上限
        # 吃掉的候选既不在 kept 也不在 folded，诊断上无法区分"折叠"与"截断"。
        pool_trimmed = 0
        if len(candidates) > _pool:
            pool_trimmed = len(candidates) - _pool
            candidates = candidates[:_pool]
        if steps is not None:
            steps[STEP_DEDUP] = { "enabled": DEDUP_BY_PARENT, "去重前候选": pre_dedup, "kept": len(candidates), "folded": len(folded) if DEDUP_BY_PARENT else 0, "folded_ids": [c["id"] for c in folded], "被候选池上限截断": pool_trimmed, "说明": (TRACE_NOTES["去重"] if DEDUP_BY_PARENT else "RAG_DEDUP_BY_PARENT=0：未做父块去重，同父兄弟子块会原样进入重排"), }
        if not candidates:
            return _empty_search_result(steps)
        if top_k > _pool:
            log(f"⚠️  top_k={top_k} 超过候选池上限 {_pool}，最多返回 {_pool} 条（可调大 RAG_RERANK_TOP_K 或 RAG_RERANK_BUDGET_MS）")
        cand_texts = [rerank_text_for(c) for c in candidates]
        rr_k = min(top_k, len(candidates))
        if RERANK_BACKEND == "http":
            # 现成重排服务（与嵌入同一个 infinity 进程）：客户端排序、服务端只出分数。
            # tokenizer 只为 trace 的 token 预算列，拿不到就整列留空（见 _get_rerank_tokenizer_only）。
            rr_tok = self._get_rerank_tokenizer_only()
            reranked = self._http_rerank(query, cand_texts, top_k=rr_k)
        else:
            rr_tok, rr_model, rr_dev = self._get_rerank()
            with self._inference_lock():
                # 复核：懒加载只在 init 锁内替换引用，两端之间存在检查-使用窗口；取到推理锁
                # 后按当前引用核对，不一致就重取，避免拿旧引用跑前向 + 随后重载第二份。
                if self._rerank is None or self._rerank[1] is not rr_model:
                    rr_tok, rr_model, rr_dev = self._get_rerank()
                reranked = rerank_indices(query, cand_texts, rr_tok, rr_model, rr_dev, top_k=rr_k)
        if steps is not None:
            rr_query_tokens = len(rr_tok.encode(query)) if rr_tok is not None else None
            # 使用 tokenizer 实际的特殊 token 数，而非硬编码
            special_tokens = rr_tok.num_special_tokens_to_add(pair=True) if rr_tok is not None else None
            rr_doc_budget = None
            if rr_query_tokens is not None and special_tokens is not None:
                rr_doc_budget = max(0, RERANK_MAX_LENGTH - rr_query_tokens - special_tokens)
            rr_rows = []
            for idx, score in reranked:
                text = cand_texts[idx]
                full_tokens = len(rr_tok.encode(text)) if rr_tok is not None else None
                # text 必须是**送进 rerank 的那段**（cand_texts[idx]），不是 child_text：
                # RERANK_ON=parent 时两者不同，旧写法展示 child_text 却用 parent_text 算
                # tokens/被截断，对外 steps 自相矛盾。scored_field 说明这段来自哪个字段。
                row = { "id": candidates[idx]["id"], "fused_rank": candidate_fused_rank.get(candidates[idx]["id"]), "score": float(score), "tokens": full_tokens, "text": text, "scored_field": ("parent_text" if RERANK_ON == "parent" else "contextual_text") }
                row["送入上限"] = rr_doc_budget
                # http 重排且本地没有 tokenizer 时这两列无从计算：写 None（明确"未知"），
                # 不写 False —— 那会读成"没被截断"，是另一种静默。
                row["被截断"] = (full_tokens > rr_doc_budget) if (full_tokens is not None and rr_doc_budget is not None) else None
                rr_rows.append(row)
            steps[STEP_RERANK] = rr_rows
        output = []
        for idx, score in reranked:
            c = candidates[idx]
            src = _source_of(c)
            # chunk_index/total_chunks 带上：资料元数据标头（_context_meta_header）的
            # "分块 x/y" 与 /ask 的 results 都读它；不带就等于标头里永远缺这一项。
            item = { "id": c["id"], "child_text": c["child_text"], "parent_text": c["parent_text"], "source": src, "contextual_text": c["contextual_text"], "rerank_score": score, "parent_id": c.get("parent_id", ""), "chunk_index": c.get("chunk_index"), "total_chunks": c.get("total_chunks"), }
            if SOURCE_FIELD != "source":
                item[SOURCE_FIELD] = src
            output.append(item)
        if len(output) < top_k:
            log(f"⚠️  去重后只有 {len(output)} 个不同父块，少于 top_k={top_k}；按实返回（不补同父子块）")
        confidence = results_confidence(output)
        if steps is not None:
            steps[STEP_RESULTS] = output
            steps[STEP_RESULT_COUNT] = { "返回": len(output), "请求 top_k": top_k, "候选池": len(candidates), "说明": TRACE_NOTES["条数"], }
            steps[STEP_CONFIDENCE] = confidence
        if return_steps:
            return output, steps
        return output


# ── 单轮编排 ──
def drain_events(events, on_thinking=None, on_delta=None, on_error=None):
    """消费事件流，返回 (thinking, answer, error, done)；回调参数为 (本段, 通道累计全文)。
    词表外的 kind 直接抛错——兜底会把「词表分叉」伪装成正常结尾。"""
    acc: dict = {EVENT_THINKING: [], EVENT_DELTA: [], EVENT_ERROR: []}
    sink = {EVENT_THINKING: on_thinking, EVENT_DELTA: on_delta, EVENT_ERROR: on_error}
    done = {}
    for kind, payload in events:
        if kind == EVENT_DONE:
            done = payload
            continue
        bucket = acc.get(kind)
        if bucket is None:
            # 词表外的 kind（含 RETRIEVAL，首事件须已被 next() 取走）= 事件契约被绕过
            raise RuntimeError(f"drain_events 收到词表外事件: {kind!r}（首事件须由消费方 next() 消费）")
        bucket.append(payload)
        callback = sink[kind]
        if callback:
            callback(payload, "".join(bucket))
    return "".join(acc[EVENT_THINKING]), "".join(acc[EVENT_DELTA]), "".join(acc[EVENT_ERROR]), done


def first_retrieval(events):
    """消费事件流的首事件（恒为 RETRIEVAL）并返回其 payload。

    事件契约被任何一侧绕过（比如上游调整了顺序）都必须在消费点显式暴露：靠 next() 的
    元组展开虽然不会崩，但会把 retrieval 误当成 thinking/delta 的下游通道。断言在此，
    让非契约形态当场报错而不是静默错位。
    """
    kind, payload = next(events)
    if kind != EVENT_RETRIEVAL:
        raise RuntimeError(f"事件契约被绕过：首事件不是 {EVENT_RETRIEVAL!r} 而是 {kind!r}")
    return payload


def run_turn(engine, query, history, *, top_k=None, source=None, book=None, use_context=None, num_ctx=None, num_predict=None, chat=None):
    source = source if source is not None else book
    chat = chat or stream_chat
    top_k = TOP_K if top_k is None else top_k
    use_context = RAG_USE_CONTEXT if use_context is None else use_context
    num_ctx = NUM_CTX if num_ctx is None else num_ctx
    # num_predict ≤ 0 = 不限长：请求发 -1，窗口是唯一上限；输入上界与 num_predict 无关
    # （见 plan_generation：输入窗口 = num_ctx - 模板余量，不做输出预留）。
    _raw_predict = int(NUM_PREDICT if num_predict is None else num_predict)
    request_predict = _raw_predict if _raw_predict > 0 else -1
    started = time.time()
    # 本函数不再做内存编排（无 gate/卸载/release，2026-10-24 定稿）：
    # 检索模型首次加载后常驻，生成模型驻留由 Ollama 服务端策略决定；进程异常由 entrypoint
    # 的进程级巡检就地重启（README §11.4）。
    results = engine.hybrid_search(query, top_k=top_k, source=source)
    confidence = results_confidence(results)
    elapsed = time.time() - started
    plan = plan_generation(results, history, query, num_ctx, use_context=use_context)
    # 唯一事件契约：首事件恒为 RETRIEVAL，之后 thinking/delta/error/done；检索为空也照常生成。
    # 诊断只报**实际发出**的内容：gen（plan["stats"]）就是装配真正执行后统计出来的
    # （context_docs_kept/context_partial/query_truncated/…），不再另拼一套 steps。
    yield EVENT_RETRIEVAL, { "query": query, "results": results, "confidence": confidence, "gen": plan["stats"], "elapsed": elapsed }
    answer_chars = 0
    # 预算参数必须真的进请求（见 stream_chat 文档）：chat 可调用对象须接受这两个关键字。
    # **无重试、无长度/时间闸**：本轮生成原样转发，窗口（num_ctx）与上下文移位是唯一边界；
    # 空答 / done_reason=length / 报错照原样交给上层（空答不入历史、错误照常显式）。
    _gen_t0 = time.time()
    _done_info = None
    _error_text = None
    for kind, delta in chat(plan["messages"], num_ctx=num_ctx, num_predict=request_predict):
        if kind == EVENT_DELTA:
            answer_chars += len(delta)
        elif kind == EVENT_DONE:
            # done 的字段在台账里被消费前先快照（本函数要捕 done_reason/计数，转发的
            # dict 会继续携带原字段，消费方无感知差异）
            _done_info = {
                "done_reason": (delta or {}).get("done_reason"),
                "eval_count": (delta or {}).get("eval_count"),
                "prompt_eval_count": (delta or {}).get("prompt_eval_count"),
            }
            delta = { **delta, "answer_chars": answer_chars, "elapsed": elapsed, }
        elif kind == EVENT_ERROR:
            _error_text = str(delta)[:300]
        yield kind, delta
    # ── 单轮台账（可选诊断）：done = 一次完整"检索→生成"循环；
    # error = 失败轮；两者都不落 = 被 Stop 中断（上游 close，本轮已知许不完整，不记）──
    _status = "done" if _done_info is not None else ("error" if _error_text else "stopped")
    if _status != "stopped":
        # prompt 估算由 plan_generation 算好存在 stats 里（整轮只算一次），这里直接取用
        _prompt_tok_est = plan["stats"].get("prompt_tokens_est")
        if _prompt_tok_est is None:
            _prompt_tok_est = message_tokens(plan["messages"])
        record_turn_stats({
            "status": _status,
            "num_ctx": num_ctx,
            "request_predict": request_predict,
            "history_turns": len(history or []),
            "retrieval_elapsed": round(elapsed, 3),
            "gen_elapsed": round(time.time() - _gen_t0, 3),
            "prompt_tokens_est": _prompt_tok_est,
            "history_tokens": plan["stats"].get("used_tokens"),
            "context_tokens": plan["stats"].get("context_tokens"),
            "query_tokens": plan["stats"].get("query_tokens"),
            "over_window": bool(plan["stats"].get("over_window")),
            "answer_chars": answer_chars,
            "prompt_eval_count": (_done_info or {}).get("prompt_eval_count"),
            "eval_count": (_done_info or {}).get("eval_count"),
            "done_reason": (_done_info or {}).get("done_reason"),
            "error": _error_text,
        })
        if _done_info is not None:
            log(
                "🧾 单轮台账: "
                f"prompt_eval_count={_done_info.get('prompt_eval_count')} "
                f"prompt_tokens_est={_prompt_tok_est} "
                f"eval_count={_done_info.get('eval_count')} answer_chars={answer_chars} "
                f"done_reason={_done_info.get('done_reason')!r} "
                f"retrieval={elapsed:.1f}s"
            )
        else:
            log(f"🧾 单轮台账: {_error_text}")


def cmd_process(args=None):
    log("阶段一: 处理数据 (读取 + 分块)")
    return build_chunks()


# ── index 单例锁：同一 artifacts 目录只允许一个重建（index/**sparse** 共用）在跑 ──
# 为什么必须 fail-fast（而不是等待/跳过）：两个 index 会写**同一个**稠密缓存 tmp
# （_save_embed_cache 用 "<path>.tmp.npz" + os.replace）—— 一方还在写、另一方就把它 rename，
# 得到的半截缓存**看起来完全正常**（维度对、条数对），却少了一批向量；而写缓存失败只被 catch
# 成一条 warning。这正是本项目反复点名的"状态"失效原型：不崩，只留下一个像正常的坏文件。
# sparse 同样必须进这把锁：它按启动时的别名目标原地更新向量，index 中途切别名并删除旧集合，
# 会让 sparse 的页更新打到已删除的集合上，留下"已更新前缀 + 旧指纹后缀"的混合态。
# 语义：拿不到锁 = 明确报错退出（exit 1）。"另一个重建正在跑"与"本次跳过重建"是两件事，
# 后者会让调用方以为索引已经更新。
INDEX_LOCK_FILE = os.path.join(_ARTIFACTS_DIR, ".index.lock")


def _ps_etime_seconds(pid):
    """进程已运行秒数（ps -o etime=）。

    ps 的输出形如 [[dd-]hh:]mm:ss：无天数时两个数字段是 **mm:ss**，
    带 `dd-` 前缀时是 hh(:mm(:ss))——小时位出现与否由有没有天数决定。
    观测工具缺席/失败返回 None（保守回退）。
    """
    import subprocess
    try:
        out = subprocess.run( ["ps", "-o", "etime=", "-p", str(pid)], capture_output=True, text=True, timeout=5 ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    dd, _, rest = out.partition("-")
    has_days = bool(rest)
    if not has_days:
        dd, rest = "0", out
    try:
        days = int(dd or "0")
        parts = [int(p) for p in rest.split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        # 无天数 → mm:ss；有天数 → hh:mm（ps 的秒位必定出现）
        if has_days:
            hours, minutes, seconds = parts[0], parts[1], 0
        else:
            hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _pid_alive(pid, since=None):
    """pid 是否还活着。判不了（权限/ps 缺席）一律按"活着"处理 —— 保守地不抢别人的锁。

    PID 复用补充判据（since = 锁的写入时刻）：这个 pid 的进程启动时间晚于锁写入时刻，
    说明它是冒名顶替的新进程（旧持有者已死、pid 被复用），锁视作陈旧。不引入该判据时，
    容器内一次异常退出后若 pid 恰被新进程复用，index 会被永久判"有人在跑"。
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    if since is not None:
        elapsed = _ps_etime_seconds(pid)
        if elapsed is None:
            # 观测工具缺席（精简容器无 ps）≠ 进程死了：按"活着"保守处理不会被误抢。
            # 但要把判据缺席说出来——否则陈旧锁需人工清理时，用户不知道为何判不了。
            log(f"⚠️  pid={pid} 的启动时长读取失败（ps 缺席或不可执行）："
                "PID 复用判据不可用，按仍存活处理；确认无重建在跑后可手动删锁")
        else:
            started = datetime.datetime.now() - datetime.timedelta(seconds=elapsed)
            if started > since:
                log(f"⚠️  pid={pid} 的启动时刻（{started:%Y-%m-%d %H:%M:%S}）晚于锁写入时刻"
                    f"（{since:%Y-%m-%d %H:%M:%S}）—— 判定为 PID 复用，锁视作陈旧")
                return False
    return True


def acquire_index_lock():
    """取 index 单例锁（index 与 sparse 共用：两个重建并发会互相毁掉共享状态）；
    拿到返回锁路径，被别人持有则抛 RuntimeError（调用方报错退出）。

    陈旧锁（持有 pid 已不存在：容器被杀、断电、崩溃残留）自动接管并告警 —— 否则一次异常退出
    就会让 index 永久跑不起来（比并发更糟的失效：功能被一把死锁永久关闭）。
    """
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    for _attempt in range(2):
        try:
            fd = os.open(INDEX_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                with open(INDEX_LOCK_FILE, encoding="utf-8") as f:
                    holder = f.read().strip()
            except OSError:
                holder = ""
            parts = holder.split()
            pid = int(parts[0]) if parts and parts[0].isdigit() else None
            # 锁的写入时刻给 PID 复用判据：启动晚于锁写入的 pid 是冒名顶替的新进程
            lock_ts = None
            if len(parts) > 1:
                try:
                    lock_ts = datetime.datetime.fromisoformat(parts[1])
                except ValueError:
                    lock_ts = None
            if pid is not None and _pid_alive(pid, since=lock_ts):
                raise RuntimeError(
                    f"另一个 index/sparse 正在运行（pid={pid}，锁 {INDEX_LOCK_FILE}，起始 {parts[1] if len(parts) > 1 else '未知'}）——"
                    f"并发重建会写坏同一份稠密缓存、把 sparse 写到被删除的集合上，本进程直接退出（不等待、不跳过）。"
                    f"确认对方已结束后重跑；确实没有别的 index/sparse 在跑时删锁: rm -f {INDEX_LOCK_FILE}"
                ) from None
            log(f"⚠️  发现陈旧 index 锁（内容 {holder or '空'}）：持有进程已不存在，自动接管 {INDEX_LOCK_FILE}")
            try:
                os.remove(INDEX_LOCK_FILE)
            except OSError as exc:
                raise RuntimeError(f"陈旧锁 {INDEX_LOCK_FILE} 删除失败（{type(exc).__name__}: {exc}），请手动清理后重跑") from None
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()} {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        return INDEX_LOCK_FILE
    raise RuntimeError(f"取 index 锁失败（{INDEX_LOCK_FILE} 连续两次被抢占），请检查后重跑")


def release_index_lock():
    """释放锁。**只删自己持有的那份**（防误删接管者的锁）；失败只告警，不影响重建结果。"""
    try:
        with open(INDEX_LOCK_FILE, encoding="utf-8") as f:
            parts = f.read().strip().split()
        if parts and parts[0].isdigit() and int(parts[0]) != os.getpid():
            log(f"⚠️  index 锁 {INDEX_LOCK_FILE} 现在属于 pid={parts[0]}，不由本进程删除")
            return
        os.remove(INDEX_LOCK_FILE)
    except FileNotFoundError:
        pass
    except Exception as exc:
        log(f"⚠️  释放 index 锁失败（不影响本次重建结果）: {type(exc).__name__}: {exc}")


def cmd_index(args=None):
    log("阶段二: 向量化入库")
    try:
        acquire_index_lock()
    except RuntimeError as exc:
        log(f"❌ {exc}")
        return 1
    try:
        # finally 覆盖正常返回、异常与 Ctrl-C（KeyboardInterrupt 也走 finally）：
        # 绝不因为一次中断把锁留在盘上。
        build_index()
    finally:
        release_index_lock()
    return 0


def _port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _streamlit_pids():
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-f", r"streamlit run .*chatbot\.py"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(p) for p in out.split() if p.isdigit() and int(p) != os.getpid()]


def _document_paths():
    """corpus 文档清单：与 build_segments 同一选路（DATA_DIR × DOC_GLOB），目录缺失=空。"""
    import glob as _glob
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        p for p in _glob.glob(os.path.join(DATA_DIR, DOC_GLOB), recursive=True)
        if os.path.isfile(p)
    )


def _corpus_has_searchable_content():
    """corpus 是否至少有一份文档含可检索内容（去 BOM/空白后的纯标点不算，同 build_chunks）。

    与**查询谓词** _has_searchable_content(text)（拦纯标点查询，见 line 742 附近）名分明确：
    前者判语料、后者判单个查询字符串，导出的语义不同，必须两个名字。
    """
    for path in _document_paths():
        try:
            text = read_book_text(path)
        except Exception:
            continue
        # 去 BOM（\ufeff 不是 str.strip() 的字符集成员，必须显式剥）：与 docstring 同一口径
        if text.lstrip("\ufeff").strip():
            return True
    return False


def _qdrant_gate():
    """serve/api 共用的启动门禁：连不上或集合为空则报错退出，返回 (coll, count)。

    门禁 client 只服务这次启动检查，不向下传（真正的 client 由 RAGEngine 自建）——
    以前把它 return 出去却被调用方丢弃，服务进程里白挂一个没人关的连接池。
    """
    from qdrant_client import QdrantClient
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
    try:
        coll = resolve_collection_name(client)
        if _state_ref().alias_degraded:
            log(f"⚠️  集合别名不可用，直接读 {COLLECTION_NAME}：当前检索不读索引目标集合，切别名（index 重建）后不会自动跟随")
        try:
            count = 0 if not client.collection_exists(coll) else client.count(collection_name=coll).count
        except Exception as exc:
            log(f"错误: 连不上向量数据库 {QDRANT_HOST}:{QDRANT_PORT}")
            log(f"  原因: {type(exc).__name__}: {exc}")
            log("  请先确认服务已启动: docker compose up -d")
            sys.exit(1)
        if count == 0:
            # 0 条 ≠ 一律"没构建"：corpus 是空目录（或全部只含封面页）时这是"已清空"
            # 的**合法终态**（见 build_chunks / build_index 的清空语义），让用户原样启动
            # 空知识库，只提示检索将零命中；语料非空才是"没建库"，指路重跑。
            if _corpus_has_searchable_content():
                log("错误: 向量数据库为空（corpus 非空但从未构建）")
                log("请先运行:")
                log("  python chatbot.py process")
                log("  python chatbot.py index")
                sys.exit(1)
            log("⚠️  语料已清空，知识库为空: 欢迎使用，但检索将一律零命中（问答退化为纯对话）")
            return coll, count
        return coll, count
    finally:
        try:
            client.close()  # 不留无主连接池；老版本 qdrant-client 没有 close 就等 GC
        except Exception:
            pass


def cmd_serve(args=None):
    coll, count = _qdrant_gate()
    log(f"向量数据库: {coll} {count} 条记录")
    log(f"Qdrant服务: {QDRANT_HOST}:{QDRANT_PORT}")
    if _port_in_use(8501):
        pids = _streamlit_pids()
        if not pids:
            log("错误: 端口 8501 已被其它进程占用，但没找到本项目启动的 Streamlit。")
            log("  请先确认占用者: lsof -nP -iTCP:8501 -sTCP:LISTEN")
            sys.exit(1)
        log("Streamlit 服务已在运行: http://localhost:8501")
        log("改过代码务必重启：进程里跑的是启动时导入的旧模块。")
        log(r"  重启: pkill -f 'streamlit run .*chatbot\.py' && python chatbot.py serve")
        log(f"  当前 PID: {', '.join(map(str, pids))}")
        sys.exit(0)
    log("启动 Streamlit 服务...")
    log("服务地址: http://localhost:8501  （按 Ctrl+C 停止）")
    log("")
    # 必须与 .streamlit/config.toml 的 address = "127.0.0.1" 一致。CLI 优先级高于该文件，
    # 传 "::" 会把只监听回环的配置静默改写成监听全部网卡（含公网 v6），与配置文件意图相反。
    os.execv(sys.executable, [ sys.executable, "-m", "streamlit", "run", __file__, "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true", ])


# ── 交互入口：Streamlit 主界面 ──
def _render_stored_message(msg):
    """历史消息渲染：思维链与正文同栏输出、不做任何区分（只有 RAG_THINK 开启时才会入历史）。"""
    import streamlit as st
    with st.chat_message(msg["role"]):
        text = msg.get("thinking", "") + msg.get("content", "")
        if text.strip():
            st.markdown(text)
        elif msg.get("error"):
            st.error(msg["error"])


def _gen_failure_hint(gen_error):
    """按错误文本给下一步动作：本地与创空间共用同一套判据，只说该错误本身指向的事实。

    顺序即优先级：具体判据在前、泛化判据在后。旧顺序把 "connection" 排在 "http 500"
    之前，含 connection 字样的 500 会被误报成"连不上生成服务"；同时整表漏掉了 404 ——
    而 entrypoint 故意把模型拉取放后台，"模型不在本地"是冷启动期最可能的一类错误。
    """
    low = (gen_error or "").lower()
    if "http 404" in low or ("not found" in low and "model" in low):
        return "生成模型不在 Ollama 里：看运行日志的 [entrypoint] 拉取行（后台拉取期间问答都会失败，拉完自动恢复）"
    if "signal: killed" in low or "out of memory" in low:
        return "生成进程被系统杀掉（内存超限）：看运行日志的 [entrypoint] [mem] 观测行，必要时调小 RAG_NUM_CTX"
    if "http 500" in low:
        return "生成服务返回 500：模型未就绪或进程已被杀，看运行日志的 [entrypoint] 与 /tmp/ollama.log"
    if "econnrefused" in low or "connection" in low or "timed out" in low:
        return "连不上生成服务：确认 ollama serve 已启动、OLLAMA_BASE_URL 可达"
    return None


def _stream_reply(events):
    """流式渲染本轮答案：生成失败与 num_predict 截断必须可见（静默失效违背项目不变量）。
    返回 (thinking, answer, gen_error, done)；done 为空即生成未正常结束（与入历史判据一致）。
    思维链与正文共用同一个输出栏、按到达顺序原样拼接——无折叠区、无分开的占位、无任何区分。"""
    import streamlit as st
    ph = st.empty()
    shown = {"thinking": "", "answer": ""}

    def render():
        ph.markdown(shown["thinking"] + shown["answer"])

    def on_thinking(_chunk, total):
        shown["thinking"] = total
        render()

    def on_delta(_chunk, total):
        shown["answer"] = total
        render()

    def on_error(chunk, _total):
        # 流中途的错误必须当场留痕：UI 侧要等整流耗尽才装配收尾元素，rag.log 第一时间可见
        log(f"⚠️  生成流中途错误: {chunk}")

    thinking, answer, gen_error, done = drain_events(
        events, on_thinking=on_thinking, on_delta=on_delta, on_error=on_error,
    )
    # 流末对齐一次全文，防尾块丢失。必须在错误装配**之前**：render() 会整块替换 ph 的内容，
    # 放在 ph.error("⚠️") 之后会把刚装进去的错误/占位符清掉 —— 用户拿到的是残缺正文、
    # 错误看不见（失效原型：静默）。错误改走独立元素（st.error），与 ph 里的残缺正文并存。
    render()
    if gen_error:
        st.error(gen_error)
        hint = _gen_failure_hint(gen_error)
        if hint:
            st.caption(hint)
    elif not answer:
        # 空答案与错误必须一样显眼：think 开启时预算被思考吃光正是这样收场的
        # （done 齐全、正文 0 字），只画一个 ⚠️ 会让用户以为是网络问题。
        ph.markdown("⚠️")
        st.warning(
            "生成正常结束但没有产出任何正文"
            + (f"（done_reason={done.get('done_reason')}, eval_count={done.get('eval_count')}）"
               if done else "")
            + ("：思维链吃掉了输出预算，关掉 RAG_THINK 或调大 RAG_NUM_PREDICT。"
               if THINK else "：调大 RAG_NUM_PREDICT 或换个问法重试。")
        )
    if done and done.get("done_reason") == "length":
        # 调大 num_predict 在 think 开启时会连带撞内存墙（预算同时归思考与正文用），
        # 那种情形下先关思维链才是对的下一步。
        st.warning(
            "⚠️ 生成被 num_predict 截断（done_reason=length）：答案没写完。"
            + ("先关 RAG_THINK（思考与正文共用预算），确实需要思维链再调大 RAG_NUM_PREDICT。"
               if THINK else "调大 RAG_NUM_PREDICT。")
        )
    return thinking, answer, gen_error, done


def answer_is_good(gen_error, done, answer):
    """生成是否算"本轮成功"：无错误、有 done 事件、正文非空（trim 后）。
    只供 Streamlit 决定是否入历史（run_turn 的唯一 UI 消费方）；空答/done_reason=length
    原样交上层（done 事件如实携带 answer_chars），不存在"两侧各写一套判据"的问题。
    将来若新增 API 消费点，应让消费方各自声明起闸与否，不要把闸移进 run_turn 层。"""
    return not gen_error and bool(done) and bool(answer and answer.strip())


def _run_assistant_turn(engine, prompt, history):
    """跑一轮检索与生成并渲染；只有生成正常结束且答案非空才把助手消息写进会话历史。"""
    import streamlit as st

    def rollback_orphan_user():
        # 本轮 user 消息由 run_streamlit 先写进会话。失败必须撤掉，否则会话里留下一条
        # 没有配对的孤儿提问，下一轮会被当成历史重发给模型（症状：同题被答两遍）。
        # 只在确认"队尾正是本轮这条 user"时弹，防止误删更早的内容。
        msgs = st.session_state.messages
        if msgs and msgs[-1].get("role") == "user" and msgs[-1].get("content") == prompt:
            msgs.pop()

    def close_events():
        # 中断时**主动**收掉上游流：关掉 events 会一路
        # 触发 stream_chat 的 finally: r.close()，让 Ollama 端看到连接关闭。不主动 close
        # 就只能等异常处理完、脚本栈解开后靠引用计数 GC 触发 GeneratorExit —— 那是不确定的，
        # 断开的生成可能继续烧完 num_predict 才停，而这台 Ollama 是单槽的，期间后续请求全在排队。
        # 已实测（2026-10-11）：close() 循 run_turn 的 for 转发链把 GeneratorExit 打进最内层
        # stream_chat，其 finally 必然执行；生成器已闭/未起时 close() 是无害的 no-op。
        if events is not None:
            try:
                events.close()
            except Exception:
                pass

    events = None
    try:
        events = run_turn(engine, prompt, history, top_k=TOP_K)
        # 首事件恒为 retrieval，必须消费掉（drain_events 的事件契约），且消费时显式断言
        # 非契约形态（first_retrieval）：最简前端暂不展示本轮资料清单，编号对照能力给 API 侧。
        first_retrieval(events)
        thinking, answer, gen_error, done = _stream_reply(events)
    except Exception as exc:
        # 只打「类型: 消息」无法定位（stat(None) 这类消息看不出是哪一行）：完整调用栈必须落盘，
        # 页面上同时给出最后一帧（file:line），部署环境读不到 rag.log 时也能直接定位。
        log(f"⚠️  本轮执行异常: {type(exc).__name__}: {exc}")
        log(traceback.format_exc().rstrip())
        frames = traceback.extract_tb(exc.__traceback__)
        own = [f for f in frames if os.path.basename(f.filename) == os.path.basename(__file__)]
        hit = own[-1] if own else (frames[-1] if frames else None)
        where = f"{hit.filename}:{hit.lineno} {hit.name}" if hit else "未知"
        st.error(
            f"本轮执行异常: {type(exc).__name__}: {exc}\n"
            f"出错位置: {where}\n"
            f"完整调用栈: {os.getenv('RAG_LOG_FILE', DEFAULT_LOG_FILE)}"
        )
        close_events()
        rollback_orphan_user()
        return
    except BaseException:
        # Stop 按钮/主动重跑抛的是 ScriptControlException（BaseException 子类），上面的
        # except Exception 接不住 —— 不回滚，孤儿 user 就永久留进会话历史（下一轮被当
        # 历史重发，同题被答两遍）。回滚后原样上抛：这个异常是 Streamlit 完成停止/重跑
        # 的通道，吞掉会把"停"变成"继续跑完"。
        # 先 close_events 再回滚：停止要求**确定**中止生成（关连接），不能只停在 UI 层。
        close_events()
        rollback_orphan_user()
        raise
    # 生成失败或空答案：不入历史（且 rollback 本轮 user）。gen_error 也进判据（不只依赖
    # "error 必不产出 done"这个上游契约）——一旦生产侧违约，残缺答案也不会被当成完整回复留下。
    if not answer_is_good(gen_error, done, answer):
        if gen_error:
            log(f"⚠️  生成失败，本轮回复未写入历史: {gen_error[:300]}")
        elif not done:
            log("⚠️  生成未正常结束（无 done 事件），本轮回复未写入历史")
        else:
            log("⚠️  生成为空答案，本轮回复未写入历史")
        rollback_orphan_user()
        return
    entry = {"role": "assistant", "content": answer}
    if THINK and thinking:
        entry["thinking"] = thinking
    st.session_state.messages.append(entry)


def _st_engine():
    import streamlit as st
    return st.cache_resource(show_spinner=False)(get_engine)()


# 登录门总开关：1=启用（默认），0=关闭 —— 关闭后即使 secrets.toml 有 [auth]
# （创空间 = 平台注入 OAUTH_* → entrypoint 生成）也不渲染登录门，直接进应用。
# 为什么需要这个开关而不是"删掉 [auth]"：平台的 OAuth 开关在**创空间设置页**（不在仓库里），
# 仓库侧改不到；要让仓库自己说了算，就只能在这里留一个开关。
LOGIN_ENABLED = env_bool("RAG_LOGIN_ENABLED", True)


def _render_login_gate(st):
    """渲染登录门。

    只能留在 iframe 里登录：创空间网关对**顶层导航**一律 302 回
    /studios/<owner>/<name>?mode=full（实测），所以"新窗口打开 /auth/login"会被弹回
    创空间页面。iframe 内 OAuth 能否成立取决于浏览器是否接受跨站 Cookie：
    Streamlit 1.6x 的 _streamlit_session / _streamlit_user_tokens 都是
    SameSite=None; Secure; Partitioned，Chrome 可正常存取；Safari / 关闭第三方 Cookie
    的浏览器会丢弃，回调就报 "Missing provider for OAuth callback"。
    这里把 Cookie 名打进日志（只名字、无值），部署后据此判断是浏览器丢了 Cookie
    还是流程本身失败。
    """
    try:
        _cks = ",".join(sorted(st.context.cookies.keys())) if hasattr(st.context, "cookies") else "-"
    except Exception:
        _cks = "?"
    log(f"[auth] 登录门：cookies=[{_cks}]")
    if st.button("", icon="spinner", key="login", use_container_width=True):
        st.login()


def run_streamlit():
    import streamlit as st
    global _STATE
    st.set_page_config(layout="wide")  # 展示最大化：宽屏铺满视口
    # ── 登录门：RAG_LOGIN_ENABLED=1 且 secrets.toml 有 [auth] 才启用 ──
    # 两个条件都满足才启用：开关（仓库侧，.env 可改）与 [auth]（运行期是否存在，
    # 取决于平台是否注入 OAUTH_*）。本地无配置自动跳过。
    _auth = LOGIN_ENABLED and hasattr(st.user, "is_logged_in")   # 有 [auth] 才为 True，无需裸表达式（避免被 magic 渲染）
    if _auth and not st.user.is_logged_in:
        _, c, _ = st.columns([2, 1, 2])
        with c:
            _render_login_gate(st)
        st.stop()
    if _auth:
        st.sidebar.caption(getattr(st.user, "name", "") or getattr(st.user, "sub", ""))
        st.sidebar.button("", icon=":material/logout:", help="退出登录",
                          on_click=st.logout, use_container_width=True)
    # 先把运行态单例钉到模块变量：脚本线程会在 cache_resource 之外再访问 load_lexicon 等
    # 懒加载，若预热/首查询线程抢先建了一份、脚本线程又建一份，就是"同一进程两套词表"。先
    # 赋值（值来自 cache_resource 指向的同一模块单例），其后所有懒加载都走这一个。
    _STATE = _streamlit_runtime_state()
    engine = _st_engine()
    # 会话内 TTL 缓存 count：rerun（点数不会变）直接用值，不白等网络；index 中途切别名
    # 是生效过程，TTL 过期后的新 rerun 自然取新数。"按会话只取一次"的旧缓存能骗一整个
    # 长会话（index 已生效仍报 stale 点数）——收集后晚点引入的 TTL 缓存修复了"用户
    # 改了点数但 24h 都没刷新"的错觉。TTL<=0 视为禁用缓存（每次 rerun 都 re-count）。
    _now = time.time()
    _cached = st.session_state.get("collection_count")
    _cached_ts = st.session_state.get("collection_count_ts", 0)
    if _cached is None or COLLECTION_COUNT_TTL_S <= 0 or _now - _cached_ts >= COLLECTION_COUNT_TTL_S:
        try:
            st.session_state["collection_count"] = engine.count()
            st.session_state["collection_count_ts"] = _now
        except Exception as exc:
            st.session_state.pop("collection_count", None)
            st.session_state.pop("collection_count_ts", None)
            st.error(f"无法访问向量库: {type(exc).__name__}: {exc}")
            st.stop()
    if st.session_state["collection_count"] == 0:
        # 同 _qdrant_gate：0 条不一定是"没建库"——corpus 空目录是清空语义的合法终态，
        # 让会话原样跑空知识库（检索零命中走纯对话兜底），语料非空才报错指路。
        # 语料全目录扫描跟随 collection_count 的 TTL 失效（B3 同款：TTL 内不重扫，防
        # 每次 rerun 都 O(corpus)——用户放/删文档最多一个 TTL 后被识别）。
        _corp = st.session_state.get("corpus_has_content")
        _corp_ts = st.session_state.get("corpus_check_ts", 0)
        if _corp is None or COLLECTION_COUNT_TTL_S <= 0 or _now - _corp_ts >= COLLECTION_COUNT_TTL_S:
            _corp = _corpus_has_searchable_content()
            st.session_state["corpus_has_content"] = _corp
            st.session_state["corpus_check_ts"] = _now
        if _corp:
            st.error("向量库为空，先跑：python chatbot.py process && python chatbot.py index")
            st.stop()
        st.info(
            "语料已清空，当前是空知识库：检索将一律零命中、问答退化为纯对话。"
            "放入文档后重跑 `python chatbot.py process && python chatbot.py index` 即可恢复检索。"
        )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in list(st.session_state.messages):
        _render_stored_message(msg)
    # 思维链有无只看 RAG_THINK 配置：开启时思考与正文同行输出（前后端均不区分）。
    if prompt := st.chat_input("", submit_mode="stop"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            _run_assistant_turn(engine, prompt, st.session_state.messages[:-1])


# ── 检索引擎单例与预热（Streamlit 与任何进程内调用共用）──
_ENGINE = None
_ENGINE_LOCK = threading.RLock()


def _warm_engine(engine):
    """后台预热：词表、嵌入/重排模型、生成模型常驻 Ollama；失败仅告警（首查询冷加载兜底）。"""
    def _warm():
        try:
            # 词表仍要预热（别名归一是查询路径上的同步调用）；jieba 已随分词组件化整体移除
            load_lexicon()
            # 嵌入/重排模型按后端预热：torch=加载进本进程；http=探一发小请求，让独立模型
            # 服务容器把冷加载提前到启动期（进程内不再持嵌入模型，rerank 仍在本进程）。
            if EMBED_BACKEND == "http":
                engine._http_embed([""], is_query=False)
            else:
                engine._get_embed()
            # 重排走 http 时模型不在本进程，预热只需探活（那些由 _http_rerank 的首个真请求负责）
            if RERANK_BACKEND == "torch":
                engine._get_rerank()
            # 必须看状态码：对未拉取的模型 Ollama 返回 404 而 requests 不抛异常，
            # 旧写法丢弃返回值、直接打印"预热完成"。而 entrypoint 正是把生成模型拉取
            # 放在后台的，冷启动窗口内这次预热必然落在 404 上 —— 唯一的"已预热"信号
            # 成了假信号，首个真故障要等用户提问时的 500 才暴露。
            # 还必须看正文：think 开启时 num_predict 预算会被思考吃光，content 恒为空，
            # 只看状态码会把"能产出 0 个字的生成链路"报成预热成功（判据恒真）。
            # 故预热一律 think=false、给 8 token —— 它验的是链路能出字，不是验思维链。
            # 超时取 OLLAMA_TIMEOUT（不是硬编码 180）：慢冷启动（模型后台拉取/加载）下
            # 探针等得起，才不会把"还在加载"误报成"预热未完成"。
            _probe_options = {"num_predict": 8, "temperature": TEMPERATURE}
            _ng_probe = _gen_num_gpu_effective()
            if _ng_probe is not None:
                # 让"这台机器起不起得来 GPU"在**预热期**就暴露：auto 模式下预热失败会翻转
                # 回落状态，首个真实问答就不必再白等一次失败。
                _probe_options["num_gpu"] = _ng_probe
            _probe_payload = { "model": MODEL, "messages": [{"role": "user", "content": ""}], "stream": False, "think": False, "options": _probe_options, }
            r = _ollama_session().post( API_URL, json=_probe_payload, timeout=TIMEOUT, )
            if r.status_code != 200:
                _gen_gpu_fallback(f"预热 HTTP {r.status_code} {r.text[:160]!r}")
                raise RuntimeError(f"生成模型预热失败 HTTP {r.status_code}: {r.text[:200]}")
            data = r.json() if r.text else {}
            _reply = ((data or {}).get("message") or {}).get("content") if isinstance(data, dict) else None
            if not (_reply or "").strip():
                raise RuntimeError(
                    f"生成模型预热返回空正文（done_reason={(data or {}).get('done_reason')!r}, "
                    f"eval_count={(data or {}).get('eval_count')!r}）—— 链路通但一个字都没产出"
                )
            log(f"模型预热完成（嵌入/重排/生成，探针 {_reply.strip()[:20]!r}）")  # type: ignore[union-attr]
            try:
                with open("/tmp/warm_ready", "w", encoding="utf-8") as _wf:
                    _wf.write("1")
            except OSError:
                pass
        except Exception as exc:
            log(f"⚠️  预热未完成（不影响服务，首次查询冷加载）: {type(exc).__name__}: {exc}")
    threading.Thread(target=_warm, daemon=True).start()


def get_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = RAGEngine()
            if PRELOAD_MODELS:
                _warm_engine(_ENGINE)
        return _ENGINE




# ── 工具：分块 A/B 比较 ──
DEFAULT_B = CHUNKS_JSON
# 比对列**由 _PAYLOAD_SPEC 派生**（不再手写）：来源字段那一项在产物里的键名由
# RAG_SOURCE_FIELD 决定，比对时统一按规范名 "source" 走 _source_of 读取。
# 手写清单是"多份字段契约表示"里最容易分叉的一份：少一列时比较器遍历不到它，
# 两边都缺就一路打"✅ 完全一致"，字段静默消失无人知（check_schema 现在双向校验兜底）。
FIELDS = [("source" if k == "source" else k) for _f, k, _d in _PAYLOAD_SPEC]
# 记录里来源字段的键名由 RAG_SOURCE_FIELD 决定（可能不是 "source"），比较器必须一并放行。
# SOURCE_PAYLOAD_FIELDS 是这三键的**唯一定义处**（_source_of / _source_filter 共用）。
_LEGACY_CHUNK_FIELDS = {"book"}
_ALLOWED_CHUNK_FIELDS = set(FIELDS) | _LEGACY_CHUNK_FIELDS | {SOURCE_FIELD}
_SOURCE_ALIASES = set(SOURCE_PAYLOAD_FIELDS)


def load(path):
    if not os.path.exists(path):
        sys.exit(f"❌ 找不到文件: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"❌ {path} 不是 JSON 列表（chunks.json 应为记录数组）")
    return data


def show(text, limit=90):
    text = text.replace("\n", "\\n")
    return text[:limit] + ("…" if len(text) > limit else "")


def check_schema(a, b, path_a, path_b):
    """记录字段契约的**双向**校验：多字段（未登记）与少字段都必须报错。

    旧实现只拦"未登记字段"（多），不拦"缺字段"（少）：产物少一列时比较器按 FIELDS
    逐列取值，两列都缺就都取到 None、一路打「✅ 完全一致」—— 字段静默消失被报成通过
    （失效原型：判据只覆盖一半）。这里对两个文件各自的键集各查一次。
    """
    seen = set()
    for rows in (a, b):
        for r in rows:
            if isinstance(r, dict):
                seen |= set(r)
    unknown = sorted(seen - _ALLOWED_CHUNK_FIELDS)
    if unknown:
        sys.exit(f"❌ 分块产物出现未登记字段 {unknown}（{path_a} / {path_b}）：比较器只遍历 FIELDS，未登记字段会被静默忽略。请先登记进 FIELDS。")
    for label, rows in ((path_a, a), (path_b, b)):
        keys: set = set()
        for r in rows:
            if isinstance(r, dict):
                keys |= set(r)
        if not keys:
            continue   # 空文件没有记录可查（总量行已给结论），不当成"缺全部字段"
        missing = sorted(_RECORD_CORE_FIELDS - keys)
        if missing or not (keys & set(SOURCE_PAYLOAD_FIELDS)):
            sys.exit(
                f"❌ {label} 的分块记录缺字段 {missing}"
                f"（来源键需是 {sorted(SOURCE_PAYLOAD_FIELDS)} 之一）——"
                f"比较器按 FIELDS 逐列比对，缺列会被静默当成 None 与空值比较，打出假的「完全一致」。"
            )


def compare_chunks_diff(a, b, allow, path_a, path_b):
    check_schema(a, b, path_a, path_b)
    log("分块结果比较")
    log(f"  A: {path_a}")
    log(f"  B: {path_b}")
    log(f"  允许不同的字段: {sorted(allow) if allow else '（无，要求逐字节相同）'}")
    ok = True
    if len(a) == len(b):
        log(f"[1] 总量: A={len(a)} 条  B={len(b)} 条  ✅")
    else:
        log(f"[1] 总量: A={len(a)} 条  B={len(b)} 条  ❌ 相差 {len(b) - len(a):+d} 条")
        ok = False
    ca, cb = Counter(_source_of(c) or "?" for c in a), Counter(_source_of(c) or "?" for c in b)
    for src in sorted(set(ca) | set(cb)):
        na, nb = ca.get(src, 0), cb.get(src, 0)
        flag = "✅" if na == nb else f"❌ 相差 {nb - na:+d}"
        log(f"      {src}: A={na} B={nb} {flag}")
        if na != nb:
            ok = False
    n = min(len(a), len(b))
    diff: dict = {f: [] for f in FIELDS}
    for i in range(n):
        ra, rb = a[i], b[i]
        for f in FIELDS:
            va = _source_of(ra) if f == "source" else ra.get(f)
            vb = _source_of(rb) if f == "source" else rb.get(f)
            if va != vb:
                diff[f].append((i, va, vb))
    log("\n[2] 字段差异（按列表位置逐条比对）")
    for f in FIELDS:
        if n == 0:
            # 空侧比对：0 差异是"没东西可比"，不是"比过了都一致" —— 报 ✅ 是把无从
            # 判断伪装成通过（总量行已给正确结论，这里给哑签即可）。
            log(f"      {f:16s} —（无可比对记录）")
            continue
        cnt = len(diff[f])
        if cnt == 0:
            log(f"      {f:16s} ✅ 完全一致")
            continue
        allowed = f in allow
        mark = "🟡 在允许范围内" if allowed else "❌ 不允许的改动"
        if not allowed:
            ok = False
        log(f"      {f:16s} {mark}  差异 {cnt} 条 ({cnt / n * 100:.2f}%)")
        for i, va, vb in diff[f][:3]:
            log(f"          位置 {i}: A={show(str(va))}")
            log(f"           {' ' * len(str(i))}      B={show(str(vb))}")
    if diff["child_text"]:
        pos = [x[0] for x in diff["child_text"]]
        log(f"\n[3] ⚠️  正文出现差异，首个位置 = {pos[0]}，最后 = {pos[-1]}（共 {len(pos)} 条）")
        log("     若差异连续延伸到列表末尾，说明是分块边界漂移而非局部文本差异，不应接受该改动。")
    log("\n" + "=" * 78)
    if ok:
        log("✅ 比较通过：差异全部落在允许范围内")
    else:
        log("❌ 比较不通过：出现了不允许的差异")
    return 0 if ok else 1


def cmd_compare_chunks(args=None):
    if args is None:
        args = _build_parser().parse_args(["compare-chunks"])
    unknown = [f for f in args.allow if f not in _ALLOWED_CHUNK_FIELDS]
    if unknown:
        sys.exit(f"❌ --allow 出现未知字段: {unknown}（可用: {sorted(_ALLOWED_CHUNK_FIELDS)}）")
    # 自定义来源键名归一到比对列 "source"
    allow = {("source" if f in _SOURCE_ALIASES else f) for f in args.allow}
    a, b = load(args.a), load(args.b)
    sys.exit(compare_chunks_diff(a, b, allow, args.a, args.b))


# ── 工具：稀疏向量重建 ──
def cmd_sparse(args=None):
    """sparse 权重重建入口：**与 index 共用单实例锁**（INDEX_LOCK_FILE）。

    为什么必须互斥：sparse 按启动时解析的集合名**原地**更新向量，index 中途会原子切
    别名并删除旧目标集合 —— 并发时 sparse 的页更新会打到被删/被换的集合上，留下
    "已更新前缀 + 旧指纹后缀"的混合态（词表指纹闸门虽能事后兜住，但重建已白做）。
    拿不到锁 = 明确报错退出（exit 1），与 index 语义一致。
    """
    try:
        acquire_index_lock()
    except RuntimeError as exc:
        log(f"错误: {exc}")
        return 1
    try:
        return _sparse_rebuild()
    finally:
        release_index_lock()


def _sparse_rebuild():
    from qdrant_client import QdrantClient, models
    # 批量重建会话较长：客户端给普通查询 2 倍超时，逐页 scroll 沿用 QDRANT_TIMEOUT。
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT * 2)
    COLL = resolve_collection_name(client)
    info = client.get_collection(COLL)
    total = info.points_count
    log(f"集合 {COLL}: {total} 条（Qdrant 内置 BM25，分词在服务端，options={bm25_options()}）")
    if not total:
        log(f"❌ 集合 {COLL} 为空，无事可做；请先运行: python chatbot.py process && python chatbot.py index")
        return 1
    lexicon_id = lexicon_fingerprint(client)
    # 写入前哨兵自检（fail-fast）：空文本短语必须查不到**任何**点（词表批量产物是惰性的，
    # online=False），一旦命中说明库里存在能把空短语编成稀疏"全词表"向量的非法条目 ——
    # 全部文档都在给它让权重，之后一路问零命中。这轮重建不落盘，先修词表。
    try:
        probe = client.query_points(
            COLL,
            query=models.Document(text=_EMPTY_SPARSE_TEXT, model=BM25_MODEL_NAME, options=bm25_options()),
            using=SPARSE_VECTOR_NAME,
            limit=1,
            timeout=QDRANT_TIMEOUT,
        )
    except Exception as exc:
        # 服务端把空 token 拒收（400/422/empty 类报错）同样符合契约：断言的是"不会算出稀疏向量"
        _msg = str(exc)
        _low = _msg.lower()
        if "empty" in _low or "400" in _msg or "422" in _msg:
            log("  哨兵自检通过: 空短语被服务端拒收/0 命中（符合契约，不落库）")
        else:
            log(f"  ⚠️  哨兵自检无法执行（{type(exc).__name__}: {exc}），校验跳过——不拦截重建")
    else:
        if probe.points:
            log(
                f"❌ 哨兵校验失败: 空短语 {_EMPTY_SPARSE_TEXT!r} 命中 {len(probe.points)} 点 —— "
                "存在把空短语编成'全词表'向量的词表项，重建中止；先修 lexicon/stopwords 再重跑"
            )
            return 1
        log("  哨兵自检通过: 空短语 0 命中")
    t0 = time.time()
    done = 0
    offset = None
    scroll_failures = 0
    # 页更新与 scroll **互不依赖**：页内 id 互不相交，set_payload 写的是同一个 lexicon_id，
    # 故主线程串行推进 scroll 游标（游标链本身必须串行），页更新交给线程池并发发 ——
    # 墙钟从"每页 (scroll + update) 之和"压到"max(scroll, update) + 余量"。
    # 每线程一个 client：带 Document 的请求走客户端私有的 ModelEmbedder，共用会互相 drain。
    workers = INDEX_WRITE_WORKERS or qdrant_write_workers(-(-total // WRITE_BATCH))
    log(f"  稀疏页并发: {workers} 路（每页 {WRITE_BATCH} 点；{hw_report()}）")
    _sparse_worker_state = threading.local()

    def _sparse_worker_client():
        c = getattr(_sparse_worker_state, "client", None)
        if c is None:
            c = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT * 2)
            _sparse_worker_state.client = c
        return c

    def _update_page(pts):
        try:
            ids = [p.id for p in pts]
            # 与 build_index 同一份文本（_index_text），否则重算的稀疏向量与稠密侧错配
            points = [ models.PointVectors( id=p.id, vector={SPARSE_VECTOR_NAME: sparse_encode(_index_text(p.payload or {}))}, ) for p in pts ]
            wc = _sparse_worker_client()
            wc.update_vectors(COLL, points=points)
            wc.set_payload(collection_name=COLL, payload={LEXICON_ID_FIELD: lexicon_id}, points=ids)
            return len(pts), None
        except Exception as exc:
            # 单页失败跳页继续，末尾 done != total 照样返回 1（继续但不静默）。
            return 0, f"{type(exc).__name__}: {exc}"

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    abort = False
    eof = False
    in_flight: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while (not (eof or abort)) or in_flight:
            # 窗口化：在飞页 ≤ 2×并发度（既不饿死 worker，也不把 21255 条 payload 全摊在内存里）
            while (not eof) and (not abort) and len(in_flight) < workers * 2:
                try:
                    pts, offset = client.scroll( COLL, limit=WRITE_BATCH, offset=offset, with_payload=True, with_vectors=False, timeout=QDRANT_TIMEOUT )
                except Exception as exc:
                    # scroll 失败推进不了页 → 同一页重试，只重试一次；连败 2 次即熔断。
                    scroll_failures += 1
                    if scroll_failures >= 2:
                        log(f"❌ scroll 连续失败 {scroll_failures} 次（{type(exc).__name__}: {exc}），无法推进，中止")
                        abort = True
                    else:
                        log(f"  ⚠️  scroll 失败（{scroll_failures}/2），重试同一页: {type(exc).__name__}: {exc}")
                    continue
                scroll_failures = 0
                if not pts:
                    eof = True
                    break
                in_flight[pool.submit(_update_page, pts)] = len(pts)
                if offset is None:
                    # 游标走完 = EOF。**必须在这里判**：qdrant 的 scroll 在 offset=None 时是
                    # "从头再来"，只看"空页"会绕回集合开头无限重扫（实测 done 涨到 130 万）。
                    eof = True
            if not in_flight:
                break
            finished, _unfinished = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in finished:
                size = in_flight.pop(fut)
                n_ok, err = fut.result()
                if err:
                    log(f"  ⚠️  批量更新失败，跳过本页 {size} 条: {err}")
                else:
                    done += n_ok
                    log(f"  {done}/{total}  ({time.time() - t0:.1f}s)")
    if abort:
        return 1
    log(f"词表指纹已回写: {lexicon_id}（{done} 条）")
    if done != total:
        log(f"❌ 只更新了 {done}/{total} 条，请重跑")
        return 1
    sample, _ = client.scroll(COLL, limit=5, with_vectors=True, timeout=30)
    log("  抽样校验（5 条点；**非全量契约**——全量契约由词表指纹闸门承担，抽样只看链路）:")
    for p in sample:
        sp = p.vector.get(SPARSE_VECTOR_NAME)  # type: ignore[union-attr]
        log(f"  校验 point {p.id}: 稠密 {len(p.vector.get(DENSE_VECTOR_NAME, []))} 维, 稀疏 {len(sp.indices) if sp else 0} terms")  # type: ignore[arg-type, union-attr]
    log(f"完成: {done} 条，耗时 {time.time() - t0:.1f}s")
    return 0


# ── 工具：词表校验 ──
# 数据目录与词表路径全部走全局配置，不写死文档名或文件名。
ALIAS_PATH = ALIASES_FILE
STOP_PATH = STOPWORDS_FILE
FORBIDDEN_NEGATION = set("不无未莫非没别勿弗毋否亡沒無別罔靡")
FORBIDDEN_DEGREE = set("很太最更极極甚颇頗稍略愈越挺蛮太殊煞")
NOISE_CHARS = set( "的了着过们这那一二三四五六七八九十百千万见说道问答叫唤令教使与在和" "是有将把被大小老好众位个名字等来去出入上下前后里外中而又也都便就却" "只遂因故然若且乃其之于以为所曰我你他她它相再从向往到至及并同连皆亦" "复可会要想知听看望走行坐立笑哭怒喜心手头身口眼声气人家儿子们兮乎者" "也矣焉哉尔汝卿彼此谁每各另别样般些点儿" )


def die(msg: str) -> None:
    """致命错误：打日志后终止本次命令行调用。-O 下 raise 不失效（sys.exit 则是真退出）。"""
    log("错误：" + msg)
    raise SystemExit(1)


def load_documents():
    """读语料文档做词表统计。与分块侧共用 read_document_text：字面量加引号归一这些
    预处理保持一致——两处各写一套读取逻辑，同一文档在两个入口读出两种文本，词表与
    入库正文就描述的不是同一份语料。坏字节转成的 \ufffd 这里剥掉（统计词面无意义）。"""
    docs = []
    name_of = _document_names(_document_paths())
    for p in name_of:
        text = read_document_text(p).replace("\ufffd", "")
        docs.append((name_of[p], text))
    return docs


def raw_byte_report():
    lines = []
    _name_of = _document_names(_document_paths())
    for p in _name_of:
        name = _name_of[p]
        with open(p, "rb") as fh:
            data = fh.read()
        try:
            data.decode("utf-8")
            lines.append((name, len(data), "OK"))
        except UnicodeDecodeError as exc:
            lines.append( (name, len(data), "非法字节 @ %d-%d" % (exc.start, exc.end - 1)) )
    return lines


def count_in_documents(docs, terms):
    out = {}
    for t in terms:
        out[t] = {name: text.count(t) for name, text in docs}
    return out


def longest_first_scan(docs, surfaces):
    by_first = defaultdict(list)
    for s in surfaces:
        by_first[s[0]].append(s)
    for ch in by_first:
        by_first[ch].sort(key=len, reverse=True)
    matched: Counter = Counter()
    crossing: defaultdict = defaultdict(Counter)
    for _name, text in docs:
        i, n = 0, len(text)
        while i < n:
            cands = by_first.get(text[i])
            hit = None
            if cands:
                for s in cands:
                    if text.startswith(s, i):
                        hit = s
                        break
            if hit is None:
                i += 1
                continue
            end = i + len(hit)
            for j in range(i + 1, min(end, n)):
                for a in by_first.get(text[j], ()):
                    if j + len(a) > end and text.startswith(a, j):
                        crossing[a][hit] += 1
                        break
            matched[hit] += 1
            i = end
    return matched, crossing


def competitor_windows(docs, surface, surface_set, limit=4, min_count=2):
    found: Counter = Counter()
    L = len(surface)
    for _name, text in docs:
        start = 0
        while True:
            idx = text.find(surface, start)
            if idx < 0:
                break
            start = idx + 1
            for ext in (1, 2):
                s = idx - ext
                if s >= 0:
                    w = text[s:idx + L]
                    extra = w[:ext]
                    if len(w) == ext + L and all( "\u4e00" <= c <= "\u9fff" for c in w ) and all(c not in NOISE_CHARS for c in extra):
                        if w not in surface_set:
                            found[w] += 1
            r = idx + L
            if r < len(text):
                w = text[idx:r + 1]
                if len(w) == L + 1 and "\u4e00" <= text[r] <= "\u9fff" and text[r] not in NOISE_CHARS and w not in surface_set:
                    found[w] += 1
    return [(w, n) for w, n in found.most_common() if n >= min_count][:limit]


def cmd_verify_lexicon(args=None):
    # 参数经 _build_parser 上收（原为手写 sys.argv 解析：-q/--quiet + 两个可选位置参数）
    if args is None:
        args = _build_parser().parse_args(["lexicon"])
    quiet = bool(args.quiet)
    alias_path = args.alias_path or ALIAS_PATH
    stop_path = args.stop_path or STOP_PATH
    def out(*a):
        if not quiet:
            log(*a)
    problems, warnings = [], []
    log("RAG 词表校验（%s + %s）" % (alias_path, stop_path))
    docs = load_documents()
    total_chars = sum(len(t) for _, t in docs)
    out("数据：%d 个文档，合计 %d 字符" % (len(docs), total_chars))
    for name, nbytes, status in raw_byte_report():
        out("    %-8s %9d 字节   UTF-8: %s" % (name, nbytes, status))
    out("    读取方式：utf-8 + errors='replace'（坏字节替换为 U+FFFD）")
    out("")
    out("-" * 78)
    out("[1] 别名表: %s" % alias_path)
    out("-" * 78)
    groups, ap = parse_aliases(alias_path)
    problems += ap
    if not groups:
        # 问题清单原本只在末尾统一打印：die 在这里先退出，根因（哪行没解析出来）就被吞，
        # 只剩"为空"这句结果，照着输出都不知道该改哪一行。
        for p in problems:
            log(f"  ❌ {p}")
        die("别名表为空或无法解析（根因见上）")
    surfaces = []
    for g in groups:
        surfaces.append(g["canonical"])
        surfaces.extend(g["aliases"])
    surface_set = set(surfaces)
    counts = count_in_documents(docs, surfaces)
    matched, crossing = longest_first_scan(docs, surfaces)
    out("别名组数：%d；词面总数：%d" % (len(groups), len(surfaces)))
    out("最长优先扫描：命中 %d 次" % sum(matched.values()))
    missing = [s for s in surfaces if sum(counts[s].values()) == 0]
    for s in missing:
        problems.append( "词面 %r 在 %s/ 中出现 0 次 —— 违反「规范名与别名都必须真实出现」" % (s, DATA_DIR) )
    out("")
    out("    规范名                次数   别名（次数）")
    out("    " + "-" * 72)
    for g in groups:
        c = sum(counts[g["canonical"]].values())
        al = "  ".join( "%s(%d)" % (a, sum(counts[a].values())) for a in g["aliases"] )
        out("    %-10s %8d   %s" % (g["canonical"], c, al))
    out("")
    out("-" * 78)
    out("[2] 子串冲突 / 遮蔽检测（对称替换 + 最长优先）")
    out("-" * 78)
    out("    独立=按最长优先扫描吃掉的次数；包含遮蔽=被更长条目吃掉的次数（同目标=安全）")
    out("    左重叠保护=更长的词面先命中，把短别名挡掉（保护性）")
    out("")
    owner_of = {}
    for g in groups:
        for s in [g["canonical"]] + g["aliases"]:
            owner_of[s] = g["canonical"]
    shadow_rows = []
    for g in groups:
        for a in g["aliases"]:
            tot = sum(counts[a].values())
            ind = matched[a]
            shadow = tot - ind
            if shadow > 0:
                contains = sorted( s for s in surface_set if s != a and a in s and len(s) > len(a) )
                overlaps = [s for s, _c in crossing[a].most_common()]
                same = all(owner_of[s] == g["canonical"] for s in contains)
                shadow_rows.append( (a, tot, shadow, ind, contains, overlaps, same, g["canonical"]) )
    if shadow_rows:
        for a, tot, shadow, ind, contains, overlaps, same, canon in shadow_rows:
            flag = "安全(同归一目标)" if same else "注意(归入别人!)"
            cross_n = sum(crossing[a].values())
            contain_n = max(shadow - cross_n, 0)
            csrc = "%d:%s" % (contain_n, "/".join(contains)) if contains else "0:-"
            osrc = ("%d:%s" % (cross_n, "/".join(overlaps))) if overlaps else "0:-"
            out("    %-8s 总%5d 独立%5d 遮蔽%5d | 包含 %-22s | 左重叠保护 %-16s %s" % (a, tot, ind, shadow, csrc, osrc, flag))
            if not same:
                warnings.append( "%s 的 %d 次出现被更长的条目 %s 先匹配走（分属 %s），不会再落到 %s，请确认更长条目归属" % (a, contain_n, "/".join(contains), "/".join(sorted({owner_of[s] for s in contains})), canon) )
    else:
        out("    （无：本表不含「别名是另一词面子串」的情况）")
    out("")
    out("    竞争窗口（不在词典、但包含某别名的更长汉字串，>=2 次才列出；启发式人工复核）")
    out("          判定原则：更长串与短别名同指=安全，异指则需剔除或加保护。")
    any_comp = False
    for g in groups:
        for a in g["aliases"]:
            wins = competitor_windows(docs, a, surface_set)
            if wins:
                any_comp = True
                out("      %-8s -> %s" % (a, "  ".join("%s(%d)" % w for w in wins)))
    if not any_comp:
        out("      （无）")
    out("")
    out("-" * 78)
    out("[3] 归一自检：最长优先 + 规范名恒等映射（调用方加载逻辑的参考实现）")
    out("-" * 78)
    flat = {}
    for g in groups:
        for s in [g["canonical"]] + g["aliases"]:
            flat[s] = g["canonical"]
    first_idx = defaultdict(list)
    for s in flat:
        first_idx[s[0]].append(s)
    for ch in first_idx:
        first_idx[ch].sort(key=len, reverse=True)
    canon_set = {g["canonical"] for g in groups}
    def normalize(text, identity_canonical=True):
        out_chars, i, n = [], 0, len(text)
        while i < n:
            hit = None
            for s in first_idx.get(text[i], ()):
                if not identity_canonical and s in canon_set:
                    continue
                if text.startswith(s, i):
                    hit = s
                    break
            if hit:
                out_chars.append(flat[hit])
                i += len(hit)
            else:
                out_chars.append(text[i])
                i += 1
        return "".join(out_chars)
    sample = []
    for _name, text in docs:
        step = max(1, len(text) // 120000)
        sample.append(text[::step])
    idem_bad, checked = [], 0
    for s in sample:
        once = normalize(s)
        twice = normalize(once)
        checked += len(s)
        if once != twice:
            bad = next( (i for i in range(min(len(once), len(twice))) if once[i] != twice[i]), min(len(once), len(twice)), )
            idem_bad.append((once[max(0, bad - 12):bad + 12], twice[max(0, bad - 12):bad + 12]))
    out("抽样 %d 字符做归一，规范名是否参与匹配：是" % checked)
    if idem_bad:
        for a, b in idem_bad[:3]:
            out("    非幂等样例：%r -> %r" % (a, b))
        problems.append( "归一不是幂等的（规范名未按恒等映射参与匹配，短别名会二次替换成叠加串）" )
    else:
        out("幂等性：通过 —— norm(norm(x)) == norm(x)，不会出现二次替换叠加")
    # 反例样例从当前词表动态构造：找「别名是规范名真子串」的一组
    demo = next(
        (g["canonical"] for g in groups
         if any(a and a in g["canonical"] and len(a) < len(g["canonical"]) for a in g["aliases"])),
        groups[0]["canonical"],
    )
    out("    反例演示（若加载方忘记把规范名映射到自身）：")
    out("      正确：%s" % normalize(demo))
    out("      错误：%s" % normalize(demo, identity_canonical=False))
    if normalize(demo) == normalize(demo, identity_canonical=False):
        warnings.append("参考实现的正/反例输出相同，说明该样例没覆盖到子串陷阱")
    # 对拍只在"校验的正是线上加载的那张表"时才有意义：normalize_aliases 读的是模块级
    # ALIASES_FILE，传入自定义路径时线上实现根本没换表，硬比只会报假"分叉"。
    same_table = os.path.abspath(alias_path) == os.path.abspath(ALIASES_FILE)
    if not same_table or not ALIASES_ENABLED:
        out("与线上实现对拍：跳过（校验 %s；线上加载 %s，别名开关 %s）" % (
            alias_path, ALIASES_FILE, "开" if ALIASES_ENABLED else "关"))
    else:
        mismatch = []
        for s in sample[:2]:
            got, want = normalize(s), normalize_aliases(s)
            if got != want:
                i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), min(len(got), len(want)))
                mismatch.append((got[max(0, i - 12):i + 12], want[max(0, i - 12):i + 12]))
        if mismatch:
            for a, b in mismatch[:2]:
                out("    与线上实现不一致：%r (本文件参照实现) vs %r (normalize_aliases)" % (a, b))
            problems.append( "参照实现与 normalize_aliases 输出不一致 —— 两份实现已分叉，自检结论不可信")
        else:
            out("与线上实现对拍：通过 —— normalize_aliases 输出逐字相同")
    out("")
    out("-" * 78)
    out("[4] 停用词表: %s" % stop_path)
    out("-" * 78)
    swords, sp = parse_stopwords(stop_path)
    problems += sp
    words = [w for w, _ in swords]
    out("停用词数量：%d" % len(words))
    danger = []
    for w in words:
        bad_neg = sorted(set(w) & FORBIDDEN_NEGATION)
        bad_deg = sorted(set(w) & FORBIDDEN_DEGREE)
        if bad_neg or bad_deg:
            danger.append((w, bad_neg, bad_deg))
    if danger:
        for w, bn, bd in danger:
            problems.append( "停用词 %r 含禁止字：否定%s 程度%s —— 过滤后会破坏语义（「不读书」与「读书」会变成同一查询）" % (w, bn or "-", bd or "-") )
    else:
        out("危险性检查：通过 —— 无否定词（不/无/未/莫/非/没/别…）、无程度词（很/太/最/更/极/甚…）")
    name_conflict = sorted(set(words) & surface_set)
    for w in name_conflict:
        problems.append("停用词 %r 同时是别名词典里的词面（会把实体名过滤掉）" % w)
    if not name_conflict:
        out("交叉检查：停用词与别名词典无交集")
    zero_sw = [w for w in words if sum(count_in_documents(docs, [w])[w].values()) == 0]
    for w in zero_sw:
        warnings.append("停用词 %r 在语料中出现 0 次（留着无害，可删）" % w)
    out("")
    out("    停用词              次数    停用词              次数")
    out("    " + "-" * 60)
    sw_counts = count_in_documents(docs, words)
    pairs = [(w, sum(sw_counts[w].values())) for w in words]
    half = (len(pairs) + 1) // 2
    for i in range(half):
        left = "%-8s %8d" % pairs[i]
        right = "%-8s %8d" % pairs[i + half] if i + half < len(pairs) else ""
        out("    %s      %s" % (left, right))
    out("")
    out("=" * 78)
    if problems:
        log("发现问题 %d 处：" % len(problems))
        for p in problems:
            log("  [错误] " + p)
        return 1
    if warnings:
        log("提示 %d 处（不阻断）：" % len(warnings))
        for w in warnings:
            log("  [提示] " + w)
    log("全部通过：别名组 %d 组 / 词面 %d 个 / 停用词 %d 个，均已数据接地，无禁止类停用词。" % (len(groups), len(surfaces), len(words)))
    return 0


# ── 工具：阶段状态与最小重跑指引 ──
def cmd_stages(args=None):
    """逐阶段比对指纹，给出最小重跑命令。
    A 分块（语料/切分变 → process） B 稠密（换模型/新内容 → index 增量）
    C 稀疏（词表/BM25/编码变 → sparse） D 入库（集合缺失/点数不符 → index）"""
    stale = []
    cd_unknown = False

    log("[A] 分块产物")
    # 产物只读一次，A/B/C/D 共用同一份数据：各段各自再 open 一遍，文件截断/损坏时就出现
    # "同一份文件半截读得出、半截读不出"的混乱结论。读失败 = 产物坏了，B/D 都判不了。
    chunks = None
    read_failed = False
    if os.path.exists(CHUNKS_JSON):
        try:
            with open(CHUNKS_JSON, encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as exc:
            log(f"  ⚠️  读取 {CHUNKS_JSON} 失败（产物可能损坏，重跑 process 修复）: {type(exc).__name__}: {exc}")
            read_failed = True
            cd_unknown = True
    n_chunks = len(chunks) if chunks is not None else None
    if read_failed:
        # chunks 读坏时 meta 仍可能是好的：只比 meta 会打出"A ✅ 新鲜"，与上方⚠️
        # 自相矛盾（坏产物被报成新鲜），D 段也会因拿不到点数而无从比对。坏产物 = A 过期。
        stale.append("A")
        log("  ❌ chunks.json 不可读（见上方）-> 需要: python chatbot.py process")
    else:
        try:
            meta = verify_chunks_freshness()
            fingerprint = meta.get("fingerprint")
            old = meta.get("sources") if "sources" in meta else meta.get("books") or {}
            log(f"  ✅ 新鲜（{len(old)} 个文档，指纹 {fingerprint}）")
        except Exception as exc:
            stale.append("A")
            log(f"  ❌ {exc}")

    log("[B] 稠密向量缓存")
    if "A" in stale:
        log("  ⏭  A 过期，先重跑 process 再谈 B")
    elif not EMBED_CACHE_ENABLED:
        log("  ⏭  RAG_EMBED_CACHE=0：无缓存，index 每次全量重算")
    elif chunks is None:
        log("  ⏭  缺/坏 chunks.json，先跑 process（见上方提示）")
    else:
        try:
            keys = [_embed_cache_key(_index_text(c)) for c in chunks]
            # 与 index 共用 _inspect_embed_caches / _full_embed_cache：这里若再按设备预测
            # 一次精度，就会与 index 的实际选择分叉（同一轮日志给出两个结论），而且预测
            # 要 import torch —— 只为读一个文件名，白花 200MB。
            cands = _inspect_embed_caches(keys)
            hit = _full_embed_cache(cands)
            if hit is not None:
                precision, path, cache, _missing = hit
                log(f"  ✅ 缓存 {os.path.basename(path)}（{precision}）：{len(cache)} 条全部命中；当前产物 0/{len(chunks)} 条需重算")
            else:
                # 没有完整覆盖的那份 = index 仍要重算：不标 B 会让 A/C/D 全新鲜时打出
                # "无需重跑"，与这一行自相矛盾（同一轮日志两个结论）。
                stale.append("B")
                detail = "，".join(
                    f"{p} 缺 {len(m)}/{len(chunks)} 条" for p, _f, _c, m in cands
                ) or "磁盘上没有任何缓存文件"
                log(f"  ❌ 没有一份缓存完整覆盖当前产物（{detail}）-> 需要: python chatbot.py index")
        except Exception as exc:
            # 检查失败 = B 没验证过：不置位会让 A/C/D 全新鲜时打出"无需重跑"、exit 0，
            # 而 B 正可能过期（判据恒真式假新鲜）。CD 类"未验证"与"已过期"同权计数。
            cd_unknown = True
            log(f"  ⚠️  缓存检查失败（B 未验证，不算新鲜）: {type(exc).__name__}: {exc}")

    log("[C] 稀疏向量（词表/BM25）与 [D] 入库（Qdrant）")
    # 产物读取与 Qdrant 连接分开兜底：chunks.json 截断/损坏是"产物坏了"，
    # 合并在一个 except 里一律报"连不上 Qdrant"会把根因指错。
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
        coll = resolve_collection_name(client)
        if _state_ref().alias_degraded:
            log(f"  ⚠️  集合别名不可用，直接读 {COLLECTION_NAME}：重跑 index 切别名后不会自动跟随")
        if not client.collection_exists(coll):
            stale.append("C")
            stale.append("D")
            log("  [C] ❌ 集合不存在，稀疏向量随之不存在 -> 需要: python chatbot.py index")
            log("  [D] ❌ 集合不存在 -> 需要: python chatbot.py index")
        else:
            qd_count = client.count(collection_name=coll, exact=True).count
            lex_now = lexicon_fingerprint(client)
            if qd_count == 0:
                # 空集合没有点可比：报"旧索引/不一致"会把根因指错（真正的问题是没入库，
                # D 段已给出 index；词表指纹也随下次 index 一并写入）。
                log("  [C] ⏭ 集合为空，词表指纹无从比对（随 D/入库一并写入）")
            else:
                # 全量滚动而非抽首点：sparse 逐页原地更新，中断/跳页留下
                # "已更新前缀 + 旧指纹后缀"，scroll 首点恒属前缀 —— 抽样必过、
                # 旧稀疏向量静默参与融合（闸门侧同款缺陷，共用 _distinct_lexicon_ids）。
                lex_ids = _distinct_lexicon_ids(client, coll)
                if len(lex_ids) > 1:
                    stale.append("C")
                    log(f"  [C] ❌ 集合内词表指纹混合（{dict(lex_ids)}，sparse 中断/跳页残留）-> 需要: sparse")
                elif lex_ids[0][0] is None:
                    stale.append("C")
                    log("  [C] ❌ 索引未记录词表指纹（旧索引）-> 需要: sparse")
                elif lex_ids[0][0] != lex_now:
                    stale.append("C")
                    log(f"  [C] ❌ 词表/稀疏编码指纹不一致（索引 {lex_ids[0][0]} / 当前 {lex_now}）-> 需要: sparse")
                else:
                    log("  [C] ✅ 新鲜（稠密向量不受影响）")
            if n_chunks is None:
                # 产物缺失/不可读（A 段已判）：点数无从比对，绝不能落进 else 打"D ✅"。
                cd_unknown = True
                log(f"  [D] ⏭ 产物缺失/不可读，点数无法比对（集合 {qd_count} 条，先跑 process，见 A 段）")
            elif qd_count != n_chunks:
                stale.append("D")
                log(f"  [D] ❌ 点数不一致（集合 {qd_count} / 产物 {n_chunks}）-> 需要: index")
            else:
                log(f"  [D] ✅ 集合 {coll} 共 {qd_count} 条")
    except Exception as exc:
        log(f"  ⏭  连不上 Qdrant（{type(exc).__name__}: {exc}）→ C/D 未验证（不算新鲜）")
        cd_unknown = True

    log("")
    if not stale and not cd_unknown:
        log("全部阶段新鲜，无需重跑。")
    elif "A" in stale:
        log("最小重跑: python chatbot.py process && python chatbot.py index")
    elif "B" in stale or "D" in stale:
        log("最小重跑: python chatbot.py index（稠密走缓存增量）")
    elif "C" in stale:
        log("最小重跑: python chatbot.py sparse")
    else:
        # stale 为空但有阶段没验过：绝不能落到上面的"全部新鲜"。
        log("有阶段未能完成验证（原因见上）→ 恢复后重跑: python chatbot.py stages")
    # 退出码与 lexicon/compare-chunks 对齐：有阶段过期或未验证 = 1，全新鲜才 0，
    # 自动化才能用退出码判断"要不要重跑"（以前恒 0，判据形同虚设）。
    return 1 if (stale or cd_unknown) else 0


# ── 命令分发 ──
COMMANDS = { "process": cmd_process, "index": cmd_index, "serve": cmd_serve, "sparse": cmd_sparse, "compare-chunks": cmd_compare_chunks, "lexicon": cmd_verify_lexicon, "stages": cmd_stages, }


def _build_parser():
    """子命令解析器：命令集的单点真相 = COMMANDS（usage 与分发共用同一张表）。

    迁移前每个带参命令自己 new ArgumentParser 再 parse_args()——"一个命令有哪些参数"
    散在 4 处，帮助也各说各话。两处有意的行为变化（都从静默变报错）：
      - `process --help` 以前被零参命令静默忽略（照跑），现在打印该子命令的帮助
      - `lexicon a b c` 第三个位置参数以前被静默丢弃，现在 unrecognized arguments
    """
    parser = argparse.ArgumentParser(prog="chatbot.py", add_help=False)
    subs = parser.add_subparsers(dest="command")
    descriptions = {"compare-chunks": "分块结果 A/B 比较"}
    for name in COMMANDS:
        sub = subs.add_parser(name, add_help=True, description=descriptions.get(name))
        if name == "compare-chunks":
            sub.add_argument("a", help="基线文件（改动前存档的分块产物）")
            sub.add_argument("b", nargs="?", default=DEFAULT_B, help=f"对照文件（默认 {DEFAULT_B}）")
            sub.add_argument("--allow", action="append", default=[], help="允许不同的字段名，可重复指定，例如 --allow contextual_text")
        elif name == "lexicon":
            sub.add_argument("-q", "--quiet", action="store_true", help="只报告问题，不打印正常明细")
            sub.add_argument("alias_path", nargs="?", default=None, help=f"别名表路径（默认 {ALIAS_PATH}）")
            sub.add_argument("stop_path", nargs="?", default=None, help=f"停用词路径（默认 {STOP_PATH}）")
    return parser


def usage():
    log("用法: python chatbot.py <命令> [参数]")
    log("  " + " ".join(COMMANDS))
    log("Streamlit 界面: streamlit run chatbot.py")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        usage()
        return 0 if argv else 1
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        log(f"未知命令: {name}")
        usage()
        return 1
    # 旧协议保留：子命令运行期间 sys.argv 形态为 ["chatbot.py <name>", *rest]。参数已
    # 上收到 _build_parser，这条改写不再服务于解析，但调试/第三方脚本读 sys.argv 仍一致。
    sys.argv = [f"chatbot.py {name}"] + rest
    try:
        ns = _build_parser().parse_args([name, *rest])
    except SystemExit as exc:
        # argparse 的 --help(0)/参数错误(2)：从"就地终止进程"改为经 main 返回，退出码不变。
        return int(exc.code) if exc.code is not None else 0
    try:
        result = COMMANDS[ns.command](ns)
    except Exception as exc:
        # 命令级兜底：受控失败（摘要 + 完整栈 + 退出码 1），不再裸 traceback 中断。
        # SystemExit 不是 Exception 子类，_qdrant_gate 等显式 sys.exit 的退出码不受影响。
        log(f"❌ 命令 {name} 执行失败: {type(exc).__name__}: {exc}")
        log(traceback.format_exc().rstrip())
        return 1
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _in_streamlit = get_script_run_ctx() is not None
    except Exception:
        _in_streamlit = False
    if _in_streamlit:
        run_streamlit()
    else:
        sys.exit(main())
