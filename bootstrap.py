#!/usr/bin/env python3
"""进程级基础设施 —— 每个模块都要用、且**必须只有一份实现**的三样东西。

  * `.env` 加载：必须在**任何常量求值之前**发生；
  * 环境变量解析：取值非法一律启动即报错，不静默回退默认值；
  * 日志：打到 stdout 的 logger（每次 emit 重新解析 sys.stdout）。

为什么必须单源：这三样此前在 app.py / ollama_client.py / rag_engine.py /
query_rewrite.py 里各写一份，于是同一类事故反复以新形式出现 ——

  * D9/D12：`.env` 生不生效**取决于 import 顺序**（谁的 load_dotenv 先跑）；
  * D13：同一个配置在两处各读一遍，`OLLAMA_TIMEOUT` 只对一个入口生效；
  * query_rewrite 的 `_env_bool` 与 rag_engine 的 `_env_bool` 是两份实现。

只要实现还在多处，这类事故就会回来。任何**读环境变量的模块**都应当先
`import bootstrap`（见各模块顶部的注释）。

依赖说明：python-dotenv 是**硬依赖**，缺失即 ImportError 而不是静默跳过 ——
静默跳过会让 .env 里的全部配置失效，程序悄悄跑在代码默认值上。
"""

import logging
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError as exc:  # pragma: no cover - 环境问题，非逻辑分支
    raise ImportError(
        "缺少依赖 python-dotenv，.env 里的全部配置会静默失效"
        "（程序会跑在代码默认值上）。\n安装: pip install python-dotenv"
    ) from exc

# override=False（默认）：命令行环境变量优先，因此 `RAG_X=1 python tools/...`
# 这类 A/B 用法不受影响。
load_dotenv()


# ============================================================================
# 环境变量解析
#
# 取值解析失败一律**启动即报错**，不静默回退默认值 —— 静默回退会让
# "我把阈值调成 0.75 了但没生效"变成最难查的那类问题（.env 里已有同类教训）。
# ============================================================================
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


def env_bool(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def env_choice(name, default, allowed):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    val = raw.strip().lower()
    if val not in allowed:
        raise ValueError(f"环境变量 {name}={raw!r} 不在允许取值 {sorted(allowed)} 内")
    return val


# ============================================================================
# 日志
#
# 自定义 Handler 而不是 logging.StreamHandler(sys.stdout)：后者在**构造时**
# 就绑定当时的 sys.stdout，而 pytest 的 capsys/redirect 会替换 sys.stdout，
# 于是测试期间日志会写进一个已经被丢掉的流 —— 表现为"日志时有时无"。
# 每次都重新取 sys.stdout 可以避免这一点。
#
# 名字相同的 logger 只会配一次 handler（`if not logger.handlers`），
# 因此 rag_engine 与 query_rewrite 共用 "rag_engine" 这个 logger 时，
# 日志仍然是顺序输出的一份，不会重复。
# ============================================================================
class _StdoutHandler(logging.StreamHandler):
    """每次 emit 时重新解析 sys.stdout，兼容 stdout 被替换的运行环境。"""

    def emit(self, record):
        self.stream = sys.stdout
        try:
            super().emit(record)
        except Exception:  # 日志绝不能反过来打断主流程
            self.handleError(record)


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_StdoutHandler())
        logger.setLevel(os.getenv("RAG_LOG_LEVEL", "INFO").strip().upper() or "INFO")
        logger.propagate = False
    return logger
