#!/usr/bin/env bash
# 本地原生四件套一键起停（GPU 档）—— Qdrant + Ollama + 检索模型服务 + 应用，各自独立进程。
#
# 为什么不用 docker compose：Mac 的 Docker 透传不了 Metal，容器里 llama.cpp 与 torch
# 都只能走 CPU；本地要"尽量 GPU"就必须原生（Ollama 走 Metal、infinity 走 MPS）。
# 容器化路径见 docker-compose.yml（自有 Linux 服务器；NVIDIA 主机叠加
# docker-compose.gpu.yml）。魔搭创空间是单容器平台、**不支持 compose**，走 Dockerfile。
#
# 「检索模型服务」= infinity 一个进程托管**两个**模型（bge-m3 + bge-reranker-v2-m3，
# 共享一份 torch 运行时），业务进程因此不再持有检索模型。.env 的
# RAG_EMBED_BACKEND / RAG_RERANK_BACKEND 任一为 http 时才会起它。
#
# 用法:
#   tools/run_local.sh start     # 依次起 Qdrant → Ollama → 检索模型服务 → 应用（已跑的复用）
#   tools/run_local.sh stop      # 只停本脚本记录的进程；Qdrant 交给 run_qdrant.sh
#   tools/run_local.sh status    # 四组件健康 + 设备（MPS）可用性
#   tools/run_local.sh logs      # tail -F 四个日志
#
# 端口: Qdrant 6333 / Ollama 11434 / 检索模型服务 8001（从 RAG_EMBED_URL 派生）/ 应用 8501
# 可覆盖: PYTHON（默认 python）、RAG_LOCAL_PORT（默认 8501）、RAG_MODELS_DEVICE（默认 mps）
# 前置: 检索模型服务需要 pip 装好 infinity-emb（infinity_emb 在 PATH 里）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python}"
APP_PORT="${RAG_LOCAL_PORT:-8501}"
MODELS_DEVICE="${RAG_MODELS_DEVICE:-mps}"
# 检索模型服务的可执行文件：优先仓库内的隔离 venv（.venv/infinity，独立于现役 Python 环境），
# 其次 PATH 里的 infinity_emb；都可用 INFINITY_BIN 显式覆盖。
INFINITY_BIN="${INFINITY_BIN:-}"
if [ -z "${INFINITY_BIN}" ]; then
    if [ -x "${ROOT}/.venv/infinity/bin/infinity_emb" ]; then
        INFINITY_BIN="${ROOT}/.venv/infinity/bin/infinity_emb"
    else
        INFINITY_BIN="infinity_emb"
    fi
fi

QDRANT_URL="http://127.0.0.1:6333"
OLLAMA_URL="http://127.0.0.1:11434"
APP_URL="http://127.0.0.1:${APP_PORT}"

# ── .env → 进程环境：唯一实现在 tools/dotenv.sh（与 entrypoint.sh 共用同一份）────────
# 旧写法在这里自带第三份解析器 _env_val：不认 export 前缀、不剥行内注释、同名键取**最后
# 一次**出现、tr -d 还会把值里任意位置的引号删掉 —— 同一份 .env 在容器与本地读出不同结果
# （失效原型：多份）。_env_val 已删除，改走 _load_dotenv；其语义（进程环境优先、同名重复
# 以最后一次为准并与 python-dotenv 对齐、白名单前缀、MODELS_DIR 在列）逐条写在
# tools/dotenv.sh 头注释里。触发场景：.env 里写 RAG_EMBED_MODEL_PATH="./models/x"、
# MODEL="a b"、或同名键出现两次 —— 旧写法会把引号删掉/取错那次的值。
# 位置很关键：_load_dotenv 只填**未设**的键（进程环境优先），故下面直接 ${VAR:-默认} 读，
# 不再对每个变量各写一遍"进程环境优先"—— 那段重复逻辑本身就是下一个分叉点。
if [ ! -f "${ROOT}/tools/dotenv.sh" ]; then
    echo "[run_local] ❌ 缺少 ${ROOT}/tools/dotenv.sh（.env 解析的唯一实现）—— 退出" >&2
    exit 1
fi
. "${ROOT}/tools/dotenv.sh"
_load_dotenv

EMBED_MODEL_PATH="${RAG_EMBED_MODEL_PATH:-}"
RERANK_MODEL_PATH="${RAG_RERANK_MODEL_PATH:-}"
EMBED_MODEL_NAME="${RAG_EMBED_MODEL_NAME:-}"
RERANK_MODEL_NAME="${RAG_RERANK_MODEL_NAME:-}"
EMBED_BACKEND="${RAG_EMBED_BACKEND:-}"
RERANK_BACKEND="${RAG_RERANK_BACKEND:-}"
# 端口单一来源：RAG_EMBED_URL（服务端 --port 由它派生，改一处即可）
MODELS_PORT="$(printf '%s' "${RAG_EMBED_URL:-}" | sed -n 's|^[a-z]*://[^:/]*:\([0-9][0-9]*\).*|\1|p')"
MODELS_PORT="${MODELS_PORT:-8001}"
MODELS_URL="http://127.0.0.1:${MODELS_PORT}"

QDRANT_LOG="${ROOT}/qdrant.log"
OLLAMA_PIDFILE="${ROOT}/.ollama.pid"
OLLAMA_LOG="${ROOT}/ollama.log"
MODELS_PIDFILE="${ROOT}/.infinity.pid"
MODELS_LOG="${ROOT}/infinity.log"
APP_PIDFILE="${ROOT}/.streamlit.pid"
APP_LOG="${ROOT}/streamlit.log"

_log() { printf '[run_local] %s\n' "$*"; }
_die() { printf '[run_local] ❌ %s\n' "$*" >&2; exit 1; }

# 与 run_qdrant.sh 同一份存活判据：kill -0 对**僵尸**返回 0，故再看一眼进程状态。
# ps 拿不到状态（不存在 / 被沙箱拒绝）时按"存活"处理：kill -0 已经通过，不能因为读不到
# 状态就把活进程误报成已死（那会让 stop 变成"删掉 pid 文件、放走孤儿进程"）。
_pid_alive() {
    local _pid="$1" _st
    [ -n "${_pid}" ] || return 1
    kill -0 "${_pid}" 2>/dev/null || return 1
    _st="$(ps -o stat= -p "${_pid}" 2>/dev/null | head -n1 | tr -d '[:space:]')" || true
    case "${_st}" in Z*) return 1 ;; esac
    return 0
}

_running_pid() {
    local _pidfile="$1" _pid
    [ -f "${_pidfile}" ] || return 1
    _pid="$(cat "${_pidfile}" 2>/dev/null || true)"
    _pid_alive "${_pid}" || return 1
    printf '%s' "${_pid}"
}

_http_ok() { curl -fsS --max-time 3 "$1" >/dev/null 2>&1; }

_fail_with_log() {
    local _msg="$1" _logfile="$2"
    printf '[run_local] ❌ %s\n' "${_msg}" >&2
    if [ -f "${_logfile}" ]; then
        printf '[run_local] 最后 20 行 %s：\n' "${_logfile}" >&2
        tail -n 20 "${_logfile}" >&2 || true
    fi
    exit 1
}

# 等 HTTP 就绪；进程提前退出立刻报错并带日志，而不是干等到超时。
# 参数: url 名称 pidfile(可空) 日志 超时秒
_wait_http() {
    local _url="$1" _name="$2" _pidfile="$3" _logfile="$4" _timeout="$5" _i _pid
    for _i in $(seq 1 "${_timeout}"); do
        _http_ok "${_url}" && return 0
        if [ -n "${_pidfile}" ] && [ -f "${_pidfile}" ]; then
            _pid="$(cat "${_pidfile}" 2>/dev/null || true)"
            _pid_alive "${_pid}" || _fail_with_log "${_name} 启动后很快退出" "${_logfile}"
        fi
        sleep 1
    done
    _fail_with_log "${_name} ${_timeout} 秒内未就绪（${_url}）" "${_logfile}"
}

# infinity 要求**绝对路径**：相对路径会被 sentence-transformers 当成 HF repo id，
# 直接抛 HFValidationError（本地实测）。.env 里写的是仓库相对路径，这里统一转绝对。
_abs_path() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *)  printf '%s/%s' "${ROOT}" "${1#./}" ;;
    esac
}

_models_backend_http() {
    [ "${EMBED_BACKEND:-torch}" = "http" ] || [ "${RERANK_BACKEND:-torch}" = "http" ]
}

_start_qdrant() {
    _log "1/4 Qdrant"
    if _http_ok "${QDRANT_URL}/healthz"; then
        _log "  已有实例在服务（${QDRANT_URL}）—— 复用"
        return 0
    fi
    "$ROOT/tools/run_qdrant.sh" start
}

_start_ollama() {
    _log "2/4 Ollama（原生，Metal GPU）"
    if _http_ok "${OLLAMA_URL}/api/version"; then
        _log "  已有实例在服务 —— 复用（Ollama.app 或上一次 ollama serve）"
        return 0
    fi
    command -v ollama >/dev/null 2>&1 || _die "PATH 里找不到 ollama（本机应在 /usr/local/bin/ollama）"
    nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &
    printf '%s' "$!" >"${OLLAMA_PIDFILE}"
    _wait_http "${OLLAMA_URL}/api/version" "Ollama" "${OLLAMA_PIDFILE}" "${OLLAMA_LOG}" 30
    _log "  就绪（日志 ${OLLAMA_LOG}）"
}

_start_models() {
    _log "3/4 检索模型服务（infinity：一个进程托管嵌入 + 重排）"
    if ! _models_backend_http; then
        _log "  .env 里 RAG_EMBED_BACKEND/RAG_RERANK_BACKEND 都不是 http —— 跳过（检索模型在应用进程内）"
        return 0
    fi
    if _http_ok "${MODELS_URL}/health"; then
        _log "  已有实例在服务（${MODELS_URL}）—— 复用"
        return 0
    fi
    if ! command -v "${INFINITY_BIN}" >/dev/null 2>&1 && [ ! -x "${INFINITY_BIN}" ]; then
        _die "找不到 infinity_emb（$INFINITY_BIN）：先 python -m venv .venv/infinity && .venv/infinity/bin/pip install 'infinity-emb[server,torch]==0.0.77'"
    fi
    local _embed_dir _rerank_dir
    _embed_dir="$(_abs_path "${EMBED_MODEL_PATH:-./models/bge-m3}")"
    _rerank_dir="$(_abs_path "${RERANK_MODEL_PATH:-./models/bge-reranker-base}")"
    local _embed_served="${EMBED_MODEL_NAME:-$(basename "${_embed_dir}")}"
    local _rerank_served="${RERANK_MODEL_NAME:-$(basename "${_rerank_dir}")}"
    for _d in "${_embed_dir}" "${_rerank_dir}"; do
        [ -d "${_d}" ] || _die "模型目录不存在: ${_d}（本地原生路径按仓库根解析）"
    done
    # infinity 的 sentence-transformers 需要 modules.json 里列出的模块子目录（bge-m3 的
    # 1_Pooling 本地没有）→ 用**影子视图**补齐，真实模型目录一个字节都不动（否则切分指纹漂移）。
    _view_root="${ROOT}/.cache/infinity-models"
    "${PY}" "${ROOT}/tools/prepare_infinity_view.py" "${_embed_dir}" "${_view_root}/$(basename "${_embed_dir}")" | sed 's/^/  /'
    # 重排还要对齐**截断长度**：应用侧按 RAG_RERANK_MAX_LENGTH（默认 384）截，infinity 默认按
    # tokenizer 的 model_max_length（bge-reranker-v2-m3 是 8192）—— 不对齐长候选的排序会变。
    _rr_max="${RAG_RERANK_MAX_LENGTH:-384}"
    "${PY}" "${ROOT}/tools/prepare_infinity_view.py" "${_rerank_dir}" "${_view_root}/$(basename "${_rerank_dir}")" --max-length "${_rr_max:-384}" | sed 's/^/  /'
    _embed_dir="${_view_root}/$(basename "${_embed_dir}")"
    _rerank_dir="${_view_root}/$(basename "${_rerank_dir}")"
    # --no-bettertransformer 必须显式关（否则走 optimum/bettertransformer，torch 依赖集里没有）
    nohup "${INFINITY_BIN}" v2 \
        --model-id "${_embed_dir}" --served-model-name "${_embed_served}" \
        --model-id "${_rerank_dir}" --served-model-name "${_rerank_served}" \
        --engine torch --device "${MODELS_DEVICE}" --no-bettertransformer \
        --url-prefix /v1 --host 127.0.0.1 --port "${MODELS_PORT}" \
        >"${MODELS_LOG}" 2>&1 &
    printf '%s' "$!" >"${MODELS_PIDFILE}"
    # 两个模型依次加载，冷启动比单模型久：给 180s（Mac 上 fp32 权重 2.1G × 2）
    _wait_http "${MODELS_URL}/health" "检索模型服务" "${MODELS_PIDFILE}" "${MODELS_LOG}" 180
    _log "  就绪：${MODELS_URL}（embed=${_embed_served} rerank=${_rerank_served} device=${MODELS_DEVICE}）"
}

_start_app() {
    _log "4/4 应用（Streamlit）"
    if _models_backend_http && ! _http_ok "${MODELS_URL}/health"; then
        _log "⚠️  .env 配的是 http 检索后端，但 ${MODELS_URL}/health 不通 —— 查询会显式报错。"
        _log "    先把检索模型服务起起来（tools/run_local.sh start 的第 3 步），或把后端改回 torch。"
    fi
    if _http_ok "${APP_URL}/_stcore/health"; then
        _log "  已有实例在服务：${APP_URL} —— 复用（改过代码要先 stop 再 start）"
        return 0
    fi
    if _running_pid "${APP_PIDFILE}" >/dev/null; then
        _log "  已有实例在运行，等待就绪"
    else
        nohup "$PY" chatbot.py serve >"${APP_LOG}" 2>&1 &
        printf '%s' "$!" >"${APP_PIDFILE}"
    fi
    _wait_http "${APP_URL}/_stcore/health" "应用" "${APP_PIDFILE}" "${APP_LOG}" 90
    _log "  就绪：${APP_URL}（日志 ${APP_LOG}）"
}

_stop_pidfile() {
    local _name="$1" _pidfile="$2" _pid _i
    if ! _pid="$(_running_pid "${_pidfile}")"; then
        _log "${_name}：本脚本未记录运行中的进程（pid 文件 ${_pidfile}）"
        rm -f "${_pidfile}"
        return 0
    fi
    _log "${_name}：停止 pid=${_pid}"
    kill "${_pid}" 2>/dev/null || true
    for _i in $(seq 1 20); do _pid_alive "${_pid}" || break; sleep 0.5; done
    _pid_alive "${_pid}" && kill -9 "${_pid}" 2>/dev/null || true
    rm -f "${_pidfile}"
    _log "${_name}：已停止"
}

_do_start() {
    _start_qdrant
    _start_ollama
    _start_models
    _start_app
    _log "四件套就绪：应用 ${APP_URL} ／ Qdrant ${QDRANT_URL} ／ Ollama ${OLLAMA_URL} ／ 检索模型 ${MODELS_URL}"
    _log "停止：tools/run_local.sh stop"
}

_do_stop() {
    _stop_pidfile "应用" "${APP_PIDFILE}"
    _stop_pidfile "检索模型服务" "${MODELS_PIDFILE}"
    _stop_pidfile "Ollama" "${OLLAMA_PIDFILE}"
    "$ROOT/tools/run_qdrant.sh" stop
}

_gpu_line() {
    local _out
    _out="$("$PY" -c 'import torch; print("MPS 可用（检索模型服务 --device mps）" if torch.backends.mps.is_available() else "MPS 不可用（将回退 CPU）")' 2>/dev/null || true)"
    printf '%s' "${_out:-无法判断（torch 导入失败）}"
}

_do_status() {
    if _http_ok "${QDRANT_URL}/healthz"; then
        _log "Qdrant: 就绪（${QDRANT_URL}）"
    else
        _log "Qdrant: 未响应（${QDRANT_URL}）"
    fi
    "$ROOT/tools/run_qdrant.sh" status || true
    if _http_ok "${OLLAMA_URL}/api/version"; then
        _log "Ollama: 就绪（$(curl -fsS --max-time 3 "${OLLAMA_URL}/api/version" 2>/dev/null || echo '?')）"
        if command -v ollama >/dev/null 2>&1; then
            ollama ps 2>/dev/null | sed 's/^/         /' || true
        fi
    else
        _log "Ollama: 未响应（${OLLAMA_URL}）"
    fi
    if ! _models_backend_http; then
        _log "检索模型服务: 未启用（后端是 torch，模型在应用进程内）"
    elif _http_ok "${MODELS_URL}/health"; then
        _log "检索模型服务: 就绪（${MODELS_URL}，设备 ${MODELS_DEVICE}）pid=$(_running_pid "${MODELS_PIDFILE}" || echo '非本脚本')"
        curl -fsS --max-time 3 "${MODELS_URL}/v1/models" 2>/dev/null | head -c 400 | sed 's/^/         /' || true
        echo
    else
        _log "检索模型服务: 未响应（${MODELS_URL}）"
    fi
    if _http_ok "${APP_URL}/_stcore/health"; then
        _log "应用: 就绪（${APP_URL}）pid=$(_running_pid "${APP_PIDFILE}" || echo '非本脚本')"
    else
        _log "应用: 未响应（${APP_URL}）"
    fi
    _log "设备: $(_gpu_line)"
}

_do_logs() {
    _log "tail -F（Ctrl+C 退出）：${QDRANT_LOG} ${OLLAMA_LOG} ${MODELS_LOG} ${APP_LOG}"
    tail -F "${QDRANT_LOG}" "${OLLAMA_LOG}" "${MODELS_LOG}" "${APP_LOG}"
}

case "${1:-status}" in
    start)  _do_start ;;
    stop)   _do_stop ;;
    status) _do_status ;;
    logs)   _do_logs ;;
    models) _start_models ;;   # 只起检索模型服务（比对/排障时不必把四件套都拉起）
    *) echo "用法: $0 {start|stop|status|logs|models}" >&2; exit 2 ;;
esac
