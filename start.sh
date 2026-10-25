#!/usr/bin/env bash
# ── 单文件启动器：Qdrant + Ollama + Streamlit（本机原生，无 Docker）──
# 本仓库唯一的 shell 脚本；原 tools/ 的起停逻辑已合并到这里。
#
# 用法:
#   ./start.sh start             起 Qdrant → Ollama → 应用（已在跑的复用）
#   ./start.sh stop              停本脚本记录的三个进程
#   ./start.sh restart           先停后起
#   ./start.sh status            健康状态 + pid
#   ./start.sh logs              tail -F 三个日志
#   ./start.sh serve             只起 Streamlit（前台，调试用）
#   ./start.sh process           分块（python chatbot.py process）
#   ./start.sh index             建索引
#   ./start.sh sparse            重建稀疏向量
#   ./start.sh stages            四阶段新鲜度检查
#   ./start.sh lexicon           词表自检
#   ./start.sh compare-chunks A [B]
#
# 前置:
#   · Python 3.10 + pip install -r requirements.txt
#   · Ollama 已安装，且 .env 的 MODEL 已 ollama pull
#   · Qdrant 可执行文件放在 .bin/qdrant/qdrant（或设 QDRANT_BIN）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python}"
APP_PORT="${RAG_LOCAL_PORT:-8501}"
QDRANT_URL="http://127.0.0.1:6333"
OLLAMA_URL="http://127.0.0.1:11434"
APP_URL="http://127.0.0.1:${APP_PORT}"
QDRANT_BIN="${QDRANT_BIN:-${ROOT}/.bin/qdrant/qdrant}"
QDRANT_LOG="${ROOT}/qdrant.log"
OLLAMA_LOG="${ROOT}/ollama.log"
APP_LOG="${ROOT}/streamlit.log"
QDRANT_PID="${ROOT}/.qdrant.pid"
OLLAMA_PID="${ROOT}/.ollama.pid"
APP_PID="${ROOT}/.streamlit.pid"

_log() { printf '[start] %s\n' "${*}"; }
_die() { printf '[start] ❌ %s\n' "${*}" >&2; exit 1; }
_http_ok() { curl -fsS --max-time 2 "${1}" >/dev/null 2>&1; }

_pid_alive() {
    local _pid="${1}" _st
    [ -n "${_pid}" ] || return 1
    kill -0 "${_pid}" 2>/dev/null || return 1
    _st="$(ps -o stat= -p "${_pid}" 2>/dev/null | head -n1 | tr -d '[:space:]')" || true
    case "${_st}" in Z*) return 1 ;; esac
    return 0
}

_running_pid() {
    local _file="${1}" _pid
    [ -f "${_file}" ] || return 1
    _pid="$(cat "${_file}" 2>/dev/null || true)"
    _pid_alive "${_pid}" || return 1
    printf '%s' "${_pid}"
}

_wait_http() {
    local _url="${1}" _name="${2}" _pidfile="${3}" _logfile="${4}" _timeout="${5}" _i
    for _i in $(seq 1 "${_timeout}"); do
        _http_ok "${_url}" && return 0
        if [ -f "${_pidfile}" ] && ! _pid_alive "$(cat "${_pidfile}" 2>/dev/null || true)"; then
            _log "${_name} 异常退出，最后 50 行日志："
            tail -n 50 "${_logfile}" 2>/dev/null || true
            return 1
        fi
        sleep 1
    done
    _log "${_name} ${_timeout}s 未就绪，最后 50 行日志："
    tail -n 50 "${_logfile}" 2>/dev/null || true
    return 1
}

# ── .env → 进程环境（原 tools/dotenv.sh 的唯一实现，合并进来）──
# 空值/注释跳过、export 前缀剥离、同名重复最后一次生效、进程环境优先、只导出白名单前缀。
_load_dotenv() {
    local _file="${1:-.env}"
    [ -f "${_file}" ] || return 0
    local _line _k _v _preset
    _preset=" $(env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' | tr '\n' ' ') "
    while IFS= read -r _line || [ -n "${_line}" ]; do
        _line="${_line%$'\r'}"
        case "${_line}" in ''|\#*) continue ;; esac
        _line="$(printf '%s' "${_line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        _line="${_line#export }"
        case "${_line}" in [A-Za-z_]*=*) ;; *) continue ;; esac
        _k="${_line%%=*}"
        _k="$(printf '%s' "${_k}" | sed -e 's/[[:space:]]*$//')"
        case "${_k}" in [A-Za-z_][A-Za-z0-9_]*) ;; *) continue ;; esac
        case "${_k}" in
            RAG_*|OLLAMA_*|QDRANT_*|SEMANTIC_*|INDEX_*|MODEL|MODELS_DIR|HF_ENDPOINT|HF_HUB_*|PYTORCH_MPS_*) ;;
            *) continue ;;
        esac
        case "${_preset}" in *" ${_k} "*) continue ;; esac
        _v="${_line#*=}"
        _v="$(printf '%s' "${_v}" | sed -e 's/^[[:space:]]*//')"
        case "${_v}" in
            \"*) _v="${_v#\"}"; _v="${_v%%\"*}" ;;
            \'*) _v="${_v#\'}"; _v="${_v%%\'*}" ;;
            *) _v="$(printf '%s' "${_v}" | sed -e 's/[[:space:]]\{1,\}#.*$//' -e 's/[[:space:]]*$//')" ;;
        esac
        export "${_k}=${_v}"
    done < "${_file}"
}

_check_backends() {
    case "${RAG_EMBED_BACKEND:-torch}/${RAG_RERANK_BACKEND:-torch}" in
        *http*) _log "⚠️  .env 配置了 http 检索后端；本发行版不内置 infinity 服务，查询会报错。"
                _log "    改回 RAG_EMBED_BACKEND=torch / RAG_RERANK_BACKEND=torch 即可进程内运行。" ;;
    esac
}

_start_qdrant() {
    if _http_ok "${QDRANT_URL}/healthz"; then
        _log "Qdrant 已在运行（复用）"
        return 0
    fi
    [ -x "${QDRANT_BIN}" ] || _die "缺少 Qdrant 可执行文件: ${QDRANT_BIN}（下载后放到 .bin/qdrant/qdrant，或设 QDRANT_BIN）"
    mkdir -p "${ROOT}/storage"
    export QDRANT__SERVICE__HOST="127.0.0.1"
    export QDRANT__STORAGE__STORAGE_PATH="${ROOT}/storage"
    _log "启动 Qdrant（storage=${ROOT}/storage, bind=127.0.0.1:6333）"
    nohup "${QDRANT_BIN}" --config-path "${ROOT}/qdrant_config/config.yaml" >"${QDRANT_LOG}" 2>&1 &
    printf '%s' "$!" >"${QDRANT_PID}"
    _wait_http "${QDRANT_URL}/healthz" "Qdrant" "${QDRANT_PID}" "${QDRANT_LOG}" 60 || _die "Qdrant 启动失败"
    _log "Qdrant 就绪：${QDRANT_URL}（日志 ${QDRANT_LOG}）"
}

_start_ollama() {
    if _http_ok "${OLLAMA_URL}/api/version"; then
        _log "Ollama 已在运行（复用；改 .env 的 OLLAMA_* 后需 stop 再 start）"
        return 0
    fi
    command -v ollama >/dev/null 2>&1 || _die "PATH 里找不到 ollama"
    _log "启动 Ollama"
    nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &
    printf '%s' "$!" >"${OLLAMA_PID}"
    _wait_http "${OLLAMA_URL}/api/version" "Ollama" "${OLLAMA_PID}" "${OLLAMA_LOG}" 30 || _die "Ollama 启动失败"
    _log "Ollama 就绪：${OLLAMA_URL}（日志 ${OLLAMA_LOG}）"
}

_start_app() {
    if _http_ok "${APP_URL}/_stcore/health"; then
        _log "应用已在运行（复用；改代码/改 .env 后需 stop 再 start）"
        return 0
    fi
    [ -f "${ROOT}/chatbot.py" ] || _die "缺少 chatbot.py"
    _log "启动 Streamlit 应用"
    nohup "${PY}" chatbot.py serve >"${APP_LOG}" 2>&1 &
    printf '%s' "$!" >"${APP_PID}"
    _wait_http "${APP_URL}/_stcore/health" "应用" "${APP_PID}" "${APP_LOG}" 90 || _die "应用启动失败"
    _log "应用就绪：${APP_URL}（日志 ${APP_LOG}）"
}

_stop_pidfile() {
    local _name="${1}" _file="${2}" _pid _i
    if ! _pid="$(_running_pid "${_file}")"; then
        rm -f "${_file}"
        return 0
    fi
    _log "停止 ${_name} pid=${_pid}"
    kill "${_pid}" 2>/dev/null || true
    for _i in $(seq 1 20); do
        _pid_alive "${_pid}" || break
        sleep 0.5
    done
    if _pid_alive "${_pid}"; then
        _log "${_name} 未在 10s 内退出，SIGKILL"
        kill -9 "${_pid}" 2>/dev/null || true
    fi
    rm -f "${_file}"
}

_stop_all() {
    _stop_pidfile "应用" "${APP_PID}"
    _stop_pidfile "Ollama" "${OLLAMA_PID}"
    _stop_pidfile "Qdrant" "${QDRANT_PID}"
}

_status_one() {
    local _name="${1}" _url="${2}" _file="${3}" _pid
    if _http_ok "${_url}"; then
        _pid="$(_running_pid "${_file}" || true)"
        printf '✅ %-8s 健康  %s  pid=%s\n' "${_name}" "${_url}" "${_pid:-外部实例}"
    else
        printf '❌ %-8s 未响应 %s\n' "${_name}" "${_url}"
    fi
}

_status_all() {
    _status_one "Qdrant" "${QDRANT_URL}/healthz" "${QDRANT_PID}"
    _status_one "Ollama" "${OLLAMA_URL}/api/version" "${OLLAMA_PID}"
    _status_one "应用" "${APP_URL}/_stcore/health" "${APP_PID}"
}

_detect_hw() {
    if [ -z "${HW_CORES:-}" ]; then
        local _c
        _c="$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
        export HW_CORES="${_c}"
    fi
    if [ -z "${HW_MEM_MB:-}" ]; then
        local _b
        _b="$(sysctl -n hw.memsize 2>/dev/null || awk '/MemTotal/{print $2*1024}' /proc/meminfo 2>/dev/null || true)"
        if [ -n "${_b}" ]; then export HW_MEM_MB="$((_b / 1048576))"; fi
    fi
}

_usage() { sed -n '2,22p' "${ROOT}/start.sh"; }

_load_dotenv
_detect_hw
case "${1:-start}" in
    start)
        _check_backends
        _start_qdrant
        _start_ollama
        _start_app
        _log "全部就绪：应用 ${APP_URL} ／ Qdrant ${QDRANT_URL} ／ Ollama ${OLLAMA_URL}"
        ;;
    stop)    _stop_all ;;
    restart) _stop_all; "${ROOT}/start.sh" start ;;
    status)  _status_all ;;
    logs)    tail -F "${QDRANT_LOG}" "${OLLAMA_LOG}" "${APP_LOG}" ;;
    serve)   _check_backends; exec "${PY}" chatbot.py serve ;;
    process|index|sparse|stages|lexicon|compare-chunks)
        _cmd="${1}"; shift
        exec "${PY}" chatbot.py "${_cmd}" "$@"
        ;;
    help|-h|--help) _usage ;;
    *) _usage; exit 2 ;;
esac
