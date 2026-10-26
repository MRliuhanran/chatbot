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
#   ./start.sh fetch-models [dest] 构建期下载模型（Docker 也复用本文件）
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

# 只复用“本脚本启动的进程”：pid 文件活着 + 命令行匹配。健康端点本身不证明身份，
# 否则别的服务/旧进程占着端口会被误当成发布成功（假发布）。
_owned_pid() {
    local _file="${1}" _marker="${2}" _pid _cmd
    _pid="$(_running_pid "${_file}")" || return 1
    _cmd="$(ps -ww -p "${_pid}" -o command= 2>/dev/null || true)"
    case "${_cmd}" in
        *"${_marker}"*) printf '%s' "${_pid}"; return 0 ;;
    esac
    # 某些沙箱/平台会拒绝 ps 读其它进程；退到 pgrep 精确匹配 pid。
    if command -v pgrep >/dev/null 2>&1 && pgrep -f "${_marker}" 2>/dev/null | grep -qx "${_pid}"; then
        printf '%s' "${_pid}"
        return 0
    fi
    return 1
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
        if _owned_pid "${QDRANT_PID}" "qdrant" >/dev/null; then
            _log "Qdrant 已在运行（本脚本实例，复用）"
            return 0
        fi
        _die "6333 上有健康服务，但不是本脚本启动的 Qdrant；先停止外部实例再 start"
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
        if _owned_pid "${OLLAMA_PID}" "ollama serve" >/dev/null; then
            _log "Ollama 已在运行（本脚本实例，复用；改 .env 的 OLLAMA_* 后需 stop 再 start）"
            return 0
        fi
        _die "11434 上有健康服务，但不是本脚本启动的 Ollama；先停止外部实例再 start"
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
        if _owned_pid "${APP_PID}" "chatbot.py" >/dev/null; then
            _log "应用已在运行（本脚本实例，复用；改代码/改 .env 后需 stop 再 start）"
            return 0
        fi
        _die "${APP_PORT} 上有健康服务，但不是本脚本启动的 Streamlit；先停止外部实例再 start"
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
    local _name="${1}" _url="${2}" _file="${3}" _marker="${4}" _pid
    if _http_ok "${_url}"; then
        _pid="$(_owned_pid "${_file}" "${_marker}" || true)"
        if [ -n "${_pid}" ]; then
            printf '✅ %-8s 健康  %s  本脚本 pid=%s\n' "${_name}" "${_url}" "${_pid}"
        else
            printf '⚠️  %-8s 健康  %s  外部实例（start 会拒绝复用）\n' "${_name}" "${_url}"
        fi
    else
        printf '❌ %-8s 未响应 %s\n' "${_name}" "${_url}"
    fi
}

_status_all() {
    _status_one "Qdrant" "${QDRANT_URL}/healthz" "${QDRANT_PID}" "qdrant"
    _status_one "Ollama" "${OLLAMA_URL}/api/version" "${OLLAMA_PID}" "ollama serve"
    _status_one "应用" "${APP_URL}/_stcore/health" "${APP_PID}" "chatbot.py"
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


# ── 构建期模型下载：唯一 shell 文件内完成，Docker 与本机共用 ────────────────
# 替代 Dockerfile 的 17 个远程 ADD（远程 ADD 不走层缓存，每次部署重下 4GB）。
_fetch_models() {
    local _dest="${1:-/models}" _input _item _rel _url _size _file _dir _need=0 _got
    local _bge="https://modelscope.cn/models/BAAI/bge-m3/resolve/master"
    local _rr="https://modelscope.cn/models/BAAI/bge-reranker-v2-m3/resolve/master"
    local _gguf_rel="_gguf/MiniCPM5-2B-Q4_K_M.gguf"
    local _gguf_url="https://modelscope.cn/models/OpenBMB/MiniCPM5-2B-gguf/resolve/master/MiniCPM5-2B-Q4_K_M.gguf"
    local _entries=(
        "bge-m3/colbert_linear.pt|${_bge}/colbert_linear.pt|2100674"
        "bge-m3/config.json|${_bge}/config.json|687"
        "bge-m3/config_sentence_transformers.json|${_bge}/config_sentence_transformers.json|123"
        "bge-m3/modules.json|${_bge}/modules.json|349"
        "bge-m3/pytorch_model.bin|${_bge}/pytorch_model.bin|2271145830"
        "bge-m3/sentence_bert_config.json|${_bge}/sentence_bert_config.json|54"
        "bge-m3/sentencepiece.bpe.model|${_bge}/sentencepiece.bpe.model|5069051"
        "bge-m3/sparse_linear.pt|${_bge}/sparse_linear.pt|3516"
        "bge-m3/special_tokens_map.json|${_bge}/special_tokens_map.json|964"
        "bge-m3/tokenizer.json|${_bge}/tokenizer.json|17098108"
        "bge-m3/tokenizer_config.json|${_bge}/tokenizer_config.json|444"
        "bge-reranker-v2-m3/config.json|${_rr}/config.json|795"
        "bge-reranker-v2-m3/model.safetensors|${_rr}/model.safetensors|2271071852"
        "bge-reranker-v2-m3/sentencepiece.bpe.model|${_rr}/sentencepiece.bpe.model|5069051"
        "bge-reranker-v2-m3/special_tokens_map.json|${_rr}/special_tokens_map.json|964"
        "bge-reranker-v2-m3/tokenizer.json|${_rr}/tokenizer.json|17098273"
        "bge-reranker-v2-m3/tokenizer_config.json|${_rr}/tokenizer_config.json|1173"
    )
    mkdir -p "${_dest}"
    _input="$(mktemp "${TMPDIR:-/tmp}/rag-models.XXXXXX")"
    for _item in "${_entries[@]}"; do
        IFS='|' read -r _rel _url _size <<< "${_item}"
        _file="${_dest}/${_rel}"
        if [ -f "${_file}" ]; then
            _got="$(wc -c < "${_file}" | tr -d '[:space:]')"
            [ "${_got}" = "${_size}" ] && { _log "跳过已就绪: ${_rel}"; continue; }
        fi
        rm -f "${_file}"
        _dir="${_dest}/$(dirname "${_rel}")"
        mkdir -p "${_dir}"
        printf '%s\n  dir=%s\n  out=%s\n' "${_url}" "${_dir}" "$(basename "${_rel}")" >> "${_input}"
        _need=1
    done
    _file="${_dest}/${_gguf_rel}"
    _got="$(wc -c < "${_file}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [ -z "${_got}" ] || [ "${_got}" -le 100000000 ]; then
        rm -f "${_file}"
        mkdir -p "$(dirname "${_file}")"
        printf '%s\n  dir=%s\n  out=%s\n' "${_gguf_url}" "$(dirname "${_file}")" "$(basename "${_file}")" >> "${_input}"
        _need=1
    fi
    if [ "${_need}" = 1 ]; then
        if command -v aria2c >/dev/null 2>&1; then
            _log "aria2 多连接下载模型"
            aria2c -i "${_input}" \
                --max-concurrent-downloads=8 --split=16 --max-connection-per-server=16 \
                --min-split-size=1M --file-allocation=none --continue=true \
                --allow-overwrite=true --auto-file-renaming=false \
                --retry-wait=2 --max-tries=5 --console-log-level=warn || \
                _log "⚠️ aria2 返回非零，继续做完整性校验"
        else
            _log "未找到 aria2c，使用 curl 顺序下载"
            for _item in "${_entries[@]}"; do
                IFS='|' read -r _rel _url _size <<< "${_item}"
                _file="${_dest}/${_rel}"
                _got="$(wc -c < "${_file}" 2>/dev/null | tr -d '[:space:]' || true)"
                [ "${_got}" = "${_size}" ] && continue
                _log "curl 下载: ${_rel}"
                curl -fL --http1.1 --connect-timeout 20 --max-time 1200 \
                    --retry 5 --retry-delay 3 --retry-all-errors -o "${_file}" "${_url}" || _die "下载失败: ${_url}"
            done
            _file="${_dest}/${_gguf_rel}"
            _got="$(wc -c < "${_file}" 2>/dev/null | tr -d '[:space:]' || true)"
            if [ -z "${_got}" ] || [ "${_got}" -le 100000000 ]; then
                curl -fL --http1.1 --connect-timeout 20 --max-time 1800 \
                    --retry 5 --retry-delay 3 --retry-all-errors -o "${_file}" "${_gguf_url}" || _die "下载失败: ${_gguf_url}"
            fi
        fi
    fi
    rm -f "${_input}"
    for _item in "${_entries[@]}"; do
        IFS='|' read -r _rel _url _size <<< "${_item}"
        _file="${_dest}/${_rel}"
        _got="$(wc -c < "${_file}" 2>/dev/null | tr -d '[:space:]' || true)"
        [ "${_got}" = "${_size}" ] || _die "模型文件大小不符: ${_rel}（期望 ${_size}，实际 ${_got:-缺失}）"
    done
    _file="${_dest}/${_gguf_rel}"
    _got="$(wc -c < "${_file}" 2>/dev/null | tr -d '[:space:]' || true)"
    [ -n "${_got}" ] && [ "${_got}" -gt 100000000 ] || _die "GGUF 文件缺失或过小: ${_gguf_rel}"
    _log "模型下载完成: ${_dest}"
}

_usage() { sed -n '2,23p' "${ROOT}/start.sh"; }

_load_dotenv
_detect_hw
case "${1:-start}" in
    fetch-models) shift; _fetch_models "$@" ;;
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
