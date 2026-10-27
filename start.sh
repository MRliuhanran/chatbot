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


# curl 单文件下载/补齐：fetch-models 的三处调用共用（aria2 残留补齐、无 aria2 时的降级下载）。
# 以前补齐分支直接调用了一个从未定义的 _fetch_curl_one —— aria2 断连必触发
# `_fetch_curl_one: command not found`（set -e 下 exit 127），救断连的逻辑自己先崩。
_fetch_curl_one() {
    local _url="${1}" _file="${2}" _max_time="${3:-1800}"
    _log "curl 下载: ${_file}"
    # -C - 先续传 aria2/上一轮 curl 留下的前缀字节（aria2 --file-allocation=none，落盘即真实
    # 已下载量，可直接作 Range 起点）；续传被拒（服务端不支持 Range / 前缀异常）再整文件重下。
    # 返回值交给调用方处置：补齐循环里单文件失败**不再 _die 杀死整次构建**——build 2026-10-27
    # 21:34 正是这样死于"一个文件补齐失败=全部报废"；外层重试轮 + 终局校验兜底。
    if curl -fL --http1.1 -C - --connect-timeout 20 --max-time "${_max_time}" \
        --retry 5 --retry-delay 3 --retry-all-errors -o "${_file}" "${_url}"; then
        return 0
    fi
    _log "curl 续传未成功，整文件重下: ${_file}"
    rm -f "${_file}"
    curl -fL --http1.1 --connect-timeout 20 --max-time "${_max_time}" \
        --retry 5 --retry-delay 3 --retry-all-errors -o "${_file}" "${_url}"
}

# ── 构建期模型下载：唯一 shell 文件内完成，Docker 与本机共用 ────────────────
# 替代 Dockerfile 的 17 个远程 ADD（远程 ADD 不走层缓存，每次部署重下 4GB）。
# 用法: fetch-models [dest] [group...]   group ∈ {gguf|bge-m3|bge-reranker-v2-m3|all}，默认 all。
# 分组的意义：Dockerfile 按组拆成 3 个 RUN —— 单组网络失败只重下该组，已完成组留在
# 层缓存；此前 5.9G 全在同一个 RUN，任何一次 CDN 抖动都全量重来（2026-10-27 两次
# 构建都死在整组下载中段）。组内再带 3 轮「aria2 → curl -C - 续传补齐 → 校验」重试。
_fetch_models() {
    local _dest="${1:-/models}"; shift
    local _groups="${*:-all}"
    # 组名 = 模型目录名（gguf 是唯一的例外，目录 "_gguf" 以外的特判）。
    # 曾经想用短名 "reranker" 当别名 —— 组按 rel 一级目录名过滤，别名永远匹配不上、
    # reranker 组会静默下载 0 个文件（本地替身测试抓到），不做映射、名字保持一致。
    case " ${_groups} " in *" all "*) _groups="gguf bge-m3 bge-reranker-v2-m3" ;; esac
    local _bge="https://modelscope.cn/models/BAAI/bge-m3/resolve/master"
    local _rr="https://modelscope.cn/models/BAAI/bge-reranker-v2-m3/resolve/master"
    local _gguf_rel="_gguf/MiniCPM5-2B-Q4_K_M.gguf"
    local _gguf_url="https://modelscope.cn/models/OpenBMB/MiniCPM5-2B-gguf/resolve/master/MiniCPM5-2B-Q4_K_M.gguf"
    # rel|url|size（size 必须与源站一字节不差；GGUF 无稳定清单大小，单独按 >100MB 校验）
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
    # 组过滤：按 rel 的一级目录名匹配（gguf 不在 _entries 里，单独开关）
    local _sel=() _item _rel _url _size
    for _item in "${_entries[@]}"; do
        _rel="${_item%%|*}"
        case " ${_groups} " in *" ${_rel%%/*} "*) _sel+=("${_item}") ;; esac
    done
    local _want_gguf=0
    case " ${_groups} " in *" gguf "*) _want_gguf=1 ;; esac

    local _attempt _input _file _dir _got _need _ok
    for _attempt in 1 2 3; do
        # —— 收集本组缺失文件（幂等：完整即跳过；不完整的把 .aria2 控制文件与数据文件
        #    一起重置 —— 只删数据会留下孤儿控制文件，反而让 aria2 续传错乱）——
        _input="$(mktemp "${TMPDIR:-/tmp}/rag-models.XXXXXX")"
        _need=0
        for _item in ${_sel[@]+"${_sel[@]}"}; do
            IFS='|' read -r _rel _url _size <<< "${_item}"
            _file="${_dest}/${_rel}"
            if [ -f "${_file}" ]; then
                _got="$(wc -c < "${_file}" | tr -d '[:space:]')"
                [ "${_got}" = "${_size}" ] && { _log "跳过已就绪: ${_rel}"; continue; }
            fi
            rm -f "${_file}" "${_file}.aria2"
            _dir="${_dest}/$(dirname "${_rel}")"
            mkdir -p "${_dir}"
            printf '%s\n  dir=%s\n  out=%s\n' "${_url}" "${_dir}" "$(basename "${_rel}")" >> "${_input}"
            _need=1
        done
        if [ "${_want_gguf}" = 1 ]; then
            _file="${_dest}/${_gguf_rel}"
            mkdir -p "$(dirname "${_file}")"
            _got="$(wc -c < "${_file}" 2>/dev/null | tr -d '[:space:]' || true)"
            if [ -z "${_got}" ] || [ "${_got}" -le 100000000 ]; then
                rm -f "${_file}" "${_file}.aria2"
                printf '%s\n  dir=%s\n  out=%s\n' "${_gguf_url}" "$(dirname "${_file}")" "$(basename "${_file}")" >> "${_input}"
                _need=1
            fi
        fi
        if [ "${_need}" = 0 ]; then
            rm -f "${_input}"
            _log "第 ${_attempt} 轮：本组文件全部就绪，无需下载"
            break
        fi
        # —— aria2 多连接（max-tries=10 抗 CDN 抖动；summary-interval=30 止住进度刷屏）。
        #    失败不致命：交给 curl 续传补齐，再不行还有外层重试轮 —— 此前的组合拳是
        #    "aria2 一次定生死 + 补齐分支调用未定义函数"，任何抖动都直接 BuildFailed。
        if command -v aria2c >/dev/null 2>&1; then
            _log "aria2 多连接下载模型（第 ${_attempt}/3 轮）"
            aria2c -i "${_input}" \
                --max-concurrent-downloads=8 --split=16 --max-connection-per-server=16 \
                --min-split-size=1M --file-allocation=none --continue=true \
                --allow-overwrite=true --auto-file-renaming=false \
                --connect-timeout=20 --timeout=120 \
                --retry-wait=3 --max-tries=10 --summary-interval=30 \
                --console-log-level=warn \
                || _log "⚠️ aria2 返回非零（第 ${_attempt} 轮），尝试 curl 续传补齐"
        else
            _log "未找到 aria2c，使用 curl 顺序下载（第 ${_attempt}/3 轮）"
            for _item in ${_sel[@]+"${_sel[@]}"}; do
                IFS='|' read -r _rel _url _size <<< "${_item}"
                _file="${_dest}/${_rel}"
                _got="$(wc -c < "${_file}" 2>/dev/null | tr -d '[:space:]' || true)"
                [ "${_got}" = "${_size}" ] && continue
                _fetch_curl_one "${_url}" "${_file}" 1200 || true
            done
            if [ "${_want_gguf}" = 1 ]; then
                _file="${_dest}/${_gguf_rel}"
                _got="$(wc -c < "${_file}" 2>/dev/null | tr -d '[:space:]' || true)"
                if [ -z "${_got}" ] || [ "${_got}" -le 100000000 ]; then
                    _fetch_curl_one "${_gguf_url}" "${_file}" || true
                fi
            fi
        fi
        # —— aria2 部分失败/断连时，curl -C - 从已落盘前缀续传补齐；失败只记日志，
        #    由本轮校验 + 外层重试接手（绝不在这里 _die）——
        for _item in ${_sel[@]+"${_sel[@]}"}; do
            IFS='|' read -r _rel _url _size <<< "${_item}"
            _file="${_dest}/${_rel}"
            _got="$(wc -c < "${_file}" 2>/dev/null | tr -d '[:space:]' || true)"
            if [ "${_got}" != "${_size}" ]; then
                _log "curl 补齐: ${_rel}"
                _fetch_curl_one "${_url}" "${_file}" || true
            fi
        done
        if [ "${_want_gguf}" = 1 ]; then
            _file="${_dest}/${_gguf_rel}"
            _got="$(wc -c < "${_file}" 2>/dev/null | tr -d '[:space:]' || true)"
            if [ -z "${_got}" ] || [ "${_got}" -le 100000000 ]; then
                _log "curl 补齐: ${_gguf_rel}"
                _fetch_curl_one "${_gguf_url}" "${_file}" || true
            fi
        fi
        rm -f "${_input}"
        # —— 本轮校验：全过提前收工；否则带着缺口进下一轮 ——
        _ok=1
        for _item in ${_sel[@]+"${_sel[@]}"}; do
            IFS='|' read -r _rel _url _size <<< "${_item}"
            _got="$(wc -c < "${_dest}/${_rel}" 2>/dev/null | tr -d '[:space:]' || true)"
            [ "${_got}" = "${_size}" ] || { _ok=0; _log "⚠️ 第 ${_attempt} 轮未完成: ${_rel}（期望 ${_size}，实际 ${_got:-缺失}）"; }
        done
        if [ "${_want_gguf}" = 1 ]; then
            _got="$(wc -c < "${_dest}/${_gguf_rel}" 2>/dev/null | tr -d '[:space:]' || true)"
            { [ -n "${_got}" ] && [ "${_got}" -gt 100000000 ]; } || { _ok=0; _log "⚠️ 第 ${_attempt} 轮未完成: ${_gguf_rel}"; }
        fi
        [ "${_ok}" = 1 ] && break
        [ "${_attempt}" = 3 ] || sleep 5
    done
    # —— 终局校验（fail-closed：缺一个字节都不放行，宁可使构建失败也不入库残缺产物）——
    for _item in ${_sel[@]+"${_sel[@]}"}; do
        IFS='|' read -r _rel _url _size <<< "${_item}"
        _got="$(wc -c < "${_dest}/${_rel}" 2>/dev/null | tr -d '[:space:]' || true)"
        [ "${_got}" = "${_size}" ] || _die "模型文件大小不符: ${_rel}（期望 ${_size}，实际 ${_got:-缺失}）"
    done
    if [ "${_want_gguf}" = 1 ]; then
        _got="$(wc -c < "${_dest}/${_gguf_rel}" 2>/dev/null | tr -d '[:space:]' || true)"
        { [ -n "${_got}" ] && [ "${_got}" -gt 100000000 ]; } || _die "GGUF 文件缺失或过小: ${_gguf_rel}"
    fi
    # 成功下载时 aria2 会自删控制文件；这里兜底清残留，绝不混进最终镜像
    find "${_dest}" -name "*.aria2" -delete 2>/dev/null || true
    _log "模型下载完成: ${_dest}（组: ${_groups}）"
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
