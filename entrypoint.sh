#!/usr/bin/env bash
# 单容器编排：Qdrant → Ollama → 嵌入/重排权重按需下载 → 生成模型按需拉取 → Streamlit（前台）
# 端口与地址全部走 CLI 参数，不改仓库内任何配置文件（CLI 优先级高于 .streamlit/config.toml）。
set -u

cd /app || exit 1

_log() { printf '[entrypoint] %s\n' "$*"; }

# ── 0.1 进程存活判据（唯一一份）─────────────────────────────────────────────
# `kill -0` 对**僵尸**返回 0：exec 之后 Streamlit 是 PID 1、没人 wait，qdrant/ollama
# 运行期崩溃只会变成僵尸，于是 `! kill -0` 恒为假 —— 存活哨兵一次都不会喊（判据恒假，
# 正是它声称要捕获的静默故障）。故必须显式排除 Z 状态。ps 不可用时退回 kill -0。
_pid_alive() {
    local _pid="$1" _st
    kill -0 "${_pid}" 2>/dev/null || return 1
    _st="$(ps -o stat= -p "${_pid}" 2>/dev/null | head -n1 | tr -d '[:space:]')"
    [ -n "${_st}" ] || return 1
    case "${_st}" in
        Z*) return 1 ;;
    esac
    return 0
}

# ── 0.2 从 .env 补齐本脚本要用的变量（与 python-dotenv 同语义）───────────────
# 旧写法只特判 MODEL 一行，于是 RAG_HTTP_PORT / RAG_SKIP_MODEL_PULL / HF_ENDPOINT 等
# 写进 .env 对本脚本**静默无效**（端口仍是 7860）—— 而本项目配置的唯一入口就是 .env。
# 规则：只认白名单前缀的键；进程环境优先（已设即跳过，含设为空串）；引号与 export 前缀
# 都认。解析后的值与应用侧 load_dotenv 看到的完全一致，不会出现"脚本与应用各读一份"。
_load_dotenv() {
    [ -f .env ] || return 0
    local _line _k _v
    while IFS= read -r _line || [ -n "${_line}" ]; do
        _line="${_line%$'\r'}"
        case "${_line}" in
            ''|\#*) continue ;;
        esac
        _line="$(printf '%s' "${_line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        _line="${_line#export }"
        case "${_line}" in
            [A-Za-z_]*=*) ;;
            *) continue ;;
        esac
        _k="${_line%%=*}"
        _k="$(printf '%s' "${_k}" | sed -e 's/[[:space:]]*$//')"
        case "${_k}" in
            [A-Za-z_][A-Za-z0-9_]*) ;;
            *) continue ;;
        esac
        case "${_k}" in
            RAG_*|OLLAMA_*|QDRANT_*|SEMANTIC_*|INDEX_*|MODEL|HF_ENDPOINT|HF_HUB_*|PYTORCH_MPS_*|RAG_SKIP_MODEL_DOWNLOAD) ;;
            *) continue ;;
        esac
        eval "[ -n \"\${${_k}+x}\" ]" && continue
        _v="${_line#*=}"
        _v="$(printf '%s' "${_v}" | sed -e 's/^[[:space:]]*//')"
        case "${_v}" in
            \"*\") _v="${_v#\"}"; _v="${_v%\"}" ;;
            \'*\') _v="${_v#\'}"; _v="${_v%\'}" ;;
            *) _v="$(printf '%s' "${_v}" | sed -e 's/[[:space:]]\{1,\}#.*$//' -e 's/[[:space:]]*$//')" ;;
        esac
        export "${_k}=${_v}"
    done < .env
}
_load_dotenv

# ── 0. 运行期目录与下载源 ────────────────────────────────────────────────────
# Ollama 权重落在 /app 下（不依赖 HOME，非 root 也可写）；平台若给持久卷可挂到这里。
export OLLAMA_MODELS="${OLLAMA_MODELS:-/app/ollama_models}"
# 嵌入/重排权重的备源 HF 镜像；主源用魔搭自己的 hub（创空间内必达）。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# 这是全脚本唯一不 exit 1 的启动前置条件（脚本没有 set -e）：磁盘满/配额超/只读挂载
# 时 Ollama 与权重目录建不出来，后面却照常往下走，最后表现为"容器起来了但一个模型都
# 没有"，且日志里没有任何一条指向根因。
if ! mkdir -p "${OLLAMA_MODELS}" /app/models 2>/tmp/mkdir.log; then
    _log "❌ 无法创建运行期目录（磁盘满 / 配额超 / 只读挂载？），详见 /tmp/mkdir.log："
    cat /tmp/mkdir.log
    exit 1
fi

if [ ! -f .env ]; then
    _log "⚠️ 缺少 .env（git 推送前没 git add -f .env？）：将退到代码默认配置"
fi

# ── 0.5 内存观测（压 8G 边界调参用）：每 30 秒一行打进 run 日志 ───────────────────
# 读数：cg_max=max 表示无 cgroup 限额；anon=匿名内存（OOM 真正判据，file 页可回收
# 不算）；top=RSS 前 4 进程。调 RAG_NUM_CTX 后看 anon 是否逼近 ~7.5G 再继续加码。
# 换算一律走 awk，不用 bash 的 $(( ))：算术展开里拿到空串是**致命**语法错误，会直接
# 杀掉整个观测子 shell。cgroup v2 的 memory.stat 没有 anon 行（未设 memory.swap.high
# 等场景）、或 memory.current 读不到内容时正是空串 —— 于是 [mem] 行整条消失、无任何
# 痕迹，而 chatbot.py 的 _gen_failure_hint 与 .env 都让运维来看这一行判断 OOM。
(
    while :; do
        _m="[mem]"
        if [ -r /proc/meminfo ]; then
            _m="${_m} $(awk '/MemTotal|MemAvailable/{printf "%s %sM ", $1, int($2/1024)}' /proc/meminfo)"
        fi
        if [ -r /sys/fs/cgroup/memory.current ] && [ -r /sys/fs/cgroup/memory.max ]; then
            _cg="$(awk 'NR==1{printf "%d", $1/1048576}' /sys/fs/cgroup/memory.current 2>/dev/null)"
            _anon="$(awk '/^anon /{printf "%d", $2/1048576; exit}' /sys/fs/cgroup/memory.stat 2>/dev/null)"
            _max="$(tr -d '[:space:]' </sys/fs/cgroup/memory.max 2>/dev/null)"
            [ -n "${_cg}" ] && _m="${_m} cg_cur:${_cg}M"
            _m="${_m} cg_max:${_max:-?}"
            [ -n "${_anon}" ] && _m="${_m} anon:${_anon}M"
        fi
        _m="${_m} top:$(ps -eo rss=,comm= --sort=-rss 2>/dev/null | head -4 | awk '{printf "%s(%sM) ", $2, int($1/1024)}')"
        printf '%s\n' "${_m}"
        sleep 30
    done
) &

# ── 1. Qdrant ──────────────────────────────────────────────────────────────
# 与本地 docker-compose 完全同配置：加载同一份 qdrant_config/config.yaml
# （memmap/indexing 阈值、log_level 等与本地一致），storage_path 用环境变量
# 覆盖成镜像内布局（qdrant 优先级：环境变量 > 配置文件）。
export QDRANT__STORAGE__STORAGE_PATH="${QDRANT__STORAGE__STORAGE_PATH:-/app/qdrant_storage}"
/opt/qdrant/qdrant --config-path /app/qdrant_config/config.yaml >/tmp/qdrant.log 2>&1 &
_qdrant_pid=$!
_log "qdrant 启动中（storage=${QDRANT__STORAGE__STORAGE_PATH}）"
_ready=0
for _i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:6333/healthz >/dev/null 2>&1; then _ready=1; break; fi
    if ! _pid_alive "${_qdrant_pid}"; then
        _log "❌ qdrant 异常退出，最后 50 行日志："
        tail -n 50 /tmp/qdrant.log
        exit 1
    fi
    sleep 1
done
if [ "${_ready}" != 1 ]; then
    _log "❌ qdrant 60 秒未就绪，最后 50 行日志："
    tail -n 50 /tmp/qdrant.log
    exit 1
fi
_log "qdrant 就绪（http://127.0.0.1:6333）"

# ── 2. Ollama ──────────────────────────────────────────────────────────────
ollama serve >/tmp/ollama.log 2>&1 &
_ollama_pid=$!
_ready=0
for _i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then _ready=1; break; fi
    if ! _pid_alive "${_ollama_pid}"; then
        _log "❌ ollama 异常退出，最后 50 行日志："
        tail -n 50 /tmp/ollama.log
        exit 1
    fi
    sleep 1
done
if [ "${_ready}" != 1 ]; then
    _log "❌ ollama 60 秒未就绪，最后 50 行日志："
    tail -n 50 /tmp/ollama.log
    exit 1
fi
_log "ollama 就绪（http://127.0.0.1:11434, models=${OLLAMA_MODELS}）"

# ── 3. 嵌入/重排权重：不打进镜像，启动时按需下载（后台，不占住 7860）───────────
#    主源魔搭 hub（创空间内必达），备源 HF 镜像。RAG_SKIP_MODEL_DOWNLOAD=1 可跳过。
#    任一权重文件已存在（如本地测试挂载了宿主 models/）即视为就绪、直接跳过。
# 权重一律下到暂存目录、清单齐了再改名就位。加载侧（chatbot.py 的 _require_model_dir）
# 只看"目录非空"，而本函数与 Streamlit 启动是并发的（后台作业 + exec）：直接写最终目录
# 会让应用在 checkpoint 写到一半时就把模型读进去（半截权重），失败分支再 rm -rf 更会
# 删掉正被 mmap 的目录。暂存 + 改名让"就绪"只剩一个含义：清单文件全部非空。
_download_model() {
    _name="$1"; _repo="$2"; _weight="$3"; shift 3
    _dir="/app/models/${_name}"
    # 就绪判据 = 清单文件全部非空。只看 checkpoint 会把"权重在、tokenizer.json 缺"的
    # 半成品判成就绪，此后每次启动都被这里短路，永不重下，from_pretrained 永久失败。
    # 不在首个缺失处 break：要把缺的全列出来，否则失败日志只说"失败"、不给可执行信息。
    _ok=1
    _miss=""
    for _f in "$@"; do
        if [ ! -s "${_dir}/${_f}" ]; then _ok=0; _miss="${_miss} ${_f}"; fi
    done
    if [ "${_ok}" = 1 ]; then
        _log "权重已就绪: ${_name}（清单 $# 个文件齐备）"
        return 0
    fi
    if [ "${RAG_SKIP_MODEL_DOWNLOAD:-0}" = 1 ]; then
        _log "RAG_SKIP_MODEL_DOWNLOAD=1，跳过 ${_name}（未就绪，缺:${_miss}）"
        return 0
    fi
    _tmp="${_dir}.part"
    rm -rf "${_tmp}"
    mkdir -p "${_tmp}"
    _log "下载权重 ${_name}（checkpoint ${_weight}，主源 ModelScope:${_repo}，清单 $# 个文件）"
    _ok=1
    for _f in "$@"; do
        mkdir -p "${_tmp}/$(dirname "${_f}")"
        if ! curl -fsSL --retry 3 --connect-timeout 20 --max-time 3600 \
                "https://modelscope.cn/models/${_repo}/resolve/master/${_f}" \
                -o "${_tmp}/${_f}"; then
            _ok=0
            break
        fi
    done
    if [ "${_ok}" != 1 ] && command -v huggingface-cli >/dev/null 2>&1; then
        _log "⚠️ ModelScope 下载不完整，改用 HF 镜像重试: ${_name}"
        # 只按清单文件下（reranker 仓库里还有 pytorch_model.bin/onnx 冗余，全量会多拉 2.1G）
        if huggingface-cli download "${_repo}" --local-dir "${_tmp}" "$@" >/dev/null 2>&1; then
            _ok=1
        fi
    fi
    if [ "${_ok}" = 1 ]; then
        for _f in "$@"; do
            if [ ! -s "${_tmp}/${_f}" ]; then _ok=0; _miss="${_miss} ${_f}(暂存)"; fi
        done
    fi
    if [ "${_ok}" = 1 ]; then
        # rm/mv 的退出码必须看：磁盘满/只读挂载（本地跑法把 models/ 挂成 :ro）时，
        # 旧写法照样打"下载完成"，与随后 _require_model_dir 的报错互相矛盾。
        if rm -rf "${_dir}" && mv "${_tmp}" "${_dir}"; then
            _log "权重下载完成: ${_name}"
        else
            _ok=0
            _miss="${_miss:-?}（落盘失败：rm/mv 返回非 0）"
            rm -rf "${_tmp}" 2>/dev/null
            _log "❌ 权重落盘失败: ${_name}（磁盘满/只读挂载？已清理暂存，保留既有 ${_dir} 不动）"
            return 1
        fi
    else
        # 只删暂存，绝不删 ${_dir}：那里可能有宿主挂载进来的完整 checkpoint（4.4G/1.1G，
        # 重下不回来）。残留的半成品由下次启动重新判定并覆盖，代价是这期间查询报
        # "模型目录缺失或为空"——而下面这行已经把缺哪个文件写清楚了。
        rm -rf "${_tmp}"
        _log "❌ 权重下载失败: ${_name}（缺:${_miss:-?}；已清理暂存，保留既有 ${_dir} 不动，重启容器重试；嵌入/重排不可用）"
    fi
}

_download_models() {
    # 本函数跑在后台（`&`），退出码无人可 wait：失败必须**自带可见结果**，
    # 否则"下载失败"与"下载成功"在日志与状态上完全无从区分。
    _dl_fail=0
    _download_model bge-base-zh-v1.5 BAAI/bge-base-zh-v1.5 pytorch_model.bin \
        config.json pytorch_model.bin tokenizer.json tokenizer_config.json vocab.txt \
        special_tokens_map.json sentence_bert_config.json config_sentence_transformers.json \
        modules.json 1_Pooling/config.json || _dl_fail=1
    _download_model bge-reranker-base BAAI/bge-reranker-base model.safetensors \
        config.json model.safetensors tokenizer.json tokenizer_config.json \
        sentencepiece.bpe.model special_tokens_map.json || _dl_fail=1
    if [ "${_dl_fail}" = 0 ]; then
        printf 'ok\n' > /tmp/model-download.status
        _log "嵌入/重排权重检查结束"
    else
        printf 'failed\n' > /tmp/model-download.status
        _log "❌ 嵌入/重排权重检查结束（有失败项，见上方日志）→ 嵌入或重排不可用，health 会点名；状态 /tmp/model-download.status"
    fi
}
_download_models &

# ── 4. 生成模型：缺失则后台拉取，不阻塞对外服务（首启多等几分钟，期间问答会失败）──
#     RAG_SKIP_MODEL_PULL=1 可跳过（本地测试挂载宿主机模型时用）。
#     MODEL 的取法已统一进脚本开头的 _load_dotenv：进程环境优先于 .env，且 `export`
#     前缀、`=` 两侧空白、单双引号都认（与 python-dotenv 同语义）。旧写法在这里单独
#     sed 特判一行 MODEL，只去空白不去引号 —— `MODEL="qwen3.5:4b-q4_K_M"` 会解析出
#     带引号的名字，ollama pull 静默失败；而 .env 里其它变量对本脚本全部静默无效。
_model="${MODEL:-}"
if [ "${RAG_SKIP_MODEL_PULL:-0}" = 1 ]; then
    _log "RAG_SKIP_MODEL_PULL=1，跳过模型检查"
elif [ -z "${_model}" ]; then
    _log "⚠️ .env 里没有 MODEL，跳过模型拉取（生成将不可用）"
elif ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -Fxq "${_model}"; then
    _log "生成模型已就绪: ${_model}"
else
    _log "生成模型缺失，后台拉取中: ${_model}（日志 /tmp/ollama-pull.log）"
    (
        if ollama pull "${_model}" >/tmp/ollama-pull.log 2>&1; then
            _log "模型拉取完成: ${_model}"
        else
            _log "❌ 模型拉取失败，详见 /tmp/ollama-pull.log"
        fi
    ) &
fi

# ── 4.5 存活哨兵 ────────────────────────────────────────────────────────────
# exec 让 Streamlit 当 PID 1，容器里没人 wait：qdrant/ollama 运行期崩了只会变成僵尸，
# 既不重启也不告警 —— 上面两处存活检查只在启动期跑一次。表现为"页面照常 200、检索或
# 生成全失败"，而日志里一条线索都没有（失效原型：状态失步 + 静默）。
# 刻意只告警不自动重启：Qdrant 崩溃循环 + 反复打开损坏的 WAL 比直接停机更糟，重启策略
# 该交给编排层（restart: unless-stopped）或人。每个进程只喊一次，避免刷屏。
(
    _said_q=0; _said_o=0
    while :; do
        sleep 30
        if [ "${_said_q}" = 0 ] && ! _pid_alive "${_qdrant_pid}"; then
            _said_q=1
            _log "🔴 qdrant 已死（pid=${_qdrant_pid}）→ 检索必然全部失败；容器不自动重启，详见 /tmp/qdrant.log"
        fi
        if [ "${_said_o}" = 0 ] && ! _pid_alive "${_ollama_pid}"; then
            _said_o=1
            _log "🔴 ollama 已死（pid=${_ollama_pid}）→ 生成必然失败；容器不自动重启，详见 /tmp/ollama.log"
        fi
    done
) &

# ── 5. Streamlit：前台运行（容器主进程），端口按创空间要求固定 7860 ────────────
_log "Streamlit 启动: http://0.0.0.0:${RAG_HTTP_PORT:-7860}"
exec python -m streamlit run chatbot.py \
    --server.address 0.0.0.0 \
    --server.port "${RAG_HTTP_PORT:-7860}" \
    --server.headless true
