#!/usr/bin/env bash
# 单容器编排：Qdrant → Ollama → 权重按需下载 → 检索模型服务(infinity，可选) → 生成模型按需拉取 → Streamlit（前台）
# 端口与地址全部走 CLI 参数，不改仓库内任何配置文件（CLI 优先级高于 .streamlit/config.toml）。
set -u

cd /app || exit 1

_log() { printf '[entrypoint] %s\n' "$*"; }

# 模型清单与拉取的唯一实现（构建期烘焙 / 运行期补下共用），见 tools/fetch_models.sh。
# 缺失时只告警不中断：镜像里必然存在，本地裸跑不会走到这里（第 6 行已 cd /app）。
. /app/tools/fetch_models.sh || _log "⚠️ 缺少 /app/tools/fetch_models.sh，模型按需下载不可用"

# ── 0.1 进程存活判据（唯一一份）─────────────────────────────────────────────
# `kill -0` 对**僵尸**返回 0：exec 之后 Streamlit 是 PID 1、没人 wait，qdrant/ollama
# 运行期崩溃只会变成僵尸，于是 `! kill -0` 恒为假 —— 存活哨兵一次都不会喊（判据恒假，
# 正是它声称要捕获的静默故障）。故必须显式排除 Z 状态。ps 不可用时退回 kill -0
# —— 这一支必须 return 0：ps 缺失时按"判死"处理会让下面的启动检查把正常进程杀掉。
_pid_alive() {
    local _pid="$1" _st
    kill -0 "${_pid}" 2>/dev/null || return 1
    command -v ps >/dev/null 2>&1 || return 0
    _st="$(ps -o stat= -p "${_pid}" 2>/dev/null | head -n1 | tr -d '[:space:]')"
    [ -n "${_st}" ] || return 1
    case "${_st}" in
        Z*) return 1 ;;
    esac
    return 0
}

# ── 0.1b 等待后台 HTTP 服务就绪（qdrant / ollama 共用一份）──────────────────
# 两段启动块完全同构：轮询 healthz → 进程死则 dump 日志退出 → 60 秒超时同样处理。
# 旧写法逐行复制，改一处漏一处（例如只给其中一段加存活判据）。name/logfile 参数
# 让各自的日志文案与日志文件保持不变。
_wait_http_ready() {
    local _name="$1" _url="$2" _pid="$3" _logf="$4" _ready=0 _i
    for _i in $(seq 1 60); do
        if curl -fsS "${_url}" >/dev/null 2>&1; then _ready=1; break; fi
        if ! _pid_alive "${_pid}"; then
            _log "❌ ${_name} 异常退出，最后 50 行日志："
            tail -n 50 "${_logf}"
            exit 1
        fi
        sleep 1
    done
    if [ "${_ready}" != 1 ]; then
        _log "❌ ${_name} 60 秒未就绪，最后 50 行日志："
        tail -n 50 "${_logf}"
        exit 1
    fi
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
        # 引号值要按**首个闭合引号**截断，不能只看首尾字符：`KEY="v" # 注释` 的首尾
        # 不是一对引号，旧写法会漏掉引号剥离、把引号带进变量（python-dotenv 不会）。
        case "${_v}" in
            \"*) _v="${_v#\"}"; _v="${_v%%\"*}" ;;
            \'*) _v="${_v#\'}"; _v="${_v%%\'*}" ;;
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
_wait_http_ready qdrant http://127.0.0.1:6333/healthz "${_qdrant_pid}" /tmp/qdrant.log
_log "qdrant 就绪（http://127.0.0.1:6333）"

# ── 2. Ollama ──────────────────────────────────────────────────────────────
# 生成模型常驻：Ollama 默认 5 分钟卸载，闲置后再问一次就要重新加载 3.4GB（首字多等数秒）。
# 这是**服务端**变量；单例模式（RAG_EXCLUSIVE_MODELS=1）下客户端每次请求都带
# keep_alive=RAG_GEN_KEEP_ALIVE（=0），**请求级字段覆盖这里**，故本默认值只在
# 非单例/其它客户端访问时生效。容器内 ollama serve 由本脚本拉起，故在此显式导出默认值
# —— 若部署方在 .env 设了 OLLAMA_KEEP_ALIVE，_load_dotenv 已导出，${VAR:-...} 会沿用。
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
# 生成侧内存三件套：KV 量化 / flash attention / 槽位并发。**镜像 ENV 已设默认**
# （见 Dockerfile），此处只做运行日志留痕——OOM 现场需要能一眼看出生效值是不是
# q8_0/1/1；若部署方用平台变量覆盖，日志也会显示实际值（不覆盖则打印镜像默认）。
_log "生成侧内存参数：KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE:-<未设=f16>} FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION:-<未设>} NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-<未设>}"
ollama serve >/tmp/ollama.log 2>&1 &
_ollama_pid=$!
_wait_http_ready ollama http://127.0.0.1:11434/api/version "${_ollama_pid}" /tmp/ollama.log
_log "ollama 就绪（http://127.0.0.1:11434, models=${OLLAMA_MODELS}）"

# ── 3. 嵌入/重排权重：清单与实现见 tools/fetch_models.sh（构建期烘焙 / 运行期补下共用）──
#     构建期已烘焙进镜像时，这里只会打"已就绪"、不产生任何下载。RAG_SKIP_MODEL_DOWNLOAD=1
#     可跳过。就绪判据=清单文件全部非空；下到暂存目录齐了再整目录改名就位，避免应用读到
#     半截权重（详见 _download_model）。后台运行，不占住 7860。
_download_models &

# ── 3.5 检索模型服务（infinity）：嵌入 + 重排同一个进程、共享一份 torch ───────────
# RAG_EMBED_BACKEND / RAG_RERANK_BACKEND 任一为 http 时才起。业务进程（Streamlit）因此
# 不再持有检索模型；查询侧按 OpenAI /v1/embeddings 与 Cohere 风格 /v1/rerank 调它。
# **必须在权重就绪之后**（infinity 启动时找不到模型目录会直接退出），故先用子 shell 轮询
# 目录、再 exec 成 infinity —— exec 保住 pid，4.5 的存活哨兵才盯得住真进程。
# 注意：单例（RAG_EXCLUSIVE_MODELS=1）下 infinity 常驻它的两个模型且**没有卸载接口**，
# 互斥只能从另一侧保证 —— 生成模型答完即卸（RAG_GEN_KEEP_ALIVE=0）。
_infinity_pid=""
if [ "${RAG_EMBED_BACKEND:-torch}" = "http" ] || [ "${RAG_RERANK_BACKEND:-torch}" = "http" ]; then
    _embed_dir="/app/models/$(basename "${RAG_EMBED_MODEL_PATH:-./models/bge-m3}")"
    _rerank_dir="/app/models/$(basename "${RAG_RERANK_MODEL_PATH:-./models/bge-reranker-base}")"
    _embed_served="${RAG_EMBED_MODEL_NAME:-$(basename "${_embed_dir}")}"
    _rerank_served="${RAG_RERANK_MODEL_NAME:-$(basename "${_rerank_dir}")}"
    # 端口从 RAG_EMBED_URL 派生（单一来源），解析不出来才退 8001
    _inf_port="$(printf '%s' "${RAG_EMBED_URL:-}" | sed -n 's|^[a-z]*://[^:/]*:\([0-9][0-9]*\).*|\1|p')"
    _inf_port="${_inf_port:-8001}"
    (
        _i=0
        while [ "${_i}" -lt 300 ]; do
            [ -d "${_embed_dir}" ] && [ -d "${_rerank_dir}" ] && break
            _i=$((_i + 1))
            sleep 2
        done
        _log "检索模型服务启动中：embed=${_embed_served} rerank=${_rerank_served} port=${_inf_port}（日志 /tmp/infinity.log）"
        exec infinity_emb v2 \
            --model-id "${_embed_dir}" --served-model-name "${_embed_served}" \
            --model-id "${_rerank_dir}" --served-model-name "${_rerank_served}" \
            --engine torch --device cpu --no-bettertransformer \
            --url-prefix /v1 --host 127.0.0.1 --port "${_inf_port}" \
            >/tmp/infinity.log 2>&1
    ) &
    _infinity_pid=$!
fi

# ── 3.5 检索模型服务（infinity）：嵌入 + 重排同一个进程、共享一份 torch ───────────
# RAG_EMBED_BACKEND / RAG_RERANK_BACKEND 任一为 http 时才起。业务进程（Streamlit）因此
# 不再持有检索模型；查询侧按 OpenAI /v1/embeddings 与 Cohere 风格 /v1/rerank 调它。
# **必须在权重就绪之后**（infinity 启动时找不到模型目录会直接退出），故先用子 shell 轮询
# 目录、再 exec 成 infinity —— exec 保住 pid，4.5 的存活哨兵才盯得住真进程。
# 注意：单例（RAG_EXCLUSIVE_MODELS=1）下 infinity 常驻它的两个模型且**没有卸载接口**，
# 互斥只能从另一侧保证 —— 生成模型答完即卸（RAG_GEN_KEEP_ALIVE=0）。
_infinity_pid=""
if [ "${RAG_EMBED_BACKEND:-torch}" = "http" ] || [ "${RAG_RERANK_BACKEND:-torch}" = "http" ]; then
    _embed_dir="/app/models/$(basename "${RAG_EMBED_MODEL_PATH:-./models/bge-m3}")"
    _rerank_dir="/app/models/$(basename "${RAG_RERANK_MODEL_PATH:-./models/bge-reranker-base}")"
    _embed_served="${RAG_EMBED_MODEL_NAME:-$(basename "${_embed_dir}")}"
    _rerank_served="${RAG_RERANK_MODEL_NAME:-$(basename "${_rerank_dir}")}"
    # 端口从 RAG_EMBED_URL 派生（单一来源），解析不出来才退 8001
    _inf_port="$(printf '%s' "${RAG_EMBED_URL:-}" | sed -n 's|^[a-z]*://[^:/]*:\([0-9][0-9]*\).*|\1|p')"
    _inf_port="${_inf_port:-8001}"
    (
        _i=0
        while [ "${_i}" -lt 300 ]; do
            [ -d "${_embed_dir}" ] && [ -d "${_rerank_dir}" ] && break
            _i=$((_i + 1))
            sleep 2
        done
        _log "检索模型服务启动中：embed=${_embed_served} rerank=${_rerank_served} port=${_inf_port}（日志 /tmp/infinity.log）"
        exec infinity_emb v2 \
            --model-id "${_embed_dir}" --served-model-name "${_embed_served}" \
            --model-id "${_rerank_dir}" --served-model-name "${_rerank_served}" \
            --engine torch --device cpu --no-bettertransformer \
            --url-prefix /v1 --host 127.0.0.1 --port "${_inf_port}" \
            >/tmp/infinity.log 2>&1
    ) &
    _infinity_pid=$!
fi

# ── 4. 生成模型：缺失则后台拉取，不阻塞对外服务（首启多等几分钟，期间问答会失败）──
#     实现与"已就绪"判据见 tools/fetch_models.sh 的 _pull_ollama_model（构建期已烘焙时
#     这里只判就绪、不下载）。RAG_SKIP_MODEL_PULL=1 可跳过。
#     MODEL 的取法已统一进脚本开头的 _load_dotenv：进程环境优先于 .env，且 `export`
#     前缀、`=` 两侧空白、单双引号都认（与 python-dotenv 同语义）。
_model="${MODEL:-}"
if [ "${RAG_SKIP_MODEL_PULL:-0}" = 1 ]; then
    _log "RAG_SKIP_MODEL_PULL=1，跳过模型检查"
elif [ -z "${_model}" ]; then
    _log "⚠️ .env 里没有 MODEL，跳过模型拉取（生成将不可用）"
else
    _pull_ollama_model "${_model}" &
fi

# ── 4.5 存活哨兵 ────────────────────────────────────────────────────────────
# exec 让 Streamlit 当 PID 1，容器里没人 wait：qdrant/ollama 运行期崩了只会变成僵尸，
# 既不重启也不告警 —— 上面两处存活检查只在启动期跑一次。表现为"页面照常 200、检索或
# 生成全失败"，而日志里一条线索都没有（失效原型：状态失步 + 静默）。
# 刻意只告警不自动重启：Qdrant 崩溃循环 + 反复打开损坏的 WAL 比直接停机更糟，重启策略
# 该交给编排层（restart: unless-stopped）或人。每个进程只喊一次，避免刷屏。
(
    _said_q=0; _said_o=0; _said_m=0
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
        if [ -n "${_infinity_pid}" ] && [ "${_said_m}" = 0 ] && ! _pid_alive "${_infinity_pid}"; then
            _said_m=1
            _log "🔴 检索模型服务（infinity）已死（pid=${_infinity_pid}）→ 嵌入/重排全部失败、检索必挂；详见 /tmp/infinity.log"
        fi
    done
) &

# ── 4.6 冷启动自检：生成链路（默认）/ 完整检索+生成（可选）──────────────────────
# 8G 档 llama-server 被 cgroup OOM 杀时（`signal: killed`），业务侧只表现为一次 500，
# 运行日志里既无 KV 规模也无峰值内存；而容器外又驱动不了 Streamlit（脚本只在浏览器
# websocket 会话建立后才跑，curl 只拿静态壳）。故把验收前移到启动期，把结果与内存
# 现场直接写进运行日志（前缀 [自检]）。
# **必须后台 + 延迟**：早期版本在 exec Streamlit 之前同步跑，自检占住 7860 端口 85~240
# 秒，平台健康探针等不到监听 → 容器反复重启（2026-10-16 实测 2 分钟内 6 次冷启动）。
# 默认只跑「生成探针」（不导入 torch，进程几十 MB，与真实查询重叠也不会撑爆 8G）；
# 需要验收检索侧时再开 RAG_BOOT_SELFTEST_FULL=1。RAG_BOOT_SELFTEST=0 可整体关掉。
if [ "${RAG_BOOT_SELFTEST:-1}" != 0 ]; then
    if [ -f /app/models/bge-m3/config.json ] || [ -f /app/models/bge-reranker-v2-m3/config.json ]; then
        (
            sleep "${RAG_BOOT_SELFTEST_DELAY:-45}"
            _log "启动自检开始（前缀 [自检]；服务已就绪，自检不阻塞端口）"
            timeout 900 python /app/tools/boot_selftest.py || \
                _log "⚠️ 启动自检超时/异常退出（非致命，服务照常提供）"
        ) &
    else
        _log "启动自检跳过：检索权重尚未就位（运行期下载中或未挂载）"
    fi
fi

# ── 5. Streamlit：前台运行（容器主进程），端口按创空间要求固定 7860 ────────────
# 0.0.0.0 是**容器语境的正确值**（容器内直连 127.0.0.1 拿不到宿主端口映射）；不要照搬
# 到裸机 —— 裸机上 Streamlit 默认绑 localhost，对外暴露需自己显式选 --server.address。
# 让应用侧知道权重仍在后台下载：模型加载时轮询等待，而不是把首查询竞态当错误抛出。
# RAG_SKIP_MODEL_DOWNLOAD=1（本地挂载模型自测）时下载根本没跑，别让应用去空等。
[ "${RAG_SKIP_MODEL_DOWNLOAD:-0}" = 1 ] || export RAG_WEIGHTS_BACKGROUND=1
_log "Streamlit 启动: http://0.0.0.0:${RAG_HTTP_PORT:-7860}"
exec python -m streamlit run chatbot.py \
    --server.address 0.0.0.0 \
    --server.port "${RAG_HTTP_PORT:-7860}" \
    --server.headless true
