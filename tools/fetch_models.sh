#!/usr/bin/env bash
# 模型清单与拉取的**唯一实现**，供两处复用（AGENTS：同一知识不得多处实现）：
#   · 构建期（Dockerfile，ARG BAKE_MODELS/BAKE_OLLAMA）：把权重/生成模型写进镜像层。
#     容器可写层只读、不持久；模型放运行期下载=每次重启重下数 GB、且"构建完"并不"能用"。
#     烘焙进镜像层后，运行期清单判据直接判"已就绪"，启动即用。
#   · 运行期（entrypoint.sh）：清单缺失时按需补下（挂载模型 / 半成品 / 未烘焙的旧镜像）。
# 本文件**只定义函数常量、不做任何下载**，可安全 source。
set -u

# _log：调用方已定义则沿用（entrypoint 写控制台+文件），否则退到 stdout（构建日志）。
if ! declare -F _log >/dev/null 2>&1; then
    _log() { printf '[fetch-models] %s\n' "$*"; }
fi

# 嵌入/重排模型存放根目录（构建与运行都在 /app 下）。
: "${MODELS_DIR:=/app/models}"

# 从 .env 取一个键（不 source，避免执行任意内容；与 entrypoint 的解析同风格）。
_env_get() {
    local _key="$1" _file="${2:-.env}" _line _v
    [ -f "${_file}" ] || return 0
    _line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${_key}=" "${_file}" | head -n1)" || return 0
    [ -n "${_line}" ] || return 0
    _v="${_line#*=}"
    _v="${_v%$'\r'}"
    _v="$(printf '%s' "${_v}" | sed -e 's/^[[:space:]]*//')"
    case "${_v}" in
        \"*) _v="${_v#\"}"; _v="${_v%%\"*}" ;;
        \'*) _v="${_v#\'}"; _v="${_v%%\'*}" ;;
        *) _v="$(printf '%s' "${_v}" | sed -e 's/[[:space:]]\{1,\}#.*$//' -e 's/[[:space:]]*$//')" ;;
    esac
    printf '%s' "${_v}"
}

# _download_model <name> <repo> <checkpoint> <file...> ─────────────────────────
# 就绪判据 = 清单文件全部非空。只看 checkpoint 会把"权重在、tokenizer.json 缺"的半成品
# 判成就绪，此后每次都被短路、永不重下，from_pretrained 永久失败。
# 全部下到暂存目录、清单齐了再把整目录改名就位：加载侧只看"目录非空"，而本函数与
# Streamlit 启动并发，直接写最终目录会让应用读到写了一半的 checkpoint；失败分支也绝不
# 删既有目录（可能有宿主挂载的完整权重）。清单内文件级并发，失败项统一在暂存校验暴露。
_download_model() {
    # 全部变量 local：两个模型**并发**调用本函数，函数级全局会互相踩。
    local _name="$1" _repo="$2" _weight="$3"; shift 3
    local _dir="${MODELS_DIR}/${_name}"
    local _f _ok=1 _miss="" _tmp
    # 不在首个缺失处 break：要把缺的全列出来，否则失败日志只说"失败"、不给可执行信息。
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
    _log "下载权重 ${_name}（checkpoint ${_weight}，主源 ModelScope:${_repo}，清单 $# 个文件，文件级并发）"
    local -a _pids=()
    local _i
    for _f in "$@"; do
        mkdir -p "${_tmp}/$(dirname "${_f}")"
        curl -fsSL --retry 3 --connect-timeout 20 --max-time 3600 \
            "https://modelscope.cn/models/${_repo}/resolve/master/${_f}" \
            -o "${_tmp}/${_f}" &
        _pids+=("$!")
    done
    _ok=1
    for _i in "${!_pids[@]}"; do
        if ! wait "${_pids[$_i]}"; then _ok=0; fi
    done
    if [ "${_ok}" != 1 ] && command -v huggingface-cli >/dev/null 2>&1; then
        _log "⚠️ ModelScope 下载不完整，改用 HF 镜像重试: ${_name}"
        # 只按清单文件下（reranker 仓库里还有 pytorch_model.bin/onnx 冗余，全量会多拉 2.1G）
        if huggingface-cli download "${_repo}" --local-dir "${_tmp}" "$@" >/dev/null 2>&1; then
            _ok=1
        fi
    fi
    if [ "${_ok}" = 1 ]; then
        _miss=""
        for _f in "$@"; do
            if [ ! -s "${_tmp}/${_f}" ]; then _ok=0; _miss="${_miss} ${_f}(暂存)"; fi
        done
    fi
    if [ "${_ok}" = 1 ]; then
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
        rm -rf "${_tmp}"
        _log "❌ 权重下载失败: ${_name}（缺:${_miss:-?}；已清理暂存，保留既有 ${_dir} 不动，重启容器重试；嵌入/重排不可用）"
        return 1
    fi
}

# _model_spec <模型目录 basename> → "repo|checkpoint|文件…" ────────────────────
# 清单**必须与宿主模型目录逐文件一致**：_model_content_fingerprint 按 (relpath, size,
# 头 4096 字节 sha256) 全量哈希，少一个文件（哪怕运行时用不到，如 bge-m3 的
# colbert_linear.pt）都会让容器算出的切分指纹与索引记录不符，_check_chunk_freshness
# 直接 fail-closed —— 就是 2026-10-10 那类事故的复发路径。
_model_spec() {
    case "$1" in
        bge-base-zh-v1.5)
            printf '%s' "BAAI/bge-base-zh-v1.5|pytorch_model.bin|config.json pytorch_model.bin tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json sentence_bert_config.json config_sentence_transformers.json modules.json 1_Pooling/config.json" ;;
        bge-m3)
            printf '%s' "BAAI/bge-m3|pytorch_model.bin|config.json pytorch_model.bin tokenizer.json tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model sentence_bert_config.json config_sentence_transformers.json modules.json colbert_linear.pt sparse_linear.pt" ;;
        bge-reranker-base)
            printf '%s' "BAAI/bge-reranker-base|model.safetensors|config.json model.safetensors tokenizer.json tokenizer_config.json sentencepiece.bpe.model special_tokens_map.json" ;;
        bge-reranker-v2-m3)
            printf '%s' "BAAI/bge-reranker-v2-m3|model.safetensors|config.json model.safetensors tokenizer.json tokenizer_config.json sentencepiece.bpe.model special_tokens_map.json" ;;
        *)
            return 1 ;;
    esac
}

# _spawn_model_download <name> <pid 变量名>：按清单表后台起一个下载并回填 pid。
# 未登记的模型名**直接判失败**，不静默跳过：那会让应用加载 A、容器里却只有 B。
_spawn_model_download() {
    local _name="$1" _target="$2" _spec _repo _ckpt _files
    printf -v "${_target}" '%s' ""
    [ -n "${_name}" ] || { _log "❌ .env 未指定检索模型路径（RAG_EMBED_MODEL_PATH / RAG_RERANK_MODEL_PATH）"; return 1; }
    if ! _spec="$(_model_spec "${_name}")"; then
        _log "❌ 未知检索模型清单: ${_name}（先在 tools/fetch_models.sh 的 _model_spec 里登记）"
        return 1
    fi
    _repo="${_spec%%|*}"; _spec="${_spec#*|}"
    _ckpt="${_spec%%|*}"; _files="${_spec#*|}"
    # shellcheck disable=SC2086  # _files 是空格分隔的清单，需按词展开
    _download_model "${_name}" "${_repo}" "${_ckpt}" ${_files} &
    printf -v "${_target}" '%s' "$!"
}

# _download_models ───────────────────────────────────────────────────────────
# 下载**当前 .env 指定的**两个检索模型（按 RAG_EMBED_MODEL_PATH / RAG_RERANK_MODEL_PATH
# 的 basename 查 _model_spec）。旧写法把模型名写死在函数里：.env 换成 bge-m3 之后容器
# 仍只下 bge-base，应用加载 m3、容器里却只有 base —— 又一处"配置不生效"的静默原型。
# 返回 0/1，并写 /tmp/model-download.status 供运行期状态通道读取。
_download_models() {
    local _dl_fail=0 _p1="" _p2=""
    local _embed_name _rerank_name
    _embed_name="$(basename "${RAG_EMBED_MODEL_PATH:-$(_env_get RAG_EMBED_MODEL_PATH)}")"
    _rerank_name="$(basename "${RAG_RERANK_MODEL_PATH:-$(_env_get RAG_RERANK_MODEL_PATH)}")"
    _spawn_model_download "${_embed_name}" _p1 || _dl_fail=1
    _spawn_model_download "${_rerank_name}" _p2 || _dl_fail=1
    if [ -n "${_p1}" ]; then wait "${_p1}" || _dl_fail=1; fi
    if [ -n "${_p2}" ]; then wait "${_p2}" || _dl_fail=1; fi
    if [ "${_dl_fail}" = 0 ]; then
        printf 'ok\n' > /tmp/model-download.status
        _log "嵌入/重排权重检查结束"
    else
        printf 'failed\n' > /tmp/model-download.status
        _log "❌ 嵌入/重排权重检查结束（有失败项，见上方日志）→ 嵌入或重排不可用，检索会失败；状态 /tmp/model-download.status"
    fi
    # 构建期（Dockerfile 的 set -e）靠这个退出码把"半成品镜像"拦在构建阶段。
    # 运行期是后台调用（&），无人读退出码，状态仍以 /tmp/model-download.status 为准。
    return "${_dl_fail}"
}

# _pull_ollama_model <model> ─────────────────────────────────────────────────
# 确保生成模型在 OLLAMA_MODELS 下就位。构建期无人预启服务，函数会自己拉起 ollama serve
# 并在结束后关掉；运行期 entrypoint 已启动服务，这里只做 list 判据、不重复起第二个。
_pull_ollama_model() {
    local _model="$1" _pid="" _we_started=0 _i
    [ -n "${_model}" ] || { _log "未指定生成模型，跳过"; return 0; }
    : "${OLLAMA_MODELS:=/app/ollama_models}"
    export OLLAMA_MODELS
    if ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -Fxq "${_model}"; then
        _log "生成模型已就绪: ${_model}"
        return 0
    fi
    if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
        _log "构建期拉起 ollama serve（models=${OLLAMA_MODELS}）"
        HOME="${HOME:-/root}" ollama serve >/tmp/ollama-bake.log 2>&1 &
        _pid=$!
        _we_started=1
        for _i in $(seq 1 60); do
            curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
            sleep 1
        done
    fi
    # 源选择：优先 ModelScope 官方 Ollama 集成（modelscope.cn/<owner>/<repo>:<quant>，
    # 走 ModelScope CDN；实测 ~8MB/s，而 registry.ollama.ai 只有 ~1.4MB/s，3.4GB 差约 40 分钟）。
    # 变量可空，空则退回官方 registry（老行为）。
    local _src="${RAG_GEN_MODEL_MODELSCOPE:-$(_env_get RAG_GEN_MODEL_MODELSCOPE)}"
    local _pulled="${_model}"
    [ -n "${_src}" ] && _pulled="${_src}"
    _log "拉取生成模型: ${_pulled}（models=${OLLAMA_MODELS}）→ /tmp/ollama-pull.log"
    if ! ollama pull "${_pulled}" >/tmp/ollama-pull.log 2>&1; then
        _log "❌ 生成模型拉取失败: ${_pulled}，详见 /tmp/ollama-pull.log"
        [ "${_we_started}" = 1 ] && { kill "${_pid}" 2>/dev/null || true; wait "${_pid}" 2>/dev/null || true; }
        return 1
    fi
    if [ "${_pulled}" != "${_model}" ]; then
        # ModelScope 源名带 host/repo，应用只认短名；ollama cp 只新建 manifest、共享同一份 blob，
        # 不复制数据；随后删源名，避免 ollama list 出现两个名字。
        ollama rm "${_model}" >/dev/null 2>&1 || true
        if ! ollama cp "${_pulled}" "${_model}" >>/tmp/ollama-pull.log 2>&1; then
            _log "❌ 生成模型重命名失败: ${_pulled} → ${_model}，详见 /tmp/ollama-pull.log"
            [ "${_we_started}" = 1 ] && { kill "${_pid}" 2>/dev/null || true; wait "${_pid}" 2>/dev/null || true; }
            return 1
        fi
        ollama rm "${_pulled}" >/dev/null 2>&1 || true
    fi
    [ "${_we_started}" = 1 ] && { kill "${_pid}" 2>/dev/null || true; wait "${_pid}" 2>/dev/null || true; }
    _log "生成模型已就绪: ${_model}"
    return 0
}
