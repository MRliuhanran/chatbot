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
# ⚠️ 默认值**不在这里**求值（评审 #8）：本文件在 entrypoint 里是 source 的，而 .env 由
# _load_dotenv 在更晚的时候导出 —— 旧写法在这行做 `: "${MODELS_DIR:=/app/models}"` 会在
# source 的瞬间把变量占住，之后 _load_dotenv 看到"已设"就跳过（旧白名单里也没有它）→
# .env 里写 MODELS_DIR= 被无声忽略。默认值现落在真正用它的函数内（见 _download_model），
# 只有"进程环境与 .env 都没给"时才生效。

# 从 .env 取一个键（不 source，避免执行任意内容；与 entrypoint 的解析同风格）。
_env_get() {
    local _key="$1" _file="${2:-.env}" _line _v
    [ -f "${_file}" ] || return 0
    # 同名重复取**最后一次**（实测 python-dotenv 3.10.0 就是这个语义，tools/dotenv.sh 的
    # _load_dotenv 亦然）：旧写法 head -n1 取第一次，会让构建期与运行期读同一份 .env
    # 得到不同的值（失效原型：多份）。
    _line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${_key}=" "${_file}" | tail -n1)" || return 0
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
    # MODELS_DIR 默认值在这里（而不是 source 时）求值：.env 的值已由 _load_dotenv 导出
    # （白名单里有它），只有谁都没设时才落这个默认值。函数内赋值不会外泄到调用方，
    # 而调用方（构建期各 RUN / 运行期后台子 shell）本来就各读各的。
    : "${MODELS_DIR:=/app/models}"
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
        # 跳过下载 ≠ 已就绪（评审 #6）：旧写法这里 return 0，_download_models 据此把
        # /tmp/model-download.status 写成 ok，而应用侧 _wait_for_weights 只认 failed →
        # 权重明明缺失也要白等到 MODEL_WAIT_SECS（900s）才报错（本文件其它失败分支返回的
        # 都是非 0）。构建期更糟：Dockerfile 的 set -e 会把"一个模型都没烘焙"判成成功。
        # 触发场景：本地挂载模型自测时挂错目录 / .env 与镜像清单不一致。
        _log "❌ ${_name} 权重未就绪（缺:${_miss}），而 RAG_SKIP_MODEL_DOWNLOAD=1 跳过下载 —— 判失败（跳过下载不等于已就绪）"
        return 1
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
        # 安全落盘（评审 #1）：旧写法 `rm -rf 目标 && mv` 在目标是挂载点/半成品时会把既有权重
        # 先删光、再因 rmdir 失败而整体失败，日志却还说"保留既有不动"——数据丢失且被日志误导。
        # 改成"改名备份 -> 换入 -> 删备份"，任一步失败都能回退。
        if [ -n "$(mount 2>/dev/null | grep -F " ${_dir} ")" ]; then
            _ok=0; _miss="${_miss:-?}（目标是挂载点，不能整体替换；请改用挂载点内的子目录）"
            rm -rf "${_tmp}" 2>/dev/null
            _log "❌ 权重落盘失败: ${_name}（目标是挂载点；既有 ${_dir} 未被触碰）"
            return 1
        fi
        rm -rf "${_dir}.old" 2>/dev/null
        if { [ ! -e "${_dir}" ] || mv "${_dir}" "${_dir}.old"; } && mv "${_tmp}" "${_dir}"; then
            rm -rf "${_dir}.old" 2>/dev/null
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

# _download_side <embed|rerank> ──────────────────────────────────────────────
# 只下载 .env 指定的**单侧**检索模型，模型名解析与 _download_models 同源（同上：知识
# 只此一处）。Dockerfile 分层烘焙按侧各起一个 RUN 时用：两侧合成一个 RUN 会产出
# 5.95G 单层，2026-10-16 两次构建实测推镜像都在该层卡死（>75 分钟无一层完成），
# 拆成 ~2.1G/2.1G/2.6G 三层后每层都能在分钟级出现 `Pushed`。
_download_side() {
    local _side="$1" _name _p="" _rc=0
    case "${_side}" in
        embed)  _name="$(basename "${RAG_EMBED_MODEL_PATH:-$(_env_get RAG_EMBED_MODEL_PATH)}")" ;;
        rerank) _name="$(basename "${RAG_RERANK_MODEL_PATH:-$(_env_get RAG_RERANK_MODEL_PATH)}")" ;;
        *) _log "❌ _download_side 只接受 embed|rerank，收到: ${_side}"; return 1 ;;
    esac
    _spawn_model_download "${_name}" _p || return 1
    wait "${_p}" || _rc=1
    return "${_rc}"
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

# _ollama_manifest_meta <model> ──────────────────────────────────────────────
# ModelScope 的 Ollama 集成**依据 GGUF 自动生成** manifest，但只落 template/params/license，
# **不落 renderer/parser**（实测对比两边 config blob：官方 qwen3.5:4b-q4_K_M 带
# "renderer":"qwen3.5"/"parser":"qwen3.5"，ModelScope 版两者皆无）。
# 后果很隐蔽：ollama show 仍报 "thinking" 能力，但 /api/chat 传 think=true 时**不会把思考
# 拆进 message.thinking 字段**——应用侧思维链永远是空的，正文却正常（实测同一份 GGUF，
# 补不补 renderer/parser，thinking 有无判然不同；正文一致）。故从 ModelScope 快源拉下来的
# 模型必须用 Modelfile 补上这两行，才能和官方 registry 那份行为一致。
# 输出为该家族需要的 Modelfile 指令（不含 FROM）；未收录的家族返回 1，调用方退回 ollama cp。
_ollama_manifest_meta() {
    case "${1%%:*}" in
        qwen3.5) printf 'RENDERER qwen3.5\nPARSER qwen3.5\n' ;;
        *) return 1 ;;
    esac
}

# ── 官方 registry 多连接直下（绕开 ollama pull 的单流）─────────────────────────
# 背景：ollama pull 每个 blob 都是**单条 TCP 流**，对官方源（registry.ollama.ai →
#   307 跳 Cloudflare R2）在境外链路常远跑不满带宽。本组按 OCI 分发协议直接取
#   manifest + blobs，对每个 blob 开 N 条 Range 连接**分片并行**拉，拼回、校验 sha256
#   后，按 Ollama 的磁盘布局落盘：
#       <OLLAMA_MODELS>/manifests/<host>/library/<name>/<tag>
#       <OLLAMA_MODELS>/blobs/sha256-<hex>
#   ollama 无需再 pull 即可识别（已用真实模型验证：手动落盘后 ollama list/show 正常）。
# 连接数：RAG_OLLAMA_DL_CONNS（默认 16）；小文件按体积自动降连接数（每连接 ≥1MiB）。
# 失败语义：任一步失败 → 返回非零 → 调用方退回 ollama pull（保底出模型，只是可能慢）。
# ⚠️ manifest/blobs 均为标准 OCI 产物，content-addressed；sha256 校验后再原子 rename 落盘，
#    半截文件绝不就位（沿用 _download_model 的"先暂存再换入"风格）。
_sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# 解析 307 后的最终直链（R2 预签名 URL）。签名只绑 host、24h 内有效，Range 可用。
_curl_final_url() {
    curl -fsSL --retry 3 --retry-all-errors --connect-timeout 20 --max-time 120 \
        -r 0-0 -o /dev/null -w '%{url_effective}' "$1"
}

# 分片并行下载：$1=直链 $2=目标文件 $3=总字节 $4=连接数
_curl_multiseg() {
    local _url="$1" _out="$2" _size="$3" _conns="$4"
    local _eff _seg _i _s _e _p _dir _rc=0
    [ "${_size}" -gt 0 ] || return 1
    _eff=$(( (_size + 1048575) / 1048576 ))          # 每连接至少 ~1MiB
    [ "${_conns}" -lt "${_eff}" ] || _conns="${_eff}"
    [ "${_conns}" -lt 1 ] && _conns=1
    _seg=$(( _size / _conns )); [ "${_seg}" -lt 1 ] && _seg=1
    _dir="$(mktemp -d "${_out}.parts.XXXXXX")" || return 1
    local -a _pids=()
    for ((_i=0; _i<_conns; _i++)); do
        _s=$(( _i * _seg ))
        [ "${_s}" -ge "${_size}" ] && break
        if [ "${_i}" -eq "$(( _conns - 1 ))" ]; then _e=$(( _size - 1 )); else _e=$(( _s + _seg - 1 )); fi
        curl -fsSL --retry 3 --retry-all-errors --connect-timeout 20 --max-time 3600 \
            -r "${_s}-${_e}" "$_url" -o "${_dir}/$(printf '%05d' "${_i}")" &
        _pids+=("$!")
    done
    for _p in "${_pids[@]}"; do wait "${_p}" || _rc=1; done
    if [ "${_rc}" = 0 ]; then
        cat "${_dir}"/* > "${_out}" || _rc=1
    fi
    rm -rf "${_dir}"
    return "${_rc}"
}

# _download_ollama_official <model>：官方源多连接直下并落到 Ollama 布局。
# 只吃无 host 的短名（如 qwen3.5:4b-q4_K_M）；带 '/' 的一律返回 1（交回原路径）。
_download_ollama_official() {
    local _model="$1"
    case "${_model}" in
        ""|*/*) return 1 ;;
    esac
    : "${OLLAMA_MODELS:=/app/ollama_models}"
    local _registry="${RAG_OLLAMA_REGISTRY:-https://registry.ollama.ai}"
    local _host="${_registry#*://}"; _host="${_host%%/*}"
    local _name="${_model%%:*}" _tag="latest"
    case "${_model}" in *:*) _tag="${_model##*:}" ;; esac
    local _conns="${RAG_OLLAMA_DL_CONNS:-16}"
    local _murl="${_registry}/v2/library/${_name}/manifests/${_tag}"
    local _mdir="${OLLAMA_MODELS}/manifests/${_host}/library/${_name}"
    local _mdest="${_mdir}/${_tag}" _mtmp="${_mdir}/.${_tag}.tmp.$$"
    local _list _hex _size _blob _btmp _final _got

    _log "官方多连接：取 manifest ${_murl}（连接数 ${_conns}）"
    mkdir -p "${OLLAMA_MODELS}/blobs" "${_mdir}" || return 1
    if ! curl -fsSL --retry 3 --retry-all-errors --connect-timeout 20 --max-time 120 \
            -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
            "${_murl}" -o "${_mtmp}"; then
        _log "⚠️  官方 manifest 获取失败: ${_murl}"; rm -f "${_mtmp}"; return 1
    fi
    if ! _list="$(python3 - "${_mtmp}" 2>/dev/null <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
blobs = [(m["config"]["digest"], m["config"]["size"])]
blobs += [(l["digest"], l["size"]) for l in m.get("layers", [])]
for d, s in blobs:
    print("%s %s" % (d.split(":", 1)[1], s))
PY
    )"; then
        _log "⚠️  官方 manifest 解析失败"; rm -f "${_mtmp}"; return 1
    fi

    while read -r _hex _size; do
        [ -n "${_hex}" ] || continue
        _blob="${OLLAMA_MODELS}/blobs/sha256-${_hex}"
        if [ -s "${_blob}" ] && [ "$(wc -c < "${_blob}" | tr -d ' ')" = "${_size}" ]; then
            continue
        fi
        if ! _final="$(_curl_final_url "${_registry}/v2/library/${_name}/blobs/sha256:${_hex}")" || [ -z "${_final}" ]; then
            _log "⚠️  解析 blob 直链失败: ${_hex:0:12}…"; rm -f "${_mtmp}"; return 1
        fi
        _btmp="${_blob}.part.$$"
        _log "  ↓ blob ${_hex:0:12}… ${_size}B"
        if ! _curl_multiseg "${_final}" "${_btmp}" "${_size}" "${_conns}"; then
            _log "⚠️  分片下载失败: ${_hex:0:12}…"; rm -f "${_btmp}" "${_mtmp}"; return 1
        fi
        _got="$(_sha256_of "${_btmp}")"
        if [ "${_got}" != "${_hex}" ]; then
            _log "⚠️  sha256 校验失败 ${_hex:0:12}…（实得 ${_got:0:12}…）"
            rm -f "${_btmp}" "${_mtmp}"; return 1
        fi
        if ! mv -f "${_btmp}" "${_blob}"; then rm -f "${_btmp}" "${_mtmp}"; return 1; fi
    done <<< "${_list}"

    if ! mv -f "${_mtmp}" "${_mdest}"; then rm -f "${_mtmp}"; return 1; fi
    _log "✅ 官方多连接落盘完成: ${_model}"
    return 0
}

# _pull_ollama_model <model> ─────────────────────────────────────────────────
# 确保生成模型在 OLLAMA_MODELS 下就位。构建期无人预启服务，函数会自己拉起 ollama serve
# 并在结束后关掉；运行期 entrypoint 已启动服务，这里只做 list 判据、不重复起第二个。
_pull_ollama_model() {
    local _model="$1" _pid="" _we_started=0 _i _meta=""
    # 空模型名的两种语境（评审 #5）—— 返回值就是构建期唯一的闸门（Dockerfile 的 RUN 带
    # set -e，调用形式是 _pull_ollama_model "$(_env_get MODEL)"，BAKE_OLLAMA=1）：
    #   · 构建期：旧写法无条件 return 0 → .env 漏了 MODEL 时"构建成功却一个生成模型都没
    #     烘焙"，与 Dockerfile 注释承诺的"模型缺失即构建失败"直接矛盾（失效原型：静默 +
    #     构建期假通过；镜像发出去后运行期才发现没有生成模型）。
    #   · 运行期（entrypoint，没有 BAKE_OLLAMA 这个 ARG）：MODEL 为空时应用自己会报"生成
    #     不可用"，容器仍应把 Qdrant/Streamlit 服务起起来，故沿用"告警后跳过"。
    if [ -z "${_model}" ]; then
        if [ "${BAKE_OLLAMA:-0}" = 1 ]; then
            _log "❌ 构建期烘焙生成模型（BAKE_OLLAMA=1），但 .env 里没有 MODEL —— 判失败，不产出没有生成模型的镜像"
            return 1
        fi
        _log "未指定生成模型，跳过（生成将不可用：.env 里没有 MODEL）"
        return 0
    fi
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
    # 源选择：
    #   · RAG_GEN_MODEL_MODELSCOPE 非空 → ModelScope 官方 Ollama 集成（境内 CDN 快，
    #     但 manifest 由平台自动生成、缺 renderer/parser，需下面用 Modelfile 补）；
    #   · 为空 → 官方 registry。此时**先试多连接直下**（绕开 ollama pull 单流），
    #     成功即免 pull，失败退回下面的 ollama pull（老行为，保底能出模型）。
    local _src="${RAG_GEN_MODEL_MODELSCOPE:-$(_env_get RAG_GEN_MODEL_MODELSCOPE)}"
    local _pulled="${_model}"
    [ -n "${_src}" ] && _pulled="${_src}"

    if [ -z "${_src}" ]; then
        if _download_ollama_official "${_model}" \
           && ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -Fxq "${_model}"; then
            [ "${_we_started}" = 1 ] && { kill "${_pid}" 2>/dev/null || true; wait "${_pid}" 2>/dev/null || true; }
            _log "生成模型已就绪（官方源多连接）: ${_model}"
            return 0
        fi
        _log "⚠️  官方源多连接不可用/未生效，退回 ollama pull（单流）: ${_model}"
    fi

    _log "拉取生成模型: ${_pulled}（models=${OLLAMA_MODELS}）→ /tmp/ollama-pull.log"
    # 构建期拉取必须扛得住瞬时抖动：一次失败就让整层 RUN 失败、整个镜像构建失败
    # （实测 2026-10-16 连续两次把公开站点打成 BuildFailed，站点下线近一小时）。
    # 3 次尝试 + 递增退避；失败原因打进构建日志（见下）。
    _try=0
    while :; do
        _try=$((_try + 1))
        if ollama pull "${_pulled}" >/tmp/ollama-pull.log 2>&1; then break; fi
        if [ "${_try}" -ge 3 ]; then
            _log "❌ 生成模型拉取失败（已尝试 ${_try} 次）: ${_pulled}"
            # ★ /tmp/ollama-pull.log 在构建容器里，构建一结束就再也看不到：不打出来，
            #   事后只能看到「拉取失败」四个字，连是网络、磁盘还是权限都无从判断（本次事故）。
            _log "---- /tmp/ollama-pull.log 末尾 40 行 ----"
            tail -n 40 /tmp/ollama-pull.log 2>/dev/null || true
            _log "---- 结束 ----"
            [ "${_we_started}" = 1 ] && { kill "${_pid}" 2>/dev/null || true; wait "${_pid}" 2>/dev/null || true; }
            return 1
        fi
        _log "⚠️  第 ${_try} 次拉取失败，$(( _try * 15 ))s 后重试: ${_pulled}"
        sleep $(( _try * 15 ))
    done
    if [ "${_pulled}" != "${_model}" ]; then
        # ModelScope 源名带 host/repo，应用只认短名；下面无论 create 还是 cp，都只新建 manifest、
        # 共享同一份 blob（不复制数据）；随后删源名，避免 ollama list 出现两个名字。
        ollama rm "${_model}" >/dev/null 2>&1 || true
        # ModelScope 自动生成的 manifest 缺 renderer/parser（原因见 _ollama_manifest_meta），
        # 缺了它 think=true 不会产出 message.thinking。已收录的家族用 Modelfile 补上再建成短名；
        # 未收录的家族退回 ollama cp（保底能出正文，但思维链不可用）。
        if _meta="$(_ollama_manifest_meta "${_model}")" && [ -n "${_meta}" ]; then
            printf 'FROM %s\n%s' "${_pulled}" "${_meta}" > /tmp/ollama-gen.Modelfile
            if ! ollama create "${_model}" -f /tmp/ollama-gen.Modelfile >>/tmp/ollama-pull.log 2>&1; then
                _log "❌ 生成模型建模失败（补 renderer/parser）: ${_pulled} → ${_model}，详见 /tmp/ollama-pull.log"
                [ "${_we_started}" = 1 ] && { kill "${_pid}" 2>/dev/null || true; wait "${_pid}" 2>/dev/null || true; }
                return 1
            fi
        else
            _log "⚠️  ${_model} 未收录 renderer/parser 家族，退回 ollama cp（思维链可能不可用）"
            if ! ollama cp "${_pulled}" "${_model}" >>/tmp/ollama-pull.log 2>&1; then
                _log "❌ 生成模型重命名失败: ${_pulled} → ${_model}，详见 /tmp/ollama-pull.log"
                [ "${_we_started}" = 1 ] && { kill "${_pid}" 2>/dev/null || true; wait "${_pid}" 2>/dev/null || true; }
                return 1
            fi
        fi
        ollama rm "${_pulled}" >/dev/null 2>&1 || true
    fi
    [ "${_we_started}" = 1 ] && { kill "${_pid}" 2>/dev/null || true; wait "${_pid}" 2>/dev/null || true; }
    _log "生成模型已就绪: ${_model}"
    return 0
}
