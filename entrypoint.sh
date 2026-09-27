#!/usr/bin/env bash
# 单容器编排：Qdrant → Ollama → 嵌入/重排权重按需下载 → 生成模型按需拉取 → Streamlit（前台）
# 端口与地址全部走 CLI 参数，不改仓库内任何配置文件（CLI 优先级高于 .streamlit/config.toml）。
set -u

cd /app || exit 1

_log() { printf '[entrypoint] %s\n' "$*"; }

# ── 0. 运行期目录与下载源 ────────────────────────────────────────────────────
# Ollama 权重落在 /app 下（不依赖 HOME，非 root 也可写）；平台若给持久卷可挂到这里。
export OLLAMA_MODELS="${OLLAMA_MODELS:-/app/ollama_models}"
# 嵌入/重排权重的备源 HF 镜像；主源用魔搭自己的 hub（创空间内必达）。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "${OLLAMA_MODELS}" /app/models

if [ ! -f .env ]; then
    _log "⚠️ 缺少 .env（git 推送前没 git add -f .env？）：将退到代码默认配置"
fi

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
    if ! kill -0 "${_qdrant_pid}" 2>/dev/null; then
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
    if ! kill -0 "${_ollama_pid}" 2>/dev/null; then
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
_download_model() {
    _name="$1"; _repo="$2"; _weight="$3"; shift 3
    _dir="/app/models/${_name}"
    if [ -s "${_dir}/${_weight}" ]; then
        _log "权重已就绪: ${_name}"
        return 0
    fi
    if [ "${RAG_SKIP_MODEL_DOWNLOAD:-0}" = 1 ]; then
        _log "RAG_SKIP_MODEL_DOWNLOAD=1，跳过 ${_name}"
        return 0
    fi
    mkdir -p "${_dir}"
    _log "下载权重 ${_name}（主源 ModelScope:${_repo}）"
    _ok=1
    for _f in "$@"; do
        mkdir -p "${_dir}/$(dirname "${_f}")"
        if ! curl -fsSL --retry 3 --connect-timeout 20 --max-time 3600 \
                "https://modelscope.cn/models/${_repo}/resolve/master/${_f}" \
                -o "${_dir}/${_f}"; then
            _ok=0
            break
        fi
    done
    if [ "${_ok}" != 1 ] && command -v huggingface-cli >/dev/null 2>&1; then
        _log "⚠️ ModelScope 下载不完整，改用 HF 镜像重试: ${_name}"
        if huggingface-cli download "${_repo}" --local-dir "${_dir}" >/dev/null 2>&1; then
            _ok=1
        fi
    fi
    if [ -s "${_dir}/${_weight}" ]; then
        _log "权重下载完成: ${_name}"
    else
        _log "❌ 权重下载失败: ${_name}（嵌入/重排不可用，请检查外网/磁盘）"
    fi
}

_download_models() {
    _download_model bge-base-zh-v1.5 BAAI/bge-base-zh-v1.5 pytorch_model.bin \
        config.json pytorch_model.bin tokenizer.json tokenizer_config.json vocab.txt \
        special_tokens_map.json sentence_bert_config.json config_sentence_transformers.json \
        modules.json 1_Pooling/config.json
    _download_model bge-reranker-base BAAI/bge-reranker-base model.safetensors \
        config.json model.safetensors tokenizer.json tokenizer_config.json \
        sentencepiece.bpe.model special_tokens_map.json
    _log "嵌入/重排权重检查结束"
}
_download_models &

# ── 4. 生成模型：缺失则后台拉取，不阻塞对外服务（首启多等几分钟，期间问答会失败）──
#     RAG_SKIP_MODEL_PULL=1 可跳过（本地测试挂载宿主机模型时用）。
_model="$(sed -n 's/^MODEL=//p' .env 2>/dev/null | head -n1 | tr -d '[:space:]')"
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

# ── 5. Streamlit：前台运行（容器主进程），端口按创空间要求固定 7860 ────────────
_log "Streamlit 启动: http://0.0.0.0:${RAG_HTTP_PORT:-7860}"
exec python -m streamlit run chatbot.py \
    --server.address 0.0.0.0 \
    --server.port "${RAG_HTTP_PORT:-7860}" \
    --server.headless true
