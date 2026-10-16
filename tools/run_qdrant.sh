#!/usr/bin/env bash
# 本地原生运行 Qdrant（不使用容器）。容器只用于魔搭构建/发布（见 Dockerfile）。
#
# 用法:
#   tools/run_qdrant.sh start     # 缺二进制则下载，然后前台后台拉起并等 /healthz
#   tools/run_qdrant.sh stop      # 停掉本脚本拉起的 qdrant
#   tools/run_qdrant.sh status    # 看进程与健康状态
#   tools/run_qdrant.sh download  # 只下载/解压二进制
#
# 版本单一来源 = Dockerfile 的 ARG QDRANT_VERSION：本地原生二进制与镜像必须同版本。
# Qdrant 存储**不支持降级**，而 魔搭镜像会打包同一份 ./storage —— 若本地原生
# 用了更新版本写过数据，镜像里的旧版本就打不开该目录。故这里直接读 Dockerfile，
# 不另写一份版本号（"同一份知识写两处必然分叉"）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DOCKERFILE="$REPO_ROOT/Dockerfile"
_default_version() {
    local _v=""
    if [ -f "$DOCKERFILE" ]; then
        _v="$(sed -n 's/^ARG QDRANT_VERSION=//p' "$DOCKERFILE" | head -n1)"
    fi
    printf '%s' "${_v:-v1.19.1}"
}
QDRANT_VERSION="${QDRANT_VERSION:-$(_default_version)}"

case "$(uname -s)" in
    Darwin) : ;;
    *) echo "本脚本只面向 macOS 本地原生运行；其它平台请用镜像。" >&2; exit 1 ;;
esac
case "$(uname -m)" in
    arm64|aarch64) QARCH="aarch64-apple-darwin" ;;
    x86_64|amd64)  QARCH="x86_64-apple-darwin" ;;
    *) echo "不支持的架构: $(uname -m)" >&2; exit 1 ;;
esac

BIN_DIR="$REPO_ROOT/.bin/qdrant"
QDRANT_BIN="$BIN_DIR/qdrant"
STORAGE="$REPO_ROOT/storage"
CONFIG="$REPO_ROOT/qdrant_config/config.yaml"
LOG="$REPO_ROOT/qdrant.log"
PIDFILE="$REPO_ROOT/.qdrant.pid"
HTTP_PORT=6333
HEALTH_URL="http://127.0.0.1:${HTTP_PORT}/healthz"

_log() { printf '[run_qdrant] %s\n' "$*"; }

_pid_alive() {
    local _pid="$1" _st
    [ -n "$_pid" ] || return 1
    kill -0 "$_pid" 2>/dev/null || return 1
    # ps 拿不到状态（不存在/被沙箱拒绝）时按"存活"处理：kill -0 已通过，不能因为读不到
    # 状态就把活进程误报成已死 —— 那会让 stop 变成"删掉 pid 文件、放走孤儿 qdrant"，
    # 而 start 随后会再拉一个、撞端口。僵尸判据只看 ps 的**成功输出**。
    _st="$(ps -o stat= -p "$_pid" 2>/dev/null | head -n1 | tr -d '[:space:]')" || true
    case "$_st" in Z*) return 1 ;; esac
    return 0
}

_running_pid() {
    [ -f "$PIDFILE" ] || return 1
    local _pid; _pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    _pid_alive "$_pid" || return 1
    printf '%s' "$_pid"
}

_download() {
    if [ -x "$QDRANT_BIN" ]; then
        _log "已存在: $QDRANT_BIN ($("$QDRANT_BIN" --version 2>/dev/null || echo '?'))"
        return 0
    fi
    mkdir -p "$BIN_DIR"
    local _asset="qdrant-${QARCH}.tar.gz"
    local _tmp="$BIN_DIR/$_asset"
    local _urls=(
        "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/${_asset}"
        "https://gh-proxy.com/https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/${_asset}"
    )
    local _u _ok=0
    for _u in "${_urls[@]}"; do
        _log "下载 $_u"
        if curl -fSL --http1.1 --connect-timeout 20 --max-time 300 \
                --speed-limit 51200 --speed-time 60 "$_u" -o "$_tmp"; then
            _ok=1; break
        fi
        rm -f "$_tmp"
    done
    [ "$_ok" = 1 ] || { echo "所有下载源均失败（版本 ${QDRANT_VERSION}，架构 ${QARCH}）" >&2; exit 1; }
    tar -xzf "$_tmp" -C "$BIN_DIR"
    rm -f "$_tmp"
    chmod +x "$QDRANT_BIN"
    test -x "$QDRANT_BIN" || { echo "解压后未见可执行文件 $QDRANT_BIN" >&2; exit 1; }
    _log "就位: $QDRANT_BIN ($("$QDRANT_BIN" --version 2>/dev/null || echo '?'))"
}

_start() {
    local _pid
    if _pid="$(_running_pid)"; then
        _log "已在运行 pid=${_pid}（$HEALTH_URL）"
        return 0
    fi
    _download
    [ -f "$CONFIG" ] || { echo "缺少配置文件 $CONFIG" >&2; exit 1; }
    mkdir -p "$STORAGE"

    # 只绑回环：默认配置是 0.0.0.0，本机所有网卡都会监听（鉴权默认关闭）。
    # 环境变量优先级高于配置文件，故不动 config.yaml（容器仍要 0.0.0.0）。
    export QDRANT__SERVICE__HOST="127.0.0.1"
    export QDRANT__STORAGE__STORAGE_PATH="$STORAGE"

    _log "启动 Qdrant ${QDRANT_VERSION}（storage=${STORAGE}，bind=127.0.0.1:${HTTP_PORT}）"
    nohup "$QDRANT_BIN" --config-path "$CONFIG" >"$LOG" 2>&1 &
    _pid=$!
    printf '%s' "$_pid" > "$PIDFILE"

    local _i _ready=0
    local _ready=0 _i
    for _i in $(seq 1 60); do
        if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then _ready=1; break; fi
        if ! _pid_alive "$_pid"; then
            _log "❌ qdrant 异常退出，最后 50 行日志："
            tail -n 50 "$LOG" || true
            rm -f "$PIDFILE"
            exit 1
        fi
        sleep 1
    done
    if [ "$_ready" != 1 ]; then
        _log "❌ qdrant 60 秒未就绪，最后 50 行日志："
        tail -n 50 "$LOG" || true
        exit 1
    fi
    _log "就绪：${HEALTH_URL}（日志 ${LOG}，pid ${_pid}）"
}

_stop() {
    local _pid
    if ! _pid="$(_running_pid)"; then
        _log "未在本脚本记录中发现运行中的 qdrant（pid 文件: $PIDFILE）"
        return 0
    fi
    _log "停止 pid=${_pid}"
    kill "$_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
        _pid_alive "$_pid" || break
        sleep 0.5
    done
    if _pid_alive "$_pid"; then
        _log "未在 10 秒内退出，发送 SIGKILL"
        kill -9 "$_pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    _log "已停止"
}

_status() {
    local _pid
    if _pid="$(_running_pid)"; then
        _log "运行中 pid=${_pid}"
        if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
            _log "健康：OK (${HEALTH_URL})"
        else
            _log "健康：无响应 (${HEALTH_URL})"
        fi
    else
        _log "未运行（本脚本）"
    fi
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
        _log "端口 ${HTTP_PORT} 上有 qdrant 在响应（可能是容器或其它实例）"
    fi
}

case "${1:-start}" in
    start)    _start ;;
    stop)     _stop ;;
    status)   _status ;;
    download) _download ;;
    *) echo "用法: $0 {start|stop|status|download}" >&2; exit 2 ;;
esac
