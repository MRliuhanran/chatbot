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

# ── 0.1c 进程自愈：运行期崩溃自动拉起（指数退避）────────────────────────────
# 旧实现（4.5）只告警不重启，顾虑是"Qdrant 崩溃循环 + 反复打开损坏 WAL 比直接停机更糟"。
# 退避公式本身就把顾虑消掉了：崩溃越频繁，间隔拉得越长，最后稳定在 120s 一次，与直接
# 停机相比差的只是"每 120s 一次的重启尝试"，而换来崩溃自愈 —— 故不再设熔断/计次上限
# （否则等于把稳定部分也判死，把自愈关了）。参数只留公式里的两个：
# 退避公式（第 n 次连续崩溃，n 从 1 起，起跑不归零；再叠加 0~20% 抖动避免固定节拍）：
#     delay(n) = min(CAP, BASE·2^(n-1)) · (1 + jitter)
#   BASE=5s CAP=120s → 5 / 10 / 20 / 40 / 80 / 120s，此后恒 120s
#   可调 RAG_SUPERVISE_BASE / RAG_SUPERVISE_CAP（公式参数，非开关）
# 只负责"运行期崩溃"：首启失败仍由 _wait_http_ready 直接 exit 1（交给平台重拉容器）。
# 子进程回收：重启出来的实例是本监督者的子进程，用 wait 收尸——否则 PID 1=Streamlit
#      不 wait，死进程会积成僵尸（_pid_alive 已排除 Z，但僵尸仍占 pid 槽）。
_svc_health() {
    case "$1" in
        qdrant)   printf 'http://127.0.0.1:6333/healthz' ;;
        ollama)   printf 'http://127.0.0.1:11434/api/version' ;;
        infinity) printf 'http://127.0.0.1:%s/health' "${_inf_port:-8001}" ;;
    esac
}
_supervise() {
    local _name="$1" _logf="$2" _startfn="$3" _pid="${4:-}"
    local _base="${RAG_SUPERVISE_BASE:-5}" _cap="${RAG_SUPERVISE_CAP:-120}"
    local _n=0 _svc_pid _i _ok _t0 _up _d _jit _url
    _url="$(_svc_health "${_name}")"
    while :; do
        if [ -n "${_pid}" ] && _pid_alive "${_pid}"; then
            : # 首启实例由主脚本拉起，这里只接管监控（不重复起第二个）
        else
            _svc_pid=""
            "${_startfn}"
            _pid="${_svc_pid}"
            _ok=0
            for _i in $(seq 1 60); do
                if curl -fsS "${_url}" >/dev/null 2>&1; then _ok=1; break; fi
                _pid_alive "${_pid}" || break
                sleep 1
            done
            # 仅"进程活着但健康检查失败"才告警；进程已死由下面的崩溃分支说明，别重复喊。
            if [ "${_ok}" != 1 ] && _pid_alive "${_pid}"; then
                _log "⚠️ ${_name} 拉起后 60 秒未就绪（pid=${_pid}，进程在但健康检查失败），继续观察"
            fi
        fi
        _t0="$(date +%s)"
        while _pid_alive "${_pid}"; do sleep 3; done
        wait "${_pid}" 2>/dev/null || true # 回收自己拉起的子进程（非自身子进程时无副作用）
        _up=$(( $(date +%s) - _t0 ))
        _n=$(( _n + 1 ))
        _d="$(awk -v b="${_base}" -v n="${_n}" -v c="${_cap}" \
            'BEGIN{d=b*2^(n-1); if(d>c)d=c; printf "%d", d}')"
        _jit=$(( RANDOM % 20 ))
        _d=$(( _d + _d * _jit / 100 ))
        _log "🔴 ${_name} 运行期崩溃（存活 ${_up}s）→ 第 ${_n} 次拉起，退避 ${_d}s（min(${_base}·2^(n-1), ${_cap})+抖动），详见 ${_logf}"
        sleep "${_d}"
        _pid=""
    done
}

# ── 0.2 从 .env 补齐本脚本要用的变量（唯一实现：tools/dotenv.sh）────────────────
# 旧写法在本脚本里自带一份解析器，run_local.sh 又自带第三份、fetch_models.sh 另有一份取值
# 器：三者对 export 前缀 / 行内注释 / 同名重复的处理各不相同，同一份 .env 在容器、本地
# 一键起停、构建期读出三种结果（失效原型：多份）。现收敛成一份 —— 本脚本只 source + 调用。
# 语义（进程环境优先、同名重复最后一次生效、引号与行内注释、白名单前缀、MODELS_DIR 在列）
# 与各自触发场景逐条写在 tools/dotenv.sh 头注释里，此处不复述 —— 同一知识只留一处实现：
# 复述一次就多一份会分叉的副本。
# 这个文件缺失 = 镜像不完整：.env 里**每一个**键都会被静默忽略（端口/后端/模型路径全部
# 退回代码默认值），正是本仓库点名的静默原型 —— 故 fail-closed 退出，不降级继续跑。
if [ -f /app/tools/dotenv.sh ]; then
    . /app/tools/dotenv.sh
else
    _log "❌ 缺少 /app/tools/dotenv.sh（.env 解析的唯一实现）—— .env 配置会全部失效，退出"
    exit 1
fi
_load_dotenv

# ── 0.2b 硬件适配参数：唯一实现 tools/hw_profile.py（经 tools/hw_profile.sh 薄封装；与本地
#         run_local.sh / run_ollama.sh 共用同一份）──
# 线程数、MALLOC_ARENA_MAX、OLLAMA_NUM_PARALLEL 这些**硬件相关**的旋钮不再写死：容器里按
# cgroup 的 quota/limit 派生（nproc 会报宿主核数，直接用会让 2 vCPU 的机器开几十个线程）。
# **必须在 _load_dotenv 之后**：派生只填"没被显式设过"的键，而 .env / 平台注入正是"显式"；
# 反过来会让 .env 永远失效（本仓库点名过的静默原型）。
# 放这么早是因为 MALLOC_ARENA_MAX 要在**每个**子进程启动前就位 —— 下面 qdrant / ollama /
# infinity / streamlit 全都继承本 shell。
# 缺它 = 镜像不完整：线程数会退回各工具自己的默认、arena 不设，在 2vCPU/8G 上就是"按宿主
# 核数开线程 + 内存碎片化"，正是本模块要消除的那件事。故与 dotenv.sh 同规格 fail-closed。
if [ -f /app/tools/hw_profile.sh ]; then
    . /app/tools/hw_profile.sh
    # 探测与派生都在 tools/hw_profile.py 里，本行只是 eval 它的 --sh 输出。任一步失败都必须
    # 拦下来：线程数/arena 静默偏离正是这个模块要消除的那类问题（与"缺 dotenv.sh 就退出"同规格）。
    _load_hw_profile || {
        _log "❌ 硬件契约加载失败（见上一行 [hw] ❌）—— 线程数/内存旋钮会静默偏离，退出"
        exit 1
    }
else
    _log "❌ 缺少 /app/tools/hw_profile.sh（硬件适配参数的薄封装；实现是 tools/hw_profile.py）—— 退出"
    exit 1
fi

# ── 0. 运行期目录与下载源 ────────────────────────────────────────────────────
# Ollama 权重落在 /app 下（不依赖 HOME，非 root 也可写）；平台若给持久卷可挂到这里。
export OLLAMA_MODELS="${OLLAMA_MODELS:-/app/ollama_models}"
# 嵌入/重排权重的备源 HF 镜像；主源用魔搭自己的 hub（创空间内必达）。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# 这是全脚本唯一不 exit 1 的启动前置条件（脚本没有 set -e）：磁盘满/配额超/只读挂载
# 时 Ollama 与权重目录建不出来，后面却照常往下走，最后表现为"容器起来了但一个模型都
# 没有"，且日志里没有任何一条指向根因。
# 权重根目录与 fetch_models.sh / infinity / 自检门禁同源（MODELS_DIR 可被 .env 覆盖，
# 见评审 #8；默认 /app/models 不变，只是不再写死）。
if ! mkdir -p "${OLLAMA_MODELS}" "${MODELS_DIR:-/app/models}" 2>/tmp/mkdir.log; then
    _log "❌ 无法创建运行期目录（磁盘满 / 配额超 / 只读挂载？），详见 /tmp/mkdir.log："
    cat /tmp/mkdir.log
    exit 1
fi

# ── 0.1 持久卷探测（no-bake 观测项，见 Dockerfile "默认关烘焙" 2026-10-22）──────────
# 纯探测、不改行为：若平台给持久盘（/mnt/workspace），权重根后续可迁过去做跨重启缓存
# （运行期下载一次、之后零成本）；探测行写进 run 日志，据此决定是否迁移 MODELS_DIR。
if [ -d /mnt/workspace ]; then
    _mnt_av="$(df -Pk /mnt/workspace 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')"
    _mnt_st="存在（可用 ${_mnt_av:-?}G"
    [ -w /mnt/workspace ] && _mnt_st="${_mnt_st}，可写）" || _mnt_st="${_mnt_st}，只读）"
    _log "持久卷探测: /mnt/workspace ${_mnt_st}"
else
    _log "持久卷探测: 无 /mnt/workspace"
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
export QDRANT__STORAGE__STORAGE_PATH="${QDRANT__STORAGE__STORAGE_PATH:-/app/storage}"
# 拉起函数：首启与运行期自愈（4.5）复用同一份命令；日志追加写，重启不清历史。
_qdrant_start() {
    /opt/qdrant/qdrant --config-path /app/qdrant_config/config.yaml >>/tmp/qdrant.log 2>&1 &
    _svc_pid=$!
}
: > /tmp/qdrant.log
_qdrant_start
_qdrant_pid="${_svc_pid}"
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
# ── 生成侧服务端参数：语义项来自 .env，硬件项来自 hw_profile（Dockerfile 两者都不再注入）──
# 为什么不再注入 ENV：ENV 与 .env 是**两套注入机制**，效果相同、来源不同，改一处必漏
# 另一处；且 _load_dotenv 是"进程环境优先"，只要 Dockerfile 预设了同名 ENV，.env 里的值就
# **永远**不生效（静默原型：配置写了却没用）。现在两边同源，且**按性质分成两类**：
#   · 语义项（KV 量化 / flash attention / 槽位并发）—— 定义行为，与核数无关：
#         .env → _load_dotenv → ollama serve
#   · 硬件项（OLLAMA_NUM_THREADS 等线程数）—— 随机器而变：
#         tools/hw_profile.py 按 cgroup 探测派生（经 hw_profile.sh 薄封装）→ ollama serve
#         （见上方 [hw] 审计行）
# 本机的对应机制：tools/run_ollama.sh / tools/run_local.sh，两者 source 的就是同一份
# dotenv.sh 与 hw_profile.sh（后者只是 hw_profile.py 的薄封装）。
#
# ★ fail-closed（本次统一最容易引入的新静默点）★
# KV_CACHE_TYPE 为空 = KV 退回 f16：65536 窗口下 KV 体积翻倍，8G 档 llama-server 必被
# cgroup OOM 杀（2026-10-16 实测 signal: killed / maxMemoryUsage 7933MB）。旧写法只用
# ${VAR:-<未设=f16>} 打一行日志就继续跑 —— 那是"看得见的降级"里最难被看见的一种（日志
# 在几万行里，事后归因只能看到一次 500）。故改为：为空直接退出，把根因摆在启动第一屏。
if [ -z "${OLLAMA_KV_CACHE_TYPE:-}" ]; then
    _log "❌ OLLAMA_KV_CACHE_TYPE 为空：KV 会退回 f16，65536 窗口下体积翻倍，8G 档必被 OOM 杀。"
    _log "   唯一来源是 /app/.env（Dockerfile 已不再注入该 ENV）：请确认 .env 存在且含"
    _log "   OLLAMA_KV_CACHE_TYPE=q8_0；本地对应机制见 tools/run_ollama.sh。拒绝在错误的内存档位下启动。"
    exit 1
fi
[ -n "${OLLAMA_FLASH_ATTENTION:-}" ] || _log "⚠️  OLLAMA_FLASH_ATTENTION 为空：KV 量化可能不生效（.env 未生效？）"
[ -n "${OLLAMA_NUM_PARALLEL:-}" ]    || _log "⚠️  OLLAMA_NUM_PARALLEL 为空：退回多槽位，65536 窗口按并发数翻倍吃内存（.env 未生效？）"
[ -n "${OLLAMA_NUM_THREADS:-}" ]     || _log "⚠️  OLLAMA_NUM_THREADS 为空：hw_profile 没生效？（启动早期应有一行 [hw]）"
# 生效值留痕：OOM 现场要能一眼看出是不是 q8_0/1/1 **以及它们来自哪里**。这一行是
# "唯一来源"改造的可证日志 —— 前三项只可能来自 .env 或平台注入的同名环境变量，
# NUM_THREADS 只可能来自 tools/hw_profile.py 的派生（或显式覆盖）。
_log "生成侧服务端参数：KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE} FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION:-<未设>} NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-<未设>}〔以上三项来源 /app/.env〕 NUM_THREADS=${OLLAMA_NUM_THREADS:-<未设>}〔来源 tools/hw_profile.py 派生，见上方 [hw] 行〕"
# 拉起函数：首启与运行期自愈（4.5）复用同一份命令。
_ollama_start() {
    ollama serve >>/tmp/ollama.log 2>&1 &
    _svc_pid=$!
}
: > /tmp/ollama.log
_ollama_start
_ollama_pid="${_svc_pid}"
_wait_http_ready ollama http://127.0.0.1:11434/api/version "${_ollama_pid}" /tmp/ollama.log
_log "ollama 就绪（http://127.0.0.1:11434, models=${OLLAMA_MODELS}）"
# 生成侧是否真的用上了量化 KV / flash attention：llama.cpp 只在**模型加载后**把
# KV 缓冲大小写进 ollama 日志，而 FC 只收 stdout —— 故此处把关键行转印到 stdout。
# 判据：出现 `KV buffer size = xxx MiB` 即知道 KV 用了什么精度（q8_0 应明显小于 f16）。
_kv_report() {
    # 窗口必须覆盖启动自检（RAG_BOOT_SELFTEST_DELAY 默认 45s）之后的首个请求：
    # 原来只轮询 10x3=30s，默认档必然走"尚未产生"分支，等于该诊断永不生效（评审 #4）。
    for _i in $(seq 1 50); do
        _line=$(grep -hoE "(KV buffer size = [0-9.]+ MiB|flash[- _]?attn[^ ]*|FLASH_ATTN[^ ]*)" /tmp/ollama.log 2>/dev/null | tail -3 | tr "\n" " ")
        if [ -n "$_line" ]; then _log "🔎 生成侧 KV/FA 实证: $_line"; return 0; fi
        sleep 3
    done
    _log "🔎 生成侧 KV/FA 实证: 尚未产生（首个请求后才会出现，属正常）"
}
_kv_report &

# ── 3. 嵌入/重排权重：清单与实现见 tools/fetch_models.sh（构建期烘焙 / 运行期补下共用）──
#     构建期已烘焙进镜像时，这里只会打"已就绪"、不产生任何下载。RAG_SKIP_MODEL_DOWNLOAD=1
#     可跳过。就绪判据=清单文件全部非空；下到暂存目录齐了再整目录改名就位，避免应用读到
#     半截权重（详见 _download_model）。后台运行，不占住 7860。
_download_models &

# ── 3.5 检索模型服务（infinity）：嵌入 + 重排同一个进程、共享一份 torch ───────────
# RAG_EMBED_BACKEND / RAG_RERANK_BACKEND 任一为 http 时才起。业务进程（Streamlit）因此
# 不再持有检索模型；查询侧按 OpenAI /v1/embeddings 与 Cohere 风格 /v1/rerank 调它。
# **必须在权重就绪之后**（infinity 启动时找不到模型目录会直接退出），故先轮询目录再拉起。
# 命令行只此一处（_infinity_argv 数组），首启与运行期自愈（4.5）共用，避免 argv 漂移。
# 注意：单例（RAG_EXCLUSIVE_MODELS=1）下 infinity 常驻它的两个模型且**没有卸载接口**，
# 互斥只能从另一侧保证 —— 生成模型答完即卸（RAG_GEN_KEEP_ALIVE=0）。
_infinity_pid=""
# http 后端 = 需要 infinity。先算"要不要"，再单独做 command -v 判据（评审 #7），最后才进
# 启动块 —— 启动块本体（argv/等待/自愈复用）一行不动，避免大段缩进带来的改错风险。
_infinity_wanted=0
if [ "${RAG_EMBED_BACKEND:-torch}" = "http" ] || [ "${RAG_RERANK_BACKEND:-torch}" = "http" ]; then
    _infinity_wanted=1
fi
# ── 前置判据（评审 #7）：本镜像**故意不装** infinity ──────────────────────────
# 依据见 Dockerfile：8G 单容器运行档定为 torch 进程内；且 infinity-emb[server] 要求
# uvicorn<0.33.0，与 requirements.txt 的锁定直接冲突（带约束安装必然构建失败）。
# 旧写法不看命令在不在就起子 shell：exec infinity_emb 立刻 127 退出，被 4.5 的自愈当成
# "运行期崩溃"无限重启（退避封顶 120s）—— 日志里全是"运行期崩溃"，
# **没有一条**说"infinity 没装"（失效原型：静默 + 自愈把根因放大成噪声）。
# 修法：先 command -v 判据 + 具名报错，并让 _infinity_pid 保持空 —— 4.5 不会为它注册自愈，
# 也就不会再有"假装崩溃"的重启风暴。触发场景：平台/部署方把两个 BACKEND 注入成 http
# （仓库自带 .env 是 torch），或 RAG_EMBED_URL 其实指向外部 infinity 服务。
# 不 exit 1：本容器还要服务 Qdrant/Ollama/Streamlit，且"外部 infinity"场景下退容器是误杀；
# 真要本地 infinity 时日志已给两条可执行出路。
if [ "${_infinity_wanted}" = 1 ] && ! command -v infinity_emb >/dev/null 2>&1; then
    _log "❌ RAG_EMBED_BACKEND/RAG_RERANK_BACKEND 配了 http，但镜像里没有 infinity_emb（本镜像按 8G 单容器档不安装它）"
    _log "   → 不拉起检索模型服务，也不注册自愈（避免被当成运行期崩溃反复重启）；嵌入/重排将全部失败"
    _log "   出路：① RAG_EMBED_URL 指向外部 infinity 时本行即预期；② 否则把两个 BACKEND 改回 torch；③ 或换用带 infinity 的镜像"
    _infinity_wanted=0
fi
if [ "${_infinity_wanted}" = 1 ]; then
    # 模型根目录与下载（fetch_models.sh）、启动自检门禁同源：MODELS_DIR 现在真的会从 .env
    # 生效（评审 #8），这里若继续写死 /app/models，"权重下到哪"与"infinity 去哪找"就会分叉。
    _embed_dir="${MODELS_DIR:-/app/models}/$(basename "${RAG_EMBED_MODEL_PATH:-./models/bge-m3}")"
    _rerank_dir="${MODELS_DIR:-/app/models}/$(basename "${RAG_RERANK_MODEL_PATH:-./models/bge-reranker-base}")"
    _embed_served="${RAG_EMBED_MODEL_NAME:-$(basename "${_embed_dir}")}"
    _rerank_served="${RAG_RERANK_MODEL_NAME:-$(basename "${_rerank_dir}")}"
    # 端口从 RAG_EMBED_URL 派生（单一来源），解析不出来才退 8001
    _inf_port="$(printf '%s' "${RAG_EMBED_URL:-}" | sed -n 's|^[a-z]*://[^:/]*:\([0-9][0-9]*\).*|\1|p')"
    _inf_port="${_inf_port:-8001}"
    _infinity_argv=(infinity_emb v2 \
        --model-id "${_embed_dir}" --served-model-name "${_embed_served}" \
        --model-id "${_rerank_dir}" --served-model-name "${_rerank_served}" \
        --engine torch --device cpu --no-bettertransformer \
        --url-prefix /v1 --host 127.0.0.1 --port "${_inf_port}")
    _infinity_wait_models() {
        local _i=0
        while [ "${_i}" -lt 300 ]; do
            [ -d "${_embed_dir}" ] && [ -d "${_rerank_dir}" ] && break
            _i=$((_i + 1))
            sleep 2
        done
        _log "检索模型服务启动中：embed=${_embed_served} rerank=${_rerank_served} port=${_inf_port}（日志 /tmp/infinity.log）"
    }
    # 自愈复用（4.5）：后台起并回填 pid。**不用 exec**——exec 会替换调用它的监督者进程。
    _infinity_start() {
        _infinity_wait_models
        "${_infinity_argv[@]}" >>/tmp/infinity.log 2>&1 &
        _svc_pid=$!
    }
    # 首启：等目录可能很久，放后台子 shell 不阻塞主流程；exec 保住 pid，自愈据此接管。
    : > /tmp/infinity.log
    ( _infinity_wait_models; exec "${_infinity_argv[@]}" >>/tmp/infinity.log 2>&1 ) &
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

# ── 4.5 运行期自愈：进程崩溃自动拉起 ─────────────────────────────────────────
# exec 让 Streamlit 当 PID 1，容器里没人 wait：qdrant/ollama 运行期崩了只会变僵尸，
# 既不重启也不告警 —— 上面两处存活检查只在启动期跑一次。表现为"页面照常 200、检索或
# 生成全失败"，而日志里一条线索都没有（失效原型：状态失步 + 静默）。
# 现改为自愈：_supervise 指数退避（公式与顾虑见 0.1c 注释），无条件启用，不设开关 ——
# 有开关就迟早有人把它关掉，然后静默与状态失步又回来了；不想要自动重启的是"编排层
# 应该重拉容器"的场景，那由首启失败 exit 1 处理，不归这里管。
_supervise qdrant   /tmp/qdrant.log   _qdrant_start   "${_qdrant_pid}"   &
_supervise ollama   /tmp/ollama.log   _ollama_start   "${_ollama_pid}"   &
[ -n "${_infinity_pid}" ] && \
    _supervise infinity /tmp/infinity.log _infinity_start "${_infinity_pid}" &

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
    # 目录名**从 .env 推导**（与下面 271-272 行同源）：写死 bge-m3/bge-reranker-v2-m3 会让
    _sel_embed="${MODELS_DIR:-/app/models}/$(basename "${RAG_EMBED_MODEL_PATH:-bge-m3}")"
    _sel_rerank="${MODELS_DIR:-/app/models}/$(basename "${RAG_RERANK_MODEL_PATH:-bge-reranker-v2-m3}")"
    if [ -f "${_sel_embed}/config.json" ] || [ -f "${_sel_rerank}/config.json" ]; then
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

# ── 5.0 创空间 OAuth：把平台注入的环境变量装配成 st.login 需要的 secrets.toml ──────
# 官方（docs/studios/oauth）在运行时**只注入环境变量**（OAUTH_CLIENT_ID / OAUTH_CLIENT_SECRET
# / OAUTH_SCOPES / OPENID_PROVIDER_URL / STUDIO_HOST），而 Streamlit 的 st.login() 只认
# .streamlit/secrets.toml 的 [auth] 段 —— chatbot.py 的登录门也据此判断是否启用。两者之间
# 的桥只能在启动期搭：缺了这一步，线上没有 [auth]，登录门被静默跳过（页面照常，谁也不登录）。
# 约束均取自 Streamlit 1.62 源码，逐条对应：
#   · redirect_uri 必须以 /oauth2callback 结尾（auth_util.get_validated_redirect_uri），
#     回调路由也写死为该路径（starlette_auth_routes._ROUTE_OAUTH_CALLBACK）—— 改路径必失败。
#     又必须用**创空间 host 域名**（https://${STUDIO_HOST}），因为创空间页面嵌在 modelscope.cn
#     的 iframe 里，官方示例同样用 host 地址 + 新窗口，不能用 studios/... 路径。
#   · 默认 scope 是 "openid email profile"，但魔搭只授权 openid/profile（无 email 项）→
#     显式覆盖成平台注入的 OAUTH_SCOPES（即设置页勾选的 openid profile）。
#   · cookie_secret 必须跨重启稳定，否则每次重部署把所有人踢下线 → 由 client_secret 派生。
#   · prompt 不动：实测魔搭 authorize 对 prompt 值原样透传、不校验（连 bogus 值都收），
#     Streamlit 的默认 select_account 安全，无需覆盖。
# 无 OAUTH_* 注入（本地裸跑 / Static）时删掉该文件：Dockerfile 会 COPY 整个 .streamlit/，
# 不清掉的话镜像里烘焙的占位 secrets.toml 会被当成真配置，登录门拿占位 client_id 去请求。
# RAG_LOGIN_ENABLED=0 时同样删掉：登录门总开关在仓库侧（chatbot.py 也读它），平台即使
# 注入 OAUTH_* 也不启用登录（否则平台设置页的 OAuth 开关会越过仓库的意图）。
# 诊断（只列**变量名**，绝不打印值）：部署后看这一行即可区分两种故障 ——
#   [空]          → 平台没开 OAuth（或没重新部署），登录门必然关闭；
#   有名字但没就绪 → 开了 OAuth 但变量名/取值与本脚本的判定对不上，据此改判定即可。
_ms_envnames="$(env | cut -d= -f1 | grep -E '^(OAUTH_|OPENID|STUDIO_)' | sort | tr '\n' ' ' | sed 's/[[:space:]]*$//' || true)"
_log "环境变量体检（仅列名）: [${_ms_envnames:-空}]"
# 登录门总开关（RAG_LOGIN_ENABLED，.env 可改，默认 1=启用）：0=关闭登录门，直接进应用。
# 为什么平台开着 OAuth 也要认这个开关：平台的 OAuth 开关在创空间设置页，仓库侧改不到，
# 仓库要能独立关掉登录门就只认这一个变量（它同时被 chatbot.py 的登录门读取，两处一致）。
_ms_login_on=1
case "$(printf '%s' "${RAG_LOGIN_ENABLED:-1}" | tr 'A-Z' 'a-z')" in
    0|false|no|off) _ms_login_on=0 ;;
esac
_ms_oauth=0
if [ -n "${OAUTH_CLIENT_ID:-}" ] && [ -n "${OAUTH_CLIENT_SECRET:-}" ] && [ -n "${STUDIO_HOST:-}" ]; then
    # TOML 基本字符串：值里出现 " \ 或换行会破坏文件，直接拒绝（正常 client id/secret 不会含）。
    case "${OAUTH_CLIENT_ID}${OAUTH_CLIENT_SECRET}" in
        *'"'*|*'\'*|*$'\n'*)
            _log "❌ OAUTH_CLIENT_ID/SECRET 含 TOML 非法字符（双引号/反斜杠/换行），拒绝生成 secrets.toml，退出"
            exit 1 ;;
    esac
    _ms_oauth=1
fi
if [ "${_ms_oauth}" = 1 ] && [ "${_ms_login_on}" = 1 ]; then
    mkdir -p /app/.streamlit
    _ms_cookie_secret="$(printf '%s' "${OAUTH_CLIENT_SECRET}" | sha256sum | awk '{print $1}')"
    _ms_scope="${OAUTH_SCOPES:-openid profile}"
    _ms_meta="${OPENID_PROVIDER_URL:-https://www.modelscope.cn}"; _ms_meta="${_ms_meta%/}/.well-known/openid-configuration"
    umask 077
    cat > /app/.streamlit/secrets.toml <<EOF
[auth]
redirect_uri = "https://${STUDIO_HOST}/oauth2callback"
cookie_secret = "${_ms_cookie_secret}"
client_id = "${OAUTH_CLIENT_ID}"
client_secret = "${OAUTH_CLIENT_SECRET}"
server_metadata_url = "${_ms_meta}"

[auth.client_kwargs]
scope = "${_ms_scope}"
EOF
    chmod 600 /app/.streamlit/secrets.toml
    _log "OAuth 登录就绪：redirect_uri=https://${STUDIO_HOST}/oauth2callback scope=${_ms_scope}"
else
    rm -f /app/.streamlit/secrets.toml
    if [ "${_ms_login_on}" = 0 ]; then
        _log "RAG_LOGIN_ENABLED=0 → 登录门关闭（平台注入的 OAUTH_* 已忽略，不生成 secrets.toml）"
    else
        _log "未检测到 OAUTH_* 注入 → 登录门关闭（本地/无 OAuth 部署，属正常）"
    fi
fi

_log "Streamlit 启动: http://0.0.0.0:${RAG_HTTP_PORT:-7860}"
exec python -m streamlit run chatbot.py \
    --server.address 0.0.0.0 \
    --server.port "${RAG_HTTP_PORT:-7860}" \
    --server.headless true
