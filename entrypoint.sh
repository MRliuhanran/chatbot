#!/usr/bin/env bash
# 单容器编排：Qdrant → Ollama → 权重按需下载 → 检索模型服务(infinity，可选) → 生成模型按需拉取 → Streamlit（前台监督）
# 运行期：每个进程（含 Streamlit）都有健康巡检与崩溃/僵死自愈，只重启进程、不重新部署。
# 端口与地址全部走 CLI 参数，不改仓库内任何配置文件（CLI 优先级高于 .streamlit/config.toml）。
set -u

cd /app || exit 1

_log() { printf '[entrypoint] %s\n' "$*"; }

# 模型清单与拉取的唯一实现（构建期烘焙 / 运行期补下共用），见 tools/fetch_models.sh。
# 缺失时只告警不中断：镜像里必然存在，本地裸跑不会走到这里（第 6 行已 cd /app）。
. /app/tools/fetch_models.sh || _log "⚠️ 缺少 /app/tools/fetch_models.sh，模型按需下载不可用"

# ── 0.1 进程存活判据（唯一一份）─────────────────────────────────────────────
# `kill -0` 对**僵尸**返回 0：进程退出后、在被父进程 wait 收尸之前一直是 Z 状态、占着
# pid。巡检/自愈必须把 Z 当死 —— 否则僵尸会被当成"还在运行"，重启永不发生（判据恒真，
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

# ── 0.1c 进程自愈：运行期崩溃 / 僵死自动拉起（指数退避 + 健康巡检）────────────
# 旧实现（4.5）只告警不重启，顾虑是"Qdrant 崩溃循环 + 反复打开损坏 WAL 比直接停机更糟"。
# 退避公式本身就把顾虑消掉了：崩溃越频繁，间隔拉得越长，最后稳定在 120s 一次，与直接
# 停机相比差的只是"每 120s 一次的重启尝试"，而换来崩溃自愈 —— 故不再设熔断/计次上限
# （否则等于把稳定部分也判死，把自愈关了）。参数只留公式里的两个：
# 退避公式（第 n 次连续崩溃，n 从 1 起，起跑不归零；再叠加 0~20% 抖动避免固定节拍）：
#     delay(n) = min(CAP, BASE·2^(n-1)) · (1 + jitter)
#   BASE=5s CAP=120s → 5 / 10 / 20 / 40 / 80 / 120s，此后恒 120s
#   可调 RAG_SUPERVISE_BASE / RAG_SUPERVISE_CAP（公式参数，非开关）
# 只负责"运行期崩溃"：首启失败仍由 _wait_http_ready 直接 exit 1（交给平台重拉容器）。
# ★ 健康巡检（2026-10-24）★：只靠"进程还在"不够 —— 进程活着但 HTTP 挂了（死锁、事件循环
# 阻塞、模型加载失败后不收请求）同样服务不可用，平台侧只看到 502/超时。故存活期间每
# RAG_WATCHDOG_INTERVAL（默认 5s）用 curl -fsS -m 3 打 _svc_health 返回的端点，连续
# RAG_WATCHDOG_FAILS（默认 6）次失败 → kill -9 该 pid、日志打 🩺，随后走下面的退避重启。
# 只重启**进程**，不动容器/部署 —— 这是本模块的核心行为。
# 子进程回收：重启出来的实例是本监督者的子进程，用 wait 收尸；首启实例（主脚本拉起）
#      由主 shell 在 Streamlit 巡检循环里顺手 wait（僵尸仍占 pid 槽，_pid_alive 已排除 Z）。
_svc_health() {
    case "$1" in
        qdrant)    printf 'http://127.0.0.1:6333/healthz' ;;
        ollama)    printf 'http://127.0.0.1:11434/api/version' ;;
        infinity)  printf 'http://127.0.0.1:%s/health' "${_inf_port:-8001}" ;;
        streamlit) printf 'http://127.0.0.1:%s/_stcore/health' "${_st_port:-${RAG_HTTP_PORT:-7860}}" ;;
    esac
}
_supervise() {
    local _name="$1" _logf="$2" _startfn="$3" _pid="${4:-}"
    local _base="${RAG_SUPERVISE_BASE:-5}" _cap="${RAG_SUPERVISE_CAP:-120}"
    local _n=0 _svc_pid _i _ok _t0 _up _d _jit _url _fails
    _url="$(_svc_health "${_name}")"
    while :; do
        # 停机中（trap 已置标记，见 0.1c4）：不再拉起，退出监督循环，别和 docker stop 抢进程。
        [ -f "${_SHUTDOWN_FLAG}" ] && return 0
        if [ -n "${_pid}" ] && _pid_alive "${_pid}"; then
            : # 首启实例由主脚本拉起，这里只接管监控（不重复起第二个）
        else
            _svc_pid=""
            "${_startfn}"
            _pid="${_svc_pid}"
        fi
        _write_pid "${_name}" "${_pid}"
        # 统一就绪宽限（60s）：首启 qdrant/ollama 已被 _wait_http_ready 等过（此处立即通过）；
        # infinity 首启在后台等权重 + 加载模型，必须给宽限，否则会被下面的巡检当"僵死"杀掉。
        _ok=0
        for _i in $(seq 1 60); do
            if curl -fsS -m 3 "${_url}" >/dev/null 2>&1; then _ok=1; break; fi
            _pid_alive "${_pid}" || break
            sleep 1
        done
        # 仅"进程活着但健康检查失败"才告警；进程已死由下面的崩溃分支说明，别重复喊。
        if [ "${_ok}" != 1 ] && _pid_alive "${_pid}"; then
            _log "⚠️ ${_name} 拉起后 60 秒未就绪（pid=${_pid}，进程在但健康检查失败），继续观察"
        fi
        # ★ 存活期健康巡检 ★：每 _watch_interval 秒打一次健康端点，连续 _watch_fails 次失败
        # = 进程僵死（在跑但不服务）→ kill -9，落到下面同一条崩溃重启路径（退避公式不变）。
        _t0="$(date +%s)"
        _fails=0
        while _pid_alive "${_pid}" && [ ! -f "${_SHUTDOWN_FLAG}" ]; do
            if curl -fsS -m 3 "${_url}" >/dev/null 2>&1; then
                _fails=0
            else
                _fails=$(( _fails + 1 ))
                if [ "${_fails}" -ge "${_watch_fails}" ]; then
                    _log "🩺 ${_name} 连续 ${_fails} 次健康检查失败（${_url}，间隔 ${_watch_interval}s）→ kill -9 pid=${_pid}，转入崩溃重启"
                    kill -9 "${_pid}" 2>/dev/null || true
                    break
                fi
            fi
            sleep "${_watch_interval}"
        done
        wait "${_pid}" 2>/dev/null || true # 回收自己拉起的子进程（非自身子进程时无副作用）
        [ -f "${_SHUTDOWN_FLAG}" ] && return 0
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

# ── 0.1c2 巡检参数（唯一一份：_supervise 与 Streamlit 监督循环共用）──────────────
# 间隔/阈值可调，但非法值（空/非数字/0）必须退回默认：sleep 0 + 紧循环会把这台 2vCPU
# 容器直接打满。case 做纯数字校验，不引入 expr/awk。
_watch_interval="${RAG_WATCHDOG_INTERVAL:-5}"
case "${_watch_interval}" in ''|*[!0-9]*) _watch_interval=5 ;; esac
[ "${_watch_interval}" -ge 1 ] || _watch_interval=5
_watch_fails="${RAG_WATCHDOG_FAILS:-6}"
case "${_watch_fails}" in ''|*[!0-9]*) _watch_fails=6 ;; esac
[ "${_watch_fails}" -ge 1 ] || _watch_fails=6

# ── 0.1c3 当前 pid 落盘（写者=各监督循环，读者=[watchdog] 汇总与 trap）──────────
# _supervise 在子 shell 里重启服务，主 shell 的 ${_qdrant_pid} 只代表首启实例、会过期；
# pid 文件是两侧"当前 pid"的唯一通道（只写一行，读不到就当没有）。
_write_pid() { printf '%s\n' "$2" > "/tmp/$1.pid" 2>/dev/null || true; }
_read_pid()  { head -n1 "/tmp/$1.pid" 2>/dev/null || true; }

# ── 0.1c3b 可中断 sleep（只在主 shell 用）────────────────────────────────────
# 主 shell 是 PID 1 且装了 TERM/INT trap，而 bash 只在"当前前台命令结束"后才执行 trap：
# 普通 sleep 会让主循环的巡检间隔、以及最长 120s 的重启退避段完全不可中断 —— docker stop
# 的租约等不到就变 SIGKILL，不叫"正常停止"。`sleep & wait $!` 让 wait 被信号打断后立刻
# 执行 trap（子 shell 的 _supervise 不需要：子 shell 不继承 trap，TERM 直接生效）。
_sleep_int() {
    local _sp
    sleep "$1" &
    _sp=$!
    wait "${_sp}" 2>/dev/null || true
}

# ── 0.1c4 停止路径：docker stop → SIGTERM/INT ────────────────────────────────
# 改造前 Streamlit 由 exec 成为 PID 1，TERM 直接结束它、容器随之退出。改成前台监督循环
# 后主 shell 是 PID 1：不显式转发 TERM 的话，docker stop 只能等租约超时再 SIGKILL，
# Qdrant 拿不到 TERM 就没有正常 flush。顺序：
#   ① 置停机标记（各 _supervise 循环看到后不再拉起，见 4.5）；
#   ② TERM 监督循环自身（它们只会 wait，不转发信号）；
#   ③ TERM Streamlit 与 qdrant/ollama/infinity 的**当前** pid（从 pid 文件读，重启过也对）；
#   ④ 等它们退出（≤6s），仍未退的 KILL，wait 收尸，exit 0 —— docker stop 正常返回。
_SHUTDOWN_FLAG="/tmp/.entrypoint-shutdown"
rm -f "${_SHUTDOWN_FLAG}"
# 同一容器内重复起脚本时，上一轮的 pid 文件可能是**已复用**的 pid：启动时清掉，别让 trap 误杀。
rm -f /tmp/qdrant.pid /tmp/ollama.pid /tmp/infinity.pid /tmp/streamlit.pid
_supervise_pids=""
_st_pid=""
_graceful_stop() {
    trap - TERM INT
    : > "${_SHUTDOWN_FLAG}"
    _log "🛑 收到停止信号（TERM/INT）：停止 streamlit / qdrant / ollama${_infinity_pid:+ / infinity}"
    local _p _pids="" _i _alive
    for _p in ${_supervise_pids:-}; do
        kill -TERM "${_p}" 2>/dev/null || true
    done
    for _p in "${_st_pid:-}" "$(_read_pid qdrant)" "$(_read_pid ollama)" "$(_read_pid infinity)"; do
        [ -n "${_p}" ] || continue
        kill -TERM "${_p}" 2>/dev/null || true
        _pids="${_pids} ${_p}"
    done
    for _i in 1 2 3 4 5 6; do
        _alive=0
        for _p in ${_pids}; do _pid_alive "${_p}" && _alive=1; done
        [ "${_alive}" = 0 ] && break
        sleep 1
    done
    for _p in ${_pids}; do
        _pid_alive "${_p}" && kill -KILL "${_p}" 2>/dev/null || true
        wait "${_p}" 2>/dev/null || true
    done
    _log "已停止（streamlit + 子进程收尸完成），退出"
    exit 0
}
trap '_graceful_stop' TERM INT

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

# ── 0.2c 启动配置指纹（一次部署打一行，事后归因"线上到底跑的是什么"）──────────────
# 2026-10-24 内存策略定稿后，应用层不再做互斥驻留/自适应/压力闸（见 chatbot.py 顶部），
# 但容器里仍在读 .env。启动第一屏把**生效值**与**已废弃变量**一次打全：
#   · .env sha256 —— 容器里的 .env 是构建期 COPY 进去的，线上与仓库是不是同一份只能看这个；
#   · MODEL / RAG_NUM_CTX / OLLAMA_KEEP_ALIVE / RAG_MODEL_DL_CONNS / RAG_GEN_TOKENIZER_PATH
#     —— 排障时最常被问"线上生效的是哪个档"的五个键（未设就写 <unset>，别让读者猜）；
#   · 已废弃键 —— 写了也不生效，但会让人误以为策略还在；告警不退出（残留值本身无害）。
_env_sha="(无 .env)"
if [ -f .env ]; then
    _env_sha="$(sha256sum .env 2>/dev/null | awk '{print $1}')"
    _env_sha="${_env_sha:-读不到}"
fi
_log "[config] .env sha256=${_env_sha} MODEL=${MODEL:-<unset>} RAG_NUM_CTX=${RAG_NUM_CTX:-<unset>} OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-<unset>} RAG_MODEL_DL_CONNS=${RAG_MODEL_DL_CONNS:-<unset>} RAG_GEN_TOKENIZER_PATH=${RAG_GEN_TOKENIZER_PATH:-<unset>}"
# 已废弃（2026-10-24 定稿，以代码为准）：chatbot.py 已不再读取下列变量。仍被 .env / 平台
# 注入时打 ⚠️ 说明"此项不生效"，避免把旧档位的残留当成现行策略。
for _dep_var in RAG_EXCLUSIVE_MODELS RAG_GEN_KEEP_ALIVE RAG_EXCLUSIVE_AUTO_FREE_MB RAG_TURN_MIN_FREE_MB; do
    eval "_dep_val=\"\${${_dep_var}:-}\""
    [ -n "${_dep_val}" ] || continue
    _log "⚠️  废弃变量 ${_dep_var}=${_dep_val}：当前代码已不再读取（以代码为准），此项不生效；可从 .env / 平台环境删除"
done
unset _dep_var _dep_val

# ── 0. 运行期目录与下载源 ────────────────────────────────────────────────────
# 嵌入/重排权重的备源 HF 镜像；主源用魔搭自己的 hub（创空间内必达）。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# ── 0.1 权重根目录三级策略（2026-10-22，与 no-bake 配套）────────────────────────
# 必须落在「目录创建 / ollama serve / 各消费方」之前：这是 MODELS_DIR / OLLAMA_MODELS 的
# 最终定值处。三层按优先级：
#   ① 烘焙镜像里已带权重（/app/models 的模型目录含 config.json，或 ollama manifests 已在
#      /app/ollama_models）→ 忠于镜像根目录（已就绪，实时可用）；
#   ② 非烘焙 + 平台持久盘 /mnt/workspace 可写 → 两个根目录都迁过去：首启下载 ~6GB 一次，
#      此后**再部署/重启直接命中盘上清单**（_download_models 与 _pull_ollama_model 的
#      就绪判据本就"清单齐备即跳过"），配合 no-bake 的 ~12min 部署 = 全程达成零重下；
#   ③ 都不满足 → /app/models、/app/ollama_models（原行为，可就地下载）。
# 显式 MODELS_DIR（.env / 环境）最高优先级，本策略不覆盖——".env 里写了就是用户的意图"。
_con_w_baked=0
[ -f "/app/models/$(basename "${RAG_EMBED_MODEL_PATH:-bge-m3}")/config.json" ] && _con_w_baked=1
[ -f "/app/models/$(basename "${RAG_RERANK_MODEL_PATH:-bge-reranker-v2-m3}")/config.json" ] && _con_w_baked=1
_oll_baked=0
[ -d /app/ollama_models/manifests ] && _oll_baked=1
_mnt_ok=0
[ -d /mnt/workspace ] && [ -w /mnt/workspace ] && _mnt_ok=1
if [ -n "${MODELS_DIR:-}" ]; then
    _log "权重根目录: MODELS_DIR=${MODELS_DIR} 已显式指定（三级策略不覆盖）"
elif [ "${_con_w_baked}" = 1 ]; then
    export MODELS_DIR=/app/models
else
    if [ "${_mnt_ok}" = 1 ]; then
        export MODELS_DIR=/mnt/workspace/models
        _log "权重根目录: 非烘焙 + 持久盘可写 → MODELS_DIR=${MODELS_DIR}（权重只下一次，再部署命中清单即跳过下载）"
    else
        export MODELS_DIR=/app/models
    fi
fi
if [ "${_oll_baked}" = 0 ] && [ "${_mnt_ok}" = 1 ]; then
    export OLLAMA_MODELS=/mnt/workspace/ollama_models
fi
# 这是全脚本唯一不 exit 1 的启动前置条件（脚本没有 set -e）：磁盘满/配额超/只读挂载
# 时 Ollama 与权重目录建不出来，后面却照常往下走，最后表现为"容器起来了但一个模型都
# 没有"，且日志里没有任何一条指向根因。
# 权重根目录与 fetch_models.sh / infinity / 自检门禁同源（MODELS_DIR 可被 .env 覆盖，
# 见评审 #8；默认 /app/models 不变，只是不再写死）。
if ! mkdir -p "${OLLAMA_MODELS}" "${MODELS_DIR}" 2>/tmp/mkdir.log; then
    _log "❌ 无法创建运行期目录（磁盘满 / 配额超 / 只读挂载？），详见 /tmp/mkdir.log："
    cat /tmp/mkdir.log
    exit 1
fi

# ── 0.1d 检索模型路径落位（2026-10-23 二次回归修复，取代"重写环境变量"一版）────
# .env 的 RAG_EMBED_MODEL_PATH / RAG_RERANK_MODEL_PATH 是**相对路径**（./models/bge-m3），
# chatbot.py 原样使用（cwd=/app → /app/models/...）。而 0.1 三级策略 ② 会把权重迁到
# /mnt/workspace/models —— 不桥接时应用侧永远看到 /app/models/bge-m3 缺失（2026-10-23
# 21:07「嵌入模型目录缺失或为空」事故链）。
# 一版修复（0.1d 首版）把相对值**重写**成 ${MODELS_DIR}/<basename>：目录缺失解决了，
# 但这个字面量直接进切分指纹（chatbot._chunking_config_fingerprint 的 embed_model
# 字段）——本地建索引时算 "./models/bge-m3"、创空间运行时算
# "/mnt/workspace/models/bge-m3"，同一 .env 两侧指纹**永远对不上**，检索闸门
# fail-closed（本地实测：字面量 923236fe7e4acc7c / 重写 e316a5d282d8ee88，后者与
# 21:33 线上报错完全同值——因果闭环）。**正解：字面量一个字不动**（指纹两侧同源），
# 落位交给 symlink：${MODELS_DIR}/<basename> 链到 ./models/<basename>。
# · 绝对值仍保留（绝对路径 = 用户意图，同 0.1 分寸）；
# · src==dst（烘焙态 MODELS_DIR=/app/models）跳过，防自环、防动真实目录；
# · ./models 下已存在**真实目录**（烘焙态部分清单）也跳过——下载会落在
#   ${MODELS_DIR}=同一目录，应用侧本来就看得见；
# · symlink 失败才回退旧重写：可用性优先于指纹，回退留痕、闸门会在首查询集中报错。
# 位置约束：必须在 mkdir 之后、任何消费方（fetch_models、infinity、自检、Streamlit
# 进程环境）之前；CWD 于第 6 行已 cd /app。fetch_models 只取 basename，不受影响。
_link_retrieval_model() {
    _var="$1"
    eval "_val=\"\${${_var}:-}\""
    if [ -z "${_val}" ]; then   # 未设时补默认值：给消费方与日志留痕一个具体值
        _val="$2"
        eval "export ${_var}=\"${_val}\""
    fi
    case "${_val}" in
        /*) _log "检索模型落位: ${_var}=${_val}（绝对路径保留）"; return 0 ;;
    esac
    _bas="$(basename "${_val}")"
    _src="${MODELS_DIR:-/app/models}/${_bas}"
    _dst="models/${_bas}"
    _dst_abs="${PWD}/models/${_bas}"
    if [ "${_src}" = "${_dst_abs}" ] || { [ -d "${_dst}" ] && [ ! -L "${_dst}" ]; }; then
        _log "检索模型落位: ${_var}=${_val}（${MODELS_DIR:-/app/models}/${_bas} 已是真实目录或同位）"
        return 0
    fi
    mkdir -p models 2>>/tmp/link.log
    [ -d "${_dst}" ] && [ ! -L "${_dst}" ] && rmdir "${_dst}" 2>>/tmp/link.log
    if ln -snf "${_src}" "${_dst}" 2>>/tmp/link.log; then
        _log "检索模型落位: ${_dst} → ${_src}（${_var} 字面量保留，切分指纹两侧同源）"
    else
        export ${_var}="${_src}"   # 回退 = 一版重写行为：先保目录可用，指纹不一致
        _log "⚠️ symlink 失败（见 /tmp/link.log），回退重写 ${_var}=${_src}——" \
             "此时切分指纹可能与本地产物不同源，检索闸门将在首次查询集中报错留痕"
    fi
}
_link_retrieval_model RAG_EMBED_MODEL_PATH "./models/bge-m3"
_link_retrieval_model RAG_RERANK_MODEL_PATH "./models/bge-reranker-base"

# ── 0.2 持久卷探测（观测行，随三级策略保留）──────────────────────────────────
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
_write_pid qdrant "${_qdrant_pid}"   # 当前 pid 落盘（[watchdog] 汇总与 trap 用）
_log "qdrant 启动中（storage=${QDRANT__STORAGE__STORAGE_PATH}）"
_wait_http_ready qdrant http://127.0.0.1:6333/healthz "${_qdrant_pid}" /tmp/qdrant.log
_log "qdrant 就绪（http://127.0.0.1:6333）"

# ── 2. Ollama ──────────────────────────────────────────────────────────────
# 生成模型驻留策略（2026-10-24 定稿）：本脚本**不再替 Ollama 设默认值**。
# 旧行为 export OLLAMA_KEEP_ALIVE=-1（永驻）是"应用层互斥驻留"方案的一部分，该方案已
# 整体废弃 —— 应用层不编排生成模型内存，每个进程按自身策略使用内存（见 chatbot.py 顶部）。
# 现在由 Ollama 服务端变量 OLLAMA_KEEP_ALIVE 决定：未显式设置时采用 **Ollama 自身默认**
# （上游 5m，闲置后自动卸载）；.env / 平台注入了显式值就原样沿用（_load_dotenv 已导出，
# ollama serve 从本 shell 继承）。这里只把"实际采用哪一种"写进日志，不设默认值。
if [ -n "${OLLAMA_KEEP_ALIVE:-}" ]; then
    _log "OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}（沿用 .env/平台显式值，覆盖 Ollama 默认）"
else
    _log "OLLAMA_KEEP_ALIVE 未设置 → 采用 Ollama 自身默认（上游 5m，闲置后自动卸载生成模型）"
fi
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
_write_pid ollama "${_ollama_pid}"   # 当前 pid 落盘（[watchdog] 汇总与 trap 用）
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
# 注意：infinity 常驻它的两个检索模型，服务本身没有卸载接口；内存策略已改为"应用层不
# 编排、各进程按自身策略用内存"（2026-10-24 定稿），这里只负责进程存活与健康巡检。
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
    _write_pid infinity "${_infinity_pid}"   # 当前 pid 落盘（[watchdog] 汇总与 trap 用）
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

# ── 4.5 运行期自愈：进程崩溃/僵死自动拉起 ────────────────────────────────────
# 旧版 exec 让 Streamlit 当 PID 1，容器里没人 wait：qdrant/ollama 运行期崩了只会变僵尸，
# 既不重启也不告警 —— 启动期的存活检查只跑一次。表现为"页面照常 200、检索或生成全失败"，
# 而日志里一条线索都没有（失效原型：状态失步 + 静默）。
# 现为三层自愈，无条件启用、不设开关（有开关就迟早被关掉，静默与状态失步又回来）：
#   ① 崩溃重启：进程退出 → 指数退避后拉起（公式与顾虑见 0.1c 注释）；
#   ② 健康巡检：进程活着但 HTTP 连续失败 → kill -9 后走 ①（见 0.1c）；
#   ③ Streamlit 同样在前台监督循环里巡检/重启（见 5.1），主 shell 从此兼作 PID 1 监督者。
# 不想要自动重启的是"编排层应该重拉容器"的场景，那由首启失败 exit 1 处理，不归这里管。
_supervise qdrant   /tmp/qdrant.log   _qdrant_start   "${_qdrant_pid}"   &
_supervise_pids="${_supervise_pids} $!"
_supervise ollama   /tmp/ollama.log   _ollama_start   "${_ollama_pid}"   &
_supervise_pids="${_supervise_pids} $!"
if [ -n "${_infinity_pid}" ]; then
    _supervise infinity /tmp/infinity.log _infinity_start "${_infinity_pid}" &
    _supervise_pids="${_supervise_pids} $!"
fi

# ── 4.5b [watchdog] 巡检汇总：每 60s 一行 pid + health ─────────────────────────
# _supervise 只在"出事"时打日志（🩺 / 🔴）；常态心跳用来区分"真的没事"与"巡检循环
# 自己死了"—— 后者完全静默，正是本仓库反复点名的失效原型。pid 读 pid 文件拿**当前**值
# （重启过的话主 shell 的 ${_qdrant_pid} 已过期），健康判据与 _supervise 同一个 _svc_health。
_watchdog_report() {
    local _n _p _u _h _line=""
    for _n in qdrant ollama streamlit infinity; do
        if [ "${_n}" = infinity ] && [ "${_infinity_wanted:-0}" != 1 ]; then
            _line="${_line} infinity=未启用"
            continue
        fi
        _p="$(_read_pid "${_n}")"
        if [ -n "${_p}" ] && _pid_alive "${_p}"; then
            _u="$(_svc_health "${_n}")"
            if curl -fsS -m 3 "${_u}" >/dev/null 2>&1; then _h=ok; else _h=fail; fi
        else
            _p="-"; _h=fail
        fi
        _line="${_line} ${_n}(pid=${_p},health=${_h})"
    done
    _log "[watchdog]${_line}"
}
(
    while :; do
        sleep 60
        [ -f "${_SHUTDOWN_FLAG}" ] && exit 0
        _watchdog_report
    done
) &

# ── 4.6 冷启动自检：生成链路（默认）/ 完整检索+生成（可选）──────────────────────
# 8G 档 llama-server 被 cgroup OOM 杀时（`signal: killed`），业务侧只表现为一次 500，
# 运行日志里既无 KV 规模也无峰值内存；而容器外又驱动不了 Streamlit（脚本只在浏览器
# websocket 会话建立后才跑，curl 只拿静态壳）。故把验收前移到启动期，把结果与内存
# 现场直接写进运行日志（前缀 [自检]）。
# **必须后台 + 延迟**：早期版本在启动 Streamlit 之前同步跑，自检占住 7860 端口 85~240
# 秒，平台健康探针等不到监听 → 容器反复重启（2026-10-16 实测 2 分钟内 6 次冷启动）。
# 默认只跑「生成探针」（不导入 torch，进程几十 MB，与真实查询重叠也不会撑爆 8G）；
# 需要验收检索侧时再开 RAG_BOOT_SELFTEST_FULL=1。RAG_BOOT_SELFTEST=0 可整体关掉。
# ★ no-bake 补跑（2026-10-23）★：旧写法"boot 时判 config.json 不在就永久跳过"，切到
# 运行期下载（BAKE_*=0）后**判定恒假、且权重晚到也不补跑**——自检与 KV/FA 实证链路
# 整体缺席（2026-10-22 两次部署实测）。改为**等待权重出现再跑**（不是"在-boot 判定）：
# 烘焙模式目录即刻存在=行为不变（延迟 RAG_BOOT_SELFTEST_DELAY 后开跑）；运行期下载
# 模式下权重就绪（清单齐备、embed/rerank 落位）后先等 RAG_BOOT_SELFTEST_DELAY 再跑。
# 等待上限 RAG_SELFTEST_WAIT_SECS（默认 900s）兜底；RAG_SKIP_MODEL_DOWNLOAD=1（本地挂
# 载自测）时挂载点无 config.json 也按"等不到"退出，不产生误导性死等行。
if [ "${RAG_BOOT_SELFTEST:-1}" != 0 ]; then
    (
        # 目录名从 .env 推导（与 271-272 行同源）；MODELS_DIR 由 0.1 三级策略已定值。
        _sel_embed="${MODELS_DIR:-/app/models}/$(basename "${RAG_EMBED_MODEL_PATH:-bge-m3}")"
        _sel_rerank="${MODELS_DIR:-/app/models}/$(basename "${RAG_RERANK_MODEL_PATH:-bge-reranker-v2-m3}")"
        _i=0
        while [ ! -f "${_sel_embed}/config.json" ] && [ ! -f "${_sel_rerank}/config.json" ]; do
            [ "${_i}" -lt "${RAG_SELFTEST_WAIT_SECS:-900}" ] || { _log "启动自检跳过：等待检索权重 ${RAG_SELFTEST_WAIT_SECS:-900}s 仍未就位"; exit 0; }
            sleep 3
            _i=$((_i + 3))
        done
        sleep "${RAG_BOOT_SELFTEST_DELAY:-45}"
        _log "启动自检开始（前缀 [自检]；服务已就绪，自检不阻塞端口）"
        timeout 900 python /app/tools/boot_selftest.py || \
            _log "⚠️ 启动自检超时/异常退出（非致命，服务照常提供）"
    ) &
fi

# ── 5. Streamlit：前台监督循环（巡检 + 自愈），端口按创空间要求固定 7860 ──────
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

# ── 5.1 Streamlit 监督循环：巡检 + 僵死/退出重启（只重启进程，不重新部署）────────
# 旧版用 exec 把 Streamlit 变成 PID 1：它崩了 / 健康检查失败没人管（页面 502 或挂起，
# 容器却"还在"），平台只能靠外部探针重拉 = 重新部署。现在主 shell 兼任 PID 1 监督者：
#   · 后台起 Streamlit，stdout/stderr **不重定向**（平台日志照收，保留原有日志形态）；
#   · 每 _watch_interval 秒 curl /_stcore/health，连续 _watch_fails 次失败 → kill -9；
#   · 进程退出或被巡检杀 → 指数退避 5s→120s 后重启（公式/参数与 _supervise 同源）。
# 首次拉起给 60s 就绪宽限（要 import torch / 加载配置），避免把还在冷启动的进程杀掉。
_st_port="${RAG_HTTP_PORT:-7860}"
_streamlit_start() {
    python -m streamlit run chatbot.py \
        --server.address 0.0.0.0 \
        --server.port "${_st_port}" \
        --server.headless true &
    _st_pid=$!
    _write_pid streamlit "${_st_pid}"
}
# 60s 就绪宽限：健康即返回 0；进程提前退出返回 1；60s 仍不健康返回 1（不杀，转入巡检）。
_streamlit_ready() {
    local _i
    for _i in $(seq 1 60); do
        curl -fsS -m 3 "$(_svc_health streamlit)" >/dev/null 2>&1 && return 0
        _pid_alive "${_st_pid}" || return 1
        sleep 1
    done
    return 1
}
# 首启实例（qdrant/ollama/infinity 由主脚本拉起）是本 shell 的子进程；被巡检 kill -9 后
# 若不 wait 会变僵尸（_pid_alive 已排除 Z，但僵尸仍占 pid 槽）。监督循环每轮顺手收尸。
_reap_initial_services() {
    local _p
    for _p in "${_qdrant_pid:-}" "${_ollama_pid:-}" "${_infinity_pid:-}"; do
        [ -n "${_p}" ] || continue
        _pid_alive "${_p}" || wait "${_p}" 2>/dev/null || true
    done
}
_log "Streamlit 启动: http://0.0.0.0:${_st_port}（前台监督：巡检 ${_watch_interval}s / 连续 ${_watch_fails} 次失败重启）"
_st_n=0
while :; do
    _streamlit_start
    if ! _streamlit_ready && _pid_alive "${_st_pid}"; then
        _log "⚠️ Streamlit 拉起后 60 秒未就绪（pid=${_st_pid}，进程在但健康检查失败），转入巡检"
    fi
    _st_fails=0
    _st_t0="$(date +%s)"
    while _pid_alive "${_st_pid}"; do
        _reap_initial_services
        if curl -fsS -m 3 "$(_svc_health streamlit)" >/dev/null 2>&1; then
            _st_fails=0
        else
            _st_fails=$(( _st_fails + 1 ))
            if [ "${_st_fails}" -ge "${_watch_fails}" ]; then
                _log "🩺 streamlit 连续 ${_st_fails} 次健康检查失败（$(_svc_health streamlit)，间隔 ${_watch_interval}s）→ kill -9 pid=${_st_pid}，重启进程（不重新部署）"
                kill -9 "${_st_pid}" 2>/dev/null || true
                break
            fi
        fi
        _sleep_int "${_watch_interval}"
    done
    wait "${_st_pid}" 2>/dev/null || true
    _st_up=$(( $(date +%s) - _st_t0 ))
    _st_n=$(( _st_n + 1 ))
    _st_d="$(awk -v b="${RAG_SUPERVISE_BASE:-5}" -v n="${_st_n}" -v c="${RAG_SUPERVISE_CAP:-120}" \
        'BEGIN{d=b*2^(n-1); if(d>c)d=c; printf "%d", d}')"
    _st_jit=$(( RANDOM % 20 ))
    _st_d=$(( _st_d + _st_d * _st_jit / 100 ))
    _log "🔴 streamlit 退出/被巡检重启（存活 ${_st_up}s）→ 第 ${_st_n} 次拉起，退避 ${_st_d}s（min(${RAG_SUPERVISE_BASE:-5}·2^(n-1), ${RAG_SUPERVISE_CAP:-120})+抖动）"
    _sleep_int "${_st_d}"
done
