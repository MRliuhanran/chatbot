#!/usr/bin/env bash
# .env → 进程环境的解析器**唯一实现**（本仓库只此一份，entrypoint.sh 与 tools/run_local.sh 共用）。
#
# 为什么单独成文件：此前同一份 .env 有**三份**解析器 —— entrypoint.sh 的 _load_dotenv、
# tools/run_local.sh 的 _env_val、tools/fetch_models.sh 的 _env_get —— 三者对 `export ` 前缀、
# 行内注释、同名重复的处理各不相同，于是同一份 .env 在容器、本地一键起停、构建期读出三种
# 结果（失效原型：多份；本仓库已为"同一知识两处实现"踩过两次坑）。
#
# 用法（只 source，不执行任何逻辑；调用方先 cd 到 .env 所在目录）：
#     . tools/dotenv.sh          # 容器内是 . /app/tools/dotenv.sh
#     _load_dotenv               # 读 ./.env；可显式给路径：_load_dotenv /path/to/.env
#
# 语义（与 python-dotenv 逐条对齐；2026-10-16 用本机 dotenv 3.10.0 实测）：
#   · 空行与 # 注释行跳过；`export KEY=v` 的 export 前缀剥掉；键名两侧空白剥掉；
#   · 值两侧空白剥掉；`KEY="v" # 注释` 按**首个闭合引号**截断（不能只看首尾字符，
#     否则引号会被带进变量）；无引号值只在"空白 + #"处剥行内注释（`v#x` 是值本身）；
#   · **同名重复：最后一次出现生效**（dotenv_values 就是字典覆盖语义，实测 DUP=a/DUP=b → b）。
#     旧 _load_dotenv 是"第一次生效"（已设即跳过），导致应用读 b、脚本读 a —— 又一种
#     "同一份 .env 两份结果"，故按 python-dotenv 对齐（见下方 _preset 快照的区分）。
#   · **进程环境优先**：调用时已在进程环境里的键（含被设为空串）不被 .env 覆盖，与
#     `load_dotenv(override=False)` 一致。实现上先把"进程环境已有的键名"快照下来，
#     这样它与"本函数刚写入的同名重复键"不会混淆（后者要覆盖，前者不能覆盖）。
#   · 只导出白名单前缀的键（见下方 case）：.env 里也有给别的工具看的键，白名单外的静默
#     不导出。MODELS_DIR 在列 —— 它曾是 fetch_models.sh 在 source 时求值的默认值，
#     白名单又漏了它，于是 .env 里写 MODELS_DIR= 被无声忽略（评审 #8）。
# 兼容性：必须能跑在 bash 3.2（macOS /bin/bash）上 —— 禁用关联数组 / mapfile / ${var^^} 等。
_load_dotenv() {
    local _file="${1:-.env}"
    [ -f "${_file}" ] || return 0
    local _line _k _v _preset
    # 进程环境里已有的键名快照（一行一个 KEY），两侧补空格以便用 case 做整词匹配。
    # `env` 的取值里若含换行会多出半行，但 sed 的 ^KEY= 形态会把它滤掉。
    _preset=" $(env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' | tr '\n' ' ') "
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
            RAG_*|OLLAMA_*|QDRANT_*|SEMANTIC_*|INDEX_*|MODEL|MODELS_DIR|HF_ENDPOINT|HF_HUB_*|PYTORCH_MPS_*) ;;
            *) continue ;;
        esac
        case "${_preset}" in
            *" ${_k} "*) continue ;;
        esac
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
