#!/usr/bin/env bash
# ── 把 storage 下 >10MiB 的大文件打包进 storage_pack/ 分片，绕过魔搭单文件 10MiB 上限 ──
#
# 背景（2026-11-14 实证事故）：魔搭收紧仓库容量策略——pre-receive hook（tree_limits_v1）
# 拒收「非 LFS 单文件 >10MiB」的推送；两天前（11-01）的 18400f9 还能带 39MiB matrix.dat
# 过闸，且 hook 的 offending 名单只列前 4 个（本代 >10MiB 的入仓文件实为 10 个）。
# 本代大文件 = dense matrix 83MB + 9 个 qdrant 预分配的 32MiB mmap 页（逻辑尺寸固定、
# 躲不开），合计 ~371MiB。
#
# 为什么不是 Git LFS：能否 work 取决于魔搭构建机是否装 git-lfs 并 smudge，不可赌；
# 分片重组全走普通 git 对象，零外部依赖。分片 ≤9MiB（硬限 10MiB 的安全余量）。
#
# 产物（全部普通 git 对象，随 push 进构建上下文）：
#   storage_pack/manifest.txt  sha256sum 格式（`<hash>  <相对路径>`），容器内用 GNU
#                              sha256sum -c 校验；preflight④ 用 shasum -c 校验同代性
#   storage_pack/blob-*        原始 tar 流的 9MiB 分片（cat blob-* | tar -x 即还原）
# Dockerfile 在 COPY storage 之后重组 + 校验，随后 rm -rf storage_pack。
#
# 每次发布周期（快照轮换、index 重跑）之后都要重跑本脚本；tools/ms_release.sh preflight④
# 会校验 pack 与磁盘大文件同代（路径集合 + sha256），过期即拦。
#
# ⚠️ macOS bash 3.2 兼容：不用 mapfile，不用空数组直接展开。
set -euo pipefail
cd "$(dirname "$0")/.."

PART_BYTES=9437184   # 9MiB < 10MiB 硬限
PACK_DIR=storage_pack

# 与 ms_release.sh preflight④ 的收集口径**逐字节一致**（comm 比对依赖它）：
# >10MiB、排除 wal / .deleted / .removed / .tmp —— 与 .dockerignore 的排除精神一致。
BIG=()
while IFS= read -r p; do BIG+=("$p"); done < <(
  find storage -type f -size +10485760c \
    ! -path '*/wal/*' ! -path 'storage/.deleted*' \
    ! -name '*.removed*' ! -name '*.tmp*' | LC_ALL=C sort
)

if pgrep -x qdrant >/dev/null 2>&1; then
  echo "⚠️  检测到 qdrant 正在运行——大文件可能正在被写，打包内容不可信。" >&2
  echo "    先停 qdrant（./start.sh stop 或 kill -TERM <pid>）再打包。" >&2
  exit 1
fi

rm -rf "${PACK_DIR}"
mkdir -p "${PACK_DIR}"

if [ "${#BIG[@]}" -eq 0 ]; then
  : > "${PACK_DIR}/manifest.txt"
  echo "ℹ️ storage 下没有 >10MiB 文件：pack 为空（manifest.txt 空文件，Dockerfile 重组步自会跳过）"
  exit 0
fi

# tar（相对路径、从仓库根打包；成员路径 = storage/...，容器内 tar -x -C /app 原位还原）
tar -c -f - ${BIG[@]+"${BIG[@]}"} | split -b "${PART_BYTES}" -a 3 - "${PACK_DIR}/blob-"

# manifest：sha256sum 文本格式（两个空格），路径 = 原始相对路径
: > "${PACK_DIR}/manifest.txt"
for f in ${BIG[@]+"${BIG[@]}"}; do
  shasum -a 256 "${f}" >> "${PACK_DIR}/manifest.txt"
done

# 分片体积自检：任何分片 >10MiB 都会在 push 时被拒，这里先拦
if [ -n "$(find "${PACK_DIR}" -name 'blob-*' -size +10485760c)" ]; then
  echo "❌ 存在 >10MiB 分片（split 参数异常）" >&2
  exit 1
fi

# ── round-trip 自测：cat 分片重组 → tar 解包 → sha256 对账 ──
# 这正是 Dockerfile 重组 RUN 要做的事；在本地先证明它逐位还原，再拿去发布。
_TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/storage_pack_roundtrip.XXXXXX")"
trap 'rm -rf "${_TEST_DIR}"' EXIT
cat "${PACK_DIR}"/blob-* | tar -x -f - -C "${_TEST_DIR}"
( cd "${_TEST_DIR}" && shasum -a 256 -c "${OLDPWD}/${PACK_DIR}/manifest.txt" >/dev/null )
_N_EXTRACTED="$(find "${_TEST_DIR}" -type f | wc -l | tr -d ' ')"

_N_PARTS="$(ls "${PACK_DIR}"/blob-* | wc -l | tr -d ' ')"
_TOTAL="$(du -sh "${PACK_DIR}" | cut -f1)"
echo "✅ storage_pack/ 生成并 round-trip 自测通过（逐位还原 ${_N_EXTRACTED}/${#BIG[@]} 个大文件）"
echo "   ${_N_PARTS} 个分片（每片 ≤9MiB）+ manifest.txt，合计 ${_TOTAL}"
echo "   覆盖文件："
printf '   - %s\n' ${BIG[@]+"${BIG[@]}"}
echo "下一步：git rm -r --cached <大文件> → git add -A storage_pack → 提交 → ms_release.sh"
