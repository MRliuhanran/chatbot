# ── 单容器 ModelScope 镜像：预构建 base（重/不常变）+ 应用与可变数据层（轻/常变）──
# base 镜像 tag：
#   mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/independent-service:366944-7e8f3a80-2026-10-28-22-16-27
#
# 分层定案（2026-10-31，指纹事故后的根治结构）：
#   · base 只承载"重且不常变"的东西：ollama/qdrant 二进制、python 依赖、双检索模型权重、
#     生成模型 GGUF。要换这些才重烤一次 base（或换 BASE_IMAGE），日常发布不重跑重层。
#   · 可变数据（storage/ 索引、corpus/、lexicon/、qdrant_config/、artifacts/index_manifest.json
#     与同代的 artifacts/chunks.meta.json）与**同一份**应用文件（.env/start.sh/chatbot.py/池化
#     元文件）在应用层一起 COPY：仓库即唯一事实源，数据不可能再落后于应用。
#   · 构建期最后一步 `chatbot.py verify-build` fail-fast：清单指纹 ≠ 容器内现算指纹就构建失败。
#
# 事故由来（2026-10-31 远程 ModelScope 实证）：base（提交 7e8f3a8，10-28）烤进旧
# storage/artifacts/corpus/lexicon；应用层只换 .env/start.sh/chatbot.py/1_Pooling，数据永远
# 停在 base 的旧快照。本地 10-31 加入 models/bge-m3/1_Pooling/config.json 后切分配置指纹变化，
# 远程启动期分块指纹 索引=923236fe7e4acc7c / 当前=3502968fe6f7039e 不一致，检索闸门 fail-closed。
ARG BASE_IMAGE=mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/independent-service:366944-7e8f3a80-2026-10-28-22-16-27
FROM ${BASE_IMAGE}

WORKDIR /app
COPY .env /app/.env
COPY start.sh /app/start.sh
COPY chatbot.py /app/chatbot.py
# base 烤入层（10-28）的 bge-m3 缺 ST 池化元文件——modules.json 要求 1_Pooling/config.json，
# 缺了 infinity 的 ST 引擎把模型路径当 HF repo id → HFValidationError，rc=3 启动即崩
# （2026-10-31 本地实证）。应用层补 191B；将来重层换 base 时此覆盖幂等可留。
COPY models/bge-m3/1_Pooling/config.json /app/models/bge-m3/1_Pooling/config.json

# http 后端依赖，一句话三块：
#   ① [torch,server]=infinity 引擎+HTTP 面板（typer<0.13 钉死 → 配套 click==8.1.8 防
#      未来 base 重建时 pip 现解 newer click 与 typer 0.12.5 CLI 解析不兼容
#      TypeError "Secondary flag is not valid for non-boolean flag"，2026-10-31 cage 实证）
#   ② optimum==1.22.0：bge-m3 走 torch 引擎时 EngineArgs.bettertransformer 默认 True，
#      acceleration.py:46 解引用 BetterTransformerManager；该符号属 optimum 可选 extra
#      （不在 [torch,server] 里），容器缺它 → NameError rc=3 启动即崩 + 30s 循环
#      （2026-10-31 cage 全量 traceback 实证：transformer/embedder/sentence_transformer.py:62
#      → acceleration.py:46）。1.22.0 = 本地实证组合的同版本。
#   ③ psutil：内存熔断（RAG_MEM_GUARD=1）的水位口径；torch==2.7.1 与 base 同版本锁死
#      防 pip 现解漂移。版本 = 本地实测组合 0.0.76。
# ⚠️ 这一行是单行 exec-form JSON 数组，**不能跨行续写**：魔搭 dockerfile parser 会把续行当成
#    独立指令（2026-10-31 23:27 场 parse error line 20 实证）。以下所有 RUN 同此约束。
RUN ["pip", "install", "--no-cache-dir", "-i", "https://mirrors.aliyun.com/pypi/simple/", "infinity-emb[torch,server]==0.0.76", "psutil==7.0.0", "torch==2.7.1", "optimum==1.22.0", "click==8.1.8"]

# ── 可变数据层（必须放在 pip RUN 之后：改数据只失效本层+verify 层，不重跑 16min 级依赖层）──
# 先删 base 里的旧快照再 COPY：COPY 目录是 merge 语义（只覆盖同名文件，不删除源端没有的文件）。
# 不删的后果：base 旧 collection 的 segment/raft 状态与新 collection 混在同一个 /app/storage
# （同一集合目录名合并后可能读到旧 segment），已从语料删掉的 txt 也会留在 /app/corpus，
# 于是容器内现算的语料/分块指纹与 manifest 永远对不上——正是 923/350 事故的形态。
RUN ["rm", "-rf", "/app/storage", "/app/corpus", "/app/lexicon"]
COPY storage /app/storage
COPY corpus /app/corpus
COPY lexicon /app/lexicon
COPY qdrant_config /app/qdrant_config
# 同代 chunks.meta.json（98KB，git 白名单内）：verify-build 的 E 项在 /app/artifacts/chunks.meta.json
# 存在时要求其 fingerprint == manifest.chunk_fingerprint。**必须 COPY**——base（10-28 快照）里
# 那份旧 meta 的分块指纹是 923…，不覆盖就会让"两代产物混进同一镜像"的检查假失败；同时让镜像内
# 排障能看到与索引同代的 meta。chunks.json（64MB）仍不进镜像，它只服务本地重建，verify-build 不读。
# 注意：只覆盖这两个小文件，/app/artifacts 不做 rm —— base 的 gen_tokenizer 等仍要复用。
COPY artifacts/chunks.meta.json /app/artifacts/chunks.meta.json
# index 成功收尾写的发布清单：chunk_fingerprint/lexicon_id/集合名。与 storage 同一次 index 的
# 产物，必须一起进镜像（.gitignore 已放行、.dockerignore 未排除）。
COPY artifacts/index_manifest.json /app/artifacts/index_manifest.json
# 构建期 fail-fast：manifest.chunk_fingerprint 必须等于容器内用同一份 corpus/切分配置现算的
# 指纹；不等 = 本地忘跑 process/index 或拿了旧数据 → 直接构建失败，而不是上线后才由检索闸门
# 拒服务（2026-10-31 远程 索引=923…/当前=350… 事故）。同一命令也是发布脚本 preflight 的第一步。
RUN ["python", "chatbot.py", "verify-build"]

EXPOSE 7860
ENTRYPOINT ["python","chatbot.py","all"]
