import io
NL = chr(10)
p = 'chatbot.py'; lines = io.open(p, encoding='utf-8').read().split(NL)
log = []
# ── A1: 空文档按 0 块正常结算，不再拖垮整轮 ──
i = next(i for i, l in enumerate(lines) if '分块后无有效内容' in l and 'raise ValueError' in l)
assert '_save_doc_chunks' in NL.join(lines[max(0,i-30):i]), '确认 _save_doc_chunks 在作用域内'
ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
new1 = [
    ind + '# 空文档（0 字节/纯空白）是**合法边界**，不是本轮失败：按 0 块正常结算并留痕。' ,
    ind + '# 旧写法在此 raise -> failures 短路 -> 整轮拒绝写 chunks.json：一篇空文件就能让',
    ind + '# 其余文档的增改全都不入库，且 process 永远退出 1（卡死式静默）。',
    ind + 'log(f"    ⚠️  文档 {source_name} 分块后无有效内容（0 块）：按空文档结算，不影响其余文档")',
    ind + '_save_doc_chunks(source_name, sha, fingerprint, [])',
    ind + 'continue',
]
lines[i:i+1] = new1; log.append('A1 空文档：raise -> 0 块正常结算')
# ── A2: build_index 加载产物后校验记录键集（改 RAG_SOURCE_FIELD 的静默路径）──
j = next(i for i, l in enumerate(lines) if l.strip() == 'all_chunks = json.load(f)')
ind2 = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
new2 = [
    ind2 + '# ★ 记录键集必须与当前契约一致：改 RAG_SOURCE_FIELD 后只跑 index（没跑 process）时，',
    ind2 + '# 切分指纹与源 sha 都不变、stages 四段全绿，而旧键读不到 -> payload 来源全空、',
    ind2 + '# 来源过滤恒零命中（静默）。这里 fail-closed 并指名要重跑 process。',
    ind2 + 'if all_chunks and set(all_chunks[0]) != _RECORD_FIELDS:',
    ind2 + '    _miss = sorted(_RECORD_FIELDS - set(all_chunks[0]))',
    ind2 + '    _extra = sorted(set(all_chunks[0]) - _RECORD_FIELDS)',
    ind2 + '    raise RuntimeError(f"{CHUNKS_JSON} 的记录键集与当前契约不一致（多 {_extra}、少 {_miss}）——通常是改了 RAG_SOURCE_FIELD 等字段配置，请重跑 process 后再 index")',
]
lines[j+1:j+1] = new2; log.append('A2 build_index：补记录键集 fail-closed 校验')
io.open(p, 'w', encoding='utf-8').write(NL.join(lines))
print(NL.join(log))