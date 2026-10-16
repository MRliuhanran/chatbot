import sqlite3, os, json, re
db = os.path.expanduser("~/.local/share/opencode/opencode.db")
con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
rows = con.execute("select data from part where data like '%modelscope%' and (data like '%studios/%' or data like '%/logs%' or data like '%openapi%') order by rowid desc limit 400")
seen, out = set(), []
for (d,) in rows:
    try:
        j = json.loads(d)
    except Exception:
        continue
    s = json.dumps(j, ensure_ascii=False)
    for m in re.findall(r"https://[a-zA-Z0-9./_%?=&{}$:-]*modelscope[a-zA-Z0-9./_%?=&{}$:-]*", s):
        if m not in seen:
            seen.add(m); out.append(m)
print("会话里出现过的 modelscope 接口（去重）：")
for u in out[:25]:
    print("  " + u)
