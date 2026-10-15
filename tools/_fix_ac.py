import io
p = "chatbot.py"
lines = io.open(p, encoding="utf-8").read().split("\n")
start = next(i for i, l in enumerate(lines) if "alias_re = re.compile(" in l)
end = next(i for i, l in enumerate(lines) if i > start and l.strip() == "alias_re = None")
bad = "\n".join(lines[start:end + 1])
assert "Automaton" in bad, "起点不是预期的坏块: " + bad[:160]
block = [
    "    alias_re = None",
    "    if surfaces:",
    "        from ahocorasick import Automaton",
    "        alias_re = Automaton()",
    "        for _surface in sorted(set(surfaces)):",
    "            alias_re.add_word(_surface, _surface)",
    "        alias_re.make_automaton()",
]
lines[start:end + 1] = block
io.open(p, "w", encoding="utf-8").write("\n".join(lines))
print("已修复 alias_re 块（AC 自动机 + 默认 None）")
