# src/json_escape_in_strings.py
from pathlib import Path
import json

SRC = Path("data/hdl_modules.json")
DST = Path("data/hdl_modules.clean.json")

s = SRC.read_text(encoding="utf-8")

out = []
in_str = False
esc = False
for ch in s:
    if in_str:
        if esc:
            out.append(ch)
            esc = False
        else:
            if ch == '\\':
                out.append(ch)
                esc = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch == '\n':
                out.append('\\n')
            elif ch == '\t':
                out.append('\\t')
            elif ch == '\r':
                out.append('\\r')
            else:
                out.append(ch)
    else:
        out.append(ch)
        if ch == '"':
            in_str = True

fixed = ''.join(out)

# validate and pretty-write
data = json.loads(fixed)  # will throw if still invalid
DST.write_text(json.dumps(data, ensure_ascii=False, indent=2))
print(f"Wrote {DST} with {len(data)} items")
