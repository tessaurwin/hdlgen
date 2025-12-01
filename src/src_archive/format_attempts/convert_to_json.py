# scripts/convert_to_jsonl.py
import json, re, sys, pathlib

src = pathlib.Path("/mnt/data/hdl prompts (1).txt")  # your uploaded file
dst_jsonl = pathlib.Path("data/hdl_100.jsonl")

def normalize_ws(s: str) -> str:
    # collapse CRLF, strip trailing spaces
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # trim lines but keep structure
    lines = [ln.rstrip() for ln in s.split("\n")]
    return "\n".join(lines).strip()

def fix_quotes(s: str) -> str:
    # convert fancy quotes to plain quotes
    return s.replace("“","\"").replace("”","\"").replace("’","'")

def slice_first_module(s: str) -> str:
    m = re.search(r"(module\b[\s\S]*?endmodule)", s, flags=re.I)
    return m.group(1).strip() if m else s.strip()

def enforce_top_header(header: str, code: str) -> str:
    want = " ".join(header.split()).rstrip()
    if not want.endswith(";"):
        want += ";"
    # replace first module header with `want`
    m = re.search(r"(module\s+\w+\s*\([^;]*\);)", code, flags=re.I)
    if m:
        code = code.replace(m.group(1), want, 1)
    else:
        code = want + "\n" + code
    if "endmodule" not in code:
        code = code.rstrip() + "\nendmodule\n"
    return code

# very tolerant loader: the file looks like a JS/JSON array; try to coerce to JSON
raw = fix_quotes(src.read_text(encoding="utf-8"))
# Quick heuristics: ensure it's a JSON array
blob = raw.strip()
if not blob.startswith("["):
    blob = "[" + blob
if not blob.endswith("]"):
    blob = blob + "]"

# Remove trailing commas before closing braces/brackets
blob = re.sub(r",\s*([}\]])", r"\1", blob)

data = json.loads(blob)  # may raise; if so, inspect the offending area

with dst_jsonl.open("w", encoding="utf-8") as f:
    for item in data:
        ex = {}
        ex["id"] = str(item.get("id","")).strip()
        ex["instruction"] = normalize_ws(item.get("instruction",""))
        ex["module_declaration"] = normalize_ws(item.get("module_declaration",""))
        out = normalize_ws(item.get("output",""))
        out = slice_first_module(out)
        if ex["module_declaration"]:
            out = enforce_top_header(ex["module_declaration"], out)
        ex["output"] = out
        # sanity checks
        assert ex["instruction"], f"missing instruction for id={ex['id']}"
        assert "module" in ex["output"] and "endmodule" in ex["output"], f"bad module for id={ex['id']}"
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {dst_jsonl}")

