# src/fix_hdl_modules.py
from pathlib import Path
import json

SRC = Path("data/hdl_modules.json")              # your current messy file
DST = Path("data/hdl_modules.clean.json")        # cleaned JSON array will be written here


def escape_in_strings(s: str) -> str:
    """
    Escape bare newlines/tabs/CR inside quoted strings only.
    We scan char-by-char and only replace when we're inside quotes and not in an escape.
    """
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
    return ''.join(out)


def extract_top_level_objects(text: str):
    """
    Walk the file and extract every top-level {...} object by brace depth.
    This works even if commas between objects are missing or extra.
    """
    objs = []
    depth = 0
    start = None
    in_str = False
    esc = False

    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            else:
                if ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i+1])
                    start = None
    return objs


def main():
    raw = SRC.read_text(encoding="utf-8", errors="ignore")

    # 1) Extract each top-level JSON object by brace matching
    chunks = extract_top_level_objects(raw)
    if not chunks:
        raise SystemExit("No JSON objects found. Check the source file path/contents.")

    cleaned = []
    bad = 0
    for idx, obj in enumerate(chunks, 1):
        candidate = obj
        # Try raw first
        try:
            cleaned.append(json.loads(candidate))
            continue
        except Exception:
            pass
        # Then try escaping newlines/tabs in strings
        candidate2 = escape_in_strings(candidate)
        try:
            cleaned.append(json.loads(candidate2))
            continue
        except Exception as e:
            bad += 1
            # Uncomment for debugging specific failures:
            # print(f"[{idx}] parse error: {e}")

    if not cleaned:
        raise SystemExit("Failed to parse any objects after fixing.")

    # Keep only entries with your required fields (and non-empty instruction/output)
    final = []
    for ex in cleaned:
        instr = (ex.get("instruction") or "").strip()
        decl  = (ex.get("module_declaration") or "").strip()
        out   = (ex.get("output") or "").strip()
        if not instr or not out:
            continue
        final.append({
            "id": str(ex.get("id", "")),
            "instruction": instr,
            "module_declaration": decl,
            "output": out
        })

    DST.write_text(json.dumps(final, ensure_ascii=False, indent=2))
    print(f"Wrote {DST} with {len(final)} items (skipped {bad} broken objects).")
        
            
if __name__ == "__main__":
    main()  
        

