# src/convert_prompts_fix.py
from pathlib import Path
import re, json, random

DATA_DIR = Path("data")
INFILE = DATA_DIR / "hdl_prompts.txt"

def read_raw():
    return INFILE.read_text(errors="ignore")

def objects_from_text(raw: str):
    return re.findall(r"\{[\s\S]*?\}", raw)

def extract(pattern, text):
    m = re.search(pattern, text, flags=re.M)
    return (m.group(1) if m else "").strip()

def clean_decl(decl: str) -> str:
    decl = decl.strip().strip('"').strip("`")
    lines = [ln.strip() for ln in decl.splitlines() if ln.strip()]
    if lines:
        s = " ".join(lines)
        s = re.sub(r"\s+([\),;])", r"\1", s)
        s = re.sub(r",\s*", ", ", s)
    else:
        s = decl
    if "module" in s and "(" in s and not s.endswith(";"):
        s += ";"
    return s

def ensure_module_block(code: str) -> str:
    code = (code or "").strip().strip('"')
    if not code:
        return ""
    mm = re.search(r"(module\b[\s\S]*?endmodule)", code, flags=re.I)
    block = mm.group(1).strip() if mm else code
    if not re.search(r"\bendmodule\b", block):
        block = block.rstrip() + "\nendmodule"
    block = "\n".join(ln.rstrip() for ln in block.splitlines())
    return block

def parse_entries(raw: str):
    items = []
    for blk in objects_from_text(raw):
        id_ = extract(r'"id"\s*:\s*"([^"]+)"', blk)

        instr = extract(r'"instruction"\s*:\s*"([\s\S]*?)"\s*,\s*"(?:module_declaration|output)"', blk)
        if not instr:
            instr = extract(r'"instruction"\s*:\s*"([\s\S]*?)"', blk)

        decl  = extract(r'"module_declaration"\s*:\s*"\s*([\s\S]*?)"\s*,\s*"?output"?', blk)
        out   = extract(r'"output"\s*:\s*"\s*([\s\S]*?)"\s*[,}]', blk)

        decl  = clean_decl(decl)
        out   = ensure_module_block(out)

        if id_ or instr or decl or out:
            items.append({
                "id": id_ or f"item_{len(items)+1:03d}",
                "instruction": instr.strip(),
                "module_declaration": decl,
                "output": out
            })
    # dedupe by instruction+decl
    seen, uniq = set(), []
    for e in items:
        key = (e["instruction"], e["module_declaration"])
        if key not in seen:
            uniq.append(e); seen.add(key)
    # stable-ish sort
    def _k(x):
        m = re.match(r"(\D*)(\d+)$", x["id"])
        return (m.group(1), int(m.group(2))) if m else (x["id"], 0)
    uniq.sort(key=_k)
    return uniq

def to_tasks(rows):
    tasks = []
    for i, e in enumerate(rows, 1):
        prompt = e["instruction"].strip()
        decl   = e["module_declaration"].strip()
        if not prompt or not decl:
            continue
        m = re.search(r"module\s+(\w+)\s*\(", decl)
        top = m.group(1) if m else "top_module"
        tasks.append({"id": f"task{i:03d}", "prompt": prompt, "module_decl": decl, "top": top})
    return tasks

def main():
    raw = read_raw()
    all_rows = parse_entries(raw)

    supervised = [e for e in all_rows if e["output"].strip()]
    TOTAL, TRAIN_N, TEST_N = 100, 80, 20

    use_rows = all_rows[:TOTAL] if len(all_rows) >= TOTAL else all_rows[:]
    random.seed(42)
    shuffled = use_rows[:]; random.shuffle(shuffled)
    train = shuffled[:TRAIN_N] if len(shuffled) >= TRAIN_N else shuffled[:]
    test  = shuffled[TRAIN_N:TRAIN_N+TEST_N] if len(shuffled) >= TRAIN_N else []

    train_ids = set(e["id"] for e in train)
    train_supervised = [e for e in supervised if e["id"] in train_ids]

    tasks_test = to_tasks(test)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR/"hdl_100_clean.json").write_text(json.dumps(use_rows, indent=2))
    (DATA_DIR/"hdl_train.json").write_text(json.dumps(train, indent=2))
    (DATA_DIR/"hdl_test.json").write_text(json.dumps(test, indent=2))
    (DATA_DIR/"hdl_train_supervised.json").write_text(json.dumps(train_supervised, indent=2))
    (DATA_DIR/"tasks_test.json").write_text(json.dumps(tasks_test, indent=2))

    print("Parsed summary:")
    print("  total objects in file   :", len(re.findall(r'^\s*\{', raw, flags=re.M)))
    print("  entries parsed (unique) :", len(all_rows))
    print("  with outputs            :", len(supervised))
    print("  kept for total          :", len(use_rows))
    print("  train / test            :", len(train), "/", len(test))
    print("  train_supervised        :", len(train_supervised))
    print("  tasks_test              :", len(tasks_test))

if __name__ == "__main__":
    main()

