# scripts/convert_prompts_to_json.py
import json, re, argparse
from pathlib import Path

def parse_blocks(text: str):
    items = []
    # Strategy A: "### Instruction / ### Module declaration / ### Verilog solution"
    parts = re.split(r"(?i)^\s*###\s*Instruction:\s*", text, flags=re.M)
    for part in parts:
        if not part.strip():
            continue
        m_decl = re.search(r"(?i)^\s*###\s*Module\s+declaration:\s*(.*?)\s*(?=^\s*###|\Z)", part, flags=re.S|re.M)
        m_out  = re.search(r"(?i)^\s*###\s*Verilog\s+solution:\s*(.*?)\s*(?=^\s*###|\Z)", part, flags=re.S|re.M)
        m_instr = re.split(r"(?i)^\s*###\s*(Module\s+declaration|Verilog\s+solution)\s*:\s*", part, maxsplit=1, flags=re.M)
        instruction = m_instr[0].strip() if m_instr else ""
        module_decl = (m_decl.group(1).strip() if m_decl else "")
        output = (m_out.group(1).strip() if m_out else "")
        if module_decl and output:
            items.append({"instruction": instruction, "module_declaration": module_decl, "output": output})
    if items:
        return items

    # Strategy B: fallback — grab module blocks & best-effort instruction
    for m in re.finditer(r"(module\s+\w+\s*\([^;]*\);\s*[\s\S]*?endmodule)", text, flags=re.I):
        block = m.group(1).strip()
        m_hdr = re.match(r"(module\s+\w+\s*\([^;]*\);)", block, flags=re.I)
        module_decl = m_hdr.group(1) if m_hdr else ""
        before = text[:m.start()]
        # nearest "Instruction:" line above, else previous non-empty line
        instr_line = ""
        m_up = list(re.finditer(r"(?i)^\s*(?:###\s*)?Instruction:\s*(.+)$", before, flags=re.M))
        if m_up:
            instr_line = m_up[-1].group(1).strip()
        else:
            lines_before = [ln.strip() for ln in before.splitlines() if ln.strip()]
            instr_line = lines_before[-1] if lines_before else ""
        items.append({"instruction": instr_line, "module_declaration": module_decl, "output": block})
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",  required=True, help="raw text file with 100 prompt+module pairs")
    ap.add_argument("--outdir",  default="data", help="where to write JSON outputs")
    ap.add_argument("--limit",   type=int, default=100, help="cap examples (default 100)")
    ap.add_argument("--train_n", type=int, default=80,  help="train size (rest is test)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    raw = Path(args.infile).read_text(errors="ignore")

    items = parse_blocks(raw)
    if not items:
        raise SystemExit("Could not parse any (instruction, module_declaration, output) triples.")

    # dedupe + cap
    seen, cleaned = set(), []
    for it in items:
        key = (it["instruction"], it["module_declaration"], it["output"])
        if key in seen: 
            continue
        seen.add(key)
        cleaned.append(it)
    items = cleaned[:args.limit]

    # write full array
    (outdir/"hdl_100.json").write_text(json.dumps(items, indent=2))

    # split
    train_n = min(args.train_n, len(items))
    train = items[:train_n]
    test  = items[train_n:]
    (outdir/"hdl_train.json").write_text(json.dumps(train, indent=2))
    (outdir/"hdl_test.json").write_text(json.dumps(test, indent=2))

    # tasks for eval script
    tasks = []
    for i, ex in enumerate(test, start=1):
        m = re.search(r"module\s+(\w+)\s*\(", ex["module_declaration"] or "", flags=re.I)
        top = m.group(1) if m else "top_module"
        tasks.append({
            "id": f"task{i:03d}",
            "prompt": ex["instruction"],
            "module_decl": ex["module_declaration"],
            "top": top
        })
    (outdir/"tasks_test.json").write_text(json.dumps(tasks, indent=2))

    print(f"Parsed: {len(items)} total → Train={len(train)} Test={len(test)}")
    print(f"Wrote: {outdir/'hdl_100.json'}, {outdir/'hdl_train.json'}, {outdir/'hdl_test.json'}, {outdir/'tasks_test.json'}")

if __name__ == "__main__":
    main()

