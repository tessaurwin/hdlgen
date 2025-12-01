import re, json, argparse, random
from pathlib import Path

# Match one "object" loosely: from { to } with id present
OBJ_RE = re.compile(
    r'\{\s*[^{}]*?"id"\s*:\s*"(?P<id>[^"]+)"[\s\S]*?\}',
    re.MULTILINE
)

# Field extractors (non-greedy, tolerate whitespace/newlines)
INSTR_RE = re.compile(r'"instruction"\s*:\s*"(?P<instr>[\s\S]*?)"\s*,?\s*$', re.MULTILINE)
DECL_RE  = re.compile(r'"module_declaration"\s*:\s*"\s*(?P<decl>module[\s\S]*?\);\s*)"\s*,?', re.IGNORECASE)
OUT_RE   = re.compile(r'"output"\s*:\s*"\s*(?P<out>module[\s\S]*?endmodule)\s*"\s*,?', re.IGNORECASE)

def clean_one_line(s: str) -> str:
    # collapse whitespace/newlines inside quoted instruction
    return re.sub(r'\s+', ' ', s).strip()

def parse_objects(raw: str):
    rows = []
    for m in OBJ_RE.finditer(raw):
        block = m.group(0)  # text of this JSON-ish object
        # instruction (may be single line in the file)
        instr_m = INSTR_RE.search(block)
        decl_m  = DECL_RE.search(block)
        out_m   = OUT_RE.search(block)

        instr = clean_one_line(instr_m.group('instr')) if instr_m else ""
        decl  = " ".join(decl_m.group('decl').split()) if decl_m else ""
        out   = out_m.group('out').strip() if out_m else ""

        # Normalize whitespace in header present inside output too
        if out:
            # normalize header line inside output
            head = re.search(r'(module\s+\w+\s*\([^;]*\);)', out, flags=re.I)
            if head and decl:
                out = out.replace(head.group(1), decl, 1)

        if instr and decl and out:
            rows.append({
                "id": m.group('id'),
                "instruction": instr,
                "module_declaration": decl,
                "output": out
            })
    return rows

def to_tasks(rows):
    tasks = []
    for i, r in enumerate(rows, 1):
        tasks.append({
            "id": f"task{i:03d}",
            "prompt": r["instruction"],
            "module_decl": r["module_declaration"],
            "top": "top_module"
        })
    return tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train_n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dryrun", action="store_true")
    args = ap.parse_args()

    raw = Path(args.infile).read_text(errors="ignore")
    rows = parse_objects(raw)

    if args.limit and len(rows) > args.limit:
        rows = rows[:args.limit]

    if not rows:
        print("No usable rows (need instruction + module_declaration + output).")
        raise SystemExit(1)

    random.seed(args.seed)
    idx = list(range(len(rows)))
    random.shuffle(idx)
    train_n = min(args.train_n, len(rows))
    train_rows = [rows[i] for i in idx[:train_n]]
    test_rows  = [rows[i] for i in idx[train_n:]]

    print(f"Parsed usable rows: {len(rows)} → train={len(train_rows)} test={len(test_rows)}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.dryrun:
        print("\n[SAMPLE TRAIN ROW]")
        print(json.dumps(train_rows[0], indent=2)[:800])
        print("\n[SAMPLE TEST TASK]")
        print(json.dumps(to_tasks(test_rows)[0], indent=2))
        return

    (outdir/"hdl_100.json").write_text(json.dumps(rows, indent=2))
    (outdir/"hdl_train.json").write_text(json.dumps(train_rows, indent=2))
    (outdir/"hdl_test.json").write_text(json.dumps(test_rows, indent=2))
    (outdir/"tasks_test.json").write_text(json.dumps(to_tasks(test_rows), indent=2))
    print(f"Wrote: {outdir/'hdl_100.json'}, {outdir/'hdl_train.json'}, {outdir/'hdl_test.json'}, {outdir/'tasks_test.json'}")

if __name__ == "__main__":
    main()

