# src/convert_prompts_txt_to_json.py
import re, json, argparse, random
from pathlib import Path

PAIR_RE = re.compile(
    r"""
    (?P<instr_tag>^\s*(Instruction|Prompt)\s*:\s*)    # "Instruction:" or "Prompt:"
    (?P<instruction>.+?)                              # instruction text (lazy)
    (?:\n+|\r+\n*)
    \s*(Module\s+declaration|Module)\s*:\s*           # "Module declaration:" or "Module:"
    (?P<module>                                      # capture the header (and optional body)
        (?:.|\n)+?                                   # up to...
    )
    (?=^\s*(Instruction|Prompt)\s*:| \Z)             # ...next "Instruction:" or EOF
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE
)

# Accept either a single-line "module top_module(...);" or a header followed by ports across lines.
HEADER_RE = re.compile(r"module\s+\w+\s*\([^;]*\);\s*", re.IGNORECASE | re.MULTILINE)

def extract_pairs(text: str):
    rows = []
    for m in PAIR_RE.finditer(text):
        instr = m.group("instruction").strip()
        module_block = m.group("module").strip()

        # Try to isolate just the signature line (module ... (...);)
        # If there's an endmodule body in the text, we still only keep the signature.
        header_match = HEADER_RE.search(module_block)
        if not header_match:
            # Try to patch: if user forgot semicolon at end, add it after the right paren.
            fix = re.search(r"(module\s+\w+\s*\([^)]*\))", module_block, re.I | re.M)
            if fix:
                sig = fix.group(1).strip()
                if not sig.endswith(";"):
                    sig += ";"
                module_decl = sig
            else:
                module_decl = ""
        else:
            module_decl = header_match.group(0).strip()

        # Clean goofy quotes or whitespace
        instr = re.sub(r'\s+', ' ', instr).strip()
        module_decl = re.sub(r'\s+', ' ', module_decl).strip()

        if instr and module_decl:
            rows.append({"instruction": instr, "module_declaration": module_decl})
    return rows

def to_tasks(test_rows):
    tasks = []
    for i, r in enumerate(test_rows, 1):
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
    ap.add_argument("--limit", type=int, default=0, help="Cap total pairs (0 = all)")
    ap.add_argument("--train_n", type=int, default=80, help="How many go to train (rest to test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.infile).read_text(errors="ignore")
    rows = extract_pairs(raw)

    if args.limit and len(rows) > args.limit:
        rows = rows[:args.limit]

    # Basic validation
    good = []
    for r in rows:
        if not r["instruction"] or not r["module_declaration"]:
            continue
        if not HEADER_RE.search(r["module_declaration"]):
            continue
        good.append(r)

    total = len(rows)
    usable = len(good)
    if usable == 0:
        raise SystemExit("No usable pairs parsed. Check your text formatting.")

    # Split
    random.seed(args.seed)
    indices = list(range(usable))
    random.shuffle(indices)
    train_n = min(args.train_n, usable)
    train_rows = [good[i] for i in indices[:train_n]]
    test_rows  = [good[i] for i in indices[train_n:]]

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Full set (for reference)
    Path(outdir/"hdl_100.json").write_text(json.dumps(good, indent=2))
    # Train/Test
    Path(outdir/"hdl_train.json").write_text(json.dumps(train_rows, indent=2))
    Path(outdir/"hdl_test.json").write_text(json.dumps(test_rows, indent=2))
    # Test in eval format (your runner expects this)
    Path(outdir/"tasks_test.json").write_text(json.dumps(to_tasks(test_rows), indent=2))

    print(f"Parsed: {total} total → Usable={usable} ({len(train_rows)} train, {len(test_rows)} test)")
    print(f"Wrote: {outdir/'hdl_100.json'}, {outdir/'hdl_train.json'}, {outdir/'hdl_test.json'}, {outdir/'tasks_test.json'}")

if __name__ == "__main__":
    main()

