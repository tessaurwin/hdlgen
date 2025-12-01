import re, json, argparse, random, sys
from pathlib import Path

# 1) Match module header that may span lines until semicolon
HEADER_RE = re.compile(r"module\s+\w+\s*\([^;]*\);\s*", re.IGNORECASE | re.MULTILINE)

def find_headers(text):
    headers = []
    for m in HEADER_RE.finditer(text):
        # join whitespace so yosys/verilator see a clean header
        headers.append((m.start(), m.end(), " ".join(m.group(0).split())))
    return headers

def make_line_starts(lines):
    starts, pos = [], 0
    for L in lines:
        starts.append(pos)
        pos += len(L) + 1  # '\n'
    return starts

def offset_to_lineno(starts, off):
    # binary-ish search
    lo, hi = 0, len(starts)-1
    if off <= starts[0]:
        return 0
    while lo < hi:
        mid = (lo + hi + 1)//2
        if starts[mid] <= off:
            lo = mid
        else:
            hi = mid-1
    return lo

INSTR_JSON_RE = re.compile(r'^\s*"instruction"\s*:\s*"(.*)"\s*,?\s*$')

def find_instruction_above(lines, start_idx, max_lookback=60):
    """
    Walk upward to find a JSON-ish line like:  "instruction": "...."
    If not found, fallback to nearest non-empty non-module line.
    """
    fallback = ""
    for i in range(start_idx-1, max(start_idx-max_lookback, -1), -1):
        raw = lines[i].rstrip("\n")
        if not raw.strip():
            continue
        m = INSTR_JSON_RE.match(raw)
        if m:
            # grab content up to end of line; escape condensed whitespace
            return re.sub(r"\s+", " ", m.group(1)).strip()
        if not fallback and not raw.lstrip().lower().startswith("module "):
            fallback = re.sub(r"\s+", " ", raw).strip()
    return fallback

def parse_pairs(text):
    lines = text.splitlines()
    headers = find_headers(text)
    if not headers:
        return []

    starts = make_line_starts(lines)
    pairs = []
    for (h_s, _h_e, header) in headers:
        line_idx = offset_to_lineno(starts, h_s)
        instr = find_instruction_above(lines, line_idx)
        if instr and header:
            pairs.append({"instruction": instr, "module_declaration": header})
    return pairs

def to_tasks(rows):
    tasks = []
    for i, r in enumerate(rows, 1):
        tasks.append({
            "id": f"task{i:03d}",
            "prompt": r["instruction"],
            "module_decl": r["module_declaration"],
            "top": "top_module",
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
    rows = parse_pairs(raw)

    if args.limit and len(rows) > args.limit:
        rows = rows[:args.limit]

    if not rows:
        print("No usable pairs parsed. Check your text formatting.", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    idx = list(range(len(rows)))
    random.shuffle(idx)
    train_n = min(args.train_n, len(rows))
    train_rows = [rows[i] for i in idx[:train_n]]
    test_rows  = [rows[i] for i in idx[train_n:]]

    print(f"Parsed usable pairs: {len(rows)} → train={len(train_rows)} test={len(test_rows)}")

    if args.dryrun:
        print("\n[TRAIN sample]")
        for r in train_rows[:2]:
            print("  instruction:", r["instruction"][:100])
            print("  module_decl:", r["module_declaration"])
        print("\n[TEST sample]")
        for r in test_rows[:2]:
            print("  instruction:", r["instruction"][:100])
            print("  module_decl:", r["module_declaration"])
        return

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/"hdl_100.json").write_text(json.dumps(rows, indent=2))
    (outdir/"hdl_train.json").write_text(json.dumps(train_rows, indent=2))
    (outdir/"hdl_test.json").write_text(json.dumps(test_rows, indent=2))
    (outdir/"tasks_test.json").write_text(json.dumps(to_tasks(test_rows), indent=2))
    print(f"Wrote: {outdir/'hdl_100.json'}, {outdir/'hdl_train.json'}, {outdir/'hdl_test.json'}, {outdir/'tasks_test.json'}")

if __name__ == "__main__":
    main()

