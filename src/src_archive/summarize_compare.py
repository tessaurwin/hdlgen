# src/summarize_compare.py
import csv, argparse, pathlib, re
from collections import Counter, defaultdict

def first_line(s: str) -> str:
    s = s.strip().splitlines()
    return s[0] if s else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/compare_tb_summary.txt")
    args = ap.parse_args()

    path = pathlib.Path(args.csv)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    counts = Counter(r["status"] for r in rows)
    by_status = defaultdict(list)
    for r in rows:
        by_status[r["status"]].append(r)

    # terminal one-liner
    total = len(rows)
    one = f"TB Summary: PASS {counts.get('PASS',0)} | MISMATCH {counts.get('MISMATCH',0)} | COMPILE_FAIL {counts.get('COMPILE_FAIL',0)} | UNKNOWN {counts.get('UNKNOWN',0)} | NO_SIM {counts.get('NO_SIM',0)} | TOTAL {total}"
    print(one)

    # txt report
    out = []
    out.append(one)
    out.append("")
    if by_status["PASS"]:
        out.append("== PASS ==")
        out.append(", ".join(r["task"] for r in by_status["PASS"]))
        out.append("")
    if by_status["MISMATCH"]:
        out.append("== MISMATCH (first error lines) ==")
        for r in by_status["MISMATCH"]:
            out.append(f"- {r['task']}: {first_line(r['detail'])}")
        out.append("")
    if by_status["COMPILE_FAIL"]:
        out.append("== COMPILE_FAIL (first error lines) ==")
        for r in by_status["COMPILE_FAIL"]:
            out.append(f"- {r['task']}: {first_line(r['detail']).replace('iverilog: ','')}")
        out.append("")
    if by_status["UNKNOWN"] or by_status["NO_SIM"]:
        out.append("== OTHER ==")
        for k in ("UNKNOWN","NO_SIM"):
            for r in by_status[k]:
                out.append(f"- {r['task']}: {first_line(r['detail'])}")
        out.append("")

    outpath = pathlib.Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {outpath}")

if __name__ == "__main__":
    main()