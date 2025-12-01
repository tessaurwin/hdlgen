#!/usr/bin/env python3
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [ROOT/"data/training_dataset.json", ROOT/"data/hdl_train.json", ROOT/"data/hdl_train.jsonl"]

def read_any(p: Path):
    txt = p.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else None
    except Exception:
        pass
    rows = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): 
            continue
        rows.append(json.loads(ln))
    return rows

def resolve_from(base: Path, rel: str) -> Path:
    cand = base.parent / rel
    if cand.exists():
        return cand
    cand = ROOT / rel
    if cand.exists():
        return cand
    return ROOT / "data" / rel

def main():
    p = next((c for c in CANDIDATES if c.exists()), None)
    if not p:
        print("no dataset at data/training_dataset.json or hdl_train.json[.l]"); sys.exit(1)
    rows = read_any(p)
    if not rows:
        print("empty/invalid dataset"); sys.exit(1)
    ok = True
    for i, r in enumerate(rows, 1):
        miss = [k for k in ("id","instruction") if not r.get(k)]
        if miss:
            ok = False; print(f"[{i}] missing {miss}")
        op = r.get("output_path")
        out = (r.get("output") or "").strip()
        if not out and not op:
            ok = False; print(f"[{i}] needs output or output_path")
        if op:
            vf = resolve_from(p, op)
            if not vf.exists():
                ok = False; print(f"[{i}] output_path not found: {op} (checked {vf})")
    print("OK" if ok else "FAIL"); sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
