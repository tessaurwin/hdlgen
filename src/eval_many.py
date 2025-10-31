# src/eval_many.py
import json, csv, pathlib
from datetime import datetime
from generate import generate
from tools import verible_lint, verilator_compile, yosys_synth

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_V = ROOT / "outputs" / "v"
OUT_LOG = ROOT / "outputs" / "logs"
OUT_RES = ROOT / "results"
DATA = ROOT / "data" / "tasks.jsonl"

for p in (OUT_V, OUT_LOG, OUT_RES):
    p.mkdir(parents=True, exist_ok=True)

def eval_one(task: dict, model="tinyllama"):
    tid, top, prompt = task["id"], task["top"], task["prompt"]
    vpath = OUT_V / f"{tid}.v"

    code = generate(prompt, top, model=model)
    vpath.write_text(code)

    v_ok, v_warns, v_log = verible_lint(vpath)
    c_ok, c_log = verilator_compile(vpath)
    y_ok, y_log = yosys_synth(vpath, top)

    (OUT_LOG / f"{tid}.verible.log").write_text(v_log)
    (OUT_LOG / f"{tid}.verilator.log").write_text(c_log)
    (OUT_LOG / f"{tid}.yosys.log").write_text(y_log)

    return {
        "task_id": tid,
        "top": top,
        "verible_ok": int(v_ok),
        "lint_warnings": v_warns,
        "verilator_ok": int(c_ok),
        "yosys_ok": int(y_ok),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }

if __name__ == "__main__":
    tasks = json.loads(DATA.read_text())
    rows = [eval_one(t) for t in tasks]

    out_csv = OUT_RES / "baseline.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"✅ wrote {out_csv}")
    print("quick summary:")
    print(f"  verible pass:  {sum(r['verible_ok'] for r in rows)}/{len(rows)}")
    print(f"  verilator ok:  {sum(r['verilator_ok'] for r in rows)}/{len(rows)}")
    print(f"  yosys ok:      {sum(r['yosys_ok'] for r in rows)}/{len(rows)}")
