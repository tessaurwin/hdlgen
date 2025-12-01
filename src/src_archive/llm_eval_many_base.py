import json, csv, sys, argparse, pathlib, re
from datetime import datetime
from generate_baseline import generate_verilog
from tools import verible_lint, verilator_compile, yosys_synth

ROOT = pathlib.Path(__file__).resolve().parents[1]
def module_from_decl(decl: str) -> str:
    if not decl: return "top_module"
    m = re.search(r"module\s+(\w+)\s*\(", decl, flags=re.I)
    return m.group(1) if m else "top_module"

def eval_one(task, out_vdir: pathlib.Path, out_logdir: pathlib.Path):
    tid  = task["id"]; top = task["top"]
    instr = task["prompt"]; decl = task.get("module_decl","")
    vpath = out_vdir / f"{tid}.v"
    code = generate_verilog(instr, decl); vpath.write_text(code)
    synth_top = module_from_decl(decl)
    v_ok, v_warns, v_log = verible_lint(vpath)
    c_ok, c_log = verilator_compile(vpath)
    y_ok, y_log = yosys_synth(vpath, synth_top)
    (out_logdir / f"{tid}.verible.log").write_text(v_log)
    (out_logdir / f"{tid}.verilator.log").write_text(c_log)
    (out_logdir / f"{tid}.yosys.log").write_text(y_log)
    return {"task_id": tid,"top": top,"verible_ok": int(v_ok),"lint_warnings": v_warns,
            "verilator_ok": int(c_ok),"yosys_ok": int(y_ok),
            "timestamp": datetime.now().isoformat(timespec="seconds")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",  default=str(ROOT/"data"/"tasks_test.json"))
    ap.add_argument("--out",    default=str(ROOT/"results"/"base_vs_test.csv"))
    ap.add_argument("--vdir",   default=str(ROOT/"outputs"/"v_base_test"))
    ap.add_argument("--logdir", default=str(ROOT/"outputs"/"logs_base_test"))
    args = ap.parse_args()
    tasks = json.loads(pathlib.Path(args.tasks).read_text())
    out_csv = pathlib.Path(args.out)
    out_vdir = pathlib.Path(args.vdir); out_vdir.mkdir(parents=True, exist_ok=True)
    out_logdir = pathlib.Path(args.logdir); out_logdir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    for i,t in enumerate(tasks, start=1):
        print(f"→ [{i}/{len(tasks)}] {t.get('id','?')} (top={t.get('top','?')})")
        try: rows.append(eval_one(t, out_vdir, out_logdir))
        except Exception as e: print(f"   ✖ error on {t.get('id','?')}: {e}", file=sys.stderr)
    if not rows: print("\nNo successful rows to summarize."); return
    with out_csv.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    n=len(rows)
    print(f"\nWrote {out_csv}")
    print(f"verible pass:  {sum(r['verible_ok'] for r in rows)}/{n}")
    print(f"verilator ok:  {sum(r['verilator_ok'] for r in rows)}/{n}")
    print(f"yosys ok:      {sum(r['yosys_ok'] for r in rows)}/{n}")

if __name__ == "__main__":
    main()

