# src/llm_eval_many_new.py
import json, csv, sys, argparse, pathlib, re
from datetime import datetime
from generate_with_lora_new import generate_verilog
from tools import verible_lint, verilator_compile, yosys_synth

ROOT = pathlib.Path(__file__).resolve().parents[1]

def _module_name_from_decl(decl: str) -> str:
    if not decl:
        return "top_module"
    m = re.search(r"module\s+(\w+)\s*\(", decl, flags=re.I)
    return m.group(1) if m else "top_module"

def eval_one(task, out_vdir: pathlib.Path, out_logdir: pathlib.Path):
    tid  = task["id"]
    want_top  = task["top"]
    instr = task["prompt"]
    decl  = task.get("module_decl", "")

    vpath = out_vdir / f"{tid}.v"

    # generate with LoRA (use decl to enforce header if needed)
    code = generate_verilog(instr, decl)
    vpath.write_text(code)

    # synth top = decl if exists, else "top_module"
    synth_top = _module_name_from_decl(decl)

    v_ok, v_warns, v_log = verible_lint(vpath)
    c_ok, c_log = verilator_compile(vpath)
    y_ok, y_log = yosys_synth(vpath, synth_top)

    (out_logdir / f"{tid}.verible.log").write_text(v_log)
    (out_logdir / f"{tid}.verilator.log").write_text(c_log)
    (out_logdir / f"{tid}.yosys.log").write_text(y_log)

    return {
        "task_id": tid,
        "top": want_top,
        "verible_ok": int(v_ok),
        "lint_warnings": v_warns,
        "verilator_ok": int(c_ok),
        "yosys_ok": int(y_ok),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",  default=str(ROOT / "data" / "tasks.jsonl"))
    ap.add_argument("--out",    default=str(ROOT / "results" / "lora_baseline.csv"))
    ap.add_argument("--vdir",   default=str(ROOT / "outputs" / "v_lora"))
    ap.add_argument("--logdir", default=str(ROOT / "outputs" / "logs_lora"))
    args = ap.parse_args()

    tasks_path = pathlib.Path(args.tasks)
    out_csv    = pathlib.Path(args.out)
    out_vdir   = pathlib.Path(args.vdir);   out_vdir.mkdir(parents=True, exist_ok=True)
    out_logdir = pathlib.Path(args.logdir); out_logdir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(tasks_path.read_text())
    if not isinstance(tasks, list) or not tasks:
        print(f"no tasks in {tasks_path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    total = len(tasks)
    for i, t in enumerate(tasks, start=1):
        print(f"→ [{i}/{total}] {t.get('id','?')} (top={t.get('top','?')})")
        try:
            rows.append(eval_one(t, out_vdir, out_logdir))
        except Exception as e:
            print(f"   ✖ error on {t.get('id','?')}: {e}", file=sys.stderr)

    # write csv even if some failed
    if rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        n = len(rows)
        print(f"\nWrote {out_csv}")
        print(f"verible pass:  {sum(r['verible_ok'] for r in rows)}/{n}")
        print(f"verilator ok:  {sum(r['verilator_ok'] for r in rows)}/{n}")
        print(f"yosys ok:      {sum(r['yosys_ok'] for r in rows)}/{n}")
    else:
        print("\nNo successful rows to summarize.")

if __name__ == "__main__":
    main()

