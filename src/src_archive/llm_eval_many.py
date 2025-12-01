# src/llm_eval_many.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence HF fork spam

import json, csv, argparse, pathlib, sys
from datetime import datetime
from llm_generate import generate_verilog
from tools import verible_lint, verilator_compile, yosys_synth

ROOT = pathlib.Path(__file__).resolve().parents[1]

def eval_one(task, out_vdir: pathlib.Path, out_logdir: pathlib.Path, idx: int, total: int):
    tid   = task["id"]
    top   = task["top"]
    instr = task["prompt"]
    mdecl = task.get("module_decl", "")

    print(f"→ [{idx}/{total}] {tid} (top={top})")
    vpath = out_vdir / f"{tid}.v"

    # 1) generate code via fine-tuned LoRA
    code = generate_verilog(instr, mdecl)
    vpath.write_text(code)

    # 2) run checks
    v_ok, v_warns, v_log = verible_lint(vpath)
    c_ok, c_log          = verilator_compile(vpath)
    y_ok, y_log          = yosys_synth(vpath, "top_module")

    # 3) save logs
    (out_logdir / f"{tid}.verible.log").write_text(v_log)
    (out_logdir / f"{tid}.verilator.log").write_text(c_log)
    (out_logdir / f"{tid}.yosys.log").write_text(y_log)

    # 4) one row of results
    return {
        "task_id": tid,
        "top": top,
        "verible_ok": int(v_ok),
        "lint_warnings": v_warns,
        "verilator_ok": int(c_ok),
        "yosys_ok": int(y_ok),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",  default=str(ROOT / "data" / "tasks.jsonl"), help="JSON array of tasks")
    ap.add_argument("--out",    default=str(ROOT / "results" / "lora_baseline.csv"))
    ap.add_argument("--vdir",   default=str(ROOT / "outputs" / "v_lora"))
    ap.add_argument("--logdir", default=str(ROOT / "outputs" / "logs_lora"))
    args = ap.parse_args()

    tasks_path = pathlib.Path(args.tasks)
    out_csv    = pathlib.Path(args.out)
    out_vdir   = pathlib.Path(args.vdir);   out_vdir.mkdir(parents=True, exist_ok=True)
    out_logdir = pathlib.Path(args.logdir); out_logdir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # load tasks
    tasks = json.loads(tasks_path.read_text())
    if not isinstance(tasks, list) or not tasks:
        print(f"no tasks in {tasks_path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    total = len(tasks)
    for i, t in enumerate(tasks, start=1):
        try:
            rows.append(eval_one(t, out_vdir, out_logdir, i, total))
        except Exception as e:
            print(f"   ✖ error on {t.get('id','?')}: {e}", file=sys.stderr)

    # write CSV even if some tasks failed
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        # fallback header if literally everything failed
        fieldnames = ["task_id","top","verible_ok","lint_warnings","verilator_ok","yosys_ok","timestamp"]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)

    # summary
    n = len(rows)
    if n:
        print(f"\nWrote {out_csv}")
        print(f"verible pass:  {sum(r['verible_ok'] for r in rows)}/{n}")
        print(f"verilator ok:  {sum(r['verilator_ok'] for r in rows)}/{n}")
        print(f"yosys ok:      {sum(r['yosys_ok'] for r in rows)}/{n}")
    else:
        print("\nNo successful rows to summarize (all tasks errored).")

if __name__ == "__main__":
    main()
