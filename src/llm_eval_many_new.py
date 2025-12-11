# src/llm_eval_many_new.py
import json, csv, sys, argparse, pathlib, re
from datetime import datetime
from generate_with_lora_new import generate_verilog
from tools import verible_lint, verilator_compile, yosys_synth

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = pathlib.Path(__file__).resolve().parents[1]

def module_name_from_declaration(decl: str) -> str:
    """If a custom module name is given in the declaration, use it; else 'top_module'."""
    if not decl:
        return "top_module"
    m = re.search(r"module\s+(\w+)\s*\(", decl, flags=re.I)
    return m.group(1) if m else "top_module"

def evaluate_one_task(task: dict, verilog_dir: pathlib.Path, logs_dir: pathlib.Path):
    task_id  = task["id"]
    want_top = task.get("top", "top_module")
    prompt   = task.get("prompt") or task.get("instruction")
    decl     = task.get("module_declaration", "")
    if not prompt:
        raise ValueError(f"task {task_id} has no 'prompt' or 'instruction'")

    # if there is no explicit module_declaration, synthesize a trivial one
    # so the generator knows the intended module name + ports (even if empty)
    if not decl:
        # at least force the top-level module name to match the task's top
        decl = f"module {want_top}();"

    vpath = verilog_dir / f"{task_id}.v"

    # generate verilog (header enforced via decl if there)
    code = generate_verilog(prompt, decl)
    vpath.write_text(code)

    # for synthesis, top = declared module name if present, else "top_module"
    synth_top = module_name_from_declaration(decl)

    # run checks
    lint_ok, lint_warns, lint_log = verible_lint(vpath)
    sim_ok, sim_log = verilator_compile(vpath)
    synth_ok, synth_log = yosys_synth(vpath, synth_top)

    # logs
    (logs_dir / f"{task_id}.verible.log").write_text(lint_log)
    (logs_dir / f"{task_id}.verilator.log").write_text(sim_log)
    (logs_dir / f"{task_id}.yosys.log").write_text(synth_log)

    return {
        "task_id": task_id,
        "top":     want_top,
        "verible_ok":   int(lint_ok),
        "lint_warnings": lint_warns,
        "verilator_ok": int(sim_ok),
        "yosys_ok":     int(synth_ok),
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
    }

def _read_tasks_any(path: pathlib.Path):
    txt = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else []
    except Exception:
        pass
    rows = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        rows.append(json.loads(ln))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",  default=str(ROOT / "data" / "tasks.jsonl"))
    ap.add_argument("--out",    default=str(ROOT / "results" / "lora_baseline.csv"))
    ap.add_argument("--vdir",   default=str(ROOT / "outputs" / "v_lora"))
    ap.add_argument("--logdir", default=str(ROOT / "outputs" / "logs_lora"))
    args = ap.parse_args()

    tasks_path = pathlib.Path(args.tasks)
    out_csv = pathlib.Path(args.out)
    verilog_dir = pathlib.Path(args.vdir); verilog_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = pathlib.Path(args.logdir); logs_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tasks = _read_tasks_any(tasks_path)
    if not tasks:
        print(f"no tasks in {tasks_path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    total = len(tasks)
    for i, t in enumerate(tasks, start=1):
        print(f"→ [{i}/{total}] {t.get('id','?')} (top={t.get('top','?')})")
        try:
            rows.append(evaluate_one_task(t, verilog_dir, logs_dir))
        except Exception as e:
            print(f"error on {t.get('id','?')}: {e}", file=sys.stderr)


    # write CSV even if some tasks failed
    if rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        n = len(rows)
        print(f"\nWrote {out_csv}")
        print(f"verible pass:  {sum(r['verible_ok'] for r in rows)}/{n}")
        print(f"verilator ok:  {sum(r['verilator_ok'] for r in rows)}/{n}")
        print(f"yosys ok:      {sum(r['yosys_ok'] for r in rows)}/{n}")
    else:
        print("\nNo successful rows to summarize.")

if __name__ == "__main__":
    main()
