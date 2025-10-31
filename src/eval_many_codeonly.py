# src/eval_many_codeonly.py
import json, csv, pathlib, re, subprocess
from datetime import datetime
from tools import verible_lint, verilator_compile, yosys_synth

# ---------- paths ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_V = ROOT / "outputs" / "v"
OUT_LOG = ROOT / "outputs" / "logs"
OUT_RES = ROOT / "results"
DATA = ROOT / "data" / "tasks.jsonl"

for p in (OUT_V, OUT_LOG, OUT_RES):
    p.mkdir(parents=True, exist_ok=True)

MODEL = "codellama:7b-instruct"


# ---------- helpers ----------
def generate_code_only(prompt, model=MODEL):
    """Query Ollama model and return only Verilog code."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True,
        text=True
    )
    raw_output = result.stdout.strip()

    # Extract fenced code block first if it exists
    match = re.search(r"```(?:verilog)?(.*?)```", raw_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        # fallback heuristic: remove commentary & explanations
        lines = [
            ln for ln in raw_output.splitlines()
            if not any(ln.strip().startswith(pfx) for pfx in ("Sure", "Here", "#", "//"))
        ]
        code = "\n".join(lines).strip()

    # ensure newline at end of file
    if not code.endswith("\n"):
        code += "\n"
    return code


# ---------- main eval ----------
def eval_one(task: dict, model=MODEL):
    tid, top, prompt = task["id"], task["top"], task["prompt"]
    vpath = OUT_V / f"{tid}.v"

    code = generate_code_only(prompt, model=model)
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
