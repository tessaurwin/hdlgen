#!/usr/bin/env python3
# batch mode: feed it a tasks file, watch the fireworks
# prints one line per task like a scoreboard, then a tiny summary, then a csv

import os, re, json, argparse, subprocess
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# env controls
os.environ["TOKENIZERS_PARALLELISM"] = "false"
QUIET = os.environ.get("HDLGEN_QUIET", "0") == "1"
KEEP_LOGS = os.environ.get("HDLGEN_KEEP_LOGS", "0") == "1"
MODEL_NAME = os.environ.get("HDLGEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# folders
root = Path(__file__).resolve().parents[1]
v_dir = root / "outputs" / "v_base"; v_dir.mkdir(parents=True, exist_ok=True)
logs_dir = root / "outputs" / "logs_base"; logs_dir.mkdir(parents=True, exist_ok=True)
results_dir = root / "results"; results_dir.mkdir(parents=True, exist_ok=True)

# device
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = get_device()

# hf objects
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.to(device).eval()

def chat_text(system_msg, user_msg):
    return tokenizer.apply_chat_template(
        [{"role":"system","content":system_msg},
        {"role":"user","content":user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
def normalize(text):
    # drop code fences and zero-width junk
    text = re.sub(r"```[\w-]*", "", text).replace("```","")
    text = re.sub(r"'''[\w-]*", "", text).replace("'''","")
    # drop any TinyLlama-style chat tags
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\xa0"):
        text = text.replace(ch, " ")
    return text.strip()

def fix_simple(code):
    # normalize bit ranges like [0:1] -> [1:0]
    def _flip(m):
        l = int(m.group(1)); r = int(m.group(2))
        return f"[{r}:{l}]" if l < r else m.group(0)
    code = re.sub(r"\[(\d+)\s*:\s*(\d+)\]", _flip, code)
    
    # if no always, coerce stray assigns to combinational assigns
    if not re.search(r"(?m)^\s*always\b", code):
        code = re.sub(r"\breg\b", "wire", code)
        code = re.sub(r"(?m)^\s*([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*<=\s*(.*?);", r"assign \1 = \2;", code)
        code = re.sub(
            r"(?m)^\s*(?!assign\b|wire\b|reg\b|input\b|output\b|inout\b|parameter\b|localparam\b|integer\b|genvar\b|if\b|case\b|for\b|always\b)"
            r"([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*=\s*(.*?);",
            r"assign \1 = \2;",
            code,
        )

    # drop trailing line comments
    code = re.sub(r"//.*", "", code)

    # collapse trivial 2-bit temp vectors to direct assign
    code = re.sub(
        r"(?ms)^\s*wire\s*\[\s*1\s*:\s*0\s*\]\s*(\w+)\s*;\s*"
        r"assign\s+\1\[\s*0\s*\]\s*=\s*([A-Za-z_]\w*)\s*;\s*"
        r"assign\s+\1\[\s*1\s*\]\s*=\s*([A-Za-z_]\w*)\s*;\s*"
        r"assign\s+([A-Za-z_]\w*)\s*=\s*\1\[\s*0\s*\]\s*&\s*\1\[\s*1\s*\]\s*;",
        r"assign \4 = \2 & \3;",
        code
    )
    code = re.sub(
        r"(?ms)^\s*wire\s*\[\s*1\s*:\s*0\s*\]\s*(\w+)\s*;\s*"
        r"assign\s+\1\[\s*0\s*\]\s*=\s*([A-Za-z_]\w*)\s*;\s*"
        r"assign\s+\1\[\s*1\s*\]\s*=\s*([A-Za-z_]\w*)\s*;\s*"
        r"assign\s+([A-Za-z_]\w*)\s*=\s*\1\[\s*1\s*\]\s*&\s*\1\[\s*0\s*\]\s*;",
        r"assign \4 = \2 & \3;",
        code
    )
    return code

def extract_module(raw, top):
    txt = normalize(raw)

    # prefer header "module <name> ("
    header = re.search(r"(?ims)^\s*module\s+[A-Za-z_]\w*\s*\(", txt)
    if header:
        tail = txt[header.start():]
        endm = re.search(r"(?is)\bendmodule\b", tail)
        if not endm:
            return None
        code = tail[:endm.end()]
    else:
        m = re.search(r"(?is)\bmodule\b.*?\bendmodule\b", txt)
        if not m:
            return None
        code = m.group(0)

    # tidy header
    code = re.sub(r"(?m)^\s*Module\b", "module", code)
    code = re.sub(r"(?im)^\s*module\s*[\"'\.]+", "module ", code)
    code = re.sub(r"(?im)^\s*module\s+([A-Za-z_]\w*)\b[^(\n]*\(", r"module \1 (", code)

    # rename top if needed
    head_name = re.search(r"(?im)^\s*module\s+([A-Za-z_]\w*)", code)
    if head_name and head_name.group(1) != top:
        code = re.sub(r"(?im)^\s*module\s+([A-Za-z_]\w*)", f"module {top}", code, count=1)

    # ensure semicolon after port list
    code = re.sub(r"(?is)(^\s*module\s+\w+\s*\([^)]*\))\s*(?!;)", r"\1;", code)

    # trim after endmodule
    end_idx = code.lower().rfind("endmodule")
    code = code[:end_idx] + "endmodule"

    # final repairs
    code = fix_simple(code)

    # smoke check
    if not re.search(r"(?m)^\s*module\s+\w+\s*\(", code):
        return None
    if not code.endswith("\n"):
        code += "\n"   
    return code

def generate_verilog(top, task_text, max_new=360):
    load_model()
    system_msg = (
        f"You are a Verilog-2001 code generator. "
        f"Return only code. No comments. No prose. No markdown. "
        f"Exactly one module named {top}. "
        f"Use input and output in the port list. "
        f"Do not use SystemVerilog. Do not use Bluespec or BSV keywords."
    )
    prompt = chat_text(system_msg, task_text)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def run_and_log(cmd, name, stem):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if KEEP_LOGS:
        (logs_dir / f"{stem}.{name}.log").write_text((p.stdout or "") + (p.stderr or ""))
    if not QUIET and p.returncode != 0:
        print(p.stdout or p.stderr)
    return p

def fallback_stub(top):
    return f"module {top}();\nendmodule\n"

def load_tasks(path_str):
    p = Path(path_str)
    txt = p.read_text().strip()
    # try plain JSON array first
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # fallback JSONL
    tasks = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(json.loads(line))
    return tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(root / "data" / "tasks.jsonl"))
    ap.add_argument("--out", default=str(results_dir / "base_tinyllama.csv"))
    ap.add_argument("--max-new", type=int, default=int(os.environ.get("HDLGEN_MAX_NEW", "420")))
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    n = len(tasks)
    print(f"Querying HF model: {MODEL_NAME} on {device} ...")

    rows = []
    for i, t in enumerate(tasks, start=1):
        tid = t["id"]
        top = t["top"]
        text = t["prompt"]
        raw = generate_verilog(top, text, max_new=args.max_new)
        if KEEP_LOGS:
            (logs_dir / f"{tid}.raw.txt").write_text(raw)

        code = extract_module(raw, top)
        source = "model"
        if code is None:
            code = fallback_stub(top)
            source = "fallback"

        vpath = v_dir / f"{tid}.v"
        vpath.write_text(code)

        lint = run_and_log(["verible-verilog-lint", str(vpath)], "verible", tid)
        veri = run_and_log(["verilator", "--lint-only", str(vpath)], "verilator", tid)
        yos = run_and_log(["yosys","-Q","-p", f"read_verilog {vpath}; hierarchy -check -top {top}; synth -top {top}; stat"], "yosys", tid)

        verible_ok = int(lint.returncode == 0)
        verilator_ok = int(veri.returncode == 0)
        yosys_ok = int(yos.returncode == 0)

        # single compact line per task
        print(f"{tid}: verible {'PASS' if verible_ok else 'FAIL'} verilator {'PASS' if verilator_ok else 'FAIL'} yosys {'PASS' if yosys_ok else 'FAIL'}")

        # count basic warnings from verible output (only if logs kept)
        warns = 0
        if KEEP_LOGS:
            out_txt = (lint.stdout or "") + (lint.stderr or "")
            warns = sum(1 for ln in out_txt.splitlines() if "warning" in ln.lower())

        rows.append({
            "task_id": tid,
            "top": top,
            "verible_ok": verible_ok,
            "lint_warnings": warns,
            "verilator_ok": verilator_ok,
            "yosys_ok": yosys_ok,
            "source": source,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("task_id,top,verible_ok,lint_warnings,verilator_ok,yosys_ok,source,timestamp\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in [
                "task_id","top","verible_ok","lint_warnings","verilator_ok","yosys_ok","source","timestamp"
            ]) + "\n")

    total = len(rows) or 1
    v_ok = sum(r["verible_ok"] for r in rows)
    vl_ok = sum(r["verilator_ok"] for r in rows)
    y_ok = sum(r["yosys_ok"] for r in rows)
    print(f"\nWrote {out}")
    print(f"verible pass {v_ok}/{total}")
    print(f"verilator ok {vl_ok}/{total}")
    print(f"yosys ok {y_ok}/{total}")

if __name__ == "__main__":
    main()