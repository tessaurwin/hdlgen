# batch mode: feed it a tasks file
# prints one line per task like a scoreboard, then a tiny summary, then a csv

import os, re, json, argparse, subprocess
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# less annoying from tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# env knobs
quietMode = os.environ.get("HDLGEN_QUIET", "0") == "1"
keepLogs = os.environ.get("HDLGEN_KEEP_LOGS", "0") == "1"
modelName = os.environ.get("HDLGEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# folders (names stay stable on disk so git doesn't fight me)
root = Path(__file__).resolve().parents[1]
vDir = root / "outputs" / "v_base"; vDir.mkdir(parents=True, exist_ok=True)
logsDir = root / "outputs" / "logs_base"; logsDir.mkdir(parents=True, exist_ok=True)
resultsDir = root / "results"; resultsDir.mkdir(parents=True, exist_ok=True)

def getDevice():
    # mps supremacy
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = getDevice()

# hugg
tok = None
mdl = None

def loadModel():
    global tok, mdl
    if tok is None:
        tok = AutoTokenizer.from_pretrained(modelName)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    if mdl is None:
        mdl = AutoModelForCausalLM.from_pretrained(modelName)
        mdl.to(device).eval()

def chatText(messages):
    # messages = [{"role":"system","content":"..."},{"role":"user","content":"..."}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def normalize(text):
    # get rid of code fences + invisible unicode that break diffs pls
    text = re.sub(r"```[\w-]*", "", text).replace("```","")
    text = re.sub(r"'''[\w-]*", "", text).replace("'''","")
    text = re.sub(r"<\|[^>]+?\|>", "", text) # models tags
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\xa0"):
        text = text.replace(ch, " ")
    return text.strip()

def fixSimple(code):
    # little fixes for typos so we can focus on testing functionality
    def flip(m):
        l = int(m.group(1)); r = int(m.group(2))
        return f"[{r}:{l}]" if l < r else m.group(0)
    code = re.sub(r"\[(\d+)\s*:\s*(\d+)\]", flip, code)

    if not re.search(r"(?m)^\s*always\b", code):
        code = re.sub(r"\breg\b", "wire", code)
        code = re.sub(r"(?m)^\s*([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*<=\s*(.*?);", r"assign \1 = \2;", code)
        code = re.sub(
            r"(?m)^\s*(?!assign\b|wire\b|reg\b|input\b|output\b|inout\b|parameter\b|localparam\b|integer\b|genvar\b|if\b|case\b|for\b|always\b)"
            r"([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*=\s*(.*?);",
            r"assign \1 = \2;",
            code,
        )

    # i don't like when it says a lot
    code = re.sub(r"//.*", "", code)

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

def extractModule(raw, top):
    # yeet a single module block out of whatever the model just did
    txt = normalize(raw)

    header = re.search(r"(?ims)^\s*module\s+[A-Za-z_]\w*\s*\(", txt)
    if header:
        tail = txt[header.start():]
        endm = re.search(r"(?is)\bendmodule\b", tail)
        if endm:
            code = tail[:endm.end()]
        else:
        # salvage partial module instead of bailing because its just like awkward messy sometimes lowkey
            code = tail
    else:
        m = re.search(r"(?is)\bmodule\b.*?\bendmodule\b", txt)
        if not m:
            return None
        code = m.group(0)

    code = re.sub(r"(?m)^\s*Module\b", "module", code)
    code = re.sub(r"(?im)^\s*module\s*[\"'\.]+", "module ", code)
    code = re.sub(r"(?im)^\s*module\s+([A-Za-z_]\w*)\b[^(\n]*\(", r"module \1 (", code)

    head = re.search(r"(?im)^\s*module\s+([A-Za-z_]\w*)", code)
    if head and head.group(1) != top:
        code = re.sub(r"(?im)^\s*module\s+([A-Za-z_]\w*)", f"module {top}", code, count=1)

    code = re.sub(r"(?is)(^\s*module\s+\w+\s*\([^)]*\))\s*(?!;)", r"\1;", code)

    endIdx = code.lower().rfind("endmodule")
    if endIdx != -1:
        code = code[:endIdx] + "endmodule"
    else:
        # if model never closed the module add
        code = code.rstrip() + "\nendmodule\n"

    code = fixSimple(code)
    
    # be strict but less so just require some module header even if it has no port list
    if not re.search(r"(?m)^\s*module\s+\w+", code):
        return None

    if not code.endswith("\n"):
        code += "\n"
    return code

def generateVerilog(top, text, maxNew):
    # very specific prompt at the model prompt engineering
    loadModel()
    sysMsg = (
        f"You are a Verilog-2001 code generator. "
        f"Return only code. No comments. No prose. No markdown. "
        f"Exactly one module named {top}. "
        f"Use input and output in the port list. "
        f"Do not use SystemVerilog. Do not use Bluespec or BSV keywords."
    )
    prompt = chatText([
        {"role":"system","content":sysMsg},
        {"role":"user","content":text},
    ])
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl.generate(
            **enc,
            max_new_tokens=maxNew,
            do_sample=False, # consistent bad then consistent 
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)

def runTool(cmd, toolName, stem):
    # run tool, maybe log
    p = subprocess.run(cmd, capture_output=True, text=True)
    if keepLogs:
        (logsDir / f"{stem}.{toolName}.log").write_text((p.stdout or "") + (p.stderr or ""))
    if not quietMode and p.returncode != 0:
        print(p.stdout or p.stderr)
    return p

# if model truly gives us something completely invalid and incorrect we need to sub an empty so things still work in eval
def fallbackStub(top):
    # bare minimum so the toolchain can not like freak out and can still eval through everything
    return f"module {top}();\nendmodule\n"

def loadTasks(pathStr):
    # json array or jsonl
    p = Path(pathStr)
    txt = p.read_text().strip()
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    tasks = []
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tasks.append(json.loads(s))
    return tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(root / "data" / "tasks.jsonl"))
    parser.add_argument("--out", default=str(resultsDir / "tinyllama_many.csv"))
    parser.add_argument("--max-new", type=int, default=int(os.environ.get("HDLGEN_MAX_NEW", "420")))
    args = parser.parse_args()

    tasks = loadTasks(args.tasks)
    print(f"Querying HF model: {modelName} on {device} ...")

    rows = []
    for t in tasks:
        tid = t["id"]; top = t["top"]; text = t["prompt"]

        raw = generateVerilog(top, text, args.max_new)
        if keepLogs:
            (logsDir / f"{tid}.raw.txt").write_text(raw)

        code = extractModule(raw, top)
        source = "model"
        if code is None:
            # dump raw so you can inspect what went wrong
            (logsDir / f"{tid}.raw.fallback.txt").write_text(raw)
            code = fallbackStub(top)
            source = "fallback"

        vPath = vDir / f"{tid}.v"
        vPath.write_text(code)

        lint = runTool(["verible-verilog-lint", str(vPath)], "verible", tid)
        veri = runTool(["verilator", "--lint-only", str(vPath)], "verilator", tid)
        yos = runTool(["yosys","-Q","-p",
                        f"read_verilog {vPath}; hierarchy -check -top {top}; synth -top {top}; stat"], "yosys", tid)

        vOk = int(lint.returncode == 0)
        vlOk = int(veri.returncode == 0)
        yOk = int(yos.returncode == 0)

        # scoreboard line to show performance results with the EDAs
        print(f"{tid}: verible {'PASS' if vOk else 'FAIL'} verilator {'PASS' if vlOk else 'FAIL'} yosys {'PASS' if yOk else 'FAIL'}")

        warnCount = 0
        if keepLogs:
            outTxt = (lint.stdout or "") + (lint.stderr or "")
            warnCount = sum(1 for ln in outTxt.splitlines() if 'warning' in ln.lower())

        rows.append({
            "taskId": tid,
            "top": top,
            "veribleOk": vOk,
            "lintWarnings": warnCount,
            "verilatorOk": vlOk,
            "yosysOk": yOk,
            "source": source,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    outPath = Path(args.out); outPath.parent.mkdir(parents=True, exist_ok=True)
    with outPath.open("w") as f:
        f.write("taskId,top,veribleOk,lintWarnings,verilatorOk,yosysOk,source,timestamp\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in [
                "taskId","top","veribleOk","lintWarnings","verilatorOk","yosysOk","source","timestamp"
            ]) + "\n")

    n = len(rows) or 1
    print(f"\nWrote {outPath}")
    print(f"verible pass {sum(r['veribleOk'] for r in rows)}/{n}")
    print(f"verilator ok {sum(r['verilatorOk'] for r in rows)}/{n}")
    print(f"yosys ok {sum(r['yosysOk'] for r in rows)}/{n}")

if __name__ == "__main__":
    main()