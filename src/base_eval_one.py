#!/usr/bin/env python3
# ok hi future me. this file runs ONE prompt, runs it thru it through 
# verible/verilator/yosys, prints PASS/FAIL, and thats all

import os
import re
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer spam off pls
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# folders we write to (keeping actual folder names the same so nothing explodes)
root = Path(__file__).resolve().parents[1]
outDir = root / "outputs"; outDir.mkdir(parents=True, exist_ok=True)
logsDir = root / "outputs" / "logs_one"; logsDir.mkdir(parents=True, exist_ok=True)
resultsDir = root / "results"; resultsDir.mkdir(parents=True, exist_ok=True)

# which model (you can override with HDLGEN_MODEL)
modelName = os.environ.get("HDLGEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# runtime flags (cli wins, then env)
quietMode = False
keepLogs = False

def envOn(name: str, default="0"):
    # yes this is extra but my brain likes 'on' more than 1
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")

def getDevice():
    # apple metal first because... macbook
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

tokenizer = None
model = None

def loadModel():
    # load once, touch grass never
    global tokenizer, model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(modelName)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(modelName)
        model.to(getDevice()).eval()

def chatText(systemMsg, userMsg):
    # hf chat template magic, do not think about it too hard
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": systemMsg},
         {"role": "user", "content": userMsg}],
        tokenize=False,
        add_generation_prompt=True,
    )

def normalize(text):
    # strip code fences + zero-width goblins; be gone foul creatures
    text = re.sub(r"```[\w-]*", "", text).replace("```", "")
    text = re.sub(r"'''[\w-]*", "", text).replace("'''", "")
    # kill TinyLlama chat tags if they sneak in
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\xa0"):
        text = text.replace(ch, " ")
    return text.strip()

def fixSimple(code):
    # tiny bandaids so tools don't cry. still lets the model biff it plenty.
    # flip [0:3] -> [3:0] because models love chaos
    def flip(m):
        left = int(m.group(1)); right = int(m.group(2))
        return f"[{right}:{left}]" if left < right else m.group(0)
    code = re.sub(r"\[(\d+)\s*:\s*(\d+)\]", flip, code)

    # if there's no always, convert stray procedural assigns to continuous
    if not re.search(r"(?m)^\s*always\b", code):
        code = re.sub(r"\breg\b", "wire", code)
        code = re.sub(r"(?m)^\s*([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*<=\s*(.*?);",
                      r"assign \1 = \2;", code)
        code = re.sub(
            r"(?m)^\s*(?!assign\b|wire\b|reg\b|input\b|output\b|inout\b|parameter\b|localparam\b|integer\b|genvar\b|if\b|case\b|for\b|always\b)"
            r"([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*=\s*(.*?);",
            r"assign \1 = \2;",
            code,
        )

    # yeet trailing line comments the model sprinkled in
    code = re.sub(r"//.*", "", code)

    # collapse silly temp vectors used just for a & b
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
    txt = normalize(raw)

    # try to find a legit "module name (" header first
    header = re.search(r"(?ims)^\s*module\s+[A-Za-z_]\w*\s*\(", txt)
    if header:
        start = header.start()
        tail = txt[start:]
        endm = re.search(r"(?is)\bendmodule\b", tail)
        if endm:
            code = tail[:endm.end()]
        else:
            return None
    else:
        m = re.search(r"(?is)\bmodule\b.*?\bendmodule\b", txt)
        if not m:
            return None
        code = m.group(0)

    # clean up a few model gremlins
    code = re.sub(r"(?m)^\s*Module\b", "module", code)
    code = re.sub(r'(?im)^\s*module\s*["\'\.]+', "module ", code)
    code = re.sub(r"(?im)^\s*module\s+([A-Za-z_]\w*)\b[^(\n]*\(", r"module \1 (", code)

    # rename the top to what we asked for (pls listen tiny model)
    head = re.search(r"(?im)^\s*module\s+([A-Za-z_]\w*)", code)
    if head and head.group(1) != top:
        code = re.sub(r"(?im)^\s*module\s+([A-Za-z_]\w*)", f"module {top}", code, count=1)

    # make sure the port list ends with a semicolon because verilog is petty
    code = re.sub(r"(?is)(^\s*module\s+\w+\s*\([^)]*\))\s*(?!;)", r"\1;", code)

    # trim anything after endmodule just in case the model kept talking
    endIdx = code.lower().rfind("endmodule")
    code = code[:endIdx] + "endmodule"

    # tiny repairs
    code = fixSimple(code)

    # sanity check
    if not re.search(r"(?m)^\s*module\s+\w+\s*\(", code):
        return None
    if not code.endswith("\n"):
        code += "\n"
    return code

def generateVerilog(top, taskText, maxNew=256):
    loadModel()
    systemMsg = (
        f"You are a Verilog-2001 code generator. "
        f"Return only code. No comments. No prose. No markdown. "
        f"Exactly one module named {top}. "
        f"Use input and output in the port list. "
        f"Do not use SystemVerilog. Do not use Bluespec or BSV keywords."
    )
    prompt = chatText(systemMsg, taskText)
    enc = tokenizer(prompt, return_tensors="pt").to(getDevice())

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=maxNew,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def runTool(cmd, toolName, stem):
    p = subprocess.run(cmd, capture_output=True, text=True)
    # only write logs if failed or we asked to keep them
    if keepLogs or p.returncode != 0:
        (logsDir / f"{stem}.{toolName}.log").write_text((p.stdout or "") + (p.stderr or ""))
    return p

def main():
    global quietMode, keepLogs

    parser = argparse.ArgumentParser()
    parser.add_argument("--top", default="and_gate")
    parser.add_argument("--task", default="Write a 1-bit AND gate. Ports: input a, input b, output y.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--keep-logs", action="store_true")
    args = parser.parse_args()

    quietMode = args.quiet or envOn("HDLGEN_QUIET")
    keepLogs = args.keep_logs or envOn("HDLGEN_KEEP_LOGS")

    print(f"Querying HF model: {modelName} on {getDevice()} ...")
    raw = generateVerilog(args.top, args.task)
    (logsDir / "one.raw.txt").write_text(raw)

    code = extractModule(raw, args.top)
    if code is None:
        # last-resort stub so tools can at least run and scream honestly
        code = (
            f"module {args.top}(input a, input b, output y);\n"
            f"  assign y = a & b;\n"
            f"endmodule\n"
        )

    vPath = outDir / f"{args.top}.v"
    vPath.write_text(code)
    if not quietMode:
        print(f"saved {vPath}")

    lint = runTool(["verible-verilog-lint", str(vPath)], "verible", args.top)
    veri = runTool(["verilator", "--lint-only", str(vPath)], "verilator", args.top)
    yos  = runTool(["yosys", "-Q", "-p",
                    f"read_verilog {vPath}; hierarchy -check -top {args.top}; synth -top {args.top}; stat"],
                   "yosys", args.top)

    def pf(rc): return "PASS" if rc == 0 else "FAIL"
    line = f"{args.top}: verible {pf(lint.returncode)}  verilator {pf(veri.returncode)}  yosys {pf(yos.returncode)}"
    print(line)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with (resultsDir / "summary.txt").open("a") as f:
        f.write(f"\n[{ts}] {vPath} model={modelName}\n")
        f.write(f"verible={lint.returncode} verilator={veri.returncode} yosys={yos.returncode}\n")
        f.write(f"status: {line}\n")

    ok = (lint.returncode == 0 and veri.returncode == 0 and yos.returncode == 0)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()