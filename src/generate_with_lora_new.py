# src/generate_with_lora_new.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import sys
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = ROOT / "checkpoints" / "tinyllama_lora_hdl"


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEV = None
tokenizer_cache = None
model_cache = None


def load_model():
    global tokenizer_cache, model_cache, DEV
    if DEV is None:
        DEV = _device()
    if tokenizer_cache is None:
        tokenizer_cache = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer_cache.pad_token is None:
            tokenizer_cache.pad_token = tokenizer_cache.eos_token
    if model_cache is None:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model_cache = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model_cache.to(DEV).eval()
    return tokenizer_cache, model_cache


def build_prompt(instruction: str, module_declaration: str) -> str:
    # let post processing handle shape
    s = (
        "You are an expert digital designer. Emit PURE Verilog-2001 ONLY.\n"
        "Rules:\n"
        "- Use ONLY the exact port names/types from the given module declaration.\n"
        "- Do not invent new top-level ports or signals unless declared locally.\n"
        "- No SystemVerilog, no Bluespec, no delays, no initial blocks.\n"
        "- End with 'endmodule'.\n\n"
        f"### Instruction:\n{instruction}\n\n"
    )
    if module_declaration:
        s += f"### Module declaration (use exactly this signature):\n{module_declaration}\n\n"
    s += "### Verilog solution:\n"
    return s


# helpers vvvvv (similar to baseline script)


def normalize(text: str) -> str:
    # remove code fences + odd tokens
    text = re.sub(r"```[\w-]*", "", text).replace("```", "")
    text = re.sub(r"'''[\w-]*", "", text).replace("'''", "")
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\xa0"):
        text = text.replace(ch, " ")
    return text.strip()


def fixSimple(code: str) -> str:
    # fix [0:3] -> [3:0] kind of stuff
    def flip(m):
        l = int(m.group(1))
        r = int(m.group(2))
        return f"[{r}:{l}]" if l < r else m.group(0)

    code = re.sub(r"\[(\d+)\s*:\s*(\d+)\]", flip, code)

    # if there is no always block
    if not re.search(r"(?m)^\s*always\b", code):
        code = re.sub(r"\breg\b", "wire", code)
        code = re.sub(
            r"(?m)^\s*([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*<=\s*(.*?);",
            r"assign \1 = \2;",
            code,
        )
        code = re.sub(
            r"(?m)^\s*(?!assign\b|wire\b|reg\b|input\b|output\b|inout\b|parameter\b|localparam\b|integer\b|genvar\b|if\b|case\b|for\b|always\b)"
            r"([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*=\s*(.*?);",
            r"assign \1 = \2;",
            code,
        )

    # strip line comments
    code = re.sub(r"//.*", "", code)

    # common 2 bit temp pattern from baseline: collapse to a single assign
    code = re.sub(
        r"(?ms)^\s*wire\s*\[\s*1\s*:\s*0\s*\]\s*(\w+)\s*;\s*"
        r"assign\s+\1\[\s*0\s*\]\s*=\s*([A-Za-z_]\w*)\s*;\s*"
        r"assign\s+\1\[\s*1\s*\]\s*=\s*([A-Za-z_]\w*)\s*;\s*"
        r"assign\s+([A-Za-z_]\w*)\s*=\s*\1\[\s*0\s*\]\s*&\s*\1\[\s*1\s*\]\s*;",
        r"assign \4 = \2 & \3;",
        code,
    )
    code = re.sub(
        r"(?ms)^\s*wire\s*\[\s*1\s*:\s*0\s*\]\s*(\w+)\s*;\s*"
        r"assign\s+\1\[\s*0\s*\]\s*=\s*([A-Za-z_]\w*)\s*;\s*"
        r"assign\s+\1\[\s*1\s*\]\s*=\s*([A-Za-z_]\w*)\s*;\s*"
        r"assign\s+([A-Za-z_]\w*)\s*=\s*\1\[\s*1\s*\]\s*&\s*\1\[\s*0\s*\]\s*;",
        r"assign \4 = \2 & \3;",
        code,
    )

    return code


def extractModule(raw: str, top: str):
    """
    Baseline-style extractor:
    - normalize text
    - find a module...endmodule block
    - clean header / capitalization
    - force module name to 'top'
    - basic fixes
    """
    txt = normalize(raw)

    # prefer a real module header line
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

    # normalize capitalization/format
    code = re.sub(r"(?m)^\s*Module\b", "module", code)
    code = re.sub(r"(?im)^\s*module\s*[\"'\.]+", "module ", code)
    code = re.sub(
        r"(?im)^\s*module\s+([A-Za-z_]\w*)\b[^(\n]*\(",
        r"module \1 (",
        code,
    )

    # force module name to match the expected top
    head = re.search(r"(?im)^\s*module\s+([A-Za-z_]\w*)", code)
    if head and head.group(1) != top:
        code = re.sub(
            r"(?im)^\s*module\s+([A-Za-z_]\w*)",
            f"module {top}",
            code,
            count=1,
        )

    # ensure header line ends with ';'
    code = re.sub(
        r"(?is)(^\s*module\s+\w+\s*\([^)]*\))\s*(?!;)",
        r"\1;",
        code,
    )

    # clip everything after the last 'endmodule'
    end_idx = code.lower().rfind("endmodule")
    if end_idx != -1:
        code = code[:end_idx] + "endmodule"

    code = fixSimple(code)

    # must at least look like a module header
    if not re.search(r"(?m)^\s*module\s+\w+\s*\(", code):
        return None

    if not code.endswith("\n"):
        code += "\n"
    return code


def _top_from_decl(module_declaration: str, default: str = "top_module") -> str:
    m = re.search(r"\bmodule\s+([A-Za-z_]\w*)", module_declaration or "")
    return m.group(1) if m else default


def _stub_from_decl(module_declaration: str, top: str) -> str:
    decl = (module_declaration or "").strip()
    if decl:
        header = " ".join(decl.split())
        if not header.endswith(";"):
            header += ";"
        return header + "\nendmodule\n"
    # worst case: no declaration at all
    return f"module {top}();\nendmodule\n"


# main generation function vvvvvv


def generate_verilog(
    instruction: str,
    module_declaration: str = "",
    max_new_tokens: int = 256,
) -> str:
    tok, model = load_model()
    prompt = build_prompt(instruction, module_declaration)
    ids = tok(prompt, return_tensors="pt")
    ids = {k: v.to(DEV) for k, v in ids.items()}

    with torch.inference_mode():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # deterministic again
            temperature=None,
            top_p=None,
            eos_token_id=tok.eos_token_id,
        )

    # strip off the prompt from the decoded text
    gen = tok.decode(out[0], skip_special_tokens=True)[len(prompt):]

    top = _top_from_decl(module_declaration, default="top_module")
    code = extractModule(gen, top)

    if code is None:
        # model failed to give us a clean module BRUH fall back to a stub
        code = _stub_from_decl(module_declaration, top)

    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--module_decl", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    text = generate_verilog(args.instruction, args.module_decl)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text + "\n")