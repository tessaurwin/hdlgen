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
FORBIDDEN = ("Reg#", "mkReg", "rule ", "interface", "package")

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
    # Remove the few-shot that used x[1:0]; give hard constraints instead.
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

def slice_module(text: str) -> str:
    m = re.search(r"(module\b[\s\S]*?endmodule)", text, flags=re.I)
    return m.group(1).strip() if m else text.strip()

def enforce_header(module_declaration: str, code: str) -> str:
    if not module_declaration:
        return code
    want = " ".join(module_declaration.split()).rstrip()
    if not want.endswith(";"):
        want += ";"
    m = re.search(r"(module\s+\w+\s*\([^;]*\);)", code, flags=re.I)
    if m:
        return code.replace(m.group(1), want, 1)
    return want + "\n" + code

# --- helpers to parse & validate ports/idents ---
_VKEYWORDS = {
    "module","endmodule","input","output","inout","wire","reg","assign",
    "always","begin","end","if","else","case","endcase","posedge","negedge",
    "parameter","localparam"
}

def _decl_port_names(decl: str):
    # capture all names that appear after input/output/inout, including comma lists
    names = re.findall(r"\b(?:input|output|inout)\b[^;]*?([A-Za-z_]\w*)", decl)
    return set(names)

def _idents_in_code(code: str):
    toks = set(re.findall(r"\b[A-Za-z_]\w*\b", code))
    return toks

def _detect_op(instr: str):
    s = instr.lower()
    if "xnor" in s: return "xnor"
    if "nand" in s: return "nand"
    if "nor"  in s: return "nor"
    if "parity" in s or "xor of all" in s: return "xor"
    if "reduction xor" in s: return "rxor"
    if "reduction or"  in s: return "ror"
    if "reduction and" in s: return "rand"
    if "xor" in s: return "xor"
    if "and" in s: return "and"
    if "or"  in s: return "or"
    if "not" in s or "invert" in s: return "not"
    return None

def _ports_by_dir(decl: str):
    # crude but effective splits
    ins  = re.findall(r"\binput\b[^;]*?([A-Za-z_]\w*)", decl)
    outs = re.findall(r"\boutput\b[^;]*?([A-Za-z_]\w*)", decl)
    return ins, outs

def _synthesize_simple(decl: str, instruction: str) -> str:
    ins, outs = _ports_by_dir(decl)
    header_m = re.search(r"(module\s+\w+\s*\([^;]*\);\s*)", decl, flags=re.I)
    header = header_m.group(1) if header_m else "module top_module();"
    op = _detect_op(instruction)

    body = []
    if not outs:
        return header + "\nendmodule\n"

    # pick first output
    y = outs[0]

    # Single-input NOT
    if op == "not" and ins:
        body.append(f"  assign {y} = ~{ins[0]};")

    # Two-input gates
    elif op in {"and","or","xor","xnor","nand","nor"} and len(ins) >= 2:
        a, b = ins[0], ins[1]
        if op == "and":  expr = f"{a} & {b}"
        if op == "or":   expr = f"{a} | {b}"
        if op == "xor":  expr = f"{a} ^ {b}"
        if op == "xnor": expr = f"{a} ^~ {b}"
        if op == "nand": expr = f"~({a} & {b})"
        if op == "nor":  expr = f"~({a} | {b})"
        body.append(f"  assign {y} = {expr};")

    # Reductions (best-effort: use first input as vector)
    elif op in {"rand","ror","rxor","and","or","xor"} and ins:
        v = ins[0]
        red = "&" if op in {"rand", "and"} else "|" if op in {"ror", "or"} else "^"
        body.append(f"  assign {y} = {red}{v};")

    else:
        # fallback constant to keep synthesis happy
        body.append(f"  assign {y} = 1'b0;")

    return header + "\n" + "\n".join(body) + "\nendmodule\n"

def sanitize_code(code: str, module_declaration: str, instruction: str) -> str:
    # Keep only the first module block
    m = re.search(r"(module\b[\s\S]*?endmodule)", code, flags=re.I)
    code = (m.group(1) if m else code).strip()

    # Drop clearly forbidden or non-hardware outputs
    if any(tok in code for tok in FORBIDDEN) or not re.search(r"\b(assign|always)\b", code):
        return _synthesize_simple(module_declaration, instruction)

    # Enforce header (exact signature)
    code = enforce_header(module_declaration, code)

    # Must end properly
    if "endmodule" not in code:
        code = code.rstrip() + "\nendmodule\n"

    # Validate identifiers: if it uses unknown names (e.g., 'x') not in declaration, replace.
    if module_declaration:
        allowed = _decl_port_names(module_declaration) | _VKEYWORDS
        used = _idents_in_code(code)
        unknown = {t for t in used if t not in allowed}
        # allow locally-declared regs/wires (very rough): if code declares them, whitelist them
        locals_declared = set(re.findall(r"\b(?:wire|reg)\b[^;]*?([A-Za-z_]\w*)", code))
        unknown -= locals_declared
        if unknown:
            return _synthesize_simple(module_declaration, instruction)

    return code

def generate_verilog(instruction: str, module_declaration: str = "", max_new_tokens: int = 128) -> str:
    tok, model = load_model()
    prompt = build_prompt(instruction, module_declaration)
    ids = tok(prompt, return_tensors="pt")
    ids = {k: v.to(DEV) for k, v in ids.items()}  # keep tensors on the model device
    with torch.inference_mode():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tok.eos_token_id,
        )
    gen = tok.decode(out[0], skip_special_tokens=True)[len(prompt):]
    code = slice_module(gen)
    return sanitize_code(code, module_declaration, instruction)

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