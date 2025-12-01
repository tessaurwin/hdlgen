# src/generate_with_lora.py
import re
import unicodedata
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = ROOT / "checkpoints" / "tinyllama_lora_hdl"

FORBIDDEN = ("Reg#", "mkReg", "rule ", "interface", "package", "endmodule%")

def _strip_to_code(text: str) -> str:
    m = re.search(r"```(?:verilog|systemverilog)?\s*(.+?)```", text, flags=re.S|re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"(module\b[\s\S]*?endmodule)", text, flags=re.I)
    return m.group(1).strip() if m else text.strip()

def _enforce_header(module_declaration: str, code: str) -> str:
    """
    Replace whatever 'module ... ;' header the model wrote with the exact
    module_declaration line. Keeps the body intact.
    """
    # normalize both to single lines
    want = " ".join(module_declaration.split()).strip()
    # ensure it ends with ';'
    if not want.rstrip().endswith(";"):
        want = want.rstrip() + ";"
    # find existing header
    m = re.search(r"(module\s+top_module\s*\([^;]*\);)", code, flags=re.I)
    if m:
        code = code.replace(m.group(1), want, 1)
    else:
        # no header found — insert one at the top
        code = want + "\n" + code
    return code

def _fallback_from_decl(module_declaration: str, instruction: str) -> str:
    """
    Build a minimal, valid Verilog-2001 module from the declaration.
    Handles common patterns: reduction AND/OR/XOR of a single vector input.
    """
    # normalize decl to one line for regex
    decl = " ".join(module_declaration.split())
    # grab the module header line if user passed full "module ... ;"
    m = re.search(r"(module\s+top_module\s*\([^;]*\);\s*)", decl)
    header = m.group(1) if m else "module top_module();"

    # find one vector input name (e.g., input [3:0] in)
    mi = re.search(r"input\s*(\[[^]]+\]\s*)?(\w+)", decl)
    in_name = mi.group(2) if mi else "in"

    # collect all output names
    outs = re.findall(r"output\s*(?:\[[^]]+\]\s*)?(\w+)", decl)
    body = []

    # helpers for reduction selection
    want_and = bool(re.search(r"reduction\s*and|\bAND4\b", instruction, re.I))
    want_or  = bool(re.search(r"reduction\s*or|\bOR4\b",  instruction, re.I))
    want_xor = bool(re.search(r"reduction\s*xor|\bXOR4\b", instruction, re.I))

    if outs:
        for o in outs:
            if want_and and re.search(rf"\b{o}\b", instruction) or o.lower().endswith("and"):
                body.append(f"  assign {o} = &{in_name};")
            elif want_or and (re.search(rf"\b{o}\b", instruction) or o.lower().endswith("or")):
                body.append(f"  assign {o} = |{in_name};")
            elif want_xor and (re.search(rf"\b{o}\b", instruction) or o.lower().endswith("xor")):
                body.append(f"  assign {o} = ^{in_name};")

    # if nothing matched, just wire zeros for safety (still valid Verilog)
    if not body and outs:
        for o in outs:
            body.append(f"  assign {o} = 1'b0;")

    return header + "\n" + "\n".join(body) + "\nendmodule\n"

def _sanitize(code: str, module_declaration: str, instruction: str) -> str:
    # keep only module..endmodule if present
    m = re.search(r"(module\b[\s\S]*?endmodule)", code, flags=re.I)
    code = m.group(1) if m else code

    # if we see forbidden tokens, bail to fallback
    if any(tok in code for tok in FORBIDDEN):
        return _fallback_from_decl(module_declaration, instruction)

    # if it doesn't actually look like plain Verilog, fallback
    if not re.search(r"\bassign\b|\balways\b", code):
        return _fallback_from_decl(module_declaration, instruction)

    # make sure it ends properly
    if "endmodule" not in code:
        code = code.rstrip() + "\nendmodule\n"
    return code

def _clean_ascii(s: str) -> str:
    # strip odd unicode like trailing '%' or smart quotes
    s = unicodedata.normalize("NFKC", s)
    return s.replace("%", "")

def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return tok, model

def format_prompt(instruction: str, module_declaration: str = "") -> str:
    # add a tiny few-shot example to bias to plain Verilog-2001
    example = """### Example:
Instruction:
From a 2-bit input x[1:0], produce single-bit output y = reduction OR of x. Use continuous assignment.

Module declaration (use EXACTLY this signature):
module top_module(input [1:0] x, output y);

Verilog solution:
module top_module(input [1:0] x, output y);
  assign y = |x;
endmodule
"""
    s = f"""### Instruction:
{instruction}

### Constraints:
- Output Verilog-2001 ONLY (no SystemVerilog, no Bluespec, no comments, no prose, no code fences).
- Emit EXACTLY one module: start with `module top_module` and end with `endmodule`.

"""
    if module_declaration:
        s += f"### Module declaration (use EXACTLY this signature):\n{module_declaration}\n\n"
    s += example
    s += "\n### Verilog solution:\n"
    return s

def generate_code(instruction: str, module_declaration: str = "", max_new_tokens=256):
    tok, model = load_model()
    prompt = format_prompt(instruction, module_declaration)
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,      # deterministic
        temperature=None,
        top_p=None,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    gen = text[len(prompt):]
    code = _strip_to_code(gen)
    code = _enforce_header(module_declaration, code) if module_declaration else code
    code = _sanitize(code, module_declaration, instruction)
    code = _clean_ascii(code)
    return _sanitize(code, module_declaration, instruction)

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", required=True)
    p.add_argument("--module_decl", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()

    code = generate_code(args.instruction, args.module_decl)
    if args.out:
        Path(args.out).write_text(code)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(code + "\n")

