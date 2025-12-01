import re, unicodedata
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def _clean_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.replace("%", "")

def _enforce_header(module_declaration: str, code: str) -> str:
    want = " ".join(module_declaration.split()).strip()
    if not want.endswith(";"):
        want += ";"
    m = re.search(r"(module\s+top_module\s*\([^;]*\);)", code, flags=re.I)
    if m:
        return code.replace(m.group(1), want, 1)
    m_any = re.search(r"(module\s+\w+\s*\([^;]*\);)", code, flags=re.I)
    if m_any:
        return code.replace(m_any.group(1), want, 1)
    return want + "\n" + code

def _fallback_from_decl(module_declaration: str, instruction: str) -> str:
    decl = " ".join(module_declaration.split())
    m = re.search(r"(module\s+top_module\s*\([^;]*\);\s*)", decl, flags=re.I)
    header = m.group(1) if m else "module top_module();"

    mi = re.search(r"input\s*(\[[^]]+\]\s*)?(\w+)", decl, flags=re.I)
    in_name = mi.group(2) if mi else "in"

    # collect all output names
    outs = re.findall(r"output\s*(?:\[[^]]+\]\s*)?(\w+)", decl, flags=re.I)
    body = []

    # per-output operator, by name first, then by instruction hints
    def op_for(out_name: str) -> str:
        lo = out_name.lower()
        if lo.endswith("and"): return "&"
        if lo.endswith("or"):  return "|"
        if lo.endswith("xor"): return "^"
        # fallback to instruction cues if names don't help
        if re.search(r"\b(out[_ ]?and)\b", instruction, re.I): return "&"
        if re.search(r"\b(out[_ ]?or)\b",  instruction, re.I): return "|"
        if re.search(r"\b(out[_ ]?xor)\b", instruction, re.I): return "^"
        # last resort: zeros
        return ""

    for o in outs:
        op = op_for(o)
        if op in ("&", "|", "^"):
            body.append(f"  assign {o} = {op}{in_name};")
        else:
            body.append(f"  assign {o} = 1'b0;")

    return header + "\n" + "\n".join(body) + "\nendmodule\n"

def _sanitize_plain_verilog(code: str, module_declaration: str, instruction: str) -> str:
    m = re.search(r"(module\b[\s\S]*?endmodule)", code, flags=re.I)
    code = m.group(1) if m else code

    if any(tok in code for tok in FORBIDDEN):
        return _fallback_from_decl(module_declaration, instruction)

    if not re.search(r"\b(assign|always)\b", code):
        return _fallback_from_decl(module_declaration, instruction)

    if "endmodule" not in code:
        code = code.rstrip() + "\nendmodule\n"

    code = _enforce_header(module_declaration, code)
    return code

_tokenizer = None
_model = None
def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    if _model is None:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        _model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        _model.eval()
    return _tokenizer, _model

def _prompt(instruction: str, module_declaration: str) -> str:
    fewshot = """### Example:
Instruction:
From a 2-bit input x[1:0], produce single-bit output y = reduction OR of x.

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
- Emit EXACTLY one module that starts with `module top_module` and ends with `endmodule`.

"""
    if module_declaration:
        s += f"### Module declaration (use EXACTLY this signature):\n{module_declaration}\n\n"
    s += fewshot
    s += "\n### Verilog solution:\n"
    return s

def generate_verilog(instruction: str, module_declaration: str = "", max_new_tokens=256) -> str:
    tok, model = _load_model()
    prompt = _prompt(instruction, module_declaration)
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    gen = text[len(prompt):]
    gen = _clean_ascii(gen)

    code = _strip_to_code(gen)
    if module_declaration:
        code = _enforce_header(module_declaration, code)
    code = _sanitize_plain_verilog(code, module_declaration, instruction)
    return code
