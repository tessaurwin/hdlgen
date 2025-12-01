# src/llm_generate.py
from pathlib import Path
import re
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = ROOT / "checkpoints" / "tinyllama_lora_hdl"  # your LoRA output

def _strip_code_fences(text: str) -> str:
    m = re.search(r"```(?:verilog|systemverilog)?\s*(.+?)```", text, flags=re.S|re.I)
    return m.group(1).strip() if m else text.strip()

def _enforce_header(module_decl: str, code: str) -> str:
    if not module_decl:
        return code
    want = " ".join(module_decl.split()).strip()
    if not want.endswith(";"):
        want += ";"
    m = re.search(r"(module\s+top_module\s*\([^;]*\);)", code, flags=re.I)
    return code.replace(m.group(1), want, 1) if m else (want + "\n" + code)

def format_prompt(instruction: str, module_decl: Optional[str]) -> str:
    fewshot = """### Example
Instruction:
From a 2-bit input x[1:0], produce single-bit output y = reduction OR of x.

Module declaration (use EXACTLY this signature):
module top_module(input [1:0] x, output y);

Verilog solution:
module top_module(input [1:0] x, output y);
  assign y = |x;
endmodule
"""
    s = f"""### Instruction
{instruction}

### Constraints
- Output Verilog-2001 ONLY (no SystemVerilog/Bluespec, no prose, no comments, no code fences).
- Emit exactly ONE module starting with `module top_module` and ending with `endmodule`.
- Use the port names exactly as declared.

"""
    if module_decl:
        s += f"### Module declaration (use EXACTLY this signature)\n{module_decl}\n\n"
    s += fewshot
    s += "\n### Verilog solution\n"
    return s

def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return tok, model

def generate_verilog(instruction: str, module_decl: str = "", max_new_tokens=256) -> str:
    tok, model = load_model()
    prompt = format_prompt(instruction, module_decl)
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,     # deterministic
        temperature=None,
        top_p=None,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    gen = text[len(prompt):]
    code = _strip_code_fences(gen)

    # keep only module...endmodule if present (don’t mutate logic otherwise)
    m = re.search(r"(module\b[\s\S]*?endmodule)", code, flags=re.I)
    code = m.group(1).strip() if m else code.strip()

    # enforce exact header if provided (prevents port name drift)
    code = _enforce_header(module_decl, code) if module_decl else code
    if "endmodule" not in code:
        code = code.rstrip() + "\nendmodule\n"
    return code

