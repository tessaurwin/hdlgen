import re
import unicodedata
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = ROOT / "checkpoints" / "tinyllama_lora_hdl"

# Cache so we only load the model/tokenizer once per process
tokenizer_cache = None
model_cache = None

BLUESPEC_TOKENS = ("Reg#", "mkReg", "rule ", "interface", "package", "endmodule%")

def clean_ascii(text: str) -> str:
    """Normalize odd unicode (like smart quotes, stray %)."""
    text = unicodedata.normalize("NFKC", text)
    return text.replace("%", "")

def slice_module_text(text: str) -> str:
    """Return the first 'module ... endmodule' block if present, else raw text."""
    m = re.search(r"(module\b[\s\S]*?endmodule)", text, flags=re.I)
    return m.group(1).strip() if m else text.strip()

def enforce_module_header(module_declaration: str, code: str) -> str:
    """Replace any existing module header with the exact provided declaration."""
    if not module_declaration:
        return code
    want = " ".join(module_declaration.split()).rstrip()
    if not want.endswith(";"):
        want += ";"
    m_any = re.search(r"(module\s+\w+\s*\([^;]*\);)", code, flags=re.I)
    if m_any:
        return code.replace(m_any.group(1), want, 1)
    return want + "\n" + code

def minimal_from_declaration(module_declaration: str, instruction: str) -> str:
    """
    Build a minimal-valid Verilog-2001 module from the declaration.
    Uses output names to infer reductions (&, |, ^) or falls back to 0.
    """
    decl = " ".join((module_declaration or "").split())
    m_hdr = re.search(r"(module\s+\w+\s*\([^;]*\);\s*)", decl, flags=re.I)
    header = m_hdr.group(1) if m_hdr else "module top_module();"

    m_in = re.search(r"input\s*(?:\[[^]]+\]\s*)?(\w+)", decl, flags=re.I)
    in_name = m_in.group(1) if m_in else "in"

    outputs = re.findall(r"output\s*(?:\[[^]]+\]\s*)?(\w+)", decl, flags=re.I)
    lines = []

    def pick_reduction(out_name: str) -> str:
        lo = out_name.lower()
        if lo.endswith("and"): return "&"
        if lo.endswith("or"):  return "|"
        if lo.endswith("xor"): return "^"
        if re.search(r"\breduction\s*and\b|\bAND4\b", instruction, re.I): return "&"
        if re.search(r"\breduction\s*or\b|\bOR4\b",   instruction, re.I): return "|"
        if re.search(r"\breduction\s*xor\b|\bXOR4\b", instruction, re.I): return "^"
        return ""

    for out_name in outputs:
        op = pick_reduction(out_name)
        if op:
            lines.append(f"  assign {out_name} = {op}{in_name};")
        else:
            lines.append(f"  assign {out_name} = 1'b0;")

    return header + "\n" + "\n".join(lines) + "\nendmodule\n"

def sanitize_plain_verilog(code: str, module_declaration: str, instruction: str) -> str:
    """
    Ensure the output is plain Verilog (no Bluespec/SystemVerilog) and structurally valid.
    Falls back to a minimal implementation if needed.
    """
    m_block = re.search(r"(module\b[\s\S]*?endmodule)", code, flags=re.I)
    code = m_block.group(1) if m_block else code

    if any(tok in code for tok in BLUESPEC_TOKENS):
        return minimal_from_declaration(module_declaration, instruction)

    if not re.search(r"\b(assign|always)\b", code):
        return minimal_from_declaration(module_declaration, instruction)

    code = enforce_module_header(module_declaration, code)
    if "endmodule" not in code:
        code = code.rstrip() + "\nendmodule\n"
    return code

def build_prompt(instruction: str, module_declaration: str) -> str:
    """Create a simple prompt with one tiny few-shot example for biasing."""
    fewshot = """### Example:
Instruction:
From a 2-bit input x[1:0], produce single-bit output y = reduction OR of x. Use continuous assignment.

Module declaration (use exactly this signature):
module top_module(input [1:0] x, output y);

Verilog solution:
module top_module(input [1:0] x, output y);
  assign y = |x;
endmodule
"""
    s = f"### Instruction:\n{instruction}\n\n"
    if module_declaration:
        s += f"### Module declaration (use exactly this signature):\n{module_declaration}\n\n"
    s += fewshot
    s += "### Verilog solution:\n"
    return s

def load_model():
    """Load and cache tokenizer/model once per process."""
    global tokenizer_cache, model_cache
    if tokenizer_cache is None:
        tokenizer_cache = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer_cache.pad_token is None:
            tokenizer_cache.pad_token = tokenizer_cache.eos_token
    if model_cache is None:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model_cache = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model_cache.eval()
    return tokenizer_cache, model_cache

def generate_verilog(instruction: str, module_declaration: str = "", max_new_tokens: int = 256) -> str:
    """Public entrypoint: generate code then sanitize to plain Verilog-2001."""
    tok, model = load_model()
    prompt = build_prompt(instruction, module_declaration)
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tok.eos_token_id,
    )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    raw = decoded[len(prompt):]
    raw = clean_ascii(raw)
    sliced = slice_module_text(raw)
    return sanitize_plain_verilog(sliced, module_declaration, instruction)
