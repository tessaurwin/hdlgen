import re, sys, argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer_cache = None
model_cache = None

def load_model():
    global tokenizer_cache, model_cache
    if tokenizer_cache is None:
        tokenizer_cache = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer_cache.pad_token is None:
            tokenizer_cache.pad_token = tokenizer_cache.eos_token
    if model_cache is None:
        model_cache = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model_cache.eval()
    return tokenizer_cache, model_cache

def build_prompt(instruction: str, module_declaration: str) -> str:
    s = f"### Instruction:\n{instruction}\n\n"
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
    return code.replace(m.group(1), want, 1) if m else want + "\n" + code

def generate_verilog(instruction: str, module_declaration: str = "", max_new_tokens: int = 256) -> str:
    tok, model = load_model()
    prompt = build_prompt(instruction, module_declaration)
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    gen = text[len(prompt):]
    code = enforce_header(module_declaration, slice_module(gen))
    if "endmodule" not in code:
        code = code.rstrip() + "\nendmodule\n"
    return code

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", required=True)
    p.add_argument("--module_decl", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()
    code = generate_verilog(args.instruction, args.module_decl)
    if args.out:
        Path(args.out).write_text(code); print(f"wrote {args.out}")
    else:
        sys.stdout.write(code + "\n")

