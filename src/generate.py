# src/generate.py
import subprocess, re

TEMPLATE = """You are an expert digital designer.
Write **pure Verilog-2001** ONLY (no Bluespec, no SystemVerilog), in a single module named {top}.
Constraints:
- No `rule`, `method`, `mkReg*`, or other Bluespec constructs.
- No `interface`, `logic`, `unique`, or typedefs (stick to Verilog-2001).
- No delays (#) or initial blocks.
- Include all port declarations and end with `endmodule`.
Task: {spec}
Return only the Verilog code. No markdown fences, no comments before/after.
"""

def call_ollama(prompt: str, model: str = "tinyllama") -> str:
    # uses `ollama run` so it stays local/offline
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

def extract_verilog(text: str) -> str:
    # try to grab fenced ```verilog blocks first (if any)
    m = re.search(r"```(?:verilog|systemverilog)?\s*(.*?)```", text, re.S|re.I)
    if m:
        return m.group(1).strip()
    # otherwise strip accidental prose and keep lines from 'module' to 'endmodule'
    m2 = re.search(r"(module\b.*?endmodule)", text, re.S)
    return m2.group(1).strip() if m2 else text.strip()

def generate(spec: str, top: str, model="codellama:7b-instruct") -> str:
    prompt = TEMPLATE.format(spec=spec, top=top)
    raw = call_ollama(prompt, model=model)
    return extract_verilog(raw)
