import subprocess
import json
import os
import re
from datetime import datetime

# run_eval.py but filters LLM output:
# forces code-only output from the model
# strips any non-Verilog text before saving


MODEL = "codellama:7b-instruct"

prompt = """You are a Verilog code generator. Respond ONLY with valid Verilog code.
Do not include explanations, markdown, or backticks.
Write a simple Verilog module named 'and_gate' that takes two inputs (a, b) and outputs y = a & b.
"""

print(f"Querying {MODEL}…")
result = subprocess.run(
    ["ollama", "run", MODEL],
    input=prompt,
    capture_output=True,
    text=True
)

verilog_code = result.stdout.strip()

# Extract only Verilog between 'module' and 'endmodule'
match = re.search(r"(module .*?endmodule)", verilog_code, re.S)
if match:
    verilog_code = match.group(1)
else:
    print("Warning: No Verilog module detected.")

# Save to outputs directory
output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
verilog_path = os.path.join(output_dir, f"gen_{timestamp}.v")

with open(verilog_path, "w") as f:
    f.write(verilog_code)

print(f"✅ Verilog code saved to {verilog_path}")

# Lint using Verible
print("Running Verible lint…")
lint = subprocess.run(["verible-verilog-lint", verilog_path], capture_output=True, text=True)
print(lint.stdout or lint.stderr)

# Simulate using Verilator
print("Running Verilator syntax check…")
verilate = subprocess.run(["verilator", "--lint-only", verilog_path], capture_output=True, text=True)
print(verilate.stdout or verilate.stderr)

# Log results
results_path = "../results"
os.makedirs(results_path, exist_ok=True)

with open(os.path.join(results_path, "summary.txt"), "a") as log:
    log.write(f"\n[{timestamp}] {verilog_path}\n")
    log.write(f"Verible: {lint.returncode}, Verilator: {verilate.returncode}\n")

print("Results appended to results/summary.txt")
