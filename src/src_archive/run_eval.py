# src/run_eval.py (original script)
import subprocess
import json
import os
from datetime import datetime

# 1 PROMPT TO SEND TO TINYLLAMA
prompt = """
Write a simple Verilog module called 'and_gate' that takes two inputs (a, b)
and outputs y = a & b.
"""

# 2 CALL THE OLLAMA API (local inference)
print("Querying TinyLlama…")
result = subprocess.run(
    ["ollama", "run", "tinyllama"],
    input=prompt,
    capture_output=True,
    text=True
)

verilog_code = result.stdout.strip()

# 3 SAVE THE OUTPUT TO A FILE
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)
verilog_path = os.path.join(output_dir, f"gen_{timestamp}.v")

with open(verilog_path, "w") as f:
    f.write(verilog_code)

print(f"Verilog code saved to {verilog_path}")

# 4 LINT THE CODE USING VERIBLE
print("Running Verible lint…")
lint = subprocess.run(
    ["verible-verilog-lint", verilog_path],
    capture_output=True,
    text=True
)
print(lint.stdout or lint.stderr)

# 5 SIMULATE USING VERILATOR
print("Running Verilator simulation…")
verilate = subprocess.run(
    ["verilator", "--lint-only", verilog_path],
    capture_output=True,
    text=True
)
print(verilate.stdout or verilate.stderr)

# 6 LOG RESULTS
results_path = "../results"
os.makedirs(results_path, exist_ok=True)
with open(os.path.join(results_path, "summary.txt"), "a") as log:
    log.write(f"\n[{timestamp}]\n{verilog_path}\n")
    log.write(f"Verible: {lint.returncode}, Verilator: {verilate.returncode}\n")
print("Results appended to results/summary.txt")
