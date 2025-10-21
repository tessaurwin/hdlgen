# HDLGen — Verilog Generation & Evaluation

Pipeline to generate Verilog from text prompts and evaluate with Verible (lint), Verilator (compile), and Yosys (synth).

## Quickstart
1) Install tools (macOS):
   - Verible, Verilator, Yosys (brew) — already installed.
   - Ollama (for local LLM): https://ollama.com — `brew install --cask ollama`, then `ollama pull tinyllama`.

2) Run baseline eval:
   ```bash
   python3 scripts/eval.py

