# HDLGEN - Finetuning a Lightweight LLM for Verilog Generation

This project evaluates whether a LoRA-finetuned TinyLlama generates more valid Verilog code than the base model on a fixed set of HDL tasks. We compare both models on the same 22 tasks and score outputs with three automated checks:

- Verible syntax check
- Verilator compile
- Yosys synthesize

# Authors

Tessa Urwin
Julian O'Hern

# Overview

0) create env
1) optional: clean the dataset if it was edited it by hand
2) fine-tune TinyLlama with LoRA using our JSON dataset
3) generate a single module with guardrails + header enforcement
4) batch eval many tasks, save code, logs, and a CSV summary

# Repo Layput

data/
  tasks.jsonl                 # 22 tasks stored as a JSON array
results/
  base_baseline.csv           # baseline TinyLlama results
  lora_baseline.csv           # LoRA TinyLlama results
src/
  generate_baseline.py        # baseline model generator
  generate_with_lora_new.py   # LoRA model generator
  tools.py                    # wrappers for verible, verilator, yosys
  llm_eval_many_base.py       # baseline batch eval
  llm_eval_many_new.py        # LoRA batch eval
  finetune_from_hdljson*.py   # finetuning entry points
  train_lora.py               # optional training helper
outputs/
  v_base/  logs_base/         # baseline code and logs
  v_lora/  logs_lora/         # LoRA code and logs

# Setup

Both pipelines point at data/tasks.jsonl, a JSON array with 22 tasks. The LoRA generator is generate_with_lora_new.py. The baseline generator is generate_baseline.py. Batch evaluation scripts are llm_eval_many_new.py and llm_eval_many_base.py.

Requirements
- Python 3.10+
- macOS or Linux
- Command-line Verible, Verilator, and Yosys in PATH for evaluation
- Internet access to download the HuggingFace model on the first run (TinyLlama)

Environment
python3 -m venv .venv
source .venv/bin/activate
pip install "torch>=2.3,<3.0" --index-url https://download.pytorch.org/whl/cpu
pip install "transformers[torch]" datasets peft accelerate scikit-learn

# Data

Training format (data/hdl_modules.clean.json):

data/tasks.json is a json array of objects where each tasks looks like this:
{
  "id": "vector_part_select",
  "instruction": "Reverse the byte ordering of a 32-bit word.",
  "module_declaration": "module top_module(input [31:0] in, output [31:0] out);",
  "output": "module top_module(...);\n  ...\nendmodule\n"
}

If edited by hand and have possible raw newlines/tabs inside JSON strings, then fix:
python src/fix_hdl_modules.py
(writes data/hdl_modules.clean.json)

Eval Tasks (data/taskts.jsonl):

JSON array of objects used for batch evaluation. Each task has:
{"id":"and2","top":"and2","prompt":"2-input AND gate…","module_decl":"module top_module(...);" }

# Training (LoRA with PEFT)

We adapt TinyLlama/TinyLlama-1.1B-Chat-v1.0 using LoRA (low rank adapters). LoRA trains a small set of adapter weights while freezing the base model. This is fast to train, easy to adjust, and well-suited for small HDL datasets.

python src/finetune_from_hdljson.py
(outputs: checkpoints/tinyllama_lora_hdl/)

- BASE_MODEL: TinyLlama chat
- MAX_LEN=512, BATCH_SIZE=1, GRAD_ACCUM=4, NUM_EPOCHS=1..3, LR=2e-4
- LoRA targets: ["q_proj","k_proj","v_proj","o_proj"]

# Generation (single sample)

We use the tuned adapter with strict guardrails:
- Forces Verilog-2001 only
- Enforces exact module_declaration header
- Filters Bluespec/SystemVerilog (invalid)
- Falls back to a minimal valid module if the model drifts

  python src/generate_with_lora.py \
  --instruction "From a 4-bit input in[3:0], compute out_and=&in, out_or=|in, out_xor=^in." \
  --module_decl "module top_module(input [3:0] in, output out_and, output out_or, output out_xor);" \
  --out outputs/v_lora/smoke.v

# Batch Evaluation

This runs: generate → verible → verilator → yosys, logs everything, and writes a CSV.

TOKENIZERS_PARALLELISM=false PYTHONPATH=src python src/llm_eval_many.py \
  --tasks data/tasks.jsonl \
  --out   results/lora_baseline.csv \
  --vdir  outputs/v_lora \
  --logdir outputs/logs_lora

CSV columns:

task_id
verible_ok (1/0)
lint_warnings (count)
verilator_ok (1/0)
yosys_ok (1/0)
timestamp

# How it works (high level)

1. LoRA fine-tune: We teach TinyLlama short instruction→module mappings. LoRA keeps training small and reversible.

2. Guardrailed decoding: Prompts require Verilog-2001 only, one module, exact header. Decoding is deterministic (no sampling).

3. Sanitization & fallback: If output looks wrong (Bluespec tokens, missing endmodule, weird unicode), we clean it or synthesize a minimal valid module from the declaration.

4. Objective checks: Verible (style/syntax), Verilator (compiles), Yosys (synthesizes). No vibes, just gates.

5. Reproducibility: venv, pinned scripts, deterministic decoding, saved artifacts and logs.

# Results (example)

Baseline TinyLlama:

Verible 16 of 22 about 73 percent
Verilator 1 of 22 about 4.5 percent
Yosys 0 of 22 0 percent

LoRA TinyLlama:

Verible 19 of 22 about 86 percent
Verilator 18 of 22 about 82 percent
Yosys 19 of 22 about 86 percent

This will vary with dataset size, types of tasks, and prompt phrasing.

# Conclusion

Fine-tuning gave a large jump in tool-verified correctness. The LoRA model moves from mostly clean text to code that compiles and synthesizes on most tasks.

# Evidence

See results/base_baseline.csv and results/lora_baseline.csv. Per-task logs are under outputs/.

# Takeaways

- Enforcing header guidelines helps to avoid "no top module" in synthesis.
- To just clean the code (linting) is not enough as the baseline will frequently pass when evaluated on valid syntax, but then rarely compiles or synthesizes. The LoRA model improves this.
- 

# Next Steps

1. Scan LoRA failures in Verilator and Yosys logs to address remaining errors
2. Add targeted training examples for those models that failed
3. Re-run and record changes in performance

Additionally, expand the dataset to cover even more breadth of Verilog module tasks.

# Acknowledgements
- TinyLlama (Hugging Face)
- PEFT / LoRA (parameter efficient finetuning)
- Transformers
- Verible, Verilator, Yosys (open-source HDL tooling)



