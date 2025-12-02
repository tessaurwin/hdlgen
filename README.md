## Finetuning TinyLlama for Verilog Generation with LoRA

This repo evaluates a small language model (TinyLlama-1.1B) on Verilog code generation, then fine-tunes it with LoRA and re-evaluates. It also checks generated modules with lightweight randomized testbenches.

# What this repo does

- Prompts TinyLlama-1.1B to generate Verilog for a set of tasks (data/tasks.jsonl).
- Runs three static checks on each module:
    - Verible formatting and basic syntax evaluation
    - Verilator compilation ability
    - Yosys read/synth success
- Fine-tunes the same base model using LoRA (PEFT + HF Transformers), then repeats the exact pipeline.
- Optionally runs simple random testbenches using Icarus Verilog (iverilog + vvp) to test actual module behavior.

# Repo Layout

data/
  tasks.jsonl                    # eval tasks (id + prompt + module_declaration)
  training_dataset.*             # small HDL training set

outputs/
  v_base/                        # baseline model .v outputs (per task)
  v_lora/                        # LoRA model .v outputs (per task)
  tb_dump_base/, tb_dump_lora/   # generated testbenches

results/
  base_*.csv / lora_*.csv        # CSVs from each phase
  *_summary.txt                  # one-line summaries

src/
  base_eval_many.py              # run baseline gen + checks over all tasks
  llm_eval_many_new.py           # run LoRA gen + checks over all tasks
  generate_with_lora_new.py      # simple generate_verilog() helper
  finetune_from_hdljson_new.py   # LoRA fine-tuning script
  auto_tb_single.py              # build + run minimal testbenches on a folder of .v files

# Setup

Python:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # if present
(or:
pip install torch transformers peft accelerate tqdm)

HDL tools (macOS)
brew install verible verilator yosys icarus-verilog

Optional: quiet HF tokenizer warnings because they are personally bothersome to me in terminal
export TOKENIZERS_PARALLELISM=false


