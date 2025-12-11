# Finetuning TinyLlama for Verilog Generation with LoRA

This repo evaluates a small language model (TinyLlama-1.1B) on Verilog code generation, then fine-tunes it with LoRA and re-evaluates. It also checks generated modules with lightweight randomized testbenches.

## Authors

Tessa Urwin
Julian O'Hern

ECE 465 Digital Systems Design Fall 2025 Graduate Project

## TLDR/our results

- Check	            Baseline	    Trained (LoRA)  Improvmenet:
- Verible syntax	    12/22 (54.5%)	21/22 (95.5%)	+40.9%
- Verilator compile	11/22 (50.0%)	18/22 (81.8%)	+31.8%
- Yosys synth	        12/22 (54.5%)	18/22 (81.8%)	+27.3%
- Testbenches	        13/22 (59.1%)	18/22 (81.8%)	+22.7%

## What this repo does

- Prompts TinyLlama-1.1B to generate Verilog for a set of tasks (data/tasks.jsonl).
- Runs three static checks on each module:
    - Verible formatting and basic syntax evaluation
    - Verilator compilation ability
    - Yosys read/synth success
- Fine-tunes the same base model using LoRA (PEFT + HF Transformers), then repeats the exact pipeline.
- Optionally runs simple random testbenches using Icarus Verilog (iverilog + vvp) to test actual module behavior.

## Repo Layout

-data/
  -tasks.jsonl                    # eval tasks (id + prompt + module_declaration)
  -training_dataset.*             # small HDL training set

-outputs/
  -v_base/                        # baseline model .v outputs (per task)
  -v_lora/                        # LoRA model .v outputs (per task)
  -tb_dump_base/, tb_dump_lora/   # generated testbenches

-results/
  -base_*.csv / lora_*.csv        # CSVs from each phase
  -*_summary.txt                  # one-line summaries

-src/
  -base_eval_many.py              # run baseline gen + checks over all tasks
  -llm_eval_many_new.py           # run LoRA gen + checks over all tasks
  -generate_with_lora_new.py      # simple generate_verilog() helper
  -finetune_from_hdljson_new.py   # LoRA fine-tuning script
  -auto_tb_single.py              # build + run minimal testbenches on a folder of .v files

## Setup

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

## Phase 1 — Baseline evaluation

Generate and evaluate TinyLlama-1.1B on all tasks.
Generate & run checks, write Verilog into outputs/v_base and a CSV into results/:

python src/base_eval_many.py --tasks data/tasks.jsonl --out results/base_baseline.csv

What it does:
- Prompts the base model for each task in data/tasks.jsonl.
- Saves a task.v into outputs/v_base/.
- Runs Verible, Verilator, Yosys for each file and logs pass/fail to the CSV.

## Phase 2 — LoRA fine-tune + evaluation

Fine-tune with LoRA:

python src/finetune_from_hdljson_new.py --train data/training_dataset.json --base tinyllama/TinyLlama-1.1B-Chat-v1.0 --out_dir runs/lora_tinyllama --lr 2e-4 --epochs 3 --batch_size 8

This produces LoRA weights in runs/lora_tinyllama/.

Generate and evaluate with the trained adapter:

python src/llm_eval_many_new.py --tasks data/tasks.jsonl --out results/lora_eval.csv --lora_dir runs/lora_tinyllama

Outputs:
- Verilog files in outputs/v_lora/
- Metrics table in results/lora_eval.csv

## Phase 3 — Functional sanity via tiny testbenches

We do a dumb but useful simulation test using Icarus Verilog (iverilog + vvp). For each .v file:
- Auto-rename the first module to dut
- Parse the port list from the known declaration
- Generate a tiny testbench that randomly toggles inputs for N iterations
- Count PASS / COMPILE_FAIL per task

Run it once for each the baseline model and the trained model verilog outputs:

Baseline:
python src/auto_tb_single.py --tasks data/tasks.jsonl --dir outputs/v_base --iters 200 --dump_dir outputs/tb_dump_base --out results/base_tb.csv

Trained:
python src/auto_tb_single.py --tasks data/tasks.jsonl --dir outputs/v_lora --iters 200 --dump_dir outputs/tb_dump_lora --out results/lora_tb.csv

Each run prints a one-line summary and writes *_tb_summary.txt.

Icarus Verilog compiles your Verilog testbench and DUT into a sim binary. iverilog does the compile. vvp runs the compiled sim and prints your $display output.

## Reproducing:

1. Run Phase 1 baseline.
2. Train LoRA, then run Phase 2.
3. Run Phase 3 on both outputs/v_base and outputs/v_lora.
4. Compare:
    Verible: baseline 12/22, trained 21/22
    Verilator: baseline 11/22, trained 18/22
    Yosys: baseline 12/22, trained 18/22
    Testbenches: baseline 13/22, trained 18/22

CSV artifacts live in results/. Generated HDL lives in outputs/v_base and outputs/v_lora.

Notes pls do not @ me on thesse:

- Training set is small. Around 30+ modules. The goal was to successfully fine-tune locally and see improvement. Larger dataset will obviously give more roburst results.
- Testbenches are minimal. They randomize inputs and check for compile failures. They are not amazing functional tests.
- Model is TinyLlama-1.1B with LoRA adapters via PEFT. You can swap the base or tune the LoRA hyper-params if desired.

## Acknowledgments because a README tutorial told me I should include these

- TinyLlama model authors
- Hugging Face Transformers, PEFT
- Verible, Verilator, Yosys, Icarus Verilog
  
