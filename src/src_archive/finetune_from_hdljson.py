# src/finetune_from_hdljson.py
import json
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ---- paths ----
ROOT = Path(__file__).resolve().parents[1]
# rename your file to this, or change this path to match reality:
DATA_FILE = ROOT / "data" / "hdl_modules.clean.json"
OUT_DIR = ROOT / "checkpoints" / "tinyllama_lora_hdl"

# ---- model + training hyperparams (CPU friendly) ----
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 1          # bump to 2–3 later if time allows
LR = 2e-4


def load_dataset_exact_schema(path: Path):
    """
    Expect a JSON ARRAY with objects like:
      {
        "id": "42",
        "instruction": "...",
        "module_declaration": "...",
        "output": "..."
      }
    We keep rows where instruction and output are non-empty.
    """
    data = json.loads(path.read_text())
    cleaned = []
    for ex in data:
        instr = (ex.get("instruction") or "").strip()
        decl  = (ex.get("module_declaration") or "").strip()
        out   = (ex.get("output") or "").strip()
        if not instr or not out:
            continue
        cleaned.append({"instruction": instr, "module_declaration": decl, "output": out})
    if not cleaned:
        raise ValueError("No usable examples (instruction/output empty).")
    return cleaned


def build_text(ex: dict) -> str:
    # Instruction + optional module_declaration + gold Verilog
    instr = ex["instruction"]
    decl  = ex["module_declaration"]
    out   = ex["output"]
    s = f"### Instruction:\n{instr}\n"
    if decl:
        s += f"\n### Module declaration:\n{decl}\n"
    s += f"\n### Verilog solution:\n{out}\n"
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_FILE}")

    rows = load_dataset_exact_schema(DATA_FILE)
    ds = Dataset.from_list([{"text": build_text(r)} for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # LoRA adapter
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # LLaMA-style
    )
    model = get_peft_model(model, lora_cfg)

    def tokenize(batch):
        tok = tokenizer(
            batch["text"],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tok_ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "runs"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=tok_ds)
    print("Starting LoRA fine-tuning…")
    trainer.train()

    print(f"Saving LoRA adapter + tokenizer to {OUT_DIR}")
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

