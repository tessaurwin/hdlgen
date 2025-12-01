# src/finetune_from_hdljson.py
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "hdl_modules.clean.json"
OUT_DIR = ROOT / "checkpoints" / "tinyllama_lora_hdl"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 1     # TODO: try 2 if time allows
LR = 2e-4          # TODO: sweep 1e-4..3e-4 later

def load_rows(path: Path):
    data = json.loads(path.read_text())
    rows = []
    for ex in data:
        instr = (ex.get("instruction") or "").strip()
        decl  = (ex.get("module_declaration") or "").strip()
        out   = (ex.get("output") or "").strip()
        if instr and out:
            s = f"### Instruction:\\n{instr}\\n"
            if decl:
                s += f"\\n### Module declaration:\\n{decl}\\n"
            s += f"\\n### Verilog solution:\\n{out}\\n"
            rows.append({"text": s})
    if not rows:
        raise ValueError("dataset had zero usable rows")
    return rows

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(DATA_FILE)
    ds = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(batch):
        t = tok(batch["text"], max_length=MAX_LEN, truncation=True, padding="max_length")
        t["labels"] = t["input_ids"].copy()
        return t

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"],  # llama-style
    )
    model = get_peft_model(base, lora)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "runs"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False, bf16=False,
        report_to=[],  # no wandb/etc
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds_tok)
    print("Starting LoRA fine-tuning…")
    trainer.train()

    print(f"Saving adapter to {OUT_DIR}")
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()

