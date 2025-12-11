# src/finetune_from_hdljson_new.py
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "training_dataset.json"
OUT_DIR = ROOT / "checkpoints" / "tinyllama_lora_hdl"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LEN = 384
BATCH_SIZE = 2
GRAD_ACCUM = 4
NUM_EPOCHS = 3     # try diff vals later if time allows
LR = 1e-4


def _read_dataset_any(path: Path):
    txt = path.read_text(encoding="utf-8").strip()
    # try json array
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # fallback jsonl
    rows = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        rows.append(json.loads(ln))
    return rows

def _resolve_output_path(base_file: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    # prefer alongside DATA_FILE parent (like data/…)
    cand = base_file.parent / rel
    if cand.exists():
        return cand
    # try repo root
    cand = ROOT / rel
    if cand.exists():
        return cand
    # final data/rel
    return ROOT / "data" / rel

def load_training_rows(path: Path):
    """Each row supports: id, instruction, module_declaration (opt),
       and either output (inline code) OR output_path (string)."""
    raw = _read_dataset_any(path)
    rows = []
    for ex in raw:
        instruction = (ex.get("instruction") or "").strip()
        decl = (ex.get("module_declaration") or "").strip()
        output = (ex.get("output") or "").strip()

        if not output:
            op = ex.get("output_path")
            if op:
                vf = _resolve_output_path(path, op)
                if vf.is_file():
                    output = vf.read_text(encoding="utf-8").replace("\r\n", "\n").strip()

        if not instruction or not output:
            continue

        if "endmodule" not in output:
            output = output.rstrip() + "\nendmodule\n"
        elif not output.endswith("\n"):
            output += "\n"

        text = f"### Instruction:\n{instruction}\n"
        if decl:
            text += f"\n### Module declaration:\n{decl}\n"
        text += f"\n### Verilog solution:\n{output}\n"

        rows.append({"text": text})

    if not rows:
        raise ValueError("Dataset had zero usable rows")
    return rows

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = load_training_rows(DATA_FILE)
    ds = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch):
        enc = tokenizer(batch["text"], max_length=MAX_LEN, truncation=True, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds_tokenized = ds.map(tokenize_batch, batched=True, remove_columns=["text"])

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)

    # initially use default params, but can change global vars at top for different iterations
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
        report_to=[],
    )

    print(f"Loaded {len(examples)} training rows from {DATA_FILE}")
    trainer = Trainer(model=model, args=args, train_dataset=ds_tokenized)
    print("Starting LoRA fine-tuning…")
    trainer.train()

    print(f"Saving adapter to {OUT_DIR}")
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
