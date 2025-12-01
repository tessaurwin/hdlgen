# src/finetune_tinyllama_lora_partner.py

import os
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hdl_prompts.json"
OUT_DIR = ROOT / "checkpoints" / "tinyllama_lora_verilog"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# --------- 1) Load + clean dataset from your JSON file ---------

def load_and_split_dataset():
    """
    Loads your JSON array file, drops empty entries, then splits into train/test.
    """

    # datasets can load a JSON array directly:
    raw_ds = load_dataset("json", data_files={"all": str(DATA_PATH)})["all"]

    # filter out rows where instruction or output is empty
    def non_empty(example):
        instr = example.get("instruction", "")
        out = example.get("output", "")
        return bool(instr.strip()) and bool(out.strip())

    cleaned = raw_ds.filter(non_empty)
    print(f"Total non-empty examples: {len(cleaned)}")

    # create train/test indices (80% / 20%)
    indices = list(range(len(cleaned)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    train_ds = cleaned.select(train_idx)
    test_ds  = cleaned.select(test_idx)

    print(f"Train examples: {len(train_ds)}, Test examples: {len(test_ds)}")

    return {"train": train_ds, "test": test_ds}


# --------- 2) Tokenizer + text formatting ---------

def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_text(example):
    """
    Build the actual text we train on using your fields:
      - instruction
      - module_declaration
      - output
    """
    instr = example.get("instruction", "").strip()
    decl  = example.get("module_declaration", "").strip()
    out   = example.get("output", "").strip()

    # In many entries 'output' already contains the whole module.
    # Including module_declaration explicitly just reinforces structure.
    text = (
        "### Prompt:\n"
        f"{instr}\n\n"
        "### Module declaration:\n"
        f"{decl}\n\n"
        "### Verilog solution:\n"
        f"{out}\n"
    )

    return {"text": text}


def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding=False,  # dynamic padding later
    )


# --------- 3) Build TinyLlama + LoRA ---------

def build_lora_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,   # if bitsandbytes works; we can switch to fp16/cpu if needed
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# --------- 4) Main ---------

if __name__ == "__main__":
    # 1) load + split
    ds_dict = load_and_split_dataset()

    # 2) tokenizer
    tokenizer = build_tokenizer()

    # 3) build "text" field from your (instruction, module_declaration, output)
    ds_with_text = {
        split: ds.map(build_text)
        for split, ds in ds_dict.items()
    }

    # 4) tokenize
    tokenized = {
        split: d.map(
            tokenize_function,
            batched=True,
            remove_columns=d.column_names,  # keep only tokenized stuff
        )
        for split, d in ds_with_text.items()
    }

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5) build model
    model = build_lora_model()

    # 6) training config
    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to=[],  # no wandb
        bf16=torch.cuda.is_available(),  # if GPU supports it; on CPU it's ignored
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # 7) save LoRA adapter + tokenizer
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    print(f"✅ Saved TinyLlama LoRA adapter to {OUT_DIR}")
