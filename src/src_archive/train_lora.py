# train_lora.py
from pathlib import Path
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
train_path = Path("data/hdl_train.jsonl")
out_dir = Path("checkpoints/tinyllama_lora_hdl_v2")
out_dir.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj","k_proj","o_proj"])
model = get_peft_model(model, lora)

def make_prompt(ex):
    p = []
    p.append("### Instruction:\n" + ex["instruction"])
    if ex.get("module_declaration"):
        p.append("### Module declaration (use exactly this signature):\n" + ex["module_declaration"])
    p.append("### Verilog solution:\n")
    return "\n\n".join(p)

train = []
for line in train_path.read_text().splitlines():
    ex = json.loads(line)
    prompt = make_prompt(ex)
    target = ex["output"]
    text = prompt + target  # supervised causal LM (inputs + labels)
    enc = tok(text, truncation=True, max_length=2048)
    enc["labels"] = enc["input_ids"].copy()
    train.append(enc)

class DS(torch.utils.data.Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

args = TrainingArguments(
    output_dir=str(out_dir),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch"
)

trainer = Trainer(model=model, args=args, train_dataset=DS(train), tokenizer=tok)
trainer.train()
model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print("Saved LoRA adapter to", out_dir)

