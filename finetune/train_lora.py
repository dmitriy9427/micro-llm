"""Дообучение Qwen2.5-0.5B-Instruct методом LoRA на вопросах про резюме.

LoRA не меняет исходные веса: к линейным слоям добавляются две маленькие матрицы (ранг r),
обучаются только они — это ~1% параметров, поэтому хватает ноутбука.

    python train_lora.py --epochs 4
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import messages

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--dtype", default="bfloat16", help="bfloat16 вдвое экономит память; float32 точнее")
ap.add_argument("--out", default=str(HERE / "out/lora"))
args = ap.parse_args()

torch.manual_seed(0)
random.seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(device)
model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()


def encode(row):
    """Токены чата; учимся только на ответе ассистента (остальное маскируем значением -100)."""
    prompt = tok.apply_chat_template(messages(row["question"]), tokenize=False, add_generation_prompt=True)
    full = prompt + row["answer"] + "<|im_end|>\n"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids, labels


rows = [json.loads(l) for l in open(HERE / "data/train.jsonl", encoding="utf-8")]
data = [encode(r) for r in rows]
print(f"примеров: {len(data)}, макс. длина {max(len(x[0]) for x in data)} токенов | устройство: {device}")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
steps = args.epochs * ((len(data) + args.batch - 1) // args.batch)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / steps))
model.train()
t0, step = time.time(), 0
for epoch in range(args.epochs):
    random.shuffle(data)
    for i in range(0, len(data), args.batch):
        batch = data[i:i + args.batch]
        n = max(len(x[0]) for x in batch)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (n - len(x[0])) for x in batch], device=device)
        lab = torch.tensor([x[1] + [-100] * (n - len(x[1])) for x in batch], device=device)
        att = torch.tensor([[1] * len(x[0]) + [0] * (n - len(x[0])) for x in batch], device=device)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        if step % 10 == 0 or step == steps:
            print(f"эпоха {epoch + 1}/{args.epochs} шаг {step}/{steps} | потери {loss.item():.3f} | {time.time() - t0:.0f} с", flush=True)

Path(args.out).mkdir(parents=True, exist_ok=True)
model.save_pretrained(args.out)
print("сохранено в", args.out)
