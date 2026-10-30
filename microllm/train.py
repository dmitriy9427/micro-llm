"""Обучение: много раз берём пачку текста, считаем ошибку предсказаний и чуть подправляем веса."""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .checkpoint import pick_device, save_checkpoint
from .config import GPTConfig, preset
from .data import get_batch, load_text, split_tokens
from .generate import generate_text
from .model import GPT
from .tokenizer import build_tokenizer, save_tokenizer


def lr_at(step: int, max_steps: int, lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Скорость обучения: сначала плавный разгон, потом плавное снижение по косинусу."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return lr * (min_ratio + (1 - min_ratio) * cosine)


@torch.no_grad()
def estimate_loss(model: GPT, data: dict, batch_size: int, block_size: int, device, iters: int = 20) -> dict:
    model.eval()
    out = {}
    for split, d in data.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(d, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    text = load_text(args.corpus)
    tok = build_tokenizer(args.tokenizer, text, args.vocab_size)
    ids = tok.encode(text)
    train_data, val_data = split_tokens(ids)
    cfg = preset(args.preset, tok.vocab_size)
    if args.block_size:
        cfg.block_size = args.block_size
    model = GPT(cfg).to(device)
    opt = model.configure_optimizer(args.lr, args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)
    save_tokenizer(tok, os.path.join(args.out_dir, "tokenizer.json"))
    log_path = os.path.join(args.out_dir, "log.jsonl")
    open(log_path, "w").close()

    print(f"устройство: {device} | токенайзер: {tok.kind}, словарь {tok.vocab_size} | "
          f"текст: {len(text):,} симв. → {len(ids):,} токенов")
    print(f"модель: {args.preset}, параметров: {model.num_params():,} | "
          f"шагов: {args.max_steps}, батч {args.batch_size}×{cfg.block_size}")

    data = {"train": train_data, "val": val_data}
    best_val = float("inf")
    t0 = time.time()
    last = {}
    model.train()
    for step in range(args.max_steps + 1):
        lr = lr_at(step, args.max_steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        if step % args.eval_interval == 0 or step == args.max_steps:
            last = estimate_loss(model, data, args.batch_size, cfg.block_size, device)
            elapsed = time.time() - t0
            print(f"шаг {step:5d} | обучение {last['train']:.3f} | проверка {last['val']:.3f} | "
                  f"lr {lr:.2e} | {elapsed:5.0f} с")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "train": last["train"], "val": last["val"], "lr": lr,
                                    "time": elapsed}) + "\n")
            if last["val"] < best_val:
                best_val = last["val"]
                save_checkpoint(os.path.join(args.out_dir, "ckpt.pt"), model, tok, {"step": step, "val": best_val})
        if step == args.max_steps:
            break

        x, y = get_batch(train_data, args.batch_size, cfg.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # не даём градиенту «взорваться»
        opt.step()

    print(f"\nлучшая проверочная ошибка: {best_val:.3f} (сохранена в {args.out_dir}/ckpt.pt)")
    if not args.no_sample:
        print("\n── пример текста ───────────────────────────────────────")
        print(generate_text(model, tok, args.prompt, device, 300, temperature=0.8, top_k=40))
    return {"train": last["train"], "val": last["val"], "best_val": best_val, "params": model.num_params()}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Обучить микро-GPT на текстовом файле")
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--vocab-size", type=int, default=1024, help="размер словаря для bpe")
    ap.add_argument("--preset", choices=["tiny", "small", "base"], default="small")
    ap.add_argument("--block-size", type=int, default=0, help="переопределить длину контекста")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-sample", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
