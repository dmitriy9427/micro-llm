"""Генерация текста: модель выдаёт распределение следующего токена, мы выбираем и повторяем."""
from __future__ import annotations

import argparse

import torch

from .checkpoint import load_checkpoint
from .model import GPT


@torch.no_grad()
def sample(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Дописывает ``max_new_tokens`` токенов к ``idx`` (B, T).

    * ``temperature`` < 1 — осторожнее и скучнее, > 1 — смелее и бредовее; 0 — всегда самый вероятный.
    * ``top_k`` — выбирать только из k самых вероятных токенов.
    * ``top_p`` — выбирать из «ядра»: наименьшего набора токенов с суммарной вероятностью p.
    """
    model.eval()
    for _ in range(max_new_tokens):
        ctx = idx[:, -model.cfg.block_size:]               # модель видит только последние block_size токенов
        logits, _ = model(ctx)
        logits = logits[:, -1, :]                           # нас интересует только следующий токен
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                drop = cum - probs > top_p                  # убираем всё, что за пределами ядра
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def generate_text(model: GPT, tok, prompt: str, device, max_new_tokens: int = 200, **kw) -> str:
    ids = tok.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = sample(model, idx, max_new_tokens, **kw)
    return tok.decode(out[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description="Сгенерировать текст обученной моделью")
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tok, device = load_checkpoint(args.ckpt, args.device)
    print(generate_text(model, tok, args.prompt, device, args.tokens,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p))


if __name__ == "__main__":
    main()
