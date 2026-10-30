"""Выгружает карты внимания в JSON — для 3D-визуализации в браузере (следующий шаг курса).

    python scripts/dump_attention.py --prompt "Привет, мир" --out attention.json

Формат: {"tokens": [...], "layers": [ [ [ [..T..] ... T строк ] ... головы ] ... слои ]}
weights[слой][голова][i][j] — сколько токен i «смотрит» на токен j (j ≤ i, строки суммируются в 1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # запуск из любой папки

from microllm.checkpoint import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="attention.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, tok, device = load_checkpoint(args.ckpt, args.device)
    ids = tok.encode(args.prompt)[-model.cfg.block_size:]
    idx = torch.tensor([ids], device=device)
    with torch.no_grad():
        _, _, attns = model(idx, return_attn=True)
    layers = [[[[round(v, 4) for v in row] for row in head] for head in a[0].cpu().tolist()] for a in attns]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tokens": tok.token_strings(ids), "layers": layers}, f, ensure_ascii=False)
    print(f"токенов: {len(ids)}, слоёв: {len(layers)}, голов: {len(layers[0])} → {args.out}")


if __name__ == "__main__":
    main()
