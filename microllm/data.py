"""Данные: текст → числа → случайные куски для обучения."""
from __future__ import annotations

from typing import Tuple

import torch


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_tokens(ids, val_fraction: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Последние val_fraction данных — проверочные: модель их в обучении не видит.

    Разница между потерями на обучающих и проверочных данных показывает переобучение:
    если на обучающих всё лучше, а на проверочных хуже — модель зубрит, а не понимает.
    """
    data = torch.tensor(ids, dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Случайные окна длины block_size. Цель y — тот же текст, сдвинутый на один токен вперёд:
    для каждой позиции модель учится угадывать следующий токен."""
    if len(data) <= block_size + 1:
        raise ValueError(f"данных ({len(data)} токенов) слишком мало для контекста {block_size}")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
