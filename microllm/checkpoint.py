"""Сохранение и загрузка обученной модели: веса + настройки + токенайзер в одном файле."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .config import GPTConfig
from .model import GPT
from .tokenizer import tokenizer_from_dict


def pick_device(name: Optional[str] = None) -> torch.device:
    """cuda (NVIDIA) → mps (Apple Silicon) → cpu."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: str, model: GPT, tok, extra: Optional[dict] = None) -> None:
    torch.save(
        {"model": model.state_dict(), "config": model.cfg.to_dict(), "tokenizer": tok.to_dict(), "extra": extra or {}},
        path,
    )


def load_checkpoint(path: str, device: Optional[str] = None) -> Tuple[GPT, object, torch.device]:
    dev = pick_device(device)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    return model, tokenizer_from_dict(ck["tokenizer"]), dev
