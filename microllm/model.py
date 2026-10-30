"""GPT: трансформер-«декодер», написанный с нуля.

Путь данных (сверху вниз):

    токены (B, T)                       B — размер батча, T — длина
      → эмбеддинги токенов + позиций    (B, T, C)    C = n_embd
      → N блоков: [LayerNorm → внимание → +] [LayerNorm → MLP → +]
      → LayerNorm → линейный слой       (B, T, vocab) — «логиты»: оценки следующего токена

Главное — слой внимания: каждый токен смотрит на предыдущие и решает, какие из них важны.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Многоголовое причинное внимание.

    Для каждого токена считаем три вектора: запрос Q («что я ищу?»), ключ K («что у меня есть?»)
    и значение V («что я отдам, если на меня посмотрят»). Сходство Q и K даёт веса внимания,
    ими взвешиваем V. «Причинное» — токен не видит будущее: верхний треугольник матрицы закрыт.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd должен делиться на n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # сразу Q, K и V одной матрицей
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # собираем головы обратно
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, голов, T, размер_головы)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, голов, T, T): кто на кого смотрит
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # закрываем будущее
        att = F.softmax(att, dim=-1)                                  # строки → вероятности, сумма = 1
        weights = att
        att = self.attn_drop(att)
        y = att @ v                                                   # смесь значений по весам внимания
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, (weights if return_attn else None)


class MLP(nn.Module):
    """Две линейные карты с нелинейностью: «подумать» над тем, что собрало внимание."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        a, w = self.attn(self.ln1(x), return_attn)
        x = x + a                       # остаточная связь: добавляем, а не заменяем
        x = x + self.mlp(self.ln2(x))
        return x, w


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # таблица «токен → вектор»
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # таблица «позиция → вектор»
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                    # общий вес: вход и выход «говорят» об одном
        self.apply(self._init_weights)
        # выходные проекции в остаточных ветках делаем меньше, иначе сумма разрастается с глубиной
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """Возвращает (logits, loss[, список карт внимания по слоям])."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"длина {T} больше контекста {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        attns: List[torch.Tensor] = []
        for block in self.blocks:
            x, w = block(x, return_attn)
            if w is not None:
                attns.append(w)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            # «следующий токен»: для каждой позиции сравниваем предсказание с настоящим продолжением
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_attn:
            return logits, loss, attns
        return logits, loss

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1) -> torch.optim.Optimizer:
        """AdamW: матрицам весов — штраф за величину (weight decay), смещениям и нормализации — нет."""
        decay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
