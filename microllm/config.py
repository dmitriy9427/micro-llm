"""Настройки модели. Три готовых размера — от «проверить, что код работает» до «заметно умная»."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 256      # сколько разных токенов знает модель
    block_size: int = 128      # максимальная длина контекста (в токенах)
    n_layer: int = 4           # сколько блоков трансформера друг за другом
    n_head: int = 4            # сколько «голов» внимания в каждом блоке
    n_embd: int = 128          # размер вектора-представления токена
    dropout: float = 0.1       # доля случайно «выключаемых» связей при обучении (против переобучения)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        return cls(**d)


# vocab_size подставляется при обучении (его знает токенайзер)
PRESETS = {
    # ~0.1 млн параметров: тесты и проверка, что всё запускается за секунды
    "tiny": dict(block_size=64, n_layer=2, n_head=2, n_embd=64, dropout=0.0),
    # ~1 млн: учится за 5–10 минут на ноутбуке, уже пишет похожий на корпус текст
    "small": dict(block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1),
    # ~10 млн: ~час на Apple Silicon, заметно связнее, но нужен корпус побольше
    "base": dict(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2),
}


def preset(name: str, vocab_size: int) -> GPTConfig:
    if name not in PRESETS:
        raise ValueError(f"нет пресета «{name}», есть: {', '.join(PRESETS)}")
    return GPTConfig(vocab_size=vocab_size, **PRESETS[name])
