"""Токенайзеры: превращают текст в числа и обратно.

Модель не умеет читать буквы — она работает с целыми числами. Токенайзер — таблица
«кусок текста ↔ число». Здесь два варианта:

* ``CharTokenizer`` — токен = один символ. Просто и надёжно, но последовательности длинные.
* ``BPETokenizer`` — Byte-Pair Encoding, как в GPT-2: частые пары символов склеиваются
  в новые токены («при», «вет», « проект»…). Последовательности короче в 3–4 раза.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Tuple


class CharTokenizer:
    kind = "char"

    def __init__(self, chars: List[str]):
        self.itos = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        # неизвестные символы пропускаем — модель их не видела
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def token_strings(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "chars": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["chars"])


# Предварительное разбиение на «слова»: слово с пробелом перед ним, число, знаки, пробелы.
# BPE склеивает пары только внутри такого куска, поэтому токены не «перетекают» через слова.
_PRETOKEN = re.compile(r"\s?\w+|\s?[^\w\s]+|\s+", re.UNICODE)

Pair = Tuple[int, int]


class BPETokenizer:
    """Байтовый BPE: алфавит — 256 байт (поэтому текст на любом языке кодируется без «неизвестных»)."""

    kind = "bpe"

    def __init__(self, merges: List[Pair]):
        self.merges = [tuple(m) for m in merges]
        # токен 0..255 — байты; токен 256+i — результат i-го слияния
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self._cache: Dict[str, List[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ── обучение ────────────────────────────────────────────────────────────
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """Находим самую частую пару соседних токенов, склеиваем её в новый токен, повторяем.

        Считаем не по всему тексту, а по *уникальным словам с частотами*: слово «проект» встречается
        тысячу раз, но разбирать его достаточно один раз. Это ускоряет обучение в сотни раз.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size должен быть не меньше 256 (по токену на байт)")
        words: Counter = Counter(_PRETOKEN.findall(text))
        pieces = {w: list(w.encode("utf-8")) for w in words}
        merges: List[Pair] = []
        for step in range(vocab_size - 256):
            stats: Counter = Counter()
            for w, ids in pieces.items():
                f = words[w]
                for pair in zip(ids, ids[1:]):
                    stats[pair] += f
            if not stats:
                break
            best, count = stats.most_common(1)[0]
            if count < 2:
                break  # дальше склеивать нечего: пары больше не повторяются
            new_id = 256 + len(merges)
            merges.append(best)
            for w, ids in pieces.items():
                pieces[w] = _merge(ids, best, new_id)
            if verbose and step % 100 == 0:
                print(f"слияние {step}: {best} → {new_id} (встретилось {count} раз)")
        return cls(merges)

    # ── кодирование ─────────────────────────────────────────────────────────
    def _encode_word(self, word: str) -> List[int]:
        if word in self._cache:
            return self._cache[word]
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            # применяем слияние с наименьшим номером — в том порядке, в каком училось
            pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, 1 << 30))
            if pair not in self.rank:
                break
            ids = _merge(ids, pair, 256 + self.rank[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for w in _PRETOKEN.findall(text):
            out.extend(self._encode_word(w))
        return out

    def decode(self, ids: List[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def token_strings(self, ids: List[int]) -> List[str]:
        # токен может оказаться половиной русской буквы (2 байта): показываем такие байты как «\\xd0», а не «�»
        return [self.vocab[i].decode("utf-8", errors="backslashreplace") for i in ids]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        return cls([tuple(m) for m in d["merges"]])


def _merge(ids: List[int], pair: Pair, new_id: int) -> List[int]:
    out: List[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def build_tokenizer(kind: str, text: str, vocab_size: int = 1024):
    if kind == "char":
        return CharTokenizer.train(text)
    if kind == "bpe":
        return BPETokenizer.train(text, vocab_size)
    raise ValueError(f"неизвестный токенайзер: {kind}")


def tokenizer_from_dict(d: dict):
    return {"char": CharTokenizer, "bpe": BPETokenizer}[d["kind"]].from_dict(d)


def save_tokenizer(tok, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok.to_dict(), f, ensure_ascii=False)


def load_tokenizer(path: str):
    with open(path, encoding="utf-8") as f:
        return tokenizer_from_dict(json.load(f))
