"""Собирает обучающий текст: все Markdown-файлы из указанных папок (по умолчанию — ваши проекты).

    python scripts/build_corpus.py ~/projects            # → data/corpus.txt
    python scripts/build_corpus.py ./docs --out data/small.txt

Модель научится писать «как ваша документация». Чем больше и разнообразнее текст,
тем лучше: хватит и нескольких сотен килобайт, чтобы увидеть, как падает ошибка.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

SKIP_DIRS = {"node_modules", ".git", "dist", ".venv", "venv", "build", ".next", ".astro", "out", "__pycache__"}


def iter_markdown(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            if name.lower().endswith(".md"):
                yield os.path.join(dirpath, name)


def clean(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)           # блоки кода выбрасываем: учим «язык», а не синтаксис
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)             # картинки
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)         # ссылки → только текст
    text = re.sub(r"<[^>\n]*>?", "", text)                      # HTML-теги (и оборванные)
    text = re.sub(r"^[\s\-:|=_~*+]+$", "", text, flags=re.M)    # линии таблиц и разделители
    text = re.sub(r"[`*_>|#]+", "", text)                        # разметка
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=[os.path.expanduser("~/projects")])
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--min-chars", type=int, default=200, help="пропускать совсем короткие файлы")
    ap.add_argument("--exclude", action="append", default=["micro-llm"], help="подстрока пути, которую пропускаем")
    args = ap.parse_args()

    parts, seen = [], set()
    for root in args.roots:
        for path in iter_markdown(root):
            if any(x in path for x in args.exclude):
                continue
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = clean(f.read())
            if len(text) < args.min_chars or text in seen:
                continue
            seen.add(text)
            parts.append(text)
    corpus = "\n\n".join(parts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"файлов: {len(parts)}, символов: {len(corpus):,}, уникальных символов: {len(set(corpus))} → {args.out}")
    if len(corpus) < 100_000:
        print("предупреждение: текста мало (<100 тыс. символов), модель будет быстро переобучаться", file=sys.stderr)


if __name__ == "__main__":
    main()
