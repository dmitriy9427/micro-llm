"""Общее для дообучения: системная подсказка, фраза отказа и сборка чата."""
from __future__ import annotations

import re

SYSTEM = {
    "ru": "Ты ассистент по резюме фронтенд-разработчика Дмитрия Рябова. Отвечай коротко, только по резюме.",
    "en": "You are an assistant for the résumé of frontend developer Dmitry Ryabov. Answer briefly, using only the résumé.",
}
REFUSAL = {
    "ru": "Я отвечаю только на вопросы о резюме Дмитрия Рябова: опыт, проекты, навыки и контакты.",
    "en": "I only answer questions about Dmitry Ryabov's résumé: experience, projects, skills and contacts.",
}
CYR = re.compile("[А-Яа-яЁё]")


def lang_of(text: str) -> str:
    return "ru" if CYR.search(text) else "en"


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return "только на вопросы о резюме" in a or "only answer questions about" in a


def messages(question: str, answer: str | None = None, context: str | None = None):
    lang = lang_of(question)
    user = question if context is None else f"Контекст из резюме:\n{context}\n\nВопрос: {question}"
    m = [{"role": "system", "content": SYSTEM[lang]}, {"role": "user", "content": user}]
    if answer is not None:
        m.append({"role": "assistant", "content": answer})
    return m
