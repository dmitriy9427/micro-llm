# micro-llm

Маленькая языковая модель в духе GPT, написанная **с нуля** ради понимания: токенайзер BPE, трансформер, обучение, генерация и выгрузка карт внимания для 3D-визуализации. Без готовых «обёрток»: вся модель — около 150 строк на PyTorch.

Это учебный проект и первая ступень плана «от фронтенда к ИИ»: сначала понять, как работает модель, потом ускорить её на Rust/WASM и показать работу трансформера в браузере в 3D.

**Учебник:** [docs/TUTORIAL.md](docs/TUTORIAL.md) — читать по порядку, к каждой главе есть упражнения.

## Быстрый старт

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest                              # 20 тестов за пару секунд
.venv/bin/python scripts/build_corpus.py ~/projects     # собрать текст из ваших .md → data/corpus.txt
.venv/bin/python -m microllm.train --preset small       # обучить (на Apple Silicon ~2–4 минуты)
.venv/bin/python -m microllm.generate --prompt "Модуль " --tokens 300
```

Устройство выбирается само: NVIDIA (`cuda`) → Apple Silicon (`mps`) → процессор.

## Что внутри

| Файл | Что это |
| --- | --- |
| `microllm/tokenizer.py` | символьный токенайзер и **BPE с нуля** (байтовый, как GPT-2) |
| `microllm/model.py` | GPT: внимание, MLP, остаточные связи, инициализация |
| `microllm/train.py` | цикл обучения: AdamW, прогрев + косинус, обрезка градиента, проверочная выборка |
| `microllm/generate.py` | выбор следующего токена: temperature, top-k, top-p |
| `microllm/data.py`, `checkpoint.py`, `config.py` | данные, сохранение, размеры модели (`tiny` / `small` / `base`) |
| `scripts/build_corpus.py` | собирает обучающий текст из Markdown-файлов |
| `scripts/dump_attention.py` | выгружает карты внимания в JSON (для 3D-визуализации) |
| `viz/` | 3D-визуализация внимания в браузере (three.js, [viz/README.md](viz/README.md)) |
| `tests/` | токенайзер, форма и причинность модели, «выучить маленькую пачку», сквозной прогон |

## Дорожная карта

1. ✅ **Шаг 1 — Python + PyTorch:** токенайзер, трансформер, обучение, генерация, карты внимания (этот репозиторий).
2. ⬜ **Шаг 2 — Rust:** быстрый токенайзер BPE (обучение сейчас на чистом Python — это узкое место), сборка в WebAssembly.
3. 🟨 **Шаг 3 — браузер и 3D:** ✅ [3D-визуализация внимания](viz/) на готовых JSON (Three.js) — готова; ⬜ запуск самой модели в браузере через WebGPU/ONNX.
4. ⬜ **Шаг 4 — взрослый уровень:** дообучение (LoRA) под задачу, оценка качества (evals), вклад в open-source.
