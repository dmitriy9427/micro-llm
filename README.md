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
2. ✅ **Шаг 2 — Rust:** токенайзер BPE на Rust, в 14–24 раза быстрее Python, собран в WebAssembly ([глава 10](docs/TUTORIAL.md)).
3. 🟨 **Шаг 3 — браузер и 3D:** ✅ [3D-визуализация внимания](viz/) на готовых JSON (Three.js) — готова; ⬜ запуск самой модели в браузере через WebGPU/ONNX.
4. 🟨 **Шаг 4 — взрослый уровень:** ✅ дообучение (LoRA) и оценка против RAG ([глава 11](docs/TUTORIAL.md)); ⬜ строгие evals, вклад в open-source.

**Новичку в Python/Rust:** [Python и PyTorch для JS-разработчика](docs/PYTHON_FOR_JS.md) · [Rust и WebAssembly для JS-разработчика](docs/RUST_FOR_JS.md).

## Токенайзер на Rust и WebAssembly

В `rust/bpe` тот же BPE, переписанный на Rust: обучение на корпусе в 1,4 МБ занимает ~1,5–3 с вместо ~40 с в Python (в 14–24 раза быстрее),
а слияния совпадают байт в байт (`scripts/check_rust_parity.py`). Тот же код собран в WebAssembly и работает в браузере:
`viz/tokenizer.html` обучает токенайзер на вашем тексте и красит токены. Не знаете Rust? Начните с [введения для JS-разработчика](docs/RUST_FOR_JS.md).

```bash
cargo test --release --manifest-path rust/bpe/Cargo.toml
cargo build --release --target wasm32-unknown-unknown --manifest-path rust/bpe/Cargo.toml
wasm-bindgen rust/bpe/target/wasm32-unknown-unknown/release/microllm_bpe.wasm --target web --out-dir viz/wasm --no-typescript
```

## Дообучение готовой модели (LoRA)

В `finetune/` — дообучение Qwen2.5-0.5B-Instruct на вопросах про резюме и сравнение с RAG. Коротко: отказы на посторонние темы
выучились (8/9 против 0/9), но факты модель частично выдумывает — для знаний лучше RAG. Подробности и честные оговорки:
[`finetune/README.md`](finetune/README.md).
