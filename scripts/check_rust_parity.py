"""Сверка Rust-токенайзера с Python: те же слияния и сколько времени на это уходит.

    cargo build --release --manifest-path rust/bpe/Cargo.toml --example train
    python scripts/check_rust_parity.py data/corpus.txt 1024
"""
import json
import subprocess
import sys
import tempfile
import time

from microllm.tokenizer import BPETokenizer

corpus, vocab = sys.argv[1], int(sys.argv[2])
text = open(corpus, encoding="utf-8").read()

t = time.time()
py = BPETokenizer.train(text, vocab)
py_s = time.time() - t

with tempfile.NamedTemporaryFile(suffix=".json") as f:
    t = time.time()
    subprocess.run(["rust/bpe/target/release/examples/train", corpus, str(vocab), f.name], check=True)
    rs_s = time.time() - t
    rs = json.load(open(f.name))["merges"]

same = [list(m) for m in py.merges] == rs
print(f"Python {py_s:.1f} с | Rust {rs_s:.1f} с | быстрее в {py_s / rs_s:.0f} раз | слияния совпадают: {same}")
sys.exit(0 if same else 1)
