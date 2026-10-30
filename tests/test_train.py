import json
import os

from microllm.checkpoint import load_checkpoint
from microllm.generate import generate_text
from microllm.train import parse_args, train

TEXT = ("Привет, мир. Это маленький тест. " * 120)


def test_end_to_end_train_save_generate(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    out = tmp_path / "out"
    args = parse_args([
        "--corpus", str(corpus), "--preset", "tiny", "--tokenizer", "char",
        "--max-steps", "60", "--eval-interval", "30", "--batch-size", "8",
        "--out-dir", str(out), "--device", "cpu", "--no-sample",
    ])
    res = train(args)
    # на таком повторяющемся тексте ошибка быстро падает ниже, чем у случайных догадок
    assert res["val"] < 3.0
    log = [json.loads(l) for l in open(out / "log.jsonl")]
    assert log[0]["train"] > log[-1]["train"]
    assert os.path.exists(out / "ckpt.pt")

    model, tok, device = load_checkpoint(str(out / "ckpt.pt"), "cpu")
    text = generate_text(model, tok, "Привет", device, 20, temperature=0.7)
    assert text.startswith("Привет") and len(text) > len("Привет")


def test_bpe_training_path(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    args = parse_args([
        "--corpus", str(corpus), "--preset", "tiny", "--tokenizer", "bpe", "--vocab-size", "300",
        "--max-steps", "20", "--eval-interval", "20", "--batch-size", "4", "--block-size", "32",
        "--out-dir", str(tmp_path / "o"), "--device", "cpu", "--no-sample",
    ])
    assert train(args)["params"] > 0
