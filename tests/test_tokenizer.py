import pytest

from microllm.tokenizer import BPETokenizer, CharTokenizer, tokenizer_from_dict

TEXT = "Привет, мир! Привет, проект. Проект — это приключение. " * 20


def test_char_roundtrip():
    tok = CharTokenizer.train(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT
    assert tok.vocab_size == len(set(TEXT))


def test_char_skips_unknown():
    tok = CharTokenizer.train("abc")
    assert tok.decode(tok.encode("abxc")) == "abc"


def test_bpe_roundtrip_on_any_text():
    tok = BPETokenizer.train(TEXT, 300)
    # байтовый BPE кодирует любой текст, даже символы, которых не было при обучении
    for s in [TEXT, "Совсем другой текст 123 !?", "emoji 🙂 и 日本語", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_compresses():
    tok = BPETokenizer.train(TEXT, 400)
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8")) / 2
    assert tok.vocab_size > 256


def test_bpe_rejects_small_vocab():
    with pytest.raises(ValueError):
        BPETokenizer.train(TEXT, 100)


def test_bpe_stops_when_nothing_to_merge():
    tok = BPETokenizer.train("ab", 1000)
    assert tok.vocab_size < 1000


def test_save_load_dict():
    tok = BPETokenizer.train(TEXT, 300)
    tok2 = tokenizer_from_dict(tok.to_dict())
    assert tok2.encode(TEXT) == tok.encode(TEXT)
    ch = tokenizer_from_dict(CharTokenizer.train(TEXT).to_dict())
    assert ch.decode(ch.encode("Привет")) == "Привет"
