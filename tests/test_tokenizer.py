"""Tests for the byte-level BPE tokenizer (pure standard library -- no torch)."""

import pytest

from localneuro.tokenizer.bpe import DEFAULT_SPECIAL_TOKENS, BPETokenizer, StreamDecoder

SAMPLE = (
    "the quick brown fox jumps over the lazy dog. " * 25
    + "Привет, мир! Машинное обучение. "
    + "café naïve 123 + 4 = 127. emoji 🚀🔥日本語"
)


def _train(vocab_size: int = 500) -> BPETokenizer:
    return BPETokenizer.train(SAMPLE, vocab_size=vocab_size)


def test_train_produces_reasonable_vocab():
    tok = _train(500)
    assert 256 < tok.vocab_size <= 500
    assert len(tok.merges) > 0


def test_roundtrip_ascii():
    tok = _train()
    text = "the quick brown fox"
    assert tok.decode(tok.encode_ordinary(text)) == text


@pytest.mark.parametrize(
    "text",
    ["Привет мир", "café", "emoji: 🚀🔥", "日本語のテスト", "1234567890", ""],
)
def test_roundtrip_unicode(text):
    # Byte-level BPE reconstructs any valid UTF-8 text exactly, even strings
    # containing characters that never appeared in the training corpus.
    tok = _train()
    assert tok.decode(tok.encode_ordinary(text)) == text


def test_special_tokens_present():
    tok = _train()
    for name in DEFAULT_SPECIAL_TOKENS:
        assert name in tok.special_tokens
    assert tok.bos_id is not None
    assert tok.eos_id is not None
    assert tok.pad_id is not None


def test_encode_ordinary_ignores_special_strings():
    tok = _train()
    ids = tok.encode_ordinary("<eos> here")
    assert tok.eos_id not in ids  # literal text, not the control token


def test_encode_with_allowed_special():
    tok = _train()
    ids = tok.encode("<bos>hello<eos>", allowed_special="all")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_save_load_roundtrip(tmp_path):
    tok = _train()
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    restored = BPETokenizer.load(path)
    assert restored.vocab_size == tok.vocab_size
    text = "the quick brown fox Привет 🚀"
    assert restored.encode_ordinary(text) == tok.encode_ordinary(text)


def test_stream_decoder_handles_split_multibyte():
    tok = _train()
    text = "Привет мир 🚀 日本語"
    ids = tok.encode_ordinary(text)
    decoder = StreamDecoder(tok)
    streamed = "".join(decoder.step(token_id) for token_id in ids) + decoder.flush()
    assert streamed == text


def test_vocab_size_too_small_raises():
    with pytest.raises(ValueError):
        BPETokenizer.train("hello", vocab_size=100)  # < 256 byte tokens
