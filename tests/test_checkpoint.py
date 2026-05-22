"""Tests for the .lnw weight format and checkpoint bundles."""

import torch

from localneuro.checkpoint import (
    load_checkpoint,
    load_weights,
    save_checkpoint,
    save_weights,
)
from localneuro.config import ModelConfig
from localneuro.model import LocalNeuroLM
from localneuro.quantization.quantize import quantize_model
from localneuro.tokenizer.bpe import BPETokenizer


def _tiny_model(vocab_size: int = 80) -> tuple[LocalNeuroLM, ModelConfig]:
    config = ModelConfig(
        vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=4,
        d_ff=64, max_seq_len=32,
    )
    return LocalNeuroLM(config).eval(), config


def test_weight_format_roundtrip(tmp_path):
    model, _ = _tiny_model()
    path = tmp_path / "weights.lnw"
    save_weights(path, model.state_dict())
    for mmap in (True, False):
        loaded = load_weights(path, mmap=mmap)
        for key, value in model.state_dict().items():
            assert torch.allclose(value, loaded[key]), f"{key} (mmap={mmap})"


def test_checkpoint_bundle_roundtrip(tmp_path):
    tokenizer = BPETokenizer.train(
        "the quick brown fox jumps over the lazy dog. " * 60, vocab_size=400
    )
    model, config = _tiny_model(vocab_size=tokenizer.vocab_size)
    save_checkpoint(
        tmp_path / "ckpt", model, config, tokenizer=tokenizer, meta={"step": 7}
    )
    loaded = load_checkpoint(tmp_path / "ckpt", build_model=True)

    assert loaded.meta["step"] == 7
    assert loaded.tokenizer is not None
    assert loaded.tokenizer.vocab_size == tokenizer.vocab_size

    x = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        original, _ = model(x)
        restored, _ = loaded.model(x)
    assert torch.allclose(original, restored, atol=1e-5)


def test_quantized_checkpoint_roundtrip(tmp_path):
    model, config = _tiny_model()
    quantize_model(model, bits=8)
    save_checkpoint(
        tmp_path / "q", model, config, meta={"step": 1, "quantization": 8}
    )
    loaded = load_checkpoint(tmp_path / "q", build_model=True)

    x = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        original, _ = model(x)
        restored, _ = loaded.model(x)
    assert torch.allclose(original, restored, atol=1e-5)
