"""Tests for the token sampling strategies."""

import pytest
import torch

from localneuro.sampling import (
    SamplingConfig,
    apply_repetition_penalty,
    sample_token,
    top_k_filter,
    top_p_filter,
)


def test_greedy_picks_argmax():
    logits = torch.tensor([0.1, 5.0, 0.2, -1.0])
    assert sample_token(logits, SamplingConfig(temperature=0.0)) == 1


def test_top_k_filter_keeps_only_k():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered = top_k_filter(logits, 2)
    assert int((filtered == float("-inf")).sum()) == 3
    assert filtered[3].item() == 4.0
    assert filtered[4].item() == 5.0


def test_top_p_filter_keeps_dominant_token():
    logits = torch.tensor([10.0, 1.0, 1.0, 1.0])
    filtered = top_p_filter(logits, 0.5)
    assert filtered[0].item() == 10.0
    assert torch.isinf(filtered[1:]).all()


def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.tensor([2.0, 2.0, 2.0])
    out = apply_repetition_penalty(logits, [0], penalty=2.0)
    assert out[0].item() < 2.0
    assert out[1].item() == 2.0  # unseen token untouched


def test_sample_token_returns_valid_index():
    torch.manual_seed(0)
    logits = torch.randn(50)
    config = SamplingConfig(temperature=1.0, top_k=10, top_p=0.9)
    for _ in range(25):
        token = sample_token(logits, config)
        assert 0 <= token < 50


def test_seeded_sampling_is_reproducible():
    logits = torch.randn(64)
    config = SamplingConfig(temperature=1.0)
    gen_a = torch.Generator().manual_seed(123)
    gen_b = torch.Generator().manual_seed(123)
    seq_a = [sample_token(logits, config, generator=gen_a) for _ in range(10)]
    seq_b = [sample_token(logits, config, generator=gen_b) for _ in range(10)]
    assert seq_a == seq_b


def test_invalid_sampling_config_rejected():
    with pytest.raises(ValueError):
        SamplingConfig(top_p=1.5)
    with pytest.raises(ValueError):
        SamplingConfig(temperature=-1.0)
