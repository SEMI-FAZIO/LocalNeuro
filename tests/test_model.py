"""Tests for the transformer model and its KV cache."""

import torch

from localneuro.config import ModelConfig
from localneuro.model import LocalNeuroLM


def _tiny_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=32
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_forward_with_targets_returns_loss():
    model = LocalNeuroLM(_tiny_config())
    x = torch.randint(0, 64, (2, 8))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 8, 64)
    assert loss.ndim == 0 and loss.item() > 0


def test_forward_without_targets_returns_last_token():
    model = LocalNeuroLM(_tiny_config())
    x = torch.randint(0, 64, (2, 8))
    logits, loss = model(x)
    assert logits.shape == (2, 1, 64)  # generation fast-path
    assert loss is None


def test_param_count_matches_analytic_estimate():
    config = _tiny_config()
    model = LocalNeuroLM(config)
    assert model.num_parameters() == config.num_params()


def test_untied_model_has_more_params_than_tied():
    tied = LocalNeuroLM(_tiny_config(tie_embeddings=True))
    untied = LocalNeuroLM(_tiny_config(tie_embeddings=False))
    assert untied.num_parameters() > tied.num_parameters()


def test_kv_cache_matches_full_forward():
    """Incremental decoding must produce the same logits as a full forward."""
    torch.manual_seed(0)
    config = _tiny_config()
    model = LocalNeuroLM(config).eval()
    seq = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        full_logits, _ = model(seq, targets=seq)
        cache = model.make_kv_cache(batch_size=1, max_seq_len=config.max_seq_len)
        steps = []
        for t in range(seq.shape[1]):
            step_logits, _ = model(seq[:, t:t + 1], kv_cache=cache, start_pos=t)
            steps.append(step_logits[:, -1, :])
    incremental = torch.stack(steps, dim=1)
    assert torch.allclose(full_logits, incremental, atol=1e-4)


def test_kv_cache_prefill_then_decode():
    """A multi-token prefill followed by single-token decodes must also match."""
    torch.manual_seed(0)
    config = _tiny_config()
    model = LocalNeuroLM(config).eval()
    seq = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        full_logits, _ = model(seq, targets=seq)
        cache = model.make_kv_cache(batch_size=1, max_seq_len=config.max_seq_len)
        prefill, _ = model(seq[:, :5], kv_cache=cache, start_pos=0)
        outputs = [prefill[:, -1, :]]  # position 4
        for t in range(5, 12):
            step_logits, _ = model(seq[:, t:t + 1], kv_cache=cache, start_pos=t)
            outputs.append(step_logits[:, -1, :])
    decoded = torch.stack(outputs, dim=1)  # positions 4..11
    assert torch.allclose(full_logits[:, 4:12, :], decoded, atol=1e-4)


def test_gradient_checkpointing_trains():
    model = LocalNeuroLM(_tiny_config())
    model.enable_gradient_checkpointing(True)
    model.train()
    x = torch.randint(0, 64, (2, 8))
    _, loss = model(x, targets=x)
    loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_grouped_query_attention_runs():
    # n_kv_heads < n_heads exercises the GQA repeat path.
    model = LocalNeuroLM(_tiny_config(n_heads=4, n_kv_heads=2))
    x = torch.randint(0, 64, (2, 8))
    logits, _ = model(x, targets=x)
    assert logits.shape == (2, 8, 64)
