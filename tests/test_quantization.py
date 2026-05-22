"""Tests for int8 / int4 quantization."""

import torch
import torch.nn as nn

from localneuro.config import ModelConfig
from localneuro.model import LocalNeuroLM
from localneuro.quantization.quant_linear import (
    QuantizedLinear,
    pack_int4,
    quantize_tensor,
    unpack_int4,
)
from localneuro.quantization.quantize import count_quantized_layers, quantize_model


def test_quantize_tensor_roundtrip_is_close():
    torch.manual_seed(0)
    weight = torch.randn(16, 32)
    q, scale = quantize_tensor(weight, bits=8)
    dequantized = q.float() * scale.unsqueeze(1)
    assert (dequantized - weight).abs().max() < 0.05


def test_int4_pack_unpack_roundtrip():
    torch.manual_seed(0)
    # Odd input dimension exercises the padding branch.
    values = torch.randint(-7, 8, (5, 9), dtype=torch.int8)
    packed = pack_int4(values)
    assert packed.dtype == torch.uint8
    assert packed.shape == (5, 5)
    assert torch.equal(unpack_int4(packed, 9), values)


def test_quantized_linear_int8_close_to_reference():
    torch.manual_seed(0)
    linear = nn.Linear(64, 48, bias=False)
    quantized = QuantizedLinear.from_linear(linear, bits=8)
    x = torch.randn(4, 64)
    reference = linear(x)
    relative_error = (quantized(x) - reference).norm() / reference.norm()
    assert relative_error < 0.05


def test_quantized_linear_int4_close_to_reference():
    torch.manual_seed(0)
    linear = nn.Linear(64, 48, bias=False)
    quantized = QuantizedLinear.from_linear(linear, bits=4)
    x = torch.randn(4, 64)
    reference = linear(x)
    relative_error = (quantized(x) - reference).norm() / reference.norm()
    assert relative_error < 0.25  # int4 is coarser but still usable


def test_quantize_model_swaps_all_linears():
    config = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=32
    )
    model = LocalNeuroLM(config)
    count = quantize_model(model, bits=8)
    assert count > 0
    assert count_quantized_layers(model) == count
    # The quantized model must still run a forward pass.
    x = torch.randint(0, 64, (1, 8))
    logits, _ = model(x)
    assert logits.shape == (1, 1, 64)
