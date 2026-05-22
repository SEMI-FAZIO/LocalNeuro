"""Weight-only quantization: a quantized drop-in for ``nn.Linear``.

LocalNeuro uses **symmetric, per-output-channel** quantization. For each row of
a weight matrix it finds the largest magnitude, derives a scale, and stores the
weight as small integers:

    scale_row = max(|W_row|) / qmax           (qmax = 127 for int8, 7 for int4)
    W_int     = round(W / scale)              (clamped to +/- qmax)
    W ~= W_int * scale                        (dequantization)

Per-row (rather than a single global) scales keep accuracy high because each
output channel gets a scale matched to its own dynamic range.

This is a *memory* optimization: int8 weights are 4x smaller than float32 and
int4 weights are 8x smaller, which is what lets a bigger model fit in a small
RAM budget. The matmul itself still runs in floating point -- the weight is
dequantized on the fly -- so it is not faster, just lighter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_tensor(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row quantization of a 2-D weight matrix.

    Returns ``(q, scale)`` where ``q`` holds signed integers in ``[-qmax, qmax]``
    (as ``int8``) and ``scale`` has one entry per row.
    """
    if weight.dim() != 2:
        raise ValueError("expected a 2-D weight matrix")
    qmax = (1 << (bits - 1)) - 1  # 8 -> 127, 4 -> 7
    max_abs = weight.abs().amax(dim=1, keepdim=True)
    scale = (max_abs / qmax).clamp(min=1e-8)
    q = torch.round(weight / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale.squeeze(1)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack a matrix of int4 values (range ``[-7, 7]``) into ``uint8`` nibbles.

    Two consecutive values along the input dimension share one byte. Values are
    biased by +7 into ``[0, 14]`` so they fit in an unsigned nibble. An odd
    input dimension is padded with a single neutral value.
    """
    out_features, in_features = q.shape
    shifted = (q.to(torch.int16) + 7).to(torch.uint8)  # [-7,7] -> [0,14]
    if in_features % 2 == 1:
        pad = torch.full((out_features, 1), 7, dtype=torch.uint8)  # 7 -> dequantizes to 0
        shifted = torch.cat((shifted, pad), dim=1)
    pairs = shifted.view(out_features, -1, 2)
    return (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4`. Returns int8 values in ``[-7, 7]``."""
    out_features = packed.shape[0]
    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)
    values = torch.stack((low, high), dim=2).view(out_features, -1)
    return (values[:, :in_features] - 7).to(torch.int8)


class QuantizedLinear(nn.Module):
    """A bias-free linear layer whose weight is stored in int8 or int4.

    Drop-in compatible with ``nn.Linear(in, out, bias=False)`` -- the model uses
    bias-free linears throughout, so bias support is intentionally omitted.
    """

    def __init__(self, in_features: int, out_features: int, bits: int = 8) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        if bits == 8:
            qweight = torch.zeros(out_features, in_features, dtype=torch.int8)
        else:
            packed_in = (in_features + 1) // 2
            qweight = torch.zeros(out_features, packed_in, dtype=torch.uint8)
        # Buffers (not parameters): quantized weights are not trained.
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float32))

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 8) -> "QuantizedLinear":
        """Build a quantized copy of an existing ``nn.Linear``."""
        if linear.bias is not None:
            raise ValueError("QuantizedLinear supports bias-free linears only")
        layer = cls(linear.in_features, linear.out_features, bits)
        layer.load_from_weight(linear.weight.detach())
        return layer

    def load_from_weight(self, weight: torch.Tensor) -> None:
        """Quantize ``weight`` and store it into this layer's buffers."""
        q, scale = quantize_tensor(weight.float(), self.bits)
        self.qweight.copy_(q if self.bits == 8 else pack_int4(q))
        self.scale.copy_(scale)

    def dequantized_weight(self) -> torch.Tensor:
        """Reconstruct the float32 weight matrix from its quantized form."""
        if self.bits == 8:
            values = self.qweight.to(torch.float32)
        else:
            values = unpack_int4(self.qweight, self.in_features).to(torch.float32)
        return values * self.scale.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize one layer's weight at a time: the transient float matrix is
        # freed immediately, so resident memory stays dominated by the small
        # integer buffers.
        return F.linear(x, self.dequantized_weight().to(x.dtype))

    def weight_numel(self) -> int:
        """Number of weight values represented (ignoring int4 bit-packing).

        Quantized weights are stored as buffers, not parameters; this lets a
        model report a parameter count that is stable across quantization.
        """
        return self.out_features * self.in_features

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}"
        )
