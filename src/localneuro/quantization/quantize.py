"""Model-level quantization: swap every ``nn.Linear`` for a ``QuantizedLinear``.

Two entry points:

* :func:`quantize_model` -- quantize a *trained* model's weights in place.
* :func:`apply_quantized_skeleton` -- swap in *empty* quantized layers so a
  quantized checkpoint can be loaded with ``load_state_dict`` afterwards.

The token embedding (an ``nn.Embedding``) is deliberately left in float32 -- it
is a lookup table, not a matmul, and keeping it full precision avoids quantizing
the tied output projection. Quantizing the embedding is noted as future work in
``docs/roadmap.md``.
"""

from __future__ import annotations

import torch.nn as nn

from .quant_linear import QuantizedLinear


def _replace_linears(module: nn.Module, bits: int, copy_weights: bool) -> int:
    """Recursively replace ``nn.Linear`` children with ``QuantizedLinear``.

    Returns the number of layers that are quantized after the pass.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if copy_weights:
                replacement: nn.Module = QuantizedLinear.from_linear(child, bits)
            else:
                replacement = QuantizedLinear(
                    child.in_features, child.out_features, bits
                )
            setattr(module, name, replacement)
            count += 1
        elif isinstance(child, QuantizedLinear):
            count += 1  # already quantized
        else:
            count += _replace_linears(child, bits, copy_weights)
    return count


def quantize_model(model: nn.Module, bits: int = 8) -> int:
    """Quantize all linear layers of a trained model **in place**.

    Returns the number of layers quantized.
    """
    return _replace_linears(model, bits, copy_weights=True)


def apply_quantized_skeleton(model: nn.Module, bits: int) -> int:
    """Swap linears for *empty* quantized layers (no weight computation).

    Used when loading a quantized checkpoint: the module structure must match
    the saved state dict before ``load_state_dict`` is called.
    """
    return _replace_linears(model, bits, copy_weights=False)


def count_quantized_layers(model: nn.Module) -> int:
    """Count :class:`QuantizedLinear` modules currently in ``model``."""
    return sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
