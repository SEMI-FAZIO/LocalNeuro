"""LocalNeuro -- an ultra-lightweight, fully self-contained tiny language model.

LocalNeuro is built from scratch: its own transformer architecture, its own
byte-level BPE tokenizer, its own training loop, its own pickle-free checkpoint
format and its own CPU-first inference engine. PyTorch is used only as a
low-level tensor / autograd library.

The public API is exposed lazily: importing :mod:`localneuro` itself is cheap
and pulls in nothing heavy, so e.g. ``from localneuro.tokenizer import
BPETokenizer`` works without importing torch. Attribute access such as
``localneuro.LocalNeuroLM`` triggers the real import on first use.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

# Public name -> module that defines it. Resolved on first attribute access.
_EXPORTS = {
    "ModelConfig": "localneuro.config",
    "TrainConfig": "localneuro.config",
    "load_model_config": "localneuro.config",
    "load_train_config": "localneuro.config",
    "LocalNeuroLM": "localneuro.model",
    "build_model": "localneuro.model",
    "BPETokenizer": "localneuro.tokenizer.bpe",
    "StreamDecoder": "localneuro.tokenizer.bpe",
    "SamplingConfig": "localneuro.sampling",
    "sample_token": "localneuro.sampling",
    "save_checkpoint": "localneuro.checkpoint",
    "load_checkpoint": "localneuro.checkpoint",
    "save_weights": "localneuro.checkpoint",
    "load_weights": "localneuro.checkpoint",
    "Trainer": "localneuro.training",
    "InferenceEngine": "localneuro.inference",
    "ChatSession": "localneuro.inference",
    "run_chat": "localneuro.inference",
    "quantize_model": "localneuro.quantization",
    "QuantizedLinear": "localneuro.quantization",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Lazily import a public symbol the first time it is accessed (PEP 562)."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'localneuro' has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
