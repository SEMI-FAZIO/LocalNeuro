"""LocalNeuro tokenizer package: a from-scratch byte-level BPE implementation."""

from .bpe import (
    ASSISTANT,
    BOS,
    DEFAULT_SPECIAL_TOKENS,
    DEFAULT_SPLIT_PATTERN,
    EOS,
    PAD,
    SYSTEM,
    USER,
    BPETokenizer,
    StreamDecoder,
)

__all__ = [
    "BPETokenizer",
    "StreamDecoder",
    "DEFAULT_SPECIAL_TOKENS",
    "DEFAULT_SPLIT_PATTERN",
    "PAD",
    "BOS",
    "EOS",
    "SYSTEM",
    "USER",
    "ASSISTANT",
]
