"""LocalNeuro data package: preprocessing, synthetic data and token datasets."""

from .dataset import (
    TokenDataset,
    build_dataloader,
    encode_corpus,
    infinite_loader,
    load_token_meta,
    write_token_bin,
)
from .preprocessing import normalize_text, read_corpus, split_train_val
from .sft import (
    InstructionExample,
    SFTDataset,
    build_sft_dataloader,
    build_sft_example,
    read_instructions_jsonl,
    write_instructions_jsonl,
)
from .synthetic import (
    generate_synthetic_corpus,
    generate_synthetic_instructions,
    write_synthetic_corpus,
)

__all__ = [
    "normalize_text",
    "read_corpus",
    "split_train_val",
    "generate_synthetic_corpus",
    "generate_synthetic_instructions",
    "write_synthetic_corpus",
    "TokenDataset",
    "build_dataloader",
    "encode_corpus",
    "infinite_loader",
    "load_token_meta",
    "write_token_bin",
    "InstructionExample",
    "SFTDataset",
    "build_sft_example",
    "build_sft_dataloader",
    "read_instructions_jsonl",
    "write_instructions_jsonl",
]
