"""Token dataset: corpus -> binary token stream -> training batches.

The training data is stored as one flat array of token ids in a ``.bin`` file
(``uint16`` while the vocabulary fits in 16 bits, otherwise ``uint32``) plus a
small JSON sidecar with metadata. At training time the ``.bin`` file is
*memory-mapped*: the OS pages token data in on demand, so a corpus far larger
than RAM can still be trained on, and start-up is instant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..tokenizer.bpe import BPETokenizer

PathLike = Union[str, Path]


def _bin_dtype(vocab_size: int) -> np.dtype:
    """Smallest unsigned integer type that can hold every token id."""
    return np.dtype(np.uint16) if vocab_size <= 65536 else np.dtype(np.uint32)


def encode_corpus(
    text: str,
    tokenizer: BPETokenizer,
    *,
    doc_separator: str = "\n\n",
    add_bos: bool = True,
    add_eos: bool = True,
) -> List[int]:
    """Tokenize a corpus into a flat list of ids.

    The corpus is split into documents on ``doc_separator``; each document is
    optionally wrapped in ``<bos>`` / ``<eos>`` so the model learns where
    documents begin and end. Pass ``doc_separator=""`` to treat the whole input
    as a single document.
    """
    ids: List[int] = []
    documents = text.split(doc_separator) if doc_separator else [text]
    for document in documents:
        document = document.strip("\n")
        if not document:
            continue
        if add_bos and tokenizer.bos_id is not None:
            ids.append(tokenizer.bos_id)
        ids.extend(tokenizer.encode_ordinary(document))
        if add_eos and tokenizer.eos_id is not None:
            ids.append(tokenizer.eos_id)
    return ids


def write_token_bin(
    path: PathLike, token_ids: Sequence[int], vocab_size: int
) -> Dict[str, object]:
    """Write a token stream to ``path`` (+ ``path.meta.json``). Returns metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = _bin_dtype(vocab_size)
    array = np.asarray(token_ids, dtype=dtype)
    array.tofile(str(path))
    meta: Dict[str, object] = {
        "dtype": dtype.name,
        "count": int(array.size),
        "vocab_size": int(vocab_size),
    }
    Path(str(path) + ".meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    return meta


def load_token_meta(path: PathLike) -> Dict[str, object]:
    """Read the JSON sidecar describing a ``.bin`` token stream."""
    return json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))


class TokenDataset(Dataset):
    """A memory-mapped dataset yielding ``(input, target)`` blocks.

    Item ``i`` is the contiguous window ``[i*block_size, i*block_size+block_size]``
    where the target is the input shifted by one position -- the standard
    next-token-prediction objective.

    The memory map is opened lazily (on first access) and is intentionally NOT
    part of the pickled state, so the dataset works correctly with multi-worker
    ``DataLoader``s, including under Windows' spawn start-method.
    """

    def __init__(self, bin_path: PathLike, block_size: int) -> None:
        self.bin_path = str(bin_path)
        self.meta = load_token_meta(self.bin_path)
        self.block_size = int(block_size)
        self.n_tokens = int(self.meta["count"])
        # Each block needs block_size inputs + 1 shifted target.
        self.n_blocks = max(0, (self.n_tokens - 1) // self.block_size)
        if self.n_blocks == 0:
            raise ValueError(
                f"token stream '{self.bin_path}' has only {self.n_tokens} tokens, "
                f"which is too short for block_size={self.block_size}"
            )
        self._data: np.memmap | None = None

    @property
    def data(self) -> np.memmap:
        """The memory-mapped token array (opened on first use)."""
        if self._data is None:
            self._data = np.memmap(
                self.bin_path, dtype=np.dtype(str(self.meta["dtype"])), mode="r"
            )
        return self._data

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        window = self.data[start : start + self.block_size + 1]
        # np.array(..) forces a contiguous int64 copy out of the memory map.
        x = torch.from_numpy(np.array(window[:-1], dtype=np.int64))
        y = torch.from_numpy(np.array(window[1:], dtype=np.int64))
        return x, y


def build_dataloader(
    bin_path: PathLike,
    block_size: int,
    batch_size: int,
    *,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """Construct a ``DataLoader`` over a token ``.bin`` file."""
    dataset = TokenDataset(bin_path, block_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False,
    )


def infinite_loader(
    loader: DataLoader,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches forever by cycling through ``loader`` -- handy for a
    step-based (rather than epoch-based) training loop."""
    while True:
        for batch in loader:
            yield batch
