"""Supervised instruction fine-tuning (SFT) data.

Pretraining teaches the model to *continue* text. SFT teaches it to *answer*:
the model is shown ``instruction -> response`` pairs laid out in the chat format
(see :mod:`localneuro.chat_format`), and the loss is applied **only** to the
response tokens. Prompt tokens are masked with ``-1``, which the model's
cross-entropy (``ignore_index=-1``) skips. After SFT the model follows the chat
layout and produces a real answer instead of rambling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from ..chat_format import format_sft_example
from ..tokenizer.bpe import BPETokenizer

PathLike = Union[str, Path]


@dataclass
class InstructionExample:
    """One instruction/response pair, optionally with a system prompt."""

    instruction: str
    response: str
    system: Optional[str] = None


def read_instructions_jsonl(path: PathLike) -> List[InstructionExample]:
    """Load instruction examples from a JSON Lines file.

    Each line is an object with ``instruction`` and ``response`` keys and an
    optional ``system`` key -- a format that is easy to inspect and hand-edit.
    """
    examples: List[InstructionExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        examples.append(
            InstructionExample(
                instruction=str(obj["instruction"]),
                response=str(obj["response"]),
                system=obj.get("system"),
            )
        )
    return examples


def write_instructions_jsonl(
    path: PathLike, examples: Sequence[InstructionExample]
) -> None:
    """Write instruction examples to a JSON Lines file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for example in examples:
            obj = {"instruction": example.instruction, "response": example.response}
            if example.system:
                obj["system"] = example.system
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_sft_example(
    example: InstructionExample,
    tokenizer: BPETokenizer,
    max_len: int,
) -> Optional[Tuple[List[int], List[int]]]:
    """Turn one instruction example into ``(input_ids, labels)``.

    ``labels`` are the next-token targets; every position that predicts a
    *prompt* token is set to ``-1`` so the loss covers only the response and its
    trailing ``<eos>``. Returns ``None`` if the response is empty or the full
    sequence is longer than ``max_len``.
    """
    if not example.response.strip():
        return None
    token_ids, prompt_len = format_sft_example(
        tokenizer, example.instruction, example.response, system=example.system
    )
    if len(token_ids) > max_len or len(token_ids) < 2:
        return None
    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    # labels[i] predicts token_ids[i+1]; mask while that target is still part
    # of the prompt (i + 1 < prompt_len).
    for i in range(min(prompt_len - 1, len(labels))):
        labels[i] = -1
    return input_ids, labels


class SFTDataset(Dataset):
    """A dataset of instruction examples, padded to a fixed block size.

    Each item is one example. The input is padded with the pad token and the
    labels with ``-1``, so both padding and prompt positions are ignored by the
    loss. Padding sits *after* the response and, because attention is causal, it
    cannot influence the real tokens -- so no attention mask is needed.
    """

    def __init__(
        self,
        examples: Sequence[InstructionExample],
        tokenizer: BPETokenizer,
        block_size: int,
    ) -> None:
        self.block_size = int(block_size)
        pad = tokenizer.pad_id
        self.pad_id = pad if pad is not None else 0
        self.samples: List[Tuple[List[int], List[int]]] = []
        self.skipped = 0
        for example in examples:
            built = build_sft_example(
                example, tokenizer, max_len=self.block_size + 1
            )
            if built is None:
                self.skipped += 1
            else:
                self.samples.append(built)
        if not self.samples:
            raise ValueError(
                "no usable SFT examples -- all were empty or longer than "
                f"block_size ({self.block_size})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.samples[index]
        x = input_ids + [self.pad_id] * (self.block_size - len(input_ids))
        y = labels + [-1] * (self.block_size - len(labels))
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def build_sft_dataloader(
    examples: Sequence[InstructionExample],
    tokenizer: BPETokenizer,
    block_size: int,
    batch_size: int,
    *,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Construct a ``DataLoader`` over an :class:`SFTDataset`.

    ``drop_last`` is intentionally false -- instruction sets are small and every
    example is worth keeping.
    """
    dataset = SFTDataset(examples, tokenizer, block_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )
