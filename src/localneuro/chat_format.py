"""Conversation formatting -- the single source of truth for how a chat maps to
token ids.

Both interactive chat (:mod:`localneuro.inference.chat`) and instruction
fine-tuning (:mod:`localneuro.data.sft`) build their token sequences here, so
the layout the model is *trained* on is byte-identical to the layout it is
*prompted* with at inference time. A mismatch would teach the model a structure
it never actually sees, and SFT would not transfer to the chat UI.

The layout is::

    <bos> [<|system|> system]
          ( <|user|> text  |  <|assistant|> text <eos> )*
          [<|assistant|>]

User turns are not terminated; assistant turns end with ``<eos>``. A trailing
``<|assistant|>`` (the "generation prompt") invites the model to produce the
next reply.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .tokenizer.bpe import ASSISTANT, SYSTEM, USER, BPETokenizer

#: A single conversation turn: ``(role, text)`` with role ``"user"``/``"assistant"``.
Turn = Tuple[str, str]


def format_chat(
    tokenizer: BPETokenizer,
    turns: Sequence[Turn],
    system: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> List[int]:
    """Render a conversation into token ids.

    Parameters
    ----------
    turns:
        Ordered ``(role, text)`` pairs; ``role`` is ``"user"`` or ``"assistant"``.
    system:
        Optional system instruction placed first.
    add_generation_prompt:
        Append a trailing ``<|assistant|>`` so the model continues with a reply.
    """
    ids: List[int] = []
    if tokenizer.bos_id is not None:
        ids.append(tokenizer.bos_id)
    if system:
        ids.append(tokenizer.special_id(SYSTEM))
        ids.extend(tokenizer.encode_ordinary(system))
    for role, text in turns:
        if role == "user":
            ids.append(tokenizer.special_id(USER))
            ids.extend(tokenizer.encode_ordinary(text))
        elif role == "assistant":
            ids.append(tokenizer.special_id(ASSISTANT))
            ids.extend(tokenizer.encode_ordinary(text))
            if tokenizer.eos_id is not None:
                ids.append(tokenizer.eos_id)
        else:
            raise ValueError(f"unknown chat role: {role!r}")
    if add_generation_prompt:
        ids.append(tokenizer.special_id(ASSISTANT))
    return ids


def format_sft_example(
    tokenizer: BPETokenizer,
    instruction: str,
    response: str,
    system: Optional[str] = None,
) -> Tuple[List[int], int]:
    """Build one instruction/response training sequence.

    Returns ``(token_ids, prompt_len)``: the first ``prompt_len`` tokens are the
    prompt (everything through the opening ``<|assistant|>``); the remainder is
    the response followed by a trailing ``<eos>``. The fine-tuning loss is
    applied only to that remainder.
    """
    prompt = format_chat(
        tokenizer, [("user", instruction)], system=system, add_generation_prompt=True
    )
    response_ids = list(tokenizer.encode_ordinary(response))
    if tokenizer.eos_id is not None:
        response_ids.append(tokenizer.eos_id)
    return prompt + response_ids, len(prompt)
