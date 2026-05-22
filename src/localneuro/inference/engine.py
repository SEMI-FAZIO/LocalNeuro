"""The inference engine: streaming, KV-cached autoregressive generation.

Generation proceeds in two phases:

1. **Prefill** -- the whole prompt is run through the model once; the keys and
   values for every prompt position are written into the KV cache.
2. **Decode** -- one token at a time. Only the single new token is fed to the
   model; the cache supplies the keys/values for the entire prefix, so each
   step costs O(context) rather than O(context^2).

The engine yields tokens as soon as they are produced, which is what makes a
responsive streaming UI possible.
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Optional, Sequence, Union

import torch

from ..sampling import SamplingConfig, sample_token
from ..tokenizer.bpe import BPETokenizer, StreamDecoder


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model, seeing through a ``torch.compile`` wrapper."""
    return getattr(model, "_orig_mod", model)


class InferenceEngine:
    """Runs text generation for a trained LocalNeuro model."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: BPETokenizer,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self._base = _unwrap(self.model)

    @property
    def max_context(self) -> int:
        """The model's maximum sequence length."""
        return self._base.config.max_seq_len

    # --------------------------------------------------------------- tokens
    def generate_stream(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 128,
        sampling: Optional[SamplingConfig] = None,
        stop_ids: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ) -> Iterator[int]:
        """Yield generated token ids one at a time.

        Stops after ``max_new_tokens``, when a token in ``stop_ids`` is drawn,
        or when the context window fills up.
        """
        sampling = sampling or SamplingConfig()
        stop = {int(s) for s in (stop_ids or [])}
        prompt = [int(t) for t in prompt_ids]
        if not prompt:
            raise ValueError("prompt_ids must not be empty")

        max_ctx = self.max_context
        if len(prompt) > max_ctx - 1:
            # Keep the most recent tokens; leave room for at least one new token.
            prompt = prompt[-(max_ctx - 1):]

        cache = self._base.make_kv_cache(
            batch_size=1, max_seq_len=max_ctx, device=self.device
        )
        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))

        recent = list(prompt)
        idx = torch.tensor([prompt], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(idx, kv_cache=cache, start_pos=0)
        position = len(prompt)

        for _ in range(max_new_tokens):
            # Sampling happens on CPU so a seeded generator behaves identically
            # regardless of the compute device.
            next_id = sample_token(
                logits[0, -1, :].detach().to("cpu"),
                sampling,
                recent_tokens=recent,
                generator=generator,
            )
            if next_id in stop:
                break
            yield next_id
            recent.append(next_id)
            if position >= max_ctx:
                break  # context window exhausted
            idx = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits, _ = self.model(idx, kv_cache=cache, start_pos=position)
            position += 1

    def generate(self, prompt_ids: Sequence[int], **kwargs) -> List[int]:
        """Non-streaming variant: return the full list of generated ids."""
        return list(self.generate_stream(prompt_ids, **kwargs))

    # ----------------------------------------------------------------- text
    def stream_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        sampling: Optional[SamplingConfig] = None,
        add_bos: bool = True,
        stop_at_eos: bool = True,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        """Generate a continuation of ``prompt``, yielding decoded text fragments.

        The :class:`StreamDecoder` guarantees that multi-byte UTF-8 characters
        are never split across two fragments.
        """
        prompt_ids: List[int] = []
        if add_bos and self.tokenizer.bos_id is not None:
            prompt_ids.append(self.tokenizer.bos_id)
        prompt_ids.extend(self.tokenizer.encode_ordinary(prompt))

        stop_ids: List[int] = []
        if stop_at_eos and self.tokenizer.eos_id is not None:
            stop_ids.append(self.tokenizer.eos_id)

        decoder = StreamDecoder(self.tokenizer)
        for token_id in self.generate_stream(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            stop_ids=stop_ids,
            seed=seed,
        ):
            fragment = decoder.step(token_id)
            if fragment:
                yield fragment
        tail = decoder.flush()
        if tail:
            yield tail

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        sampling: Optional[SamplingConfig] = None,
        add_bos: bool = True,
        stop_at_eos: bool = True,
        seed: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate a continuation of ``prompt`` and return it as one string.

        If ``on_token`` is given it receives each text fragment as soon as it is
        ready -- use it to stream output to a console.
        """
        pieces: List[str] = []
        for fragment in self.stream_text(
            prompt,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            add_bos=add_bos,
            stop_at_eos=stop_at_eos,
            seed=seed,
        ):
            pieces.append(fragment)
            if on_token is not None:
                on_token(fragment)
        return "".join(pieces)
