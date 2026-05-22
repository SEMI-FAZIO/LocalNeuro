"""Token sampling strategies for text generation.

Given the model's raw logits for the next token, these helpers shape the
probability distribution before a token is drawn from it:

* **temperature**        -- flattens (>1) or sharpens (<1) the distribution.
* **top-k**              -- keep only the ``k`` most likely tokens.
* **top-p (nucleus)**    -- keep the smallest set of tokens whose cumulative
                            probability reaches ``p``.
* **repetition penalty** -- down-weight tokens that appeared recently, which
                            curbs the degenerate looping small models love.

Setting ``temperature = 0`` switches to deterministic greedy decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class SamplingConfig:
    """Bundle of decoding hyper-parameters."""

    temperature: float = 0.8
    top_k: int = 40            # 0 disables
    top_p: float = 0.95        # 1.0 disables
    repetition_penalty: float = 1.1   # 1.0 disables
    repetition_window: int = 128      # how many recent tokens the penalty sees

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.repetition_window < 0:
            raise ValueError("repetition_window must be >= 0")


def apply_repetition_penalty(
    logits: torch.Tensor, recent_tokens: Sequence[int], penalty: float
) -> torch.Tensor:
    """Down-weight the logits of tokens that occur in ``recent_tokens``.

    Uses the CTRL-paper rule: positive logits are divided by the penalty and
    negative logits are multiplied by it, which always moves a token's score
    *towards* being less likely regardless of its sign.
    """
    if penalty == 1.0 or not recent_tokens:
        return logits
    unique = torch.tensor(
        sorted(set(int(t) for t in recent_tokens)),
        dtype=torch.long,
        device=logits.device,
    )
    logits = logits.clone()
    selected = logits[unique]
    logits[unique] = torch.where(
        selected > 0, selected / penalty, selected * penalty
    )
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask every logit outside the top ``k`` with ``-inf``."""
    if k <= 0 or k >= logits.numel():
        return logits
    threshold = torch.topk(logits, k).values[-1]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest token set with cumulative prob >= ``p``."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # Remove a token if the probability mass *before* it already exceeds p.
    # This keeps the token that crosses the threshold and always keeps the top-1.
    remove = (cumulative - probs) > p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    restored = torch.empty_like(logits)
    restored.scatter_(0, sorted_idx, sorted_logits)
    return restored


def sample_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    recent_tokens: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Pick the next token id from a 1-D logits vector.

    The pipeline is: repetition penalty -> temperature -> top-k -> top-p ->
    multinomial draw. ``temperature == 0`` short-circuits to greedy argmax.
    """
    if logits.dim() != 1:
        raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")
    logits = logits.detach().float()

    if recent_tokens and config.repetition_penalty != 1.0:
        window = recent_tokens[-config.repetition_window:] if config.repetition_window else recent_tokens
        logits = apply_repetition_penalty(logits, window, config.repetition_penalty)

    if config.temperature == 0.0:
        return int(torch.argmax(logits))

    logits = logits / config.temperature
    logits = top_k_filter(logits, config.top_k)
    logits = top_p_filter(logits, config.top_p)

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_token)
