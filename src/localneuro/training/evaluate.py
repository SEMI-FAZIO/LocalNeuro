"""Model evaluation: average loss and perplexity.

Perplexity is the exponential of the average per-token cross-entropy loss. It
has an intuitive reading: "on average the model is as uncertain as if it had to
choose uniformly among this many tokens". Lower is better; a perfect model
scores 1.0, a uniform-random model over an N-token vocabulary scores N.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .optim import autocast_context


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    max_batches: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    """Return the mean cross-entropy loss over up to ``max_batches`` batches.

    The model's train/eval mode is saved and restored, so this is safe to call
    from inside a training loop.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    batches = 0
    for index, (inputs, targets) in enumerate(loader):
        if index >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        with autocast_context(device, amp_dtype):
            _, loss = model(inputs, targets=targets)
        total_loss += float(loss.item())
        batches += 1
    if was_training:
        model.train()
    return total_loss / batches if batches else float("nan")


def perplexity(loss: float) -> float:
    """Convert an average cross-entropy loss into perplexity."""
    if loss != loss:  # NaN
        return float("nan")
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")
