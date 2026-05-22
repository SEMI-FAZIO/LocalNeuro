"""Optimizer construction, learning-rate schedule and mixed-precision helpers."""

from __future__ import annotations

import contextlib
import math
from typing import ContextManager, Optional, Tuple

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: Tuple[float, float],
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer with sensible weight-decay groups.

    Weight decay is applied only to parameters that take part in matrix
    multiplications (2-D and larger). Bias-like and normalization parameters
    (1-D) are excluded -- decaying them slightly hurts and is standard practice.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)


def cosine_lr(
    step: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Learning rate at ``step``: linear warm-up then cosine decay to ``min_lr``.

    Warm-up avoids the large, noisy updates of the first few steps; cosine decay
    spends most of training at a high rate and gently anneals at the end.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Overwrite the learning rate of every parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = lr


def resolve_precision(
    precision: str, device: torch.device
) -> Tuple[Optional[torch.dtype], bool]:
    """Map a precision string to ``(autocast_dtype, use_grad_scaler)``.

    * ``fp32`` -- no autocast.
    * ``bf16`` -- bfloat16 autocast (works on modern CPUs and CUDA; the wide
      exponent range means no gradient scaler is needed).
    * ``fp16`` -- float16 autocast, CUDA only, and needs a gradient scaler.

    On Apple MPS autocast is unreliable for tiny models, so fp32 is forced.
    """
    if precision == "fp32":
        return None, False
    if device.type == "mps":
        return None, False
    if precision == "bf16":
        return torch.bfloat16, False
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 mixed precision requires CUDA; use 'bf16' on CPU")
        return torch.float16, True
    raise ValueError(f"unknown precision: {precision!r}")


def autocast_context(
    device: torch.device, amp_dtype: Optional[torch.dtype]
) -> ContextManager:
    """Return an autocast context (or a no-op context for fp32)."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def make_grad_scaler(enabled: bool) -> "torch.amp.GradScaler":
    """Construct a gradient scaler, portably across torch versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)  # torch >= 2.3
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)
