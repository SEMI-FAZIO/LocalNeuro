"""LocalNeuro training package: optimizer, schedule, evaluation and trainer."""

from .evaluate import evaluate, perplexity
from .optim import build_optimizer, cosine_lr, resolve_precision, set_lr
from .trainer import Trainer

__all__ = [
    "Trainer",
    "build_optimizer",
    "cosine_lr",
    "set_lr",
    "resolve_precision",
    "evaluate",
    "perplexity",
]
