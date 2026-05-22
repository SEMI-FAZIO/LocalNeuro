"""Shared utilities: determinism, device selection, logging and formatting.

These helpers are intentionally tiny and dependency-light so they can be reused
by training, inference and the test-suite alike.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG that LocalNeuro touches for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_console() -> None:
    """Make stdout/stderr tolerant of characters the console cannot encode.

    Generated text is byte-level and may contain any Unicode character -- or
    U+FFFD where the model emitted an invalid byte sequence. On a console with a
    narrow code page (e.g. Windows cp1251) printing such a character would raise
    ``UnicodeEncodeError``; switching the error handler to ``replace`` prints a
    placeholder instead of crashing. The console's encoding is left untouched so
    text it *can* represent still renders correctly.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # stream not reconfigurable (e.g. piped)
            pass


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a compute device.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU. Any explicit string is
    honoured as-is so users can force CPU even when a GPU is present.
    """
    preference = preference.lower()
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_threads(num_threads: int) -> None:
    """Pin the CPU thread count. ``0`` leaves the PyTorch default untouched."""
    if num_threads > 0:
        torch.set_num_threads(num_threads)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a module."""
    return sum(
        p.numel()
        for p in model.parameters()
        if (p.requires_grad or not trainable_only)
    )


def human_count(n: float) -> str:
    """Render a number compactly, e.g. ``7_340_032 -> '7.34M'``."""
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000.0:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def human_bytes(n: float) -> str:
    """Render a byte count compactly, e.g. ``'28.0 MiB'``."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def format_duration(seconds: float) -> str:
    """Human-readable elapsed time."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@dataclass
class Logger:
    """Minimal logger: prints to stdout and appends structured JSONL records.

    A JSONL log is far easier to post-process than scraped console output, and
    it costs nothing to keep both.
    """

    log_path: Optional[Path] = None
    quiet: bool = False
    _start: float = 0.0

    def __post_init__(self) -> None:
        self._start = time.time()
        if self.log_path is not None:
            self.log_path = Path(self.log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str) -> None:
        """Print a free-form line."""
        if not self.quiet:
            print(message, flush=True)

    def log(self, record: Dict[str, Any]) -> None:
        """Record a structured metric event (printed and persisted as JSONL)."""
        record = {"elapsed": round(time.time() - self._start, 2), **record}
        if not self.quiet:
            pretty = "  ".join(
                f"{k}={_fmt(v)}" for k, v in record.items() if k != "elapsed"
            )
            print(f"[{format_duration(record['elapsed'])}] {pretty}", flush=True)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def progress_bar(current: int, total: int, width: int = 28) -> str:
    """Return a textual progress bar string (no carriage returns added)."""
    total = max(total, 1)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * current / total
    return f"[{bar}] {pct:5.1f}% ({current}/{total})"
