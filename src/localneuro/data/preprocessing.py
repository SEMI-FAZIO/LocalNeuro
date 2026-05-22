"""Text preprocessing for the training corpus.

Preprocessing is deliberately light: language models learn best from text that
looks like the text they will be asked to produce. We only fix things that are
genuinely noise -- inconsistent newlines, stray control characters and
non-canonical Unicode -- and leave casing, punctuation and spacing intact.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Union

# Control characters except tab (\t), newline (\n) and carriage return (\r).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Three or more consecutive blank lines collapse to two.
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(
    text: str,
    *,
    unicode_form: str = "NFC",
    strip_control: bool = True,
    normalize_newlines: bool = True,
    collapse_blank_lines: bool = True,
) -> str:
    """Clean a single string.

    Parameters
    ----------
    unicode_form:
        Unicode normalization form (``NFC`` recommended). Empty string skips it.
    strip_control:
        Remove non-printable control characters (keeps tab/newline).
    normalize_newlines:
        Convert Windows/Mac line endings to ``\\n``.
    collapse_blank_lines:
        Collapse runs of 3+ newlines into 2.
    """
    if unicode_form:
        text = unicodedata.normalize(unicode_form, text)
    if normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if strip_control:
        text = _CONTROL_RE.sub("", text)
    if collapse_blank_lines:
        text = _BLANK_LINES_RE.sub("\n\n", text)
    return text


def read_corpus(
    paths: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    normalize: bool = True,
    separator: str = "\n\n",
) -> str:
    """Read one or more UTF-8 text files and concatenate them into a corpus.

    Files are decoded with ``errors="replace"`` so a single bad byte never
    aborts a long preprocessing run.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    chunks: List[str] = []
    for path in paths:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        chunks.append(normalize_text(raw) if normalize else raw)
    return separator.join(chunks)


def split_train_val(text: str, val_fraction: float = 0.1) -> tuple[str, str]:
    """Split a corpus into train / validation halves at a line boundary.

    Splitting on a newline (rather than mid-line) avoids cutting a word in two.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    cut = int(len(text) * (1.0 - val_fraction))
    newline = text.find("\n", cut)
    if newline != -1:
        cut = newline + 1
    return text[:cut], text[cut:]
