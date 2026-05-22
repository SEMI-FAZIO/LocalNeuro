"""A from-scratch byte-level Byte-Pair-Encoding (BPE) tokenizer.

Why byte-level BPE?

* **Self-contained.** The base alphabet is the 256 possible byte values, so the
  tokenizer needs no external vocabulary file and can represent *any* input.
* **Multilingual for free.** Text is UTF-8 encoded before training, so Latin,
  Cyrillic, CJK, emoji -- everything -- is just a sequence of bytes. There is
  never an out-of-vocabulary token.
* **Trainable.** BPE starts from single bytes and repeatedly merges the most
  frequent adjacent pair, growing a vocabulary tuned to the training corpus.

The vocabulary id layout is::

    0   .. 255                         raw byte values
    256 .. 255 + num_merges            learned merge tokens
    ...        .. vocab_size - 1        special tokens (<pad>, <bos>, ...)

This module depends only on the Python standard library -- no torch, no numpy.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------
# Special tokens
# --------------------------------------------------------------------------
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"

#: Default special tokens. ``<pad>``/``<bos>``/``<eos>`` are the core trio; the
#: chat-role tokens let the same tokenizer drive the interactive chat REPL.
DEFAULT_SPECIAL_TOKENS: List[str] = [PAD, BOS, EOS, SYSTEM, USER, ASSISTANT]

#: Pre-tokenization pattern. Text is first split into "words" along this regex
#: so that merges never cross obvious boundaries (a merge spanning a word and
#: the following punctuation would waste vocabulary). ``\w`` is Unicode-aware,
#: which is what gives multilingual coverage. Digit groups are capped at three
#: characters to stop the model from minting one token per long number.
DEFAULT_SPLIT_PATTERN: str = (
    r"'(?:[sdmt]|ll|ve|re)"   # common English contractions
    r"| ?\d{1,3}"             # 1-3 digit groups, optional leading space
    r"| ?[^\s\d\W]+"          # letter runs, optional leading space
    r"| ?[^\s\w]+"            # punctuation / symbol runs, optional leading space
    r"|\s+"                   # whitespace runs
)

Pair = Tuple[int, int]


def _merge_symbols(symbols: List[int], pair: Pair, new_id: int) -> List[int]:
    """Return ``symbols`` with every non-overlapping ``pair`` replaced by ``new_id``."""
    a, b = pair
    out: List[int] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class BPETokenizer:
    """A trainable byte-level BPE tokenizer."""

    def __init__(
        self,
        merges: Dict[Pair, int],
        special_tokens: Dict[str, int],
        pattern: str = DEFAULT_SPLIT_PATTERN,
    ) -> None:
        # ``merges`` is ordered by rank (insertion order == merge priority).
        self.merges: Dict[Pair, int] = dict(merges)
        self.special_tokens: Dict[str, int] = dict(special_tokens)
        self.pattern = pattern

        self._compiled = re.compile(pattern)
        # Rank table used during encoding to pick which merge to apply next.
        self._merge_ranks: Dict[Pair, int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        # Reverse map id -> bytes for the non-special part of the vocabulary.
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        # Reverse map id -> special-token string.
        self._special_ids: Dict[int, str] = {
            idx: name for name, idx in self.special_tokens.items()
        }
        # Per-word encoding cache -- natural text repeats words constantly.
        self._cache: Dict[bytes, List[int]] = {}

        self.vocab_size = 256 + len(self.merges) + len(self.special_tokens)

    # ------------------------------------------------------------------ repr
    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"merges={len(self.merges)}, special={len(self.special_tokens)})"
        )

    # -------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: Optional[Sequence[str]] = None,
        pattern: str = DEFAULT_SPLIT_PATTERN,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Learn a BPE vocabulary from ``text``.

        Parameters
        ----------
        text:
            The training corpus as a single string.
        vocab_size:
            Target vocabulary size. The number of learned merges is
            ``vocab_size - 256 - len(special_tokens)``. If the corpus runs out
            of merge candidates first, the resulting vocabulary is simply
            smaller -- callers should read :attr:`vocab_size` afterwards.
        special_tokens:
            Names of the reserved special tokens (defaults to
            :data:`DEFAULT_SPECIAL_TOKENS`).
        pattern:
            Pre-tokenization regex.
        verbose:
            Print progress every 500 merges.
        """
        specials = list(special_tokens) if special_tokens is not None else list(
            DEFAULT_SPECIAL_TOKENS
        )
        num_merges = vocab_size - 256 - len(specials)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) is too small for 256 byte tokens "
                f"plus {len(specials)} special tokens"
            )

        compiled = re.compile(pattern)

        # 1. Pre-tokenize into words; count frequencies. Working on *unique*
        #    words keeps the merge loop fast on natural text.
        word_freqs: Counter = Counter()
        for chunk in compiled.findall(text):
            word_freqs[chunk.encode("utf-8")] += 1

        # 2. Each word starts as its list of raw byte ids.
        splits: Dict[bytes, List[int]] = {
            word: list(word) for word in word_freqs
        }

        # 3. Build the global pair-frequency table and an inverted index
        #    (pair -> words containing it) so updates stay local.
        pair_counts: Counter = Counter()
        pair_to_words: Dict[Pair, set] = defaultdict(set)
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(word)

        # 4. Greedily merge the most frequent pair, ``num_merges`` times.
        merges: Dict[Pair, int] = {}
        for rank in range(num_merges):
            if not pair_counts:
                break
            # Deterministic tie-break on the pair tuple itself.
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] <= 0:
                break
            new_id = 256 + rank
            merges[best] = new_id

            for word in list(pair_to_words[best]):
                freq = word_freqs[word]
                symbols = splits[word]
                # Withdraw this word's contribution to every pair it had.
                for i in range(len(symbols) - 1):
                    p = (symbols[i], symbols[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    bucket = pair_to_words.get(p)
                    if bucket is not None:
                        bucket.discard(word)
                # Apply the merge and re-add the updated word's pairs.
                new_symbols = _merge_symbols(symbols, best, new_id)
                splits[word] = new_symbols
                for i in range(len(new_symbols) - 1):
                    p = (new_symbols[i], new_symbols[i + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(word)

            if verbose and (rank + 1) % 500 == 0:
                print(f"  merges {rank + 1}/{num_merges}", flush=True)

        # 5. Assign ids to the special tokens after the merge tokens.
        base = 256 + len(merges)
        special_map = {name: base + i for i, name in enumerate(specials)}
        return cls(merges=merges, special_tokens=special_map, pattern=pattern)

    # -------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk: bytes) -> List[int]:
        """Encode one pre-tokenized word (its raw bytes) into token ids."""
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached

        ids: List[int] = list(chunk)
        while len(ids) >= 2:
            # Find the adjacent pair with the smallest (i.e. earliest) merge
            # rank -- that is the next merge BPE would have applied in training.
            best_pair: Optional[Pair] = None
            best_rank = len(self._merge_ranks) + 1
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            ids = _merge_symbols(ids, best_pair, self.merges[best_pair])

        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> List[int]:
        """Encode plain text. Special-token strings are NOT recognised here."""
        ids: List[int] = []
        for chunk in self._compiled.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(
        self,
        text: str,
        allowed_special: Union[None, str, Iterable[str]] = None,
    ) -> List[int]:
        """Encode text, optionally recognising special-token substrings.

        ``allowed_special`` may be ``None`` (recognise none -- the safe
        default), ``"all"``, or an explicit iterable of special-token names.
        """
        if allowed_special is None:
            return self.encode_ordinary(text)
        if allowed_special == "all":
            allowed = set(self.special_tokens)
        else:
            allowed = set(allowed_special)  # type: ignore[arg-type]
            unknown = allowed - set(self.special_tokens)
            if unknown:
                raise ValueError(f"unknown special tokens: {sorted(unknown)}")
        if not allowed:
            return self.encode_ordinary(text)

        # Longer names first so e.g. "<|assistant|>" is not shadowed.
        ordered = sorted(allowed, key=len, reverse=True)
        splitter = "(" + "|".join(re.escape(s) for s in ordered) + ")"
        ids: List[int] = []
        for part in re.split(splitter, text):
            if not part:
                continue
            if part in allowed:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    # -------------------------------------------------------------- decoding
    def id_to_bytes(self, token_id: int) -> bytes:
        """Raw bytes for a token id. Special tokens return ``b""``.

        This is the primitive used by :class:`StreamDecoder`; special tokens
        are control symbols and should not be rendered into generated text.
        """
        return self.vocab.get(token_id, b"")

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        """Decode a sequence of token ids back into text.

        Special tokens are rendered as their literal name (``<eos>`` etc.)
        unless ``skip_special`` is set. Invalid UTF-8 byte sequences are
        replaced rather than raising.
        """
        parts: List[bytes] = []
        for token_id in ids:
            special = self._special_ids.get(token_id)
            if special is not None:
                if not skip_special:
                    parts.append(special.encode("utf-8"))
            else:
                parts.append(self.vocab.get(token_id, b""))
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------- special tokens
    def is_special(self, token_id: int) -> bool:
        return token_id in self._special_ids

    def special_id(self, name: str) -> int:
        """Return the id of a special token by name (raises if absent)."""
        return self.special_tokens[name]

    @property
    def pad_id(self) -> Optional[int]:
        return self.special_tokens.get(PAD)

    @property
    def bos_id(self) -> Optional[int]:
        return self.special_tokens.get(BOS)

    @property
    def eos_id(self) -> Optional[int]:
        return self.special_tokens.get(EOS)

    # ----------------------------------------------------------- (de)serialize
    def save(self, path: Union[str, Path]) -> None:
        """Persist the tokenizer to a compact JSON file.

        Only the merge list (in rank order) and the special-token map need to
        be stored -- the byte alphabet and the full vocabulary are derived on
        load. The new id of merge ``rank`` is implicitly ``256 + rank``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": "localneuro-bpe",
            "version": 1,
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for (a, b) in self.merges],
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """Load a tokenizer previously written by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "localneuro-bpe":
            raise ValueError(f"{path} is not a LocalNeuro tokenizer file")
        merges: Dict[Pair, int] = {}
        for rank, pair in enumerate(data["merges"]):
            merges[(pair[0], pair[1])] = 256 + rank
        return cls(
            merges=merges,
            special_tokens={k: int(v) for k, v in data["special_tokens"].items()},
            pattern=data["pattern"],
        )


class StreamDecoder:
    """Incrementally decode token ids into text, UTF-8 safe.

    A single multi-byte UTF-8 character can be split across two tokens. Naively
    decoding token-by-token would emit replacement characters at every such
    boundary. This decoder buffers raw bytes and only emits text once it forms
    complete characters -- which is exactly what a streaming chat UI needs.
    """

    def __init__(self, tokenizer: BPETokenizer) -> None:
        self._tokenizer = tokenizer
        self._buffer = b""

    def step(self, token_id: int) -> str:
        """Feed one token id, return whatever text is now safe to emit."""
        data = self._tokenizer.id_to_bytes(token_id)
        if not data:
            return ""
        self._buffer += data
        try:
            text = self._buffer.decode("utf-8")
            self._buffer = b""
            return text
        except UnicodeDecodeError as exc:
            if exc.start == 0:
                # No decodable prefix yet. A UTF-8 character is at most 4 bytes,
                # so 4+ undecodable leading bytes must be genuinely invalid.
                if len(self._buffer) >= 4:
                    bad, self._buffer = self._buffer[:1], self._buffer[1:]
                    return bad.decode("utf-8", errors="replace")
                return ""
            good = self._buffer[: exc.start]
            self._buffer = self._buffer[exc.start :]
            return good.decode("utf-8")

    def flush(self) -> str:
        """Emit any remaining buffered bytes (with replacement for leftovers)."""
        text = self._buffer.decode("utf-8", errors="replace")
        self._buffer = b""
        return text
