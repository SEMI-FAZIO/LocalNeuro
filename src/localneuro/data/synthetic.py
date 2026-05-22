"""Synthetic dataset generation.

Real corpora are great, but for smoke-testing the whole pipeline -- and for
teaching -- a procedurally generated corpus is invaluable: it is deterministic,
needs no download, and contains *crisp, learnable patterns*. A tiny model
trained for a few minutes on this data will visibly learn to count, to continue
arithmetic, and to complete the templated sentence forms below. That makes it
an honest end-to-end test that the architecture, tokenizer, trainer and
inference engine all actually work.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Union

_NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]
_NOUNS = [
    "cat", "dog", "bird", "fish", "robot", "river", "mountain", "star",
    "engine", "garden", "wizard", "captain", "forest", "machine", "planet",
    "signal", "bridge", "candle", "mirror", "anchor",
]
_ADJECTIVES = [
    "small", "bright", "quiet", "ancient", "rapid", "gentle", "frozen",
    "golden", "hidden", "clever", "silent", "distant", "warm", "steady",
]
_VERBS = [
    "sees", "finds", "builds", "carries", "watches", "follows", "opens",
    "keeps", "draws", "sends", "holds", "moves", "guards", "repairs",
]
_PLACES = [
    "the harbor", "the valley", "the city", "the desert", "the station",
    "the library", "the workshop", "the meadow", "the tower", "the canyon",
]
_COLORS = ["red", "blue", "green", "yellow", "silver", "orange", "violet", "white"]


def _color_of(noun: str) -> str:
    """Deterministically map a noun to a colour (stable across processes)."""
    return _COLORS[_NOUNS.index(noun) % len(_COLORS)]


def _gen_sentence(rng: random.Random) -> str:
    return (
        f"the {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} "
        f"{rng.choice(_VERBS)} {rng.choice(_PLACES)}."
    )


def _gen_count_words(rng: random.Random) -> str:
    start = rng.randint(0, 14)
    length = rng.randint(4, 6)
    return "count: " + " ".join(_NUMBER_WORDS[start:start + length]) + " ."


def _gen_count_digits(rng: random.Random) -> str:
    start = rng.randint(0, 80)
    length = rng.randint(4, 7)
    return "numbers: " + " ".join(str(start + i) for i in range(length)) + " ."


def _gen_arithmetic(rng: random.Random) -> str:
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        a, b = rng.randint(0, 30), rng.randint(0, 30)
        result = a + b
    elif op == "-":
        a = rng.randint(0, 40)
        b = rng.randint(0, a)
        result = a - b
    else:
        a, b = rng.randint(0, 12), rng.randint(0, 12)
        result = a * b
    return f"{a} {op} {b} = {result}"


def _gen_fact(rng: random.Random) -> str:
    noun = rng.choice(_NOUNS)
    return f"the {noun} is {_color_of(noun)}."


def _gen_question(rng: random.Random) -> str:
    noun = rng.choice(_NOUNS)
    return (
        f"question: what color is the {noun}? "
        f"answer: the {noun} is {_color_of(noun)}."
    )


def _gen_list(rng: random.Random) -> str:
    items = rng.sample(_NOUNS, k=rng.randint(3, 5))
    return "list: " + ", ".join(items) + "."


def _gen_echo(rng: random.Random) -> str:
    word = rng.choice(_NOUNS)
    return "echo: " + " ".join([word] * rng.randint(2, 4)) + "."


_GENERATORS: List[Callable[[random.Random], str]] = [
    _gen_sentence,
    _gen_count_words,
    _gen_count_digits,
    _gen_arithmetic,
    _gen_fact,
    _gen_question,
    _gen_list,
    _gen_echo,
]


def generate_synthetic_corpus(num_samples: int = 4000, seed: int = 0) -> str:
    """Generate a deterministic synthetic corpus of ``num_samples`` lines."""
    rng = random.Random(seed)
    lines = [rng.choice(_GENERATORS)(rng) for _ in range(num_samples)]
    return "\n".join(lines) + "\n"


def write_synthetic_corpus(
    path: Union[str, Path], num_samples: int = 4000, seed: int = 0
) -> int:
    """Generate a synthetic corpus and write it to ``path``. Returns char count."""
    text = generate_synthetic_corpus(num_samples=num_samples, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return len(text)
