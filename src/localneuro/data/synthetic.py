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
from typing import TYPE_CHECKING, Callable, List, Tuple, Union

if TYPE_CHECKING:  # avoids importing the torch-dependent sft module at load time
    from .sft import InstructionExample

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


def _gen_two_sentence(rng: random.Random) -> str:
    noun = rng.choice(_NOUNS)
    return (
        f"the {noun} was {rng.choice(_ADJECTIVES)}. "
        f"it {rng.choice(_VERBS)} {rng.choice(_PLACES)}."
    )


def _gen_because(rng: random.Random) -> str:
    return (
        f"the {rng.choice(_NOUNS)} {rng.choice(_VERBS)} {rng.choice(_PLACES)} "
        f"because the night was {rng.choice(_ADJECTIVES)}."
    )


def _gen_compare(rng: random.Random) -> str:
    first, second = rng.sample(_NOUNS, k=2)
    return (
        f"the {first} is {rng.choice(_ADJECTIVES)}, "
        f"but the {second} is {rng.choice(_ADJECTIVES)}."
    )


_GENERATORS: List[Callable[[random.Random], str]] = [
    _gen_sentence,
    _gen_count_words,
    _gen_count_digits,
    _gen_arithmetic,
    _gen_fact,
    _gen_question,
    _gen_list,
    _gen_echo,
    _gen_two_sentence,
    _gen_because,
    _gen_compare,
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


# --------------------------------------------------------------------------
# Synthetic instruction / response pairs (for supervised fine-tuning)
# --------------------------------------------------------------------------

#: Hand-written question/answer pairs that give the chat sensible, on-theme
#: replies. They are mixed in with the procedurally generated pairs below.
_FIXED_QA: List[Tuple[str, str]] = [
    ("Hello.", "Hello. How can I help you?"),
    ("Hi.", "Hi there. What would you like to know?"),
    ("Who are you?", "I am LocalNeuro, a small language model."),
    ("What is LocalNeuro?",
     "LocalNeuro is a small language model that runs on your own computer."),
    ("What is a language model?",
     "A language model predicts the next piece of text from the text before it."),
    ("What can you do?",
     "I can answer simple questions, do small arithmetic, and continue text."),
    ("What is a token?",
     "A token is a small piece of text that the model works with."),
    ("What is attention?",
     "Attention lets each token look back at the tokens that came before it."),
    ("What is training?",
     "Training adjusts the model's weights so its predictions get better."),
    ("How do you work?",
     "I read the words so far and predict the next one, again and again."),
    ("Thank you.", "You are welcome."),
    ("Goodbye.", "Goodbye. Come back any time."),
]


def _si_add(rng: random.Random) -> Tuple[str, str]:
    a, b = rng.randint(0, 30), rng.randint(0, 30)
    return f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}."


def _si_sub(rng: random.Random) -> Tuple[str, str]:
    a = rng.randint(0, 40)
    b = rng.randint(0, a)
    return f"What is {a} minus {b}?", f"{a} minus {b} is {a - b}."


def _si_mul(rng: random.Random) -> Tuple[str, str]:
    a, b = rng.randint(0, 12), rng.randint(0, 12)
    return f"What is {a} times {b}?", f"{a} times {b} is {a * b}."


def _si_count(rng: random.Random) -> Tuple[str, str]:
    start = rng.randint(0, 80)
    length = rng.randint(4, 7)
    sequence = " ".join(str(start + i) for i in range(length))
    return f"Count from {start} to {start + length - 1}.", sequence


def _si_next_number(rng: random.Random) -> Tuple[str, str]:
    index = rng.randint(0, len(_NUMBER_WORDS) - 2)
    return (
        f"What comes after {_NUMBER_WORDS[index]}?",
        f"The next number is {_NUMBER_WORDS[index + 1]}.",
    )


def _si_color(rng: random.Random) -> Tuple[str, str]:
    noun = rng.choice(_NOUNS)
    return f"What color is the {noun}?", f"The {noun} is {_color_of(noun)}."


def _si_describe(rng: random.Random) -> Tuple[str, str]:
    noun = rng.choice(_NOUNS)
    return (
        f"Describe the {noun}.",
        f"The {rng.choice(_ADJECTIVES)} {noun} {rng.choice(_VERBS)} "
        f"{rng.choice(_PLACES)}.",
    )


def _si_list(rng: random.Random) -> Tuple[str, str]:
    count = rng.randint(3, 4)
    return f"List {count} things.", ", ".join(rng.sample(_NOUNS, k=count)) + "."


def _si_reverse(rng: random.Random) -> Tuple[str, str]:
    items = rng.sample(_NOUNS, k=3)
    return (
        "Reverse this list: " + ", ".join(items) + ".",
        ", ".join(reversed(items)) + ".",
    )


def _si_first(rng: random.Random) -> Tuple[str, str]:
    items = rng.sample(_NOUNS, k=3)
    return (
        "What is the first item in: " + ", ".join(items) + "?",
        f"The first item is {items[0]}.",
    )


def _si_last(rng: random.Random) -> Tuple[str, str]:
    items = rng.sample(_NOUNS, k=3)
    return (
        "What is the last item in: " + ", ".join(items) + "?",
        f"The last item is {items[-1]}.",
    )


def _si_repeat(rng: random.Random) -> Tuple[str, str]:
    word = rng.choice(_NOUNS)
    count = rng.randint(2, 4)
    return f"Repeat the word {word} {count} times.", " ".join([word] * count) + "."


def _si_fixed(rng: random.Random) -> Tuple[str, str]:
    return rng.choice(_FIXED_QA)


_INSTRUCTION_GENERATORS: List[Callable[[random.Random], Tuple[str, str]]] = [
    _si_add, _si_sub, _si_mul, _si_count, _si_next_number, _si_color,
    _si_describe, _si_list, _si_reverse, _si_first, _si_last, _si_repeat,
    # _si_fixed appears twice so the hand-written Q&A is seen more often.
    _si_fixed, _si_fixed,
]


def generate_synthetic_instructions(
    num_samples: int = 3000, seed: int = 0
) -> List["InstructionExample"]:
    """Generate deterministic synthetic instruction/response pairs for SFT.

    The categories (arithmetic, counting, facts, lists) deliberately overlap the
    plain-text synthetic corpus so fine-tuning builds on what the base model has
    already learned during pretraining.
    """
    from .sft import InstructionExample  # local import keeps this module light

    rng = random.Random(seed)
    examples: List[InstructionExample] = []
    for _ in range(num_samples):
        instruction, response = rng.choice(_INSTRUCTION_GENERATORS)(rng)
        examples.append(
            InstructionExample(instruction=instruction, response=response)
        )
    return examples

