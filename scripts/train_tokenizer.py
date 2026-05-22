"""Train a byte-level BPE tokenizer from one or more text files.

Examples
--------
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --input my_corpus.txt --vocab-size 16000
    python scripts/train_tokenizer.py --input a.txt b.txt --synthetic-samples 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `localneuro` importable when running straight from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.data.preprocessing import read_corpus
from localneuro.data.synthetic import generate_synthetic_corpus
from localneuro.tokenizer.bpe import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", nargs="*", default=["datasets/demo/tiny_corpus.txt"],
        help="text file(s) to learn the vocabulary from",
    )
    parser.add_argument(
        "--output", default="checkpoints/tokenizer.json",
        help="where to write the tokenizer JSON",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=4096,
        help="target vocabulary size (actual size may be smaller for a tiny corpus)",
    )
    parser.add_argument(
        "--synthetic-samples", type=int, default=0,
        help="append this many lines of synthetic data to the training corpus",
    )
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="print merge progress")
    args = parser.parse_args()

    parts = []
    existing = [p for p in args.input if Path(p).exists()]
    missing = [p for p in args.input if not Path(p).exists()]
    for path in missing:
        print(f"warning: input file not found, skipping: {path}")
    if existing:
        parts.append(read_corpus(existing))
    if args.synthetic_samples > 0:
        parts.append(generate_synthetic_corpus(args.synthetic_samples, args.synthetic_seed))
    if not parts:
        raise SystemExit("no training text: pass --input files and/or --synthetic-samples")

    corpus = "\n\n".join(parts)
    print(f"corpus: {len(corpus):,} characters")
    print(f"training BPE tokenizer (target vocab {args.vocab_size}) ...")

    tokenizer = BPETokenizer.train(
        corpus, vocab_size=args.vocab_size, verbose=args.verbose
    )
    tokenizer.save(args.output)

    print(f"done: {tokenizer.vocab_size} tokens "
          f"({len(tokenizer.merges)} merges) -> {args.output}")

    # A quick round-trip sanity check on a multilingual sample.
    sample = "Hello, LocalNeuro! Привет. 123 + 4 = 127."
    restored = tokenizer.decode(tokenizer.encode_ordinary(sample))
    status = "ok" if restored == sample else "MISMATCH"
    print(f"round-trip check: {status}")


if __name__ == "__main__":
    main()
