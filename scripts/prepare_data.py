"""Tokenize a text corpus into the binary token streams used for training.

Produces ``train.bin`` and ``val.bin`` (plus ``.meta.json`` sidecars) in the
output directory.

Examples
--------
    python scripts/prepare_data.py
    python scripts/prepare_data.py --input my_corpus.txt --output-dir datasets/mine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.data.dataset import encode_corpus, write_token_bin
from localneuro.data.preprocessing import read_corpus, split_train_val
from localneuro.data.synthetic import generate_synthetic_corpus
from localneuro.tokenizer.bpe import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    parser.add_argument("--input", nargs="*", default=["datasets/demo/tiny_corpus.txt"])
    parser.add_argument("--output-dir", default="datasets/demo")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--synthetic-samples", type=int, default=0,
        help="append this many lines of synthetic data to the corpus",
    )
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument(
        "--doc-separator", default="\\n\\n",
        help="string that separates documents (default: a blank line)",
    )
    args = parser.parse_args()

    if not Path(args.tokenizer).exists():
        raise SystemExit(
            f"tokenizer not found: {args.tokenizer}\n"
            "Run scripts/train_tokenizer.py first."
        )
    tokenizer = BPETokenizer.load(args.tokenizer)
    separator = args.doc_separator.encode().decode("unicode_escape")

    parts = []
    existing = [p for p in args.input if Path(p).exists()]
    for path in (p for p in args.input if not Path(p).exists()):
        print(f"warning: input file not found, skipping: {path}")
    if existing:
        parts.append(read_corpus(existing))
    if args.synthetic_samples > 0:
        parts.append(generate_synthetic_corpus(args.synthetic_samples, args.synthetic_seed))
    if not parts:
        raise SystemExit("no input text: pass --input files and/or --synthetic-samples")

    corpus = "\n\n".join(parts)
    train_text, val_text = split_train_val(corpus, args.val_fraction)
    print(f"corpus: {len(corpus):,} chars -> "
          f"train {len(train_text):,} / val {len(val_text):,}")

    train_ids = encode_corpus(train_text, tokenizer, doc_separator=separator)
    val_ids = encode_corpus(val_text, tokenizer, doc_separator=separator)

    out_dir = Path(args.output_dir)
    train_meta = write_token_bin(out_dir / "train.bin", train_ids, tokenizer.vocab_size)
    val_meta = write_token_bin(out_dir / "val.bin", val_ids, tokenizer.vocab_size)

    print(f"train.bin: {train_meta['count']:,} tokens ({train_meta['dtype']})")
    print(f"val.bin  : {val_meta['count']:,} tokens ({val_meta['dtype']})")
    if train_ids:
        ratio = len(train_text) / max(1, len(train_ids))
        print(f"compression: {ratio:.2f} characters per token")
    print(f"output directory: {out_dir}")


if __name__ == "__main__":
    main()
