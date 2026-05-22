"""End-to-end LocalNeuro demo, runnable with no arguments.

It performs the entire pipeline in a single process:

    1. generate a synthetic corpus
    2. train a BPE tokenizer on it
    3. tokenize the data into binary token streams
    4. train a ~1M parameter model for a few hundred steps
    5. sample text from the trained model

On a typical laptop CPU this finishes in a couple of minutes and is the
fastest way to confirm that every part of the project works on your machine.

Run it from the repository root:

    python examples/end_to_end.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `localneuro` importable from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.config import ModelConfig, TrainConfig
from localneuro.data.dataset import encode_corpus, write_token_bin
from localneuro.data.preprocessing import read_corpus, split_train_val
from localneuro.data.synthetic import generate_synthetic_corpus
from localneuro.inference.engine import InferenceEngine
from localneuro.model import LocalNeuroLM
from localneuro.sampling import SamplingConfig
from localneuro.tokenizer.bpe import BPETokenizer
from localneuro.training.trainer import Trainer
from localneuro.utils import configure_console, set_seed

WORKDIR = Path("checkpoints/demo")


def main() -> None:
    configure_console()
    set_seed(1337)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Build a corpus: synthetic patterns plus the bundled natural-text demo.
    print("[1/5] building the training corpus ...")
    corpus = generate_synthetic_corpus(num_samples=8000, seed=1)
    demo_file = Path("datasets/demo/tiny_corpus.txt")
    if demo_file.exists():
        corpus = corpus + "\n\n" + read_corpus(demo_file)
    print(f"      corpus: {len(corpus):,} characters")

    # 2. Train the tokenizer.
    print("[2/5] training the BPE tokenizer ...")
    tokenizer = BPETokenizer.train(corpus, vocab_size=2048)
    tokenizer.save(WORKDIR / "tokenizer.json")
    print(f"      vocabulary: {tokenizer.vocab_size} tokens")

    # 3. Tokenize into train / validation binary streams.
    print("[3/5] tokenizing the dataset ...")
    train_text, val_text = split_train_val(corpus, val_fraction=0.1)
    train_ids = encode_corpus(train_text, tokenizer, doc_separator="\n")
    val_ids = encode_corpus(val_text, tokenizer, doc_separator="\n")
    write_token_bin(WORKDIR / "train.bin", train_ids, tokenizer.vocab_size)
    write_token_bin(WORKDIR / "val.bin", val_ids, tokenizer.vocab_size)
    print(f"      train: {len(train_ids):,} tokens   val: {len(val_ids):,} tokens")

    # 4. Build and train a tiny model.
    print("[4/5] training a tiny model ...")
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=384,
        max_seq_len=256,
    )
    model = LocalNeuroLM(model_config)
    print(model.summary())
    train_config = TrainConfig(
        train_bin=str(WORKDIR / "train.bin"),
        val_bin=str(WORKDIR / "val.bin"),
        out_dir=str(WORKDIR),
        batch_size=16,
        block_size=128,
        max_steps=600,
        learning_rate=6e-4,
        warmup_steps=60,
        log_interval=50,
        eval_interval=200,
        eval_iters=20,
        checkpoint_interval=600,
    )
    trainer = Trainer(model, model_config, train_config, tokenizer=tokenizer)
    trainer.train()

    # 5. Sample from the trained model.
    print("[5/5] sampling from the trained model ...")
    engine = InferenceEngine(model, tokenizer, device="cpu")
    sampling = SamplingConfig(
        temperature=0.7, top_k=30, top_p=0.9, repetition_penalty=1.2
    )
    prompts = ["count: one two three", "12 + 7 =", "the ", "question: what color"]
    for prompt in prompts:
        completion = engine.generate_text(
            prompt, max_new_tokens=40, sampling=sampling, seed=0
        )
        print(f"      {prompt!r}\n        -> {completion!r}")

    elapsed = time.time() - started
    print(f"\ndone in {elapsed:.1f}s. Checkpoint saved to: {WORKDIR}")
    print(f"Next:  python scripts/chat.py --checkpoint {WORKDIR}")


if __name__ == "__main__":
    main()
