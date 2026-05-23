"""End-to-end instruction fine-tuning demo.

This script shows the whole point of instruction tuning. It:

    1. pretrains a tiny base model on plain text,
    2. asks it a few questions -- the base model just rambles,
    3. instruction-fine-tunes it on synthetic question/answer pairs,
    4. asks the same questions again -- now it answers.

It runs on a CPU in several minutes. Run it from the repository root:

    python examples/instruction_tuning.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.config import ModelConfig, TrainConfig
from localneuro.data.dataset import encode_corpus, write_token_bin
from localneuro.data.preprocessing import read_corpus, split_train_val
from localneuro.data.sft import build_sft_dataloader
from localneuro.data.synthetic import (
    generate_synthetic_corpus,
    generate_synthetic_instructions,
)
from localneuro.inference.chat import ChatSession
from localneuro.inference.engine import InferenceEngine
from localneuro.model import LocalNeuroLM
from localneuro.sampling import SamplingConfig
from localneuro.tokenizer.bpe import BPETokenizer
from localneuro.training.trainer import Trainer
from localneuro.utils import configure_console, set_seed

WORKDIR = Path("checkpoints/sft-demo")
QUESTIONS = [
    "What is 5 plus 3?",
    "What color is the river?",
    "Count from 2 to 6.",
    "Hello.",
]


def ask(model: LocalNeuroLM, tokenizer: BPETokenizer) -> list[tuple[str, str]]:
    """Put each question to the model through the chat format."""
    engine = InferenceEngine(model, tokenizer, device="cpu")
    # Greedy decoding: show the model's most confident answer, no sampling noise.
    sampling = SamplingConfig(temperature=0.0, repetition_penalty=1.2)
    results = []
    for question in QUESTIONS:
        session = ChatSession(engine, sampling=sampling, max_new_tokens=32)
        results.append((question, session.send(question)))
    return results


def main() -> None:
    configure_console()
    set_seed(1337)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Corpus and tokenizer ---------------------------------------------
    print("[1/5] building corpus and tokenizer ...")
    corpus = generate_synthetic_corpus(num_samples=7000, seed=1)
    demo = Path("datasets/demo/tiny_corpus.txt")
    if demo.exists():
        corpus = corpus + "\n\n" + read_corpus(demo)
    tokenizer = BPETokenizer.train(corpus, vocab_size=2048)
    tokenizer.save(WORKDIR / "tokenizer.json")
    print(f"      vocabulary: {tokenizer.vocab_size} tokens")

    # 2. Pretrain the base model ------------------------------------------
    print("[2/5] pretraining the base model ...")
    train_text, val_text = split_train_val(corpus, val_fraction=0.1)
    write_token_bin(
        WORKDIR / "train.bin",
        encode_corpus(train_text, tokenizer, doc_separator="\n"),
        tokenizer.vocab_size,
    )
    write_token_bin(
        WORKDIR / "val.bin",
        encode_corpus(val_text, tokenizer, doc_separator="\n"),
        tokenizer.vocab_size,
    )
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size, d_model=160, n_layers=4,
        n_heads=5, d_ff=448, max_seq_len=256,
    )
    model = LocalNeuroLM(model_config)
    print(model.summary())
    base_config = TrainConfig(
        train_bin=str(WORKDIR / "train.bin"),
        val_bin=str(WORKDIR / "val.bin"),
        out_dir=str(WORKDIR / "base"),
        batch_size=24, block_size=128, max_steps=550,
        learning_rate=6e-4, warmup_steps=60,
        log_interval=150, eval_interval=300, eval_iters=20,
        checkpoint_interval=550,
    )
    Trainer(model, model_config, base_config, tokenizer=tokenizer).train()

    # 3. Ask the base model (it cannot really answer) ---------------------
    print("\n[3/5] BASE model answers (before instruction tuning):")
    before = ask(model, tokenizer)
    for question, answer in before:
        print(f"      Q: {question}\n        A: {answer!r}")

    # 4. Instruction fine-tuning ------------------------------------------
    print("\n[4/5] instruction fine-tuning the same model ...")
    instructions = generate_synthetic_instructions(num_samples=4000, seed=2)
    val_count = 200
    block_size = 160
    train_loader = build_sft_dataloader(
        instructions[val_count:], tokenizer, block_size, 24, shuffle=True
    )
    val_loader = build_sft_dataloader(
        instructions[:val_count], tokenizer, block_size, 24, shuffle=False
    )
    sentinel = str(WORKDIR / "_unused")
    sft_config = TrainConfig(
        train_bin=sentinel, val_bin=sentinel, out_dir=str(WORKDIR / "sft"),
        batch_size=24, block_size=block_size, max_steps=500,
        learning_rate=5e-5, min_learning_rate=5e-6, warmup_steps=50,
        weight_decay=0.0, log_interval=150, eval_interval=180, eval_iters=20,
        checkpoint_interval=500,
    )
    Trainer(
        model, model_config, sft_config, tokenizer=tokenizer,
        train_loader=train_loader, val_loader=val_loader,
        extra_meta={"stage": "sft"},
    ).train()

    # 5. Ask the fine-tuned model -----------------------------------------
    print("\n[5/5] FINE-TUNED model answers (after instruction tuning):")
    after = ask(model, tokenizer)
    for question, answer in after:
        print(f"      Q: {question}\n        A: {answer!r}")

    print(f"\ndone in {time.time() - started:.1f}s.")
    print("The base model cannot use the chat format -- it never trained on it.")
    print("After instruction tuning the model replies in the proper answer form.")
    print("Answer *accuracy* scales with model size and training: for genuinely")
    print("good answers, fine-tune a base-40m model (see the README).")
    print(f"\nfine-tuned checkpoint saved to: {WORKDIR / 'sft'}")
    print(f"try:  python scripts/chat.py --checkpoint {WORKDIR / 'sft'}")


if __name__ == "__main__":
    main()
