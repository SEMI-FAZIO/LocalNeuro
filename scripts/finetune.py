"""Instruction fine-tuning (SFT) for a trained LocalNeuro checkpoint.

Pretraining (scripts/train.py) teaches the model to continue text. This second
stage teaches it to *answer*: it trains on instruction/response pairs laid out
in the chat format, with the loss applied only to the response tokens.

Examples
--------
    # fine-tune on generated synthetic instructions
    python scripts/finetune.py --checkpoint checkpoints/run --out-dir checkpoints/run-sft

    # fine-tune on your own instruction data (JSONL: {"instruction":..,"response":..})
    python scripts/finetune.py --checkpoint checkpoints/run --data my_instructions.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.checkpoint import checkpoint_exists, load_checkpoint
from localneuro.config import TrainConfig
from localneuro.data.sft import build_sft_dataloader, read_instructions_jsonl, write_instructions_jsonl
from localneuro.data.synthetic import generate_synthetic_instructions
from localneuro.training.trainer import Trainer
from localneuro.utils import configure_console, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="checkpoints/run",
                        help="base (pretrained) checkpoint to fine-tune")
    parser.add_argument("--out-dir", default="checkpoints/run-sft",
                        help="where to write the fine-tuned checkpoint")
    parser.add_argument("--data", nargs="*", default=[],
                        help="instruction JSONL file(s): {instruction, response[, system]}")
    parser.add_argument("--synthetic-instructions", type=int, default=4000,
                        help="generate this many synthetic instruction pairs (0 to disable)")
    parser.add_argument("--save-data", default=None,
                        help="optional path to dump the instruction data as JSONL")

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5, help="fine-tuning learning rate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256,
                        help="sequence length (capped to the model's max_seq_len)")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main() -> None:
    configure_console()
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    if not checkpoint_exists(args.checkpoint):
        raise SystemExit(
            f"base checkpoint not found: {args.checkpoint}\n"
            "Pretrain one first with scripts/train.py."
        )

    print(f"loading base checkpoint: {args.checkpoint}")
    bundle = load_checkpoint(args.checkpoint, build_model=True, map_location="cpu")
    if bundle.tokenizer is None:
        raise SystemExit("base checkpoint has no tokenizer.json")
    if bundle.meta.get("quantization"):
        raise SystemExit(
            "cannot fine-tune a quantized checkpoint (its weights are frozen "
            "integer buffers). Fine-tune the float checkpoint, then quantize."
        )
    tokenizer = bundle.tokenizer
    model_config = bundle.model_config

    # --- gather instruction data -----------------------------------------
    examples = []
    for data_path in args.data:
        loaded = read_instructions_jsonl(data_path)
        examples.extend(loaded)
        print(f"loaded {len(loaded)} instructions from {data_path}")
    if args.synthetic_instructions > 0:
        synthetic = generate_synthetic_instructions(
            args.synthetic_instructions, seed=args.seed
        )
        examples.extend(synthetic)
        print(f"generated {len(synthetic)} synthetic instructions")
    if not examples:
        raise SystemExit(
            "no instruction data -- pass --data and/or --synthetic-instructions"
        )
    if args.save_data:
        write_instructions_jsonl(args.save_data, examples)
        print(f"wrote instruction data -> {args.save_data}")

    # --- train / val split ------------------------------------------------
    random.Random(args.seed).shuffle(examples)
    block_size = min(args.block_size, model_config.max_seq_len)
    val_count = (
        max(8, int(len(examples) * args.val_fraction))
        if len(examples) >= 32 else 0
    )
    val_examples = examples[:val_count]
    train_examples = examples[val_count:]

    train_loader = build_sft_dataloader(
        train_examples, tokenizer, block_size, args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = None
    if val_examples:
        val_loader = build_sft_dataloader(
            val_examples, tokenizer, block_size, args.batch_size,
            shuffle=False, num_workers=args.num_workers,
        )

    dataset = train_loader.dataset
    print(f"SFT dataset: {len(dataset)} train / "
          f"{len(val_loader.dataset) if val_loader else 0} val examples "
          f"(block_size {block_size})")
    if dataset.skipped:
        print(f"  ({dataset.skipped} examples skipped: empty or too long)")

    # --- fine-tune --------------------------------------------------------
    sentinel = str(Path(args.out_dir) / "_unused")  # no .bin fallback for SFT
    train_config = TrainConfig(
        train_bin=sentinel,
        val_bin=sentinel,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        block_size=block_size,
        max_steps=args.steps,
        learning_rate=args.lr,
        min_learning_rate=args.lr * 0.1,
        warmup_steps=max(1, args.steps // 20),
        weight_decay=0.0,
        precision=args.precision,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        log_interval=max(1, args.steps // 60),
        eval_interval=max(1, args.steps // 5),
        eval_iters=20,
        checkpoint_interval=max(args.steps, 1),
    )

    trainer = Trainer(
        bundle.model,
        model_config,
        train_config,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        extra_meta={"stage": "sft", "base_checkpoint": str(args.checkpoint)},
    )
    trainer.train()

    print(f"fine-tuned checkpoint saved to: {args.out_dir}")
    print(f"try:  python scripts/chat.py --checkpoint {args.out_dir}")


if __name__ == "__main__":
    main()
