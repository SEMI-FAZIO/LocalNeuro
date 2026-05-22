"""Train a LocalNeuro language model.

Examples
--------
    # quick CPU run on the synthetic + demo data
    python scripts/train.py --config configs/tiny-7m.json --steps 1500 --block-size 128

    # resume an interrupted run
    python scripts/train.py --out-dir checkpoints/run --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.checkpoint import checkpoint_exists, load_checkpoint
from localneuro.config import ModelConfig, TrainConfig
from localneuro.data.dataset import load_token_meta
from localneuro.model import LocalNeuroLM
from localneuro.tokenizer.bpe import BPETokenizer
from localneuro.training.trainer import Trainer
from localneuro.utils import set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/tiny-7m.json", help="model config JSON")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    parser.add_argument("--train-bin", default="datasets/demo/train.bin")
    parser.add_argument("--val-bin", default="datasets/demo/val.bin")
    parser.add_argument("--out-dir", default="checkpoints/run")

    parser.add_argument("--steps", type=int, default=2000, help="number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256, help="training sequence length")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="trade compute for lower activation memory")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 = torch default")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="resume from --out-dir")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    train_cfg = TrainConfig(
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.steps,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        precision=args.precision,
        grad_checkpoint=args.grad_checkpoint,
        dropout=args.dropout,
        device=args.device,
        num_workers=args.num_workers,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
        compile_model=args.compile,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )

    resume_state = None
    if args.resume and checkpoint_exists(args.out_dir):
        print(f"resuming from checkpoint: {args.out_dir}")
        loaded = load_checkpoint(args.out_dir, build_model=True, map_location="cpu")
        model = loaded.model
        model_cfg = loaded.model_config
        tokenizer = loaded.tokenizer
        resume_state = loaded.trainer_state
        if tokenizer is None:
            tokenizer = BPETokenizer.load(args.tokenizer)
    else:
        if not Path(args.tokenizer).exists():
            raise SystemExit(
                f"tokenizer not found: {args.tokenizer}\n"
                "Run scripts/train_tokenizer.py first."
            )
        if not Path(str(args.train_bin) + ".meta.json").exists():
            raise SystemExit(
                f"token data not found: {args.train_bin}\n"
                "Run scripts/prepare_data.py first."
            )
        tokenizer = BPETokenizer.load(args.tokenizer)
        model_cfg = ModelConfig.load(args.config)  # type: ignore[assignment]
        # The vocabulary the data was tokenized with is authoritative.
        data_vocab = int(load_token_meta(args.train_bin)["vocab_size"])
        if data_vocab != model_cfg.vocab_size:
            print(f"adjusting model vocab_size {model_cfg.vocab_size} -> {data_vocab} "
                  "to match the tokenized data")
            model_cfg.vocab_size = data_vocab
        model_cfg.dropout = train_cfg.dropout
        model = LocalNeuroLM(model_cfg)

    trainer = Trainer(model, model_cfg, train_cfg, tokenizer=tokenizer)
    if resume_state is not None:
        trainer.load_trainer_state(resume_state)
    trainer.train()


if __name__ == "__main__":
    main()
