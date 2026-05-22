"""Quantize a trained checkpoint to int8 or int4 for low-memory inference.

Examples
--------
    python scripts/quantize.py --input checkpoints/run --output checkpoints/run-int8
    python scripts/quantize.py --input checkpoints/run --output checkpoints/run-int4 --bits 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.checkpoint import load_checkpoint, save_checkpoint
from localneuro.quantization.quantize import quantize_model
from localneuro.utils import human_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="checkpoints/run", help="source checkpoint directory")
    parser.add_argument("--output", required=True, help="destination checkpoint directory")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8)
    args = parser.parse_args()

    weights_path = Path(args.input) / "weights.lnw"
    if not weights_path.exists():
        raise SystemExit(f"no checkpoint found at {args.input}")
    size_before = weights_path.stat().st_size

    print(f"loading checkpoint: {args.input}")
    loaded = load_checkpoint(args.input, build_model=True, map_location="cpu")

    print(f"quantizing all linear layers to int{args.bits} ...")
    layers = quantize_model(loaded.model, bits=args.bits)

    meta = dict(loaded.meta)
    meta["quantization"] = args.bits
    save_checkpoint(
        args.output,
        loaded.model,
        loaded.model_config,
        tokenizer=loaded.tokenizer,
        meta=meta,
    )

    size_after = (Path(args.output) / "weights.lnw").stat().st_size
    ratio = size_before / max(size_after, 1)
    print(f"quantized {layers} linear layers")
    print(f"weights: {human_bytes(size_before)} -> {human_bytes(size_after)} "
          f"({ratio:.2f}x smaller)")
    print(f"saved quantized checkpoint -> {args.output}")
    print("Note: the token embedding stays in float32, so the ratio is below the "
          "theoretical 4x (int8) / 8x (int4).")


if __name__ == "__main__":
    main()
