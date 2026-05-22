"""Generate text from a trained LocalNeuro checkpoint.

Examples
--------
    python scripts/generate.py --prompt "the model learns"
    python scripts/generate.py --checkpoint checkpoints/run/best \\
        --prompt "2 + 2 =" --temperature 0.7 --max-new-tokens 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.checkpoint import load_checkpoint
from localneuro.inference.engine import InferenceEngine
from localneuro.sampling import SamplingConfig
from localneuro.utils import configure_console, human_count, resolve_device


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/run", help="checkpoint directory")
    parser.add_argument("--prompt", default="the model", help="text prompt to continue")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible sampling")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--no-mmap", action="store_true", help="load weights without memory mapping")
    args = parser.parse_args()

    device = resolve_device(args.device)
    loaded = load_checkpoint(
        args.checkpoint,
        build_model=True,
        map_location=str(device),
        mmap=not args.no_mmap,
    )
    if loaded.tokenizer is None:
        raise SystemExit(f"checkpoint {args.checkpoint} has no tokenizer.json")

    engine = InferenceEngine(loaded.model, loaded.tokenizer, device=device)
    sampling = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    params = human_count(loaded.model.num_parameters())
    step = loaded.meta.get("step", "?")
    print(f"[LocalNeuro {params} params | step {step} | device {device}]")
    print("-" * 60)
    print(args.prompt, end="", flush=True)
    engine.generate_text(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        sampling=sampling,
        seed=args.seed,
        on_token=lambda fragment: print(fragment, end="", flush=True),
    )
    print()


if __name__ == "__main__":
    main()
