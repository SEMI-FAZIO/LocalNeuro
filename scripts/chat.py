"""Interactive chat with a trained LocalNeuro checkpoint.

Example
-------
    python scripts/chat.py --checkpoint checkpoints/run

Type your message and press Enter. Commands: /reset clears the history,
/exit quits (Ctrl-D / Ctrl-C also quit).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.checkpoint import load_checkpoint
from localneuro.inference.chat import run_chat
from localneuro.inference.engine import InferenceEngine
from localneuro.sampling import SamplingConfig
from localneuro.utils import configure_console, resolve_device


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/run", help="checkpoint directory")
    parser.add_argument("--system", default=None, help="optional system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--no-mmap", action="store_true")
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
    run_chat(
        engine,
        system_prompt=args.system,
        sampling=sampling,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
