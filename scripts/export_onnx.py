"""Export a trained checkpoint to ONNX (optional).

The exported graph maps ``input_ids`` -> next-token ``logits`` using the plain
(no KV cache) forward pass. It is handy for running LocalNeuro inside other
runtimes; KV-cache decoding is not part of the exported graph.

Requires the optional dependency:  pip install onnx onnxruntime

Example
-------
    python scripts/export_onnx.py --checkpoint checkpoints/run --output localneuro.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn

from localneuro.checkpoint import load_checkpoint


class _ExportWrapper(nn.Module):
    """Adapts the model's ``(logits, loss)`` return to a single ONNX output."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)  # targets=None -> last-token logits
        return logits


def main() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise SystemExit(
            "ONNX export requires extra packages:\n"
            "    pip install onnx onnxruntime"
        )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/run")
    parser.add_argument("--output", default="localneuro.onnx")
    parser.add_argument("--seq-len", type=int, default=64, help="example sequence length for tracing")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    loaded = load_checkpoint(args.checkpoint, build_model=True, map_location="cpu")
    model = loaded.model
    model.eval()
    model.enable_gradient_checkpointing(False)

    if args.seq_len > model.config.max_seq_len:
        raise SystemExit(
            f"--seq-len {args.seq_len} exceeds the model's max_seq_len "
            f"({model.config.max_seq_len})"
        )

    wrapper = _ExportWrapper(model).eval()
    dummy = torch.zeros(1, args.seq_len, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy,),
        args.output,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
    )
    print(f"exported ONNX graph -> {args.output}")

    # Optional smoke test if onnxruntime is available.
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("install onnxruntime to verify the exported graph")
        return
    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input_ids": dummy.numpy().astype(np.int64)})
    print(f"onnxruntime check ok: logits shape = {outputs[0].shape}")


if __name__ == "__main__":
    main()
