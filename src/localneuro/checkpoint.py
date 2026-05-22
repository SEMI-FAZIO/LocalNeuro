"""Checkpointing and a pickle-free, memory-mappable weight format.

LocalNeuro stores model weights in its own ``.lnw`` container instead of using
``torch.save`` (which is pickle-based). The format is:

    bytes  0..3    magic  b"LNW1"
    bytes  4..11   header length (uint64, little-endian)
    bytes  12..    header (UTF-8 JSON): per-tensor dtype / shape / offset
    then           raw tensor data, tightly packed

Advantages:

* **Safe to share.** No pickle means loading a downloaded model cannot execute
  arbitrary code.
* **Memory-mappable.** Tensors can be read as views into the file, so loading a
  model never needs more RAM than the model itself.
* **Transparent.** The header is plain JSON; anyone can inspect it.

A *checkpoint* is a directory bundle containing the weights, the model config,
the tokenizer and (for resumable training) the optimizer state.
"""

from __future__ import annotations

import json
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from .config import ModelConfig
from .model import LocalNeuroLM
from .quantization.quantize import apply_quantized_skeleton
from .tokenizer.bpe import BPETokenizer

_MAGIC = b"LNW1"

# torch dtype -> numpy dtype. bfloat16 is intentionally absent: it has no numpy
# equivalent and is up-cast to float32 on save.
_TORCH_TO_NUMPY: Dict[torch.dtype, type] = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

PathLike = Union[str, Path]


# --------------------------------------------------------------------------
# Raw weight container (.lnw)
# --------------------------------------------------------------------------
def save_weights(path: PathLike, state_dict: Dict[str, torch.Tensor]) -> None:
    """Write a state dict to a ``.lnw`` file.

    Performed in two passes (compute the layout, then stream the data) so peak
    extra memory is only a single tensor, not a copy of the whole model.
    """
    # Pass 1: compute the on-disk layout.
    tensors_meta: Dict[str, Dict[str, object]] = {}
    offset = 0
    for name, tensor in state_dict.items():
        export_dtype = (
            torch.float32 if tensor.dtype == torch.bfloat16 else tensor.dtype
        )
        if export_dtype not in _TORCH_TO_NUMPY:
            raise TypeError(f"unsupported tensor dtype for '{name}': {tensor.dtype}")
        np_dtype = np.dtype(_TORCH_TO_NUMPY[export_dtype])
        nbytes = tensor.numel() * np_dtype.itemsize
        tensors_meta[name] = {
            "dtype": np_dtype.name,
            "shape": list(tensor.shape),
            "offset": offset,
            "nbytes": nbytes,
        }
        offset += nbytes

    header = json.dumps(
        {"format": "localneuro-weights", "version": 1, "tensors": tensors_meta}
    ).encode("utf-8")

    # Pass 2: write the file, one tensor at a time.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(_MAGIC)
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        for name, tensor in state_dict.items():
            cpu_tensor = tensor.detach().to("cpu").contiguous()
            if cpu_tensor.dtype == torch.bfloat16:
                cpu_tensor = cpu_tensor.float()
            fh.write(cpu_tensor.numpy().tobytes())


def _read_header(path: Path) -> tuple[dict, int]:
    """Return ``(header_dict, data_section_offset)`` for a ``.lnw`` file."""
    with path.open("rb") as fh:
        if fh.read(4) != _MAGIC:
            raise ValueError(f"{path} is not a LocalNeuro .lnw weight file")
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len).decode("utf-8"))
    return header, 4 + 8 + header_len


def load_weights(path: PathLike, mmap: bool = True) -> Dict[str, torch.Tensor]:
    """Load a ``.lnw`` file into a state dict.

    With ``mmap=True`` (default) each tensor is a lazy view into the file. The
    operating system pages weight data in on demand, so peak memory stays close
    to the model size even for models larger than RAM. With ``mmap=False`` the
    file is read fully into owned memory.
    """
    path = Path(path)
    header, data_start = _read_header(path)
    tensors_meta = header["tensors"]
    result: Dict[str, torch.Tensor] = {}

    if mmap:
        mapped = np.memmap(path, dtype=np.uint8, mode="r")
        for name, meta in tensors_meta.items():
            np_dtype = np.dtype(str(meta["dtype"]))
            count = int(meta["nbytes"]) // np_dtype.itemsize
            array = np.frombuffer(
                mapped, dtype=np_dtype, count=count,
                offset=data_start + int(meta["offset"]),
            ).reshape(meta["shape"])
            with warnings.catch_warnings():
                # The view is read-only by design; copy_ into the model later.
                warnings.simplefilter("ignore")
                result[name] = torch.from_numpy(array)
    else:
        blob = path.read_bytes()
        for name, meta in tensors_meta.items():
            np_dtype = np.dtype(str(meta["dtype"]))
            count = int(meta["nbytes"]) // np_dtype.itemsize
            array = np.frombuffer(
                blob, dtype=np_dtype, count=count,
                offset=data_start + int(meta["offset"]),
            ).reshape(meta["shape"])
            result[name] = torch.from_numpy(array.copy())
    return result


# --------------------------------------------------------------------------
# Checkpoint bundles (directories)
# --------------------------------------------------------------------------
@dataclass
class LoadedCheckpoint:
    """The result of :func:`load_checkpoint`."""

    model_config: ModelConfig
    model: Optional[LocalNeuroLM]
    tokenizer: Optional[BPETokenizer]
    trainer_state: Optional[dict]
    meta: dict


def save_checkpoint(
    out_dir: PathLike,
    model: LocalNeuroLM,
    model_config: ModelConfig,
    *,
    tokenizer: Optional[BPETokenizer] = None,
    trainer_state: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> Path:
    """Write a checkpoint bundle to ``out_dir``.

    The bundle contains ``config.json``, ``weights.lnw``, optionally
    ``tokenizer.json``, optionally ``trainer_state.pt`` (optimizer/step state
    for resuming) and ``meta.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_config.save(out_dir / "config.json")
    save_weights(out_dir / "weights.lnw", model.state_dict())
    if tokenizer is not None:
        tokenizer.save(out_dir / "tokenizer.json")
    if trainer_state is not None:
        # The optimizer state is nested and torch-specific; torch.save is the
        # pragmatic choice. This file is only ever read to resume your *own*
        # runs -- shareable artifacts use the pickle-free weights.lnw instead.
        torch.save(trainer_state, out_dir / "trainer_state.pt")
    (out_dir / "meta.json").write_text(
        json.dumps(meta or {}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_dir


def load_checkpoint(
    checkpoint_dir: PathLike,
    *,
    build_model: bool = True,
    map_location: str = "cpu",
    mmap: bool = True,
) -> LoadedCheckpoint:
    """Load a checkpoint bundle.

    Parameters
    ----------
    build_model:
        When true, instantiate the model and load its weights. When false only
        the config / tokenizer / metadata are read (useful for inspection).
    map_location:
        Device the model is moved to after loading.
    mmap:
        Memory-map the weight file (low-memory loading).
    """
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")

    model_config = ModelConfig.load(directory / "config.json")  # type: ignore[assignment]

    tokenizer = None
    tok_path = directory / "tokenizer.json"
    if tok_path.exists():
        tokenizer = BPETokenizer.load(tok_path)

    meta: dict = {}
    meta_path = directory / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    trainer_state = None
    state_path = directory / "trainer_state.pt"
    if state_path.exists():
        trainer_state = torch.load(
            state_path, map_location=map_location, weights_only=False
        )

    model = None
    if build_model:
        model = LocalNeuroLM(model_config)
        # A quantized checkpoint stores int8/int4 buffers instead of float
        # weights; the module tree must be converted before loading.
        quant_bits = meta.get("quantization")
        if quant_bits:
            apply_quantized_skeleton(model, int(quant_bits))
        weights = load_weights(directory / "weights.lnw", mmap=mmap)
        model.load_state_dict(weights)
        model.to(map_location)
        model.eval()

    return LoadedCheckpoint(
        model_config=model_config,
        model=model,
        tokenizer=tokenizer,
        trainer_state=trainer_state,
        meta=meta,
    )


def checkpoint_exists(checkpoint_dir: PathLike) -> bool:
    """True if ``checkpoint_dir`` looks like a complete checkpoint bundle."""
    directory = Path(checkpoint_dir)
    return (directory / "config.json").exists() and (
        directory / "weights.lnw"
    ).exists()
