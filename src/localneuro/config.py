"""Configuration objects for LocalNeuro.

Everything that describes *what* a model is (its shape) and *how* it should be
trained lives here as plain dataclasses with JSON serialization. Keeping these
free of any ``torch`` import means configs can be inspected, generated and
validated by lightweight tooling without loading the deep-learning stack.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


class _ConfigBase:
    """Mixin that gives a dataclass round-trip JSON serialization.

    ``from_dict`` deliberately ignores unknown keys so that human-friendly
    fields (``name``, ``description``) can live in the JSON files without
    breaking construction.
    """

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_ConfigBase":
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[call-arg]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "_ConfigBase":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class ModelConfig(_ConfigBase):
    """Describes the shape of a LocalNeuro transformer.

    The architecture is a decoder-only transformer with rotary position
    embeddings (RoPE), RMSNorm pre-normalization, SwiGLU feed-forward blocks
    and optional grouped-query attention (GQA). See ``docs/architecture.md``.
    """

    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    # Number of key/value heads. When < n_heads this enables grouped-query
    # attention, which shrinks the KV cache (the dominant inference memory
    # cost). ``None`` means "same as n_heads" (plain multi-head attention).
    n_kv_heads: Optional[int] = None
    # Hidden width of the SwiGLU feed-forward block. ``None`` -> derived from
    # d_model using the standard 8/3 ratio rounded up to a multiple of 64.
    d_ff: Optional[int] = None
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5
    # Tie the input embedding matrix and the output projection. This removes
    # ``vocab_size * d_model`` parameters and tends to help small models.
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even for rotary embeddings"
            )
        if self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size and max_seq_len must be positive")

    @property
    def head_dim(self) -> int:
        """Dimensionality of a single attention head."""
        return self.d_model // self.n_heads

    @property
    def ff_dim(self) -> int:
        """Resolved feed-forward hidden width."""
        if self.d_ff is not None:
            return self.d_ff
        hidden = int(8 * self.d_model / 3)
        return ((hidden + 63) // 64) * 64

    @property
    def n_rep(self) -> int:
        """How many query heads share each key/value head (GQA factor)."""
        assert self.n_kv_heads is not None
        return self.n_heads // self.n_kv_heads

    def num_params(self, include_embedding: bool = True) -> int:
        """Analytically estimate the parameter count without building the model."""
        assert self.n_kv_heads is not None
        d, ff = self.d_model, self.ff_dim
        kv = self.n_kv_heads * self.head_dim
        per_layer = (
            d * d            # query projection
            + d * kv         # key projection
            + d * kv         # value projection
            + d * d          # output projection
            + 3 * d * ff     # SwiGLU: gate, up, down
            + 2 * d          # two RMSNorm weights
        )
        total = self.n_layers * per_layer + d  # + final norm
        embedding = self.vocab_size * d
        if include_embedding:
            total += embedding
        if not self.tie_embeddings:
            total += embedding  # separate output projection
        return total


@dataclass
class TrainConfig(_ConfigBase):
    """Hyper-parameters and runtime options for a training run."""

    # --- data -------------------------------------------------------------
    train_bin: str = "datasets/demo/train.bin"
    val_bin: str = "datasets/demo/val.bin"

    # --- optimization -----------------------------------------------------
    batch_size: int = 16
    block_size: int = 256          # training sequence length (<= model.max_seq_len)
    grad_accum_steps: int = 1      # effective batch = batch_size * grad_accum_steps
    max_steps: int = 2000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0         # 0 disables gradient clipping

    # --- precision / memory ----------------------------------------------
    precision: str = "fp32"        # "fp32" | "bf16" | "fp16"
    grad_checkpoint: bool = False  # trade compute for activation memory
    dropout: float = 0.0           # overrides ModelConfig.dropout during training

    # --- runtime ----------------------------------------------------------
    device: str = "auto"           # "auto" | "cpu" | "cuda" | "mps"
    num_workers: int = 0           # DataLoader workers (0 is safest on Windows)
    cpu_threads: int = 0           # 0 keeps the torch default thread count
    seed: int = 1337
    compile_model: bool = False    # torch.compile (off by default for portability)

    # --- logging / checkpointing -----------------------------------------
    out_dir: str = "checkpoints/run"
    log_interval: int = 10
    eval_interval: int = 200
    eval_iters: int = 50
    checkpoint_interval: int = 500
    always_save_checkpoint: bool = False
    resume: bool = False

    def __post_init__(self) -> None:
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"unknown precision: {self.precision!r}")
        if self.batch_size <= 0 or self.block_size <= 0:
            raise ValueError("batch_size and block_size must be positive")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")


def load_model_config(path: str | Path) -> ModelConfig:
    """Load a :class:`ModelConfig` from a JSON preset file."""
    return ModelConfig.load(path)  # type: ignore[return-value]


def load_train_config(path: str | Path) -> TrainConfig:
    """Load a :class:`TrainConfig` from a JSON file."""
    return TrainConfig.load(path)  # type: ignore[return-value]
