"""The LocalNeuro language model -- a compact decoder-only transformer.

Data flow for a forward pass::

    token ids
      -> token embedding            (vocab_size x d_model lookup table)
      -> N x TransformerBlock       (RMSNorm -> attention -> RMSNorm -> SwiGLU)
      -> final RMSNorm
      -> output projection          (tied with the embedding by default)
      -> logits over the vocabulary

The same ``forward`` serves both training (full sequence, returns a loss) and
incremental decoding (one token at a time, backed by a :class:`KVCache`).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .kv_cache import KVCache
from .modules import RMSNorm, RotaryEmbedding, TransformerBlock
from .utils import human_count


class LocalNeuroLM(nn.Module):
    """A decoder-only transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

        # Weight tying: reuse the embedding matrix as the output projection.
        # ``lm_head is None`` is the signal that tying is active.
        if config.tie_embeddings:
            self.lm_head: Optional[nn.Linear] = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._grad_checkpoint = False

        self.apply(self._init_weights)
        self._scaled_residual_init()

    # ------------------------------------------------------------------ init
    def _init_weights(self, module: nn.Module) -> None:
        """Standard small-transformer initialization."""
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scaled_residual_init(self) -> None:
        """Down-scale the projections that write into the residual stream.

        Every layer adds its attention and feed-forward output to the residual
        stream. Without correction the stream variance grows with depth. Scaling
        these two projections by ``1/sqrt(2 * n_layers)`` (the GPT-2 recipe)
        keeps activations well-conditioned at initialization.
        """
        scale = self.config.init_std / math.sqrt(2 * self.config.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=scale)

    # --------------------------------------------------------------- forward
    def _project_to_vocab(self, x: torch.Tensor) -> torch.Tensor:
        """Map hidden states to vocabulary logits (honouring weight tying)."""
        if self.lm_head is None:
            return F.linear(x, self.token_embedding.weight)
        return self.lm_head(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the model.

        Parameters
        ----------
        idx:
            Long tensor of token ids, shape ``[batch, seq]``.
        targets:
            Optional next-token targets, shape ``[batch, seq]``. When given the
            method returns *all* logits plus the cross-entropy loss (training
            mode). When omitted only the final position's logits are computed
            (the generation fast-path -- it avoids a full vocab projection over
            the whole prompt).
        kv_cache:
            Optional cache for incremental decoding.
        start_pos:
            Absolute position of the first token in ``idx`` -- needed so rotary
            embeddings and the causal mask line up with the cached prefix.

        Returns
        -------
        ``(logits, loss)``. ``loss`` is ``None`` unless ``targets`` was given.
        """
        _, seq = idx.shape
        if start_pos + seq > self.config.max_seq_len:
            raise ValueError(
                f"sequence positions up to {start_pos + seq} exceed the model's "
                f"max_seq_len ({self.config.max_seq_len})"
            )

        x = self.embed_dropout(self.token_embedding(idx))
        cos, sin = self.rotary(start_pos, seq)

        # Gradient checkpointing only makes sense while training without a cache.
        use_checkpoint = self._grad_checkpoint and self.training and kv_cache is None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            if use_checkpoint:
                x = checkpoint(
                    block, x, cos, sin, layer_cache, start_pos, use_reentrant=False
                )
            else:
                x = block(x, cos, sin, layer_cache, start_pos)

        x = self.final_norm(x)

        if targets is not None:
            logits = self._project_to_vocab(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss

        # Generation fast-path: only the last token's logits are needed.
        logits = self._project_to_vocab(x[:, -1:, :])
        return logits, None

    # ----------------------------------------------------------- utilities
    def make_kv_cache(
        self,
        batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> KVCache:
        """Allocate a :class:`KVCache` sized for this model."""
        cfg = self.config
        ref = self.token_embedding.weight
        return KVCache(
            n_layers=cfg.n_layers,
            batch_size=batch_size,
            n_kv_heads=cfg.n_kv_heads or cfg.n_heads,
            max_seq_len=max_seq_len or cfg.max_seq_len,
            head_dim=cfg.head_dim,
            dtype=dtype or ref.dtype,
            device=device or ref.device,
        )

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Toggle activation checkpointing (saves memory, costs ~30% compute)."""
        self._grad_checkpoint = enabled

    def num_parameters(self, include_embedding: bool = True) -> int:
        """Total parameter count.

        ``include_embedding=False`` reports the "transformer core" size, the
        figure usually quoted when comparing model capacity, since the
        embedding table scales with vocabulary rather than depth/width.
        """
        total = sum(p.numel() for p in self.parameters())
        # Quantized layers keep their weights as buffers rather than trainable
        # parameters; count the values they represent (duck-typed, so model.py
        # need not import the quantization package) for a figure that stays
        # stable whether or not the model has been quantized.
        for module in self.modules():
            weight_numel = getattr(module, "weight_numel", None)
            if callable(weight_numel):
                total += weight_numel()
        if not include_embedding:
            total -= self.token_embedding.weight.numel()
            if self.lm_head is not None:
                total -= self.lm_head.weight.numel()
        return total

    @property
    def device(self) -> torch.device:
        return self.token_embedding.weight.device

    def summary(self) -> str:
        """Return a human-readable description of the model."""
        cfg = self.config
        lines = [
            "LocalNeuroLM",
            f"  parameters     : {human_count(self.num_parameters())} "
            f"({self.num_parameters():,})",
            f"  core (no embed): {human_count(self.num_parameters(False))}",
            f"  d_model        : {cfg.d_model}",
            f"  layers         : {cfg.n_layers}",
            f"  heads          : {cfg.n_heads} query / {cfg.n_kv_heads} kv",
            f"  head_dim       : {cfg.head_dim}",
            f"  ff_dim         : {cfg.ff_dim}",
            f"  vocab_size     : {cfg.vocab_size}",
            f"  max_seq_len    : {cfg.max_seq_len}",
            f"  tied embeddings: {cfg.tie_embeddings}",
        ]
        return "\n".join(lines)


def build_model(config: ModelConfig) -> LocalNeuroLM:
    """Construct a model from a config (a stable public factory function)."""
    return LocalNeuroLM(config)
