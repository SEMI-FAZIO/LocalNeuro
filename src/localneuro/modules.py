"""Neural network building blocks for the LocalNeuro transformer.

Every layer here is implemented explicitly (no ``transformers`` / no fused
black-box kernels) so the data flow is easy to read and to teach:

* :class:`RMSNorm`        -- cheap, bias-free normalization.
* :class:`RotaryEmbedding`-- relative position information injected into Q/K.
* :class:`CausalSelfAttention` -- multi-head / grouped-query causal attention
                                  with an optional KV cache for fast decoding.
* :class:`SwiGLU`         -- gated feed-forward network.
* :class:`TransformerBlock` -- one pre-norm residual block.

The attention score computation is written out with an explicit softmax rather
than calling a fused primitive, because clarity is a stated goal of this
project and the models are small enough that the manual path is fast on CPU.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .kv_cache import LayerKVCache

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "repeat_kv",
    "CausalSelfAttention",
    "SwiGLU",
    "TransformerBlock",
]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Compared with standard LayerNorm this drops the mean-subtraction and the
    bias term, so it needs one fewer reduction and half the parameters while
    performing just as well for language models. The normalization statistic
    is always computed in float32 for numerical stability, even when the rest
    of the network runs in bfloat16.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normed.type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate the two halves of the last dimension (the RoPE trick)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate ``x`` by the precomputed angles.

    ``x``    -- ``[batch, n_heads, seq, head_dim]``
    ``cos``/``sin`` -- ``[seq, head_dim]``

    The rotation is done in float32 and cast back, so position information
    stays precise regardless of the model's compute dtype.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_fp32 = x.float()
    rotated = (x_fp32 * cos) + (_rotate_half(x_fp32) * sin)
    return rotated.type_as(x)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Instead of *adding* a position vector to the token embedding, RoPE *rotates*
    the query and key vectors by an angle proportional to their absolute
    position. The dot-product of two rotated vectors then depends only on their
    *relative* distance -- which is exactly what attention needs, and it
    extrapolates to sequence lengths beyond those seen in training far better
    than learned absolute embeddings.

    The ``cos``/``sin`` tables are precomputed once for ``max_seq_len`` and
    stored as non-persistent buffers (they are pure functions of the config and
    therefore never need to be saved to disk).
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        # Frequencies: lower dimensions rotate fast, higher dimensions slow.
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)   # [max_seq_len, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, start_pos: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the ``cos``/``sin`` slices for absolute positions
        ``[start_pos, start_pos + seq_len)``."""
        end = start_pos + seq_len
        if end > self.max_seq_len:
            raise ValueError(
                f"requested positions up to {end} exceed RoPE table "
                f"size {self.max_seq_len}"
            )
        return self.cos_cached[start_pos:end], self.sin_cached[start_pos:end]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads to match the number of query heads (GQA).

    With grouped-query attention several query heads share one key/value head.
    This repeats each KV head ``n_rep`` times so the subsequent matmul lines up,
    without ever materialising the repeats in the cache itself.
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq, head_dim)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary embeddings and GQA.

    Supports two modes through a single code path:

    * **Training / full forward** -- ``cache=None``; a square causal mask.
    * **Incremental decoding** -- a :class:`LayerKVCache` is supplied; only the
      new tokens are projected, keys/values for the prefix come from the cache.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_rep
        self.scale = 1.0 / math.sqrt(self.head_dim)

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.d_model, q_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[LayerKVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        # 1. Project to queries, keys and values, then split into heads.
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, n_heads, seq, head_dim]
        k = k.transpose(1, 2)  # [batch, n_kv_heads, seq, head_dim]
        v = v.transpose(1, 2)

        # 2. Inject position information via rotary embeddings.
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 3. Extend the cache (decoding) so keys/values cover the whole prefix.
        if cache is not None:
            if cache.length != start_pos:
                raise RuntimeError(
                    f"cache length {cache.length} disagrees with start_pos {start_pos}"
                )
            k, v = cache.append(k, v)
        kv_len = k.shape[2]

        # 4. Expand KV heads for grouped-query attention.
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 5. Scaled dot-product attention with a causal mask.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = self._causal_mask(seq, kv_len, start_pos, x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        # Softmax in float32 keeps the probabilities stable under bf16/fp16.
        weights = torch.softmax(scores.float(), dim=-1).type_as(q)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)  # [batch, n_heads, seq, head_dim]

        # 6. Merge heads and project back to the model dimension.
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.n_heads * self.head_dim)
        return self.resid_dropout(self.o_proj(out))

    @staticmethod
    def _causal_mask(
        seq: int, kv_len: int, start_pos: int, device: torch.device
    ) -> torch.Tensor:
        """Boolean mask ``[seq, kv_len]``; ``True`` marks positions to hide.

        Query token ``i`` lives at absolute position ``start_pos + i`` and may
        attend to any key whose absolute position is ``<=`` its own. This single
        formula covers both the square training mask and the wide decode mask
        (one query row, many cached keys).
        """
        q_pos = torch.arange(start_pos, start_pos + seq, device=device).unsqueeze(1)
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        return k_pos > q_pos


class SwiGLU(nn.Module):
    """Gated feed-forward network (the SwiGLU variant).

    A plain MLP applies one non-linearity: ``down(act(up(x)))``. SwiGLU instead
    *gates* one linear projection with a SiLU-activated sibling projection:
    ``down(silu(gate(x)) * up(x))``. The multiplicative gate lets the network
    suppress or amplify features per-channel and consistently improves quality
    per parameter, which matters a lot when the parameter budget is tiny.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        ff = config.ff_dim
        self.gate_proj = nn.Linear(config.d_model, ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, ff, bias=False)
        self.down_proj = nn.Linear(ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(hidden))


class TransformerBlock(nn.Module):
    """One pre-norm transformer block: attention then feed-forward.

    Pre-norm means normalization is applied *inside* the residual branch
    (``x + sublayer(norm(x))``). This keeps a clean identity path from input to
    output, which makes deep stacks train stably without learning-rate warmup
    tricks beyond the standard schedule.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[LayerKVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x
