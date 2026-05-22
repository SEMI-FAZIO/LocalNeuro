"""Key/value cache for autoregressive inference.

During generation every new token attends to *all* previous tokens. Without a
cache that means recomputing the keys and values for the whole prefix on every
step -- O(n^2) work. The cache stores already-computed keys/values so each new
step is O(n). It is the single most important inference optimization, and on a
small model it is also the dominant memory consumer, which is why LocalNeuro
supports grouped-query attention to shrink it.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


class LayerKVCache:
    """Pre-allocated key/value buffers for a single attention layer."""

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.max_seq_len = max_seq_len
        shape = (batch_size, n_kv_heads, max_seq_len, head_dim)
        # Buffers are allocated once up-front so that generation never triggers
        # a reallocation (which would fragment memory on long runs).
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0

    def append(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write ``k``/``v`` for the new tokens and return the full prefix.

        ``k`` and ``v`` have shape ``[batch, n_kv_heads, new_tokens, head_dim]``.
        The returned tensors cover positions ``0 .. length`` (inclusive of the
        tokens just written).
        """
        new_tokens = k.shape[2]
        end = self.length + new_tokens
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: tried to store {end} tokens "
                f"but capacity is {self.max_seq_len}"
            )
        self.keys[:, :, self.length:end] = k
        self.values[:, :, self.length:end] = v
        self.length = end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def reset(self) -> None:
        """Forget all cached tokens (buffers are kept allocated)."""
        self.length = 0


class KVCache:
    """A bundle of :class:`LayerKVCache` objects, one per transformer layer."""

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.layers: List[LayerKVCache] = [
            LayerKVCache(batch_size, n_kv_heads, max_seq_len, head_dim, dtype, device)
            for _ in range(n_layers)
        ]

    def __getitem__(self, index: int) -> LayerKVCache:
        return self.layers[index]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def length(self) -> int:
        """Number of tokens currently cached (identical across layers)."""
        return self.layers[0].length

    def reset(self) -> None:
        """Reset every layer's cache -- used to start a fresh conversation."""
        for layer in self.layers:
            layer.reset()

    def memory_bytes(self) -> int:
        """Total bytes held by the cache buffers."""
        total = 0
        for layer in self.layers:
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
        return total
