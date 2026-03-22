from typing import Optional

import torch


class EvictedKVCache:
    """Mutable container for per-layer evicted KV cache.

    Used as the return value from store_op so that compute_prefill_op can
    mutate it in-place after computing attention scores for eviction.

    During prefill, store_op creates an empty EvictedKVCache and returns it
    as both keys_return and values_return. compute_prefill_op then fills in
    the .keys and .values fields after performing eviction. Because Python
    passes objects by reference, the mutation is visible to
    MultiHeadAttention.forward() which holds the same reference.

    During decode, the cache holds (B, kvheads, S_evicted, D) tensors and
    new tokens are concatenated directly.

    Tracks seq_len_logical separately from the physical cache size so that
    RoPE position encoding computes correct positions after eviction.

    This is not a torch.Tensor, so _make_cache_contiguous() in generation.py
    will skip contiguity enforcement (isinstance check returns False).
    """

    def __init__(
        self,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        seq_len_logical: int = 0,
    ):
        self.keys = keys  # (B, kvheads, S_evicted, D) or None
        self.values = values  # (B, kvheads, S_evicted, D) or None
        # Logical sequence length: tracks the actual number of tokens processed
        # (pre-eviction), NOT the physical cache size. This is critical for RoPE
        # position encoding — RotaryEmbedding.adjusted_qk() calls .size(2) on
        # the cache to compute position_ids for new tokens.
        self.seq_len_logical = seq_len_logical

    # Tensor-like accessors needed by RotaryEmbedding.adjusted_qk()
    # which calls past_kv_state[0].numel() and past_kv_state[0].size(2)
    # to compute position_ids when they are not explicitly provided.

    def numel(self) -> int:
        if self.seq_len_logical == 0:
            return 0
        if self.keys is None:
            return 0
        return self.keys.numel()

    def size(self, dim=None):
        if self.keys is None:
            if dim is not None:
                return 0
            return torch.Size([0, 0, 0, 0])
        if dim == 2:
            # Return logical sequence length for RoPE position computation,
            # NOT the physical cache size after eviction.
            return self.seq_len_logical
        return self.keys.size(dim) if dim is not None else self.keys.size()

    def is_contiguous(self) -> bool:
        """Support _make_cache_contiguous checks."""
        if self.keys is None:
            return True
        return self.keys.is_contiguous() and self.values.is_contiguous()
