"""MiniKV attention op: KV cache eviction for FMS models.

Registers a "minikv" attention op via FMS's register_attention_op() system.
During prefill, computes full causal attention then evicts low-importance KV
pairs using H2O, SnapKV, or Pyramid selection. During decode, uses the evicted
(smaller) cache with standard SDPA.

Limitations:
- Not compatible with torch.compile dynamic_shapes mode (_make_cache_dynamic
  in generation.py will fail on EvictedKVCache objects).
- Explicit attention masks from pad_input_ids are dropped after prefill since
  the mask dimension doesn't match the evicted cache size. Decode runs without
  masking (is_causal=False for single-token queries).
- Decode tokens accumulate without re-eviction. For very long generation,
  the cache grows back toward full size.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from fms.modules.attention import AttentionKwargs, register_attention_op
from fms.utils.minikv.cache import EvictedKVCache
from fms.utils.minikv.selection import (
    H2OSelection,
    SnapKVSelection,
    create_selector,
)


@dataclass
class MiniKVConfig:
    """Configuration for MiniKV KV cache eviction."""

    selection_method: str = "h2o"  # "h2o", "snapkv", "pyramid_h2o", "pyramid_snapkv"
    heavy_ratio: float = 0.25  # H2O: fraction of heavy hitter tokens to keep
    recent_ratio: float = 0.25  # H2O: fraction of most recent tokens to keep
    window_size: int = 64  # SnapKV: attention window size for scoring
    prompt_sparsity_ratio: float = 0.25  # SnapKV: overall keep ratio
    kernel_size: int = 5  # SnapKV: pooling kernel size
    pooling: str = "avgpool"  # SnapKV: pooling method ("avgpool", "maxpool", or None)


def create_minikv_kwargs(
    config: MiniKVConfig,
    num_layers: int,
    is_causal_mask: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> dict:
    """Create the attn_kwargs dict for a MiniKV generation run.

    Args:
        config: MiniKV eviction configuration
        num_layers: Number of transformer layers in the model
        is_causal_mask: Whether to use causal masking
        mask: Optional explicit attention mask

    Returns:
        Dict to pass as extra_kwargs to generate(), or unpack into model forward.
    """
    selectors = [
        create_selector(
            config.selection_method,
            layer_id=i,
            num_layers=num_layers,
            heavy_ratio=config.heavy_ratio,
            recent_ratio=config.recent_ratio,
            window_size=config.window_size,
            prompt_sparsity_ratio=config.prompt_sparsity_ratio,
            kernel_size=config.kernel_size,
            pooling=config.pooling,
        )
        for i in range(num_layers)
    ]

    kwargs: dict[str, Any] = {
        "attn_name": "minikv",
        "is_causal_mask": is_causal_mask,
        "minikv_state": {
            "layer_counter": 0,
            "selectors": selectors,
            "config": config,
            "is_prefill": True,
        },
    }
    if mask is not None:
        kwargs["mask"] = mask
    return kwargs


# ---------------------------------------------------------------------------
# Attention Op implementations
# ---------------------------------------------------------------------------


def _minikv_store_op(
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: Optional[Any],
    value_cache: Optional[Any],
    **attn_kwargs,
) -> Tuple[Any, Any, Any, Any]:
    """Store op for MiniKV attention.

    During prefill (key_cache is None):
        - Transpose keys/values to (B, kvheads, S, D)
        - Create empty EvictedKVCache; compute_prefill_op will fill it in-place
        - Return full KV for compute, mutable wrapper for return

    During decode (key_cache is EvictedKVCache):
        - Transpose new single-token KV to (B, kvheads, 1, D)
        - Concatenate with existing cache
        - Return concatenated cache for both compute and return
    """
    state = attn_kwargs["minikv_state"]
    layer_idx = state["layer_counter"]
    state["layer_counter"] += 1

    # keys/values arrive as (B, S, kvheads, D) from MultiHeadAttention
    keys_t = keys.transpose(2, 1)  # (B, kvheads, S, D)
    values_t = values.transpose(2, 1)

    if key_cache is None:
        # === PREFILL ===
        seq_len = keys_t.shape[2]
        evicted_cache = EvictedKVCache(seq_len_logical=seq_len)
        state[f"_pending_{layer_idx}"] = evicted_cache
        return keys_t, values_t, evicted_cache, evicted_cache

    else:
        # === DECODE ===
        # key_cache and value_cache are the same EvictedKVCache object
        # (store_op returns it for both keys_return and values_return).
        cache: EvictedKVCache = key_cache
        cache.keys = torch.cat((cache.keys, keys_t), dim=2)
        cache.values = torch.cat((cache.values, values_t), dim=2)
        cache.seq_len_logical += keys_t.shape[2]  # +1 per decode step
        return cache.keys, cache.values, cache, cache


def _minikv_compute_prefill_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs,
) -> torch.Tensor:
    """Prefill compute: full causal attention + eviction.

    1. Compute full causal attention at nheads level (for accurate output)
    2. Aggregate cumulative attention scores from nheads → kvheads
    3. Evict tokens at kvheads granularity
    4. Mutate the EvictedKVCache returned by store_op in-place
    5. Return attention output from full (pre-eviction) attention
    """
    state = attn_kwargs["minikv_state"]
    layer_idx = state["layer_counter"] - 1  # store_op already incremented
    selector = state["selectors"][layer_idx]

    # query: (B, S, nheads, D), key/value_cache: (B, kvheads, S, D)
    queries = query.transpose(2, 1)  # (B, nheads, S, D)
    keys = key_cache  # (B, kvheads, S, D) from store_op
    values = value_cache

    # Expand KV heads for full attention computation
    expansion = nheads // kvheads
    if expansion != 1:
        keys_e = (
            keys.unsqueeze(2)
            .expand(-1, -1, expansion, -1, -1)
            .flatten(1, 2)
        )
        values_e = (
            values.unsqueeze(2)
            .expand(-1, -1, expansion, -1, -1)
            .flatten(1, 2)
        )
    else:
        keys_e, values_e = keys, values

    B, H, S, D = queries.shape  # H = nheads

    # 1. Full causal attention
    attn_weights = torch.matmul(queries, keys_e.transpose(2, 3))
    if scale_factor is not None:
        attn_weights = attn_weights * scale_factor
    else:
        attn_weights = attn_weights / math.sqrt(D)

    # Apply causal mask
    causal_mask = torch.full(
        (S, S), torch.finfo(attn_weights.dtype).min, device=attn_weights.device
    )
    mask_cond = torch.arange(S, device=attn_weights.device)
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(S, 1), 0)
    attn_weights = attn_weights + causal_mask[None, None, :, :]

    # Apply explicit mask if provided
    mask = attn_kwargs.get("mask", None)
    if mask is not None:
        while len(mask.size()) != 4:
            mask = mask.unsqueeze(1)
        attn_weights = attn_weights + mask

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(queries.dtype)

    # 2. Attention output from full (pre-eviction) attention
    attn_output = torch.matmul(attn_weights, values_e)  # (B, nheads, S, D)

    # 3. Aggregate attention scores nheads → kvheads for eviction
    # Use nansum to handle left-padded batches: padding queries where all keys
    # are masked produce NaN after softmax. Regular sum would propagate NaN to
    # all cumulative scores, breaking H2O token selection.
    cumulative_attn_map = torch.nansum(attn_weights, dim=2)  # (B, nheads, S)
    if expansion != 1:
        cumulative_attn_map_kv = cumulative_attn_map.view(
            B, kvheads, expansion, S
        ).sum(dim=2)
    else:
        cumulative_attn_map_kv = cumulative_attn_map

    # 4. Eviction at kvheads granularity
    if isinstance(selector, H2OSelection):
        k_evicted, v_evicted = selector.select(
            keys, values, cumulative_attn_map_kv
        )
    else:  # SnapKV
        # Aggregate queries to kvheads for windowed scoring
        if expansion != 1:
            queries_kv = queries.view(B, kvheads, expansion, S, D).mean(dim=2)
        else:
            queries_kv = queries
        k_evicted, v_evicted = selector.select(keys, values, queries_kv)

    # 5. Mutate the EvictedKVCache that store_op returned
    pending: EvictedKVCache = state[f"_pending_{layer_idx}"]
    pending.keys = k_evicted  # (B, kvheads, S_evicted, D)
    pending.values = v_evicted
    del state[f"_pending_{layer_idx}"]

    logger.info(
        f"[MiniKV] Layer {layer_idx}: prefill {S} -> kept {k_evicted.shape[2]} tokens "
        f"({k_evicted.shape[2]/S*100:.1f}%)"
    )

    # Return (B, S, nheads, D)
    return attn_output.transpose(2, 1).contiguous()


def _minikv_compute_decode_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs,
) -> torch.Tensor:
    """Decode compute: standard SDPA with GQA expansion.

    Cache is at kvheads granularity; expanded to nheads for the dot product.
    """
    # query: (B, 1, nheads, D), key/value_cache: (B, kvheads, S, D)
    queries = query.transpose(2, 1)  # (B, nheads, 1, D)

    # Expand KV heads for GQA
    expansion = nheads // kvheads
    if expansion != 1:
        keys_e = (
            key_cache.unsqueeze(2)
            .expand(-1, -1, expansion, -1, -1)
            .flatten(1, 2)
        )
        values_e = (
            value_cache.unsqueeze(2)
            .expand(-1, -1, expansion, -1, -1)
            .flatten(1, 2)
        )
    else:
        keys_e, values_e = key_cache, value_cache

    mask = attn_kwargs.get("mask", None)
    if mask is not None:
        while len(mask.size()) != 4:
            mask = mask.unsqueeze(1)

    attn_mask = mask
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(dtype=queries.dtype)

    attn = F.scaled_dot_product_attention(
        queries,
        keys_e,
        values_e,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale_factor,
    )
    # Return (B, 1, nheads, D)
    return attn.transpose(2, 1).contiguous()


def _minikv_is_prefill_op(**attn_kwargs) -> bool:
    return attn_kwargs["minikv_state"]["is_prefill"]


def _minikv_update_attn_kwargs_op(**attn_kwargs) -> dict:
    """Reset layer counter and transition from prefill to decode between generation steps."""
    state = attn_kwargs["minikv_state"]
    state["layer_counter"] = 0
    state["is_prefill"] = False

    # Drop the mask for decode. After eviction the KV cache has S_evicted tokens,
    # but the original mask was sized for S_prefill tokens. These dimensions don't
    # match, so the mask cannot be reused. Single-token decode queries against
    # the full cache don't need causal masking (is_causal=False in compute_decode).
    if "mask" in attn_kwargs:
        del attn_kwargs["mask"]

    return attn_kwargs


# ---------------------------------------------------------------------------
# Register the MiniKV attention op
# ---------------------------------------------------------------------------

register_attention_op(
    "minikv",
    _minikv_store_op,
    _minikv_compute_prefill_op,
    is_prefill_op=_minikv_is_prefill_op,
    compute_decode_op=_minikv_compute_decode_op,
    update_attn_kwargs_op=_minikv_update_attn_kwargs_op,
)
