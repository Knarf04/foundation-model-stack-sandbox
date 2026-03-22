import logging
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class H2OSelection:
    """Heavy Hitter Oracle: keep top-k by cumulative attention score + most recent tokens.

    All operations are at kvheads granularity. The caller is responsible for
    aggregating nheads-level attention scores to kvheads before calling select().
    """

    def __init__(self, heavy_ratio: float = 0.25, recent_ratio: float = 0.25):
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        logger.info(
            f"[H2O Selection] heavy_ratio={heavy_ratio}, recent_ratio={recent_ratio}"
        )

    def select(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cumulative_attn_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select tokens to keep from KV cache based on cumulative attention scores.

        Args:
            key_states:          (B, kvheads, S, D)
            value_states:        (B, kvheads, S, D)
            cumulative_attn_map: (B, kvheads, S) — sum of attention weights over queries

        Returns:
            (evicted_keys, evicted_values) both (B, kvheads, S_kept, D)
        """
        bsz, num_heads, q_len, head_dim = key_states.shape
        num_hh = int(q_len * self.heavy_ratio)
        num_rw = int(q_len * self.recent_ratio)

        if num_hh + num_rw >= q_len:
            return key_states, value_states

        hh_score = cumulative_attn_map

        # Select top-k heavy hitters from non-recent tokens
        select_hh_scores = hh_score[:, :, : q_len - num_rw]
        _, keep_topk = torch.topk(select_hh_scores, num_hh, dim=-1)
        keep_topk = keep_topk.sort().values

        # Always keep most recent tokens
        keep_recent = torch.arange(
            q_len - num_rw, q_len, device=keep_topk.device
        ).repeat(keep_topk.shape[0], keep_topk.shape[1], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(hh_score.shape, dtype=torch.bool, device=keep_topk.device)
        mask = mask.scatter(-1, keep_idx, 1)

        key_states_compress = key_states[mask].view(bsz, num_heads, -1, head_dim)
        value_states_compress = value_states[mask].view(bsz, num_heads, -1, head_dim)

        return key_states_compress, value_states_compress


class PyramidH2OSelection(H2OSelection):
    """H2O with per-layer budget: more tokens kept in lower layers, fewer in upper.

    Uses a linear interpolation with delta=7 to distribute the total heavy_ratio
    budget across layers, giving lower layers a higher ratio.
    """

    def __init__(
        self,
        heavy_ratio: float = 0.25,
        recent_ratio: float = 0.25,
        layer_id: int = 0,
        num_layers: int = 32,
    ):
        m = num_layers
        delta = 7
        k_total = heavy_ratio * m
        k = [0.0] * m
        k[m - 1] = k_total / m / delta
        k[0] = 2 * k_total / m - k[m - 1]
        for l in range(1, m - 1):
            k[l] = k[0] + (k[m - 1] - k[0]) / (m - 1) * l

        self.heavy_ratio = k[layer_id]
        self.recent_ratio = recent_ratio
        self.layer_id = layer_id
        self.num_layers = num_layers
        logger.info(
            f"[Pyramid H2O] heavy_ratio={self.heavy_ratio:.4f}, "
            f"recent_ratio={self.recent_ratio}, layer={layer_id}/{num_layers}"
        )


class SnapKVSelection:
    """Attention-pattern-based sparsity: use windowed attention to score token importance.

    Uses the last `window_size` query tokens to compute attention weights over all
    key positions, then applies pooling to smooth the scores. Top-k tokens are kept
    along with the most recent window_size tokens.

    All operations are at kvheads granularity.
    """

    def __init__(
        self,
        window_size: int = 64,
        prompt_sparsity_ratio: float = 0.25,
        kernel_size: int = 5,
        pooling: str = "avgpool",
    ):
        self.window_size = window_size
        self.prompt_sparsity_ratio = prompt_sparsity_ratio
        self.kernel_size = kernel_size
        self.pooling = pooling
        logger.info(
            f"[SnapKV Selection] window_size={window_size}, "
            f"prompt_sparsity_ratio={prompt_sparsity_ratio}, "
            f"kernel_size={kernel_size}, pooling={pooling}"
        )

    def select(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select tokens to keep using windowed attention pattern analysis.

        Args:
            key_states:    (B, kvheads, S, D)
            value_states:  (B, kvheads, S, D)
            query_states:  (B, kvheads, S, D) — queries aggregated to kvheads

        Returns:
            (evicted_keys, evicted_values) both (B, kvheads, S_kept, D)
        """
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        retained_tokens = int(self.prompt_sparsity_ratio * q_len)
        if retained_tokens < self.window_size:
            retained_tokens = max(1, retained_tokens)
            key_states = key_states[:, :, -retained_tokens:, :]
            value_states = value_states[:, :, -retained_tokens:, :]
            return key_states, value_states

        # Compute windowed attention: last window_size queries attend to all keys
        attn_weights = torch.matmul(
            query_states[..., -self.window_size :, :], key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        # Causal mask for the window region
        mask = torch.full(
            (self.window_size, self.window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attention_mask = mask[None, None, :, :]

        attn_weights[
            :, :, -self.window_size :, -self.window_size :
        ] += attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # Sum attention to non-window tokens across window queries
        attn_weights_sum = attn_weights[
            :, :, -self.window_size :, : -self.window_size
        ].sum(dim=-2)

        # Apply pooling for smoother token importance scores
        if self.pooling == "avgpool":
            attn_cache = F.avg_pool1d(
                attn_weights_sum,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        elif self.pooling == "maxpool":
            attn_cache = F.max_pool1d(
                attn_weights_sum,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        elif self.pooling is None:
            attn_cache = attn_weights_sum
        else:
            raise ValueError(f"Pooling method '{self.pooling}' not supported")

        # Select top-k important tokens from non-window region
        num_keep = retained_tokens - self.window_size
        indices = attn_cache.topk(num_keep, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )
        v_past_compress = value_states[:, :, : -self.window_size, :].gather(
            dim=2, index=indices
        )

        # Keep the most recent window tokens
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states


class PyramidSnapKVSelection(SnapKVSelection):
    """SnapKV with per-layer budget: more tokens kept in lower layers, fewer in upper.

    Uses the same linear interpolation as PyramidH2O (delta=7) to distribute
    the prompt_sparsity_ratio across layers.
    """

    def __init__(
        self,
        window_size: int = 64,
        prompt_sparsity_ratio: float = 0.25,
        kernel_size: int = 5,
        pooling: str = "avgpool",
        layer_id: int = 0,
        num_layers: int = 32,
    ):
        m = num_layers
        delta = 7
        k_total = prompt_sparsity_ratio * m
        k = [0.0] * m
        k[m - 1] = k_total / m / delta
        k[0] = 2 * k_total / m - k[m - 1]
        for l in range(1, m - 1):
            k[l] = k[0] + (k[m - 1] - k[0]) / (m - 1) * l

        self.window_size = window_size
        self.prompt_sparsity_ratio = min(1.0, k[layer_id])
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.layer_id = layer_id
        self.num_layers = num_layers
        logger.info(
            f"[Pyramid SnapKV] prompt_sparsity_ratio={self.prompt_sparsity_ratio:.4f}, "
            f"window_size={window_size}, layer={layer_id}/{num_layers}"
        )


def create_selector(
    method: str,
    layer_id: int,
    num_layers: int,
    heavy_ratio: float = 0.25,
    recent_ratio: float = 0.25,
    window_size: int = 64,
    prompt_sparsity_ratio: float = 0.25,
    kernel_size: int = 5,
    pooling: str = "avgpool",
) -> Union[H2OSelection, SnapKVSelection]:
    """Factory for creating selection mechanisms.

    Args:
        method: One of "h2o", "snapkv", "pyramid_h2o", "pyramid_snapkv"
        layer_id: Index of the current layer
        num_layers: Total number of layers in the model
    """
    if method == "h2o":
        return H2OSelection(heavy_ratio=heavy_ratio, recent_ratio=recent_ratio)
    elif method == "pyramid_h2o":
        return PyramidH2OSelection(
            heavy_ratio=heavy_ratio,
            recent_ratio=recent_ratio,
            layer_id=layer_id,
            num_layers=num_layers,
        )
    elif method == "snapkv":
        return SnapKVSelection(
            window_size=window_size,
            prompt_sparsity_ratio=prompt_sparsity_ratio,
            kernel_size=kernel_size,
            pooling=pooling,
        )
    elif method == "pyramid_snapkv":
        return PyramidSnapKVSelection(
            window_size=window_size,
            prompt_sparsity_ratio=prompt_sparsity_ratio,
            kernel_size=kernel_size,
            pooling=pooling,
            layer_id=layer_id,
            num_layers=num_layers,
        )
    else:
        raise ValueError(
            f"Unknown selection method '{method}'. "
            f"Choose from: h2o, snapkv, pyramid_h2o, pyramid_snapkv"
        )
