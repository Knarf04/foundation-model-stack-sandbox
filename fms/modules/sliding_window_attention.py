"""Causal sliding-window attention (SWA), backed by flex_attention.

The window is expressed as a block-sparse BlockMask, giving O(q_len * window)
compute/memory instead of a dense O(q_len * k_len) additive mask through SDPA.
Requires torch >= 2.5 (torch.nn.attention.flex_attention); construction of
SlidingWindowMultiHeadAttention raises otherwise — there is no dense fallback.
"""

from typing import Any, Mapping, Optional, Tuple

from typing_extensions import NotRequired, Unpack

import torch
import torch.distributed
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional as F

from fms import distributed
from fms.distributed.tensorparallel import (
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    TPMultiHeadAttention,
    _sdpa_store_op,
    get_attention_type,
    register_attention_op,
)
from fms.modules.flex_utils import (
    additive_score_mod as _additive_score_mod,
    compose_block_mask as _compose_block_mask,
    flex_attention_available as _flex_attention_available,
    flex_attention_with_sinks as _flex_attention_with_sinks,
    get_sliding_window_block_mask as _get_sliding_window_block_mask,
    normalize_attention_mask as _normalize_attention_mask,
    sliding_window_mask_mod as _sliding_window_mask_mod,
)
from fms.modules.linear import get_linear_type
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule


class FlexSlidingWindowAttentionKwargs(AttentionKwargs):
    sliding_window: NotRequired[int]
    sinks: NotRequired[torch.Tensor]
    # user mask (padding/packing/document boundaries), True/0 = attend;
    # composed with the sliding window inside the op
    mask: NotRequired[torch.Tensor]


def _flex_swa_compute_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs,
) -> torch.Tensor:
    if p_dropout:
        raise NotImplementedError(
            "flex_sliding_window has no attention-dropout support"
        )

    queries = query.transpose(2, 1)
    # key_cache/value_cache are always b x kvheads x kv_len x ds: the store op
    # and the uncached forward path normalize the layout before this op runs.
    # Do not reintroduce shape-based layout inference here — when
    # kv_len == kvheads the two layouts are indistinguishable by shape, and a
    # wrong guess silently swaps the token and head axes

    window_size = attn_kwargs["sliding_window"]
    sinks = attn_kwargs.get("sinks", None)
    mask = attn_kwargs.get("mask", None)
    batch_size = queries.shape[0]
    q_len = queries.shape[2]
    k_len = key_cache.shape[2]
    enable_gqa = nheads != kvheads
    device = str(queries.device)

    # A user mask (padding, packing, document boundaries) composes with the
    # window: allowed(b,q,k) = window(q,k) AND user(b,q,k). Boolean masks fold
    # into the BlockMask; additive float masks become a score_mod, so the lse
    # (and hence the sink gate) accounts for them automatically.

    # This op is only registered when flex is importable (torch >= 2.5), so
    # SDPA's enable_gqa is also available and manual kv expansion is unneeded.
    if q_len == 1:
        # Decode: the single (newest) query attends to exactly the last
        # min(window_size, k_len) cache slots, so dense attention over that
        # slice is exact — no window BlockMask construction per step.
        keys_s = key_cache[:, :, -window_size:, :]
        values_s = value_cache[:, :, -window_size:, :]
        s_len = keys_s.shape[2]
        user = (
            None
            if mask is None
            else _normalize_attention_mask(mask, batch_size, 1, s_len)
        )
        if sinks is None:
            attn_mask = None
            if user is not None:
                attn_mask = user.unsqueeze(1)  # b x 1 x 1 x s_len
                if attn_mask.dtype != torch.bool:
                    attn_mask = attn_mask.to(dtype=queries.dtype)
            attn = F.scaled_dot_product_attention(
                queries,
                keys_s,
                values_s,
                attn_mask=attn_mask,
                is_causal=False,
                scale=scale_factor,
                enable_gqa=enable_gqa,
            )
        else:
            # Sink gating needs the row logsumexp, which SDPA cannot expose —
            # flex over the slice yields it as a free byproduct.
            if user is None:
                block_mask, score_mod = None, None
            elif user.dtype == torch.bool:
                block_mask = _compose_block_mask(None, user, 1, s_len, device)
                score_mod = None
            else:
                block_mask, score_mod = None, _additive_score_mod(user)
            attn = _flex_attention_with_sinks(
                queries,
                keys_s,
                values_s,
                block_mask,
                scale_factor,
                enable_gqa,
                sinks,
                score_mod=score_mod,
            )
    else:
        user = (
            None
            if mask is None
            else _normalize_attention_mask(mask, batch_size, q_len, k_len)
        )
        score_mod = None
        if user is None:
            block_mask = _get_sliding_window_block_mask(
                q_len, k_len, window_size, device
            )
        elif user.dtype == torch.bool:
            block_mask = _compose_block_mask(
                _sliding_window_mask_mod(k_len - q_len, window_size),
                user,
                q_len,
                k_len,
                device,
            )
        else:
            block_mask = _get_sliding_window_block_mask(
                q_len, k_len, window_size, device
            )
            score_mod = _additive_score_mod(user)
        attn = _flex_attention_with_sinks(
            queries,
            key_cache,
            value_cache,
            block_mask,
            scale_factor,
            enable_gqa,
            sinks,
            score_mod=score_mod,
        )

    # b x h x qlen x ds -> b x qlen x h x ds
    return attn.transpose(2, 1).contiguous()


if _flex_attention_available:
    register_attention_op(
        "flex_sliding_window",
        _sdpa_store_op,
        _flex_swa_compute_op,
    )


class SlidingWindowMultiHeadAttention(MultiHeadAttention):
    """
    MultiHeadAttention with a causal sliding window: each query attends to at
    most `sliding_window` most-recent keys (including itself), and the returned
    KV cache is trimmed to the last `sliding_window` entries so cache size
    stays bounded during generation.
    ...
    Args
    ----
    Check MultiHeadAttention for the shared args.

    sliding_window : int
        Causal sliding-window attention size. Defaults to 512.
    use_sinks : bool
        If True, add a learned attention sink (see MultiHeadAttention): one
        trainable zero-init logit per query head entering the softmax
        denominator with value 0.
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        fused: bool = True,
        linear_config: Optional[Mapping[str, Any]] = None,
        scale_factor: Optional[float] = None,
        sliding_window: int = 512,
        use_sinks: bool = False,
    ):
        if not _flex_attention_available:
            raise ImportError(
                "SlidingWindowMultiHeadAttention requires "
                "torch.nn.attention.flex_attention (torch >= 2.5)"
            )
        if p_dropout:
            raise ValueError(
                "SlidingWindowMultiHeadAttention does not support attention "
                "dropout (flex_attention has no dropout); use p_dropout=0"
            )
        if not isinstance(sliding_window, int) or sliding_window <= 0:
            raise ValueError(
                f"sliding_window must be a positive integer, got {sliding_window}"
            )
        super().__init__(
            emb_dim,
            emb_kq,
            emb_v,
            nheads,
            kvheads,
            p_dropout,
            use_bias,
            position_encoder,
            fused,
            linear_config,
            scale_factor,
            use_sinks=use_sinks,
        )
        self.sliding_window = sliding_window

        # Local absolute-position counter for RoPE under trimmed SWA cache.
        # Plain Python int (not a buffer) so DDP does not try to broadcast it.
        # Only consistent for batch_size=1, single-stream generation; caller must
        # invoke reset_position_counter() before each new prompt.
        self.curr_id = 0

    def reset_position_counter(self):
        """Reset the local absolute-position counter. Call before each new
        single-stream generation. A top-level model wrapper can walk modules
        and invoke this on every SlidingWindowMultiHeadAttention submodule."""
        self.curr_id = 0

    def to_tp(self, group: ProcessGroup) -> "TPSlidingWindowMultiHeadAttention":
        return TPSlidingWindowMultiHeadAttention.import_module(self, group)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_value_state: Optional[Tuple[Tensor | None, Tensor | None]] = None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        Check MultiHeadAttention for the shared arguments and return format.
        This forward always computes through the "flex_sliding_window" op;
        other attn_name values are rejected. A user mask (padding/packing;
        True/0 = attend, FMS bs x q_len x kv_len convention) is composed with
        the sliding window inside the op.
        """
        # q, k, v: batch_size x seq_len x emb_dim
        batch_size, q_len, _ = q.size()

        attn_name = attn_kwargs.get("attn_name", "flex_sliding_window")
        if attn_name not in ("sdpa_causal", "flex_sliding_window"):
            raise ValueError(
                "SlidingWindowMultiHeadAttention only supports the "
                f"flex_sliding_window attention op, got attn_name=`{attn_name}`"
            )
        attn_kwargs["attn_name"] = "flex_sliding_window"
        attn_kwargs["sliding_window"] = self.sliding_window
        if self.sinks is not None:
            attn_kwargs["sinks"] = self.sinks

        q_out, k_out, v_out = self.in_proj(q, k, v)

        # note: transposes will be moved in a later PR to fix dis-contiguous tensor issues
        queries = q_out.view(batch_size, q_len, self.nheads, self.emb_kq_per_head)
        keys = k_out.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head)
        values = v_out.view(batch_size, q_len, self.kvheads, self.emb_v_per_head)

        # Sliding-window KV cache trimming invalidates implicit cache-length position
        # derivation in RoPE. Fall back to a local absolute-position counter when no
        # position_ids is given. Restricted to batch_size=1 single-stream generation.
        if self.position_encoder is not None and use_cache:
            if position_ids is None:
                if batch_size != 1:
                    raise ValueError(
                        "Local RoPE position counter only supports batch_size=1. "
                        "Pass explicit absolute position_ids for batched generation."
                    )
                position_ids = torch.arange(
                    self.curr_id,
                    self.curr_id + q_len,
                    device=q.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                self.curr_id += q_len
            elif position_ids.shape[0] == 1:
                # Keep local counter synchronized when a single-stream caller
                # provides positions.
                self.curr_id = int(position_ids[0, -1].item()) + 1
            # batch > 1 with explicit absolute position_ids: use them as given;
            # the local counter only tracks single-stream generation. Note that
            # uniform cache trimming assumes right-aligned sequences — batched
            # generation with padding needs left padding plus a user mask.

        # You want to apply rotary embeddings pre-cache
        if self.position_encoder is not None:
            queries, keys = self.position_encoder.adjusted_qk(
                queries, keys, position_ids, past_key_value_state, use_cache
            )

        attn_compute_dict = get_attention_type(**attn_kwargs)

        if use_cache:
            if past_key_value_state is None:
                past_key_value_state = (None, None)

            keys_compute, values_compute, keys_return, values_return = (
                attn_compute_dict["store"](
                    keys,
                    values,
                    past_key_value_state[0],
                    past_key_value_state[1],
                    **attn_kwargs,
                )
            )
        else:
            # Normalize K/V to b x kvheads x seq_len x ds here so the compute
            # op never has to infer the layout from tensor shapes: when
            # seq_len == kvheads, b x seq_len x kvheads x ds and
            # b x kvheads x seq_len x ds have identical shapes, so a shape-based
            # guess can silently swap the token and head axes. The store op
            # upholds the same b x kvheads x seq_len x ds contract on the
            # cached path.
            keys_compute = keys.transpose(2, 1)
            values_compute = values.transpose(2, 1)

        if attn_compute_dict["is_prefill"](**attn_kwargs):
            attn = attn_compute_dict["compute_prefill"](
                queries,
                keys_compute,
                values_compute,
                self.nheads,
                self.kvheads,
                self.p_dropout if self.training else 0.0,
                self.scale_factor,
                **attn_kwargs,
            )
        else:
            attn = attn_compute_dict["compute_decode"](
                queries,
                keys_compute,
                values_compute,
                self.nheads,
                self.kvheads,
                self.p_dropout if self.training else 0.0,
                self.scale_factor,
                **attn_kwargs,
            )

        attn = attn.view(batch_size, q_len, self.nheads * self.emb_v_per_head)
        gate = F.silu(self.gate_proj(q))
        attn = gate * attn
        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache.
        # Trim the returned cache to at most `sliding_window` most-recent entries so
        # cache size stays bounded — keep computed attention intact for this step.
        if use_cache:
            if keys_return.shape[2] > self.sliding_window:
                keys_return = keys_return[:, :, -self.sliding_window :, :].contiguous()
                values_return = values_return[
                    :, :, -self.sliding_window :, :
                ].contiguous()
            return out, (keys_return, values_return)
        else:
            return out


class TPSlidingWindowMultiHeadAttention(SlidingWindowMultiHeadAttention, TPModule):
    """
    SlidingWindowMultiHeadAttention with Tensor Parallel support.
    ...
    Args
    ----
    Check SlidingWindowMultiHeadAttention for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        fused: bool = True,
        group: Optional[ProcessGroup] = None,
        linear_config: Optional[Mapping[str, Any]] = None,
        scale_factor: Optional[float] = None,
        sliding_window: int = 512,
        use_sinks: bool = False,
    ):
        assert torch.distributed.is_initialized()

        rank, world_size = distributed.rank_and_world(group)
        assert nheads % world_size == 0, (
            "The number of heads must be divisible by world size"
        )
        assert (kvheads >= world_size and kvheads % world_size == 0) or (
            kvheads < world_size and world_size % kvheads == 0
        ), (
            "the kv heads must be divisible by the world size or the world size must be divisible by kv heads"
        )
        SlidingWindowMultiHeadAttention.__init__(
            self,
            emb_dim,
            emb_kq,
            emb_v,
            nheads // world_size,
            (kvheads // world_size) if kvheads >= world_size else 1,
            p_dropout,
            use_bias,
            position_encoder,
            fused,
            linear_config,
            scale_factor,
            sliding_window,
            use_sinks=use_sinks,
        )
        self.pre_tp_nheads = nheads
        self.pre_tp_kvheads = kvheads
        self.setup_tp(rank, group)

        # linear_type must handle module_name = None to support TP of MHA
        self.linear_type = get_linear_type(self.linear_config)

    # Sharding layout and TP input replication are identical to the full
    # attention TP module (same submodules: in_proj, gate_proj, dense).
    load_weights = TPMultiHeadAttention.load_weights
    _copy_to_tp_region = TPMultiHeadAttention._copy_to_tp_region

    @staticmethod
    def import_module(
        mha: SlidingWindowMultiHeadAttention, group: ProcessGroup
    ) -> "TPSlidingWindowMultiHeadAttention":
        tp_mha = TPSlidingWindowMultiHeadAttention(
            emb_dim=mha.emb_dim,
            emb_kq=mha.emb_kq_per_head,
            emb_v=mha.emb_v_per_head,
            nheads=mha.nheads,
            kvheads=mha.kvheads,
            p_dropout=mha.p_dropout,
            use_bias=mha.use_bias,
            position_encoder=mha.position_encoder,
            group=group,
            fused=mha.fused,
            linear_config=mha.linear_config,
            scale_factor=mha.scale_factor,
            sliding_window=mha.sliding_window,
            use_sinks=mha.sinks is not None,
        )
        return tp_mha

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_value_state: Optional[Tuple[Tensor | None, Tensor | None]] = None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        Check SlidingWindowMultiHeadAttention for up-to-date arguments and docs
        """

        q_par, k_par, v_par = self._copy_to_tp_region(q, k, v)

        out_par = SlidingWindowMultiHeadAttention.forward(
            self,
            q_par,
            k_par,
            v_par,
            position_ids,
            past_key_value_state,
            use_cache,
            **attn_kwargs,
        )

        # if use_cache=True, we return the hidden_state as well as the kv cache.
        # We only reduce the output, and keep the cache thread-local
        if use_cache:
            out = reduce_from_tensor_model_parallel_region(out_par[0], self.group)
            return out, out_par[1]
        else:
            out = reduce_from_tensor_model_parallel_region(out_par, self.group)
            return out
