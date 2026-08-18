import abc
import functools
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
)
from typing_extensions import NotRequired, Unpack

import torch
import torch.distributed
from torch import Tensor, nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional as F

from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules import flex_utils
from fms.modules.linear import (
    LinearModuleShardingInfo,
    get_all_linear_type_to_sharding_maps,
    get_linear,
    get_linear_type,
)
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule

__sdpa_previous_flash: bool = torch.backends.cuda.flash_sdp_enabled()
__sdpa_previous_mem_efficient: bool = torch.backends.cuda.mem_efficient_sdp_enabled()
__sdpa_previous_math: bool = torch.backends.cuda.math_sdp_enabled()


__type_factory_map: dict[str, dict[str, Callable]] = {}


class AttentionKwargs(TypedDict, total=False):
    """
    The attention kwargs to be passed to fms model forward.

    attn_name: str
        this is the name corresponding to the attention op registered in register_attention_op
    """

    attn_name: str


# TODO: add adjusted_mask for alibi as part of attn_compute_dict
def register_attention_op(
    attn_type: str,
    store_op: Callable[
        [
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Unpack[AttentionKwargs],
        ],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    compute_op: Callable[
        [
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            int,
            float,
            float,
            Unpack[AttentionKwargs],
        ],
        torch.Tensor,
    ],
    is_prefill_op: Optional[Callable[[Unpack[AttentionKwargs]], bool]] = None,
    compute_decode_op: Optional[
        Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
                int,
                float,
                float,
                Unpack[AttentionKwargs],
            ],
            torch.Tensor,
        ]
    ] = None,
    update_attn_kwargs_op: Optional[
        Callable[[Unpack[AttentionKwargs]], AttentionKwargs]
    ] = None,
    validate_attn_kwargs_op: Optional[
        Callable[
            [
                torch.Tensor,
                torch.Tensor,
                Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
                Unpack["AttentionKwargs"],
            ],
            None,
        ]
    ] = None,
) -> None:
    """Register a custom attention operation to be used within MultiHeadAttention. This method also provides the ability to register other useful constructs related to the attention type.

    Args:
        attn_type: str
            the name for the attention_op. This should correspond directly to the AttentionKwargs implementation
        store_op: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Unpack["AttentionKwargs"]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
            This function has the following contract (keys, values, key_cache, value_cache, **attn_kwargs) -> (keys_compute, values_compute, keys_return, values_return). The intention
            of this function is to provide a method of storing the keys in the key_cache and the values in the value_cache. The return of this method will include what keys/values to compute
            on as well as what keys/values to return from MultiHeadAttention. Note: Reason for keeping these separate is that in some cases the keys to compute will be different than those
            that are to be returned from MultiHeadAttention. For example, in Paged Attention, we may use sdpa as prefill (utilitizing the initial computed keys/values), but the returned cache
            should be the larger cache that we stored to.
        compute_op: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, int, float, float, Unpack["AttentionKwargs"]], torch.Tensor]
            This function has the following contract (query, key_cache, value_cache, nheads, kvheads, p_dropout, scale_factor, **attn_kwargs) -> (attn_output) --
            query - b x qlen x h x ds, attn_output - b x qlen x h x ds. The intention of this function is perform attention computation. Note: the kv-cache may be very different in shape
            depending on the type of attention
        is_prefill_op: Optional[Callable[[Unpack["AttentionKwargs"]], bool]]
            This function has the following contract (**attn_kwargs) -> bool. The intention of this function is to denote given the attention kwargs whether prefill or decode is being performed.
            If prefill is being performed, the compute_op will be called, otherwise the compute_decode_op will be called. If set to None, this funcion will always return True.
        compute_decode_op: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, int, float, float, Unpack["AttentionKwargs"]], torch.Tensor]
            This function has the following contract (query, key_cache, value_cache, nheads, kvheads, p_dropout, scale_factor, **attn_kwargs) -> (attn_output) --
            query - b x qlen x h x ds, attn_output - b x qlen x h x ds. The intention of this function to provide a separate attention computation for decode. If this is set to something other than
            compute_op, is_prefill_op should also be provided. If set to None, this will default to the compute_op. Note: the kv-cache may be very different in shape depending on the type of attention
        update_attn_kwargs_op: Optional[Callable[[Unpack["AttentionKwargs"]], "AttentionKwargs"]]
            This function has the following contract (**attn_kwargs) -> updated_attn_kwargs. The intention of this function is to act as a helper to update the attn_kwargs between each step within a
            generation loop. If set to None, will return the attn_kwargs with no changes.
        validate_attn_kwargs_op: Optional[Callable[[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Unpack["AttentionKwargs"]], None]]
            This function has the following contract (input_ids, position_ids, past_key_value_states, **attn_kwargs) -> None. The intention of this function is do further validation against the
            attn_kwargs for a given forward pass. If set to None, this function will perform no extra validation.
    """
    if attn_type in __type_factory_map:
        raise KeyError(
            f"Module mapping of attention type `{attn_type}` already registered"
        )
    if compute_decode_op is None:
        compute_decode_op = compute_op

    compute_dict = {
        "store": store_op,
        "is_prefill": (lambda **_: True) if is_prefill_op is None else is_prefill_op,
        "compute_prefill": compute_op,
        "compute_decode": compute_decode_op,
        "update_attn_kwargs": (lambda **attn_kwargs: attn_kwargs)
        if update_attn_kwargs_op is None
        else update_attn_kwargs_op,
        "validate_attn_kwargs": (lambda **_: None)
        if validate_attn_kwargs_op is None
        else validate_attn_kwargs_op,
    }
    __type_factory_map[attn_type] = compute_dict


class SDPAAttentionKwargs(AttentionKwargs):
    mask: NotRequired[torch.Tensor]
    attn_algorithm: NotRequired[str]
    is_causal_mask: bool


def _sdpa_store_op(
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: Optional[torch.Tensor],
    value_cache: Optional[torch.Tensor],
    **attn_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = keys.transpose(2, 1)
    values = values.transpose(2, 1)

    if key_cache is not None and value_cache is not None and value_cache.numel() > 0:
        key_cache_result = torch.cat((key_cache, keys), dim=2)
        value_cache_result = torch.cat((value_cache, values), dim=2)
        return (
            key_cache_result,
            value_cache_result,
            key_cache_result,
            value_cache_result,
        )
    else:
        return (keys, values, keys, values)


def _sdpa_compute_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs,
) -> torch.Tensor:
    queries = query.transpose(2, 1)

    # key_cache/value_cache are always b x kvheads x kv_len x ds: store ops and
    # the uncached forward path normalize the layout before this op is called.
    # Do not reintroduce shape-based layout inference here — when
    # kv_len == kvheads the two layouts are indistinguishable by shape, and a
    # wrong guess silently swaps the token and head axes
    mask = attn_kwargs.get("mask", None)

    # TODO: Once we add alibi support, merge rel pos bias and mask into single float mask
    if mask is not None:
        # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
        # we need to create the nheads dimension
        while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
            mask = mask.unsqueeze(1)

    # Expand kv so black-box attn will work
    expansion = nheads // kvheads
    # k/v: b h l d
    if expansion != 1:
        keys_e = key_cache.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        values_e = (
            value_cache.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        )
    else:
        keys_e = key_cache
        values_e = value_cache

    attn_algorithm = attn_kwargs.get("attn_algorithm", None)
    if attn_algorithm:
        # Pick which fused attn kernels will run.
        use_flash = attn_algorithm == "flash"
        use_mem_efficient = attn_algorithm == "mem"
        use_math = attn_algorithm == "math"

        torch.backends.cuda.enable_flash_sdp(use_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(use_mem_efficient)
        torch.backends.cuda.enable_math_sdp(use_math)

    attn_mask = mask
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(dtype=queries.dtype)

    # For the causal op (is_causal_mask absent or True), causality is intrinsic
    # and a user mask is an ADDITIONAL restriction: effective = causal AND user.
    # Without this, any supplied mask silently disabled causality (is_causal
    # became False), so e.g. the HF adapter's dense pad-only masks made full
    # attention bidirectional. Composing is idempotent for masks that already
    # encode causality. sdpa_bidirectional passes is_causal_mask=False and is
    # unaffected. Bottom-right aligned for cached calls; a single decode query
    # attends every key, so composition is skipped there as an identity.
    is_causal_op = attn_kwargs.get("is_causal_mask", True)
    q_len = queries.shape[2]
    k_len = keys_e.shape[2]
    if is_causal_op and attn_mask is not None and q_len > 1:
        q_pos = torch.arange(k_len - q_len, k_len, device=queries.device)
        k_pos = torch.arange(k_len, device=queries.device)
        causal = (k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)).view(1, 1, q_len, k_len)
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask & causal
        else:
            attn_mask = attn_mask.masked_fill(~causal, float("-inf"))
        is_causal = False
    else:
        is_causal = is_causal_op and attn_mask is None and not (k_len != 1 and q_len == 1)

    # TODO: when updating to 2.7, use enable_gqa and stop using keys_e and values_e
    attn = F.scaled_dot_product_attention(
        queries,
        keys_e,
        values_e,
        attn_mask=attn_mask,
        dropout_p=p_dropout,
        is_causal=is_causal,
        scale=scale_factor,
    )

    if attn_algorithm:
        torch.backends.cuda.enable_flash_sdp(__sdpa_previous_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(__sdpa_previous_mem_efficient)
        torch.backends.cuda.enable_math_sdp(__sdpa_previous_math)

    # attn: bs x seq_len x nheads*emb_v_per_head
    # attn: b x h x qlen x ds
    # attn after permute: b x qlen x h x ds
    # b x qlen x (d)
    attn = attn.transpose(2, 1).contiguous()
    return attn


def _sdpa_update_attn_kwargs(**attn_kwargs):
    # this is updating the mask for decoding
    mask = attn_kwargs.get("mask", None)
    if mask is not None:
        # get the last row of the 3d mask
        mask = mask[:, -1:, :]
        # extend the mask one slot
        mask = torch.cat(
            (
                mask,
                torch.zeros(mask.size(0), 1, 1, device=mask.device),
            ),
            dim=2,
        )
        if torch._dynamo.config.dynamic_shapes:
            torch._dynamo.mark_dynamic(mask, 2)

        attn_kwargs["mask"] = mask
    return attn_kwargs


register_attention_op(
    "sdpa_causal",
    _sdpa_store_op,
    _sdpa_compute_op,
    update_attn_kwargs_op=_sdpa_update_attn_kwargs,
)
register_attention_op(
    "sdpa_bidirectional",
    _sdpa_store_op,
    functools.partial(_sdpa_compute_op, is_causal_mask=False),
)


class FlexCausalAttentionKwargs(AttentionKwargs):
    sinks: NotRequired[torch.Tensor]
    # user mask (padding/packing), True/0 = attend; composed with causality
    mask: NotRequired[torch.Tensor]


def _flex_causal_compute_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs,
) -> torch.Tensor:
    """Causal full attention through flex_attention, with optional learned
    sinks. Used by MultiHeadAttention when use_sinks=True: SDPA cannot expose
    the row logsumexp the sink gate needs, while flex produces it as a free
    byproduct of its streaming softmax — no logits recomputation.

    A user mask (padding/packing, FMS bs x q_len x kv_len convention) is
    composed with causality: allowed(b,q,k) = causal(q,k) AND user(b,q,k).
    Boolean masks fold into the BlockMask; additive float masks become a
    score_mod, so the lse (and hence the sink gate) accounts for them."""
    if p_dropout:
        raise NotImplementedError("flex_causal has no attention-dropout support")

    queries = query.transpose(2, 1)
    # key_cache/value_cache are always b x kvheads x kv_len x ds: the store op
    # and the uncached forward path normalize the layout before this op runs.
    # Do not reintroduce shape-based layout inference here — when
    # kv_len == kvheads the two layouts are indistinguishable by shape

    batch_size = queries.shape[0]
    q_len = queries.shape[2]
    k_len = key_cache.shape[2]
    device = str(queries.device)

    mask = attn_kwargs.get("mask", None)
    user = (
        None
        if mask is None
        else flex_utils.normalize_attention_mask(mask, batch_size, q_len, k_len)
    )
    # Decode: the single (newest) query attends to every key — a dense row
    # needs no causal BlockMask and stays streamed inside the kernel.
    base_mod = None if q_len == 1 else flex_utils.causal_mask_mod(k_len - q_len)
    score_mod = None
    if user is None:
        block_mask = (
            None
            if base_mod is None
            else flex_utils.get_causal_block_mask(q_len, k_len, device)
        )
    elif user.dtype == torch.bool:
        block_mask = flex_utils.compose_block_mask(
            base_mod, user, q_len, k_len, device
        )
    else:
        block_mask = (
            None
            if base_mod is None
            else flex_utils.get_causal_block_mask(q_len, k_len, device)
        )
        score_mod = flex_utils.additive_score_mod(user)

    attn = flex_utils.flex_attention_with_sinks(
        queries,
        key_cache,
        value_cache,
        block_mask,
        scale_factor,
        nheads != kvheads,
        attn_kwargs.get("sinks", None),
        score_mod=score_mod,
    )

    # b x h x qlen x ds -> b x qlen x h x ds
    return attn.transpose(2, 1).contiguous()


if flex_utils.flex_attention_available:
    register_attention_op(
        "flex_causal",
        _sdpa_store_op,
        _flex_causal_compute_op,
    )


def get_attention_type(**attn_kwargs: Unpack[AttentionKwargs]) -> dict[str, Callable]:
    attn_name = attn_kwargs.get("attn_name", "sdpa_causal")
    if attn_name not in __type_factory_map:
        # we can add sdpa default here
        raise KeyError(
            f"Unknown attention type `{attn_name}`; registered types: "
            f"{sorted(__type_factory_map.keys())}"
        )

    return __type_factory_map[attn_name]


class QKV(nn.Module, metaclass=abc.ABCMeta):
    """Simple module for applying qkv in attention"""

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_kq_per_head = emb_kq_per_head
        self.emb_v_per_head = emb_v_per_head
        self.use_bias = use_bias
        self.linear_config = linear_config

    @abc.abstractmethod
    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """applies query/key/value transformations on q, k, v inputs respectively and returns the resulting values

        Args:
            q: torch.Tensor
                the query tensor
            k: Optional[torch.Tensor]
                the optional key tensor
            v: Optional[torch.Tensor]
                the optional value tensor

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            the query, key, and value computed
        """
        pass

    @abc.abstractmethod
    def reset_parameters(self):
        """resets the query, key, and value weights for training

        Args:
            gain: int
                gain for std in norm (default is 1)
        """
        pass


class UnfusedQKV(QKV):
    """
    Unfused Weights implementation of QKV
    """

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
            linear_config,
            *args,
            **kwargs,
        )

        self.query = get_linear(
            self.emb_dim,
            self.nheads * self.emb_kq_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.key = get_linear(
            self.emb_dim,
            self.kvheads * self.emb_kq_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.value = get_linear(
            self.emb_dim,
            self.kvheads * self.emb_v_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if k is None and v is None:
            k = q
            v = q
        elif k is None or v is None:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        # b x h x qlen x ds
        queries = self.query(q)
        keys = self.key(k)
        values = self.value(v)
        return queries, keys, values


class FusedQKV(QKV):
    """
    Fused Weights implementation of QKV
    """

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
            linear_config,
            *args,
            **kwargs,
        )
        self.splits = [
            self.nheads * self.emb_kq_per_head,
            self.kvheads * self.emb_kq_per_head,
            self.kvheads * self.emb_v_per_head,
        ]

        self.qkv_fused = get_linear(
            self.emb_dim,
            sum(self.splits),
            bias=self.use_bias,
            linear_config=linear_config,
        )

    def unfuse_weights(self):
        with torch.device("meta"):
            result = UnfusedQKV(
                self.emb_dim,
                self.nheads,
                self.kvheads,
                self.emb_kq_per_head,
                self.emb_v_per_head,
                self.use_bias,
            )
        query, key, value = torch.split(self.qkv_fused.weight, self.splits, dim=0)
        result.query.weight = torch.nn.Parameter(query)
        result.key.weight = torch.nn.Parameter(key)
        result.value.weight = torch.nn.Parameter(value)
        if self.use_bias:
            query_bias, key_bias, value_bias = torch.split(
                self.qkv_fused.bias, self.splits, dim=0
            )
            result.query.bias = torch.nn.Parameter(query_bias)
            result.key.bias = torch.nn.Parameter(key_bias)
            result.value.bias = torch.nn.Parameter(value_bias)
        return result

    def reset_parameters(self):
        nn.init.trunc_normal_(self.qkv_fused.weight, mean=0.0, std=0.02)
        if self.qkv_fused.bias is not None:
            self.qkv_fused.bias.data.zero_()

    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (k is None and v is None) or (k is q and v is q):
            qkv = q
        else:
            raise ValueError("q, k, and v must be the same or k and v must be None")
        return self.qkv_fused(qkv).split(self.splits, dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    ...
    Args
    ----
    emb_dim : int
        Latent dimensionality of input and output tensors.
    emb_kq : int
        Latent dimensionality of each head in key and query projections (attention dimension).
    emb_v : int
        Latent dimensionality of each head in value projection (mixing dimension).
    nheads : int
        Number of attention heads.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    fused : bool
        If True, qkv weights will be fused, otherwise qkv weights will be unfused.
    linear_config : Mapping[str, Any] | None
        Configuration for selection of linear modules (QKV, dense).
        Pass as {"linear_type": [str | callable], <other kwargs>}.
        "linear_type" should provide the string identifier of a registered type
        (e.g., "torch_linear", "gptq", ...) or a callable for module selection depending
        on module name. Additional config options should be provided as kwargs in
        linear_config.
    use_sinks : bool
        If True, add a learned attention sink: one trainable logit per query
        head (zero-init), entering the softmax denominator with value 0
        (gpt-oss style). Routes attention through flex_attention (torch >= 2.5)
        so the sink needs no logits recomputation; incompatible with attention
        dropout and user-supplied masks.
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
        use_sinks: bool = False,
    ):
        super(MultiHeadAttention, self).__init__()
        if nheads <= 0 or kvheads <= 0:
            raise ValueError(
                f"nheads and kvheads must be positive, got nheads={nheads}, kvheads={kvheads}"
            )
        if nheads % kvheads != 0:
            raise ValueError(
                f"nheads must be divisible by kvheads, got nheads={nheads}, kvheads={kvheads}"
            )
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.fused = fused
        self.linear_config = linear_config
        self.scale_factor = scale_factor

        if use_sinks:
            if not flex_utils.flex_attention_available:
                raise ImportError(
                    "use_sinks requires torch.nn.attention.flex_attention "
                    "(torch >= 2.5) to obtain the row logsumexp without "
                    "recomputing attention logits"
                )
            if self.p_dropout:
                raise ValueError(
                    "use_sinks is incompatible with attention dropout "
                    "(flex_attention has no dropout); use p_dropout=0"
                )
            # gpt-oss-style learned attention sink: one logit per query head,
            # an extra softmax entry with value 0. Zero-init, trained through
            # the LM loss.
            self.sinks: Optional[nn.Parameter] = nn.Parameter(torch.zeros(nheads))
        else:
            self.sinks = None

        self.in_proj: QKV = (FusedQKV if self.fused else UnfusedQKV)(
            self.emb_dim,
            self.nheads,
            self.kvheads,
            self.emb_kq_per_head,
            self.emb_v_per_head,
            self.use_bias,
            linear_config=linear_config,
        )

        self.gate_proj = get_linear(
            self.emb_dim,
            self.nheads * self.emb_v_per_head,
            bias=False,
            linear_config=linear_config,
        )

        self.dense = get_linear(
            self.nheads * self.emb_v_per_head,
            self.emb_dim,
            bias=use_bias,
            linear_config=linear_config,
        )

        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        self.position_encoder = position_encoder

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, QKV):
                m.reset_parameters()
        if self.sinks is not None:
            self.sinks.data.zero_()

    def to_tp(self, group: ProcessGroup) -> "TPMultiHeadAttention":
        return TPMultiHeadAttention.import_module(self, group)

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
        past_key_value_state: tuple
            the cache to be used in attention of the form (<self/cross>_key, <self/cross>_value)
        position_ids: Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. Used for RoPE embeddings
        use_cache: bool
            if True, the kv states for self/cross attention will be saved, otherwise they will not be saved

        Returns
        -------
        tensor or tuple
            If use_cache=False, only the hidden state will be returned as a tensor. If use_cache=True, a tuple will be
            returned in the form (hidden_state, cache) where hidden_state is a tensor and cache is of the form specified
            in past_key_value_state
        """
        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()

        # if this is self attention, we always recompute
        # cross attention only gets computed when a cache does not exist
        # if we dont have the cache yet, we need to compute
        # d x (h x ds)
        # b x kvlen x d
        # b x kvlen x h x ds
        # b x h x kvlen x ds
        # todo: Cross attention (This always is true for now)
        q_out, k_out, v_out = self.in_proj(q, k, v)

        # note: transposes will be moved in a later PR to fix dis-contiguous tensor issues
        queries = q_out.view(batch_size, q_len, self.nheads, self.emb_kq_per_head)
        keys = k_out.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head)
        values = v_out.view(batch_size, q_len, self.kvheads, self.emb_v_per_head)

        # You want to apply rotary embeddings pre-cache
        if self.position_encoder is not None:
            queries, keys = self.position_encoder.adjusted_qk(
                queries, keys, position_ids, past_key_value_state, use_cache
            )

        # Learned sinks need the row logsumexp, which the SDPA ops cannot
        # expose without recomputing logits — route through the flex op, which
        # yields it as a free byproduct of its streaming softmax.
        if self.sinks is not None:
            attn_name = attn_kwargs.get("attn_name", "flex_causal")
            if attn_name not in ("sdpa_causal", "flex_causal"):
                raise ValueError(
                    "use_sinks only supports the flex_causal attention op, "
                    f"got attn_name=`{attn_name}`"
                )
            attn_kwargs["attn_name"] = "flex_causal"
            attn_kwargs["sinks"] = self.sinks

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
            # Normalize K/V to b x kvheads x seq_len x ds here so compute ops
            # never have to infer the layout from tensor shapes: when
            # seq_len == kvheads, b x seq_len x kvheads x ds and
            # b x kvheads x seq_len x ds have identical shapes, so a shape-based
            # guess can silently swap the token and head axes. Store ops uphold
            # the same b x kvheads x seq_len x ds contract on the cached path.
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

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            return out, (keys_return, values_return)
        else:
            return out


class TPMultiHeadAttention(MultiHeadAttention, TPModule):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    This subclass adds support for Tensor Parallel
    ...
    Args
    ----
    Check MultiHeadAttention for up-to-date docs

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
        MultiHeadAttention.__init__(
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
            use_sinks=use_sinks,
        )
        self.pre_tp_nheads = nheads
        self.pre_tp_kvheads = kvheads
        self.setup_tp(rank, group)

        # linear_type must handle module_name = None to support TP of MHA
        self.linear_type = get_linear_type(self.linear_config)

    def load_weights(
        self,
        tensor_values: dict[str, torch.Tensor],
    ) -> Optional[set]:
        """Define sharding info of MHA module as:
        {'module_name': (module_obj, sharding_dim, max_partition)}
        Then, call the pre-registered sharding function associated with
        self.linear_type.

        `sharding_dim` is sharding dimension of the `weights` parameter
        of nn.Linear. It may differ for other types of linear or other
        parameters.

        The numbers in `max_partition` signify the largest world size
        till we need to duplicate. For instance if we have nheads=16 and
        world_size=32, then first 2 ranks will get first 1/16th of query
        """

        # The learned sink logits are a plain per-query-head parameter, not a
        # linear module — shard them with the same query-head partition as
        # gate_proj (dim 0, blocked by pre-TP head count) before delegating
        # the linear weights to the registered sharding map.
        if self.sinks is not None:
            tensor_values = dict(tensor_values)
            sinks_used: set = set()
            sinks_value = self._get_sd_weight(tensor_values, sinks_used, ["sinks"])
            self.sharded_copy(self.sinks, sinks_value, 0, [self.pre_tp_nheads])
            for used_key in sinks_used:
                del tensor_values[used_key]

        if self.fused:
            module_sharding_info = {
                "qkv_fused": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("qkv_fused"),
                    0,
                    [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
                ),
                "gate_proj": LinearModuleShardingInfo(
                    self.gate_proj, 0, [self.pre_tp_nheads]
                ),
                "dense": LinearModuleShardingInfo(self.dense, 1, [self.world_size]),
            }
        else:
            module_sharding_info = {
                "query": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("query"), 0, [self.pre_tp_nheads]
                ),
                "key": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("key"), 0, [self.pre_tp_kvheads]
                ),
                "value": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("value"), 0, [self.pre_tp_kvheads]
                ),
                "gate_proj": LinearModuleShardingInfo(
                    self.gate_proj, 0, [self.pre_tp_nheads]
                ),
                "dense": LinearModuleShardingInfo(self.dense, 1, [self.world_size]),
            }

        type_sharding_map = get_all_linear_type_to_sharding_maps()
        unused_keys = type_sharding_map[self.linear_type](
            tensor_values,
            self,
            module_sharding_info,
        )
        return unused_keys

    @staticmethod
    def import_module(
        mha: MultiHeadAttention, group: ProcessGroup
    ) -> "TPMultiHeadAttention":
        tp_mha = TPMultiHeadAttention(
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
            use_sinks=mha.sinks is not None,
        )
        return tp_mha

    def _copy_to_tp_region(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ):
        if (k is None and v is None) or (k is q and v is q):
            q_par = copy_to_tensor_model_parallel_region(q, self.group)
            if self.fused:
                k_par = None
                v_par = None
            else:
                k_par = copy_to_tensor_model_parallel_region(k, self.group)
                v_par = copy_to_tensor_model_parallel_region(v, self.group)
        else:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        return q_par, k_par, v_par

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
        Check MultiHeadAttention for up-to-date arguments and docs
        """

        q_par, k_par, v_par = self._copy_to_tp_region(q, k, v)

        out_par = MultiHeadAttention.forward(
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


def get_attention(
    attn_type: str,
    *args,
    window_size: int = 512,
    **kwargs,
) -> MultiHeadAttention:
    """Construct an attention layer by type name.

    attn_type: "attn" / "full_attention" for MultiHeadAttention,
        "swa" / "sliding_window" for the flex-attention-backed
        SlidingWindowMultiHeadAttention.
    window_size: sliding-window size; only used for the swa types.

    Remaining args/kwargs are forwarded to the class constructor (see
    MultiHeadAttention for the shared signature).
    """
    if attn_type in ("attn", "full_attention"):
        return MultiHeadAttention(*args, **kwargs)
    if attn_type in ("swa", "sliding_window"):
        # local import to avoid a circular dependency at module load
        from fms.modules.sliding_window_attention import (
            SlidingWindowMultiHeadAttention,
        )

        return SlidingWindowMultiHeadAttention(
            *args, sliding_window=window_size, **kwargs
        )
    raise ValueError(
        f"Unknown attention type `{attn_type}`; expected one of "
        "'attn', 'full_attention', 'swa', 'sliding_window'"
    )
