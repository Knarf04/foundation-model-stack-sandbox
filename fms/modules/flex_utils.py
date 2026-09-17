"""Shared flex_attention machinery for the attention modules.

Kept free of any fms imports so both fms.modules.attention and
fms.modules.sliding_window_attention can depend on it without cycles.
Requires torch >= 2.5 for the flex paths; `flex_attention_available` gates all
callers.
"""

import functools
import inspect
from typing import Callable, Optional, Tuple

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    flex_attention_available: bool = True
except ImportError:
    flex_attention_available = False

# Version-adaptive lse retrieval: return_lse is deprecated (slated for removal
# in torch 2.10) in favor of return_aux=AuxRequest(lse=True) (torch >= 2.9).
AuxRequest: Optional[type] = None
_flex_supports_return_aux: bool = False
if flex_attention_available:
    try:
        from torch.nn.attention.flex_attention import AuxRequest

        _flex_supports_return_aux = (
            "return_aux" in inspect.signature(flex_attention).parameters
        )
    except ImportError:
        pass

# torch's flex_attention lowering rejects head dims below 16:
#   "NYI: embedding dimension of the query, key, and value must be at least 16"
FLEX_MIN_HEADDIM = 16

# Block-sparse callers (telescoping) make a fresh shape per schedule under
# dynamic=False; past dynamo's default limit of 8 it falls back to an unfused
# eager path that materializes the full scores matrix and OOMs at large N.
if flex_attention_available:
    torch._dynamo.config.recompile_limit = max(
        getattr(torch._dynamo.config, "recompile_limit", 0), 64
    )

# One compiled flex per `dynamic` setting: dynamic=True for shape-stable
# callers, dynamic=False where every schedule is its own shape anyway.
_compiled_flex_attention: dict = {}
# (device_type, head_dim_qk, head_dim_v) signatures inductor cannot lower --
# e.g. head dims below 16 ("NYI: embedding dimension ... must be at least 16").
# Whether a shape compiles is a property of the shape, not of the process, so
# this must be remembered per signature: a single global compiled/eager flag
# makes the backend depend on whichever shape happened to run first.
_eager_only_signatures: set = set()


def _flex_signature(query: torch.Tensor, value: torch.Tensor) -> tuple:
    return (query.device.type, query.shape[-1], value.shape[-1])


def _compiled_flex(dynamic: bool) -> Callable:
    if dynamic not in _compiled_flex_attention:
        _compiled_flex_attention[dynamic] = torch.compile(
            flex_attention, dynamic=dynamic
        )
    return _compiled_flex_attention[dynamic]


def _must_run_eager(signature: tuple) -> bool:
    """CPU, or a shape inductor has already refused to lower. See
    call_flex_attention for why CPU is never compiled here."""
    return signature[0] == "cpu" or signature in _eager_only_signatures


def call_flex_attention(query, key, value, *, dynamic: bool = True, **kwargs):
    """Call flex_attention through torch.compile where that is both available
    and correct, else eager.

    Eager flex_attention materializes the full score matrix, so compilation is
    what actually delivers the fused sparse kernel. Two cases force eager:

    * CPU. The CPU inductor lowering returns all-NaN for a mask_mod that
      captures a tensor (as composed user masks do) instead of failing, so the
      compiled path there is not trustworthy. CPU is not a throughput path.
    * Signatures inductor refuses to lower (head dim < 16). These fall back on
      first use and are remembered, so the choice is per shape rather than a
      process-wide flag set by whichever shape ran first.

    A genuine caller bug (bad shapes, invalid block mask) raises from eager too
    and is propagated without being recorded, so it cannot poison the backend
    selection for later, fixed callers."""
    signature = _flex_signature(query, value)
    if _must_run_eager(signature):
        return flex_attention(query, key, value, **kwargs)
    try:
        return _compiled_flex(dynamic)(query, key, value, **kwargs)
    except Exception:
        # Retry eagerly to tell a lowering failure from a real invocation bug;
        # if eager also raises, propagate and record nothing.
        result = flex_attention(query, key, value, **kwargs)
        _eager_only_signatures.add(signature)
        return result


# enable_gqa lets the kernel broadcast Hkv heads over the query heads instead
# of materializing the expansion: bit-identical, and `expansion` times less KV.
_GQA_FALLBACK_BLOCK_M = 64
_gqa_needs_pin: dict = {}


def _gqa_key(q, k, v) -> tuple:
    return (q.shape[1], q.shape[2], q.shape[3],
            k.shape[1], k.shape[2], v.shape[3], q.dtype)


def flex_attention_gqa(query, key, value, **kwargs):
    """flex_attention with native GQA: query may carry more heads than key and
    value, and the kernel broadcasts rather than materializing the expansion.

    Compiled with dynamic=False -- block-sparse callers give every schedule its
    own shape, so there is nothing for dynamic shapes to reuse.

    Where enable_gqa leaves inductor no autotune choice, retry with a pinned
    BLOCK_M. The failing window depends on (Q_LEN, head dim, ratio), so it is
    discovered per shape; pinning globally costs 3x on the Q_LEN=1 decode
    shape. Any other lowering failure falls back to eager exactly as
    call_flex_attention does."""
    kwargs = {**kwargs, "enable_gqa": True}
    signature = _flex_signature(query, value)
    if _must_run_eager(signature):
        return flex_attention(query, key, value, **kwargs)

    gqa_signature = _gqa_key(query, key, value)
    pinned = {"kernel_options": {"BLOCK_M": _GQA_FALLBACK_BLOCK_M}}
    if _gqa_needs_pin.get(gqa_signature):
        kwargs = {**kwargs, **pinned}
    try:
        return _compiled_flex(False)(query, key, value, **kwargs)
    except Exception as exc:
        if "NoValidChoicesError" in str(exc) and not _gqa_needs_pin.get(
            gqa_signature
        ):
            _gqa_needs_pin[gqa_signature] = True
            return _compiled_flex(False)(query, key, value, **kwargs, **pinned)
        result = flex_attention(query, key, value, **kwargs)
        _eager_only_signatures.add(signature)
        return result


def pad_head_dims(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_dim_qk: int,
    head_dim_v: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Zero-pad the feature dim up to flex's minimum head dim.
    -> (query, key, value, qk_pad, v_pad).

    Exact: padded lanes contribute 0 to every q.k. Callers must pin the scale
    to the TRUE head dim, and apply positional encoding BEFORE this --
    rotate-half splits at D/2 and would otherwise mix real lanes with pad
    lanes."""
    qk_pad = max(0, FLEX_MIN_HEADDIM - head_dim_qk)
    v_pad = max(0, FLEX_MIN_HEADDIM - head_dim_v)
    if qk_pad:
        query = torch.nn.functional.pad(query, (0, qk_pad))
        key = torch.nn.functional.pad(key, (0, qk_pad))
    if v_pad:
        value = torch.nn.functional.pad(value, (0, v_pad))
    return query, key, value, qk_pad, v_pad


def sliding_window_mask_mod(offset: int, window_size: int) -> Callable:
    """mask_mod for causal sliding-window attention: query at cache-offset
    position `q_idx + offset` attends to keys in (q_pos - window_size, q_pos]."""

    def mask_mod(b, h, q_idx, kv_idx):
        q_pos = q_idx + offset
        return (kv_idx <= q_pos) & (kv_idx > q_pos - window_size)

    return mask_mod


def causal_mask_mod(offset: int) -> Callable:
    """mask_mod for bottom-right-aligned causal attention: query at
    cache-offset position `q_idx + offset` attends to keys [0, q_pos]."""

    def mask_mod(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx + offset

    return mask_mod


@functools.lru_cache(maxsize=32)
def get_sliding_window_block_mask(
    q_len: int, k_len: int, window_size: int, device: str
):
    """Build (and cache) a causal sliding-window BlockMask. Cache is keyed on
    shapes only, so all layers of a model share one mask per step."""
    return create_block_mask(
        sliding_window_mask_mod(k_len - q_len, window_size),
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=k_len,
        device=device,
    )


@functools.lru_cache(maxsize=32)
def get_causal_block_mask(q_len: int, k_len: int, device: str):
    """Build (and cache) a bottom-right-aligned causal BlockMask."""
    return create_block_mask(
        causal_mask_mod(k_len - q_len),
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=k_len,
        device=device,
    )


def normalize_attention_mask(
    mask: torch.Tensor, batch_size: int, q_len: int, k_len: int
) -> torch.Tensor:
    """Normalize a user attention mask to (batch_size, q_len, k_len), bottom-
    right aligned. Accepts (q, k), (b, q, k), or (b, 1, q, k); True/0 = attend.

    Rows/columns beyond (q_len, k_len) are sliced off the top/left: with a
    trimmed SWA cache the cache holds the most recent k_len keys, so the last
    k_len mask columns are the ones that correspond to it."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4:
        if mask.shape[1] != 1:
            raise ValueError(
                "per-head attention masks are not supported on the flex path; "
                f"got mask of shape {tuple(mask.shape)}"
            )
        mask = mask.squeeze(1)
    elif mask.dim() != 3:
        raise ValueError(f"unsupported attention mask rank: {tuple(mask.shape)}")
    if mask.shape[-2] < q_len or mask.shape[-1] < k_len:
        raise ValueError(
            f"attention mask of shape {tuple(mask.shape)} is smaller than "
            f"(q_len={q_len}, k_len={k_len})"
        )
    mask = mask[:, -q_len:, -k_len:]
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1)
    return mask.contiguous()


def compose_block_mask(
    base_mask_mod: Optional[Callable],
    user_mask: torch.Tensor,
    q_len: int,
    k_len: int,
    device: str,
):
    """BlockMask for `base & user`: allowed(b,q,k) = base(q,k) AND user[b,q,k].

    user_mask must be bool (batch, q_len, k_len), True = attend. Built per call
    (the captured tensor prevents shape-keyed caching); pass base_mask_mod=None
    when the base allows everything (e.g. a single decode query)."""
    if base_mask_mod is None:

        def mask_mod(b, h, q_idx, kv_idx):
            return user_mask[b][q_idx][kv_idx]
    else:

        def mask_mod(b, h, q_idx, kv_idx):
            return base_mask_mod(b, h, q_idx, kv_idx) & user_mask[b][q_idx][kv_idx]

    return create_block_mask(
        mask_mod,
        B=user_mask.shape[0],
        H=None,
        Q_LEN=q_len,
        KV_LEN=k_len,
        device=device,
    )


def additive_score_mod(bias: torch.Tensor) -> Callable:
    """score_mod adding a float mask (batch, q_len, k_len) to the logits
    (0 = attend, large negative = masked). The lse then includes the bias, so
    sink gating composes with it for free."""

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias[b][q_idx][kv_idx]

    return score_mod


def apply_sink_gate(
    attn: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """Scale attention output by the learned-sink gate sigmoid(lse - s_h).

    Mathematically equivalent to an extra softmax entry with logit s_h and
    value 0: the effective weights become e^{l_ij} / (e^{s_h} + sum_k e^{l_ik}).

    attn: b x h x q_len x ds, lse: b x h x q_len (natural-log logsumexp of the
    masked scaled logits, as returned by flex_attention's return_lse),
    sinks: (h,) per-query-head logits.
    """
    gate = torch.sigmoid(lse - sinks.float().view(1, -1, 1))
    return attn * gate.unsqueeze(-1).to(attn.dtype)


def _flex_attention_with_lse(queries, keys, values, **flex_kwargs):
    """Run flex_attention and also return the row logsumexp, using whichever
    API this torch provides: return_aux=AuxRequest(lse=True) on newer torch,
    the deprecated return_lse=True on older."""
    if _flex_supports_return_aux:
        attn, aux = call_flex_attention(
            queries, keys, values, return_aux=AuxRequest(lse=True), **flex_kwargs
        )
        return attn, aux.lse
    attn, lse = call_flex_attention(
        queries, keys, values, return_lse=True, **flex_kwargs
    )
    return attn, lse


def flex_attention_with_sinks(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_mask,
    scale: Optional[float],
    enable_gqa: bool,
    sinks: Optional[torch.Tensor],
    score_mod: Optional[Callable] = None,
) -> torch.Tensor:
    """flex_attention with an optional learned sink, without recomputing any
    logits: the row logsumexp comes back as a byproduct of the streaming
    softmax and the output is renormalized by sigmoid(lse - s_h). An optional
    score_mod (e.g. an additive float mask) is folded into the logits, so the
    lse — and therefore the sink gate — accounts for it automatically.

    Stock PyTorch flex_attention has no native sink argument; this LSE
    renormalization is the canonical implementation (it is also how Hugging
    Face's FlexAttention backend realizes gpt-oss-style sinks). The lse
    participates in flex's backward interface, so the sink trains normally."""
    flex_kwargs = dict(
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    if sinks is None:
        return call_flex_attention(queries, keys, values, **flex_kwargs)
    if queries.device.type == "cpu":
        # The CPU flex backend does not expose the lse (and has no backward),
        # so sink gating cannot run there. Fail early with a clear message
        # instead of deep inside the kernel.
        raise RuntimeError(
            "attention sinks need the row logsumexp from flex_attention, "
            "which the CPU flex backend does not provide; run sink-enabled "
            "attention on CUDA"
        )
    attn, lse = _flex_attention_with_lse(queries, keys, values, **flex_kwargs)
    return apply_sink_gate(attn, lse, sinks)
