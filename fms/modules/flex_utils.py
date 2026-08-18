"""Shared flex_attention machinery for the attention modules.

Kept free of any fms imports so both fms.modules.attention and
fms.modules.sliding_window_attention can depend on it without cycles.
Requires torch >= 2.5 for the flex paths; `flex_attention_available` gates all
callers.
"""

import functools
import inspect
from typing import Callable, Optional

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

_flex_attention_fn: Optional[Callable] = None


def call_flex_attention(*args, **kwargs):
    """Call flex_attention through torch.compile (compiled lazily, once).
    Eager flex_attention materializes the full score matrix, so compilation is
    what actually delivers the fused sparse kernel; if compilation is
    unavailable in this environment, fall back to eager.

    The eager fallback is only cached when eager actually succeeds: a genuine
    caller bug (bad shapes, invalid block mask) raises from both paths and must
    not permanently poison the backend selection for later, fixed callers."""
    global _flex_attention_fn
    if _flex_attention_fn is None:
        compiled = torch.compile(flex_attention)
        try:
            result = compiled(*args, **kwargs)
        except Exception:
            # Retry in eager to distinguish a compilation failure from a real
            # invocation bug; if eager also raises, propagate without caching.
            result = flex_attention(*args, **kwargs)
            _flex_attention_fn = flex_attention
            return result
        _flex_attention_fn = compiled
        return result
    return _flex_attention_fn(*args, **kwargs)


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
