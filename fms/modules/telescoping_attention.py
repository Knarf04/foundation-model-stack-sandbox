"""Telescoping-cache attention, backed by flex_attention.

Instead of one flat KV cache, keys/values are merged into a dyadic summary
tree: level l node k summarizes tokens [k*2^l, (k+1)*2^l), and a schedule
(`fmap`) says when each level becomes visible. A query attends to one
contiguous interval per level, so recent tokens are seen at full resolution and
older ones only through progressively coarser summaries -- bounded cache,
unbounded context.

The algorithm lives in fms.modules.telescoping_attn (the extracted reference
building blocks plus the flex kernels); this module is the FMS layer around it.
Requires torch >= 2.5 and, in practice, CUDA -- flex_attention is the only
backend here, there is no dense fallback.

Unlike full attention and SWA, the cache returned under use_cache=True is a
DecodeState (per-level rings + merge weights + conv state + token counter), not
a (keys, values) tensor pair: the tree has no flat-tensor representation that
decode can be resumed from. It passes through fms.utils.generation untouched,
the same way SSMCacheUnit does.
"""

from typing import Any, Dict, Mapping, Optional, Tuple

from typing_extensions import Unpack

import torch
import torch.distributed
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask
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
    register_attention_op,
)
from fms.modules.flex_utils import (
    FLEX_MIN_HEADDIM,
    flex_attention_available as _flex_available,
    flex_attention_gqa,
    normalize_attention_mask,
    pad_head_dims,
)
from fms.modules.linear import get_linear, get_linear_type
from fms.modules.positions import PositionEncoder
from fms.modules.telescoping_attn.decode_state import (
    DecodeState,
    advance_decode_state,
    init_decode_state,
)
from fms.modules.telescoping_attn.packing import PackedKV
from fms.modules.telescoping_attn.range_spec import (
    IntOps,
    activation_times_from_fmap,
    level_offsets as packed_level_offsets,
    level_row_bounds,
    range_bounds,
)
from fms.modules.telescoping_attn.packing import pack_levels
from fms.modules.telescoping_attn.position import (
    apply_rope,
    compute_relative_states,
    rope_tables,
)
from fms.modules.telescoping_attn.short_conv import ShortConv1d
from fms.modules.telescoping_attn.sink_gate import apply_attention_sink
from fms.modules.telescoping_attn.summaries import build_dyadic_summaries
from fms.modules.tp import TPModule

# Activation schedule: level l becomes visible fmap[l] queries in. The default
# is the schedule the reference implementation is pinned against.
DEFAULT_FMAP = {1: 64, 2: 72, 3: 80}

WEIGHT_MODES = ("qk", "linear")
POSITION_MODES = ("none", "rope", "relative")


# ---------------------------------------------------------------------------
# FlexAttention over the dyadic tree, with analytic ranges.
#
# Selection is ANALYTIC rather than a materialized mask: for a query q and
# level l the visible entries are the contiguous level-local interval [lo, hi)
# that range_spec computes in closed form from the schedule, so `mask_mod`
# recomputes the predicate from (q_idx, kv_idx) arithmetically and
# `create_block_mask` turns it straight into block sparsity. Two consequences
# beyond taste: visible entries are contiguous per level, so they cluster
# inside blocks instead of scattering; and the positional metadata is closed
# form rather than recovered from a scan plan -- a summary's RoPE position is
# `(local + 1) * 2**level - 1` and its relative bin distance is a pure integer
# function of (q, level, local), so neither needs a table.
#
# Frozen contract, identical to the reference implementation's
# multilevel_attention_forward:
#     out: [B, N, Hq, Dv]
#     lse: [B, N, Hq], float32, natural log, over exactly A(q).
# ---------------------------------------------------------------------------


def _mx(a, b):
    """max() for the IntOps protocol, tolerant of a python-int operand."""
    if isinstance(b, int):
        return torch.clamp(a, min=b)
    return torch.maximum(a, b)


def _mn(a, b):
    if isinstance(b, int):
        return torch.clamp(a, max=b)
    return torch.minimum(a, b)


# The torch binding of range_spec's IntOps: every `max` in those closed forms
# is against a literal 0, so clamp works.
TORCH_OPS = IntOps(minimum=_mn, maximum=_mx)


def validate_position_args(
    position_mode: str,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
    relative_states: Optional[torch.Tensor],
    relative_proj: Optional[torch.Tensor],
) -> None:
    """
    Raises unless the tables for `position_mode` are present and those for the
    other modes absent.
    """
    if position_mode not in ("none", "rope", "relative"):
        raise ValueError(f"unknown position_mode {position_mode!r}")
    if position_mode == "rope" and (rope_cos is None or rope_sin is None):
        raise ValueError("position_mode='rope' needs rope_cos and rope_sin")
    if position_mode != "rope" and (rope_cos is not None
                                    or rope_sin is not None):
        raise ValueError("rope tables given but position_mode is not 'rope'")
    if position_mode == "relative" and (relative_states is None
                                        or relative_proj is None):
        raise ValueError(
            "position_mode='relative' needs relative_states and relative_proj")
    if position_mode != "relative" and (relative_states is not None
                                        or relative_proj is not None):
        raise ValueError(
            "relative tensors given but position_mode is not 'relative'")


def make_mask_mod(activation_times, cache_size: int, offsets):
    """`mask_mod(b, h, q_idx, kv_idx) -> bool`, analytic, no mask tensor.

    A packed index belongs to level l iff offsets[l] <= kv < offsets[l+1];
    within that level it is visible iff its level-local index lies in the
    closed-form interval [lo_l, hi_l) for this query. The level count is a
    trace-time constant, so the loop unrolls.
    """
    a = tuple(activation_times)
    offs = tuple(int(o) for o in offsets)

    def mask_mod(b, h, q_idx, kv_idx):
        bounds = level_row_bounds(q_idx, a, cache_size, TORCH_OPS)
        visible = None
        for level, (lo, hi) in enumerate(bounds):
            local = kv_idx - offs[level]
            in_level = (kv_idx >= offs[level]) & (kv_idx < offs[level + 1])
            hit = in_level & (local >= lo) & (local < hi)
            visible = hit if visible is None else (visible | hit)
        return visible

    return mask_mod


def packed_slot_positions(seq_len, num_levels, offsets, device):
    """RoPE position of every packed slot: the right endpoint of its dyadic
    interval, `(local + 1) * 2**level - 1` -- `summary_token_position` in
    closed form. Level 0 reduces to the token's own index.
    """
    pos = torch.empty(offsets[-1], dtype=torch.long, device=device)
    for level in range(num_levels):
        n_l = seq_len >> level
        local = torch.arange(n_l, dtype=torch.long, device=device)
        pos[offsets[level]:offsets[level + 1]] = ((local + 1) << level) - 1
    return pos


def bin_distances(q_idx, kv_idx, activation_times, cache_size, offsets):
    """0-based chronological bin distance for packed indices.

    q_idx, kv_idx: broadcastable index tensors. -> bin distance, 0 = newest.

    The visible levels partition the past IN ORDER -- coarse is older, fine is
    newer, no interleaving -- so an entry's rank is its position in the
    concatenation (level L, then L-1, ... then level 0), each ascending in j:

        bin = base[l] - (j - lo_l),  base[l] = M - 1 - (entries above level l)

    One subtract per key once the L+1 per-query counts are in hand, instead of
    `relative_bin_distance`'s clamp/ceil sum. That form answers the more
    general "rank of an arbitrary token position"; its clamps are no-ops for
    the visible entries that are the only inputs here. The suite checks this
    against it exhaustively.
    """
    a = tuple(activation_times)
    num_levels = len(a)
    raw = level_row_bounds(q_idx, a, cache_size, TORCH_OPS)
    bounds = [(lo, _mx(hi, lo)) for lo, hi in raw]      # empties as lo == hi
    counts = [hi - lo for lo, hi in bounds]

    total = counts[0]
    for c in counts[1:]:
        total = total + c

    dist = torch.zeros_like(kv_idx)
    coarser = None                                      # entries above level l
    for level in reversed(range(num_levels)):
        lo, _hi = bounds[level]
        base = (total - 1) if coarser is None else (total - 1 - coarser)
        in_level = (kv_idx >= offsets[level]) & (kv_idx < offsets[level + 1])
        local = kv_idx - offsets[level]
        dist = torch.where(in_level, base - (local - lo), dist)
        coarser = counts[level] if coarser is None else coarser + counts[level]
    return dist


_BLOCK_MASK_CACHE: Dict[tuple, object] = {}


def get_block_mask(activation_times, cache_size, offsets, n_q, n_kv, device):
    """Block sparsity for one schedule. Cached: it depends only on the
    schedule and the two lengths, never on the data."""
    key = (tuple(activation_times), cache_size, tuple(offsets),
           n_q, n_kv, str(device))
    bm = _BLOCK_MASK_CACHE.get(key)
    if bm is None:
        bm = create_block_mask(
            make_mask_mod(activation_times, cache_size, offsets),
            B=None, H=None, Q_LEN=n_q, KV_LEN=n_kv, device=device,
        )
        _BLOCK_MASK_CACHE[key] = bm
    return bm


def telescope_flex_forward(
    q: torch.Tensor,
    packed: PackedKV,
    fmap: Dict[int, int],
    cache_size: int,
    softcap: Optional[float] = 20.0,
    *,
    position_mode: str = "none",
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    relative_states: Optional[torch.Tensor] = None,
    relative_proj: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q: [B, N, Hq, Dk]; packed: level-major K/V from `pack_levels`.
    -> (out [B, N, Hq, Dv], lse [B, N, Hq] float32, natural log)

    Matches `reference.multilevel_attention_forward`'s frozen boundary,
    including the keyword-only positional-encoding contract:

      "rope"     post-summary RoPE. Q rotates at its token position, each
                 packed entry at its interval's RIGHT ENDPOINT. V untouched.
      "relative" learned summary-bin bias added to the logit BEFORE softcap.
    """
    validate_position_args(position_mode, rope_cos, rope_sin,
                           relative_states, relative_proj)
    B, N, Hq, Dk = q.shape
    Hkv, Dv = packed.k.shape[2], packed.v.shape[3]
    expansion = Hq // Hkv
    device = q.device
    a = activation_times_from_fmap(fmap)
    num_levels = len(a)
    offsets = packed_level_offsets(N, num_levels)
    cache_len = packed.k.shape[1]
    if offsets[-1] != cache_len:
        raise ValueError(
            f"packed buffer has {cache_len} rows, schedule implies "
            f"{offsets[-1]} -- pass the tight packing from pack_levels")

    # b h n d. The KV heads stay at Hkv: flex_attention broadcasts them over
    # the query heads natively with enable_gqa=True below, so materializing the
    # expansion here would just read `expansion` times more KV for the same
    # result (bit-identical, verified across ratios 1..16 incl. non-powers-of-2).
    q_t = q.transpose(1, 2)
    k_t = packed.k.transpose(1, 2)
    v_t = packed.v.transpose(1, 2)

    # Positional encoding, before the padding below: rotate-half splits at
    # D/2, so rotating a padded tensor would mix real lanes with pad lanes.
    rel_bias = None
    if position_mode == "rope":
        k_pos = packed_slot_positions(N, num_levels, offsets, device)
        q_t = apply_rope(q_t, torch.arange(N, device=device)[None, None, :],
                         rope_cos, rope_sin)
        k_t = apply_rope(k_t, k_pos[None, None, :], rope_cos, rope_sin)
    elif position_mode == "relative":
        # [B, N, Hq, bins] -> [B, Hq, N, bins], indexed by (b, h, q_i) in the
        # score_mod; the bin index itself is recomputed arithmetically.
        rel_logits = torch.einsum(
            "bnhd,dr->bnhr",
            relative_states.float(), relative_proj.float(),
        ).permute(0, 2, 1, 3).contiguous()
        max_bins = rel_logits.shape[-1]
        if max_bins < 1:
            raise ValueError("relative_proj must have >= 1 bins")

    # Zero-padding to flex's minimum head dim is exact: padded lanes add 0 to
    # every q.k, so scores and LSE are unchanged, and the padded V columns
    # come back 0 and are sliced off. The scale must stay 1/sqrt(true Dk).
    q_t, k_t, v_t, dk_pad, dv_pad = pad_head_dims(q_t, k_t, v_t, Dk, Dv)

    block_mask = get_block_mask(a, cache_size, offsets, N, cache_len, device)

    cap = None if softcap is None else float(softcap)
    if position_mode == "relative":
        # X = QK/sqrt(Dk) + b, BEFORE softcap (reference's forward contract).
        def score_mod(score, b, h, q_i, kv_i):
            dist = bin_distances(q_i, kv_i, a, cache_size, offsets)
            bias = rel_logits[b, h, q_i, dist.clamp(0, max_bins - 1)]
            s = score + bias
            return s if cap is None else cap * s.div(cap).tanh()
    elif cap is None:
        score_mod = None
    else:
        def score_mod(score, b, h, q_i, kv_i):
            return cap * score.div(cap).tanh()

    out, lse = flex_attention_gqa(
        q_t, k_t, v_t, block_mask=block_mask, score_mod=score_mod,
        scale=Dk ** -0.5, return_lse=True,
    )
    if dv_pad:
        out = out[..., :Dv]
    return out.transpose(1, 2), lse.transpose(1, 2).float()


# ---------------------------------------------------------------------------
# Incremental decode.
#
# Everything except the attention itself is reused from reference.py: the
# ring cache, its capacities, the short-conv rolling state and the tree
# update all live in DecodeState / advance_decode_state and are backend
# agnostic. Only `multilevel_attention_decode` is replaced here.
#
# Two things differ from the prefill path above:
#   * storage is a RING per level -- node j sits at
#     ring_offsets[l] + j % caps[l] -- so a contiguous node range [lo, hi)
#     becomes one or two runs of slots once it wraps. Slot -> node is still
#     exact: the visible j is the unique value in [lo, hi) congruent to the
#     slot's ring position, so per-slot metadata is recovered, not guessed.
#   * there is ONE query row over sum(caps) <= a few hundred keys, so block
#     sparsity has nothing to exploit. Masking goes through the score_mod as
#     a -inf fill instead of a block_mask, which avoids rebuilding
#     create_block_mask every token -- that construction would otherwise
#     cost far more than the attention it gates.
# ---------------------------------------------------------------------------


def _decode_slot_metadata(state: DecodeState, t: int, validate: bool = False,
                          row: Optional[int] = None):
    """Per-slot visibility / node index for query t, plus the gather-order
    entry list. Small (sum(caps) elements), rebuilt each step because the
    live set moves with t. `row` only labels errors; the tensors are shared
    across rows that sit at the same t."""
    device = state.k.device
    S = state.k.shape[1]
    spec = state.spec(t + 1)

    visible = torch.zeros(S, dtype=torch.bool, device=device)
    slot_node = torch.zeros(S, dtype=torch.long, device=device)
    slot_level = torch.zeros(S, dtype=torch.long, device=device)

    ent_level, ent_j, ent_slot, bounds = [], [], [], []
    for level in range(spec.num_levels):
        lo, hi = range_bounds(spec, t, level)
        bounds.append((lo, hi))
        if lo >= hi:
            continue
        j = torch.arange(lo, hi, device=device, dtype=torch.long)
        slots = state.ring_offsets[level] + torch.remainder(
            j, state.caps[level])
        if validate:
            # A device->host sync per level, per token: it stalls the step and
            # makes the region uncapturable by a CUDA graph. Ring capacities
            # are a property of the schedule, not the data, and are already
            # pinned by test_decode_capacities -- so this is a debug aid, off
            # by default, not a runtime invariant.
            if visible[slots].any():
                raise RuntimeError(
                    f"ring collision at level {level} for query {t}: two live "
                    f"nodes share a slot (capacities are wrong)")
        # `= True` would stage a host scalar and copy it H2D, which stalls the
        # step and makes the region uncapturable by a CUDA graph.
        visible[slots] = torch.ones_like(slots, dtype=torch.bool)
        slot_node[slots] = j
        slot_level[slots] = level
        ent_level.append(torch.full((hi - lo,), level, dtype=torch.long,
                                    device=device))
        ent_j.append(j)
        ent_slot.append(slots)

    if not ent_j:
        where = "" if row is None else f" (row {row})"
        raise RuntimeError(f"query {t} has no visible entries{where}")
    return (visible, slot_node, slot_level, torch.cat(ent_level),
            torch.cat(ent_j), torch.cat(ent_slot), bounds)


def _decode_metadata_batched(state: DecodeState, t_rows):
    """Every row's per-slot visibility and node index, in a fixed number of
    tensor ops -- independent of the batch size and of how many distinct
    positions it holds.

    The per-row loop this replaces was O(distinct positions) small kernel
    launches on a dispatch-bound path. Two facts make the batched form
    possible: `level_row_bounds` is the branch-free closed form and already
    runs on tensors (that is what TORCH_OPS is for), and the slot -> node map
    inverts without a branch -- slot s at level l holds the node congruent to
    its ring index mod cap[l], and the live range is never longer than the
    ring, so `lo + (ring - lo) % cap` is the only candidate.

    -> (visible [B, S] bool, slot_node [B, S], slot_level [S],
        lo [B, L+1], hi [B, L+1])
    """
    dev = state.k.device
    slot_level, slot_ring, cap_of_slot = state.slot_maps()
    t_vec = torch.as_tensor(list(t_rows), device=dev, dtype=torch.long)
    bounds = level_row_bounds(
        t_vec, state.activation_times, state.cache_size, TORCH_OPS)
    lo = torch.stack([p[0] for p in bounds], dim=1)          # [B, L+1]
    hi = torch.stack([p[1] for p in bounds], dim=1)
    lo_s = lo[:, slot_level]                                 # [B, S]
    hi_s = hi[:, slot_level]
    node = lo_s + torch.remainder(slot_ring - lo_s, cap_of_slot)
    visible = node < hi_s
    # A stale slot still yields a congruent node index; zero it so downstream
    # index arithmetic (RoPE table lookup) stays in range. Those lanes are
    # masked out of the softmax anyway.
    return visible, torch.where(visible, node, torch.zeros_like(node)), \
        slot_level, lo, hi


def _batched_bin_distances(node, slot_level, lo, hi, visible):
    """`base[l] - (j - lo_l)` for every slot of every row, vectorized.

    The visible levels partition the past in order (coarse is older), so an
    entry's rank is its offset within its level plus everything coarser.
    """
    counts = (hi - lo).clamp(min=0)                          # [B, L+1]
    total = counts.sum(dim=1, keepdim=True)
    # entries strictly coarser than each level = suffix sum, exclusive
    coarser = counts.flip(1).cumsum(1).flip(1) - counts
    base = total - 1 - coarser                               # [B, L+1]
    dist = base[:, slot_level] - (node - lo[:, slot_level])
    return torch.where(visible, dist, torch.zeros_like(dist))


def _decode_bin_distances(ent_level, ent_j, bounds):
    """Bin distance for every visible decode entry.

    ent_level, ent_j: the enumerated entries. -> bin distance, 0 = newest.

    Same `base[l] - (j - lo_l)` identity as `bin_distances`; here the entries
    are already enumerated, so the per-level constants are host ints and the
    per-entry work is a single subtract.
    """
    counts = [max(hi - lo, 0) for lo, hi in bounds]
    total = sum(counts)
    dist = torch.zeros_like(ent_j)
    coarser = 0
    for level in reversed(range(len(bounds))):
        lo, _hi = bounds[level]
        base = total - 1 - coarser
        dist = torch.where(ent_level == level, base - (ent_j - lo), dist)
        coarser += counts[level]
    return dist


def telescope_flex_decode(
    q_t: torch.Tensor,
    state: DecodeState,
    *,
    softcap: Optional[float] = 20.0,
    position_mode: str = "none",
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    relative_states: Optional[torch.Tensor] = None,
    relative_proj: Optional[torch.Tensor] = None,
    return_stats: bool = True,
):
    """
    FlexAttention counterpart of `reference.multilevel_attention_decode`.

    q_t: [B, 1, Hq, Dk] for query index t = state.t - 1, attending over the
    ring cache. Returns (out [B, 1, Hq, Dv], lse [B, 1, Hq] float32 natural
    log, stats) -- the same three-value contract, with `stats` carrying
    `entries` in gather order plus the aligned `positions` / `distances`.
    """
    validate_position_args(position_mode, rope_cos, rope_sin,
                           relative_states, relative_proj)
    if q_t.dim() != 4 or q_t.shape[1] != 1:
        raise ValueError(f"q_t must be [B, 1, Hq, Dk], got {tuple(q_t.shape)}")
    if min(state.t) < 1:
        raise ValueError("no token has been consumed yet")

    B, _, Hq, Dk = q_t.shape
    Bs, S, Hkv, Dks = state.k.shape
    Dv = state.v.shape[-1]
    if Bs != B or Dks != Dk:
        raise ValueError(
            f"q_t {tuple(q_t.shape)} incompatible with the state's keys "
            f"{tuple(state.k.shape)}")
    if Hq % Hkv != 0:
        raise ValueError(f"Hq={Hq} must be divisible by Hkv={Hkv}.")
    expansion = Hq // Hkv
    device = q_t.device
    uniform = state.uniform_t
    t_rows = [x - 1 for x in state.t]
    t = (uniform - 1) if uniform is not None else max(t_rows)

    # All rows' per-slot visibility and node index in a fixed number of tensor
    # ops, whatever the batch holds. `slot_level` comes back as the static
    # [S] map; visible / slot_node are [B, S].
    visible, slot_node, slot_level, bnd_lo, bnd_hi = _decode_metadata_batched(
        state, t_rows)

    stats = None
    if return_stats:
        # Debug aid only: it wants the entries in gather order, which is the
        # one thing the per-slot form does not give. Host ints throughout, so
        # no device->host sync on the normal path.
        if uniform is None:
            raise ValueError(
                "return_stats is a single-position debug aid; it has no "
                "meaning for rows at different positions"
            )
        _, _, _, ent_level, ent_j, _, bounds = _decode_slot_metadata(state, t)
        entries = [(lv, j) for lv, (lo, hi) in enumerate(bounds)
                   for j in range(lo, hi)]
        stats = {"entries": entries, "positions": [], "distances": []}

    q_bhnd = q_t.transpose(1, 2)                       # [B, Hq, 1, Dk]
    k_bhsd = state.k.transpose(1, 2)                   # [B, Hkv, S, Dk]
    v_bhsd = state.v.transpose(1, 2)                   # broadcast, not expanded

    rel_bias = None
    if position_mode == "rope":
        # A summary is a virtual token at its interval's right endpoint.
        slot_pos = (((slot_node + 1) << slot_level) - 1).clamp(min=0)
        if return_stats:
            stats["positions"] = [((j + 1) << lv) - 1
                                  for lv, j in stats["entries"]]
        if uniform is not None:
            q_pos = torch.full((1, 1, 1), t, device=device, dtype=torch.long)
        else:
            q_pos = torch.as_tensor(
                t_rows, device=device, dtype=torch.long)[:, None, None]
        k_pos = slot_pos[:, None, :]                           # [B, 1, S]
        q_bhnd = apply_rope(q_bhnd, q_pos, rope_cos, rope_sin)
        k_bhsd = apply_rope(k_bhsd, k_pos, rope_cos, rope_sin)
    elif position_mode == "relative":
        rel_logits = torch.einsum(
            "bnhd,dr->bnhr",
            relative_states.float(), relative_proj.float(),
        )                                              # [B, 1, Hq, bins]
        max_bins = rel_logits.shape[-1]

        slot_dist = _batched_bin_distances(          # [B, S]
            slot_node, slot_level, bnd_lo, bnd_hi, visible)
        if return_stats:
            # The only remaining sync, and only when stats are requested.
            stats["distances"] = _decode_bin_distances(
                ent_level, ent_j, bounds).tolist()
        idx = slot_dist.clamp(0, max_bins - 1)
        # each row gathers its own bins out of its own logits
        rel_bias = torch.gather(
            rel_logits[:, 0], 2,
            idx[:, None, :].expand(-1, rel_logits.shape[2], -1),
        )                                              # [B, Hq, S]

    q_bhnd, k_bhsd, v_bhsd, dk_pad, dv_pad = pad_head_dims(
        q_bhnd, k_bhsd, v_bhsd, Dk, Dv)

    cap = None if softcap is None else float(softcap)

    def score_mod(score, b, h, q_i, kv_i):
        s = score
        if rel_bias is not None:
            s = s + rel_bias[b, h, kv_i]
        if cap is not None:
            s = cap * s.div(cap).tanh()
        # Mask last, matching the reference: bias -> softcap -> select.
        # masked_fill takes the sentinel as a kernel argument; building a
        # `torch.tensor(-inf, device=...)` here would stage it on the host and
        # copy H2D on every step, which also blocks CUDA graph capture.
        return s.masked_fill(~visible[b, kv_i], float("-inf"))

    out, lse = flex_attention_gqa(
        q_bhnd, k_bhsd, v_bhsd, score_mod=score_mod, scale=Dk ** -0.5,
        return_lse=True,
    )
    if dv_pad:
        out = out[..., :Dv]
    return out.transpose(1, 2), lse.transpose(1, 2).float(), stats


class TelescopingAttentionKwargs(AttentionKwargs):
    """No per-call kwargs: the schedule is layer state, and the tree defines
    what each query attends to, so there is no mask to pass."""


def _unsupported_op(*args, **kwargs):
    raise NotImplementedError(
        "telescoping attention does not go through the store/compute op "
        "registry: the cache is a summary tree, not a (keys, values) pair, "
        "and building it needs the queries and the pre-projection hidden "
        "state. Use TelescopingMultiHeadAttention, which overrides forward. "
        "The `telescoping` attn_name is registered only so model-level "
        "attn_kwargs validation resolves."
    )


if _flex_available:
    register_attention_op("telescoping", _unsupported_op, _unsupported_op)


class TelescopingMultiHeadAttention(MultiHeadAttention):
    """
    MultiHeadAttention over a dyadic summary tree instead of a flat cache.

    Args
    ----
    Check MultiHeadAttention for the shared args.

    fmap : Mapping[int, int] | None
        Activation schedule {level: first query that sees it}, keys a
        consecutive 1..L. Defaults to DEFAULT_FMAP.
    cache_size : int
        Entries a single query may attend to. Bounds the coarsest level.
    softcap : float | None
        tanh logit softcap, applied before the softmax. None disables it.
    weight_mode : str
        How sibling merge weights are formed: "qk" (logsumexp of the query-key
        logits) or "linear" (a learned projection of the hidden state).
    position_mode : str
        "rope" gives each summary a position at its interval's RIGHT ENDPOINT
        and rotates there; "relative" adds a learned bias keyed on how many
        visible entries back the summary is; "none" leaves summaries
        unpositioned. With "rope" or "relative", position_encoder MUST be None
        -- the layer positions after summarization, so an ordinary pre-cache
        RoPE would rotate the keys a second time.
    rope_theta : float
        RoPE base for position_mode="rope".
    relative_dim : int
        Rank of the learned relative bias (position_mode="relative").
    use_kv_short_conv : bool
        Prepend a causal depthwise conv (+ residual) to the projected K/V.
        Zero-init, so enabling it starts from the unconvolved model.
    kv_conv_kernel_size : int
        Conv width when use_kv_short_conv is set.
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
        fmap: Optional[Mapping[int, int]] = None,
        cache_size: int = 512,
        softcap: Optional[float] = 20.0,
        weight_mode: str = "qk",
        position_mode: str = "rope",
        rope_theta: float = 10_000.0,
        relative_dim: int = 8,
        use_kv_short_conv: bool = False,
        kv_conv_kernel_size: int = 4,
        use_sinks: bool = False,
    ):
        if not _flex_available:
            raise ImportError(
                "TelescopingMultiHeadAttention requires "
                "torch.nn.attention.flex_attention (torch >= 2.5)"
            )
        if p_dropout:
            raise ValueError(
                "TelescopingMultiHeadAttention does not support attention "
                "dropout (flex_attention has no dropout); use p_dropout=0"
            )
        if weight_mode not in WEIGHT_MODES:
            raise ValueError(
                f"weight_mode must be one of {WEIGHT_MODES}, got {weight_mode!r}"
            )
        if position_mode not in POSITION_MODES:
            raise ValueError(
                f"position_mode must be one of {POSITION_MODES}, got "
                f"{position_mode!r}"
            )
        if position_mode != "none" and position_encoder is not None:
            raise ValueError(
                f"position_mode={position_mode!r} positions entries AFTER "
                "summarization; a position_encoder would rotate the keys a "
                "second time. Pass position_encoder=None, or use "
                "position_mode='none' to position before the tree instead."
            )
        if not isinstance(cache_size, int) or cache_size <= 0:
            raise ValueError(f"cache_size must be a positive int, got {cache_size}")

        fmap = DEFAULT_FMAP if fmap is None else dict(fmap)
        if sorted(fmap) != list(range(1, len(fmap) + 1)):
            raise ValueError(
                f"fmap keys must be consecutive 1..L, got {sorted(fmap)}"
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
        self.fmap = fmap
        self.cache_size = cache_size
        self.softcap = softcap
        self.weight_mode = weight_mode
        self.position_mode = position_mode
        self.rope_theta = rope_theta
        self.relative_dim = relative_dim
        self.use_kv_short_conv = use_kv_short_conv
        self.kv_conv_kernel_size = kv_conv_kernel_size

        if weight_mode == "linear":
            # Merge weights as a learned projection of the ORIGINAL hidden
            # state; `.weight` is [kvheads, emb_dim], the layout the builder
            # wants.
            self.w = get_linear(
                emb_dim, kvheads, bias=False, linear_config=linear_config
            )
        if use_kv_short_conv:
            self.k_sconv = ShortConv1d(kvheads * emb_kq, kv_conv_kernel_size)
            self.v_sconv = ShortConv1d(kvheads * emb_v, kv_conv_kernel_size)
        if position_mode == "relative":
            # A table with >= cache_size bins always suffices: a query never
            # sees more than cache_size entries.
            self.rel_w = nn.Parameter(torch.zeros(nheads * relative_dim, emb_dim))
            self.rel_proj = nn.Parameter(torch.zeros(relative_dim, cache_size))

        self._rope_cache: dict = {}

    def reset_parameters(self):
        # MultiHeadAttention handles nn.Linear, QKV and sinks. The conv and the
        # relative-bias tables are the two things it cannot reach: the conv is
        # not a Linear, and rel_w / rel_proj are raw Parameters. Both are
        # zero-init identities, so reset returns the layer to plain telescoped
        # attention.
        super().reset_parameters()
        for m in self.modules():
            if isinstance(m, ShortConv1d):
                m.reset_parameters()
        for name in ("rel_w", "rel_proj"):
            param = getattr(self, name, None)
            if param is not None:
                param.data.zero_()

    def to_tp(self, group: ProcessGroup) -> "TPTelescopingMultiHeadAttention":
        return TPTelescopingMultiHeadAttention.import_module(self, group)

    def _rope(self, n_pos: int, device) -> Tuple[Tensor, Tensor]:
        """(cos, sin) covering at least n_pos positions, cached and grown."""
        cached = self._rope_cache.get(device)
        if cached is None or cached[0].shape[0] < n_pos:
            self._rope_cache[device] = rope_tables(
                max(n_pos, 2 * (cached[0].shape[0] if cached else 0)),
                self.emb_kq_per_head,
                base=self.rope_theta,
                device=device,
            )
        cos, sin = self._rope_cache[device]
        return cos, sin

    def _position_kwargs(self, x: Tensor, n_pos: int) -> dict:
        """Positional tables for the configured mode, for a [B, N, emb] x."""
        if self.position_mode == "rope":
            cos, sin = self._rope(n_pos, x.device)
            return dict(rope_cos=cos, rope_sin=sin)
        if self.position_mode == "relative":
            return dict(
                relative_states=compute_relative_states(
                    x, self.rel_w, self.nheads
                ),
                relative_proj=self.rel_proj,
            )
        return {}

    def _conv_weights(self) -> dict:
        if not self.use_kv_short_conv:
            return {}
        return dict(
            k_conv_weight=self.k_sconv.weight, v_conv_weight=self.v_sconv.weight
        )

    def _project(self, q, k, v, batch_size, q_len):
        q_out, k_out, v_out = self.in_proj(q, k, v)
        queries = q_out.view(batch_size, q_len, self.nheads, self.emb_kq_per_head)
        keys = k_out.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head)
        values = v_out.view(batch_size, q_len, self.kvheads, self.emb_v_per_head)
        return queries, keys, values

    def _finish(self, out, lse, x, batch_size, q_len):
        """Sink, gate and output projection -- shared by prefill and decode."""
        if self.sinks is not None:
            out = apply_attention_sink(out, lse, self.sinks)
        attn = out.reshape(batch_size, q_len, self.nheads * self.emb_v_per_head)
        attn = F.silu(self.gate_proj(x)) * attn
        return self.dense(attn)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_value_state: Optional[DecodeState] = None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        Check MultiHeadAttention for the shared arguments.

        `position_ids` is ignored: positions come from the tree (a query's own
        index, a summary's interval endpoint), not from the caller. With
        use_cache=True the returned cache is a DecodeState, and passing one
        back in continues decoding from it.
        """
        attn_name = attn_kwargs.get("attn_name", "telescoping")
        if attn_name not in ("sdpa_causal", "telescoping"):
            raise ValueError(
                "TelescopingMultiHeadAttention only supports the telescoping "
                f"attention path, got attn_name=`{attn_name}`"
            )
        batch_size, q_len, _ = q.size()

        # A padding mask is handled by REALIGNING, not by masking: gather each
        # row's real tokens to the front so real token i sits at absolute
        # position i. The dyadic tree is anchored at position 0, so this is
        # what makes the result independent of how much padding a row carries
        # -- left padding would otherwise regroup every token into different
        # summaries. Trailing pad positions then need no masking at all: a node
        # is first attendable at a query at or after its own last covered
        # token (a[l] >= 2^l - 1), so a node straddling the real/pad boundary
        # is only ever visible to a pad query, whose output is discarded.
        order, n_real = self._padding_order(
            attn_kwargs.get("mask", None), batch_size, q_len
        )
        if order is not None:
            q = self._gather_rows(q, order)

        # The ORIGINAL pre-projection hidden state: merge weights, relative
        # states and the gate all read it, not the projected Q.
        x = q
        queries, keys, values = self._project(q, k, v, batch_size, q_len)

        if past_key_value_state is None:
            out, lse, state = self._prefill(
                x, queries, keys, values, use_cache=use_cache, n_real=n_real
            )
        else:
            if not isinstance(past_key_value_state, DecodeState):
                raise TypeError(
                    "telescoping attention resumes from the DecodeState it "
                    f"returned, got {type(past_key_value_state).__name__}. A "
                    "(keys, values) cache from another attention type cannot "
                    "be continued here."
                )
            out, lse, state = self._decode(
                x, queries, keys, values, past_key_value_state
            )

        dense_out = self._finish(out, lse, x, batch_size, q_len)
        if order is not None:
            dense_out = self._scatter_rows(dense_out, order)
        return (dense_out, state) if use_cache else dense_out

    # ---- padding realignment -------------------------------------------

    def _padding_order(self, mask, batch_size: int, q_len: int):
        """
        -> (order, n_real): `order` is [B, q_len] long, listing each row's real
        token positions in order followed by its pad positions, or None when
        there is nothing to realign. `n_real` is the per-row real length.

        Only padding masks are accepted. A packing mask (one that varies with
        the query beyond causality) has no realignment that makes it go away,
        and quietly treating it as padding would train on cross-document
        context, so it is rejected.
        """
        if mask is None:
            return None, None
        if mask.dtype != torch.bool:
            raise NotImplementedError(
                "telescoping attention takes a boolean padding mask; an "
                "additive float mask has no meaning for a summary, whose "
                "tokens may carry different biases. Pass True = attend."
            )
        mask = normalize_attention_mask(mask, batch_size, q_len, q_len)

        # Per-token validity, read off the newest query row -- the last query
        # is real whenever anything is.
        valid = mask[:, -1, :]
        causal = torch.ones(q_len, q_len, dtype=torch.bool,
                            device=mask.device).tril()
        expected = valid[:, None, :] & causal[None] & valid[:, :, None]
        if not torch.equal(mask & causal[None], expected):
            raise NotImplementedError(
                "telescoping attention supports padding masks only; this mask "
                "varies with the query beyond causality (packing / document "
                "boundaries), which realignment cannot express. A node spans "
                "a fixed dyadic interval, so a boundary inside one cannot be "
                "masked away after the merge."
            )
        n_real = valid.sum(dim=1)
        if bool((n_real == q_len).all()):
            return None, None                       # nothing padded
        if bool((n_real == 0).any()):
            raise ValueError("every row must contain at least one real token")
        # Stable argsort puts the real positions first, in order, pads after.
        order = torch.argsort(
            (~valid).to(torch.int8), dim=1, stable=True
        )
        return order, n_real

    @staticmethod
    def _gather_rows(t: Tensor, order: Tensor) -> Tensor:
        return t.gather(1, order[..., None].expand(-1, -1, t.shape[-1]))

    @staticmethod
    def _scatter_rows(t: Tensor, order: Tensor) -> Tensor:
        out = torch.empty_like(t)
        out.scatter_(1, order[..., None].expand(-1, -1, t.shape[-1]), t)
        return out

    def _prefill(self, x, queries, keys, values, use_cache: bool, n_real=None):
        """Build the tree over the whole sequence and attend to it."""
        q_len = queries.shape[1]
        weight_kwargs = (
            dict(x=x, w_proj=self.w.weight) if self.weight_mode == "linear" else {}
        )
        k_levels, v_levels, w_levels = build_dyadic_summaries(
            queries,
            keys,
            values,
            len(self.fmap),
            **weight_kwargs,
            **self._conv_weights(),
        )
        packed = pack_levels(k_levels, v_levels)
        out, lse = telescope_flex_forward(
            queries,
            packed,
            self.fmap,
            self.cache_size,
            softcap=self.softcap,
            position_mode=self.position_mode,
            **self._position_kwargs(x, q_len),
        )

        state = None
        if use_cache:
            # Hand off as of the last REAL token, not the last padded slot.
            # The tree is prefix-stable -- node j of level l depends only on
            # tokens below (j+1) * 2^l -- so the first `R >> l` nodes of a
            # padded row's tree ARE the tree of its length-R prefix.
            real_lens = None if n_real is None else [int(n) for n in n_real]
            conv_kwargs = {}
            if self.use_kv_short_conv:
                # init_decode_state wants the RAW pre-conv rows: the ring cache
                # holds post-conv K/V, the conv state holds pre-conv rows.
                conv_kwargs = dict(
                    k_pre_conv=keys,
                    v_pre_conv=values,
                    conv_kernel_size=self.kv_conv_kernel_size,
                )
            state = init_decode_state(
                k_levels,
                v_levels,
                w_levels,
                self.fmap,
                self.cache_size,
                real_lens=real_lens,
                **conv_kwargs,
            )
        return out, lse, state

    def _decode(self, x, queries, keys, values, state: DecodeState):
        """
        Consume the new tokens one at a time: each must enter the tree before
        it is queried. q_len == 1 is the generation case; longer runs loop.
        """
        outs, lses = [], []
        for i in range(queries.shape[1]):
            x_t = x[:, i : i + 1]
            q_t = queries[:, i : i + 1]
            weight_kwargs = (
                dict(x_t=x_t, w_proj=self.w.weight)
                if self.weight_mode == "linear"
                else {}
            )
            advance_decode_state(
                state,
                q_t,
                keys[:, i : i + 1],
                values[:, i : i + 1],
                **weight_kwargs,
                **self._conv_weights(),
            )
            out_t, lse_t, _ = telescope_flex_decode(
                q_t,
                state,
                softcap=self.softcap,
                position_mode=self.position_mode,
                return_stats=False,
                # the table must cover the furthest-along row
                **self._position_kwargs(x_t, max(state.t)),
            )
            outs.append(out_t)
            lses.append(lse_t)
        out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        lse = torch.cat(lses, dim=1) if len(lses) > 1 else lses[0]
        return out, lse, state


class TPTelescopingMultiHeadAttention(TelescopingMultiHeadAttention, TPModule):
    """
    TelescopingMultiHeadAttention with Tensor Parallel support.

    Args
    ----
    Check TelescopingMultiHeadAttention for up-to-date docs

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
        fmap: Optional[Mapping[int, int]] = None,
        cache_size: int = 512,
        softcap: Optional[float] = 20.0,
        weight_mode: str = "qk",
        position_mode: str = "rope",
        rope_theta: float = 10_000.0,
        relative_dim: int = 8,
        use_kv_short_conv: bool = False,
        kv_conv_kernel_size: int = 4,
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
        TelescopingMultiHeadAttention.__init__(
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
            fmap=fmap,
            cache_size=cache_size,
            softcap=softcap,
            weight_mode=weight_mode,
            position_mode=position_mode,
            rope_theta=rope_theta,
            relative_dim=relative_dim,
            use_kv_short_conv=use_kv_short_conv,
            kv_conv_kernel_size=kv_conv_kernel_size,
            use_sinks=use_sinks,
        )
        # load_weights below shards in_proj / gate_proj / dense only. The
        # telescoping-only parameters are per-KV-head or per-query-head and so
        # would need their own sharding rules; without them a checkpoint would
        # load unsharded and silently compute the wrong thing.
        unsharded = [
            name
            for name, on in (
                ("weight_mode='linear' (w)", weight_mode == "linear"),
                ("position_mode='relative' (rel_w/rel_proj)",
                 position_mode == "relative"),
                ("use_kv_short_conv (k_sconv/v_sconv)", use_kv_short_conv),
            )
            if on
        ]
        if unsharded:
            raise NotImplementedError(
                "tensor parallelism for telescoping attention covers only the "
                "shared projections; no sharding rules exist yet for "
                + ", ".join(unsharded)
            )
        self.pre_tp_nheads = nheads
        self.pre_tp_kvheads = kvheads
        self.setup_tp(rank, group)

        # linear_type must handle module_name = None to support TP of MHA
        self.linear_type = get_linear_type(self.linear_config)

    # Sharding layout and TP input replication match the full attention TP
    # module: same in_proj / gate_proj / dense submodules.
    load_weights = TPMultiHeadAttention.load_weights
    _copy_to_tp_region = TPMultiHeadAttention._copy_to_tp_region

    @staticmethod
    def import_module(
        mha: TelescopingMultiHeadAttention, group: ProcessGroup
    ) -> "TPTelescopingMultiHeadAttention":
        return TPTelescopingMultiHeadAttention(
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
            fmap=mha.fmap,
            cache_size=mha.cache_size,
            softcap=mha.softcap,
            weight_mode=mha.weight_mode,
            position_mode=mha.position_mode,
            rope_theta=mha.rope_theta,
            relative_dim=mha.relative_dim,
            use_kv_short_conv=mha.use_kv_short_conv,
            kv_conv_kernel_size=mha.kv_conv_kernel_size,
            use_sinks=mha.sinks is not None,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_value_state: Optional[DecodeState] = None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        Check TelescopingMultiHeadAttention for up-to-date arguments and docs
        """
        q_par, k_par, v_par = self._copy_to_tp_region(q, k, v)

        out_par = TelescopingMultiHeadAttention.forward(
            self,
            q_par,
            k_par,
            v_par,
            position_ids,
            past_key_value_state,
            use_cache,
            **attn_kwargs,
        )

        # Only the output is reduced; the cache stays thread-local.
        if use_cache:
            out = reduce_from_tensor_model_parallel_region(out_par[0], self.group)
            return out, out_par[1]
        out = reduce_from_tensor_model_parallel_region(out_par, self.group)
        return out
