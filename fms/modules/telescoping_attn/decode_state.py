"""Incremental decode state: per-level rings, capacities, tree update."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from fms.modules.telescoping_attn.checks import check_paired, check_shape
from fms.modules.telescoping_attn.range_spec import (
    RangeSpec,
    activation_times_from_fmap,
    prefix_offsets,
    range_bounds,
)
from fms.modules.telescoping_attn.short_conv import (
    conv_history,
    short_conv_step,
)
from fms.modules.telescoping_attn.summaries import (
    compute_linear_weights,
    compute_summary_weights,
    merge_children,
)

@dataclass
class DecodeState:
    """
    One layer's decode cache, as level-major rings:

        k [B, sum(caps), Hkv, Dk]   UNROTATED, post-conv keys
        v [B, sum(caps), Hkv, Dv]
        w [B, sum(caps), Hkv, 1]    merge log-weights
        conv_k / conv_v [B, K-1, Hkv*D] | None   last K-1 PRE-conv flat rows
        t               [B] host ints, tokens consumed per row == that row's
                        next query index

    Node j of level l lives at ring_slot(state, l, j). Keys are unrotated
    because the merge re-reads them and RoPE rotates after aggregation.

    `t` is per row so a batch of prompts of differing length can decode
    together, and it is deliberately HOST state (plain ints, not a device
    tensor): every step reads it to pick ring slots, and a device tensor would
    force a sync. Rows advance in lockstep -- one step consumes one token
    everywhere -- so the values stay offset by their prompt lengths. When they
    are all equal, `uniform_t` lets the hot paths take the cheaper
    single-index route.
    """

    k: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    conv_k: Optional[torch.Tensor]
    conv_v: Optional[torch.Tensor]
    t: Tuple[int, ...]
    activation_times: Tuple[int, ...]
    cache_size: int
    caps: Tuple[int, ...]
    ring_offsets: Tuple[int, ...]

    def __post_init__(self):
        if isinstance(self.t, int):            # accept a scalar, store per row
            self.t = (self.t,) * self.k.shape[0]
        else:
            self.t = tuple(int(x) for x in self.t)
        if len(self.t) != self.k.shape[0]:
            raise ValueError(
                f"t has {len(self.t)} entries for a batch of {self.k.shape[0]}"
            )

    @property
    def num_levels(self) -> int:
        return len(self.activation_times)

    def slot_maps(self):
        """Static per-slot (level, ring index, capacity), built once.

        The ring layout never changes, so these are constants of the schedule.
        Having them lets a decode step recover "which node sits in slot s"
        for every row at once, instead of scattering entry lists per row.
        """
        cached = getattr(self, "_slot_maps", None)
        if cached is not None and cached[0].device == self.k.device:
            return cached
        dev = self.k.device
        S = self.k.shape[1]
        level = torch.zeros(S, dtype=torch.long, device=dev)
        ring = torch.zeros(S, dtype=torch.long, device=dev)
        for l, cap in enumerate(self.caps):
            lo = self.ring_offsets[l]
            level[lo:lo + cap] = l
            ring[lo:lo + cap] = torch.arange(cap, device=dev)
        cap_of = torch.as_tensor(self.caps, device=dev)[level]
        maps = (level, ring, cap_of)
        self._slot_maps = maps
        return maps

    @property
    def uniform_t(self) -> Optional[int]:
        """The shared token count when every row is at the same position,
        else None. Host-side, so branching on it costs no sync."""
        first = self.t[0]
        return first if all(x == first for x in self.t) else None

    def spec(self, seq_len: int) -> RangeSpec:
        """The prefill schedule truncated to `seq_len` tokens (one row's)."""
        return RangeSpec(self.activation_times, self.cache_size, seq_len)

def decode_capacities(
    activation_times: Tuple[int, ...], cache_size: int
) -> Tuple[int, ...]:
    """
    -> caps[l], the ring size per level. sum(caps) may exceed cache_size: each
    ring holds its own level's peak, not the joint peak.
    """
    a = tuple(activation_times)
    if not a or a[0] != 0:
        raise ValueError("activation_times must start with a[0] = 0")
    if cache_size <= 0:
        raise ValueError("cache_size must be positive")
    L = len(a) - 1
    horizon = 2 * (a[L] + (1 << (L + 1))) + 1
    spec = RangeSpec(a, cache_size, horizon)
    caps = [0] * (L + 1)
    for q in range(horizon):
        used = 0
        for level in range(L):
            lo, hi = range_bounds(spec, q, level)
            caps[level] = max(caps[level], hi - lo)
            used += hi - lo
        if used > cache_size:
            raise ValueError(
                f"fine levels need {used} slots > cache_size={cache_size} "
                f"at q={q}"
            )
        caps[L] = max(caps[L], cache_size - used)
    return tuple(caps)

def ring_slot(state: DecodeState, level: int, j: int) -> int:
    """
    Physical slot of node j at `level`: ring_offsets[level] + j % caps[level].
    """
    if not 0 <= level < state.num_levels:
        raise ValueError(
            f"level {level} out of range [0, {state.num_levels - 1}]"
        )
    if j < 0:
        raise ValueError(f"node index must be nonnegative, got {j}")
    return state.ring_offsets[level] + j % state.caps[level]

def _ring_slots(
    state: DecodeState, level: int, indices: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized ring_slot for a level-local index tensor.
    """
    return state.ring_offsets[level] + torch.remainder(
        indices, state.caps[level]
    )

def init_decode_state(
    k_levels: List[torch.Tensor],
    v_levels: List[torch.Tensor],
    w_levels: List[torch.Tensor],
    fmap: Dict[int, int],
    cache_size: int,
    *,
    k_pre_conv: Optional[torch.Tensor] = None,
    v_pre_conv: Optional[torch.Tensor] = None,
    conv_kernel_size: Optional[int] = None,
    real_lens: Optional[Sequence[int]] = None,
) -> DecodeState:
    """
    Prefill -> decode hand-off: k/v/w_levels are build_dyadic_summaries' outputs
    for N tokens, and the state receives the nodes live at query N-1, t = N.
    Short-conv mode also takes the RAW pre-conv k/v and conv_kernel_size.

    `real_lens` gives each row's true prefix length when the batch was padded;
    the row is then handed off as of ITS last real token. This is sound
    because the tree is prefix-stable -- node j of level l depends only on
    tokens below (j+1) * 2^l -- so the first `R >> l` nodes of a padded row's
    tree are exactly the tree of its length-R prefix. Defaults to all rows at
    the full length N.
    """
    check_paired("k_pre_conv", k_pre_conv, "v_pre_conv", v_pre_conv)
    check_paired("k_pre_conv", k_pre_conv, "conv_kernel_size", conv_kernel_size)
    a = activation_times_from_fmap(fmap)
    L = len(a) - 1
    if min(len(k_levels), len(v_levels), len(w_levels)) < L + 1:
        raise ValueError(
            f"need {L + 1} levels, got {len(k_levels)}/{len(v_levels)}/"
            f"{len(w_levels)}"
        )
    B, N, Hkv, Dk = k_levels[0].shape
    Dv = v_levels[0].shape[-1]
    if N < 1:
        raise ValueError("prefill must contain at least one token")
    check_shape("w_levels[0]", w_levels[0], (B, N, Hkv, 1))
    caps = decode_capacities(a, cache_size)
    offsets = prefix_offsets(caps)
    total = offsets[-1]
    state = DecodeState(
        k=k_levels[0].new_zeros(B, total, Hkv, Dk),
        v=v_levels[0].new_zeros(B, total, Hkv, Dv),
        w=w_levels[0].new_zeros(B, total, Hkv, 1),
        conv_k=None,
        conv_v=None,
        t=N,
        activation_times=a,
        cache_size=cache_size,
        caps=caps,
        ring_offsets=offsets,
    )
    lens = [N] * B if real_lens is None else [int(x) for x in real_lens]
    if len(lens) != B:
        raise ValueError(f"real_lens has {len(lens)} entries for batch {B}")
    if min(lens) < 1 or max(lens) > N:
        raise ValueError(f"real_lens must lie in [1, {N}], got {lens}")
    state.t = tuple(lens)

    dev = state.k.device
    # Rows at the same length share a live range; the common (unpadded) case
    # is a single group and reduces to the original whole-batch assignment.
    for group_len in sorted(set(lens)):
        rows = [b for b, x in enumerate(lens) if x == group_len]
        whole = len(rows) == B
        row_idx = None if whole else torch.as_tensor(rows, device=dev)[:, None]
        spec = RangeSpec(a, cache_size, group_len)
        for level in range(L + 1):
            lo, hi = range_bounds(spec, group_len - 1, level)
            if lo == hi:
                continue
            if any(hi > lv[level].shape[1]
                   for lv in (k_levels, v_levels, w_levels)):
                raise ValueError(
                    f"level {level}: live range [{lo},{hi}) exceeds the given "
                    f"level tensors"
                )
            slots = _ring_slots(
                state, level,
                torch.arange(lo, hi, device=dev, dtype=torch.long),
            )
            if whole:
                state.k[:, slots] = k_levels[level][:, lo:hi]
                state.v[:, slots] = v_levels[level][:, lo:hi]
                state.w[:, slots] = w_levels[level][:, lo:hi]
            else:
                sl = slots[None, :]
                state.k[row_idx, sl] = k_levels[level][rows, lo:hi]
                state.v[row_idx, sl] = v_levels[level][rows, lo:hi]
                state.w[row_idx, sl] = w_levels[level][rows, lo:hi]
    if k_pre_conv is not None:
        if conv_kernel_size <= 0:
            raise ValueError(
                f"conv kernel size must be positive, got {conv_kernel_size}"
            )
        if k_pre_conv.shape[:2] != (B, N) or v_pre_conv.shape[:2] != (B, N):
            raise ValueError(
                f"pre-conv rows must be [B={B}, N={N}, ...], got "
                f"{tuple(k_pre_conv.shape)} / {tuple(v_pre_conv.shape)}"
            )
        k_flat = k_pre_conv.reshape(B, N, -1)
        v_flat = v_pre_conv.reshape(B, N, -1)
        if len(set(lens)) == 1:
            r = lens[0]
            state.conv_k = conv_history(k_flat[:, :r], conv_kernel_size)
            state.conv_v = conv_history(v_flat[:, :r], conv_kernel_size)
        else:
            # Each row's last K-1 PRE-conv rows end at its own real length.
            state.conv_k = torch.stack([
                conv_history(k_flat[b:b + 1, :r], conv_kernel_size)[0]
                for b, r in enumerate(lens)])
            state.conv_v = torch.stack([
                conv_history(v_flat[b:b + 1, :r], conv_kernel_size)[0]
                for b, r in enumerate(lens)])
    return state

def _fired_level(activation_times: Tuple[int, ...], t: int) -> Optional[int]:
    """The single level >= 1 that activates a node at token t, or None.

    Dyadic alignment guarantees at most one; two would mean the schedule is
    malformed, which is an assertion rather than a supported case.
    """
    fired = None
    for level in range(1, len(activation_times)):
        span = 1 << level
        if t < activation_times[level] or (t - activation_times[level]) % span:
            continue
        if fired is not None:
            raise AssertionError(
                f"levels {fired} and {level} both activate at t={t}; the "
                f"schedule is not dyadically aligned"
            )
        fired = level
    return fired


def advance_decode_state(
    state: DecodeState,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    *,
    x_t: Optional[torch.Tensor] = None,
    w_proj: Optional[torch.Tensor] = None,
    k_conv_weight: Optional[torch.Tensor] = None,
    v_conv_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Consume each row's token t = state.t[b] in place; every t becomes t + 1.
    q_t/k_t/v_t: [B, 1, H*, D*] RAW projections. -> (k_t, v_t) as stored.

    Order matters: short conv -> merge weight -> the one level activating at t
    merges its two live children -> token t written at level 0, AFTER the merge.
    """
    t_rows = state.t
    uniform = state.uniform_t
    t = uniform if uniform is not None else max(t_rows)
    B, _, Hkv, Dk = state.k.shape
    Dv = state.v.shape[-1]
    if q_t.dim() != 4 or q_t.shape[0] != B or q_t.shape[1] != 1 \
            or q_t.shape[-1] != Dk or q_t.shape[2] % Hkv != 0:
        raise ValueError(
            f"q_t shape {tuple(q_t.shape)} incompatible with [B={B}, 1, "
            f"Hq (multiple of {Hkv}), Dk={Dk}]"
        )
    check_shape("k_t", k_t, (B, 1, Hkv, Dk))
    check_shape("v_t", v_t, (B, 1, Hkv, Dv))

    # 1. short conv, rolling pre-conv state
    check_paired("k_conv_weight", k_conv_weight, "v_conv_weight", v_conv_weight)
    if k_conv_weight is not None:
        if state.conv_k is None or state.conv_v is None:
            raise ValueError(
                "conv weights given but the state was initialised without "
                "pre-conv rows (init_decode_state k_pre_conv/v_pre_conv)"
            )
        k_flat, state.conv_k = short_conv_step(
            k_t.reshape(B, 1, Hkv * Dk), state.conv_k, k_conv_weight
        )
        v_flat, state.conv_v = short_conv_step(
            v_t.reshape(B, 1, Hkv * Dv), state.conv_v, v_conv_weight
        )
        k_t = k_flat.reshape(B, 1, Hkv, Dk)
        v_t = v_flat.reshape(B, 1, Hkv, Dv)
    elif state.conv_k is not None:
        raise ValueError(
            "the state carries a conv state but no conv weights were given"
        )

    # 2. merge weight from the post-conv key, unpositioned q/k
    check_paired("x_t", x_t, "w_proj", w_proj,
                 "linear weight mode; omit both for QK mode")
    if x_t is not None:
        w_t = compute_linear_weights(x_t, w_proj)
        check_shape("linear weights", w_t, (B, 1, Hkv, 1))
    else:
        w_t = compute_summary_weights(q_t, k_t)

    # 3. ruler tick: at most one level >= 1 activates a node at each row's t.
    # The whole decision is host-side (t is host state), so the per-row case
    # costs integer work here and one extra gather/scatter below -- never an
    # extra kernel launch: rows firing DIFFERENT levels still merge together,
    # because a ring slot index is global across levels and merge_children is
    # level-agnostic.
    a = state.activation_times
    L = len(a) - 1
    rows, c0s, c1s, parents = [], [], [], []
    for b, t_b in enumerate(t_rows):
        fired = _fired_level(a, t_b)
        if fired is None:
            continue
        j = (t_b - a[fired]) >> fired
        # Both children leave level-1's range exactly at t_b, so they are live
        # at t_b - 1 (and t_b >= 1 here).
        lo_c, hi_c = range_bounds(state.spec(t_b), t_b - 1, fired - 1)
        if not (lo_c <= 2 * j and 2 * j + 1 < hi_c):
            raise AssertionError(
                f"children of node ({fired}, {j}) not live at t-1={t_b - 1} "
                f"(row {b}): level-{fired - 1} range [{lo_c},{hi_c})"
            )
        rows.append(b)
        c0s.append(ring_slot(state, fired - 1, 2 * j))
        c1s.append(ring_slot(state, fired - 1, 2 * j + 1))
        parents.append(ring_slot(state, fired, j))

    if rows:
        if uniform is not None:
            # Every row fired the same level at the same node: basic indexing
            # gives views and a strided write, no gather/scatter.
            c0, c1, parent = c0s[0], c1s[0], parents[0]
            k_ch, v_ch, w_ch = (
                torch.stack([src[:, c0], src[:, c1]], dim=1)
                for src in (state.k, state.v, state.w)
            )
            (state.k[:, parent], state.v[:, parent],
             state.w[:, parent]) = merge_children(k_ch, v_ch, w_ch, dim=1)
        else:
            dev = state.k.device
            bi = torch.as_tensor(rows, device=dev)
            c0v = torch.as_tensor(c0s, device=dev)
            c1v = torch.as_tensor(c1s, device=dev)
            pv = torch.as_tensor(parents, device=dev)
            k_ch, v_ch, w_ch = (
                torch.stack([src[bi, c0v], src[bi, c1v]], dim=1)
                for src in (state.k, state.v, state.w)
            )
            k_p, v_p, w_p = merge_children(k_ch, v_ch, w_ch, dim=1)
            # index_put_ will not downcast the way a basic-indexing assignment
            # does, and merge_children returns fp32 (alpha is fp32).
            state.k[bi, pv] = k_p.to(state.k.dtype)
            state.v[bi, pv] = v_p.to(state.v.dtype)
            state.w[bi, pv] = w_p.to(state.w.dtype)

    # capacity invariant at each row's new query; also validates the schedule
    for t_b in set(t_rows):
        spec_next = state.spec(t_b + 1)
        for level in range(L + 1):
            lo, hi = range_bounds(spec_next, t_b, level)
            if hi - lo > state.caps[level]:
                raise AssertionError(
                    f"level {level} needs {hi - lo} live slots > cap "
                    f"{state.caps[level]} at t={t_b}"
                )

    # 4. each row's token at its own level-0 slot, after the merge
    if uniform is not None:
        slot0 = ring_slot(state, 0, t)
        state.k[:, slot0] = k_t[:, 0]
        state.v[:, slot0] = v_t[:, 0]
        state.w[:, slot0] = w_t[:, 0]
    else:
        dev = state.k.device
        bi = torch.arange(B, device=dev)
        s0 = torch.as_tensor(
            [ring_slot(state, 0, t_b) for t_b in t_rows], device=dev)
        state.k[bi, s0] = k_t[:, 0].to(state.k.dtype)
        state.v[bi, s0] = v_t[:, 0].to(state.v.dtype)
        state.w[bi, s0] = w_t[:, 0].to(state.w.dtype)
    state.t = tuple(x + 1 for x in t_rows)
    return k_t, v_t
