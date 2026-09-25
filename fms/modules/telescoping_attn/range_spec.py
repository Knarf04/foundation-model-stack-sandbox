"""
Pure-integer range spec for telescoping attention, and the branch-free closed
forms that implement it.

Node k at level l summarizes tokens [k*2^l, (k+1)*2^l); a schedule (`fmap`)
gives activation times a[l]. Per query the attended nodes at a level are ONE
interval R_l(q) = [k_lo, k_hi); a node's inverse Q_l(k) = { q : k in R_l(q) }
is likewise one interval.

    range_bounds / elem_mask             exact, per query
    fwd_bounds / bwd_bounds              hulls over a query block / KV tile
    node_query_bounds                    hull of Q_l(k)
    packed_bounds                        range_bounds offset into storage

fwd/bwd_bounds are HULLS: containment is the contract, exactness is not, so a
backward MUST keep the element mask. Coordinates are level-local except in
packed_bounds. Empty is always (0, 0), and out-of-domain input raises rather
than clamping.
"""

from dataclasses import dataclass
from typing import Callable, Dict, NamedTuple, Sequence, Tuple

EMPTY: Tuple[int, int] = (0, 0)

def activation_times_from_fmap(fmap: Dict[int, int]) -> Tuple[int, ...]:
    """
    a[l] = first query at which node 0 of level l becomes visible, plus the
    dyadic-alignment check. Deliberately duplicates the suite's oracle so this
    module has no test dependency.

    What alignment is FOR: level l promotes a node at every t with
    t == a[l] (mod 2^l), and the tree update assumes at most one level does so
    at any t. Since 2^l divides 2^L, level L's firing times collide with
    level l exactly when a[L] == 2^(l-1) (mod 2^l) -- that is, when a[L]'s
    lowest set bit sits at position l-1.

    A level with a parent must therefore interleave with it exactly:
    a[l] == 2^(l-1) (mod 2^l), the binary-carry ("ruler") pattern.

    THE COARSEST LEVEL IS DIFFERENT: nothing is ever promoted out of it, so it
    has no parent to interleave with and only has to dodge the levels below.
    That holds iff its lowest set bit is at or above L-1, i.e.
    a[L] % 2^(L-1) == 0 -- which admits a[L] == 0 (mod 2^L) as well as the
    2^(L-1) phase the stricter rule allowed. Requiring the strict phase there
    too would reject sound schedules: with L=2 and a=(0, 3, 8) every query
    sees 2-3 tokens, 1-3 two-token summaries and 1-2 four-token summaries,
    with no collision and no dangling child, yet a[2] % 4 == 0.
    """
    if not fmap:
        return (0,)
    L = max(fmap)
    if sorted(fmap) != list(range(1, L + 1)):
        raise ValueError(
            f"fmap keys must be consecutive 1..L; got {sorted(fmap)}"
        )
    a = [0] * (L + 1)
    a[1] = fmap[1] + 1
    for level in range(2, L + 1):
        a[level] = (
            a[level - 1]
            + (1 << (level - 1)) * (fmap[level] - fmap[level - 1])
            + (1 << (level - 2))
        )
    for level in range(1, L):
        if a[level] % (1 << level) != (1 << (level - 1)):
            raise ValueError(
                f"fmap not dyadically aligned at level {level}: "
                f"a[{level}]={a[level]} has phase {a[level] % (1 << level)} "
                f"mod {1 << level}, expected {1 << (level - 1)} so that "
                f"level {level} interleaves with its parent, level "
                f"{level + 1}."
            )
    # Coarsest level: no parent, so only the levels below constrain it.
    if a[L] % (1 << (L - 1)) != 0:
        raise ValueError(
            f"fmap not dyadically aligned at the coarsest level {L}: "
            f"a[{L}]={a[L]} has phase {a[L] % (1 << (L - 1))} mod "
            f"{1 << (L - 1)}, expected 0 -- its lowest set bit must sit at or "
            f"above bit {L - 1}, or it promotes on the same step as a finer "
            f"level."
        )
    # Each level must outlive its children long enough for both to still be
    # there at promotion time. Alignment alone does not imply it: a schedule
    # that advances too little between levels would have a parent form from a
    # child that has already been evicted.
    for level in range(2, L + 1):
        if a[level] - a[level - 1] < (1 << (level - 1)) + (1 << (level - 2)):
            raise ValueError(
                f"fmap advances too little into level {level}: "
                f"a[{level}]-a[{level - 1}]={a[level] - a[level - 1]} leaves "
                f"level-{level - 1} nodes evicted before their parent merges "
                f"them; need at least "
                f"{(1 << (level - 1)) + (1 << (level - 2))}."
            )
    return tuple(a)


@dataclass(frozen=True)
class RangeSpec:
    """Schedule parameters: activation times a[0..L], cache size C, seq len N."""

    activation_times: Tuple[int, ...]
    cache_size: int
    seq_len: int

    @classmethod
    def from_fmap(
        cls, fmap: Dict[int, int], cache_size: int, seq_len: int
    ) -> "RangeSpec":
        return cls(activation_times_from_fmap(fmap), cache_size, seq_len)

    def __post_init__(self):
        if self.cache_size <= 0 or self.seq_len <= 0:
            raise ValueError("cache_size and seq_len must be positive")
        if len(self.activation_times) < 1 or self.activation_times[0] != 0:
            raise ValueError("activation_times must start with a[0] = 0")

    @property
    def num_levels(self) -> int:
        return len(self.activation_times)

    @property
    def coarsest(self) -> int:
        return len(self.activation_times) - 1

    def level_len(self, level: int) -> int:
        """Number of nodes stored at `level`: N, N//2, N//4, ... (floor)."""
        self._check_level(level)
        return self.seq_len >> level

    def level_offsets(self) -> Tuple[int, ...]:
        """
        Where each level starts in the packed level-major buffer, plus the
        total. Bound-method spelling of the module function of the same name.
        """
        return level_offsets(self.seq_len, self.num_levels)

    @property
    def total_len(self) -> int:
        """Total number of packed entries, sum_l level_len(l)."""
        return self.level_offsets()[-1]

    # domain checks

    def _check_level(self, level: int) -> None:
        if not 0 <= level <= self.coarsest:
            raise ValueError(
                f"level {level} out of range [0, {self.coarsest}]"
            )

    def _check_q(self, q: int) -> None:
        if not 0 <= q < self.seq_len:
            raise ValueError(f"q={q} out of range [0, {self.seq_len})")

    def _check_k(self, k: int, level: int) -> None:
        n = self.level_len(level)
        if not 0 <= k < n:
            raise ValueError(
                f"k={k} out of range [0, {n}) at level {level}"
            )


# The implementation: only + - * >> min max, no `if` on a data value, and the
# (min, max) pair arrives as `ops`, so one source runs on Python ints, torch
# tensors and CuTeDSL Int32. Empty is lo >= hi here; the wrappers normalize it.
class IntOps(NamedTuple):
    """The only backend-dependent operations. Everything else is +, -, *, >>."""

    minimum: Callable
    maximum: Callable


PY_OPS = IntOps(minimum=min, maximum=max)


def level_row_bounds(
    q,
    activation_times: Sequence[int],
    cache_size: int,
    ops: IntOps = PY_OPS,
) -> Tuple[Tuple, ...]:
    """
    All levels' [lo, hi) for one query row in one pass. q may be a Python int, a
    torch tensor or a CuTeDSL Int32. -> L+1 pairs, empty as lo >= hi.
    """
    a = tuple(activation_times)
    L = len(a) - 1
    mx = ops.maximum

    bounds = []
    used = 0
    for level in range(L):
        span = 1 << level
        hi = mx(q - a[level] + span, 0) >> level
        lo = (mx(q - a[level + 1] + 2 * span, 0) >> (level + 1)) * 2
        bounds.append((lo, hi))
        # Empty pairs have lo > hi; max(.., 0) stops them crediting the cache
        # with negative occupancy.
        used = used + mx(hi - lo, 0)

    span_L = 1 << L
    hi_L = mx(q - a[L] + span_L, 0) >> L
    lo_L = mx(hi_L - (cache_size - used), 0)
    bounds.append((lo_L, hi_L))
    return tuple(bounds)


def level_row_bounds_one(
    q,
    level: int,
    activation_times: Sequence[int],
    cache_size: int,
    ops: IntOps = PY_OPS,
) -> Tuple:
    """One level's [lo, hi) for query row q."""
    return level_row_bounds(q, activation_times, cache_size, ops)[level]


def block_hull(
    q_lo: int,
    q_hi: int,
    level: int,
    activation_times: Sequence[int],
    cache_size: int,
    ops: IntOps = PY_OPS,
):
    """O(1) hull of R_l(q) over a query block. -> (lo, hi), empty as lo >= hi."""
    # lo(q) and hi(q) are non-decreasing in q, so the hull is
    # [lo(q_lo), hi(q_hi - 1)).
    lo, _ = level_row_bounds_one(q_lo, level, activation_times, cache_size, ops)
    _, hi = level_row_bounds_one(q_hi - 1, level, activation_times, cache_size, ops)
    return lo, hi


def normalize_empty(bounds: Tuple[int, int]) -> Tuple[int, int]:
    """(lo, hi) with this module's empty convention -> range_spec's (0, 0)."""
    lo, hi = bounds
    return (lo, hi) if lo < hi else (0, 0)


def prefix_offsets(lengths: Sequence[int]) -> Tuple[int, ...]:
    """Start offset of each run, plus the total. -> len(lengths) + 1 values."""
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    return tuple(offsets)


def level_lens(seq_len: int, num_levels: int) -> Tuple[int, ...]:
    """Node count per level. -> num_levels values."""
    return tuple(seq_len >> level for level in range(num_levels))


def level_offsets(seq_len: int, num_levels: int) -> Tuple[int, ...]:
    """Level-major start offsets. -> num_levels + 1 values."""
    return prefix_offsets(level_lens(seq_len, num_levels))


def node_query_hull(
    k,
    level: int,
    activation_times: Sequence[int],
    coarsest_span: int,
    seq_len: int,
    ops: IntOps = PY_OPS,
):
    """Branch-free hull of Q_l(k). -> (q_lo, q_hi) clamped to [0, N)."""
    # Fine level: created at a[l] + 2^l k, removed when its pair is promoted
    # at a[l+1] + 2^(l+1) (k // 2). Coarsest: created at a[L] + 2^L k, removed
    # no later than `coarsest_span` queries later.
    a = tuple(activation_times)
    L = len(a) - 1
    mn = ops.minimum

    q_lo = a[level] + (k << level)
    if level < L:
        q_hi = a[level + 1] + ((k >> 1) << (level + 1))
    else:
        q_hi = q_lo + coarsest_span
    return q_lo, mn(q_hi, seq_len)


def tile_query_hull(
    k_lo: int,
    k_hi: int,
    level: int,
    activation_times: Sequence[int],
    coarsest_span: int,
    seq_len: int,
    ops: IntOps = PY_OPS,
):
    """O(1) hull of Q_l(k) over a KV tile. -> (q_lo, q_hi)."""
    # q_lo(k) and q_hi(k) are non-decreasing in k, so the hull is
    # [q_lo(k_lo), q_hi(k_hi - 1)) -- the mirror of block_hull.
    # Clamp to the last attended node: one created at or after seq_len is never
    # attended, and without this the tile would inherit q_hi from a node that
    # does not exist. k_lo > k_max then leaves lo >= seq_len >= hi, i.e. empty.
    k_max = (seq_len - activation_times[level] - 1) >> level
    k_last = ops.minimum(k_hi - 1, k_max)
    lo, _ = node_query_hull(
        k_lo, level, activation_times, coarsest_span, seq_len, ops
    )
    _, hi = node_query_hull(
        k_last, level, activation_times, coarsest_span, seq_len, ops
    )
    return lo, hi


_SPAN_CACHE = {}


def coarsest_lifetime_span(spec) -> int:
    """
    Longest lifetime of any coarsest-level node -- the constant
    node_query_hull adds to a node's creation time.
    """
    # k_lo(q) is non-decreasing, so one sweep over q evicts coarsest nodes in
    # order. Host-side and memoised: once per schedule, to derive the constant
    # the device path consumes. Nodes alive at seq_len are skipped -- their
    # lifetime is truncated by N, not eviction, and would understate the span.
    key = (spec.activation_times, spec.cache_size, spec.seq_len)
    hit = _SPAN_CACHE.get(key)
    if hit is not None:
        return hit
    a = spec.activation_times
    L = spec.coarsest
    N = spec.seq_len
    n_nodes = spec.level_len(L)
    span = 0
    evicting = 0
    for q in range(N):
        k_lo, _ = level_row_bounds(q, a, spec.cache_size, PY_OPS)[L]
        while evicting < k_lo and evicting < n_nodes:
            born = a[L] + (evicting << L)
            if born < N:
                span = max(span, q - born)
            evicting += 1
    if span == 0:
        # Nothing is ever evicted before the sequence ends; N bounds it.
        span = N
    _SPAN_CACHE[key] = span
    return span


# Spec-facing wrappers: the closed forms above plus domain checks and the
# normalized EMPTY.


def range_bounds(spec: RangeSpec, q: int, level: int) -> Tuple[int, int]:
    """Exact R_l(q). -> (k_lo, k_hi) level-local, EMPTY if nothing attended."""
    spec._check_q(q)
    spec._check_level(level)
    lo, hi = normalize_empty(level_row_bounds_one(
        q, level, spec.activation_times, spec.cache_size))
    if hi > spec.level_len(level):
        raise AssertionError(
            f"range [{lo},{hi}) exceeds level_len={spec.level_len(level)} "
            f"at q={q}, level={level}")
    return (lo, hi)


def packed_bounds(spec: RangeSpec, q: int, level: int) -> Tuple[int, int]:
    """
    range_bounds in packed coordinates: (offset[level] + k_lo, + k_hi). The only
    function here that speaks packed indices.
    """
    lo, hi = range_bounds(spec, q, level)
    if lo == hi:
        return EMPTY
    base = spec.level_offsets()[level]
    return (base + lo, base + hi)


def elem_mask(spec: RangeSpec, q: int, k_local: int, level: int) -> bool:
    """k_local in R_l(q). -> bool; the kernel applies this vectorized."""
    spec._check_k(k_local, level)
    lo, hi = range_bounds(spec, q, level)
    return lo <= k_local < hi


def fwd_bounds(
    spec: RangeSpec, q_lo: int, q_hi: int, level: int
) -> Tuple[int, int]:
    """Hull of the R_l(q) a query block streams. -> (k_lo, k_hi) or EMPTY."""
    if not 0 <= q_lo <= q_hi <= spec.seq_len:
        raise ValueError(
            f"[q_lo, q_hi)=[{q_lo},{q_hi}) not within [0, {spec.seq_len}]")
    spec._check_level(level)
    if q_lo == q_hi:
        return EMPTY
    return normalize_empty(block_hull(
        q_lo, q_hi, level, spec.activation_times, spec.cache_size))


def node_query_bounds(spec: RangeSpec, k: int, level: int) -> Tuple[int, int]:
    """Hull of Q_l(k). -> (q_lo, q_hi), EMPTY if the node is never attended."""
    spec._check_k(k, level)
    return normalize_empty(node_query_hull(
        k, level, spec.activation_times, coarsest_lifetime_span(spec),
        spec.seq_len))


def bwd_bounds(
    spec: RangeSpec, k_lo: int, k_hi: int, level: int
) -> Tuple[int, int]:
    """
    Exact hull of the Q_l(k) over a KV tile. -> (q_lo, q_hi), EMPTY if empty.
    Conservative at fine levels, so the backward must keep the element mask.
    """
    spec._check_level(level)
    n = spec.level_len(level)
    if not 0 <= k_lo <= k_hi <= n:
        raise ValueError(
            f"[k_lo, k_hi)=[{k_lo},{k_hi}) not within [0, {n}]")
    if k_lo == k_hi:
        return EMPTY
    return normalize_empty(tile_query_hull(
        k_lo, k_hi, level, spec.activation_times,
        coarsest_lifetime_span(spec), spec.seq_len))
