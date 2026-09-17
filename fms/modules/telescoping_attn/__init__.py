"""
Building blocks for telescoping attention, one module per concern.

    checks.py        the two argument checks every torch module here shares
    range_spec.py    the frozen contract and its branch-free closed forms
    position.py      post-summary RoPE + the learned summary-bin bias
    short_conv.py    short causal conv: functional, nn.Module, decode step
    summaries.py     merge weights, the shared merge, dyadic tree builder
    packing.py       PackedKV and the level-major packing
    sink_gate.py     learned attention sink, output gate
    decode_state.py  per-level rings, capacities, incremental tree update

No flex here: `fms.modules.telescoping_attention` is the implementation --
the FlexAttention prefill/decode kernels and the FMS layer around them --
and the generic flex plumbing lives in `fms.modules.flex_utils`. This package
imports nothing from the rest of fms, and range_spec imports nothing at all -- it is the one module the research repo's independent oracle shares,
and that comparison only has force while the two share the contract and no
implementation.

Extracted from the telescope_cache reference implementation; the equivalence
suite that pins these semantics lives there.
"""
