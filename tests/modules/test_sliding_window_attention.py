import unittest

import torch

from fms.modules import sliding_window_attention as swa_module
from fms.modules.attention import MultiHeadAttention
from fms.modules.positions import RotaryEmbedding
from fms.modules.sliding_window_attention import SlidingWindowMultiHeadAttention

_flex_available = swa_module._flex_attention_available


def _build_swa(sliding_window: int, with_rope: bool = True, use_sinks: bool = False):
    emb_dim = 32
    nheads, kvheads = 4, 2
    emb_kq = emb_v = emb_dim // nheads
    rope = RotaryEmbedding(emb_kq, max_seq_len=64) if with_rope else None
    swa = SlidingWindowMultiHeadAttention(
        emb_dim=emb_dim,
        emb_kq=emb_kq,
        emb_v=emb_v,
        nheads=nheads,
        kvheads=kvheads,
        position_encoder=rope,
        fused=False,
        sliding_window=sliding_window,
        use_sinks=use_sinks,
    )
    swa.eval()
    return swa


def _dense_window_mask(q_len: int, k_len: int, window_size: int) -> torch.Tensor:
    """(1, q_len, k_len) bool reference mask: query at cache-offset position
    k_len - q_len + i attends to keys in (q_pos - window_size, q_pos]."""
    q_idx = torch.arange(k_len - q_len, k_len).unsqueeze(1)
    k_idx = torch.arange(k_len).unsqueeze(0)
    return ((k_idx <= q_idx) & (k_idx > q_idx - window_size)).unsqueeze(0)


@unittest.skipIf(
    _flex_available, "construction-failure test only applies without flex"
)
class FlexUnavailableTests(unittest.TestCase):
    def test_construction_raises_without_flex(self):
        with self.assertRaises(ImportError):
            _build_swa(sliding_window=4)


@unittest.skipUnless(_flex_available, "flex_attention requires torch >= 2.5")
class SlidingWindowConstructionTests(unittest.TestCase):
    def test_dropout_rejected(self):
        with self.assertRaises(ValueError):
            SlidingWindowMultiHeadAttention(
                emb_dim=32,
                emb_kq=8,
                emb_v=8,
                nheads=4,
                kvheads=2,
                p_dropout=0.1,
                fused=False,
            )

    def test_sliding_window_must_be_positive(self):
        """window_size = 0 would make the decode slice [:, :, -0:, :] silently
        attend over the full cache; non-positive windows must be rejected."""
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                _build_swa(sliding_window=bad)

    def test_foreign_attn_name_rejected(self):
        swa = _build_swa(sliding_window=4)
        x = torch.randn(1, 6, 32)
        with self.assertRaises(ValueError):
            swa(x, attn_name="sdpa_bidirectional", use_cache=False)


@unittest.skipUnless(_flex_available, "flex_attention requires torch >= 2.5")
class SlidingWindowAttentionTests(unittest.TestCase):
    def test_batched_no_position_ids_raises(self):
        """Local position counter is single-stream; batch>1 without explicit
        position_ids must raise."""
        swa = _build_swa(sliding_window=4, with_rope=True)
        x = torch.randn(2, 6, 32)  # batch_size = 2
        with self.assertRaises(ValueError):
            swa(x, use_cache=True)

    def test_batched_explicit_position_ids_ok(self):
        """Batched cached forward with explicit absolute position_ids is
        supported: the positions are used as given, the single-stream counter
        stays untouched, and each row matches its own single-stream run."""
        torch.manual_seed(0)
        swa = _build_swa(sliding_window=4, with_rope=True)
        x = torch.randn(2, 6, 32)
        pos = torch.arange(6, dtype=torch.long).unsqueeze(0).expand(2, -1)

        counter_before = swa.curr_id
        with torch.no_grad():
            out, _ = swa(x, position_ids=pos, use_cache=True)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(swa.curr_id, counter_before)

        with torch.no_grad():
            out_row0 = swa(x[0:1], position_ids=pos[0:1], use_cache=False)
        torch.testing.assert_close(out[0:1], out_row0, atol=1e-4, rtol=1e-4)

    def test_user_bool_mask_composes(self):
        """A user mask must AND with the window. Reference: the dense sdpa
        path on the same weights with (window & user) as an explicit mask."""
        torch.manual_seed(0)
        sliding_window, L = 4, 12
        swa = _build_swa(sliding_window=sliding_window, with_rope=True)
        x = torch.randn(1, L, 32)
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
        torch.manual_seed(1)
        user = torch.rand(1, L, L) > 0.3
        user |= torch.eye(L, dtype=torch.bool).unsqueeze(0)  # no empty rows

        with torch.no_grad():
            out = swa(x, position_ids=pos, mask=user, use_cache=False)
            out_ref = MultiHeadAttention.forward(
                swa,
                x,
                position_ids=pos,
                use_cache=False,
                attn_name="sdpa_causal",
                mask=_dense_window_mask(L, L, sliding_window) & user,
            )
        torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)

    def test_user_float_mask_composes(self):
        """Additive float masks (0 = attend, -1e9 = masked) go through a
        score_mod and must match the equivalent boolean mask."""
        torch.manual_seed(0)
        sliding_window, L = 4, 12
        swa = _build_swa(sliding_window=sliding_window, with_rope=True)
        x = torch.randn(1, L, 32)
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
        torch.manual_seed(1)
        user = torch.rand(1, L, L) > 0.3
        user |= torch.eye(L, dtype=torch.bool).unsqueeze(0)
        float_mask = torch.where(user, 0.0, -1e9)

        with torch.no_grad():
            out_float = swa(x, position_ids=pos, mask=float_mask, use_cache=False)
            out_bool = swa(x, position_ids=pos, mask=user, use_cache=False)
        torch.testing.assert_close(out_float, out_bool, atol=1e-4, rtol=1e-4)

    def test_masked_prefill_vs_decode_equivalence(self):
        """Padding-style mask through cached decode: masked columns must not
        change parity between full prefill and token-by-token decode. Uses a
        left-pad-style mask that blanks the first two keys for every query."""
        torch.manual_seed(0)
        sliding_window, L = 4, 10
        swa = _build_swa(sliding_window=sliding_window, with_rope=True)
        x = torch.randn(1, L, 32)
        keep = torch.ones(1, L, L, dtype=torch.bool)
        keep[:, :, :2] = False
        keep |= torch.eye(L, dtype=torch.bool).unsqueeze(0)

        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out_prefill = swa(x, position_ids=pos, mask=keep, use_cache=False)

        swa.reset_position_counter()
        cache = None
        outs = []
        with torch.no_grad():
            for t in range(L):
                out_t, cache = swa(
                    x[:, t : t + 1, :],
                    position_ids=torch.tensor([[t]], dtype=torch.long),
                    mask=keep[:, t : t + 1, : t + 1],
                    past_key_value_state=cache,
                    use_cache=True,
                )
                outs.append(out_t)
        out_decode = torch.cat(outs, dim=1)
        torch.testing.assert_close(out_prefill, out_decode, atol=1e-4, rtol=1e-4)

    def test_position_counter_lifecycle(self):
        """curr_id starts at 0, advances by q_len each forward, syncs to last+1
        when explicit position_ids is supplied, and resets via the helper."""
        swa = _build_swa(sliding_window=4, with_rope=True)
        self.assertEqual(swa.curr_id, 0)

        x_prefill = torch.randn(1, 10, 32)
        with torch.no_grad():
            _, cache = swa(x_prefill, use_cache=True)
        self.assertEqual(swa.curr_id, 10)

        x_decode = torch.randn(1, 1, 32)
        with torch.no_grad():
            _, cache = swa(x_decode, past_key_value_state=cache, use_cache=True)
        self.assertEqual(swa.curr_id, 11)

        # Explicit position_ids must sync the counter.
        with torch.no_grad():
            swa(
                torch.randn(1, 1, 32),
                position_ids=torch.tensor([[20]], dtype=torch.long),
                past_key_value_state=cache,
                use_cache=True,
            )
        self.assertEqual(swa.curr_id, 21)

        swa.reset_position_counter()
        self.assertEqual(swa.curr_id, 0)

    def test_cache_bound(self):
        """Decode for L > W and assert returned cache is always capped at W."""
        sliding_window = 4
        L = 12
        swa = _build_swa(sliding_window=sliding_window, with_rope=True)
        swa.reset_position_counter()
        x = torch.randn(1, L, 32)

        cache = None
        with torch.no_grad():
            for t in range(L):
                _, cache = swa(
                    x[:, t : t + 1, :],
                    past_key_value_state=cache,
                    use_cache=True,
                )
                self.assertLessEqual(cache[0].shape[2], sliding_window)
                self.assertLessEqual(cache[1].shape[2], sliding_window)

    def test_prefill_vs_decode_equivalence(self):
        """Full prefill SWA vs token-by-token cached decoding must match for
        L > sliding_window when both use absolute position_ids."""
        torch.manual_seed(0)
        sliding_window = 4
        L = 12
        B, D = 1, 32

        swa = _build_swa(sliding_window=sliding_window, with_rope=True)
        x = torch.randn(B, L, D)

        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out_prefill = swa(x, position_ids=pos, use_cache=False)

        swa.reset_position_counter()
        cache = None
        outs = []
        with torch.no_grad():
            for t in range(L):
                pos_t = torch.tensor([[t]], dtype=torch.long)
                out_t, cache = swa(
                    x[:, t : t + 1, :],
                    position_ids=pos_t,
                    past_key_value_state=cache,
                    use_cache=True,
                )
                outs.append(out_t)

        out_decode = torch.cat(outs, dim=1)
        torch.testing.assert_close(out_prefill, out_decode, atol=1e-4, rtol=1e-4)

    def test_flex_matches_dense_reference(self):
        """Flex-backed SWA prefill must match a dense-mask reference computed
        through the full-attention machinery on the same weights: the SWA
        module is a MultiHeadAttention subclass, so MultiHeadAttention.forward
        with an explicit boolean window mask is an exact reference."""
        torch.manual_seed(0)
        sliding_window = 4
        L = 12
        swa = _build_swa(sliding_window=sliding_window, with_rope=True)
        x = torch.randn(1, L, 32)
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            out_flex = swa(x, position_ids=pos, use_cache=False)
            out_ref = MultiHeadAttention.forward(
                swa,
                x,
                position_ids=pos,
                use_cache=False,
                attn_name="sdpa_causal",
                mask=_dense_window_mask(L, L, sliding_window),
            )

        torch.testing.assert_close(out_flex, out_ref, atol=1e-4, rtol=1e-4)

    def test_seq_len_equals_kvheads_layout(self):
        """Regression: with seq_len == kvheads, the uncached K/V layout used to
        be inferred from tensor shape and b x l x kvh x d was misread as
        b x kvh x l x d, swapping the token and head axes. Cached prefill
        stores in the normalized layout, so it is the reference."""
        torch.manual_seed(0)
        swa = _build_swa(sliding_window=4, with_rope=True)
        L = swa.kvheads
        x = torch.randn(1, L, 32)
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            out_nocache = swa(x, position_ids=pos, use_cache=False)
            swa.reset_position_counter()
            out_cached, _ = swa(x, position_ids=pos, use_cache=True)
        torch.testing.assert_close(out_nocache, out_cached, atol=1e-4, rtol=1e-4)

    def test_gate_path_active(self):
        """Zeroing gate_proj must zero the output — the inherited gate is on
        the SWA output path too."""
        torch.manual_seed(0)
        swa = _build_swa(sliding_window=4, with_rope=True)
        x = torch.randn(1, 6, 32)
        pos = torch.arange(6, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            out_normal = swa(x, position_ids=pos, use_cache=False)
            self.assertFalse(torch.allclose(out_normal, torch.zeros_like(out_normal)))

            swa.gate_proj.weight.data.zero_()
            out_zero_gate = swa(x, position_ids=pos, use_cache=False)
        torch.testing.assert_close(out_zero_gate, torch.zeros_like(out_zero_gate))


@unittest.skipUnless(_flex_available, "flex_attention requires torch >= 2.5")
class SlidingWindowSinkCpuGuardTests(unittest.TestCase):
    def test_sinks_on_cpu_raise(self):
        """The CPU flex backend exposes no lse (and has no backward), so
        sink-enabled attention must fail early with a clear error."""
        swa = _build_swa(sliding_window=4, use_sinks=True)
        x = torch.randn(1, 6, 32)
        pos = torch.arange(6, dtype=torch.long).unsqueeze(0)
        with self.assertRaises(RuntimeError):
            swa(x, position_ids=pos, use_cache=False)


@unittest.skipUnless(
    _flex_available and torch.cuda.is_available(),
    "sink gating needs the flex lse, which the CPU backend does not provide",
)
class SlidingWindowSinkTests(unittest.TestCase):
    """Learned sink = extra softmax entry with logit s_h and value 0, applied
    as a sigmoid(lse - s_h) output gate from flex's lse — no logits recompute."""

    def _twins(self, sliding_window: int = 4):
        """Two modules with identical weights (same seed; the zero-init sink
        parameter consumes no RNG), one with sinks and one without."""
        torch.manual_seed(42)
        with_sinks = _build_swa(sliding_window=sliding_window, use_sinks=True)
        torch.manual_seed(42)
        without = _build_swa(sliding_window=sliding_window, use_sinks=False)
        return with_sinks.to("cuda"), without.to("cuda")

    def test_large_negative_sink_matches_no_sink(self):
        """s_h -> -inf makes the sink weight e^{s_h} vanish: outputs must match
        the sink-free twin on both the prefill and decode paths."""
        swa_sink, swa_base = self._twins()
        swa_sink.sinks.data.fill_(-1e4)
        L = 12
        x = torch.randn(1, L, 32, device="cuda")
        pos = torch.arange(L, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            out_sink = swa_sink(x, position_ids=pos, use_cache=False)
            out_base = swa_base(x, position_ids=pos, use_cache=False)
        torch.testing.assert_close(out_sink, out_base, atol=1e-4, rtol=1e-4)

    def test_large_positive_sink_suppresses_output(self):
        """s_h -> +inf absorbs all attention mass into the zero-valued sink."""
        swa_sink, _ = self._twins()
        swa_sink.sinks.data.fill_(1e4)
        x = torch.randn(1, 12, 32, device="cuda")
        pos = torch.arange(12, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            out = swa_sink(x, position_ids=pos, use_cache=False)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=0)

    def test_prefill_vs_decode_equivalence_with_sinks(self):
        """Nonzero sinks, L > window: block-mask flex prefill and sliced
        block_mask=None flex decode must agree — cross-validates the two sink
        paths (and would catch a return_lse log-base mismatch)."""
        swa_sink, _ = self._twins(sliding_window=4)
        torch.manual_seed(7)
        swa_sink.sinks.data.normal_(std=1.0)
        L = 12
        x = torch.randn(1, L, 32, device="cuda")

        pos = torch.arange(L, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            out_prefill = swa_sink(x, position_ids=pos, use_cache=False)

        swa_sink.reset_position_counter()
        cache = None
        outs = []
        with torch.no_grad():
            for t in range(L):
                out_t, cache = swa_sink(
                    x[:, t : t + 1, :],
                    position_ids=torch.tensor(
                        [[t]], dtype=torch.long, device="cuda"
                    ),
                    past_key_value_state=cache,
                    use_cache=True,
                )
                outs.append(out_t)
        out_decode = torch.cat(outs, dim=1)
        torch.testing.assert_close(out_prefill, out_decode, atol=1e-4, rtol=1e-4)

    def test_sink_gradients_flow(self):
        swa_sink, _ = self._twins()
        x = torch.randn(1, 8, 32, device="cuda")
        pos = torch.arange(8, dtype=torch.long, device="cuda").unsqueeze(0)
        out = swa_sink(x, position_ids=pos, use_cache=False)
        out.sum().backward()
        self.assertIsNotNone(swa_sink.sinks.grad)
        self.assertFalse(
            torch.allclose(swa_sink.sinks.grad, torch.zeros_like(swa_sink.sinks.grad))
        )


if __name__ == "__main__":
    unittest.main()
