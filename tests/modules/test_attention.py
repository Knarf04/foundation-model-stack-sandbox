import unittest

import torch

from fms.modules import flex_utils
from fms.modules.attention import MultiHeadAttention, get_attention
from fms.modules.positions import RotaryEmbedding


def _build_mha(with_rope: bool = True, use_sinks: bool = False):
    emb_dim = 32
    nheads, kvheads = 4, 2
    emb_kq = emb_v = emb_dim // nheads
    rope = RotaryEmbedding(emb_kq, max_seq_len=64) if with_rope else None
    mha = MultiHeadAttention(
        emb_dim=emb_dim,
        emb_kq=emb_kq,
        emb_v=emb_v,
        nheads=nheads,
        kvheads=kvheads,
        position_encoder=rope,
        fused=False,
        use_sinks=use_sinks,
    )
    mha.eval()
    return mha


class FullAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        mha = _build_mha()
        x = torch.randn(2, 6, 32)
        with torch.no_grad():
            out = mha(x, use_cache=False)
        self.assertEqual(out.shape, x.shape)

    def test_cache_grows_unbounded(self):
        """Full attention keeps the entire KV history; cache length equals the
        number of tokens processed so far."""
        L = 12
        mha = _build_mha()
        x = torch.randn(1, L, 32)

        cache = None
        with torch.no_grad():
            for t in range(L):
                _, cache = mha(
                    x[:, t : t + 1, :],
                    position_ids=torch.tensor([[t]], dtype=torch.long),
                    past_key_value_state=cache,
                    use_cache=True,
                )
                self.assertEqual(cache[0].shape[2], t + 1)
                self.assertEqual(cache[1].shape[2], t + 1)

    def test_prefill_vs_decode_equivalence(self):
        """Full prefill vs token-by-token cached decoding must match."""
        torch.manual_seed(0)
        L = 12
        B, D = 1, 32

        mha = _build_mha()
        x = torch.randn(B, L, D)

        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out_prefill = mha(x, position_ids=pos, use_cache=False)

        cache = None
        outs = []
        with torch.no_grad():
            for t in range(L):
                pos_t = torch.tensor([[t]], dtype=torch.long)
                out_t, cache = mha(
                    x[:, t : t + 1, :],
                    position_ids=pos_t,
                    past_key_value_state=cache,
                    use_cache=True,
                )
                outs.append(out_t)

        out_decode = torch.cat(outs, dim=1)
        torch.testing.assert_close(out_prefill, out_decode, atol=1e-4, rtol=1e-4)

    def test_seq_len_equals_kvheads_layout(self):
        """Regression: with seq_len == kvheads, the uncached K/V layout used to
        be inferred from tensor shape and b x l x kvh x d was misread as
        b x kvh x l x d, swapping the token and head axes. Cached prefill
        stores in the normalized layout, so it is the reference."""
        torch.manual_seed(0)
        mha = _build_mha()
        L = mha.kvheads
        x = torch.randn(1, L, 32)
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            out_nocache = mha(x, position_ids=pos, use_cache=False)
            out_cached, _ = mha(x, position_ids=pos, use_cache=True)
        torch.testing.assert_close(out_nocache, out_cached, atol=1e-4, rtol=1e-4)

    def test_user_mask_composes_with_causality(self):
        """Regression: on sdpa_causal a user mask is an additional restriction,
        not a replacement for causality. An all-True mask (what HF adapters
        produce for unpadded batches via pad-only mask_2d_to_3d) used to set
        is_causal=False and silently make full attention bidirectional."""
        torch.manual_seed(0)
        mha = _build_mha()
        L = 8
        x = torch.randn(1, L, 32)
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
        ones = torch.ones(1, L, L, dtype=torch.bool)

        with torch.no_grad():
            out_masked = mha(x, position_ids=pos, mask=ones, use_cache=False)
            out_causal = mha(x, position_ids=pos, use_cache=False)
        torch.testing.assert_close(out_masked, out_causal, atol=1e-4, rtol=1e-4)

    def test_float_user_mask_composes_with_causality(self):
        """Additive all-zeros float mask must likewise keep causality."""
        torch.manual_seed(0)
        mha = _build_mha()
        L = 8
        x = torch.randn(1, L, 32)
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
        zeros = torch.zeros(1, L, L)

        with torch.no_grad():
            out_masked = mha(x, position_ids=pos, mask=zeros, use_cache=False)
            out_causal = mha(x, position_ids=pos, use_cache=False)
        torch.testing.assert_close(out_masked, out_causal, atol=1e-4, rtol=1e-4)

    def test_bidirectional_mask_stays_bidirectional(self):
        """Causality composition must not leak into sdpa_bidirectional: an
        all-True mask there still means full bidirectional attention."""
        torch.manual_seed(0)
        mha = _build_mha(with_rope=False)
        L = 8
        x = torch.randn(1, L, 32)
        ones = torch.ones(1, L, L, dtype=torch.bool)

        with torch.no_grad():
            out_masked = mha(
                x, mask=ones, attn_name="sdpa_bidirectional", use_cache=False
            )
            out_free = mha(x, attn_name="sdpa_bidirectional", use_cache=False)
        torch.testing.assert_close(out_masked, out_free, atol=1e-4, rtol=1e-4)

    def test_gate_path_active(self):
        """Zeroing gate_proj must zero the output (no bias on the gate, and
        the default dense path has use_bias=False). Confirms the gate is
        actually on the output path."""
        torch.manual_seed(0)
        mha = _build_mha()
        x = torch.randn(1, 6, 32)
        pos = torch.arange(6, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            out_normal = mha(x, position_ids=pos, use_cache=False)
            self.assertFalse(torch.allclose(out_normal, torch.zeros_like(out_normal)))

            mha.gate_proj.weight.data.zero_()
            out_zero_gate = mha(x, position_ids=pos, use_cache=False)
        torch.testing.assert_close(out_zero_gate, torch.zeros_like(out_zero_gate))


@unittest.skipIf(
    flex_utils.flex_attention_available,
    "construction-failure test only applies without flex",
)
class SinksUnavailableTests(unittest.TestCase):
    def test_use_sinks_requires_flex(self):
        with self.assertRaises(ImportError):
            _build_mha(use_sinks=True)


@unittest.skipUnless(
    flex_utils.flex_attention_available, "flex_attention requires torch >= 2.5"
)
class FullAttentionSinkConstructionTests(unittest.TestCase):
    def test_dropout_rejected(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(
                32, 8, 8, 4, 2, p_dropout=0.1, fused=False, use_sinks=True
            )

    def test_sinks_on_cpu_raise(self):
        """The CPU flex backend exposes no lse (and has no backward), so
        sink-enabled attention must fail early with a clear error."""
        mha = _build_mha(use_sinks=True)
        x = torch.randn(1, 6, 32)
        pos = torch.arange(6, dtype=torch.long).unsqueeze(0)
        with self.assertRaises(RuntimeError):
            mha(x, position_ids=pos, use_cache=False)


@unittest.skipUnless(
    flex_utils.flex_attention_available and torch.cuda.is_available(),
    "sink gating needs the flex lse, which the CPU backend does not provide",
)
class FullAttentionSinkTests(unittest.TestCase):
    """Learned sink = extra softmax entry with logit s_h and value 0, applied
    as an output gate sigmoid(lse - s_h) without recomputing logits."""

    def _twins(self):
        """Two modules with identical weights (same seed; the zero-init sink
        parameter consumes no RNG), one with sinks and one without."""
        torch.manual_seed(42)
        with_sinks = _build_mha(use_sinks=True)
        torch.manual_seed(42)
        without = _build_mha(use_sinks=False)
        return with_sinks.to("cuda"), without.to("cuda")

    def test_masked_neg_sink_matches_masked_base(self):
        """Both paths now compose causality with a user mask internally
        (flex_causal via BlockMask, sdpa_causal via causal AND user), so with
        the sink disabled (s -> -inf) the two twins must agree given the same
        raw user mask — cross-validating the two composition implementations."""
        mha_sink, mha_base = self._twins()
        mha_sink.sinks.data.fill_(-1e4)
        L = 12
        x = torch.randn(1, L, 32, device="cuda")
        pos = torch.arange(L, dtype=torch.long, device="cuda").unsqueeze(0)
        torch.manual_seed(3)
        user = torch.rand(1, L, L) > 0.3
        user |= torch.eye(L, dtype=torch.bool).unsqueeze(0)
        user = user.to("cuda")
        with torch.no_grad():
            out_sink = mha_sink(x, position_ids=pos, mask=user, use_cache=False)
            out_base = mha_base(x, position_ids=pos, mask=user, use_cache=False)
        torch.testing.assert_close(out_sink, out_base, atol=1e-4, rtol=1e-4)

    def test_large_negative_sink_matches_no_sink(self):
        """s_h -> -inf makes the sink weight e^{s_h} vanish: outputs must match
        the sink-free twin. Also cross-validates flex_causal vs sdpa_causal."""
        mha_sink, mha_base = self._twins()
        mha_sink.sinks.data.fill_(-1e4)
        x = torch.randn(1, 12, 32, device="cuda")
        pos = torch.arange(12, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            out_sink = mha_sink(x, position_ids=pos, use_cache=False)
            out_base = mha_base(x, position_ids=pos, use_cache=False)
        torch.testing.assert_close(out_sink, out_base, atol=1e-4, rtol=1e-4)

    def test_large_positive_sink_suppresses_output(self):
        """s_h -> +inf absorbs all attention mass into the zero-valued sink."""
        mha_sink, _ = self._twins()
        mha_sink.sinks.data.fill_(1e4)
        x = torch.randn(1, 12, 32, device="cuda")
        pos = torch.arange(12, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            out = mha_sink(x, position_ids=pos, use_cache=False)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=0)

    def test_prefill_vs_decode_equivalence_with_sinks(self):
        """Nonzero sinks: block-mask prefill and block_mask=None decode must
        agree — cross-validates the two flex sink paths (and would catch a
        return_lse log-base mismatch)."""
        mha_sink, _ = self._twins()
        torch.manual_seed(7)
        mha_sink.sinks.data.normal_(std=1.0)
        L = 12
        x = torch.randn(1, L, 32, device="cuda")

        pos = torch.arange(L, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.no_grad():
            out_prefill = mha_sink(x, position_ids=pos, use_cache=False)

        cache = None
        outs = []
        with torch.no_grad():
            for t in range(L):
                out_t, cache = mha_sink(
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
        mha_sink, _ = self._twins()
        x = torch.randn(1, 8, 32, device="cuda")
        pos = torch.arange(8, dtype=torch.long, device="cuda").unsqueeze(0)
        out = mha_sink(x, position_ids=pos, use_cache=False)
        out.sum().backward()
        self.assertIsNotNone(mha_sink.sinks.grad)
        self.assertFalse(
            torch.allclose(mha_sink.sinks.grad, torch.zeros_like(mha_sink.sinks.grad))
        )


class GetAttentionFactoryTests(unittest.TestCase):
    def test_full_attention_types(self):
        for attn_type in ("attn", "full_attention"):
            attn = get_attention(
                attn_type,
                32,
                8,
                8,
                4,
                2,
                fused=False,
            )
            self.assertIsInstance(attn, MultiHeadAttention)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            get_attention("not_a_type", 32, 8, 8, 4, 2)


if __name__ == "__main__":
    unittest.main()
