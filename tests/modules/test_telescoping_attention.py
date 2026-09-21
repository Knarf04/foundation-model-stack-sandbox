import unittest

import torch

from fms.models.llama import LLaMA, LLaMAConfig, _resolve_layer_attn
from fms.modules import telescoping_attention as tele_module
from fms.modules.telescoping_attention import (
    DEFAULT_FMAP,
    TelescopingMultiHeadAttention,
)
from fms.modules.telescoping_attn.decode_state import DecodeState
from fms.modules.telescoping_attn.short_conv import ShortConv1d
from fms.modules.positions import RotaryEmbedding

_flex_available = tele_module._flex_available
_cuda = torch.cuda.is_available()

# A tiny schedule: level 1 visible from query 4, level 2 from 6, level 3 from 8.
TINY_FMAP = {1: 4, 2: 6, 3: 8}
EMB, NHEADS, KVHEADS = 32, 4, 2


def _build(**kwargs):
    """A telescoping layer on the tiny schedule, eval mode, reset weights."""
    kwargs.setdefault("fmap", TINY_FMAP)
    kwargs.setdefault("cache_size", 16)
    tele = TelescopingMultiHeadAttention(
        emb_dim=EMB,
        emb_kq=EMB // NHEADS,
        emb_v=EMB // NHEADS,
        nheads=NHEADS,
        kvheads=KVHEADS,
        fused=False,
        **kwargs,
    )
    tele.reset_parameters()
    tele.eval()
    return tele


@unittest.skipUnless(_flex_available, "flex_attention requires torch >= 2.5")
class TelescopingConstructionTests(unittest.TestCase):
    def test_dropout_rejected(self):
        with self.assertRaises(ValueError):
            _build(p_dropout=0.1)

    def test_unknown_modes_rejected(self):
        with self.assertRaises(ValueError):
            _build(weight_mode="magic")
        with self.assertRaises(ValueError):
            _build(position_mode="alibi")

    def test_position_encoder_conflicts_with_post_summary_rope(self):
        """A pre-cache RoPE plus post-summary RoPE would rotate keys twice;
        the combination must be rejected rather than silently double-applied."""
        rope = RotaryEmbedding(EMB // NHEADS, max_seq_len=64)
        for mode in ("rope", "relative"):
            with self.assertRaises(ValueError):
                _build(position_mode=mode, position_encoder=rope)
        # position_mode="none" defers positioning to the encoder, so it is fine
        _build(position_mode="none", position_encoder=rope)

    def test_bad_schedule_rejected(self):
        with self.assertRaises(ValueError):
            _build(cache_size=0)
        with self.assertRaises(ValueError):
            _build(fmap={2: 6, 3: 8})  # keys must start at 1 and be consecutive

    def test_foreign_attn_name_rejected(self):
        tele = _build()
        with self.assertRaises(ValueError):
            tele(torch.randn(1, 12, EMB), attn_name="flex_sliding_window")

    def test_user_mask_rejected(self):
        """Summaries merge tokens before a mask could apply, so a padding or
        packing mask must raise rather than be silently dropped."""
        tele = _build()
        with self.assertRaises(NotImplementedError):
            tele(torch.randn(1, 12, EMB), mask=torch.ones(1, 12, 12, dtype=torch.bool))

    def test_reset_parameters_reaches_every_parameter(self):
        """MultiHeadAttention's loop sees modules and sinks only; the conv and
        the relative tables are a non-Linear module and raw Parameters."""
        tele = _build(
            use_kv_short_conv=True, weight_mode="linear", position_mode="relative"
        )
        with torch.no_grad():
            for p in tele.parameters():
                p.fill_(7.0)
        tele.reset_parameters()
        untouched = [n for n, p in tele.named_parameters() if bool((p == 7).all())]
        self.assertEqual(untouched, [])
        self.assertTrue(bool((tele.k_sconv.weight == 0).all()))
        self.assertTrue(bool((tele.rel_w == 0).all()))

    def test_zero_init_short_conv_is_identity(self):
        """The conv carries a residual, so zero weights mean enabling it starts
        from exactly the unconvolved model."""
        torch.manual_seed(0)
        plain = _build()
        torch.manual_seed(0)
        conv = _build(use_kv_short_conv=True)
        conv.load_state_dict(
            {**plain.state_dict(),
             "k_sconv.weight": conv.k_sconv.weight,
             "v_sconv.weight": conv.v_sconv.weight}
        )
        x = torch.randn(1, 20, EMB)
        with torch.no_grad():
            torch.testing.assert_close(plain(x), conv(x))


@unittest.skipUnless(_flex_available, "flex_attention requires torch >= 2.5")
class TelescopingForwardTests(unittest.TestCase):
    def test_forward_shape_and_finiteness(self):
        tele = _build()
        x = torch.randn(2, 24, EMB)
        with torch.no_grad():
            out = tele(x)
        self.assertEqual(tuple(out.shape), (2, 24, EMB))
        self.assertTrue(torch.isfinite(out).all())

    def test_cache_is_a_decode_state(self):
        tele = _build()
        with torch.no_grad():
            _, cache = tele(torch.randn(1, 20, EMB), use_cache=True)
        self.assertIsInstance(cache, DecodeState)
        self.assertEqual(cache.t, (20,))   # per row

    def test_foreign_cache_rejected(self):
        """A (keys, values) cache from another attention type cannot be
        continued here; that must be an error, not a wrong answer."""
        tele = _build()
        kv = (torch.randn(1, KVHEADS, 4, EMB // NHEADS),) * 2
        with self.assertRaises(TypeError):
            tele(torch.randn(1, 1, EMB), past_key_value_state=kv)

    def test_prefill_vs_decode_equivalence(self):
        """The contract that matters: prefilling N tokens and then stepping one
        at a time must reproduce a single full-sequence prefill."""
        L, split = 40, 36
        for kwargs in (
            {},
            {"use_kv_short_conv": True},
            {"weight_mode": "linear"},
            {"position_mode": "none"},
            {"position_mode": "relative"},
            {"softcap": None},
        ):
            with self.subTest(**kwargs):
                torch.manual_seed(1)
                tele = _build(**kwargs)
                x = torch.randn(1, L, EMB)
                with torch.no_grad():
                    full = tele(x)
                    rows, state = tele(x[:, :split], use_cache=True)
                    outs = [rows]
                    for i in range(split, L):
                        row, state = tele(
                            x[:, i : i + 1],
                            past_key_value_state=state,
                            use_cache=True,
                        )
                        outs.append(row)
                torch.testing.assert_close(
                    torch.cat(outs, dim=1), full, atol=1e-4, rtol=1e-4
                )

    def test_multi_token_decode_matches_single_token(self):
        """A q_len > 1 continuation consumes tokens one at a time internally
        and must match stepping them individually."""
        torch.manual_seed(2)
        tele = _build()
        x = torch.randn(1, 30, EMB)
        with torch.no_grad():
            _, s_chunk = tele(x[:, :24], use_cache=True)
            chunk, _ = tele(x[:, 24:], past_key_value_state=s_chunk, use_cache=True)
            _, s_step = tele(x[:, :24], use_cache=True)
            steps = []
            for i in range(24, 30):
                row, s_step = tele(
                    x[:, i : i + 1], past_key_value_state=s_step, use_cache=True
                )
                steps.append(row)
        torch.testing.assert_close(chunk, torch.cat(steps, dim=1))

    def test_gate_path_active(self):
        """The inherited SiLU gate is on: zeroing gate_proj zeroes the output."""
        tele = _build()
        with torch.no_grad():
            tele.gate_proj.weight.zero_()
            out = tele(torch.randn(1, 20, EMB))
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_sinks_change_the_output(self):
        """A zero sink is NOT an identity: it adds e^0 = 1 to the denominator."""
        torch.manual_seed(3)
        x = torch.randn(1, 20, EMB)
        torch.manual_seed(4)
        plain = _build()
        torch.manual_seed(4)
        sunk = _build(use_sinks=True)
        sunk.load_state_dict({**plain.state_dict(), "sinks": sunk.sinks})
        with torch.no_grad():
            self.assertFalse(torch.allclose(plain(x), sunk(x)))
            # a very negative sink removes the extra denominator entry again
            sunk.sinks.fill_(-30.0)
            torch.testing.assert_close(plain(x), sunk(x), atol=1e-5, rtol=1e-5)


def _pad_mask(lens, total, device="cpu", left=True):
    """FMS-convention bool mask for a left- (or right-) padded batch."""
    valid = torch.zeros(len(lens), total, dtype=torch.bool, device=device)
    for b, r in enumerate(lens):
        if left:
            valid[b, total - r:] = True
        else:
            valid[b, :r] = True
    return (valid[:, None, :] & valid[:, :, None]).tril()


@unittest.skipUnless(_flex_available, "flex_attention requires torch >= 2.5")
class TelescopingPaddingTests(unittest.TestCase):
    """A padding mask is handled by realigning each row's real tokens to the
    front, not by masking: the dyadic tree is anchored at position 0, so left
    padding would otherwise regroup every token into different summaries."""

    def test_left_padding_matches_the_unpadded_run(self):
        torch.manual_seed(0)
        tele = _build()
        real = torch.randn(1, 37, EMB)
        with torch.no_grad():
            gold = tele(real)
            for pad in (3, 11, 27):
                x = torch.cat([torch.randn(1, pad, EMB), real], dim=1)
                got = tele(x, mask=_pad_mask([37], 37 + pad))[:, pad:]
                torch.testing.assert_close(got, gold, atol=1e-6, rtol=1e-6)

    def test_pad_content_cannot_leak(self):
        """Same padded length, wildly different pad values -> bitwise equal.
        A node straddling the real/pad boundary is only ever visible to a pad
        query, because a node activates no earlier than its own last token."""
        torch.manual_seed(0)
        tele = _build()
        real = torch.randn(1, 37, EMB)
        mask = _pad_mask([37], 64)
        outs = []
        with torch.no_grad():
            for scale in (1.0, 1000.0, 0.0):
                x = torch.cat([torch.randn(1, 27, EMB) * scale, real], dim=1)
                # only the real rows are claimed: the pad rows are garbage
                # computed from garbage, and the loss masks them out
                outs.append(tele(x, mask=mask)[:, 27:])
        self.assertTrue(torch.equal(outs[0], outs[1]))
        self.assertTrue(torch.equal(outs[0], outs[2]))

    def test_float_mask_rejected(self):
        tele = _build()
        with self.assertRaises(NotImplementedError):
            tele(torch.randn(1, 12, EMB), mask=torch.zeros(1, 12, 12))

    def test_packing_mask_rejected(self):
        """A mask that varies with the query beyond causality cannot be fixed
        by realignment -- a boundary inside a dyadic node survives the merge."""
        tele = _build()
        blocks = torch.zeros(1, 24, 24, dtype=torch.bool)
        blocks[:, :12, :12] = True
        blocks[:, 12:, 12:] = True
        with self.assertRaises(NotImplementedError):
            tele(torch.randn(1, 24, EMB), mask=blocks.tril())

    def test_cache_hands_off_at_the_real_length(self):
        tele = _build()
        real = torch.randn(1, 37, EMB)
        x = torch.cat([torch.randn(1, 27, EMB), real], dim=1)
        with torch.no_grad():
            _, state = tele(x, mask=_pad_mask([37], 64), use_cache=True)
        self.assertEqual(state.t, (37,))


@unittest.skipUnless(_flex_available, "flex_attention requires torch >= 2.5")
class TelescopingPerRowPositionTests(unittest.TestCase):
    """DecodeState.t is per row, so prompts of differing length decode in one
    batch. Rows at the same position share the uniform fast path."""

    def test_uniform_batch_reports_one_position(self):
        tele = _build()
        with torch.no_grad():
            _, state = tele(torch.randn(4, 20, EMB), use_cache=True)
        self.assertEqual(state.t, (20,) * 4)
        self.assertEqual(state.uniform_t, 20)

    def test_ragged_batch_decodes_as_separate_rows(self):
        lens = [37, 30, 22]
        for mode in ("rope", "relative", "none"):
            with self.subTest(position_mode=mode):
                torch.manual_seed(5)
                tele = _build(position_mode=mode)
                reals = [torch.randn(1, r, EMB) for r in lens]
                padded = torch.cat(
                    [torch.cat([torch.randn(1, 40 - r, EMB), t], dim=1)
                     for t, r in zip(reals, lens)], dim=0)
                new = torch.randn(len(lens), 3, EMB)
                with torch.no_grad():
                    _, st = tele(padded, mask=_pad_mask(lens, 40), use_cache=True)
                    self.assertEqual(st.t, tuple(lens))
                    self.assertIsNone(st.uniform_t)
                    got = []
                    for i in range(3):
                        row, st = tele(new[:, i:i + 1],
                                       past_key_value_state=st, use_cache=True)
                        got.append(row)
                    got = torch.cat(got, dim=1)
                    # each row decoded on its own, unpadded
                    want = []
                    for b, xr in enumerate(reals):
                        _, s = tele(xr, use_cache=True)
                        rows = []
                        for i in range(3):
                            o, s = tele(new[b:b + 1, i:i + 1],
                                        past_key_value_state=s, use_cache=True)
                            rows.append(o)
                        want.append(torch.cat(rows, dim=1))
                    want = torch.cat(want, dim=0)
                self.assertEqual(st.t, tuple(x + 3 for x in lens))
                torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)

    def test_rows_advance_independently(self):
        tele = _build()
        lens = [20, 14]
        padded = torch.cat(
            [torch.cat([torch.randn(1, 24 - r, EMB), torch.randn(1, r, EMB)], 1)
             for r in lens], dim=0)
        with torch.no_grad():
            _, st = tele(padded, mask=_pad_mask(lens, 24), use_cache=True)
            for _ in range(5):
                _, st = tele(torch.randn(2, 1, EMB),
                             past_key_value_state=st, use_cache=True)
        self.assertEqual(st.t, (25, 19))     # offsets preserved, lockstep


@unittest.skipUnless(
    _flex_available and _cuda, "flex_attention backward requires CUDA"
)
class TelescopingGradientTests(unittest.TestCase):
    def test_gradients_flow_to_every_parameter(self):
        tele = _build(
            use_kv_short_conv=True, weight_mode="linear", use_sinks=True
        ).cuda()
        tele.train()
        out = tele(torch.randn(1, 40, EMB, device="cuda"))
        out.sum().backward()
        missing = [n for n, p in tele.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])
        for name, p in tele.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_relative_bias_gradients_flow(self):
        tele = _build(position_mode="relative").cuda()
        tele.train()
        tele(torch.randn(1, 40, EMB, device="cuda")).sum().backward()
        self.assertIsNotNone(tele.rel_w.grad)
        self.assertIsNotNone(tele.rel_proj.grad)


@unittest.skipUnless(_flex_available, "flex_attention requires torch >= 2.5")
class TelescopingLLaMAIntegrationTests(unittest.TestCase):
    @staticmethod
    def _config(**kwargs):
        return LLaMAConfig(
            src_vocab_size=64,
            emb_dim=EMB,
            nheads=NHEADS,
            kvheads=KVHEADS,
            nlayers=4,
            max_expected_seq_len=128,
            **kwargs,
        )

    def test_layer_selection_and_position_encoder(self):
        from fms.modules.attention import MultiHeadAttention
        from fms.modules.sliding_window_attention import (
            SlidingWindowMultiHeadAttention,
        )

        model = LLaMA(
            self._config(
                tele={"layers": [1], "fmap": TINY_FMAP, "cache_size": 16},
                swa={"layers": [2], "window_size": 8},
            )
        )
        kinds = [type(layer.attn) for layer in model.layers]
        self.assertEqual(
            kinds,
            [
                MultiHeadAttention,
                TelescopingMultiHeadAttention,
                SlidingWindowMultiHeadAttention,
                MultiHeadAttention,
            ],
        )
        # telescoping positions after summarization, so it must not also carry
        # the shared pre-cache RoPE
        self.assertIsNone(model.layers[1].attn.position_encoder)
        self.assertIsNotNone(model.layers[2].attn.position_encoder)

    def test_position_mode_none_keeps_the_shared_rope(self):
        model = LLaMA(
            self._config(
                tele={
                    "layers": [1],
                    "fmap": TINY_FMAP,
                    "cache_size": 16,
                    "position_mode": "none",
                }
            )
        )
        self.assertIsNotNone(model.layers[1].attn.position_encoder)

    def test_layer_in_two_dicts_rejected(self):
        with self.assertRaises(ValueError):
            LLaMA(
                self._config(
                    tele={"layers": [1], "fmap": TINY_FMAP},
                    swa={"layers": [1]},
                )
            )

    def test_rope_scaling_rejected_for_post_summary_rope(self):
        """rope_tables takes a base, not a scaling schedule; silently dropping
        the scaling would give these layers a different RoPE than configured."""
        with self.assertRaises(ValueError):
            LLaMA(
                self._config(
                    rope_scaling={"factor": 8.0},
                    tele={"layers": [1], "fmap": TINY_FMAP},
                )
            )

    def test_resolve_layer_attn(self):
        config = self._config(
            tele={"layers": [1, 2], "fmap": TINY_FMAP}, swa={"layers": [3]}
        )
        self.assertEqual(_resolve_layer_attn(config, 0)[0], "attn")
        self.assertEqual(_resolve_layer_attn(config, 1)[0], "tele")
        self.assertEqual(_resolve_layer_attn(config, 3)[0], "swa")

    def test_model_forward_and_mixed_cache(self):
        model = LLaMA(
            self._config(
                tele={"layers": [1], "fmap": TINY_FMAP, "cache_size": 16},
            )
        )
        model.reset_parameters()
        model.eval()
        ids = torch.randint(0, 64, (1, 24))
        with torch.no_grad():
            # LLaMA.forward always returns (preds, aux); aux is the cache when
            # use_cache=True and a placeholder otherwise.
            logits, _ = model(ids)
            self.assertEqual(tuple(logits.shape), (1, 24, 64))
            _, cache = model(ids, use_cache=True)
        # the telescoping layer contributes a DecodeState; the rest keep tuples
        self.assertIsInstance(cache[1], DecodeState)
        self.assertNotIsInstance(cache[0], DecodeState)

    def test_model_stepwise_decode_matches_full_forward(self):
        model = LLaMA(
            self._config(
                tele={"layers": [1, 2], "fmap": TINY_FMAP, "cache_size": 16},
            )
        )
        model.reset_parameters()
        model.eval()
        ids = torch.randint(0, 64, (1, 30))
        with torch.no_grad():
            full, _ = model(ids, only_last_token=True)
            logits, cache = model(ids[:, :26], use_cache=True)
            for i in range(26, 30):
                logits, cache = model(
                    ids[:, i : i + 1],
                    past_key_value_states=cache,
                    use_cache=True,
                    only_last_token=True,
                )
        torch.testing.assert_close(logits, full, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
