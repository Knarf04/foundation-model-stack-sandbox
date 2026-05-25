import unittest

import torch

from fms.modules.attention import (
    MultiHeadAttention,
    _make_sliding_window_causal_mask,
)
from fms.modules.positions import RotaryEmbedding


def _build_mha(sliding_window: int, with_rope: bool = True):
    emb_dim = 32
    nheads, kvheads = 4, 2
    emb_kq = emb_v = emb_dim // nheads
    rope = (
        RotaryEmbedding(emb_kq, max_seq_len=64) if with_rope else None
    )
    mha = MultiHeadAttention(
        emb_dim=emb_dim,
        emb_kq=emb_kq,
        emb_v=emb_v,
        nheads=nheads,
        kvheads=kvheads,
        position_encoder=rope,
        fused=False,
        sliding_window=sliding_window,
    )
    mha.eval()
    return mha


class SlidingWindowMaskTests(unittest.TestCase):
    def test_mask_shape_and_pattern(self):
        """Each query t attends exactly to keys in [max(0, t-W+1), t]."""
        L = 8
        W = 4
        m = _make_sliding_window_causal_mask(L, L, W, torch.device("cpu"))
        self.assertEqual(m.shape, (L, L))
        self.assertEqual(m.dtype, torch.bool)
        for t in range(L):
            for j in range(L):
                expected = (j <= t) and (j >= max(0, t - W + 1))
                self.assertEqual(
                    bool(m[t, j].item()),
                    expected,
                    msg=f"mask[{t},{j}] expected {expected}",
                )

    def test_mask_decode_step(self):
        """Decode step: q_len=1, k_len=L. Single query at position L-1 attends to
        the most-recent W keys."""
        L = 10
        W = 4
        m = _make_sliding_window_causal_mask(1, L, W, torch.device("cpu"))
        self.assertEqual(m.shape, (1, L))
        attended = m[0].tolist()
        # Indices L-W..L-1 should be True; everything before False.
        expected = [j >= L - W for j in range(L)]
        self.assertEqual(attended, expected)


class SlidingWindowAttentionTests(unittest.TestCase):
    def test_batched_no_position_ids_raises(self):
        """Local position counter is single-stream; batch>1 without explicit
        position_ids must raise."""
        mha = _build_mha(sliding_window=4, with_rope=True)
        x = torch.randn(2, 6, 32)  # batch_size = 2
        with self.assertRaises(ValueError):
            mha(x, use_cache=True)

    def test_batched_explicit_position_ids_raises(self):
        """Local counter only supports batch=1, even with explicit positions."""
        mha = _build_mha(sliding_window=4, with_rope=True)
        x = torch.randn(2, 6, 32)
        pos = torch.arange(6, dtype=torch.long).unsqueeze(0).expand(2, -1)
        with self.assertRaises(ValueError):
            mha(x, position_ids=pos, use_cache=True)

    def test_position_counter_lifecycle(self):
        """curr_id starts at 0, advances by q_len each forward, syncs to last+1
        when explicit position_ids is supplied, and resets via the helper."""
        mha = _build_mha(sliding_window=4, with_rope=True)
        self.assertEqual(mha.curr_id, 0)

        x_prefill = torch.randn(1, 10, 32)
        with torch.no_grad():
            _, cache = mha(x_prefill, use_cache=True)
        self.assertEqual(mha.curr_id, 10)

        x_decode = torch.randn(1, 1, 32)
        with torch.no_grad():
            _, cache = mha(
                x_decode, past_key_value_state=cache, use_cache=True
            )
        self.assertEqual(mha.curr_id, 11)

        # Explicit position_ids must sync the counter.
        with torch.no_grad():
            mha(
                torch.randn(1, 1, 32),
                position_ids=torch.tensor([[20]], dtype=torch.long),
                past_key_value_state=cache,
                use_cache=True,
            )
        self.assertEqual(mha.curr_id, 21)

        mha.reset_position_counter()
        self.assertEqual(mha.curr_id, 0)

    def test_cache_bound(self):
        """Decode for L > W and assert returned cache is always capped at W."""
        sliding_window = 4
        L = 12
        mha = _build_mha(sliding_window=sliding_window, with_rope=True)
        mha.reset_position_counter()
        x = torch.randn(1, L, 32)

        cache = None
        with torch.no_grad():
            for t in range(L):
                _, cache = mha(
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

        mha = _build_mha(sliding_window=sliding_window, with_rope=True)
        x = torch.randn(B, L, D)

        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out_prefill = mha(x, position_ids=pos, use_cache=False)

        mha.reset_position_counter()
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
        torch.testing.assert_close(
            out_prefill, out_decode, atol=1e-4, rtol=1e-4
        )

    def test_gate_path_active(self):
        """Zeroing gate_proj must zero the output (no bias on the gate, and
        the default dense path has use_bias=False). Confirms the gate is
        actually on the output path, i.e. this is gated SWA, not vanilla SWA."""
        torch.manual_seed(0)
        mha = _build_mha(sliding_window=4, with_rope=True)
        x = torch.randn(1, 6, 32)
        pos = torch.arange(6, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            out_normal = mha(x, position_ids=pos, use_cache=False)
            self.assertFalse(torch.allclose(out_normal, torch.zeros_like(out_normal)))

            mha.gate_proj.weight.data.zero_()
            out_zero_gate = mha(x, position_ids=pos, use_cache=False)
        torch.testing.assert_close(out_zero_gate, torch.zeros_like(out_zero_gate))


if __name__ == "__main__":
    unittest.main()
