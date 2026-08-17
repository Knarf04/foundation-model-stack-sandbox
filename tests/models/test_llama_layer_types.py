import unittest

import torch

from fms.models.llama import LLaMA, LLaMAConfig
from fms.modules.attention import MultiHeadAttention
from fms.modules.sliding_window_attention import (
    SlidingWindowMultiHeadAttention,
    _flex_attention_available,
)


def _small_config(**kwargs) -> LLaMAConfig:
    return LLaMAConfig(
        src_vocab_size=128,
        emb_dim=64,
        nheads=4,
        kvheads=2,
        nlayers=4,
        max_expected_seq_len=64,
        hidden_grow_factor=2.0,
        multiple_of=1,
        **kwargs,
    )


class LlamaLayerTypeConfigTests(unittest.TestCase):
    def test_default_all_full_attention(self):
        model = LLaMA(_small_config())
        for block in model.layers:
            self.assertIs(type(block.attn), MultiHeadAttention)

    def test_overlapping_layers_raise(self):
        config = _small_config(
            attn={"layers": [1, 2]},
            swa={"layers": [2, 3]},
        )
        with self.assertRaises(ValueError):
            LLaMA(config)

    def test_out_of_range_layers_raise(self):
        with self.assertRaises(ValueError):
            LLaMA(_small_config(attn={"layers": [7]}))

    def test_missing_layers_key_raises(self):
        with self.assertRaises(ValueError):
            LLaMA(_small_config(swa={"window_size": 8}))


@unittest.skipUnless(_flex_attention_available, "flex_attention requires torch >= 2.5")
class LlamaMixedLayerTypeTests(unittest.TestCase):
    def test_mixed_layer_types_and_overrides(self):
        config = _small_config(
            attn={"layers": [1], "num_heads": 8, "num_kv_heads": 4},
            swa={"layers": [2, 3], "window_size": 8},
        )
        model = LLaMA(config)

        self.assertIs(type(model.layers[0].attn), MultiHeadAttention)
        self.assertIs(type(model.layers[1].attn), MultiHeadAttention)
        self.assertIsInstance(
            model.layers[2].attn, SlidingWindowMultiHeadAttention
        )
        self.assertIsInstance(
            model.layers[3].attn, SlidingWindowMultiHeadAttention
        )

        # per-type overrides applied
        self.assertEqual(model.layers[1].attn.nheads, 8)
        self.assertEqual(model.layers[1].attn.kvheads, 4)
        self.assertEqual(model.layers[1].attn.emb_kq_per_head, 64 // 8)
        self.assertEqual(model.layers[2].attn.sliding_window, 8)

        # default layers keep the global config
        self.assertEqual(model.layers[0].attn.nheads, 4)
        self.assertEqual(model.layers[0].attn.kvheads, 2)

    def test_sinks_wiring(self):
        """"sinks": true in a per-type dict creates the per-head sink parameter
        on exactly those layers; all other layers stay sink-free."""
        config = _small_config(
            attn={"layers": [0], "sinks": True},
            swa={"layers": [2], "window_size": 8, "sinks": True},
        )
        model = LLaMA(config)
        self.assertIsInstance(model.layers[0].attn.sinks, torch.nn.Parameter)
        self.assertEqual(model.layers[0].attn.sinks.shape, (4,))
        self.assertIsNone(model.layers[1].attn.sinks)
        self.assertIsInstance(model.layers[2].attn.sinks, torch.nn.Parameter)
        self.assertIsNone(model.layers[3].attn.sinks)

    def test_mixed_model_forward(self):
        config = _small_config(swa={"layers": [1, 3], "window_size": 8})
        model = LLaMA(config)
        model.eval()
        ids = torch.randint(0, 128, (1, 16))
        with torch.no_grad():
            out, _ = model(ids)
        self.assertEqual(out.shape, (1, 16, 128))


if __name__ == "__main__":
    unittest.main()
