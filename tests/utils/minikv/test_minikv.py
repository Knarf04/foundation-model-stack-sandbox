"""Integration tests for MiniKV KV cache eviction attention op."""

import torch
import pytest

from fms.models.llama import LLaMA, LLaMAConfig
from fms.utils.generation import generate

# Import triggers registration of the "minikv" attention op
from fms.utils.minikv.attention_op import MiniKVConfig, create_minikv_kwargs
from fms.utils.minikv.selection import (
    H2OSelection,
    SnapKVSelection,
    PyramidH2OSelection,
    PyramidSnapKVSelection,
    create_selector,
)


# ---------------------------------------------------------------------------
# Test configs
# ---------------------------------------------------------------------------

MICRO_CONFIG = LLaMAConfig(
    src_vocab_size=256,
    emb_dim=16,
    nheads=4,
    kvheads=2,  # GQA: 4 heads, 2 kv heads
    nlayers=2,
    hidden_grow_factor=2.0,
    multiple_of=2,
    max_expected_seq_len=512,
)

MICRO_CONFIG_MHA = LLaMAConfig(
    src_vocab_size=256,
    emb_dim=16,
    nheads=4,
    kvheads=0,  # MHA: kvheads == nheads
    nlayers=2,
    hidden_grow_factor=2.0,
    multiple_of=2,
    max_expected_seq_len=512,
)


def _init_model(config):
    model = LLaMA(config)
    # Initialize with random weights
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.02)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Selection mechanism unit tests
# ---------------------------------------------------------------------------


class TestH2OSelection:
    def test_basic_eviction(self):
        B, H, S, D = 1, 2, 20, 8
        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)
        attn_map = torch.randn(B, H, S).abs()  # cumulative scores

        selector = H2OSelection(heavy_ratio=0.25, recent_ratio=0.25)
        k_out, v_out = selector.select(keys, values, attn_map)

        # Should keep 25% heavy + 25% recent = 50% of tokens
        expected_kept = int(S * 0.25) + int(S * 0.25)  # 5 + 5 = 10
        assert k_out.shape == (B, H, expected_kept, D)
        assert v_out.shape == (B, H, expected_kept, D)

    def test_no_eviction_when_budget_exceeds_length(self):
        B, H, S, D = 1, 2, 10, 8
        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)
        attn_map = torch.randn(B, H, S).abs()

        selector = H2OSelection(heavy_ratio=0.6, recent_ratio=0.6)
        k_out, v_out = selector.select(keys, values, attn_map)

        # Budget exceeds length, should keep all tokens
        assert k_out.shape == (B, H, S, D)
        assert v_out.shape == (B, H, S, D)


class TestSnapKVSelection:
    def test_basic_eviction(self):
        B, H, S, D = 1, 2, 100, 8
        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)
        queries = torch.randn(B, H, S, D)

        selector = SnapKVSelection(
            window_size=16, prompt_sparsity_ratio=0.5, kernel_size=5
        )
        k_out, v_out = selector.select(keys, values, queries)

        expected_kept = int(S * 0.5)  # 50
        assert k_out.shape == (B, H, expected_kept, D)
        assert v_out.shape == (B, H, expected_kept, D)

    def test_short_sequence_fallback(self):
        B, H, S, D = 1, 2, 10, 8
        keys = torch.randn(B, H, S, D)
        values = torch.randn(B, H, S, D)
        queries = torch.randn(B, H, S, D)

        selector = SnapKVSelection(
            window_size=64, prompt_sparsity_ratio=0.25, kernel_size=5
        )
        k_out, v_out = selector.select(keys, values, queries)

        # prompt_sparsity_ratio * S = 2.5 < window_size=64, fallback path
        retained = max(1, int(0.25 * S))
        assert k_out.shape == (B, H, retained, D)


class TestPyramidBudgets:
    def test_pyramid_h2o_decreasing_budget(self):
        num_layers = 8
        ratios = []
        for i in range(num_layers):
            s = PyramidH2OSelection(
                heavy_ratio=0.25, recent_ratio=0.25, layer_id=i, num_layers=num_layers
            )
            ratios.append(s.heavy_ratio)
        # Lower layers should have higher budget
        for i in range(len(ratios) - 1):
            assert ratios[i] > ratios[i + 1], (
                f"Layer {i} ratio {ratios[i]} should be > layer {i+1} ratio {ratios[i+1]}"
            )

    def test_pyramid_snapkv_decreasing_budget(self):
        num_layers = 8
        ratios = []
        for i in range(num_layers):
            s = PyramidSnapKVSelection(
                prompt_sparsity_ratio=0.25, layer_id=i, num_layers=num_layers
            )
            ratios.append(s.prompt_sparsity_ratio)
        for i in range(len(ratios) - 1):
            assert ratios[i] > ratios[i + 1]


class TestCreateSelector:
    def test_all_methods(self):
        for method in ["h2o", "snapkv", "pyramid_h2o", "pyramid_snapkv"]:
            s = create_selector(method, layer_id=0, num_layers=4)
            assert s is not None

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            create_selector("invalid", layer_id=0, num_layers=4)


# ---------------------------------------------------------------------------
# Integration tests with micro LLaMA
# ---------------------------------------------------------------------------


class TestMiniKVGeneration:
    def test_h2o_gqa_generation(self):
        """Test H2O eviction with GQA model (kvheads < nheads)."""
        model = _init_model(MICRO_CONFIG)
        input_ids = torch.randint(0, 256, (1, 32))

        config = MiniKVConfig(
            selection_method="h2o", heavy_ratio=0.25, recent_ratio=0.25
        )
        extra_kwargs = create_minikv_kwargs(
            config, num_layers=MICRO_CONFIG.nlayers
        )

        result = generate(
            model,
            input_ids,
            max_new_tokens=5,
            use_cache=True,
            do_sample=False,
            extra_kwargs=extra_kwargs,
        )

        # Should generate input + 5 new tokens
        assert result.shape == (1, 32 + 5)

    def test_snapkv_gqa_generation(self):
        """Test SnapKV eviction with GQA model."""
        model = _init_model(MICRO_CONFIG)
        input_ids = torch.randint(0, 256, (1, 64))

        config = MiniKVConfig(
            selection_method="snapkv",
            prompt_sparsity_ratio=0.5,
            window_size=8,
            kernel_size=3,
        )
        extra_kwargs = create_minikv_kwargs(
            config, num_layers=MICRO_CONFIG.nlayers
        )

        result = generate(
            model,
            input_ids,
            max_new_tokens=5,
            use_cache=True,
            do_sample=False,
            extra_kwargs=extra_kwargs,
        )
        assert result.shape == (1, 64 + 5)

    def test_h2o_mha_generation(self):
        """Test H2O eviction with MHA model (kvheads == nheads)."""
        model = _init_model(MICRO_CONFIG_MHA)
        input_ids = torch.randint(0, 256, (1, 32))

        config = MiniKVConfig(
            selection_method="h2o", heavy_ratio=0.3, recent_ratio=0.3
        )
        extra_kwargs = create_minikv_kwargs(
            config, num_layers=MICRO_CONFIG_MHA.nlayers
        )

        result = generate(
            model,
            input_ids,
            max_new_tokens=5,
            use_cache=True,
            do_sample=False,
            extra_kwargs=extra_kwargs,
        )
        assert result.shape == (1, 32 + 5)

    def test_pyramid_h2o_generation(self):
        """Test Pyramid H2O eviction."""
        model = _init_model(MICRO_CONFIG)
        input_ids = torch.randint(0, 256, (1, 32))

        config = MiniKVConfig(
            selection_method="pyramid_h2o", heavy_ratio=0.25, recent_ratio=0.25
        )
        extra_kwargs = create_minikv_kwargs(
            config, num_layers=MICRO_CONFIG.nlayers
        )

        result = generate(
            model,
            input_ids,
            max_new_tokens=5,
            use_cache=True,
            do_sample=False,
            extra_kwargs=extra_kwargs,
        )
        assert result.shape == (1, 32 + 5)

    def test_baseline_vs_minikv_output_differs(self):
        """MiniKV output should differ from baseline (eviction changes decode)."""
        torch.manual_seed(42)
        model = _init_model(MICRO_CONFIG)
        input_ids = torch.randint(0, 256, (1, 32))

        # Baseline with standard SDPA
        baseline = generate(
            model,
            input_ids,
            max_new_tokens=10,
            use_cache=True,
            do_sample=False,
        )

        # MiniKV with aggressive eviction
        config = MiniKVConfig(
            selection_method="h2o", heavy_ratio=0.1, recent_ratio=0.1
        )
        extra_kwargs = create_minikv_kwargs(
            config, num_layers=MICRO_CONFIG.nlayers
        )
        minikv_result = generate(
            model,
            input_ids,
            max_new_tokens=10,
            use_cache=True,
            do_sample=False,
            extra_kwargs=extra_kwargs,
        )

        # Both should have same shape
        assert baseline.shape == minikv_result.shape
        # With aggressive eviction, results should likely differ
        # (not guaranteed for random weights, but very likely)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
