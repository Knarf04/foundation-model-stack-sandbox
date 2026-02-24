"""
GatedDeltaNet model for foundation-model-stack (fms).

Mirrors LLaMABlock structure exactly: LayerNormParameterized pre-norms,
GatedLinearUnit (SwiGLU) MLP, pre-norm → mixer → residual pattern.

The sequence mixer (self.attn in GDNBlock) is:
  - fla.layers.gated_deltanet.GatedDeltaNet  for pure-GDN layers
  - fms.modules.attention.MultiHeadAttention for hybrid attention layers
    (those listed in GatedDeltaNetConfig.attn["layers"])

Model layout (mirrors LLaMA):
    nn.Embedding → GDNBlock × N → LayerNormParameterized → nn.Linear (lm_head)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from fla.layers.gated_deltanet import GatedDeltaNet as _FLAGDNLayer

from fms import models
from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


@dataclass
class GatedDeltaNetConfig(ModelConfig):
    """
    FMS-style configuration for GatedDeltaNet language models.

    `src_vocab_size` follows fms convention (cf. LLaMAConfig, BambaConfig).

    Hybrid full-attention layers
    ----------------------------
    Set `attn` to a dict to interleave fms MultiHeadAttention with GDN layers::

        attn={
            "layers": [6, 13, 20, 27],  # layer indices that become full-attn
            "num_heads": 12,
            "num_kv_heads": 4,
            "qkv_bias": False,
            "window_size": -1,       # -1 = global attention
            "rope_theta": 500000.0,
        }
    """

    src_vocab_size: int = 128256
    hidden_size: int = 2048
    num_hidden_layers: int = 21
    num_heads: int = 6
    head_dim: int = 256
    num_v_heads: Optional[int] = None
    expand_v: float = 2.0
    hidden_ratio: int = 4
    intermediate_size: Optional[int] = None
    use_short_conv: bool = True
    conv_size: int = 4
    use_gate: bool = True
    allow_neg_eigval: bool = False
    attn_mode: str = "chunk"
    norm_eps: float = 1e-6
    max_position_embeddings: int = 4096
    pad_id: int = -1
    initializer_range: float = 0.02
    # Optional hybrid-attention config dict (see class docstring).
    attn: Optional[dict] = None
    # Legacy fields kept for config-file compatibility; not used by GDNBlock.
    fuse_norm: bool = True
    fuse_swiglu: bool = True


class GDNBlock(nn.Module):
    """
    A transformer block with fla's GatedDeltaNet as the sequence mixer.

    Structurally identical to LLaMABlock:
      - self.ln           : LayerNormParameterized  (pre-attention norm)
      - self.attn         : _FLAGDNLayer or MultiHeadAttention
      - self.ff_ln        : LayerNormParameterized  (pre-MLP norm)
      - self.ff_sub_layer : GatedLinearUnit (SwiGLU)

    For layers listed in config.attn["layers"], self.attn is fms
    MultiHeadAttention (Universal Attention).  All other layers use fla's
    efficient GatedDeltaNet mixer.
    """

    def __init__(
        self,
        config: GatedDeltaNetConfig,
        layer_idx: int,
        rot_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.ln = LayerNormParameterized(
            config.hidden_size,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = LayerNormParameterized(
            config.hidden_size,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=config.norm_eps,
            use_high_precision_pow=True,
        )

        # --- Sequence mixer ---
        attn_layers = config.attn.get("layers", []) if config.attn else []
        if layer_idx in attn_layers:
            # Hybrid layer: fms MultiHeadAttention (same as LLaMABlock)
            _ac = config.attn
            _nheads = _ac.get("num_heads", config.num_heads)
            _kvheads = _ac.get("num_kv_heads", _nheads)
            _emb_kq = config.hidden_size // _nheads
            self.attn = MultiHeadAttention(
                emb_dim=config.hidden_size,
                emb_kq=_emb_kq,
                emb_v=_emb_kq,
                nheads=_nheads,
                kvheads=_kvheads,
                use_bias=_ac.get("qkv_bias", False),
                position_encoder=rot_emb,
            )
        else:
            # GDN layer: fla's efficient GatedDeltaNet mixer
            self.attn = _FLAGDNLayer(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_v_heads=config.num_v_heads,
                use_gate=config.use_gate,
                use_short_conv=config.use_short_conv,
                allow_neg_eigval=config.allow_neg_eigval,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
            )

        # --- MLP: SwiGLU via GatedLinearUnit, identical to LLaMABlock ---
        hidden_grow_factor = (
            config.intermediate_size / config.hidden_size
            if config.intermediate_size is not None
            else float(config.hidden_ratio)
        )
        self.ff_sub_layer = GatedLinearUnit(
            config.hidden_size,
            hidden_grow_factor=hidden_grow_factor,
            multiple_of=1,
            activation_fn=str_to_activation("swish"),
            p_dropout=0.0,
            use_bias=False,
            fused=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids=None,
        past_key_value_state=None,
        use_cache: bool = False,
        **attn_kwargs,
    ):
        # Pre-norm → mixer → residual  (identical pattern to LLaMABlock)
        residual = x
        x = self.ln(x)

        if isinstance(self.attn, MultiHeadAttention):
            # fms MHA interface — identical to LLaMABlock
            x = self.attn(
                q=x,
                position_ids=position_ids,
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
                **attn_kwargs,
            )
            cache = None
            if use_cache:
                x, cache = x
            else:
                x, _ = x
        else:
            # fla GatedDeltaNet layer interface
            out = self.attn(
                hidden_states=x,
                past_key_values=past_key_value_state,
                use_cache=use_cache,
            )
            cache = None
            if isinstance(out, tuple):
                x = out[0]
                if use_cache and len(out) > 1:
                    cache = out[-1]
            else:
                x = out

        x = x + residual

        # Pre-norm → MLP → residual
        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        x = x + residual

        if use_cache:
            return x, cache
        else:
            return x, None


class GatedDeltaNet(nn.Module):
    """
    FMS-style GatedDeltaNet causal language model.

    Model layout (mirrors LLaMA):
        nn.Embedding → GDNBlock × N → LayerNormParameterized → nn.Linear (lm_head)
    """

    def __init__(
        self,
        config: Optional[GatedDeltaNetConfig] = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = GatedDeltaNetConfig()
        self.config = config.updated(**kwargs)

        padding_idx = self.config.pad_id if self.config.pad_id >= 0 else None
        self.embedding = nn.Embedding(
            self.config.src_vocab_size,
            self.config.hidden_size,
            padding_idx=padding_idx,
        )

        # Shared RotaryEmbedding for hybrid attention layers.
        # RotaryEmbedding is not an nn.Module; freqs are cached lazily on first forward.
        self._rot_emb = None
        if self.config.attn:
            _ac = self.config.attn
            _nheads = _ac.get("num_heads", self.config.num_heads)
            self._rot_emb = RotaryEmbedding(
                dim=self.config.hidden_size // _nheads,
                ratio=_ac.get("rope_theta", 10_000.0),
                max_seq_len=self.config.max_position_embeddings,
            )

        self.layers = nn.ModuleList([
            GDNBlock(self.config, layer_idx=i, rot_emb=self._rot_emb)
            for i in range(self.config.num_hidden_layers)
        ])

        self.dec_norm = LayerNormParameterized(
            self.config.hidden_size,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )

        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.src_vocab_size,
            bias=False,
        )

    def get_config(self) -> GatedDeltaNetConfig:
        return self.config

    @classmethod
    def from_config(cls, config: GatedDeltaNetConfig) -> "GatedDeltaNet":
        return cls(config)

    def reset_parameters(self):
        """
        Weight initialization mirroring LLaMA's reset_parameters.

        LayerNormParameterized, MultiHeadAttention, and GatedLinearUnit all
        expose their own reset_parameters().  fla's GatedDeltaNet layer
        initializes its own weights during construction.
        """
        for m in self.modules():
            if isinstance(m, (MultiHeadAttention, GatedLinearUnit, LayerNormParameterized)):
                m.reset_parameters()
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_key_values=None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : LongTensor [batch, seq_len]

        Returns
        -------
        logits : FloatTensor [batch, seq_len, vocab_size]
        """
        hidden = self.embedding(x)
        for layer in self.layers:
            hidden, _ = layer(hidden, use_cache=False)
        hidden = self.dec_norm(hidden)
        return self.lm_head(hidden)


# ---------------------------------------------------------------------------
# FMS model registry
# ---------------------------------------------------------------------------

_architecture_name = "gdn"


def _gdn_factory(config: GatedDeltaNetConfig):
    def factory(**kwargs):
        return GatedDeltaNet(config, **kwargs)
    return factory


models.register_model(
    _architecture_name,
    "gdn_1b",
    _gdn_factory(
        GatedDeltaNetConfig(
            src_vocab_size=128256,
            hidden_size=1536,
            num_hidden_layers=28,
            num_heads=12,
            head_dim=64,
            expand_v=2.0,
            hidden_ratio=4,
            use_short_conv=True,
            conv_size=4,
            use_gate=True,
            attn_mode="chunk",
            max_position_embeddings=4096,
        )
    ),
)

models.register_model(
    _architecture_name,
    "gdn_1b_hybrid",
    _gdn_factory(
        GatedDeltaNetConfig(
            src_vocab_size=128256,
            hidden_size=1536,
            num_hidden_layers=28,
            num_heads=12,
            head_dim=64,
            expand_v=2.0,
            hidden_ratio=4,
            use_short_conv=True,
            conv_size=4,
            use_gate=True,
            attn_mode="chunk",
            max_position_embeddings=4096,
            attn={
                "layers": [6, 13, 20, 27],
                "num_heads": 12,
                "num_kv_heads": 4,
                "qkv_bias": False,
                "window_size": -1,
                "rope_theta": 500000.0,
            },
        )
    ),
)
