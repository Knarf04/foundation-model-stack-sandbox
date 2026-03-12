from typing import Optional

from transformers import PretrainedConfig

from fms.models.gdn import GatedDeltaNetConfig


class HFAdaptedGDNConfig(PretrainedConfig):
    model_type = "hf_adapted_gated_delta_net"
    attribute_map = {
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_hidden_layers",
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 21,
        num_heads: int = 6,
        head_dim: int = 256,
        num_v_heads: Optional[int] = None,
        expand_v: float = 2.0,
        hidden_ratio: int = 4,
        intermediate_size: Optional[int] = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        attn_mode: str = "chunk",
        hidden_act: str = "swish",
        norm_eps: float = 1e-6,
        max_position_embeddings: int = 2048,
        pad_token_id: Optional[int] = None,
        initializer_range: float = 0.02,
        attn: Optional[dict] = None,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        use_cache: bool = True,
        eos_token_id: int = 2,
        bos_token_id: int = 1,
        is_decoder: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_v_heads = num_v_heads
        self.expand_v = expand_v
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.attn_mode = attn_mode
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.attn = attn
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            is_decoder=is_decoder,
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_fms_config(cls, config: GatedDeltaNetConfig, **hf_kwargs):
        config_dict = config.as_dict()
        # fms uses src_vocab_size / pad_id; fla uses vocab_size / pad_token_id
        if "src_vocab_size" in config_dict:
            config_dict["vocab_size"] = config_dict.pop("src_vocab_size")
        if "pad_id" in config_dict:
            config_dict["pad_token_id"] = config_dict.pop("pad_id")
        return cls.from_dict(config_dict, **hf_kwargs)

    def to_fla_config(self):
        """Convert this HF adapter config to a fla GatedDeltaNetConfig."""
        from fla.models.gated_deltanet import GatedDeltaNetConfig as FLAGDNConfig
        return FLAGDNConfig(**{
            k: v for k, v in self.to_dict().items()
            if k in FLAGDNConfig().__dict__
        })
