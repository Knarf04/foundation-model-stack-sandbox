from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fla.models.gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetConfig as FLAGDNConfig
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetModel

from fms.models.hf.gated_delta_net.configuration_gated_delta_net_hf import HFAdaptedGDNConfig
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture


class HFAdaptedGDNDecoder(HFDecoder):
    """Adapter for the fla GatedDeltaNetModel decoder"""

    def __init__(self, model: GatedDeltaNetModel, config: PretrainedConfig):
        super().__init__(model, config, attention_mask_dim=3)

    def set_input_embeddings(self, value: nn.Module):
        self.model.embeddings = value

    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        # fla expects attention_mask as 2D [batch, seq_len] or None.
        # The base adapter expands it to 3D causal; convert back for fla.
        if attention_mask is not None and attention_mask.dim() > 2:
            attention_mask = None

        fla_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=fla_output.last_hidden_state,
            past_key_values=fla_output.past_key_values,
        )


class HFAdaptedGDNHeadless(HFDecoderModelArchitecture):
    """Adapter for the base GatedDeltaNet architecture using the fla model"""

    config_class = HFAdaptedGDNConfig
    base_model_prefix = "hf_adapted_gated_delta_net"

    def __init__(
        self,
        config: PretrainedConfig,
        decoder: Optional[nn.Module] = None,
        embedding: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        if decoder is None or embedding is None:
            # Build a fla GatedDeltaNetModel from config
            fla_config = self._to_fla_config(config)
            fla_model = GatedDeltaNetModel(fla_config)
            decoder = fla_model if decoder is None else decoder
            embedding = fla_model.embeddings if embedding is None else embedding

        decoder = HFAdaptedGDNDecoder(decoder, config)
        super().__init__(decoder, embedding, config, *args, **kwargs)

    @staticmethod
    def _to_fla_config(config: PretrainedConfig) -> FLAGDNConfig:
        """Convert HFAdaptedGDNConfig to fla GatedDeltaNetConfig."""
        config_dict = config.to_dict()
        # Remove HF-only keys that fla config doesn't accept
        for key in ("model_type", "transformers_version", "is_decoder",
                     "torch_dtype", "_name_or_path"):
            config_dict.pop(key, None)
        return FLAGDNConfig(**{
            k: v for k, v in config_dict.items()
            if k in FLAGDNConfig().__dict__
        })

    def _prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs,
    ) -> dict:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            **model_kwargs,
        }


class HFAdaptedGDNForCausalLM(LMHeadModelLMHeadMixin, HFAdaptedGDNHeadless):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["embedding.weight", "lm_head.weight"]

    def __init__(self, config: HFAdaptedGDNConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    @classmethod
    def _hf_model_from_fms(
        cls, model, config: HFAdaptedGDNConfig
    ) -> "HFAdaptedGDNForCausalLM":
        """
        Build an HFAdaptedGDNForCausalLM from either:
        - a fla GatedDeltaNetForCausalLM (extracts model.model, model.lm_head)
        - a fla GatedDeltaNetModel (uses it directly as decoder)
        - an fms GatedDeltaNet (legacy path)
        """
        if isinstance(model, GatedDeltaNetForCausalLM):
            # fla ForCausalLM: inner model is model.model (GatedDeltaNetModel)
            return cls(
                config=config,
                decoder=model.model,
                embedding=model.model.embeddings,
                lm_head=model.lm_head,
            )
        elif isinstance(model, GatedDeltaNetModel):
            return cls(
                config=config,
                decoder=model,
                embedding=model.embeddings,
            )
        else:
            # Legacy fms GatedDeltaNet path
            return cls(
                config=config,
                decoder=model,
                embedding=model.embedding,
                lm_head=model.lm_head,
            )

    def _lm_head(self, input_ids, *args, **kwargs):
        return self.lm_head(input_ids)
