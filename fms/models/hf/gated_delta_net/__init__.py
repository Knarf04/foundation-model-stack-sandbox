import os
from typing import Union

from fms.models.hf.gated_delta_net.modeling_gated_delta_net_hf import HFAdaptedGDNForCausalLM


def get_model(model_name_or_path: Union[str, os.PathLike]) -> HFAdaptedGDNForCausalLM:
    """
    Get a Huggingface adapted FMS model from a fla GatedDeltaNetForCausalLM checkpoint.

    Parameters
    ----------
    model_name_or_path: Union[str, os.PathLike]
        Either the name of the model in huggingface hub or the absolute path to
        the model checkpoint

    Returns
    -------
    HFAdaptedGDNForCausalLM
        A Huggingface adapted FMS wrapper around fla's GatedDeltaNet
    """
    import torch
    from fla.models.gated_deltanet import GatedDeltaNetForCausalLM

    from fms.models.hf.utils import register_fms_models

    register_fms_models()
    fla_model = GatedDeltaNetForCausalLM.from_pretrained(model_name_or_path)
    result_model: HFAdaptedGDNForCausalLM = HFAdaptedGDNForCausalLM.from_fms_model(
        fla_model,
        torch_dtype=torch.float16,
    )
    return result_model
