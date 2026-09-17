"""
Short causal depthwise convolution: the layer, its decode cache, and the
one-token step that consumes it.

An optional stage on projected K/V, before the merge weights and the tree. The
attention cache holds POST-conv K/V; this cache is PRE-conv rows -- different
objects, do not conflate them.
"""

from typing import Tuple

import torch
from torch import nn

from fms.modules.telescoping_attn.checks import check_shape

def _kernel_size(weight: torch.Tensor, channels: int) -> int:
    """Validate a [C, K] conv weight against C. -> K."""
    if weight.dim() != 2 or weight.shape[0] != channels:
        raise ValueError(
            f"conv weight shape {tuple(weight.shape)} incompatible with "
            f"C={channels}; expected [C, K]"
        )
    K = weight.shape[1]
    if K <= 0:
        raise ValueError(f"conv kernel size must be positive, got {K}")
    return K

def short_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x: [B, N, C]  weight: [C, K]  -> [B, N, C], causal depthwise + residual.
    y[t, c] = sum_j weight[c, j] * x[t-K+1+j, c]: past-and-current only, no
    channel mixing (groups=C), no bias, no activation.

    The residual is ADDED IN FP32 and only the sum is cast back -- contract, not
    x + conv(x).to(dtype). minimal_reference._short_conv is the independent
    oracle; the suite, not shared code, keeps the two equal.

    TODO(padding/packing): assumes one continuous sequence per batch row -- no
    state reset across padding or packed-example boundaries (a mask gates
    attention, it does not zero K/V rows).
    """
    B, N, C = x.shape
    K = _kernel_size(weight, C)
    input_dtype = x.dtype
    x_fp32 = x.float()
    # conv1d is cross-correlation; padding=K-1 plus the [:N] crop gives the
    # causal sum above, no kernel flip.
    y = torch.nn.functional.conv1d(
        x_fp32.transpose(1, 2),
        weight.float().unsqueeze(1),
        padding=K - 1,
        groups=C,
    )[:, :, :N].transpose(1, 2)
    return (x_fp32 + y).to(input_dtype)

class ShortConv1d(nn.Module):
    """
    nn.Module wrapper over `short_conv`. weight: [channels, kernel_size].

    Zero-init on purpose: the residual makes W=0 an identity, so enabling the
    conv starts exactly from the original attention model. That is a finetuning
    choice, not part of the definition.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(channels, kernel_size))

    def reset_parameters(self):
        self.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return short_conv(x, self.weight)

def conv_history(x_pre: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    -> the last K-1 rows of x_pre [B, N, C], newest last, left zero-padded if
    N < K-1. K == 1 needs the explicit branch: x[:, -0:] is the whole tensor.
    """
    history = kernel_size - 1
    if history == 0:
        return x_pre[:, :0]
    pad_rows = max(history - x_pre.shape[1], 0)
    return torch.nn.functional.pad(x_pre, (0, 0, pad_rows, 0))[:, -history:]

def short_conv_step(
    x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [B, 1, C] current PRE-conv row  state: [B, K-1, C] previous rows, newest
    last  weight: [C, K]  -> (y [B, 1, C], new_state [B, K-1, C])
    Same FP32 residual contract as short_conv.
    """
    if x.dim() != 3 or x.shape[1] != 1:
        raise ValueError(f"x must be [B, 1, C], got {tuple(x.shape)}")
    B, _, C = x.shape
    K = _kernel_size(weight, C)
    check_shape("conv state", state, (B, K - 1, C))
    full = torch.cat([state, x], dim=1)  # [B, K, C]
    y = (full.float() * weight.float().t().unsqueeze(0)).sum(dim=1)  # [B, C]
    out = (x[:, 0].float() + y).to(x.dtype).unsqueeze(1)  # [B, 1, C]
    return out, full[:, 1:]
