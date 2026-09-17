"""Summary-tree construction: merge weights and the dyadic builder."""

import math
from typing import Optional

import torch

from fms.modules.telescoping_attn.checks import check_paired, check_shape
from fms.modules.telescoping_attn.short_conv import short_conv

def compute_summary_weights(
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    q: [B, N, Hq, Dk]  k: [B, N, Hkv, Dk]  -> w: [B, N, Hkv, 1]
    logsumexp_h(q_h^T k / sqrt(Dk)). Phase-1 MERGE weight, not the attention LSE.
    """
    B, N, Hq, Dk = q.shape
    _, Nk, Hkv, Dkk = k.shape

    assert N == Nk
    assert Dk == Dkk
    assert Hq % Hkv == 0

    expansion = Hq // Hkv
    q_grouped = q.reshape(B, N, Hkv, expansion, Dk)

    scores = (
        q_grouped * k.unsqueeze(3)
    ).sum(dim=-1) / math.sqrt(Dk)  # [B, N, Hkv, expansion]

    return torch.logsumexp(scores, dim=3, keepdim=True)

def compute_linear_weights(
    x: torch.Tensor,
    w_proj: torch.Tensor,
) -> torch.Tensor:
    """
    x: [B, N, emb_dim]  w_proj: [Hkv, emb_dim]  -> w: [B, N, Hkv, 1] = x @ w_proj^T
    """
    B, N, E = x.shape
    Hkv, Ew = w_proj.shape

    assert E == Ew

    return x.matmul(w_proj.t()).unsqueeze(-1)

def merge_children(k_children, v_children, w_children, dim: int):
    """
    One dyadic merge along `dim`: alpha = softmax(w), K/V averaged by alpha, w
    combined by logsumexp. -> (k, v, w) parents with `dim` gone.

    The prefill builder (dim=2) and the decode tick (dim=1) share this so they
    cannot drift; decode-equivalence in the suite is what would catch it if
    they did.
    """
    alpha = torch.softmax(w_children, dim=dim)
    return (
        (k_children * alpha).sum(dim=dim),
        (v_children * alpha).sum(dim=dim),
        torch.logsumexp(w_children, dim=dim),
    )

def build_dyadic_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_summary_levels: int,
    detach_weights: bool = False,
    x: Optional[torch.Tensor] = None,
    w_proj: Optional[torch.Tensor] = None,
    k_conv_weight: Optional[torch.Tensor] = None,
    v_conv_weight: Optional[torch.Tensor] = None,
):
    """
    -> (k_levels, v_levels, w_levels), length num_summary_levels + 1, with
    k_levels[l]: [B, N >> l, Hkv, Dk]. Only complete aligned pairs merge.

    Merge weights from q/k, or from x @ w_proj^T if both are given. The conv
    weights apply short_conv to k/v (flattened, c = h*D + d) BEFORE the weights
    and the tree. Each pair is both-or-neither. detach_weights cuts the gradient
    into the merge weights only.
    """
    check_paired("k_conv_weight", k_conv_weight, "v_conv_weight", v_conv_weight)
    if k_conv_weight is not None:
        B, N, Hkv, Dk = k.shape
        Dv = v.shape[-1]
        k = short_conv(
            k.reshape(B, N, Hkv * Dk), k_conv_weight
        ).reshape(B, N, Hkv, Dk)
        v = short_conv(
            v.reshape(B, N, Hkv * Dv), v_conv_weight
        ).reshape(B, N, Hkv, Dv)

    check_paired("x", x, "w_proj", w_proj,
                 "linear weight mode; omit both for QK mode")
    if x is not None:
        w = compute_linear_weights(x, w_proj)
        check_shape("linear weights", w, (*k.shape[:3], 1))
    else:
        w = compute_summary_weights(q, k)
    if detach_weights:
        w = w.detach()

    k_levels = [k]
    v_levels = [v]
    w_levels = [w]

    for _level in range(1, num_summary_levels + 1):
        k_prev = k_levels[-1]
        v_prev = v_levels[-1]
        w_prev = w_levels[-1]

        usable = (k_prev.shape[1] // 2) * 2

        if usable == 0:
            B, _, Hkv, Dk = k_prev.shape
            Dv = v_prev.shape[-1]
            k_levels.append(k_prev.new_empty(B, 0, Hkv, Dk))
            v_levels.append(v_prev.new_empty(B, 0, Hkv, Dv))
            w_levels.append(w_prev.new_empty(B, 0, Hkv, 1))
            continue

        # even/odd siblings onto a new child axis: [B, N_parent, 2, Hkv, D]
        k_ch, v_ch, w_ch = (
            torch.stack([t[:, 0:usable:2], t[:, 1:usable:2]], dim=2)
            for t in (k_prev, v_prev, w_prev)
        )
        k_parent, v_parent, w_parent = merge_children(k_ch, v_ch, w_ch, dim=2)

        k_levels.append(k_parent)
        v_levels.append(v_parent)
        w_levels.append(w_parent)

    return k_levels, v_levels, w_levels
