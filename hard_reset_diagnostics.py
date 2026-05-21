"""Hard-reset reference diagnostics for LP-SN Proposition 2b.

These helpers intentionally use the hard-reset LIF spike history. They are
therefore diagnostics for the one-sided hard-reset bound, not the A2
``exact_detached`` soft-reset forward path.
"""
from __future__ import annotations

import math

import torch

from .lpsn_neuron import build_causal_exp_kernel, soft_reset_correction


def _stable_tau(tau: float, eps: float = 1e-6) -> float:
    return max(float(tau), 1.0 + eps)


def hard_reset_lif_reference(
    x_seq: torch.Tensor,
    *,
    tau: float,
    v_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hard-reset LIF pre-threshold membrane, state, and spikes."""
    x_seq = x_seq.detach().float()
    tau = _stable_tau(tau)
    lam = 1.0 - 1.0 / tau
    gain = 1.0 / tau
    v = torch.zeros_like(x_seq[0])
    h_list: list[torch.Tensor] = []
    v_list: list[torch.Tensor] = []
    s_list: list[torch.Tensor] = []
    for t in range(x_seq.shape[0]):
        h = lam * v + gain * x_seq[t]
        spikes = (h >= v_threshold).float()
        v = h * (1.0 - spikes)
        h_list.append(h.clone())
        v_list.append(v.clone())
        s_list.append(spikes.clone())
    return torch.stack(h_list), torch.stack(v_list), torch.stack(s_list)


def prop2b_hard_reset_reference_bound(
    x_seq: torch.Tensor,
    *,
    tau: float,
    v_threshold: float,
) -> dict[str, torch.Tensor]:
    """Compute Prop. 2b using hard-reset LIF's own spike history.

    Returns the hard-reset reference membrane, the LP-SN soft-reset-corrected
    logits built from the hard-reset spike train, and the closed-form bound
    gap. The spike source is deliberately hard-reset LIF, which keeps this
    diagnostic separate from A2's exact soft-reset forward computation.
    """
    x_seq = x_seq.detach().float()
    tau = _stable_tau(tau)
    log_tau = torch.tensor(math.log(tau), device=x_seq.device, dtype=x_seq.dtype)
    alpha = torch.tensor(1.0 / tau, device=x_seq.device, dtype=x_seq.dtype)
    beta = torch.tensor(float(v_threshold), device=x_seq.device, dtype=x_seq.dtype)
    h_hard, v_hard, s_hard = hard_reset_lif_reference(
        x_seq,
        tau=tau,
        v_threshold=v_threshold,
    )

    flat = x_seq.reshape(x_seq.shape[0], -1)
    kernel = build_causal_exp_kernel(x_seq.shape[0], log_tau, alpha)
    no_reset_logits = (kernel @ flat).reshape(x_seq.shape)
    correction = soft_reset_correction(s_hard.detach(), log_tau, beta)
    corrected_logits = no_reset_logits - correction
    actual_gap = corrected_logits - h_hard

    lam = 1.0 - 1.0 / tau
    predicted_gap = torch.zeros_like(actual_gap)
    for target in range(x_seq.shape[0]):
        for source in range(target):
            predicted_gap[target] += (
                s_hard[source]
                * (h_hard[source] - v_threshold)
                * (lam ** (target - source))
            )

    return {
        "h_hard": h_hard,
        "v_hard": v_hard,
        "s_hard": s_hard,
        "corrected_logits_from_hard_spikes": corrected_logits,
        "actual_gap": actual_gap,
        "predicted_gap": predicted_gap,
    }


def prop2b_hard_reset_reference_stats(
    x_seq: torch.Tensor,
    *,
    tau: float,
    v_threshold: float,
) -> dict[str, float]:
    """Summarize the Prop. 2b hard-reset reference bound for plotting."""
    result = prop2b_hard_reset_reference_bound(
        x_seq,
        tau=tau,
        v_threshold=v_threshold,
    )
    gap = result["predicted_gap"]
    h_hard = result["h_hard"]
    s_hard = result["s_hard"]
    overshoot = (h_hard - v_threshold).clamp_min(0.0) * s_hard
    spike_count = float(s_hard.sum())
    rms_by_timestep = [
        float(torch.sqrt(torch.mean(gap[t].pow(2)))) for t in range(gap.shape[0])
    ]
    return {
        "bound_mean_abs": float(gap.abs().mean()),
        "bound_rms": sum(rms_by_timestep) / max(len(rms_by_timestep), 1),
        "mean_spike_overshoot": float(overshoot.sum()) / max(spike_count, 1.0),
        "spike_count": spike_count,
    }


__all__ = [
    "hard_reset_lif_reference",
    "prop2b_hard_reset_reference_bound",
    "prop2b_hard_reset_reference_stats",
]
