"""Common neuron adapters used by the controlled comparison.

The RPSU equations are independently implemented from the authors' public
IJCNN-2024 repository (Brain-Cog-Lab/PSU, commit 89fd724365cd4970), rather
than importing its full BrainCog training stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from spikingjelly.activation_based import base, neuron, surrogate

try:
    from ..lpsn_variants import build_lpsn_neuron
except ImportError:  # direct execution from the public-artifact root
    from lpsn_variants import build_lpsn_neuron


@dataclass(frozen=True)
class NeuronMetadata:
    name: str
    source: str
    source_revision: str
    temporal_profile: str


NEURON_METADATA = {
    "lif": NeuronMetadata("LIF", "SpikingJelly", "environment lock", "sequential_recurrence"),
    "psn": NeuronMetadata("PSN", "SpikingJelly", "environment lock", "dense_gemm"),
    "masked_psn": NeuronMetadata("Masked PSN", "SpikingJelly", "environment lock", "masked_dense_gemm"),
    "sliding_psn": NeuronMetadata("Sliding PSN", "SpikingJelly", "environment lock", "sliding_window"),
    "rpsu": NeuronMetadata(
        "RPSU", "https://github.com/Brain-Cog-Lab/PSU", "89fd724365cd4970cf3cf3802122c9fc22285a71", "masked_dense_gemm"
    ),
    "lpsn_a1_lif_init": NeuronMetadata("LP-SN A1", "this artifact", "current commit", "dense_gemm"),
    "lpsn_a2_soft": NeuronMetadata("LP-SN A2", "this artifact", "current commit", "exact_detached_scan"),
}


def _rpsu_matrices(time_steps: int, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(time_steps).view(-1, 1)
    column = torch.arange(time_steps).view(1, -1)
    lag = row - column
    causal = lag >= 0
    strict = lag > 0
    m1 = torch.where(causal, torch.pow(torch.tensor(1.0 / tau), lag + 1), torch.zeros_like(lag, dtype=torch.float32))
    m2 = torch.where(strict, -torch.pow(torch.tensor(1.0 / tau), lag), torch.zeros_like(lag, dtype=torch.float32))
    return m1.float(), m2.float()


class RPSU(nn.Module, base.StepModule):
    """Reset-aware Parallel Spiking Unit with a causal learned reset proxy."""

    def __init__(self, time_steps: int, *, tau: float = 2.0, threshold: float = 1.0) -> None:
        super().__init__()
        self.step_mode = "m"
        if time_steps <= 0 or tau <= 1.0:
            raise ValueError("time_steps must be positive and tau must be greater than one")
        self.time_steps = int(time_steps)
        self.threshold = float(threshold)
        m1, m2 = _rpsu_matrices(time_steps, tau)
        self.register_buffer("m1", m1)
        self.register_buffer("m2", m2)
        self.weight = nn.Parameter(torch.empty(time_steps, time_steps))
        self.bias = nn.Parameter(torch.zeros(time_steps, 1))
        self.register_buffer("causal_mask", torch.tril(torch.ones(time_steps, time_steps)))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.surrogate = surrogate.ATan(alpha=2.0, spiking=True)

    def supported_step_mode(self) -> tuple[str, ...]:
        return ("m",)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.shape[0] != self.time_steps:
            raise ValueError(f"RPSU expected T={self.time_steps}, got {x_seq.shape[0]}")
        shape = x_seq.shape
        flattened = x_seq.flatten(1)
        direct = self.m1 @ flattened
        reset_proxy = (self.weight * self.causal_mask) @ flattened + self.bias
        membrane = direct + self.m2 @ reset_proxy.sigmoid()
        return self.surrogate(membrane.view(shape) - self.threshold)


def build_neuron(name: str, time_steps: int, params: dict[str, Any] | None = None) -> nn.Module:
    params = {} if params is None else dict(params)
    if name == "lif":
        node = neuron.LIFNode(tau=float(params.get("tau", 2.0)), step_mode="m", backend="torch")
    elif name == "psn":
        node = neuron.PSN(T=time_steps)
    elif name == "masked_psn":
        node = neuron.MaskedPSN(
            k=int(params.get("k", time_steps)),
            T=time_steps,
            lambda_init=float(params.get("lambda_init", 0.0)),
            step_mode="m",
        )
    elif name == "sliding_psn":
        node = neuron.SlidingPSN(
            k=int(params.get("k", min(8, time_steps))),
            exp_init=bool(params.get("exp_init", True)),
            step_mode="m",
            backend="gemm",
        )
    elif name == "rpsu":
        node = RPSU(time_steps, tau=float(params.get("tau", 2.0)), threshold=float(params.get("threshold", 1.0)))
    elif name in {"lpsn_a1_lif_init", "lpsn_a2_soft"}:
        node = build_lpsn_neuron(T=time_steps, variant=name, **params)
    else:
        raise ValueError(name)
    return node


def neuron_metadata(name: str) -> dict[str, str]:
    metadata = NEURON_METADATA[name]
    return {
        "name": metadata.name,
        "source": metadata.source,
        "source_revision": metadata.source_revision,
        "temporal_profile": metadata.temporal_profile,
    }
