"""Shared VGG-Small spiking backbone for every controlled comparator."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, layer

from .neurons import build_neuron


class SharedVGGSNN(nn.Module):
    def __init__(
        self,
        *,
        neuron_name: str,
        time_steps: int,
        num_classes: int,
        image_size: int = 128,
        neuron_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if image_size < 32:
            raise ValueError("The five-block backbone requires image_size >= 32")
        channels = (64, 128, 256, 256, 256)
        modules: list[nn.Module] = []
        in_channels = 2
        spatial = image_size
        for out_channels in channels:
            modules.extend(
                (
                    layer.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    layer.BatchNorm2d(out_channels),
                    build_neuron(neuron_name, time_steps, neuron_params),
                    layer.MaxPool2d(2),
                )
            )
            spatial //= 2
            in_channels = out_channels
        modules.extend(
            (
                layer.Flatten(),
                layer.Dropout(0.5),
                layer.Linear(in_channels * spatial * spatial, 512, bias=False),
                build_neuron(neuron_name, time_steps, neuron_params),
                layer.Dropout(0.5),
                layer.Linear(512, num_classes, bias=False),
                build_neuron(neuron_name, time_steps, neuron_params),
            )
        )
        self.network = nn.Sequential(*modules)
        self.time_steps = int(time_steps)
        functional.set_step_mode(self, "m")
        functional.set_backend(self, "torch")

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(frames.shape)}")
        if frames.shape[1] != self.time_steps:
            raise ValueError(f"Model was constructed for T={self.time_steps}, got {frames.shape[1]}")
        return self.network(frames.transpose(0, 1).contiguous())


def regularization_loss(model: nn.Module, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for module in model.modules():
        method = getattr(module, "regularization_loss", None)
        if callable(method):
            total = total + method().to(device)
    return total
