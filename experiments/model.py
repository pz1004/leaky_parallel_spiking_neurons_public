"""VGG-Small, half-width VGG, and spiking ResNet-18 comparison backbones."""

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
        half_width: bool = False,
    ) -> None:
        super().__init__()
        if image_size < 32:
            raise ValueError("The five-block backbone requires image_size >= 32")
        channels = (32, 64, 128, 128, 128) if half_width else (64, 128, 256, 256, 256)
        hidden = 256 if half_width else 512
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
                layer.Linear(in_channels * spatial * spatial, hidden, bias=False),
                build_neuron(neuron_name, time_steps, neuron_params),
                layer.Dropout(0.5),
                layer.Linear(hidden, num_classes, bias=False),
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


class SpikingBasicBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride, make_neuron):
        super().__init__()
        self.conv1 = layer.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(outgoing)
        self.spike1 = make_neuron()
        self.conv2 = layer.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(outgoing)
        self.shortcut = nn.Identity() if stride == 1 and incoming == outgoing else nn.Sequential(
            layer.Conv2d(incoming, outgoing, 1, stride=stride, bias=False),
            layer.BatchNorm2d(outgoing),
        )
        self.spike2 = make_neuron()

    def forward(self, x):
        return self.spike2(self.bn2(self.conv2(self.spike1(self.bn1(self.conv1(x))))) + self.shortcut(x))


class SpikingResNet18(nn.Module):
    """Two-channel 7x7/stride-2 stem; post-addition spikes; [T,B,C] readout.

    The common engine averages the final spike sequence over time, as for VGG.
    """
    def __init__(self, *, neuron_name, time_steps, num_classes, image_size=128, neuron_params=None):
        super().__init__()
        make_neuron = lambda: build_neuron(neuron_name, time_steps, neuron_params)
        self.stem = nn.Sequential(
            layer.Conv2d(2, 64, 7, stride=2, padding=3, bias=False),
            layer.BatchNorm2d(64), make_neuron(), layer.MaxPool2d(3, stride=2, padding=1),
        )
        blocks = []
        incoming = 64
        for stage, outgoing in enumerate((64, 128, 256, 512)):
            for block in range(2):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(SpikingBasicBlock(incoming, outgoing, stride, make_neuron))
                incoming = outgoing
        self.stages = nn.Sequential(*blocks)
        self.readout = nn.Sequential(layer.AdaptiveAvgPool2d(1), layer.Flatten(),
                                     layer.Linear(512, num_classes, bias=False), make_neuron())
        self.time_steps = time_steps
        functional.set_step_mode(self, "m")
        functional.set_backend(self, "torch")

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[1] != self.time_steps:
            raise ValueError(f"Expected [B,{self.time_steps},C,H,W], got {tuple(frames.shape)}")
        return self.readout(self.stages(self.stem(frames.transpose(0, 1).contiguous())))


def build_model(architecture="vgg_small", **kwargs):
    if architecture == "vgg_small":
        return SharedVGGSNN(**kwargs)
    if architecture == "vgg_small_half":
        return SharedVGGSNN(half_width=True, **kwargs)
    if architecture == "spiking_resnet18":
        return SpikingResNet18(**kwargs)
    raise ValueError(f"Unknown architecture {architecture!r}")


def regularization_loss(model: nn.Module, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for module in model.modules():
        method = getattr(module, "regularization_loss", None)
        if callable(method):
            total = total + method().to(device)
    return total
