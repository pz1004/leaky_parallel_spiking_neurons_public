"""Training and evaluation primitives with validation-only model selection."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional

from .config import ExperimentConfig, MetricsRecord, SCHEMA_VERSION
from .model import regularization_loss


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_loader(dataset, config: ExperimentConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + (1 if shuffle else 2))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda") and torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def confusion_metrics(confusion: np.ndarray) -> dict[str, float]:
    values = confusion.astype(np.float64)
    diagonal = np.diag(values)
    support = values.sum(axis=1)
    predicted = values.sum(axis=0)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) > 0,
    )
    total = values.sum()
    return {
        "accuracy": float(diagonal.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "balanced_accuracy": float(recall.mean()),
    }


def _is_spiking(module: torch.nn.Module) -> bool:
    return module.__class__.__name__ in {
        "LIFNode",
        "PSN",
        "MaskedPSN",
        "SlidingPSN",
        "RPSU",
        "LPSpikingNeuron",
        "FixedTemporalOperator",
    }


class LayerFiringCollector:
    def __init__(self, model: torch.nn.Module) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.hooks = []
        for name, module in model.named_modules():
            if _is_spiking(module):
                self.sums[name] = 0.0
                self.counts[name] = 0
                self.hooks.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(_module, _inputs, output) -> None:
            if torch.is_tensor(output):
                value = output.detach().float()
                self.sums[name] += float(value.sum().item())
                self.counts[name] += int(value.numel())

        return hook

    def finish(self) -> dict[str, float]:
        for hook in self.hooks:
            hook.remove()
        return {name: self.sums[name] / max(self.counts[name], 1) for name in self.sums}


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.train()
    device = torch.device(config.device)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    correct = 0
    samples = 0
    firing = 0.0
    start = time.perf_counter()
    for batch_index, (frames, labels, _ids, _statistics) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
            spikes = model(frames)
            logits = spikes.mean(0)
            task_loss = F.cross_entropy(logits, labels)
            loss = task_loss + regularization_loss(model, device)
            scaled_loss = loss / config.accumulation_steps
        scaler.scale(scaled_loss).backward()
        should_step = (batch_index + 1) % config.accumulation_steps == 0 or batch_index + 1 == len(loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        functional.reset_net(model)
        size = int(labels.numel())
        samples += size
        loss_sum += float(loss.item()) * size
        correct += int((logits.argmax(1) == labels).sum().item())
        firing += float(spikes.detach().float().mean().item()) * size
    seconds = max(time.perf_counter() - start, 1e-12)
    return {
        "loss": loss_sum / max(samples, 1),
        "accuracy": correct / max(samples, 1),
        "firing_rate": firing / max(samples, 1),
        "samples": samples,
        "seconds": seconds,
        "throughput": samples / seconds,
    }


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    *,
    split: str,
    epoch: int | None,
    num_classes: int,
    collect_samples: bool = False,
) -> tuple[MetricsRecord, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    collector = LayerFiringCollector(model)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    loss_sum = 0.0
    firing = 0.0
    samples = 0
    statistic_sums = {"event_count": 0.0, "active_bin_fraction": 0.0, "event_mass": 0.0, "duration": 0.0}
    sample_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for frames, labels, identifiers, statistics in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        spikes = model(frames)
        logits = spikes.mean(0)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(1)
        functional.reset_net(model)
        size = int(labels.numel())
        samples += size
        loss_sum += float(loss.item()) * size
        firing += float(spikes.detach().float().mean().item()) * size
        np.add.at(confusion, (labels.cpu().numpy(), predictions.cpu().numpy()), 1)
        for name in statistic_sums:
            statistic_sums[name] += float(statistics[name].double().sum().item())
        if collect_samples:
            for index, sample_id in enumerate(identifiers):
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "label": int(labels[index].item()),
                        "prediction": int(predictions[index].item()),
                        **{name: float(statistics[name][index].item()) for name in statistic_sums},
                    }
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = max(time.perf_counter() - start, 1e-12)
    scores = confusion_metrics(confusion)
    condition = config.condition
    record = MetricsRecord(
        schema_version=SCHEMA_VERSION,
        split=split,
        dataset=config.dataset,
        model=config.model,
        seed=config.seed,
        time_steps=config.time_steps,
        condition=condition.name,
        severity=condition.severity,
        draw=condition.draw,
        epoch=epoch,
        samples=samples,
        loss=loss_sum / max(samples, 1),
        firing_rate=firing / max(samples, 1),
        seconds=seconds,
        throughput=samples / seconds,
        peak_cuda_memory_mb=(torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0),
        mean_event_count=statistic_sums["event_count"] / max(samples, 1),
        mean_active_bin_fraction=statistic_sums["active_bin_fraction"] / max(samples, 1),
        mean_event_mass=statistic_sums["event_mass"] / max(samples, 1),
        mean_duration=statistic_sums["duration"] / max(samples, 1),
        layer_firing_rates=collector.finish(),
        **scores,
    )
    return record, confusion, sample_rows


def record_dict(record: MetricsRecord) -> dict[str, Any]:
    return asdict(record)
