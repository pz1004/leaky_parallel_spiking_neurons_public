"""Command-line interface for the LP-SN reviewer-remediation experiments.

Examples:
    python -m experiments.cli prepare --dataset cifar10_dvs --data-root ../data \
        --output manifests/cifar10_dvs.json
    python -m experiments.cli train --config configs/smoke.yml
    python -m experiments.cli evaluate --config configs/smoke.yml
    python -m experiments.cli analyze --runs-root runs_revision --output aggregates/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

# NVIDIA documents this workspace setting as a prerequisite for reproducible
# cuBLAS GEMM on CUDA 10.2 and later. Set it before importing torch so direct
# CLI use and the matrix scheduler follow the same deterministic contract.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from spikingjelly.activation_based import functional

from .config import (
    ConditionConfig,
    ExperimentConfig,
    environment_summary,
    load_config,
    runtime_manifest,
    stable_hash,
    utc_now,
    write_json,
)
from .data import (
    DATASET_META,
    build_dataset,
    frame_cache_namespace,
    load_split_manifest,
    prepare_split_manifest,
)
from .engine import evaluate_loader, make_loader, record_dict, seed_everything, train_epoch
from .model import SharedVGGSNN
from .neurons import neuron_metadata


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _model(config: ExperimentConfig) -> SharedVGGSNN:
    model = SharedVGGSNN(
        neuron_name=config.model,
        time_steps=config.time_steps,
        num_classes=int(DATASET_META[config.dataset]["classes"]),
        image_size=config.image_size,
        neuron_params=config.model_params,
    )
    return model.to(torch.device(config.device))


def _load_checkpoint(config: ExperimentConfig, checkpoint_path: Path | None = None):
    path = checkpoint_path or config.run_dir / "checkpoint_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("dataset", "model", "time_steps", "image_size"):
        if checkpoint["config"][key] != getattr(config, key):
            raise ValueError(f"Checkpoint/config mismatch for {key}: {checkpoint['config'][key]} != {getattr(config, key)}")
    manifest = load_split_manifest(Path(config.split_manifest))
    if checkpoint["split_hash"] != manifest["split_hash"]:
        raise ValueError("Checkpoint and current split manifest have different hashes")
    model = _model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint, path


def command_prepare(args: argparse.Namespace) -> int:
    manifest = prepare_split_manifest(args.dataset, args.data_root, args.output, seed=args.seed)
    counts = {split: len(rows) for split, rows in manifest["splits"].items()}
    print(json.dumps({"output": str(args.output), "split_hash": manifest["split_hash"], "counts": counts}, indent=2))
    return 0


def command_cache(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    namespace = frame_cache_namespace(config)
    if namespace is None:
        raise ValueError("This configuration does not enable a cacheable frame source")
    manifest = load_split_manifest(Path(config.split_manifest))
    marker_payload = {
        "schema_version": 1,
        "dataset": config.dataset,
        "time_steps": config.time_steps,
        "image_size": config.image_size,
        "source_namespace": str(namespace),
        "source_condition": namespace.name,
        "split_hash": manifest["split_hash"],
        "samples": {name: len(rows) for name, rows in manifest["splits"].items()},
    }
    marker_hash = stable_hash(marker_payload)
    marker = namespace / f".complete_{marker_hash}.json"
    if marker.exists() and not args.verify:
        print(json.dumps({"status": "already_complete", **marker_payload}, indent=2))
        return 0
    seen = 0
    started = time.perf_counter()
    for split in ("train", "val", "test"):
        dataset, selected_manifest = build_dataset(config, split)
        if selected_manifest["split_hash"] != manifest["split_hash"]:
            raise RuntimeError("Cache source and marker split hashes differ")
        loader = make_loader(dataset, config, shuffle=False)
        split_seen = 0
        for _frames, labels, _identifiers, _statistics in loader:
            split_seen += int(labels.numel())
        expected = len(manifest["splits"][split])
        if split_seen != expected:
            raise RuntimeError(f"Cached {split_seen} {split} samples, expected {expected}")
        seen += split_seen
        print(json.dumps({"split": split, "cached_samples": split_seen}), flush=True)
    marker_payload["cached_samples"] = seen
    marker_payload["seconds"] = time.perf_counter() - started
    write_json(marker, marker_payload)
    print(json.dumps({"status": "complete", **marker_payload}, indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    seed_everything(config.seed)
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "test_evaluation_ledger.jsonl").exists():
        raise RuntimeError(f"Refusing to train in a run directory whose locked test set was already opened: {run_dir}")
    train_dataset, manifest = build_dataset(config, "train")
    validation_dataset, validation_manifest = build_dataset(config, "val")
    if manifest["split_hash"] != validation_manifest["split_hash"]:
        raise RuntimeError("Training and validation split hashes differ")
    train_loader = make_loader(train_dataset, config, shuffle=True)
    validation_loader = make_loader(validation_dataset, config, shuffle=False)
    model = _model(config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    _write_yaml(run_dir / "config.yml", asdict(config))
    write_json(run_dir / "environment.json", environment_summary())
    train_manifest = runtime_manifest(config, phase="train_validation", split_hash=manifest["split_hash"])
    write_json(run_dir / "run_manifest.json", train_manifest)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_primary = -math.inf
    best_accuracy = -math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, config)
        validation_record, confusion, _sample_rows = evaluate_loader(
            model,
            validation_loader,
            config,
            split="validation",
            epoch=epoch,
            num_classes=int(DATASET_META[config.dataset]["classes"]),
        )
        primary = getattr(validation_record, config.selection_metric)
        improved = primary > best_primary + 1e-12 or (
            math.isclose(primary, best_primary, abs_tol=1e-12)
            and validation_record.accuracy > best_accuracy + 1e-12
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": record_dict(validation_record),
            "selection_metric": config.selection_metric,
            "improved": improved,
        }
        _append_jsonl(metrics_path, row)
        if improved:
            best_primary = float(primary)
            best_accuracy = float(validation_record.accuracy)
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "schema_version": config.schema_version,
                    "created_at": utc_now(),
                    "config": asdict(config),
                    "config_hash": config.config_hash,
                    "split_hash": manifest["split_hash"],
                    "selection_split": "validation",
                    "selection_metric": config.selection_metric,
                    "selection_value": best_primary,
                    "selection_accuracy_tiebreak": best_accuracy,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "neuron_metadata": neuron_metadata(config.model),
                },
                run_dir / "checkpoint_best.pt",
            )
            np.save(run_dir / "validation_confusion_best.npy", confusion)
        else:
            stale += 1
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_accuracy": train_metrics["accuracy"],
                    "validation_accuracy": validation_record.accuracy,
                    "validation_macro_f1": validation_record.macro_f1,
                    "best_epoch": best_epoch,
                }
            ),
            flush=True,
        )
        if config.patience > 0 and stale >= config.patience:
            break
    result = {
        "completed_at": utc_now(),
        "dataset": config.dataset,
        "model": config.model,
        "seed": config.seed,
        "time_steps": config.time_steps,
        "condition": asdict(config.condition),
        "selection_split": "validation",
        "selection_metric": config.selection_metric,
        "best_validation_value": best_primary,
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "executed_epochs": epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "test_evaluated": False,
    }
    write_json(run_dir / "train_result.json", result)
    print(json.dumps(result, indent=2))
    return 0


def _condition_from_args(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if getattr(args, "condition", None) is None:
        return config
    return replace(
        config,
        condition=ConditionConfig(
            name=args.condition,
            severity=float(args.severity),
            draw=int(args.draw),
            renormalize=bool(getattr(args, "renormalize", False)),
        ),
    )


def _evaluation_key(config: ExperimentConfig) -> str:
    condition = config.condition
    return f"{condition.name}|{condition.severity:.12g}|{condition.draw}"


def _ledger_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["evaluation_key"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def evaluate_checkpoint(config: ExperimentConfig, checkpoint_path: Path | None = None) -> dict[str, Any]:
    model, checkpoint, selected_path = _load_checkpoint(config, checkpoint_path)
    ledger = config.run_dir / "test_evaluation_ledger.jsonl"
    key = _evaluation_key(config)
    if key in _ledger_keys(ledger):
        raise RuntimeError(f"Locked-test condition was already evaluated: {key}")
    test_dataset, manifest = build_dataset(config, "test")
    test_loader = make_loader(test_dataset, config, shuffle=False)
    record, confusion, sample_rows = evaluate_loader(
        model,
        test_loader,
        config,
        split="test_locked",
        epoch=int(checkpoint["epoch"]),
        num_classes=int(DATASET_META[config.dataset]["classes"]),
        collect_samples=True,
    )
    output_hash = stable_hash(key)[:12]
    sample_path = config.run_dir / f"per_sample_{output_hash}.jsonl"
    payload = {
        **record_dict(record),
        "experiment_id": config.experiment_id,
        "config_hash": config.config_hash,
        "training_config_hash": checkpoint["config_hash"],
        "split_hash": manifest["split_hash"],
        "checkpoint": str(selected_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_split": checkpoint["selection_split"],
        "selection_metric": checkpoint["selection_metric"],
        "selection_value": checkpoint["selection_value"],
        "evaluated_at": utc_now(),
        "per_sample_path": str(sample_path),
    }
    _append_jsonl(config.run_dir / "locked_test_metrics.jsonl", payload)
    np.save(config.run_dir / f"confusion_{output_hash}.npy", confusion)
    sample_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample_rows), encoding="utf-8"
    )
    _append_jsonl(
        ledger,
        {
            "evaluation_key": key,
            "evaluated_at": payload["evaluated_at"],
            "checkpoint_epoch": checkpoint["epoch"],
            "split_hash": manifest["split_hash"],
        },
    )
    return payload


def command_evaluate(args: argparse.Namespace) -> int:
    config = _condition_from_args(load_config(args.config), args)
    result = evaluate_checkpoint(config, args.checkpoint)
    print(json.dumps(result, indent=2))
    return 0


def corruption_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = [{"name": "clean", "severity": 0.0, "draws": 1}]
    severities = {
        "event_dropout": [0.10, 0.30, 0.50],
        "background_activity": [0.10, 0.30, 0.50],
        "polarity_flip": [0.05, 0.15, 0.30],
        "temporal_jitter": [0.005, 0.02, 0.05],
        "packet_loss": [0.10, 0.25, 0.40],
        "hot_pixels": [0.10, 0.30, 0.50],
    }
    for name, values in severities.items():
        grid.extend({"name": name, "severity": value, "draws": 10} for value in values)
    return grid


def command_corrupt(args: argparse.Namespace) -> int:
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "base_seed": args.seed,
        "shared_across_models": True,
        "conditions": corruption_grid(),
        "note": "Sensor-motivated synthetic corruptions; not a calibrated physical-sensor noise model.",
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(args.output, manifest)
    print(json.dumps({"output": str(args.output), "manifest_hash": manifest["manifest_hash"]}, indent=2))
    return 0


def command_evaluate_grid(args: argparse.Namespace) -> int:
    base = load_config(args.config)
    grid = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if "manifest_hash" in grid:
        expected_hash = stable_hash({key: value for key, value in grid.items() if key != "manifest_hash"})
        if grid["manifest_hash"] != expected_hash:
            raise ValueError(f"Corruption-manifest hash mismatch: {args.manifest}")
    if "grids" in grid:
        if not args.grid:
            raise ValueError("--grid is required when the manifest contains named grids")
        grid = {"conditions": grid["grids"][args.grid]}
    completed = 0
    for item in grid["conditions"]:
        for draw in range(int(item["draws"])):
            config = replace(
                base,
                condition=ConditionConfig(name=item["name"], severity=float(item["severity"]), draw=draw),
            )
            try:
                evaluate_checkpoint(config, args.checkpoint)
                completed += 1
            except RuntimeError as exc:
                if "already evaluated" not in str(exc) or not args.skip_completed:
                    raise
    print(json.dumps({"completed": completed, "manifest": str(args.manifest)}, indent=2))
    return 0


def _bootstrap_interval(rows: list[dict[str, Any]], metric: str, seed: int = 20240723) -> tuple[float, float]:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row[metric]))
    seeds = sorted(by_seed)
    rng = np.random.default_rng(seed)
    estimates = np.empty(2000, dtype=np.float64)
    for index in range(len(estimates)):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for selected in selected_seeds:
            values = by_seed[int(selected)]
            sampled = rng.choice(values, size=len(values), replace=True)
            seed_means.append(float(np.mean(sampled)))
        estimates[index] = float(np.mean(seed_means))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def command_analyze(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for path in sorted(args.runs_root.rglob("locked_test_metrics.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not rows:
        raise FileNotFoundError(f"No locked_test_metrics.jsonl files below {args.runs_root}")
    groups: dict[tuple, list[dict[str, Any]]] = {}
    keys = ("experiment_id", "dataset", "model", "time_steps", "condition", "severity")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, selected in sorted(groups.items()):
        summary = dict(zip(keys, group_key))
        summary["n_seeds"] = len({int(row["seed"]) for row in selected})
        summary["n_draws"] = len({int(row["draw"]) for row in selected})
        for metric in (
            "accuracy",
            "macro_f1",
            "firing_rate",
            "throughput",
            "peak_cuda_memory_mb",
            "mean_event_count",
            "mean_active_bin_fraction",
            "mean_event_mass",
            "mean_duration",
        ):
            metric_rows = [row for row in selected if row.get(metric) is not None]
            if not metric_rows:
                summary[f"{metric}_mean"] = math.nan
                summary[f"{metric}_std"] = math.nan
                summary[f"{metric}_ci_low"] = math.nan
                summary[f"{metric}_ci_high"] = math.nan
                continue
            values = np.asarray([float(row[metric]) for row in metric_rows])
            low, high = _bootstrap_interval(metric_rows, metric)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        summaries.append(summary)
    clean_lookup = {
        (row["experiment_id"], row["dataset"], row["model"], row["time_steps"]): row["accuracy_mean"]
        for row in summaries
        if row["condition"] == "clean"
    }
    for row in summaries:
        clean = clean_lookup.get((row["experiment_id"], row["dataset"], row["model"], row["time_steps"]))
        row["clean_accuracy_mean"] = clean
        row["absolute_accuracy_drop"] = None if clean is None else clean - row["accuracy_mean"]
        row["relative_accuracy_drop"] = None if not clean else (clean - row["accuracy_mean"]) / clean
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    robustness_groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in summaries:
        if row["condition"] != "clean":
            key = (row["experiment_id"], row["dataset"], row["model"], row["time_steps"], row["condition"])
            robustness_groups.setdefault(key, []).append(row)
    robustness_rows: list[dict[str, Any]] = []
    for key, selected in sorted(robustness_groups.items()):
        ordered = sorted(selected, key=lambda row: float(row["severity"]))
        x = np.asarray([float(row["severity"]) for row in ordered])
        y = np.asarray([float(row["accuracy_mean"]) for row in ordered])
        robustness_rows.append(
            {
                **dict(zip(("experiment_id", "dataset", "model", "time_steps", "condition"), key)),
                "severity_auc_accuracy": float(np.trapezoid(y, x) / (x[-1] - x[0])) if len(x) > 1 and x[-1] > x[0] else float(y[0]),
                "worst_severity": float(x[np.argmin(y)]),
                "worst_accuracy": float(y.min()),
            }
        )
    if robustness_rows:
        robustness_path = args.output.with_name(args.output.stem + "_robustness.csv")
        with robustness_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(robustness_rows[0]))
            writer.writeheader()
            writer.writerows(robustness_rows)

    class_counts: dict[tuple, list[int]] = {}
    strata_counts: dict[tuple, list[int]] = {}
    for metric_row in rows:
        sample_path_value = metric_row.get("per_sample_path")
        if not sample_path_value or not Path(sample_path_value).exists():
            continue
        samples_for_evaluation = [
            json.loads(line) for line in Path(sample_path_value).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        prefix = (
            metric_row["experiment_id"], metric_row["dataset"], metric_row["model"], metric_row["time_steps"],
            metric_row["condition"], metric_row["severity"],
        )
        for sample in samples_for_evaluation:
            class_key = (*prefix, int(sample["label"]))
            counter = class_counts.setdefault(class_key, [0, 0])
            counter[0] += int(sample["prediction"] == sample["label"])
            counter[1] += 1
        for axis in ("event_count", "duration"):
            values = np.asarray([float(sample[axis]) for sample in samples_for_evaluation])
            edges = np.quantile(values, [0.25, 0.50, 0.75])
            for sample, value in zip(samples_for_evaluation, values):
                quartile = int(np.searchsorted(edges, value, side="right") + 1)
                stratum_key = (*prefix, axis, quartile)
                counter = strata_counts.setdefault(stratum_key, [0, 0])
                counter[0] += int(sample["prediction"] == sample["label"])
                counter[1] += 1
    class_rows = [
        {
            **dict(zip((*keys, "class_index"), key)),
            "correct": counts[0], "support": counts[1], "recall": counts[0] / counts[1],
        }
        for key, counts in sorted(class_counts.items())
    ]
    strata_rows = [
        {
            **dict(zip((*keys, "stratification", "quartile"), key)),
            "correct": counts[0], "support": counts[1], "accuracy": counts[0] / counts[1],
        }
        for key, counts in sorted(strata_counts.items())
    ]
    for suffix, output_rows in (("_classwise.csv", class_rows), ("_strata.csv", strata_rows)):
        if output_rows:
            path = args.output.with_name(args.output.stem + suffix)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
                writer.writeheader()
                writer.writerows(output_rows)
    write_json(args.output.with_suffix(".manifest.json"), {
        "created_at": utc_now(),
        "source_root": str(args.runs_root),
        "source_files": len(list(args.runs_root.rglob("locked_test_metrics.jsonl"))),
        "rows": len(rows),
        "groups": len(summaries),
        "bootstrap": "hierarchical seed then within-seed draw resampling, 2000 replicates",
    })
    print(json.dumps({"output": str(args.output), "rows": len(rows), "groups": len(summaries)}, indent=2))
    return 0


def command_temporal_influence(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model, checkpoint, _path = _load_checkpoint(config, args.checkpoint)
    dataset, manifest = build_dataset(config, "test", condition=ConditionConfig())
    loader = make_loader(dataset, config, shuffle=False)
    device = torch.device(config.device)
    saliency = torch.zeros(config.time_steps, device=device)
    samples = 0
    for batch_index, (frames, labels, _ids, _statistics) in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        frames = frames.to(device).requires_grad_(True)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        spikes = model(frames)
        logits = spikes.mean(0)
        logits.gather(1, labels[:, None]).sum().backward()
        saliency += frames.grad.detach().abs().sum(dim=(0, 2, 3, 4))
        samples += int(labels.numel())
        functional.reset_net(model)
    values = saliency.cpu().numpy()
    normalized = values / max(values.sum(), 1e-12)
    cumulative = np.cumsum(normalized[::-1])[::-1]
    horizon_candidates = np.flatnonzero(cumulative >= 0.90)
    result = {
        "created_at": utc_now(),
        "dataset": config.dataset,
        "model": config.model,
        "seed": config.seed,
        "time_steps": config.time_steps,
        "checkpoint_epoch": checkpoint["epoch"],
        "split_hash": manifest["split_hash"],
        "samples": samples,
        "normalized_absolute_input_gradient": normalized.tolist(),
        "reverse_cumulative_influence": cumulative.tolist(),
        "effective_horizon_90_percent": int(config.time_steps - horizon_candidates[-1]) if len(horizon_candidates) else 0,
        "limitation": "Input gradients are an attribution diagnostic, not causal proof of memory.",
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model, checkpoint, _path = _load_checkpoint(config, args.checkpoint)
    device = torch.device(config.device)
    batch = torch.zeros(
        config.batch_size, config.time_steps, 2, config.image_size, config.image_size, device=device
    )
    with torch.no_grad():
        for _ in range(args.warmup):
            model(batch)
            functional.reset_net(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            model(batch)
            functional.reset_net(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
    result = {
        "created_at": utc_now(),
        "dataset": config.dataset,
        "model": config.model,
        "time_steps": config.time_steps,
        "checkpoint_epoch": checkpoint["epoch"],
        "batch_size": config.batch_size,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "latency_ms_per_batch": seconds * 1000 / args.iterations,
        "throughput_samples_per_second": config.batch_size * args.iterations / seconds,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
        "synchronization": device.type == "cuda",
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


def command_reproduce(args: argparse.Namespace) -> int:
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    selected = [task for task in matrix["targets"] if task["name"] == args.target]
    if not selected:
        raise KeyError(f"Unknown target {args.target!r}")
    commands = selected[0]["commands"]
    if not args.execute:
        print("\n".join(" ".join(command) for command in commands))
        return 0
    for command in commands:
        subprocess.run(command, check=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create and validate a split manifest")
    prepare.add_argument("--dataset", choices=sorted(DATASET_META), required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=20240723)
    prepare.set_defaults(function=command_prepare)

    cache = subparsers.add_parser("cache", help="Precompute lossless frame caches for repeated training epochs")
    cache.add_argument("--config", type=Path, required=True)
    cache.add_argument(
        "--verify",
        action="store_true",
        help="Read every cached sample even when a completion marker exists",
    )
    cache.set_defaults(function=command_cache)

    train = subparsers.add_parser("train", help="Train with validation-only checkpoint selection")
    train.add_argument("--config", type=Path, required=True)
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a selected checkpoint on the locked test split")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--condition")
    evaluate.add_argument("--severity", type=float, default=0.0)
    evaluate.add_argument("--draw", type=int, default=0)
    evaluate.add_argument("--renormalize", action="store_true")
    evaluate.set_defaults(function=command_evaluate)

    corrupt = subparsers.add_parser("corrupt", help="Create the shared corruption/draw manifest")
    corrupt.add_argument("--output", type=Path, required=True)
    corrupt.add_argument("--seed", type=int, default=20240723)
    corrupt.set_defaults(function=command_corrupt)

    grid = subparsers.add_parser("evaluate-grid", help="Evaluate every corruption severity and draw")
    grid.add_argument("--config", type=Path, required=True)
    grid.add_argument("--manifest", type=Path, required=True)
    grid.add_argument("--checkpoint", type=Path)
    grid.add_argument("--grid", help="Named grid when --manifest contains a grids mapping")
    grid.add_argument("--skip-completed", action="store_true")
    grid.set_defaults(function=command_evaluate_grid)

    analyze = subparsers.add_parser("analyze", help="Aggregate locked-test metrics with hierarchical intervals")
    analyze.add_argument("--runs-root", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(function=command_analyze)

    influence = subparsers.add_parser("temporal-influence", help="Compute per-timestep input-gradient influence")
    influence.add_argument("--config", type=Path, required=True)
    influence.add_argument("--checkpoint", type=Path)
    influence.add_argument("--output", type=Path, required=True)
    influence.add_argument("--max-batches", type=int, default=0)
    influence.set_defaults(function=command_temporal_influence)

    benchmark = subparsers.add_parser("benchmark", help="Run the fixed warm-up/timed GPU protocol")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--checkpoint", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--warmup", type=int, default=100)
    benchmark.add_argument("--iterations", type=int, default=500)
    benchmark.set_defaults(function=command_benchmark)

    reproduce = subparsers.add_parser("reproduce", help="Print or execute a named reproduction target")
    reproduce.add_argument("--matrix", type=Path, required=True)
    reproduce.add_argument("--target", required=True)
    reproduce.add_argument("--execute", action="store_true")
    reproduce.set_defaults(function=command_reproduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
