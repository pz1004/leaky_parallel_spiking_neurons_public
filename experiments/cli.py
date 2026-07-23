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
    file_sha256,
    load_config,
    runtime_manifest,
    stable_hash,
    utc_now,
    write_json,
)
from .data import (
    DATASET_META,
    FRAME_CACHE_SCHEMA,
    PERTURBATION_NAMESPACE_SEED,
    build_dataset,
    cache_source_condition,
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


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def _portable_path(path: Path) -> str:
    resolved = _project_path(path).resolve()
    project = Path(__file__).resolve().parents[1]
    try:
        return str(resolved.relative_to(project))
    except ValueError:
        return str(resolved)


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
    config = _condition_from_args(load_config(args.config), args)
    namespace = frame_cache_namespace(config)
    if namespace is None:
        raise ValueError("This configuration does not enable a cacheable frame source")
    manifest = load_split_manifest(Path(config.split_manifest))
    source_condition = cache_source_condition(config.condition)
    if source_condition is None:
        raise ValueError("This configuration does not have a cache source condition")
    marker_payload = {
        "schema_version": 1,
        "frame_cache_schema": FRAME_CACHE_SCHEMA,
        "dataset": config.dataset,
        "time_steps": config.time_steps,
        "image_size": config.image_size,
        "source_namespace": str(namespace),
        "source_condition": asdict(source_condition),
        "split_hash": manifest["split_hash"],
        "splits": args.split or ["train", "val", "test"],
        "samples": {
            name: len(manifest["splits"][name])
            for name in (args.split or ["train", "val", "test"])
        },
    }
    marker_hash = stable_hash(marker_payload)
    marker = namespace / f".complete_{marker_hash}.json"
    if marker.exists() and not args.verify:
        print(json.dumps({"status": "already_complete", **marker_payload}, indent=2))
        return 0
    seen = 0
    started = time.perf_counter()
    for split in marker_payload["splits"]:
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
        "schema_version": config.schema_version,
        "completed_at": utc_now(),
        "experiment_id": config.experiment_id,
        "config_hash": config.config_hash,
        "split_hash": manifest["split_hash"],
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
        "checkpoint": _portable_path(run_dir / "checkpoint_best.pt"),
        "checkpoint_sha256": file_sha256(
            _project_path(run_dir / "checkpoint_best.pt")
        ),
        "independent_replication": True,
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
    confusion_path = config.run_dir / f"confusion_{output_hash}.npy"
    evaluation_manifest_path = (
        config.run_dir / "evaluation_manifests" / f"{output_hash}.json"
    )
    payload = {
        **record_dict(record),
        "experiment_id": config.experiment_id,
        "config_hash": config.config_hash,
        "training_config_hash": checkpoint["config_hash"],
        "split_hash": manifest["split_hash"],
        "checkpoint": _portable_path(selected_path),
        "checkpoint_sha256": file_sha256(_project_path(selected_path)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_split": checkpoint["selection_split"],
        "selection_metric": checkpoint["selection_metric"],
        "selection_value": checkpoint["selection_value"],
        "evaluated_at": utc_now(),
        "per_sample_path": _portable_path(sample_path),
        "evaluation_manifest_path": _portable_path(evaluation_manifest_path),
    }
    _append_jsonl(config.run_dir / "locked_test_metrics.jsonl", payload)
    np.save(confusion_path, confusion)
    sample_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample_rows), encoding="utf-8"
    )
    evaluation_manifest = asdict(
        runtime_manifest(
            config,
            phase="locked_test_evaluation",
            split_hash=manifest["split_hash"],
        )
    )
    evaluation_manifest.update(
        {
            "evaluation_key": key,
            "evaluation_config_hash": config.config_hash,
            "training_config_hash": checkpoint["config_hash"],
            "checkpoint": _portable_path(selected_path),
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "selection_split": checkpoint["selection_split"],
            "selection_metric": checkpoint["selection_metric"],
            "selection_value": checkpoint["selection_value"],
            "test_evaluated_at": payload["evaluated_at"],
            "artifacts": {
                "per_sample_path": _portable_path(sample_path),
                "per_sample_sha256": file_sha256(_project_path(sample_path)),
                "confusion_path": _portable_path(confusion_path),
                "confusion_sha256": file_sha256(_project_path(confusion_path)),
                "metrics_payload_sha256": stable_hash(payload),
            },
        }
    )
    evaluation_manifest["manifest_hash"] = stable_hash(evaluation_manifest)
    write_json(evaluation_manifest_path, evaluation_manifest)
    _append_jsonl(
        ledger,
        {
            "evaluation_key": key,
            "evaluated_at": payload["evaluated_at"],
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "split_hash": manifest["split_hash"],
            "evaluation_manifest_path": _portable_path(evaluation_manifest_path),
            "evaluation_manifest_hash": evaluation_manifest["manifest_hash"],
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
    if args.seed != PERTURBATION_NAMESPACE_SEED:
        raise ValueError(
            "The perturbation namespace is protocol-locked to "
            f"{PERTURBATION_NAMESPACE_SEED}; received {args.seed}"
        )
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "base_seed": PERTURBATION_NAMESPACE_SEED,
        "shared_across_models_and_initialization_seeds": True,
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


def _exact_signed_rank(values: list[float]) -> dict[str, float | int]:
    nonzero = np.asarray(
        [float(value) for value in values if not math.isclose(float(value), 0.0, abs_tol=1e-15)],
        dtype=np.float64,
    )
    count = len(nonzero)
    if count == 0:
        return {
            "wilcoxon_nonzero_seed_means": 0,
            "wilcoxon_w_plus": 0.0,
            "wilcoxon_p_greater_raw": 1.0,
            "wilcoxon_p_two_sided_raw": 1.0,
        }
    absolute = np.abs(nonzero)
    order = np.argsort(absolute, kind="stable")
    ranks = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and math.isclose(
            float(absolute[order[stop]]),
            float(absolute[order[start]]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    observed = float(ranks[nonzero > 0].sum())
    total = float(ranks.sum())
    possibilities = np.asarray(
        [
            sum(ranks[index] for index in range(count) if mask & (1 << index))
            for mask in range(1 << count)
        ],
        dtype=np.float64,
    )
    greater = float(np.mean(possibilities >= observed - 1e-15))
    distance = abs(observed - total / 2.0)
    two_sided = float(
        np.mean(np.abs(possibilities - total / 2.0) >= distance - 1e-15)
    )
    return {
        "wilcoxon_nonzero_seed_means": count,
        "wilcoxon_w_plus": observed,
        "wilcoxon_p_greater_raw": greater,
        "wilcoxon_p_two_sided_raw": min(1.0, two_sided),
    }


def _paired_test_family(row: dict[str, Any]) -> str:
    experiment = str(row["experiment_id"])
    condition = str(row["condition"])
    if experiment == "temporal_native" and condition == "clean":
        return "temporal_native_clean_models_vs_psn"
    if experiment == "hardvs_main" and condition == "clean":
        return "hardvs_main_models_vs_psn"
    if experiment.startswith("density_sparsity"):
        return "density_sparsity_models_vs_psn"
    if condition in {
        "event_dropout",
        "background_activity",
        "polarity_flip",
        "temporal_jitter",
        "packet_loss",
        "hot_pixels",
    }:
        return "sensor_corruption_models_vs_psn"
    return "temporal_diagnostic_models_vs_psn"


def _holm_adjust(rows: list[dict[str, Any]], raw_field: str, output_field: str) -> None:
    families: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        families.setdefault(str(row["test_family"]), []).append(index)
    for indices in families.values():
        ordered = sorted(indices, key=lambda index: float(rows[index][raw_field]))
        running = 0.0
        total = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (total - rank) * float(rows[index][raw_field]))
            running = max(running, adjusted)
            rows[index][output_field] = running


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
        same_experiment_key = (
            row["experiment_id"],
            row["dataset"],
            row["model"],
            row["time_steps"],
        )
        reference_key = same_experiment_key
        clean = clean_lookup.get(reference_key)
        reference_type = "same_experiment"
        if clean is None:
            reference_key = (
                "temporal_native",
                row["dataset"],
                row["model"],
                row["time_steps"],
            )
            clean = clean_lookup.get(reference_key)
            reference_type = (
                "temporal_native_matched_fallback"
                if clean is not None
                else "unavailable"
            )
        row["clean_reference_experiment_id"] = (
            reference_key[0] if clean is not None else None
        )
        row["clean_reference_type"] = reference_type
        row["clean_accuracy_mean"] = clean
        row["absolute_accuracy_drop"] = None if clean is None else clean - row["accuracy_mean"]
        row["relative_accuracy_drop"] = (
            None if clean is None or clean == 0 else (clean - row["accuracy_mean"]) / clean
        )
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
        x_values = [float(row["severity"]) for row in ordered]
        y_values = [float(row["accuracy_mean"]) for row in ordered]
        clean_reference = ordered[0].get("clean_accuracy_mean")
        if clean_reference is not None and x_values[0] > 0:
            x_values.insert(0, 0.0)
            y_values.insert(0, float(clean_reference))
        x = np.asarray(x_values)
        y = np.asarray(y_values)
        worst_accuracy = float(np.min(y[1:] if len(y) > len(ordered) else y))
        robustness_rows.append(
            {
                **dict(zip(("experiment_id", "dataset", "model", "time_steps", "condition"), key)),
                "clean_reference_experiment_id": ordered[0].get(
                    "clean_reference_experiment_id"
                ),
                "clean_accuracy_mean": clean_reference,
                "severity_auc_accuracy": float(np.trapezoid(y, x) / (x[-1] - x[0])) if len(x) > 1 and x[-1] > x[0] else float(y[0]),
                "worst_severity": float(
                    ordered[
                        int(np.argmin([float(row["accuracy_mean"]) for row in ordered]))
                    ]["severity"]
                ),
                "worst_accuracy": worst_accuracy,
                "worst_absolute_accuracy_drop": (
                    None
                    if clean_reference is None
                    else float(clean_reference) - worst_accuracy
                ),
                "worst_relative_accuracy_drop": (
                    None
                    if clean_reference is None or float(clean_reference) == 0
                    else (float(clean_reference) - worst_accuracy)
                    / float(clean_reference)
                ),
            }
        )
    if robustness_rows:
        robustness_path = args.output.with_name(args.output.stem + "_robustness.csv")
        with robustness_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(robustness_rows[0]))
            writer.writeheader()
            writer.writerows(robustness_rows)

    pair_key_fields = (
        "experiment_id",
        "dataset",
        "time_steps",
        "condition",
        "severity",
        "seed",
        "draw",
    )
    baseline_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row["model"] != "psn":
            continue
        key = tuple(row[field] for field in pair_key_fields)
        if key in baseline_rows:
            raise RuntimeError(f"Duplicate PSN pairing key: {key}")
        baseline_rows[key] = row
    paired_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    paired_group_fields = (
        "experiment_id",
        "dataset",
        "model",
        "time_steps",
        "condition",
        "severity",
    )
    for row in rows:
        if row["model"] == "psn":
            continue
        baseline_key = tuple(row[field] for field in pair_key_fields)
        baseline = baseline_rows.get(baseline_key)
        if baseline is None:
            continue
        paired_groups.setdefault(
            tuple(row[field] for field in paired_group_fields), []
        ).append(
            {
                "seed": int(row["seed"]),
                "draw": int(row["draw"]),
                "accuracy_delta": float(row["accuracy"])
                - float(baseline["accuracy"]),
                "macro_f1_delta": float(row["macro_f1"])
                - float(baseline["macro_f1"]),
            }
        )
    paired_rows: list[dict[str, Any]] = []
    for key, selected in sorted(paired_groups.items()):
        result = {
            **dict(zip(paired_group_fields, key)),
            "baseline_model": "psn",
            "matched_pairs": len(selected),
            "n_seeds": len({row["seed"] for row in selected}),
            "n_draws": len({row["draw"] for row in selected}),
        }
        result["test_family"] = _paired_test_family(result)
        for metric in ("accuracy_delta", "macro_f1_delta"):
            values = np.asarray([float(row[metric]) for row in selected])
            low, high = _bootstrap_interval(selected, metric)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
            result[f"{metric}_wins"] = int(np.sum(values > 0))
            result[f"{metric}_ties"] = int(np.sum(values == 0))
            result[f"{metric}_losses"] = int(np.sum(values < 0))
            seed_means = [
                float(
                    np.mean(
                        [
                            row[metric]
                            for row in selected
                            if int(row["seed"]) == seed
                        ]
                    )
                )
                for seed in sorted({int(row["seed"]) for row in selected})
            ]
            result[f"{metric}_seed_mean_wins"] = int(
                np.sum(np.asarray(seed_means) > 0)
            )
            result[f"{metric}_seed_mean_ties"] = int(
                np.sum(np.asarray(seed_means) == 0)
            )
            result[f"{metric}_seed_mean_losses"] = int(
                np.sum(np.asarray(seed_means) < 0)
            )
            signed_rank = _exact_signed_rank(seed_means)
            for field, value in signed_rank.items():
                result[f"{metric}_{field}"] = value
        paired_rows.append(result)
    if paired_rows:
        for metric in ("accuracy_delta", "macro_f1_delta"):
            _holm_adjust(
                paired_rows,
                f"{metric}_wilcoxon_p_greater_raw",
                f"{metric}_wilcoxon_p_greater_holm",
            )
            _holm_adjust(
                paired_rows,
                f"{metric}_wilcoxon_p_two_sided_raw",
                f"{metric}_wilcoxon_p_two_sided_holm",
            )
        paired_path = args.output.with_name(args.output.stem + "_paired.csv")
        with paired_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)

    class_counts: dict[tuple, list[int]] = {}
    strata_counts: dict[tuple, list[int]] = {}
    pooled_clean_confusions: dict[tuple[Any, ...], np.ndarray] = {}
    for metric_row in rows:
        sample_path_value = metric_row.get("per_sample_path")
        if not sample_path_value:
            continue
        sample_path = _project_path(Path(sample_path_value))
        if not sample_path.exists():
            continue
        samples_for_evaluation = [
            json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()
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
        if metric_row["condition"] == "clean":
            confusion_key = (
                metric_row["experiment_id"],
                metric_row["dataset"],
                metric_row["model"],
                metric_row["time_steps"],
            )
            matrix = pooled_clean_confusions.setdefault(
                confusion_key,
                np.zeros(
                    (
                        int(DATASET_META[metric_row["dataset"]]["classes"]),
                        int(DATASET_META[metric_row["dataset"]]["classes"]),
                    ),
                    dtype=np.int64,
                ),
            )
            for sample in samples_for_evaluation:
                matrix[int(sample["label"]), int(sample["prediction"])] += 1
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
    confusion_rows: list[dict[str, Any]] = []
    for key, matrix in sorted(pooled_clean_confusions.items()):
        support = matrix.sum(axis=1)
        for true_class, predicted_class in np.argwhere(matrix > 0):
            count = int(matrix[true_class, predicted_class])
            confusion_rows.append(
                {
                    **dict(
                        zip(
                            ("experiment_id", "dataset", "model", "time_steps"),
                            key,
                        )
                    ),
                    "true_class": int(true_class),
                    "predicted_class": int(predicted_class),
                    "count": count,
                    "true_class_support": int(support[true_class]),
                    "row_normalized_fraction": (
                        count / int(support[true_class])
                        if support[true_class]
                        else 0.0
                    ),
                    "pooling_note": (
                        "Predictions pooled across matched seeds; repeated "
                        "test examples are correlated, not independent samples."
                    ),
                }
            )
    if confusion_rows:
        path = args.output.with_name(args.output.stem + "_confusion_long.csv")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(confusion_rows[0]))
            writer.writeheader()
            writer.writerows(confusion_rows)

    training_rows: list[dict[str, Any]] = []
    for path in sorted(args.runs_root.rglob("train_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(args.runs_root)
        if not relative.parts:
            continue
        condition = row.get("condition", {})
        training_rows.append(
            {
                **row,
                "experiment_id": row.get("experiment_id", relative.parts[0]),
                "condition": str(condition.get("name", "clean")),
                "severity": float(condition.get("severity", 0.0)),
                "draw": int(condition.get("draw", 0)),
            }
        )
    training_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    training_keys = (
        "experiment_id",
        "dataset",
        "model",
        "time_steps",
        "condition",
        "severity",
    )
    for row in training_rows:
        training_groups.setdefault(
            tuple(row[key] for key in training_keys), []
        ).append(row)
    training_summaries: list[dict[str, Any]] = []
    for key, selected in sorted(training_groups.items()):
        result = {
            **dict(zip(training_keys, key)),
            "n_seeds": len({int(row["seed"]) for row in selected}),
            "n_draws": len({int(row["draw"]) for row in selected}),
            "parameter_count": int(selected[0]["parameter_count"]),
            "trainable_parameter_count": int(
                selected[0]["trainable_parameter_count"]
            ),
        }
        for metric in (
            "best_validation_value",
            "best_validation_accuracy",
            "best_epoch",
            "executed_epochs",
        ):
            metric_rows = [
                {"seed": int(row["seed"]), metric: float(row[metric])}
                for row in selected
            ]
            values = np.asarray([row[metric] for row in metric_rows])
            low, high = _bootstrap_interval(metric_rows, metric)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        training_summaries.append(result)
    if training_summaries:
        path = args.output.with_name(args.output.stem + "_training.csv")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(training_summaries[0])
            )
            writer.writeheader()
            writer.writerows(training_summaries)

    source_paths = sorted(args.runs_root.rglob("locked_test_metrics.jsonl"))
    training_source_paths = sorted(args.runs_root.rglob("train_result.json"))
    aggregate_manifest = {
        "created_at": utc_now(),
        "source_root": str(args.runs_root),
        "source_files": [
            {
                "path": _portable_path(path),
                "sha256": file_sha256(path),
            }
            for path in source_paths + training_source_paths
        ],
        "rows": len(rows),
        "groups": len(summaries),
        "training_rows": len(training_rows),
        "training_groups": len(training_summaries),
        "pooled_clean_confusion_groups": len(pooled_clean_confusions),
        "pooled_clean_confusion_nonzero_cells": len(confusion_rows),
        "bootstrap": "hierarchical seed then within-seed draw resampling, 2000 replicates",
        "paired_tests": (
            "Exact signed-rank tests on per-seed mean paired deltas; zero "
            "differences dropped; raw greater and two-sided p-values with "
            "Holm adjustment within the declared test_family."
        ),
    }
    aggregate_manifest["manifest_hash"] = stable_hash(aggregate_manifest)
    write_json(args.output.with_suffix(".manifest.json"), aggregate_manifest)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "groups": len(summaries)}, indent=2))
    return 0


def command_temporal_influence(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model, checkpoint, checkpoint_path = _load_checkpoint(config, args.checkpoint)
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
        "schema_version": 1,
        "created_at": utc_now(),
        "dataset": config.dataset,
        "model": config.model,
        "seed": config.seed,
        "time_steps": config.time_steps,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint": _portable_path(checkpoint_path),
        "checkpoint_sha256": file_sha256(_project_path(checkpoint_path)),
        "training_config_hash": checkpoint["config_hash"],
        "diagnostic_config_hash": config.config_hash,
        "split_hash": manifest["split_hash"],
        "samples": samples,
        "normalized_absolute_input_gradient": normalized.tolist(),
        "reverse_cumulative_influence": cumulative.tolist(),
        "effective_horizon_90_percent": int(config.time_steps - horizon_candidates[-1]) if len(horizon_candidates) else 0,
        "limitation": "Input gradients are an attribution diagnostic, not causal proof of memory.",
        "runtime": asdict(
            runtime_manifest(
                config,
                phase="temporal_input_gradient_diagnostic",
                split_hash=manifest["split_hash"],
            )
        ),
    }
    result["manifest_hash"] = stable_hash(result)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model, checkpoint, checkpoint_path = _load_checkpoint(config, args.checkpoint)
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
        "schema_version": 1,
        "created_at": utc_now(),
        "dataset": config.dataset,
        "model": config.model,
        "time_steps": config.time_steps,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint": _portable_path(checkpoint_path),
        "checkpoint_sha256": file_sha256(_project_path(checkpoint_path)),
        "training_config_hash": checkpoint["config_hash"],
        "benchmark_config_hash": config.config_hash,
        "batch_size": config.batch_size,
        "input_shape": list(batch.shape),
        "input_dtype": str(batch.dtype),
        "input_distribution": "all-zero synthetic tensor already resident on device",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "latency_ms_per_batch": seconds * 1000 / args.iterations,
        "throughput_samples_per_second": config.batch_size * args.iterations / seconds,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
        "synchronization": device.type == "cuda",
        "scope": (
            "Model-forward latency only; excludes event decoding, frame construction, "
            "host-to-device transfer, and neuromorphic hardware energy."
        ),
        "runtime": asdict(
            runtime_manifest(
                config,
                phase="fixed_forward_benchmark",
                split_hash=checkpoint["split_hash"],
            )
        ),
    }
    result["manifest_hash"] = stable_hash(result)
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
    cache.add_argument(
        "--split",
        action="append",
        choices=["train", "val", "test"],
        help="Cache only the selected split; may be repeated (default: all splits)",
    )
    cache.add_argument("--condition")
    cache.add_argument("--severity", type=float, default=0.0)
    cache.add_argument("--draw", type=int, default=0)
    cache.add_argument("--renormalize", action="store_true")
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
