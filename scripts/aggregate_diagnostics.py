#!/usr/bin/env python3
"""Aggregate fixed benchmarks and temporal-influence manifests by seed."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.config import file_sha256, load_config, stable_hash  # noqa: E402
from experiments.data import DATASET_META, load_split_manifest  # noqa: E402


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT / path


def run_dir(config) -> Path:
    return project_path(config.run_dir)


def read_hashed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("manifest_hash", None)
    if claimed != stable_hash(payload):
        raise RuntimeError(f"Manifest hash mismatch: {path}")
    payload["manifest_hash"] = claimed
    return payload


def interval(values: Iterable[float], seed: int = 20240723) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(2000, len(array)))
    estimates = array[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output: list[dict[str, Any]] = []
    for key, selected in sorted(groups.items()):
        result = dict(zip(group_keys, key))
        result["n_seeds"] = len({int(row["seed"]) for row in selected})
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in selected], dtype=np.float64
            )
            low, high = interval(values)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            result[f"{metric}_median"] = float(np.median(values))
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        output.append(result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs-root", type=Path, default=PROJECT / "generated_configs"
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT / "aggregates"
    )
    args = parser.parse_args()

    benchmarks: list[dict[str, Any]] = []
    influence_timesteps: list[dict[str, Any]] = []
    influence_horizons: list[dict[str, Any]] = []
    dataset_characteristics: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    split_paths: dict[str, Path] = {}
    for config_path in sorted(args.configs_root.rglob("*.yml")):
        config = load_config(config_path)
        split_paths.setdefault(config.dataset, project_path(Path(config.split_manifest)))
        common = {
            "experiment_id": config.experiment_id,
            "dataset": config.dataset,
            "model": config.model,
            "seed": config.seed,
            "time_steps": config.time_steps,
        }
        benchmark_path = run_dir(config) / "benchmark.json"
        if benchmark_path.is_file():
            payload = read_hashed(benchmark_path)
            benchmarks.append(
                {
                    **common,
                    "latency_ms_per_batch": payload["latency_ms_per_batch"],
                    "throughput_samples_per_second": payload[
                        "throughput_samples_per_second"
                    ],
                    "peak_cuda_memory_mb": payload["peak_cuda_memory_mb"],
                    "batch_size": payload["batch_size"],
                    "warmup_iterations": payload["warmup_iterations"],
                    "timed_iterations": payload["timed_iterations"],
                    "device_name": payload["device_name"],
                    "checkpoint_sha256": payload["checkpoint_sha256"],
                }
            )
            sources.append(
                {
                    "path": str(benchmark_path.relative_to(PROJECT)),
                    "sha256": file_sha256(benchmark_path),
                }
            )
        influence_path = run_dir(config) / "temporal_influence.json"
        if influence_path.is_file():
            payload = read_hashed(influence_path)
            values = payload["normalized_absolute_input_gradient"]
            if len(values) != config.time_steps:
                raise RuntimeError(
                    f"Influence length mismatch in {influence_path}: {len(values)}"
                )
            for timestep, value in enumerate(values):
                influence_timesteps.append(
                    {**common, "timestep": timestep, "normalized_influence": value}
                )
            influence_horizons.append(
                {
                    **common,
                    "effective_horizon_90_percent": payload[
                        "effective_horizon_90_percent"
                    ],
                    "samples": payload["samples"],
                }
            )
            sources.append(
                {
                    "path": str(influence_path.relative_to(PROJECT)),
                    "sha256": file_sha256(influence_path),
                }
            )

    for dataset, split_path in sorted(split_paths.items()):
        manifest = load_split_manifest(split_path)
        class_count = int(DATASET_META[dataset]["classes"])
        for split_name in ("train", "val", "test"):
            counts = np.bincount(
                [
                    int(row["label"])
                    for row in manifest["splits"][split_name]
                ],
                minlength=class_count,
            ).astype(np.float64)
            dataset_characteristics.append(
                {
                    "dataset": dataset,
                    "split": split_name,
                    "samples": int(counts.sum()),
                    "classes": class_count,
                    "samples_per_class_min": int(counts.min()),
                    "samples_per_class_max": int(counts.max()),
                    "samples_per_class_mean": float(counts.mean()),
                    "samples_per_class_population_std": float(counts.std(ddof=0)),
                    "samples_per_class_cv": (
                        float(counts.std(ddof=0) / counts.mean())
                        if counts.mean() > 0
                        else 0.0
                    ),
                    "split_hash": manifest["split_hash"],
                    "integrity_mode": manifest.get("dataset_integrity", {}).get(
                        "integrity_mode"
                    ),
                }
            )
        sources.append(
            {
                "path": str(split_path.relative_to(PROJECT)),
                "sha256": file_sha256(split_path),
            }
        )

    benchmark_summary = summarize(
        benchmarks,
        group_keys=("experiment_id", "dataset", "model", "time_steps"),
        metrics=(
            "latency_ms_per_batch",
            "throughput_samples_per_second",
            "peak_cuda_memory_mb",
        ),
    )
    influence_summary = summarize(
        influence_timesteps,
        group_keys=(
            "experiment_id",
            "dataset",
            "model",
            "time_steps",
            "timestep",
        ),
        metrics=("normalized_influence",),
    )
    horizon_summary = summarize(
        influence_horizons,
        group_keys=("experiment_id", "dataset", "model", "time_steps"),
        metrics=("effective_horizon_90_percent",),
    )
    write_csv(args.output_root / "benchmark_results.csv", benchmarks)
    write_csv(args.output_root / "benchmark_summary.csv", benchmark_summary)
    write_csv(
        args.output_root / "temporal_influence_results.csv",
        influence_timesteps,
    )
    write_csv(
        args.output_root / "temporal_influence_summary.csv",
        influence_summary,
    )
    write_csv(
        args.output_root / "temporal_horizon_summary.csv",
        horizon_summary,
    )
    write_csv(
        args.output_root / "dataset_split_characteristics.csv",
        dataset_characteristics,
    )
    manifest = {
        "schema_version": 1,
        "benchmark_rows": len(benchmarks),
        "benchmark_groups": len(benchmark_summary),
        "influence_timestep_rows": len(influence_timesteps),
        "influence_groups": len(influence_summary),
        "horizon_rows": len(influence_horizons),
        "horizon_groups": len(horizon_summary),
        "dataset_split_rows": len(dataset_characteristics),
        "bootstrap": "seed bootstrap, 2000 replicates",
        "sources": sorted(sources, key=lambda row: row["path"]),
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path = args.output_root / "diagnostics.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "benchmark_rows": len(benchmarks),
                "influence_rows": len(influence_timesteps),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
