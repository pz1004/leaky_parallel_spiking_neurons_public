#!/usr/bin/env python3
"""Resume the declared study matrix with cache-aware GPU scheduling.

Configurations that are mathematically identical at severity 1.0 share one
physical deterministic training run. Their run directories retain explicit
equivalence provenance; they are not counted as independent replications.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.config import ExperimentConfig, load_config, stable_hash  # noqa: E402
from experiments.data import frame_cache_namespace, load_split_manifest  # noqa: E402

IDENTITY_CONDITIONS = {"event_thin", "event_thin_renorm", "amplitude_scale", "temporal_sparse"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_dir(config: ExperimentConfig) -> Path:
    path = config.run_dir
    return path if path.is_absolute() else PROJECT / path


def evaluation_key(config: ExperimentConfig) -> str:
    condition = config.condition
    return f"{condition.name}|{condition.severity:.12g}|{condition.draw}"


def evaluation_complete(config: ExperimentConfig) -> bool:
    ledger = run_dir(config) / "test_evaluation_ledger.jsonl"
    if not ledger.exists():
        return False
    keys = {
        json.loads(line)["evaluation_key"]
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return evaluation_key(config) in keys


def execute(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] {' '.join(command)}\n")
        log.flush()
        subprocess.run(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_one(
    config_path: Path,
    configs_root: Path,
    python: str,
    logs_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    config = load_config(config_path)
    relative = config_path.relative_to(configs_root)
    log_path = logs_root / relative.with_suffix(".log")
    if not (run_dir(config) / "train_result.json").exists():
        execute([python, "-u", "-m", "experiments.cli", "train", "--config", str(config_path)], log_path, environment)
    if not evaluation_complete(config):
        execute([python, "-u", "-m", "experiments.cli", "evaluate", "--config", str(config_path)], log_path, environment)
    return {"config": str(relative), "run_dir": str(run_dir(config)), "status": "complete"}


def training_signature(config: ExperimentConfig) -> tuple[Any, ...]:
    values = asdict(config)
    for key in ("experiment_id", "condition"):
        values.pop(key)
    return tuple(sorted((key, json.dumps(value, sort_keys=True)) for key, value in values.items()))


def alias_map(configs: list[tuple[Path, ExperimentConfig]]) -> dict[Path, Path]:
    clean_by_signature: dict[tuple[Any, ...], list[tuple[Path, ExperimentConfig]]] = {}
    for path, config in configs:
        if config.condition.name == "clean":
            clean_by_signature.setdefault(training_signature(config), []).append((path, config))

    aliases: dict[Path, Path] = {}
    for signature, candidates in clean_by_signature.items():
        preferred = sorted(
            candidates,
            key=lambda item: (
                item[1].experiment_id != "temporal_native",
                item[1].experiment_id,
                str(item[0]),
            ),
        )[0]
        for path, _config in candidates:
            if path != preferred[0]:
                aliases[path] = preferred[0]

    for path, config in configs:
        condition = config.condition
        if condition.name not in IDENTITY_CONDITIONS or condition.severity != 1.0:
            continue
        candidates = clean_by_signature.get(training_signature(config), [])
        if candidates:
            source = sorted(
                candidates,
                key=lambda item: (
                    item[1].experiment_id != "temporal_native",
                    item[1].experiment_id,
                    str(item[0]),
                ),
            )[0]
            aliases[path] = source[0]
    return aliases


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def materialize_alias(target_path: Path, source_path: Path) -> dict[str, Any]:
    target = load_config(target_path)
    source = load_config(source_path)
    source_dir = run_dir(source)
    target_dir = run_dir(target)
    if not (source_dir / "train_result.json").exists() or not evaluation_complete(source):
        raise RuntimeError(f"canonical source is incomplete: {source_path}")
    target_dir.mkdir(parents=True, exist_ok=True)

    proof = {
        "kind": "deterministic_identity_reuse",
        "created_at": now(),
        "source_config": str(source_path.relative_to(PROJECT)),
        "target_config": str(target_path.relative_to(PROJECT)),
        "source_config_hash": source.config_hash,
        "target_config_hash": target.config_hash,
        "basis": (
            "All non-condition hyperparameters and the seed are identical. "
            "The target transform is the identity at severity 1.0, or the "
            "clean target is a duplicate declaration in another study."
        ),
        "independent_replication": False,
    }
    (target_dir / "equivalence_provenance.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target_dir / "config.yml").write_text(
        yaml.safe_dump(asdict(target), sort_keys=True),
        encoding="utf-8",
    )
    for name in (
        "checkpoint_best.pt",
        "validation_confusion_best.npy",
        "metrics.jsonl",
        "environment.json",
    ):
        link_or_copy(source_dir / name, target_dir / name)

    result = json.loads((source_dir / "train_result.json").read_text(encoding="utf-8"))
    result.update(
        {
            "experiment_id": target.experiment_id,
            "condition": asdict(target.condition),
            "config_hash": target.config_hash,
            "equivalent_training_source": proof["source_config"],
            "independent_replication": False,
        }
    )
    (target_dir / "train_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split = load_split_manifest(Path(target.split_manifest))
    run_manifest = {
        "schema_version": 1,
        "created_at": now(),
        "phase": "train_validation_equivalent_reuse",
        "experiment_id": target.experiment_id,
        "config_hash": target.config_hash,
        "split_hash": split["split_hash"],
        "dataset": target.dataset,
        "model": target.model,
        "seed": target.seed,
        "time_steps": target.time_steps,
        "condition": asdict(target.condition),
        "equivalence_provenance": proof,
    }
    (target_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_rows = [
        json.loads(line)
        for line in (source_dir / "locked_test_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_clean = next(
        row
        for row in source_rows
        if row["condition"] == source.condition.name
        and float(row["severity"]) == source.condition.severity
        and int(row["draw"]) == source.condition.draw
    )
    source_samples = Path(source_clean["per_sample_path"])
    if not source_samples.is_absolute():
        source_samples = PROJECT / source_samples
    output_hash = stable_hash(evaluation_key(target))[:12]
    target_samples = target_dir / f"per_sample_{output_hash}.jsonl"
    link_or_copy(source_samples, target_samples)

    source_hash = stable_hash(evaluation_key(source))[:12]
    link_or_copy(
        source_dir / f"confusion_{source_hash}.npy",
        target_dir / f"confusion_{output_hash}.npy",
    )
    payload = dict(source_clean)
    payload.update(
        {
            "experiment_id": target.experiment_id,
            "config_hash": target.config_hash,
            "condition": target.condition.name,
            "severity": target.condition.severity,
            "draw": target.condition.draw,
            "evaluated_at": now(),
            "per_sample_path": str(target_samples.relative_to(PROJECT)),
            "equivalent_evaluation_source": proof["source_config"],
            "independent_replication": False,
        }
    )
    (target_dir / "locked_test_metrics.jsonl").write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target_dir / "test_evaluation_ledger.jsonl").write_text(
        json.dumps(
            {
                "evaluation_key": evaluation_key(target),
                "evaluated_at": payload["evaluated_at"],
                "checkpoint_epoch": payload["checkpoint_epoch"],
                "split_hash": split["split_hash"],
                "equivalent_evaluation_source": proof["source_config"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"target": str(target_path.relative_to(PROJECT)), "source": proof["source_config"]}


def safe_remove_cache(namespace: Path, frame_cache_root: Path) -> None:
    namespace = namespace.resolve()
    root = frame_cache_root.resolve()
    if namespace == root or root not in namespace.parents:
        raise RuntimeError(f"refusing unsafe cache removal: {namespace} is not below {root}")
    if namespace.exists():
        shutil.rmtree(namespace)
        print(f"[removed completed frame cache] {namespace}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-root", type=Path, default=PROJECT / "generated_configs")
    parser.add_argument("--dataset", action="append", choices=["cifar10_dvs", "dvs_gesture", "hardvs"])
    parser.add_argument("--study", action="append")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--temporal-worker-budget",
        type=int,
        default=64,
        help=(
            "Maximum sum of T across concurrent jobs in one cache group. "
            "The default permits 4xT16, 2xT32, or 1xT64 on the reference A100."
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--logs-root", type=Path, default=PROJECT / "orchestration_logs")
    parser.add_argument("--retain-cache", action="store_true")
    parser.add_argument("--no-equivalent-reuse", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.temporal_worker_budget < 1:
        parser.error("--temporal-worker-budget must be positive")
    configs_root = args.configs_root.resolve()

    loaded: list[tuple[Path, ExperimentConfig]] = []
    for path in sorted(configs_root.rglob("*.yml")):
        config = load_config(path)
        if args.dataset and config.dataset not in args.dataset:
            continue
        if args.study and config.experiment_id.split("_", 1)[0] not in args.study and not any(
            config.experiment_id.startswith(value) for value in args.study
        ):
            continue
        loaded.append((path.resolve(), config))
    if not loaded:
        raise FileNotFoundError("No matching generated configurations")

    aliases = {} if args.no_equivalent_reuse else alias_map(loaded)
    physical = [(path, config) for path, config in loaded if path not in aliases]
    groups: dict[Path, list[Path]] = {}
    for path, config in physical:
        namespace = frame_cache_namespace(config)
        if namespace is None:
            raise RuntimeError(f"matrix training condition is not cacheable: {path}")
        groups.setdefault(namespace.resolve(), []).append(path)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment.setdefault("OMP_NUM_THREADS", "4")
    report: dict[str, Any] = {
        "started_at": now(),
        "selected_configs": len(loaded),
        "physical_runs": len(physical),
        "equivalent_aliases": len(aliases),
        "groups": len(groups),
        "completed": [],
        "aliases": [],
    }
    report_path = args.logs_root / "matrix_status.json"
    args.logs_root.mkdir(parents=True, exist_ok=True)

    for group_index, (namespace, paths) in enumerate(sorted(groups.items(), key=lambda item: str(item[0])), start=1):
        representative = paths[0]
        print(
            f"[group {group_index}/{len(groups)}] cache={namespace} runs={len(paths)}",
            flush=True,
        )
        representative_config = load_config(representative)
        group_workers = min(
            args.workers,
            max(1, args.temporal_worker_budget // representative_config.time_steps),
        )
        print(
            f"[group {group_index}/{len(groups)}] workers={group_workers} "
            f"(T={representative_config.time_steps}, temporal budget={args.temporal_worker_budget})",
            flush=True,
        )
        cache_log = args.logs_root / "cache" / f"{stable_hash(str(namespace))[:16]}.log"
        execute(
            [args.python, "-u", "-m", "experiments.cli", "cache", "--config", str(representative)],
            cache_log,
            environment,
        )
        failures: list[tuple[Path, BaseException]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=group_workers) as executor:
            futures = {
                executor.submit(
                    run_one,
                    path,
                    configs_root,
                    args.python,
                    args.logs_root / "runs",
                    environment,
                ): path
                for path in paths
            }
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    report["completed"].append(result)
                    print(f"[complete] {result['config']}", flush=True)
                except BaseException as error:
                    failures.append((path, error))
                    print(f"[failed] {path}: {error}", file=sys.stderr, flush=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if failures:
            raise RuntimeError(
                "matrix group failed; cache retained for resume: "
                + "; ".join(f"{path}: {error}" for path, error in failures)
            )
        if not args.retain_cache:
            safe_remove_cache(namespace, Path(str(representative_config.frame_cache_root)))

    for target, source in sorted(aliases.items(), key=lambda item: str(item[0])):
        result = materialize_alias(target, source)
        report["aliases"].append(result)
        print(f"[equivalent reuse] {result['target']} <- {result['source']}", flush=True)

    report["completed_at"] = now()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"completed", "aliases"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
