#!/usr/bin/env python3
"""Resume locked-test diagnostics for the matched clean T=32 checkpoints.

Raw-event corruptions are cached one condition/draw at a time and reused by
all matched models. Frame-only temporal probes share one clean-frame cache.
Evaluations for a single run directory remain sequential, while different
models may evaluate concurrently. Temporal gradients and fixed forward
benchmarks run sequentially to avoid cross-process GPU timing interference.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.config import (  # noqa: E402
    ConditionConfig,
    ExperimentConfig,
    file_sha256,
    load_config,
    stable_hash,
)
from experiments.data import (  # noqa: E402
    PERTURBATION_NAMESPACE_SEED,
    frame_cache_namespace,
)

DEFAULT_DATASETS = ("cifar10_dvs", "dvs_gesture", "hardvs")
DEFAULT_MODELS = ("psn", "lpsn_a1_lif_init", "lpsn_a2_soft")
COMPARATOR_ONLY_MODELS = ("lif", "masked_psn", "sliding_psn", "rpsu")
DEFAULT_SEEDS = (42, 123, 256, 512, 1024)
DEFAULT_GRIDS = (
    "robustness",
    "temporal_order",
    "causal_prefix",
    "temporal_sparsity_draws",
)


@dataclass(frozen=True)
class EvaluationTask:
    config_path: Path
    condition: ConditionConfig


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT / path


def run_dir(config: ExperimentConfig) -> Path:
    return project_path(config.run_dir)


def evaluation_key(condition: ConditionConfig) -> str:
    return f"{condition.name}|{condition.severity:.12g}|{condition.draw}"


def ledger_keys(config: ExperimentConfig) -> set[str]:
    path = run_dir(config) / "test_evaluation_ledger.jsonl"
    if not path.exists():
        return set()
    return {
        json.loads(line)["evaluation_key"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


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


def selected_configs(
    configs_root: Path,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    time_steps: int,
) -> list[tuple[Path, ExperimentConfig]]:
    found: dict[tuple[str, str, int], tuple[Path, ExperimentConfig]] = {}
    for path in sorted(configs_root.rglob("*.yml")):
        config = load_config(path)
        if (
            config.experiment_id != "temporal_native"
            or config.dataset not in datasets
            or config.model not in models
            or config.seed not in seeds
            or config.time_steps != time_steps
            or config.condition.name != "clean"
        ):
            continue
        key = (config.dataset, config.model, config.seed)
        if key in found:
            raise RuntimeError(f"Duplicate selected configuration for {key}")
        found[key] = (path.resolve(), config)
    expected = {
        (dataset, model, seed)
        for dataset in datasets
        for model in models
        for seed in seeds
    }
    missing = sorted(expected - set(found))
    if missing:
        raise FileNotFoundError(
            "Missing clean temporal-native diagnostic configurations: "
            + ", ".join(map(str, missing))
        )
    selected = [found[key] for key in sorted(found)]
    incomplete = [
        str(path.relative_to(PROJECT))
        for path, config in selected
        if not (run_dir(config) / "train_result.json").is_file()
        or "clean|0|0" not in ledger_keys(config)
    ]
    if incomplete:
        raise RuntimeError(
            "Diagnostics require completed clean training and locked-test runs: "
            + ", ".join(incomplete)
        )
    return selected


def benchmark_configs(
    configs_root: Path,
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[tuple[Path, ExperimentConfig]]:
    """Select native-T scaling runs plus non-duplicated HARDVS comparators."""

    selected: dict[Path, ExperimentConfig] = {}
    for path in sorted(configs_root.rglob("*.yml")):
        config = load_config(path)
        if config.seed not in seeds or config.condition.name != "clean":
            continue
        native = (
            config.experiment_id == "temporal_native"
            and config.dataset in datasets
            and config.model in DEFAULT_MODELS
        )
        comparator = (
            config.experiment_id == "hardvs_main"
            and config.dataset == "hardvs"
            and "hardvs" in datasets
            and config.model in COMPARATOR_ONLY_MODELS
            and config.time_steps == 16
        )
        if native or comparator:
            selected[path.resolve()] = config
    expected = (
        len(datasets) * len(DEFAULT_MODELS) * 4 * len(seeds)
        + (len(COMPARATOR_ONLY_MODELS) * len(seeds) if "hardvs" in datasets else 0)
    )
    if len(selected) != expected:
        raise FileNotFoundError(
            f"Benchmark selection has {len(selected)} configs, expected {expected}"
        )
    incomplete = [
        str(path.relative_to(PROJECT))
        for path, config in selected.items()
        if not (run_dir(config) / "train_result.json").is_file()
        or config_evaluation_key(config) not in ledger_keys(config)
    ]
    if incomplete:
        raise RuntimeError(
            "Benchmarks require completed declared evaluations: "
            + ", ".join(incomplete)
        )
    return sorted(selected.items(), key=lambda item: str(item[0]))


def config_evaluation_key(config: ExperimentConfig) -> str:
    return evaluation_key(config.condition)


def load_conditions(path: Path, grids: tuple[str, ...]) -> list[ConditionConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    claimed_hash = payload.pop("manifest_hash", None)
    if claimed_hash != stable_hash(payload):
        raise RuntimeError(f"Diagnostic-grid manifest hash mismatch: {path}")
    if int(payload.get("base_seed", -1)) != PERTURBATION_NAMESPACE_SEED:
        raise RuntimeError("Diagnostic-grid perturbation seed differs from the code")
    conditions: dict[str, ConditionConfig] = {}
    for grid in grids:
        if grid not in payload["grids"]:
            raise KeyError(f"Unknown diagnostic grid {grid!r}")
        for row in payload["grids"][grid]:
            for draw in range(int(row["draws"])):
                condition = ConditionConfig(
                    name=str(row["name"]),
                    severity=float(row["severity"]),
                    draw=draw,
                )
                conditions[evaluation_key(condition)] = condition
    return [conditions[key] for key in sorted(conditions)]


def cache_command(
    python: str,
    config_path: Path,
    condition: ConditionConfig,
) -> list[str]:
    return [
        python,
        "-u",
        "-m",
        "experiments.cli",
        "cache",
        "--config",
        str(config_path),
        "--split",
        "test",
        "--condition",
        condition.name,
        "--severity",
        str(condition.severity),
        "--draw",
        str(condition.draw),
    ]


def evaluate_command(
    python: str,
    task: EvaluationTask,
) -> list[str]:
    return [
        python,
        "-u",
        "-m",
        "experiments.cli",
        "evaluate",
        "--config",
        str(task.config_path),
        "--condition",
        task.condition.name,
        "--severity",
        str(task.condition.severity),
        "--draw",
        str(task.condition.draw),
    ]


def safe_remove_cache(namespace: Path, cache_root: Path) -> None:
    namespace = namespace.resolve()
    cache_root = project_path(cache_root).resolve()
    if namespace == cache_root or cache_root not in namespace.parents:
        raise RuntimeError(
            f"Refusing unsafe cache removal: {namespace} is not below {cache_root}"
        )
    if namespace.exists():
        shutil.rmtree(namespace)


def run_model_task_sequence(
    tasks: list[EvaluationTask],
    *,
    python: str,
    logs_root: Path,
    environment: dict[str, str],
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for task in tasks:
        config = load_config(task.config_path)
        key = evaluation_key(task.condition)
        if key in ledger_keys(config):
            continue
        relative = task.config_path.relative_to(PROJECT)
        log_name = stable_hash(f"{relative}|{key}")[:20] + ".log"
        execute(
            evaluate_command(python, task),
            logs_root / "evaluations" / log_name,
            environment,
        )
        completed.append({"config": str(relative), "evaluation_key": key})
    return completed


def valid_hashed_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    claimed = payload.pop("manifest_hash", None)
    return bool(claimed) and claimed == stable_hash(payload)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs-root", type=Path, default=PROJECT / "generated_configs"
    )
    parser.add_argument(
        "--grids-manifest",
        type=Path,
        default=PROJECT / "configs" / "evaluation_grids.yml",
    )
    parser.add_argument("--dataset", action="append", choices=DEFAULT_DATASETS)
    parser.add_argument("--model", action="append", choices=DEFAULT_MODELS)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument("--grid", action="append")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--temporal-worker-budget", type=int, default=64)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--logs-root", type=Path, default=PROJECT / "diagnostic_logs"
    )
    parser.add_argument("--retain-cache", action="store_true")
    parser.add_argument("--skip-grids", action="store_true")
    parser.add_argument("--skip-influence", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--benchmark-scope",
        choices=["temporal_and_comparators", "selected"],
        default="temporal_and_comparators",
        help=(
            "Benchmark all native-T scaling cells plus the four additional "
            "HARDVS comparators, or only the T=32 diagnostic selection."
        ),
    )
    parser.add_argument("--max-influence-batches", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    if args.workers < 1 or args.temporal_worker_budget < 1:
        parser.error("worker counts and temporal budget must be positive")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("benchmark iteration counts are invalid")

    datasets = tuple(args.dataset or DEFAULT_DATASETS)
    models = tuple(args.model or DEFAULT_MODELS)
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    grids = tuple(args.grid or DEFAULT_GRIDS)
    selected = selected_configs(
        args.configs_root.resolve(),
        datasets,
        models,
        seeds,
        args.time_steps,
    )
    benchmark_selected = (
        selected
        if args.benchmark_scope == "selected" or args.skip_benchmark
        else benchmark_configs(args.configs_root.resolve(), datasets, seeds)
    )
    conditions = load_conditions(args.grids_manifest, grids)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("OMP_NUM_THREADS", "4")

    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": now(),
        "selection": {
            "datasets": datasets,
            "models": models,
            "seeds": seeds,
            "time_steps": args.time_steps,
            "configs": len(selected),
            "grids": grids,
            "grids_manifest": str(args.grids_manifest),
            "grids_manifest_sha256": file_sha256(args.grids_manifest),
            "unique_grid_conditions": len(conditions),
            "benchmark_scope": args.benchmark_scope,
            "benchmark_configs": len(benchmark_selected),
        },
        "grid_evaluations_completed_this_run": [],
        "influence_completed_this_run": [],
        "benchmarks_completed_this_run": [],
    }
    report_path = args.logs_root / "diagnostic_status.json"
    write_report(report_path, report)

    if not args.skip_grids:
        groups: dict[Path, list[EvaluationTask]] = {}
        for config_path, base in selected:
            completed_keys = ledger_keys(base)
            for condition in conditions:
                key = evaluation_key(condition)
                if key in completed_keys:
                    continue
                configured = replace(base, condition=condition)
                namespace = frame_cache_namespace(configured)
                if namespace is None:
                    raise RuntimeError(
                        f"Condition has no reusable cache namespace: {condition}"
                    )
                groups.setdefault(project_path(namespace).resolve(), []).append(
                    EvaluationTask(config_path, condition)
                )

        for index, (namespace, tasks) in enumerate(
            sorted(groups.items(), key=lambda item: str(item[0])), start=1
        ):
            representative = tasks[0]
            representative_config = load_config(representative.config_path)
            print(
                f"[diagnostic cache {index}/{len(groups)}] {namespace} "
                f"evaluations={len(tasks)}",
                flush=True,
            )
            cache_log = (
                args.logs_root
                / "cache"
                / f"{stable_hash(str(namespace))[:20]}.log"
            )
            execute(
                cache_command(
                    args.python,
                    representative.config_path,
                    representative.condition,
                ),
                cache_log,
                environment,
            )
            by_config: dict[Path, list[EvaluationTask]] = {}
            for task in tasks:
                by_config.setdefault(task.config_path, []).append(task)
            group_workers = min(
                args.workers,
                len(by_config),
                max(1, args.temporal_worker_budget // args.time_steps),
            )
            failures: list[tuple[Path, BaseException]] = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=group_workers
            ) as executor:
                futures = {
                    executor.submit(
                        run_model_task_sequence,
                        sorted(
                            model_tasks,
                            key=lambda task: evaluation_key(task.condition),
                        ),
                        python=args.python,
                        logs_root=args.logs_root,
                        environment=environment,
                    ): config_path
                    for config_path, model_tasks in by_config.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    config_path = futures[future]
                    try:
                        completed = future.result()
                        report["grid_evaluations_completed_this_run"].extend(
                            completed
                        )
                    except BaseException as error:
                        failures.append((config_path, error))
            write_report(report_path, report)
            if failures:
                raise RuntimeError(
                    "Diagnostic group failed; cache retained for resume: "
                    + "; ".join(f"{path}: {error}" for path, error in failures)
                )
            if not args.retain_cache:
                cache_root = Path(str(representative_config.frame_cache_root))
                safe_remove_cache(namespace, cache_root)

    for config_path, config in selected:
        relative = config_path.relative_to(PROJECT)
        if not args.skip_influence:
            output = run_dir(config) / "temporal_influence.json"
            if not valid_hashed_json(output):
                command = [
                    args.python,
                    "-u",
                    "-m",
                    "experiments.cli",
                    "temporal-influence",
                    "--config",
                    str(config_path),
                    "--output",
                    str(output),
                ]
                if args.max_influence_batches:
                    command.extend(
                        ["--max-batches", str(args.max_influence_batches)]
                    )
                execute(
                    command,
                    args.logs_root
                    / "influence"
                    / f"{stable_hash(str(relative))[:20]}.log",
                    environment,
                )
                report["influence_completed_this_run"].append(str(relative))
                write_report(report_path, report)

    if not args.skip_benchmark:
        for config_path, config in benchmark_selected:
            relative = config_path.relative_to(PROJECT)
            output = run_dir(config) / "benchmark.json"
            if valid_hashed_json(output):
                continue
            execute(
                [
                    args.python,
                    "-u",
                    "-m",
                    "experiments.cli",
                    "benchmark",
                    "--config",
                    str(config_path),
                    "--warmup",
                    str(args.warmup),
                    "--iterations",
                    str(args.iterations),
                    "--output",
                    str(output),
                ],
                args.logs_root
                / "benchmark"
                / f"{stable_hash(str(relative))[:20]}.log",
                environment,
            )
            report["benchmarks_completed_this_run"].append(str(relative))
            write_report(report_path, report)

    report["completed_at"] = now()
    report["manifest_hash"] = stable_hash(report)
    write_report(report_path, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_configs": len(selected),
                "grid_conditions": len(conditions),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
