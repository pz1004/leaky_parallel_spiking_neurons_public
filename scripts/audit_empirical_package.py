#!/usr/bin/env python3
"""Audit matrix completeness, sealed test artifacts, and diagnostic coverage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.config import file_sha256, load_config, stable_hash  # noqa: E402
from experiments.data import (  # noqa: E402
    DATASET_META,
    PERTURBATION_NAMESPACE_SEED,
    load_split_manifest,
)

DIAGNOSTIC_MODELS = {"psn", "lpsn_a1_lif_init", "lpsn_a2_soft"}
DIAGNOSTIC_DATASETS = {"cifar10_dvs", "dvs_gesture", "hardvs"}
COMPARATOR_ONLY_MODELS = {"lif", "masked_psn", "sliding_psn", "rpsu"}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def run_dir(config) -> Path:
    return project_path(config.run_dir)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluation_key(row: dict[str, Any]) -> str:
    return f"{row['condition']}|{float(row['severity']):.12g}|{int(row['draw'])}"


def config_evaluation_key(config) -> str:
    condition = config.condition
    return f"{condition.name}|{condition.severity:.12g}|{condition.draw}"


def hashed_json(path: Path) -> tuple[dict[str, Any], bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("manifest_hash", None)
    valid = bool(claimed) and claimed == stable_hash(payload)
    if claimed is not None:
        payload["manifest_hash"] = claimed
    return payload, valid


def diagnostic_keys(path: Path, grids: tuple[str, ...]) -> set[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    claimed_hash = payload.pop("manifest_hash", None)
    if claimed_hash != stable_hash(payload):
        raise RuntimeError(f"Diagnostic-grid manifest hash mismatch: {path}")
    if int(payload.get("base_seed", -1)) != PERTURBATION_NAMESPACE_SEED:
        raise RuntimeError("Diagnostic-grid perturbation seed differs from the code")
    keys: set[str] = set()
    for grid in grids:
        for row in payload["grids"][grid]:
            for draw in range(int(row["draws"])):
                keys.add(
                    f"{row['name']}|{float(row['severity']):.12g}|{draw}"
                )
    return keys


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
    parser.add_argument(
        "--grid",
        action="append",
        default=None,
        help="Required diagnostic grid (default: all four committed grids)",
    )
    parser.add_argument("--expected-configs", type=int, default=695)
    parser.add_argument("--require-diagnostics", action="store_true")
    parser.add_argument("--skip-artifact-hashes", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "aggregates" / "empirical_audit.json",
    )
    args = parser.parse_args()

    grids = tuple(
        args.grid
        or (
            "robustness",
            "temporal_order",
            "causal_prefix",
            "temporal_sparsity_draws",
        )
    )
    expected_diagnostics = diagnostic_keys(args.grids_manifest, grids)
    errors: list[str] = []
    warnings: list[str] = []
    hardvs_sources: dict[str, dict[str, Any]] = {}
    stats: dict[str, int] = {
        "configs": 0,
        "complete_training_runs": 0,
        "complete_declared_evaluations": 0,
        "evaluation_manifests": 0,
        "evaluation_rows_audited": 0,
        "aliases": 0,
        "native_physical_runs": 0,
        "diagnostic_configs": 0,
        "diagnostic_evaluations_expected": 0,
        "diagnostic_evaluations_present": 0,
        "influence_manifests": 0,
        "benchmark_manifests_expected": 0,
        "benchmark_manifests": 0,
    }
    digest_cache: dict[tuple[int, int, int, int], str] = {}

    def digest(path: Path) -> str:
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in digest_cache:
            digest_cache[key] = file_sha256(path)
        return digest_cache[key]

    config_paths = sorted(args.configs_root.rglob("*.yml"))
    stats["configs"] = len(config_paths)
    if len(config_paths) != args.expected_configs:
        errors.append(
            f"generated config count {len(config_paths)} != {args.expected_configs}"
        )

    for config_path in config_paths:
        config = load_config(config_path)
        directory = run_dir(config)
        label = display_path(config_path)
        split_path = project_path(config.split_manifest)
        try:
            split = load_split_manifest(split_path)
        except Exception as error:
            errors.append(f"{label}: invalid split manifest: {error}")
            continue
        test_count = len(split["splits"]["test"])
        if config.smoke_samples > 0:
            test_count = min(test_count, config.smoke_samples)
        classes = int(DATASET_META[config.dataset]["classes"])
        if config.dataset == "hardvs":
            integrity = split.get("dataset_integrity", {})
            hardvs_sources.setdefault(
                str(split["split_hash"]),
                {
                    "split_hash": split["split_hash"],
                    "integrity_mode": integrity.get("integrity_mode"),
                    "source_deviation": integrity.get("source_deviation"),
                    "archive_md5": integrity.get("archive_md5"),
                    "archive_sha256": integrity.get("archive_sha256"),
                    "acquisition_record_hash": integrity.get(
                        "acquisition_record_hash"
                    ),
                    "preparation_git_commit": integrity.get(
                        "preparation_git_commit"
                    ),
                    "preparation_git_dirty": integrity.get(
                        "preparation_git_dirty"
                    ),
                    "preparation_script_sha256": integrity.get(
                        "preparation_script_sha256"
                    ),
                    "effective_counts": integrity.get("effective_counts"),
                    "excluded_sample_ids": integrity.get("excluded_sample_ids"),
                },
            )
            if (
                integrity.get("integrity_mode")
                == "known_mirror_exclusion"
                and not integrity.get("source_deviation")
            ):
                errors.append(
                    f"{label}: known-mirror HARDVS source deviation is not disclosed"
                )
        required = (
            "config.yml",
            "environment.json",
            "run_manifest.json",
            "train_result.json",
            "checkpoint_best.pt",
            "locked_test_metrics.jsonl",
            "test_evaluation_ledger.jsonl",
        )
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            errors.append(f"{label}: missing {missing}")
            continue

        try:
            environment = json.loads(
                (directory / "environment.json").read_text(encoding="utf-8")
            )
            training_manifest = json.loads(
                (directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            train_result = json.loads(
                (directory / "train_result.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{label}: invalid training metadata: {error}")
            continue
        if training_manifest.get("config_hash") != config.config_hash:
            errors.append(f"{label}: training manifest config hash mismatch")
        if training_manifest.get("split_hash") != split["split_hash"]:
            errors.append(f"{label}: training manifest split hash mismatch")
        if bool(training_manifest.get("git_dirty")):
            errors.append(f"{label}: training manifest records a dirty worktree")
        if bool(environment.get("git_dirty")):
            errors.append(f"{label}: environment records a dirty worktree")
        if training_manifest.get("git_commit") != environment.get("git_commit"):
            errors.append(f"{label}: training/environment git commits differ")
        if train_result.get("best_epoch") is None:
            errors.append(f"{label}: training result lacks best_epoch")
        proof_path = directory / "equivalence_provenance.json"
        if proof_path.is_file():
            stats["aliases"] += 1
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            if proof.get("independent_replication") is not False:
                errors.append(f"{label}: alias is not marked non-independent")
        else:
            stats["native_physical_runs"] += 1
        stats["complete_training_runs"] += 1

        try:
            metrics = read_jsonl(directory / "locked_test_metrics.jsonl")
            ledger = read_jsonl(directory / "test_evaluation_ledger.jsonl")
        except Exception as error:
            errors.append(f"{label}: invalid evaluation JSONL: {error}")
            continue
        ledger_by_key = {str(row.get("evaluation_key")): row for row in ledger}
        if len(ledger_by_key) != len(ledger):
            errors.append(f"{label}: duplicate ledger evaluation keys")
        metric_by_key = {evaluation_key(row): row for row in metrics}
        if len(metric_by_key) != len(metrics):
            errors.append(f"{label}: duplicate metric evaluation keys")
        if set(metric_by_key) != set(ledger_by_key):
            errors.append(f"{label}: metric/ledger evaluation-key sets differ")
        declared_key = config_evaluation_key(config)
        if declared_key not in metric_by_key:
            errors.append(f"{label}: declared evaluation {declared_key} is missing")
        else:
            stats["complete_declared_evaluations"] += 1

        checkpoint_path = directory / "checkpoint_best.pt"
        checkpoint_sha = (
            None if args.skip_artifact_hashes else digest(checkpoint_path)
        )
        for key, row in metric_by_key.items():
            stats["evaluation_rows_audited"] += 1
            ledger_row = ledger_by_key.get(key, {})
            if int(row.get("samples", -1)) != test_count:
                errors.append(
                    f"{label} {key}: metric samples {row.get('samples')} != {test_count}"
                )
            if row.get("split_hash") != split["split_hash"]:
                errors.append(f"{label} {key}: metric split hash mismatch")
            manifest_value = row.get("evaluation_manifest_path")
            if not manifest_value:
                errors.append(f"{label} {key}: missing evaluation manifest pointer")
                continue
            manifest_path = project_path(manifest_value)
            if not manifest_path.is_file():
                errors.append(f"{label} {key}: missing {manifest_path}")
                continue
            try:
                evaluation_manifest, valid = hashed_json(manifest_path)
            except Exception as error:
                errors.append(f"{label} {key}: invalid evaluation manifest: {error}")
                continue
            if not valid:
                errors.append(f"{label} {key}: evaluation manifest hash mismatch")
            stats["evaluation_manifests"] += 1
            if bool(evaluation_manifest.get("git_dirty")):
                errors.append(f"{label} {key}: evaluation records a dirty worktree")
            if evaluation_manifest.get("split_hash") != split["split_hash"]:
                errors.append(f"{label} {key}: evaluation manifest split mismatch")
            if evaluation_manifest.get("evaluation_key") != key:
                errors.append(f"{label} {key}: evaluation manifest key mismatch")
            if (
                ledger_row.get("evaluation_manifest_hash")
                != evaluation_manifest.get("manifest_hash")
            ):
                errors.append(f"{label} {key}: ledger/manifest hashes differ")
            sample_path = project_path(row["per_sample_path"])
            output_hash = stable_hash(key)[:12]
            confusion_path = directory / f"confusion_{output_hash}.npy"
            if not sample_path.is_file() or not confusion_path.is_file():
                errors.append(f"{label} {key}: missing sample/confusion artifact")
                continue
            try:
                sample_rows = sum(
                    bool(line.strip())
                    for line in sample_path.read_text(encoding="utf-8").splitlines()
                )
                confusion = np.load(confusion_path, allow_pickle=False)
            except Exception as error:
                errors.append(f"{label} {key}: unreadable evaluation artifact: {error}")
                continue
            if sample_rows != test_count:
                errors.append(
                    f"{label} {key}: per-sample rows {sample_rows} != {test_count}"
                )
            if confusion.shape != (classes, classes):
                errors.append(
                    f"{label} {key}: confusion shape {confusion.shape} "
                    f"!= {(classes, classes)}"
                )
            if int(confusion.sum()) != test_count:
                errors.append(
                    f"{label} {key}: confusion sum {int(confusion.sum())} != {test_count}"
                )
            if not args.skip_artifact_hashes:
                artifacts = evaluation_manifest.get("artifacts", {})
                if artifacts.get("per_sample_sha256") != digest(sample_path):
                    errors.append(f"{label} {key}: per-sample SHA-256 mismatch")
                if artifacts.get("confusion_sha256") != digest(confusion_path):
                    errors.append(f"{label} {key}: confusion SHA-256 mismatch")
                if evaluation_manifest.get("checkpoint_sha256") != checkpoint_sha:
                    errors.append(f"{label} {key}: checkpoint SHA-256 mismatch")
                if row.get("checkpoint_sha256") != checkpoint_sha:
                    errors.append(f"{label} {key}: metric checkpoint SHA-256 mismatch")
                if artifacts.get("metrics_payload_sha256") != stable_hash(row):
                    errors.append(f"{label} {key}: metric-payload hash mismatch")

        if (
            args.require_diagnostics
            and config.experiment_id == "temporal_native"
            and config.dataset in DIAGNOSTIC_DATASETS
            and config.model in DIAGNOSTIC_MODELS
            and config.time_steps == 32
            and config.condition.name == "clean"
        ):
            stats["diagnostic_configs"] += 1
            stats["diagnostic_evaluations_expected"] += len(expected_diagnostics)
            present = expected_diagnostics & set(metric_by_key)
            stats["diagnostic_evaluations_present"] += len(present)
            missing_diagnostics = sorted(expected_diagnostics - set(metric_by_key))
            if missing_diagnostics:
                errors.append(
                    f"{label}: missing {len(missing_diagnostics)} diagnostic evaluations"
                )
            for name, stat_name in (
                ("temporal_influence.json", "influence_manifests"),
            ):
                path = directory / name
                if not path.is_file():
                    errors.append(f"{label}: missing {name}")
                    continue
                try:
                    _payload, valid = hashed_json(path)
                except Exception as error:
                    errors.append(f"{label}: invalid {name}: {error}")
                    continue
                if not valid:
                    errors.append(f"{label}: {name} hash mismatch")
                stats[stat_name] += 1

        benchmark_required = args.require_diagnostics and config.condition.name == "clean" and (
            (
                config.experiment_id == "temporal_native"
                and config.dataset in DIAGNOSTIC_DATASETS
                and config.model in DIAGNOSTIC_MODELS
            )
            or (
                config.experiment_id == "hardvs_main"
                and config.dataset == "hardvs"
                and config.model in COMPARATOR_ONLY_MODELS
                and config.time_steps == 16
            )
        )
        if benchmark_required:
            stats["benchmark_manifests_expected"] += 1
            path = directory / "benchmark.json"
            if not path.is_file():
                errors.append(f"{label}: missing benchmark.json")
            else:
                try:
                    _payload, valid = hashed_json(path)
                except Exception as error:
                    errors.append(f"{label}: invalid benchmark.json: {error}")
                else:
                    if not valid:
                        errors.append(f"{label}: benchmark.json hash mismatch")
                    stats["benchmark_manifests"] += 1

    report = {
        "schema_version": 1,
        "project": str(PROJECT),
        "config_root": str(args.configs_root),
        "required_diagnostic_grids": grids if args.require_diagnostics else [],
        "artifact_hashes_checked": not args.skip_artifact_hashes,
        "stats": stats,
        "hardvs_sources": sorted(
            hardvs_sources.values(), key=lambda row: row["split_hash"]
        ),
        "errors": errors,
        "warnings": warnings,
        "status": "complete" if not errors else "incomplete",
    }
    report["manifest_hash"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "stats": stats,
                "errors": len(errors),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if errors:
        for error in errors[:50]:
            print(f"[audit error] {error}", file=sys.stderr)
        if len(errors) > 50:
            print(
                f"[audit error] ... {len(errors) - 50} additional errors",
                file=sys.stderr,
            )
    return 0 if not errors or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
