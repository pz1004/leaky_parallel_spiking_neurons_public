#!/usr/bin/env python3
"""Transparently add locked-test provenance manifests to legacy completed runs.

This does not rerun or alter predictions. It hashes the already-sealed
checkpoint, per-sample rows, and confusion matrix, and derives the historical
git state from the matching clean training/runtime records. The resulting
manifest is explicitly marked as a provenance backfill rather than native
evaluation-time capture.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.config import file_sha256, stable_hash  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def portable_path(path: Path) -> str:
    resolved = project_path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT))
    except ValueError:
        return str(resolved)


def evaluation_key(row: dict[str, Any]) -> str:
    return f"{row['condition']}|{float(row['severity']):.12g}|{int(row['draw'])}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.backfill.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def backfill_run(metrics_path: Path, *, dry_run: bool) -> list[dict[str, str]]:
    run_dir = metrics_path.parent
    ledger_path = run_dir / "test_evaluation_ledger.jsonl"
    run_manifest_path = run_dir / "run_manifest.json"
    environment_path = run_dir / "environment.json"
    checkpoint_path = run_dir / "checkpoint_best.pt"
    required = (
        ledger_path,
        run_manifest_path,
        environment_path,
        checkpoint_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot backfill {run_dir}; missing {missing}")

    metrics = read_jsonl(metrics_path)
    ledger = read_jsonl(ledger_path)
    by_key = {str(row["evaluation_key"]): row for row in ledger}
    if len(by_key) != len(ledger):
        raise RuntimeError(f"Duplicate ledger keys in {ledger_path}")
    training_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if (
        training_manifest.get("git_commit") != environment.get("git_commit")
        or bool(training_manifest.get("git_dirty")) != bool(environment.get("git_dirty"))
    ):
        raise RuntimeError(
            f"Training manifest/environment git state mismatch in {run_dir}"
        )

    changed: list[dict[str, str]] = []
    checkpoint_sha256 = file_sha256(checkpoint_path)
    for row in metrics:
        key = evaluation_key(row)
        ledger_row = by_key.get(key)
        if ledger_row is None:
            raise RuntimeError(f"Metric row has no ledger entry: {run_dir} {key}")
        output_hash = stable_hash(key)[:12]
        manifest_path = run_dir / "evaluation_manifests" / f"{output_hash}.json"
        if manifest_path.is_file():
            continue
        sample_path = project_path(row["per_sample_path"])
        confusion_path = run_dir / f"confusion_{output_hash}.npy"
        if not sample_path.is_file() or not confusion_path.is_file():
            raise FileNotFoundError(
                f"Missing sealed evaluation artifact for {run_dir} {key}"
            )
        row["checkpoint"] = portable_path(checkpoint_path)
        row["checkpoint_sha256"] = checkpoint_sha256
        row["evaluation_manifest_path"] = portable_path(manifest_path)
        manifest = {
            "schema_version": 1,
            "created_at": row["evaluated_at"],
            "phase": "locked_test_evaluation_provenance_backfill",
            "experiment_id": row["experiment_id"],
            "config_hash": row["config_hash"],
            "evaluation_config_hash": row["config_hash"],
            "training_config_hash": row["training_config_hash"],
            "split_hash": row["split_hash"],
            "git_commit": training_manifest["git_commit"],
            "git_dirty": training_manifest["git_dirty"],
            "python": training_manifest.get("python", environment.get("python")),
            "torch": training_manifest.get("torch", environment.get("torch")),
            "cuda": training_manifest.get("cuda", environment.get("cuda")),
            "device": training_manifest.get("device"),
            "dataset": row["dataset"],
            "model": row["model"],
            "seed": row["seed"],
            "time_steps": row["time_steps"],
            "condition": {
                "name": row["condition"],
                "severity": row["severity"],
                "draw": row["draw"],
            },
            "evaluation_key": key,
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": row["checkpoint_epoch"],
            "selection_split": row["selection_split"],
            "selection_metric": row["selection_metric"],
            "selection_value": row["selection_value"],
            "test_evaluated_at": row["evaluated_at"],
            "provenance_backfill": {
                "backfilled_at": now(),
                "predictions_recomputed": False,
                "historical_git_state_basis": (
                    "The training run manifest and environment record agree. "
                    "The matrix scheduler invoked locked-test evaluation in the "
                    "same clean checkout immediately after training."
                ),
                "training_run_manifest_path": portable_path(run_manifest_path),
                "training_run_manifest_sha256": file_sha256(run_manifest_path),
                "environment_path": portable_path(environment_path),
                "environment_sha256": file_sha256(environment_path),
            },
            "artifacts": {
                "per_sample_path": portable_path(sample_path),
                "per_sample_sha256": file_sha256(sample_path),
                "confusion_path": portable_path(confusion_path),
                "confusion_sha256": file_sha256(confusion_path),
                "metrics_payload_sha256": stable_hash(row),
            },
        }
        manifest["manifest_hash"] = stable_hash(manifest)
        ledger_row.update(
            {
                "checkpoint_sha256": checkpoint_sha256,
                "evaluation_manifest_path": portable_path(manifest_path),
                "evaluation_manifest_hash": manifest["manifest_hash"],
                "provenance_backfilled": True,
            }
        )
        changed.append({"run_dir": portable_path(run_dir), "evaluation_key": key})
        if not dry_run:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    if changed and not dry_run:
        write_jsonl(metrics_path, metrics)
        write_jsonl(ledger_path, ledger)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root", type=Path, default=PROJECT / "runs_baseline"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    changed: list[dict[str, str]] = []
    for path in sorted(args.runs_root.rglob("locked_test_metrics.jsonl")):
        changed.extend(backfill_run(path, dry_run=args.dry_run))
    report = {
        "schema_version": 1,
        "created_at": now(),
        "dry_run": args.dry_run,
        "backfilled_evaluations": len(changed),
        "rows": changed,
    }
    report["manifest_hash"] = stable_hash(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
