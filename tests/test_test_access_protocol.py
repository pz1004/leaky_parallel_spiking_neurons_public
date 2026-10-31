"""Regression fixtures for checkpoint identity and irreversible test-read records.

These tests use temporary records and mocked dataset/model boundaries. Merely
importing this module does not train a model or open an experimental dataset.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import yaml
from argparse import Namespace
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from experiments.cli import (
    _assert_training_allowed,
    _evaluation_key,
    _load_checkpoint,
    _reservation_path,
    _reserve_test_read,
    _run_protocol_lock,
    _validate_completed_training,
    command_analyze,
    command_train,
    evaluate_checkpoint,
)
from experiments.config import ConditionConfig, ExperimentConfig, file_sha256


def configuration(root: Path, experiment_id: str = "ordinary_run") -> ExperimentConfig:
    return ExperimentConfig.from_mapping({
        "schema_version": 1, "experiment_id": experiment_id,
        "dataset": "synthetic", "model": "psn", "seed": 42,
        "time_steps": 8, "data_root": str(root),
        "split_manifest": str(root / "unused_split.json"),
        "output_root": str(root / "runs"), "device": "cpu",
    })


def completed_checkpoint(config: ExperimentConfig):
    config.run_dir.mkdir(parents=True, exist_ok=True)
    selected = config.run_dir / "checkpoint_best.pt"
    selected.write_bytes(b"mock checkpoint; never deserialized")
    checkpoint = {
        "config": asdict(config), "config_hash": config.config_hash,
        "selection_split": "validation", "split_hash": "fixture-split",
    }
    (config.run_dir / "train_result.json").write_text(json.dumps({
        "config_hash": config.config_hash,
        "checkpoint_sha256": file_sha256(selected),
    }), encoding="utf-8")
    return checkpoint, selected


class TestAccessProtocolTests(unittest.TestCase):
    def test_interruption_reserves_condition_before_dataset_open_for_every_prefix(self):
        for prefix in ("smoke", "temporal_native", "component_study"):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as directory:
                config = configuration(Path(directory), prefix)
                checkpoint, selected = completed_checkpoint(config)

                def stop_at_dataset(*_args, **_kwargs):
                    self.assertTrue(_reservation_path(config, _evaluation_key(config)).is_file())
                    raise RuntimeError("simulated interruption at test open")

                with patch("experiments.cli._load_checkpoint", return_value=(object(), checkpoint, selected)), \
                     patch("experiments.cli.build_dataset", side_effect=stop_at_dataset) as dataset:
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        evaluate_checkpoint(config)
                    with self.assertRaisesRegex(RuntimeError, "refusing another test read"):
                        evaluate_checkpoint(config)
                    self.assertEqual(dataset.call_count, 1)

    def test_distinct_condition_draws_reserve_independently_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = configuration(Path(directory))
            corrupt = replace(clean, condition=ConditionConfig("event_dropout", 0.3, draw=2))
            with _run_protocol_lock(clean.run_dir):
                first = _reserve_test_read(clean, _evaluation_key(clean), "frozen-checkpoint")
                second = _reserve_test_read(corrupt, _evaluation_key(corrupt), "frozen-checkpoint")
                self.assertNotEqual(first, second)
                with self.assertRaisesRegex(RuntimeError, "refusing another test read"):
                    _reserve_test_read(corrupt, _evaluation_key(corrupt), "frozen-checkpoint")
            with self.assertRaisesRegex(RuntimeError, "Refusing to train"):
                _assert_training_allowed(clean.run_dir)

    def test_named_renormalization_preserves_historical_ledger_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            base = configuration(Path(directory))
            scaled = replace(base, condition=ConditionConfig("event_thin", 0.5))
            normalized = replace(base, condition=ConditionConfig("event_thin_renorm", 0.5, renormalize=True))
            self.assertEqual(_evaluation_key(normalized), "event_thin_renorm|0.5|0")
            self.assertNotEqual(_evaluation_key(scaled), _evaluation_key(normalized))

    def test_started_test_blocks_training_before_seeding_or_dataset_access(self):
        with tempfile.TemporaryDirectory() as directory:
            config = configuration(Path(directory))
            with _run_protocol_lock(config.run_dir):
                _reserve_test_read(config, _evaluation_key(config), "frozen-checkpoint")
            with patch("experiments.cli.load_config", return_value=config), \
                 patch("experiments.cli.seed_everything") as seed, \
                 patch("experiments.cli.build_dataset") as dataset:
                with self.assertRaisesRegex(RuntimeError, "Refusing to train"):
                    command_train(Namespace(config="unused.yml"))
                seed.assert_not_called()
                dataset.assert_not_called()

    def test_checkpoint_metadata_mismatch_is_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            config = configuration(Path(directory))
            _, selected = completed_checkpoint(config)
            cases = {"seed": 123, "architecture": "spiking_resnet18", "model_params": {"tau": 3.0}}
            for field, value in cases.items():
                with self.subTest(field=field):
                    saved = asdict(config)
                    saved[field] = value
                    with patch("experiments.cli.torch.load", return_value={"config": saved}), \
                         patch("experiments.cli._model") as model:
                        with self.assertRaisesRegex(ValueError, field):
                            _load_checkpoint(config, selected)
                        model.assert_not_called()

    def test_frozen_training_identity_allows_only_evaluation_condition_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            training = configuration(Path(directory))
            checkpoint, selected = completed_checkpoint(training)
            evaluation = replace(training, condition=ConditionConfig("event_dropout", 0.3, draw=1))
            self.assertNotEqual(training.config_hash, evaluation.config_hash)
            self.assertEqual(
                _validate_completed_training(evaluation, checkpoint, selected), file_sha256(selected)
            )
            selected.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(RuntimeError, "changed after training completion"):
                _validate_completed_training(evaluation, checkpoint, selected)

    def test_legacy_interruption_marker_remains_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = configuration(Path(directory))
            config.run_dir.mkdir(parents=True)
            (config.run_dir / "test_evaluation_started.json").write_text("{}", encoding="utf-8")
            with patch("experiments.cli._load_checkpoint") as load:
                with self.assertRaisesRegex(RuntimeError, "refusing another test read"):
                    evaluate_checkpoint(config)
                load.assert_not_called()

    def test_declared_identity_alias_keeps_frozen_source_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = configuration(Path(directory), "temporal_native")
            checkpoint, selected = completed_checkpoint(source)
            target = replace(source, experiment_id="density_sparsity_event_thin_renorm_1",
                             condition=ConditionConfig("event_thin_renorm", 1.0, renormalize=True))
            target.run_dir.mkdir(parents=True)
            target_checkpoint = target.run_dir / "checkpoint_best.pt"
            target_checkpoint.write_bytes(selected.read_bytes())
            proof = {
                "kind": "deterministic_identity_reuse", "independent_replication": False,
                "source_config": "canonical.yml", "target_config": "alias.yml",
                "source_config_hash": source.config_hash, "target_config_hash": target.config_hash,
            }
            completed = {
                "config_hash": target.config_hash, "checkpoint_sha256": file_sha256(target_checkpoint),
                "equivalent_training_source": "canonical.yml", "independent_replication": False,
            }
            (target.run_dir / "config.yml").write_text(yaml.safe_dump(asdict(target)), encoding="utf-8")
            (target.run_dir / "equivalence_provenance.json").write_text(json.dumps(proof), encoding="utf-8")
            (target.run_dir / "train_result.json").write_text(json.dumps(completed), encoding="utf-8")
            self.assertEqual(_validate_completed_training(target, checkpoint, target_checkpoint),
                             file_sha256(target_checkpoint))
            # An apparent proof cannot authorize a nonidentity transform, even
            # when its target hashes are internally consistent.
            nonidentity = replace(target, condition=replace(target.condition, severity=0.5))
            proof["target_config_hash"] = nonidentity.config_hash
            completed["config_hash"] = nonidentity.config_hash
            (target.run_dir / "config.yml").write_text(yaml.safe_dump(asdict(nonidentity)), encoding="utf-8")
            (target.run_dir / "equivalence_provenance.json").write_text(json.dumps(proof), encoding="utf-8")
            (target.run_dir / "train_result.json").write_text(json.dumps(completed), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "declared identity condition"):
                _validate_completed_training(nonidentity, checkpoint, target_checkpoint)

    def test_missing_declared_evidence_fails_before_writing_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "locked_test_metrics.jsonl").write_text(json.dumps({
                "per_sample_path": str(root / "missing_predictions.jsonl"),
            }) + "\n", encoding="utf-8")
            output = root / "summary.csv"
            with self.assertRaisesRegex(FileNotFoundError, "Missing sealed per-example evidence"):
                command_analyze(Namespace(runs_root=root, output=output))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
