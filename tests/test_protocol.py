from __future__ import annotations

import tempfile
import unittest
import csv
import json
from argparse import Namespace
from pathlib import Path

from dataclasses import replace
import yaml

from experiments.config import ConditionConfig, ExperimentConfig
from experiments.data import build_dataset, load_split_manifest, prepare_split_manifest
from experiments.cli import _exact_signed_rank, command_analyze, command_cache


class ProtocolTests(unittest.TestCase):
    def test_retained_matrix_preserves_only_hardvs_t16(self) -> None:
        project = Path(__file__).resolve().parents[1]
        spec = yaml.safe_load((project / "configs/studies.yml").read_text())
        declarations = 0
        for study in spec["studies"].values():
            for dataset in study["datasets"]:
                lengths = study.get("time_steps_by_dataset", {}).get(dataset, study["time_steps"])
                if dataset == "hardvs":
                    self.assertEqual(lengths, [16])
                    self.assertNotIn("conditions", study)
                declarations += len(lengths) * len(study["models"]) * len(study["seeds"]) * len(study.get("conditions", [None]))
        self.assertEqual(declarations, 410)
        historical = yaml.safe_load((project / "configs/historical/studies_695.yml").read_text())
        historical_count = sum(len(study["datasets"]) * len(study["time_steps"]) * len(study["models"]) * len(study["seeds"]) * len(study.get("conditions", [None])) for study in historical["studies"].values())
        self.assertEqual(historical_count, 695)

    def test_scheduler_refuses_stale_hardvs_configs_without_deleting_them(self) -> None:
        from scripts.run_study_matrix import load_declared_configs
        project = Path(__file__).resolve().parents[1]
        spec = yaml.safe_load((project / "configs/studies.yml").read_text())
        raw = {**spec["defaults"], **spec["dataset_overrides"]["hardvs"],
               "schema_version": 1, "experiment_id": "temporal_native", "dataset": "hardvs",
               "model": "psn", "time_steps": 32, "seed": 42, "data_root": "../data",
               "split_manifest": "manifests/hardvs.json", "frame_cache_root": "../data/.lpsn_frame_cache",
               "condition": {"name": "clean", "severity": 0.0, "draw": 0, "renormalize": False}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retired.yml"
            path.write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(RuntimeError, "undeclared/stale"):
                load_declared_configs(Path(directory), project / "configs/studies.yml")
            self.assertTrue(path.exists())
            selected = load_declared_configs(Path(directory), project / "configs/historical/studies_695.yml")
            self.assertEqual(len(selected), 1)
            (Path(directory) / "duplicate.yml").write_text(path.read_text())
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                load_declared_configs(Path(directory), project / "configs/historical/studies_695.yml")

    def test_extension_templates_are_portable_and_complete(self) -> None:
        project = Path(__file__).resolve().parents[1]
        counts = {"architecture": 0, "components": 0}
        runs = set()
        for path in (project / "configs/extension").rglob("*.yml"):
            raw = yaml.safe_load(path.read_text())
            config = ExperimentConfig.from_mapping(raw)
            counts[path.relative_to(project / "configs/extension").parts[0]] += 1
            self.assertNotIn(config.run_dir, runs)
            runs.add(config.run_dir)
            for field in ("data_root", "split_manifest", "output_root", "frame_cache_root"):
                self.assertFalse(Path(raw[field]).is_absolute(), f"Non-portable {field}: {path}")
        self.assertEqual(counts, {"architecture": 60, "components": 50})

    def test_cache_completion_uses_smoke_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "split.json"
            prepare_split_manifest("synthetic", root, manifest_path)
            config_path = root / "smoke_cache.yml"
            values = {
                "schema_version": 1,
                "experiment_id": "smoke_cache",
                "dataset": "synthetic",
                "model": "psn",
                "seed": 42,
                "time_steps": 8,
                "data_root": str(root),
                "split_manifest": str(manifest_path),
                "output_root": str(root / "runs"),
                "image_size": 32,
                "batch_size": 1,
                "effective_batch_size": 1,
                "num_workers": 0,
                "smoke_samples": 3,
                "cache_frames": True,
                "frame_cache_root": str(root / "cache"),
            }
            config_path.write_text(
                yaml.safe_dump(values, sort_keys=True),
                encoding="utf-8",
            )
            result = command_cache(
                Namespace(
                    config=config_path,
                    condition=None,
                    severity=None,
                    draw=0,
                    split=["train"],
                    verify=False,
                )
            )
            self.assertEqual(result, 0)
            markers = list((root / "cache").rglob(".complete_*.json"))
            self.assertEqual(len(markers), 1)
            marker = json.loads(markers[0].read_text(encoding="utf-8"))
            self.assertEqual(marker["samples"], {"train": 3})
            self.assertEqual(marker["cached_samples"], 3)

    def test_exact_signed_rank_uses_seed_level_granularity(self) -> None:
        result = _exact_signed_rank([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(result["wilcoxon_nonzero_seed_means"], 5)
        self.assertEqual(result["wilcoxon_w_plus"], 15.0)
        self.assertEqual(result["wilcoxon_p_greater_raw"], 1 / 32)
        self.assertEqual(result["wilcoxon_p_two_sided_raw"], 1 / 16)

    def test_synthetic_manifest_is_disjoint_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            prepare_split_manifest("synthetic", Path(directory), path)
            manifest = load_split_manifest(path)
            identifiers = [{row["id"] for row in manifest["splits"][name]} for name in ("train", "val", "test")]
            self.assertFalse(identifiers[0] & identifiers[1])
            self.assertFalse(identifiers[0] & identifiers[2])
            self.assertFalse(identifiers[1] & identifiers[2])

    def test_config_rejects_non_native_length(self) -> None:
        values = {
            "schema_version": 1,
            "experiment_id": "bad",
            "dataset": "synthetic",
            "model": "psn",
            "seed": 1,
            "time_steps": 12,
            "data_root": ".",
            "split_manifest": "split.json",
        }
        with self.assertRaises(ValueError):
            ExperimentConfig.from_mapping(values)

    def test_condition_draw_does_not_change_source_clip_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            prepare_split_manifest("synthetic", Path(directory), path)
            config = ExperimentConfig.from_mapping(
                {
                    "schema_version": 1,
                    "experiment_id": "source_invariance",
                    "dataset": "synthetic",
                    "model": "psn",
                    "seed": 42,
                    "time_steps": 8,
                    "data_root": directory,
                    "split_manifest": str(path),
                    "image_size": 32,
                    "batch_size": 1,
                    "effective_batch_size": 1,
                }
            )
            clean, _ = build_dataset(config, "test")
            corrupt, _ = build_dataset(
                replace(config, condition=ConditionConfig("event_dropout", 0.5, draw=9)), "test"
            )
            self.assertEqual(clean[0][3]["duration"], corrupt[0][3]["duration"])

    def test_analysis_uses_declared_temporal_native_clean_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "dataset": "cifar10_dvs",
                "model": "psn",
                "seed": 42,
                "time_steps": 32,
                "draw": 0,
                "firing_rate": 0.1,
                "throughput": 1.0,
                "peak_cuda_memory_mb": 1.0,
                "mean_event_count": 1.0,
                "mean_active_bin_fraction": 0.1,
                "mean_event_mass": 1.0,
                "mean_duration": 1.0,
                "macro_f1": 0.5,
            }
            rows = [
                {
                    **common,
                    "experiment_id": "temporal_native",
                    "condition": "clean",
                    "severity": 0.0,
                    "accuracy": 0.8,
                },
                {
                    **common,
                    "experiment_id": "density_sparsity",
                    "condition": "event_thin",
                    "severity": 0.5,
                    "accuracy": 0.6,
                },
            ]
            for index, row in enumerate(rows):
                path = root / f"run_{index}" / "locked_test_metrics.jsonl"
                path.parent.mkdir()
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                training = {
                    "experiment_id": "shared_training",
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "time_steps": row["time_steps"],
                    "condition": {
                        "name": row["condition"],
                        "severity": row["severity"],
                        "draw": row["draw"],
                    },
                    "parameter_count": 10,
                    "trainable_parameter_count": 10,
                    "best_validation_value": row["accuracy"],
                    "best_validation_accuracy": row["accuracy"],
                    "best_epoch": 1,
                    "executed_epochs": 2,
                }
                (path.parent / "train_result.json").write_text(
                    json.dumps(training) + "\n", encoding="utf-8"
                )
            output = root / "aggregates" / "results.csv"
            command_analyze(Namespace(runs_root=root, output=output))
            with output.open(encoding="utf-8", newline="") as handle:
                results = list(csv.DictReader(handle))
            density = next(
                row for row in results if row["experiment_id"] == "density_sparsity"
            )
            self.assertEqual(
                density["clean_reference_type"],
                "temporal_native_matched_fallback",
            )
            self.assertEqual(
                density["clean_reference_experiment_id"], "temporal_native"
            )
            self.assertAlmostEqual(float(density["absolute_accuracy_drop"]), 0.2)
            with output.with_name("results_training.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                training_rows = list(csv.DictReader(handle))
            self.assertEqual(len(training_rows), 2)
            self.assertEqual(
                {row["condition"] for row in training_rows},
                {"clean", "event_thin"},
            )


if __name__ == "__main__":
    unittest.main()
