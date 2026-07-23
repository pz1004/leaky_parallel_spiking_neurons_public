from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataclasses import replace

from experiments.config import ConditionConfig, ExperimentConfig
from experiments.data import build_dataset, load_split_manifest, prepare_split_manifest


class ProtocolTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
