from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np

from experiments.config import ConditionConfig, ExperimentConfig
from experiments.data import build_dataset, corrupt_events, events_to_frames, prepare_split_manifest


def fixture_events(count: int = 100):
    return {
        "x": np.arange(count) % 16,
        "y": np.arange(count) % 12,
        "p": np.arange(count) % 2,
        "t": np.arange(count, dtype=np.float64),
    }


class CorruptionTests(unittest.TestCase):
    def test_clean_frame_integration_conserves_event_mass(self) -> None:
        events = fixture_events()
        frames = events_to_frames(events, 8, 12, 16)
        self.assertEqual(tuple(frames.shape), (8, 2, 12, 16))
        self.assertEqual(float(frames.sum()), 100.0)

    def test_corruptions_are_deterministic_and_bounded(self) -> None:
        conditions = [
            ConditionConfig("event_dropout", 0.3),
            ConditionConfig("background_activity", 0.3),
            ConditionConfig("polarity_flip", 0.3),
            ConditionConfig("temporal_jitter", 0.02),
            ConditionConfig("packet_loss", 0.25),
            ConditionConfig("hot_pixels", 0.3),
        ]
        for condition in conditions:
            first, _ = corrupt_events(fixture_events(), condition, np.random.default_rng(7), 12, 16)
            second, _ = corrupt_events(fixture_events(), condition, np.random.default_rng(7), 12, 16)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            self.assertTrue(np.all((first["x"] >= 0) & (first["x"] < 16)))
            self.assertTrue(np.all((first["y"] >= 0) & (first["y"] < 12)))
            self.assertTrue(np.all((first["p"] == 0) | (first["p"] == 1)))

    def test_thinning_renormalization_restores_expected_scale(self) -> None:
        thinned, scale = corrupt_events(
            fixture_events(10000), ConditionConfig("event_thin_renorm", 0.25), np.random.default_rng(4), 12, 16
        )
        self.assertEqual(scale, 4.0)
        self.assertLess(abs(len(thinned["t"]) * scale - 10000), 500)

    def test_original_time_bounds_prevent_post_corruption_rescaling(self) -> None:
        events = fixture_events(100)
        retained = {key: value[50:] for key, value in events.items()}
        fixed = events_to_frames(retained, 8, 12, 16, time_bounds=(0.0, 99.0))
        rescaled = events_to_frames(retained, 8, 12, 16)
        self.assertEqual(float(fixed[:4].sum()), 0.0)
        self.assertGreater(float(rescaled[0].sum()), 0.0)

    def test_renormalized_thinning_uses_identical_retained_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "split.json"
            prepare_split_manifest("synthetic", root, manifest_path)
            config = ExperimentConfig.from_mapping(
                {
                    "schema_version": 1,
                    "experiment_id": "matched_thinning",
                    "dataset": "synthetic",
                    "model": "psn",
                    "seed": 42,
                    "time_steps": 8,
                    "data_root": str(root),
                    "split_manifest": str(manifest_path),
                    "image_size": 32,
                    "batch_size": 1,
                    "effective_batch_size": 1,
                    "cache_frames": True,
                    "frame_cache_root": str(root / "cache"),
                }
            )
            plain, _ = build_dataset(
                replace(config, condition=ConditionConfig("event_thin", 0.25)), "test"
            )
            renormalized, _ = build_dataset(
                replace(config, condition=ConditionConfig("event_thin_renorm", 0.25)), "test"
            )
            plain_frames, _label, _identifier, plain_stats = plain[0]
            renorm_frames, _label, _identifier, renorm_stats = renormalized[0]
            self.assertEqual(plain_stats["event_count"], renorm_stats["event_count"])
            np.testing.assert_allclose(renorm_frames.numpy(), plain_frames.numpy() * 4.0)
            self.assertEqual(len(list((root / "cache").rglob("*.npz"))), 1)

    def test_sensor_corruption_cache_is_shared_across_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "split.json"
            prepare_split_manifest("synthetic", root, manifest_path)
            values = {
                "schema_version": 1,
                "experiment_id": "sensor_cache",
                "dataset": "synthetic",
                "model": "psn",
                "seed": 42,
                "time_steps": 8,
                "data_root": str(root),
                "split_manifest": str(manifest_path),
                "image_size": 32,
                "batch_size": 1,
                "effective_batch_size": 1,
                "cache_frames": True,
                "frame_cache_root": str(root / "cache"),
                "condition": {
                    "name": "event_dropout",
                    "severity": 0.3,
                    "draw": 4,
                },
            }
            psn = ExperimentConfig.from_mapping(values)
            lpsn = replace(psn, model="lpsn_a1_lif_init", seed=1024)
            psn_dataset, _ = build_dataset(psn, "test")
            lpsn_dataset, _ = build_dataset(lpsn, "test")
            first = psn_dataset[0][0]
            second = lpsn_dataset[0][0]
            np.testing.assert_array_equal(first.numpy(), second.numpy())
            self.assertEqual(len(list((root / "cache").rglob("*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()
