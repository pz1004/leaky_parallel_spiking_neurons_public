from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from experiments.config import ExperimentConfig, stable_hash
from experiments.data import prepare_split_manifest
from scripts import aggregate_diagnostics


class DiagnosticAggregationTests(unittest.TestCase):
    def test_influence_is_collected_for_every_configuration(self) -> None:
        project = Path(aggregate_diagnostics.PROJECT)
        with tempfile.TemporaryDirectory(dir=project) as directory:
            root = Path(directory)
            configs_root = root / "configs"
            output_root = root / "aggregates"
            split_path = root / "split.json"
            prepare_split_manifest("synthetic", root, split_path)
            configs_root.mkdir()

            for model in ("psn", "lpsn_a1_lif_init"):
                values = {
                    "schema_version": 1,
                    "experiment_id": "diagnostic_fixture",
                    "dataset": "synthetic",
                    "model": model,
                    "seed": 42,
                    "time_steps": 8,
                    "data_root": str(root),
                    "split_manifest": str(split_path),
                    "output_root": str(root / "runs"),
                    "image_size": 32,
                    "batch_size": 1,
                    "effective_batch_size": 1,
                }
                config = ExperimentConfig.from_mapping(values)
                (configs_root / f"{model}.yml").write_text(
                    yaml.safe_dump(values, sort_keys=True),
                    encoding="utf-8",
                )
                influence = {
                    "normalized_absolute_input_gradient": [0.125] * 8,
                    "effective_horizon_90_percent": 8,
                    "samples": 4,
                }
                influence["manifest_hash"] = stable_hash(influence)
                path = config.run_dir / "temporal_influence.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(influence, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            with patch.object(
                sys,
                "argv",
                [
                    "aggregate_diagnostics.py",
                    "--configs-root",
                    str(configs_root),
                    "--output-root",
                    str(output_root),
                ],
            ):
                self.assertEqual(aggregate_diagnostics.main(), 0)

            manifest = json.loads(
                (output_root / "diagnostics.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["influence_timestep_rows"], 16)
            self.assertEqual(manifest["horizon_rows"], 2)
            self.assertEqual(
                len(
                    [
                        row
                        for row in manifest["sources"]
                        if row["path"].endswith("temporal_influence.json")
                    ]
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
