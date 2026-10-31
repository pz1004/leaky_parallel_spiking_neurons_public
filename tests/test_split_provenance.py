"""Synthetic metadata fixtures for explicitly permitted preparation provenance."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.config import stable_hash
from experiments.data import load_split_manifest, prepare_split_manifest, validate_split_manifest


class SplitProvenanceTests(unittest.TestCase):
    @staticmethod
    def fixture():
        splits = {
            split: [{"id": f"{split}/{label}", "path": None, "label": label} for label in range(300)]
            for split in ("train", "val", "test")
        }
        record = {
            "integrity_mode": "official_exact", "source_deviation": None,
            "archive": {"md5": "a" * 32, "sha256": "b" * 64, "acquisition_record_hash": "c" * 64},
            "preparation_runtime": {
                "git_commit": "d" * 40, "git_dirty": True,
                "script_sha256": "e" * 64,
                "argv": ["prepare_hardvs.py", "--allow-dirty-code"],
            },
            "splits": {
                "effective_counts": {"train": 300, "val": 300, "test": 300, "assigned": 900},
                "excluded_sample_ids": [],
            },
        }
        record["record_hash"] = stable_hash(record)
        return splits, record

    def prepare_fixture(self, root, splits, record):
        hardvs = root / "HARDVS"
        hardvs.mkdir(exist_ok=True)
        (hardvs / "integrity_record.json").write_text(json.dumps(record), encoding="utf-8")
        with patch("experiments.data._scan_hardvs", return_value=splits):
            return prepare_split_manifest("hardvs", root, root / "split.json")

    def test_explicit_dirty_opt_in_survives_hash_protected_split_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits, record = self.fixture()
            manifest = self.prepare_fixture(root, splits, record)
            integrity = manifest["dataset_integrity"]
            self.assertIs(integrity["preparation_git_dirty"], True)
            self.assertIs(integrity["preparation_dirty_code_allowed"], True)
            self.assertEqual(load_split_manifest(root / "split.json"), manifest)

            # Even an explicitly authorized dirty preparation cannot bypass a
            # modified archive digest in the frozen split record.
            manifest["dataset_integrity"]["archive_sha256"] = "0" * 64
            (root / "split.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Split-manifest hash mismatch"):
                load_split_manifest(root / "split.json")

    def test_dirty_preparation_without_recorded_opt_in_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            splits, record = self.fixture()
            record["preparation_runtime"]["argv"] = ["prepare_hardvs.py"]
            record.pop("record_hash")
            record["record_hash"] = stable_hash(record)
            with self.assertRaisesRegex(ValueError, "recorded --allow-dirty-code"):
                self.prepare_fixture(Path(directory), splits, record)

    def test_dirty_opt_in_does_not_relax_record_counts_or_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits, record = self.fixture()
            manifest = self.prepare_fixture(root, splits, record)
            bad_counts = copy.deepcopy(manifest)
            bad_counts["dataset_integrity"]["effective_counts"]["train"] -= 1
            with self.assertRaisesRegex(ValueError, "counts disagree"):
                validate_split_manifest(bad_counts)
            bad_exclusions = copy.deepcopy(manifest)
            bad_exclusions["dataset_integrity"]["integrity_mode"] = "known_mirror_exclusion"
            with self.assertRaisesRegex(ValueError, "13 exclusions"):
                validate_split_manifest(bad_exclusions)
            record["archive"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "integrity-record hash mismatch"):
                self.prepare_fixture(root, splits, record)

    def test_clean_preparation_keeps_historical_provenance_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            splits, record = self.fixture()
            record["preparation_runtime"]["git_dirty"] = False
            record["preparation_runtime"]["argv"] = ["prepare_hardvs.py"]
            record.pop("record_hash")
            record["record_hash"] = stable_hash(record)
            manifest = self.prepare_fixture(Path(directory), splits, record)
            self.assertNotIn("preparation_dirty_code_allowed", manifest["dataset_integrity"])


if __name__ == "__main__":
    unittest.main()
