from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import prepare_hardvs


class HardvsIntegrityTests(unittest.TestCase):
    def _archive_fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        member = "MINIHARDVS_EVENT_files/action_001/sample/sample.npz"
        archive = root / "fixture.zip"
        payload = b"synthetic-npz-payload"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr(member, payload)
        hashes = prepare_hardvs.file_digests(archive, ("md5", "sha256"))
        manifest: dict[str, object] = {
            "archive_name": archive.name,
            "archive_size": archive.stat().st_size,
            "archive_entries": 1,
            "npz_entries": 1,
            "uncompressed_bytes": len(payload),
            "official_archive_md5": hashes["md5"],
            "known_mirror_archive_md5": "0" * 32,
            "known_mirror_archive_sha256": "0" * 64,
            "known_mirror_corrupt_samples": [],
        }
        record = {
            "schema_version": 2,
            "integrity_mode": "official_exact",
            "archive_md5": hashes["md5"],
            "archive_sha256": hashes["sha256"],
            "archive_entries": 1,
            "npz_entries": 1,
            "uncompressed_bytes": len(payload),
            "full_crc_members": 1,
            "full_crc_failures": [],
            "manifest_sha256": "fixture-manifest",
        }
        record["record_hash"] = prepare_hardvs.stable_hash(record)
        (root / "acquisition_record.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        return archive, manifest

    def test_exact_archive_requires_matching_full_crc_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self._archive_fixture(root)
            result = prepare_hardvs.validate_archive(
                manifest,
                archive,
                acquisition_record_path=root / "acquisition_record.json",
                allow_known_corrupt_mirror=False,
                manifest_sha256="fixture-manifest",
            )
            self.assertEqual(result["integrity_mode"], "official_exact")
            self.assertEqual(result["excluded_members"], [])

            record_path = root / "acquisition_record.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["full_crc_members"] = 0
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                prepare_hardvs.validate_archive(
                    manifest,
                    archive,
                    acquisition_record_path=record_path,
                    allow_known_corrupt_mirror=False,
                    manifest_sha256="fixture-manifest",
                )

    def test_known_mirror_is_rejected_without_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self._archive_fixture(root)
            hashes = prepare_hardvs.file_digests(archive, ("md5", "sha256"))
            manifest["official_archive_md5"] = "f" * 32
            manifest["known_mirror_archive_md5"] = hashes["md5"]
            manifest["known_mirror_archive_sha256"] = hashes["sha256"]
            with self.assertRaisesRegex(RuntimeError, "--allow-known-corrupt-mirror"):
                prepare_hardvs.validate_archive(
                    manifest,
                    archive,
                    acquisition_record_path=root / "acquisition_record.json",
                    allow_known_corrupt_mirror=False,
                    manifest_sha256="fixture-manifest",
                )

    def test_split_materialization_excludes_only_declared_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels"
            extracted = root / "extracted"
            raw = root / "raw"
            labels.mkdir()
            rows = {
                "train": "action_001/train_sample 1 0\n",
                "val": "action_001/val_sample 1 0\n",
                "test": "action_001/test_sample 1 0\n",
            }
            label_meta = []
            for split, content in rows.items():
                path = labels / f"{split}_label.txt"
                path.write_text(content, encoding="utf-8")
                label_meta.append(
                    {
                        "name": path.name,
                        "size": path.stat().st_size,
                        "md5": prepare_hardvs.file_digest(path, "md5"),
                        "sha256": prepare_hardvs.file_digest(path, "sha256"),
                        "samples": 1,
                    }
                )
                sample = f"{split}_sample"
                source = extracted / "action_001" / sample / f"{sample}.npz"
                source.parent.mkdir(parents=True)
                source.write_bytes(split.encode("ascii"))

            excluded = "action_001/train_sample"
            manifest = {
                "labels": label_meta,
                "npz_entries": 4,
                "known_mirror_corrupt_samples": [
                    {"sample_id": excluded, "member": "unused", "split": "train"}
                ],
                "known_mirror_effective_counts": {
                    "train": 0,
                    "val": 1,
                    "test": 1,
                    "assigned": 2,
                    "valid_unassigned_archive_npz": 1,
                },
            }
            result = prepare_hardvs.materialize_splits(
                manifest,
                labels,
                extracted,
                raw,
                excluded_sample_ids={excluded},
            )
            self.assertEqual(result["official_counts"]["assigned"], 3)
            self.assertEqual(result["effective_counts"]["assigned"], 2)
            self.assertFalse(
                (raw / "train" / "action_001" / "train_sample.npz").exists()
            )
            self.assertTrue(
                (raw / "val" / "action_001" / "val_sample.npz").is_symlink()
            )
            self.assertTrue(
                (raw / "test" / "action_001" / "test_sample.npz").is_symlink()
            )


if __name__ == "__main__":
    unittest.main()
