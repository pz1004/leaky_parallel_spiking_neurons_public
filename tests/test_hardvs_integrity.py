from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import acquire_hardvs, prepare_hardvs
from scripts.audit_empirical_package import validate_hardvs_source


class HardvsIntegrityTests(unittest.TestCase):
    def test_committed_source_audit_matches_integrity_manifest(self) -> None:
        project = Path(__file__).resolve().parents[1]
        source_audit = json.loads(
            (project / "manifests" / "hardvs_source_audit.json").read_text(
                encoding="utf-8"
            )
        )
        claimed_hash = source_audit.pop("manifest_hash")
        self.assertEqual(claimed_hash, prepare_hardvs.stable_hash(source_audit))
        manifest = json.loads(
            (
                project / "manifests" / "hardvs_compact_archive.json"
            ).read_text(encoding="utf-8")
        )
        declared = {
            row["sample_id"]
            for key in (
                "known_mirror_corrupt_samples",
                "known_mirror_missing_assigned_samples",
            )
            for row in manifest[key]
        }
        published = {
            row["sample_id"]
            for row in source_audit["excluded_assigned_samples"]
        }
        self.assertEqual(published, declared)
        self.assertEqual(
            source_audit["effective_counts"],
            {
                key: value
                for key, value in manifest[
                    "known_mirror_effective_counts"
                ].items()
                if key != "valid_unassigned_archive_npz"
            },
        )

    def test_fresh_source_accepts_new_provenance_but_keeps_content_checks(self) -> None:
        project = Path(__file__).resolve().parents[1]
        spec = json.loads((project / "manifests/hardvs_compact_archive.json").read_text())
        historical = json.loads((project / "manifests/hardvs_source_audit.json").read_text())
        split = json.loads((project / "manifests/hardvs.json").read_text())
        integrity = dict(split["dataset_integrity"])
        integrity.update(acquisition_record_hash="1" * 64, record_hash="2" * 64,
                         preparation_git_commit="3" * 40, preparation_script_sha256="4" * 64)
        self.assertEqual(validate_hardvs_source(integrity, spec), [])
        self.assertTrue(validate_hardvs_source(integrity, spec, historical=historical))
        integrity["archive_sha256"] = "0" * 64
        self.assertTrue(validate_hardvs_source(integrity, spec))

    def test_fresh_official_source_uses_official_population(self) -> None:
        project = Path(__file__).resolve().parents[1]
        spec = json.loads((project / "manifests/hardvs_compact_archive.json").read_text())
        counts = {row["name"].split("_", 1)[0]: row["samples"] for row in spec["labels"]}
        counts["assigned"] = spec["official_assigned_samples"]
        integrity = {"integrity_mode": "official_exact", "archive_md5": spec["official_archive_md5"],
                     "effective_counts": counts, "excluded_sample_ids": [],
                     **{key: "1" * 64 for key in ("archive_sha256", "acquisition_record_hash", "record_hash", "preparation_script_sha256")}}
        self.assertEqual(validate_hardvs_source(integrity, spec), [])
        integrity["excluded_sample_ids"] = ["unexpected/sample"]
        self.assertTrue(validate_hardvs_source(integrity, spec))

    def test_acquisition_reconciles_missing_assigned_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels"
            labels.mkdir()
            (labels / "train_label.txt").write_text(
                "action_001/assigned_sample 1 0\n", encoding="utf-8"
            )
            (labels / "val_label.txt").write_text("", encoding="utf-8")
            (labels / "test_label.txt").write_text("", encoding="utf-8")
            archive = root / "mirror.zip"
            member = (
                "MINIHARDVS_EVENT_files/action_001/archive_only/"
                "archive_only.npz"
            )
            payload = b"valid"
            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_DEFLATED
            ) as handle:
                handle.writestr(member, payload)
            hashes = prepare_hardvs.file_digests(
                archive, ("md5", "sha256")
            )
            manifest = {
                "archive_size": archive.stat().st_size,
                "archive_entries": 1,
                "npz_entries": 1,
                "uncompressed_bytes": len(payload),
                "official_assigned_samples": 1,
                "official_unassigned_archive_npz": 0,
                "official_archive_md5": "f" * 32,
                "known_mirror_archive_md5": hashes["md5"],
                "known_mirror_archive_sha256": hashes["sha256"],
                "known_mirror_corrupt_samples": [],
                "known_mirror_missing_assigned_samples": [
                    {
                        "sample_id": "action_001/assigned_sample",
                        "split": "train",
                    }
                ],
                "known_mirror_effective_counts": {
                    "train": 0,
                    "val": 0,
                    "test": 0,
                    "assigned": 0,
                    "valid_unassigned_archive_npz": 1,
                },
            }
            result = acquire_hardvs.verify_zip(
                manifest,
                archive,
                labels_dir=labels,
                allow_known_corrupt_mirror=True,
            )
            reconciliation = result["label_reconciliation"]
            self.assertEqual(result["integrity_mode"], "known_mirror_exclusion")
            self.assertEqual(
                reconciliation["missing_assigned_sample_ids"],
                ["action_001/assigned_sample"],
            )
            self.assertEqual(
                reconciliation["valid_unassigned_archive_sample_ids"],
                ["action_001/archive_only"],
            )

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
            "official_assigned_samples": 1,
            "official_unassigned_archive_npz": 0,
            "labels": [
                {"name": "train_label.txt", "samples": 1},
                {"name": "val_label.txt", "samples": 0},
                {"name": "test_label.txt", "samples": 0},
            ],
            "official_archive_md5": hashes["md5"],
            "known_mirror_archive_md5": "0" * 32,
            "known_mirror_archive_sha256": "0" * 64,
            "known_mirror_corrupt_samples": [],
        }
        record = {
            "schema_version": 3,
            "integrity_mode": "official_exact",
            "archive_md5": hashes["md5"],
            "archive_sha256": hashes["sha256"],
            "archive_entries": 1,
            "npz_entries": 1,
            "uncompressed_bytes": len(payload),
            "full_crc_members": 1,
            "full_crc_failures": [],
            "manifest_sha256": "fixture-manifest",
            "label_reconciliation": {
                "assigned_label_rows": 1,
                "archive_sample_ids": 1,
                "missing_assigned_sample_ids": [],
                "corrupt_assigned_sample_ids": [],
                "excluded_assigned_sample_ids": [],
                "effective_counts": {
                    "train": 1,
                    "val": 0,
                    "test": 0,
                    "assigned": 1,
                },
                "valid_unassigned_archive_count": 0,
            },
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

    def test_extraction_audit_rejects_same_size_content_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, _manifest = self._archive_fixture(root)
            with zipfile.ZipFile(archive) as handle:
                infos = handle.infolist()
            destination = root / "extracted"
            target = destination / infos[0].filename
            target.parent.mkdir(parents=True)
            target.write_bytes(b"x" * infos[0].file_size)
            with self.assertRaisesRegex(RuntimeError, "wrong_crc"):
                prepare_hardvs.validate_extracted_members(
                    infos,
                    destination,
                    excluded_members=set(),
                )

    def test_split_materialization_excludes_only_declared_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels"
            extracted = root / "extracted"
            raw = root / "raw"
            labels.mkdir()
            rows = {
                "train": (
                    "action_001/train_sample 1 0\n"
                    "action_001/missing_sample 1 0\n"
                ),
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
                        "samples": len(content.splitlines()),
                    }
                )
                sample = f"{split}_sample"
                source = extracted / "action_001" / sample / f"{sample}.npz"
                source.parent.mkdir(parents=True)
                source.write_bytes(split.encode("ascii"))

            corrupt = "action_001/train_sample"
            missing = "action_001/missing_sample"
            manifest = {
                "labels": label_meta,
                "npz_entries": 4,
                "known_mirror_corrupt_samples": [
                    {"sample_id": corrupt, "member": "unused", "split": "train"}
                ],
                "known_mirror_missing_assigned_samples": [
                    {"sample_id": missing, "split": "train"}
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
                excluded_sample_ids={corrupt, missing},
            )
            self.assertEqual(result["official_counts"]["assigned"], 4)
            self.assertEqual(result["effective_counts"]["assigned"], 2)
            self.assertEqual(result["valid_archive_npz"], 3)
            self.assertEqual(result["valid_unassigned_archive_npz"], 1)
            self.assertEqual(
                result["missing_assigned_sample_ids"],
                [missing],
            )
            self.assertFalse(
                (raw / "train" / "action_001" / "train_sample.npz").exists()
            )
            self.assertFalse(
                (raw / "train" / "action_001" / "missing_sample.npz").exists()
            )
            self.assertTrue(
                (raw / "val" / "action_001" / "val_sample.npz").is_symlink()
            )
            self.assertTrue(
                (raw / "test" / "action_001" / "test_sample.npz").is_symlink()
            )


if __name__ == "__main__":
    unittest.main()
