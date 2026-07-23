#!/usr/bin/env python3
"""Extract HARDVS compact events and materialize the official split layout."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def file_digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def file_digests(path: Path, algorithms: tuple[str, ...]) -> dict[str, str]:
    values = {algorithm: hashlib.new(algorithm) for algorithm in algorithms}
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            for value in values.values():
                value.update(block)
    return {algorithm: value.hexdigest() for algorithm, value in values.items()}


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_state(project: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_commit": (
            commit.stdout.strip() if commit.returncode == 0 else "unknown"
        ),
        "git_dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def known_corrupt_members(manifest: dict[str, Any]) -> set[str]:
    return {
        str(row["member"]) for row in manifest.get("known_mirror_corrupt_samples", [])
    }


def known_corrupt_sample_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        str(row["sample_id"])
        for row in manifest.get("known_mirror_corrupt_samples", [])
    }


def known_missing_assigned_sample_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        str(row["sample_id"])
        for row in manifest.get("known_mirror_missing_assigned_samples", [])
    }


def known_excluded_sample_ids(manifest: dict[str, Any]) -> set[str]:
    return known_corrupt_sample_ids(manifest) | known_missing_assigned_sample_ids(
        manifest
    )


def read_label_rows(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"{path}:{number}: expected three fields, got {len(fields)}")
        sample_id, _frames, label = fields
        sample_path = PurePosixPath(sample_id)
        if len(sample_path.parts) != 2 or not sample_path.parts[0].startswith("action_"):
            raise ValueError(f"{path}:{number}: invalid HARDVS sample identifier {sample_id!r}")
        class_index = int(sample_path.parts[0].split("_", 1)[1]) - 1
        if int(label) != class_index:
            raise ValueError(
                f"{path}:{number}: label {label} disagrees with class directory {class_index}"
            )
        rows.append((sample_id, class_index))
    return rows


def validate_acquisition_record(
    record_path: Path,
    *,
    archive_md5: str,
    archive_sha256: str,
    expected_bad: set[str],
    expected_npz_members: int,
    expected_archive_entries: int,
    expected_uncompressed_bytes: int,
    expected_integrity_mode: str,
    expected_manifest_sha256: str,
    expected_missing_assigned: set[str],
    expected_corrupt_assigned: set[str],
    expected_excluded_assigned: set[str],
    expected_assigned_label_rows: int,
    expected_archive_sample_ids: int,
    expected_effective_counts: dict[str, int],
    expected_valid_unassigned_count: int,
) -> dict[str, Any]:
    if not record_path.is_file():
        raise RuntimeError(
            "Preparation requires acquisition_record.json from "
            "scripts/acquire_hardvs.py so the complete archive CRC audit is recorded."
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record_hash = record.pop("record_hash", None)
    if record_hash != stable_hash(record):
        raise RuntimeError(f"Acquisition-record hash mismatch: {record_path}")
    if int(record.get("schema_version", -1)) < 3:
        raise RuntimeError("Acquisition record predates label/archive reconciliation")
    if record.get("archive_md5") != archive_md5:
        raise RuntimeError("Acquisition record and current archive MD5 differ")
    if record.get("archive_sha256") != archive_sha256:
        raise RuntimeError("Acquisition record and current archive SHA-256 differ")
    if record.get("integrity_mode") != expected_integrity_mode:
        raise RuntimeError(
            "Acquisition-record integrity mode does not match the current archive"
        )
    if record.get("manifest_sha256") != expected_manifest_sha256:
        raise RuntimeError(
            "Acquisition record was not created from the current integrity manifest"
        )
    actual_bad = {
        str(row["member"]) for row in record.get("full_crc_failures", [])
    }
    if actual_bad != expected_bad:
        raise RuntimeError(
            "Acquisition record does not contain the exact known corruption set: "
            f"{sorted(actual_bad)}"
        )
    if int(record.get("full_crc_members", -1)) != expected_npz_members:
        raise RuntimeError(
            "Acquisition record does not cover all declared NPZ members: "
            f"{record.get('full_crc_members')} != {expected_npz_members}"
        )
    if int(record.get("archive_entries", -1)) != expected_archive_entries:
        raise RuntimeError("Acquisition-record ZIP entry count differs from the manifest")
    if int(record.get("npz_entries", -1)) != expected_npz_members:
        raise RuntimeError("Acquisition-record NPZ count differs from the manifest")
    if int(record.get("uncompressed_bytes", -1)) != expected_uncompressed_bytes:
        raise RuntimeError(
            "Acquisition-record uncompressed-byte count differs from the manifest"
        )
    reconciliation = record.get("label_reconciliation", {})
    actual_missing = set(reconciliation.get("missing_assigned_sample_ids", []))
    actual_corrupt = set(reconciliation.get("corrupt_assigned_sample_ids", []))
    actual_excluded = set(reconciliation.get("excluded_assigned_sample_ids", []))
    if actual_missing != expected_missing_assigned:
        raise RuntimeError("Acquisition-record missing-assigned set differs")
    if actual_corrupt != expected_corrupt_assigned:
        raise RuntimeError("Acquisition-record corrupt-assigned set differs")
    if actual_excluded != expected_excluded_assigned:
        raise RuntimeError("Acquisition-record excluded-assigned set differs")
    if (
        int(reconciliation.get("assigned_label_rows", -1))
        != expected_assigned_label_rows
    ):
        raise RuntimeError("Acquisition-record assigned-label count differs")
    if (
        int(reconciliation.get("archive_sample_ids", -1))
        != expected_archive_sample_ids
    ):
        raise RuntimeError("Acquisition-record archive-sample count differs")
    actual_effective_counts = {
        str(key): int(value)
        for key, value in reconciliation.get("effective_counts", {}).items()
    }
    if actual_effective_counts != expected_effective_counts:
        raise RuntimeError("Acquisition-record effective split counts differ")
    if (
        int(reconciliation.get("valid_unassigned_archive_count", -1))
        != expected_valid_unassigned_count
    ):
        raise RuntimeError("Acquisition-record valid-unassigned count differs")
    return {**record, "record_hash": record_hash}


def validate_archive(
    manifest: dict[str, Any],
    archive: Path,
    *,
    acquisition_record_path: Path,
    allow_known_corrupt_mirror: bool,
    manifest_sha256: str,
) -> dict[str, Any]:
    if archive.stat().st_size != int(manifest["archive_size"]):
        raise RuntimeError(f"archive-size mismatch: {archive.stat().st_size}")
    archive_hashes = file_digests(archive, ("md5", "sha256"))
    archive_md5 = archive_hashes["md5"]
    official_md5 = str(manifest["official_archive_md5"])
    mirror_md5 = str(manifest["known_mirror_archive_md5"])
    expected_bad: set[str]
    if archive_md5 == official_md5:
        integrity_mode = "official_exact"
        expected_bad = set()
        expected_missing_assigned: set[str] = set()
        expected_corrupt_assigned: set[str] = set()
        expected_excluded_assigned: set[str] = set()
        expected_effective_counts = {
            str(row["name"]).removesuffix("_label.txt"): int(row["samples"])
            for row in manifest["labels"]
        }
        expected_effective_counts["assigned"] = sum(
            expected_effective_counts.values()
        )
        expected_valid_unassigned_count = int(
            manifest["official_unassigned_archive_npz"]
        )
    elif archive_md5 == mirror_md5:
        if not allow_known_corrupt_mirror:
            raise RuntimeError(
                "The available archive is the known-corrupt mirror, not the "
                "official HARDVS object. Use an exact official archive or pass "
                "--allow-known-corrupt-mirror to apply the recorded 13-sample "
                "exclusion protocol."
            )
        integrity_mode = "known_mirror_exclusion"
        expected_bad = known_corrupt_members(manifest)
        expected_missing_assigned = known_missing_assigned_sample_ids(manifest)
        expected_corrupt_assigned = known_corrupt_sample_ids(manifest)
        expected_excluded_assigned = known_excluded_sample_ids(manifest)
        expected_effective_counts = {
            str(key): int(value)
            for key, value in manifest["known_mirror_effective_counts"].items()
            if key != "valid_unassigned_archive_npz"
        }
        expected_valid_unassigned_count = int(
            manifest["known_mirror_effective_counts"][
                "valid_unassigned_archive_npz"
            ]
        )
        expected_sha256 = str(manifest["known_mirror_archive_sha256"])
        if archive_hashes["sha256"] != expected_sha256:
            raise RuntimeError("Current archive SHA-256 is not the pinned mirror SHA-256")
    else:
        raise RuntimeError(
            "Archive MD5 matches neither the official object nor the pinned "
            f"known mirror: {archive_md5}"
        )
    acquisition_record = validate_acquisition_record(
        acquisition_record_path,
        archive_md5=archive_md5,
        archive_sha256=archive_hashes["sha256"],
        expected_bad=expected_bad,
        expected_npz_members=int(manifest["npz_entries"]),
        expected_archive_entries=int(manifest["archive_entries"]),
        expected_uncompressed_bytes=int(manifest["uncompressed_bytes"]),
        expected_integrity_mode=integrity_mode,
        expected_manifest_sha256=manifest_sha256,
        expected_missing_assigned=expected_missing_assigned,
        expected_corrupt_assigned=expected_corrupt_assigned,
        expected_excluded_assigned=expected_excluded_assigned,
        expected_assigned_label_rows=int(manifest["official_assigned_samples"]),
        expected_archive_sample_ids=int(manifest["npz_entries"]),
        expected_effective_counts=expected_effective_counts,
        expected_valid_unassigned_count=expected_valid_unassigned_count,
    )
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe ZIP member: {info.filename}")
        if not info.is_dir() and path.suffix.lower() != ".npz":
            raise RuntimeError(f"unexpected non-NPZ ZIP member: {info.filename}")
        if not path.parts or path.parts[0] != "MINIHARDVS_EVENT_files":
            raise RuntimeError(f"unexpected ZIP root: {info.filename}")
    npz_count = sum(info.filename.lower().endswith(".npz") for info in infos)
    uncompressed_bytes = sum(info.file_size for info in infos)
    if len(infos) != int(manifest["archive_entries"]) or npz_count != int(manifest["npz_entries"]):
        raise RuntimeError(f"archive central-directory mismatch: entries={len(infos)}, npz={npz_count}")
    if uncompressed_bytes != int(manifest["uncompressed_bytes"]):
        raise RuntimeError(
            "archive uncompressed-byte mismatch: "
            f"{uncompressed_bytes} != {manifest['uncompressed_bytes']}"
        )
    return {
        "integrity_mode": integrity_mode,
        "archive_md5": archive_md5,
        "archive_sha256": archive_hashes["sha256"],
        "excluded_members": sorted(expected_bad),
        "excluded_sample_ids": sorted(expected_excluded_assigned),
        "infos": infos,
        "acquisition_record": acquisition_record,
    }


def quarantine_excluded_members(
    destination: Path,
    quarantine: Path,
    excluded_members: set[str],
) -> list[str]:
    moved: list[str] = []
    for member in sorted(excluded_members):
        source = destination / PurePosixPath(member)
        if not source.is_file():
            continue
        relative = PurePosixPath(member)
        target = quarantine / Path(*relative.parts[1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            repeated = target.with_name(f"{target.stem}.repeat{target.suffix}")
            if repeated.exists():
                raise RuntimeError(
                    f"Refusing to overwrite repeated quarantined member: {repeated}"
                )
            source.replace(repeated)
        else:
            source.replace(target)
        moved.append(member)
    return moved


def validate_extracted_members(
    infos: list[zipfile.ZipInfo],
    destination: Path,
    excluded_members: set[str],
) -> dict[str, int]:
    good = 0
    missing: list[str] = []
    wrong_size: list[tuple[str, int, int]] = []
    wrong_crc: list[tuple[str, str, str]] = []
    exposed_exclusions: list[str] = []
    for info in infos:
        if info.is_dir() or not info.filename.lower().endswith(".npz"):
            continue
        path = destination / PurePosixPath(info.filename)
        if info.filename in excluded_members:
            if path.exists():
                exposed_exclusions.append(info.filename)
            continue
        good += 1
        if not path.is_file():
            missing.append(info.filename)
            continue
        size = path.stat().st_size
        if size != info.file_size:
            wrong_size.append((info.filename, size, info.file_size))
            continue
        checksum = 0
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                checksum = binascii.crc32(block, checksum)
        checksum &= 0xFFFFFFFF
        if checksum != info.CRC:
            wrong_crc.append(
                (info.filename, f"{checksum:08x}", f"{info.CRC:08x}")
            )
    if missing or wrong_size or wrong_crc or exposed_exclusions:
        raise RuntimeError(
            "Extracted-member audit failed: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"wrong_size={wrong_size[:5]} ({len(wrong_size)} total), "
            f"wrong_crc={wrong_crc[:5]} ({len(wrong_crc)} total), "
            f"exposed_exclusions={exposed_exclusions}"
        )
    return {
        "good_npz_members": good,
        "missing_good_members": 0,
        "wrong_size_good_members": 0,
        "wrong_crc_good_members": 0,
        "crc_verified_good_members": good,
        "exposed_excluded_members": 0,
    }


def extract(
    archive: Path,
    destination: Path,
    *,
    infos: list[zipfile.ZipInfo],
    integrity_mode: str,
    archive_md5: str,
    excluded_members: set[str],
    quarantine: Path,
) -> dict[str, Any]:
    marker = destination / ".extraction_complete"
    destination.mkdir(parents=True, exist_ok=True)
    expected_marker = {
        "schema_version": 2,
        "integrity_mode": integrity_mode,
        "archive_md5": archive_md5,
        "excluded_members": sorted(excluded_members),
    }
    if marker.is_file():
        try:
            current_marker = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current_marker = {}
        if all(current_marker.get(key) == value for key, value in expected_marker.items()):
            return {**current_marker, **validate_extracted_members(infos, destination, excluded_members)}

    quarantine_excluded_members(destination, quarantine, excluded_members)
    try:
        extraction_audit = validate_extracted_members(
            infos, destination, excluded_members
        )
        reused_existing_extraction = True
        unzip_returncode: int | None = None
    except RuntimeError:
        result = subprocess.run(
            ["unzip", "-q", "-o", str(archive), "-d", str(destination)],
            check=False,
        )
        unzip_returncode = int(result.returncode)
        if integrity_mode == "official_exact" and unzip_returncode != 0:
            raise RuntimeError(f"Exact official archive extraction failed: exit {unzip_returncode}")
        if integrity_mode == "known_mirror_exclusion" and unzip_returncode == 0:
            raise RuntimeError("Known-corrupt mirror unexpectedly extracted with no errors")
        quarantine_excluded_members(destination, quarantine, excluded_members)
        extraction_audit = validate_extracted_members(
            infos, destination, excluded_members
        )
        reused_existing_extraction = False

    marker_payload = {
        **expected_marker,
        **extraction_audit,
        "reused_existing_extraction": reused_existing_extraction,
        "unzip_returncode": unzip_returncode,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    marker.write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return marker_payload


def locate_source(extracted_root: Path, sample_id: str) -> Path:
    class_name, sample_name = sample_id.split("/")
    candidates = (
        extracted_root / class_name / sample_name / f"{sample_name}.npz",
        extracted_root / class_name / f"{sample_name}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"archive member missing for {sample_id}: {candidates}")


def materialize_splits(
    manifest: dict[str, Any],
    labels_dir: Path,
    extracted_root: Path,
    raw_root: Path,
    *,
    excluded_sample_ids: set[str],
) -> dict[str, Any]:
    expected_label_meta = {row["name"]: row for row in manifest["labels"]}
    split_rows: dict[str, list[tuple[str, int]]] = {}
    sample_split: dict[str, str] = {}
    all_ids: set[str] = set()
    for split in ("train", "val", "test"):
        label_path = labels_dir / f"{split}_label.txt"
        meta = expected_label_meta[label_path.name]
        if label_path.stat().st_size != int(meta["size"]):
            raise RuntimeError(f"label-size mismatch: {label_path}")
        if file_digest(label_path, "md5") != meta["md5"]:
            raise RuntimeError(f"label MD5 mismatch: {label_path}")
        if file_digest(label_path, "sha256") != meta["sha256"]:
            raise RuntimeError(f"label SHA-256 mismatch: {label_path}")
        rows = read_label_rows(label_path)
        if len(rows) != int(meta["samples"]):
            raise RuntimeError(f"label count mismatch: {label_path}")
        ids = {sample_id for sample_id, _label in rows}
        if len(ids) != len(rows) or ids & all_ids:
            raise RuntimeError(f"duplicate or cross-split identifiers in {label_path}")
        all_ids.update(ids)
        sample_split.update({sample_id: split for sample_id in ids})
        split_rows[split] = rows

    unknown_exclusions = excluded_sample_ids - all_ids
    if unknown_exclusions:
        raise RuntimeError(
            f"Excluded sample IDs are absent from official labels: {sorted(unknown_exclusions)}"
        )
    declared_exclusions = known_excluded_sample_ids(manifest)
    declared_rows = [
        *manifest.get("known_mirror_corrupt_samples", []),
        *manifest.get("known_mirror_missing_assigned_samples", []),
    ]
    if len(declared_rows) != len(declared_exclusions):
        raise RuntimeError("Duplicate known-mirror exclusion IDs in the manifest")
    for row in declared_rows:
        sample_id = str(row["sample_id"])
        if sample_split.get(sample_id) != row.get("split"):
            raise RuntimeError(
                "Known-corrupt sample split disagrees with official labels: "
                f"{sample_id}"
            )
    if excluded_sample_ids and excluded_sample_ids != declared_exclusions:
        raise RuntimeError(
            "Effective exclusion set differs from the manifest: "
            f"{sorted(excluded_sample_ids)}"
        )

    desired_targets: set[Path] = set()
    effective_counts: dict[str, int] = {}
    exclusions_by_split: dict[str, list[str]] = {}
    for split, rows in split_rows.items():
        split_root = raw_root / split
        for class_index in range(300):
            (split_root / f"action_{class_index + 1:03d}").mkdir(parents=True, exist_ok=True)
        for sample_id, _label in rows:
            if sample_id in excluded_sample_ids:
                continue
            class_name, sample_name = sample_id.split("/")
            target = split_root / class_name / f"{sample_name}.npz"
            desired_targets.add(target)
            source = locate_source(extracted_root, sample_id)
            relative_source = Path(os.path.relpath(source, target.parent))
            if target.is_symlink():
                if target.readlink() != relative_source:
                    raise RuntimeError(f"existing split link has wrong target: {target}")
            elif target.exists():
                raise RuntimeError(f"refusing to replace non-symlink split file: {target}")
            else:
                target.symlink_to(relative_source)
        effective_counts[split] = sum(
            sample_id not in excluded_sample_ids for sample_id, _label in rows
        )
        exclusions_by_split[split] = sorted(
            sample_id
            for sample_id, _label in rows
            if sample_id in excluded_sample_ids
        )

    removed_stale_links: list[str] = []
    if raw_root.exists():
        for path in raw_root.glob("*/*/*.npz"):
            if path in desired_targets:
                continue
            if not path.is_symlink():
                raise RuntimeError(f"unexpected non-symlink split artifact: {path}")
            path.unlink()
            removed_stale_links.append(str(path.relative_to(raw_root)))

    actual_targets = set(raw_root.glob("*/*/*.npz"))
    if actual_targets != desired_targets:
        missing = sorted(str(path) for path in desired_targets - actual_targets)
        unexpected = sorted(str(path) for path in actual_targets - desired_targets)
        raise RuntimeError(
            "Materialized split audit failed: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
        )

    official_counts = {split: len(rows) for split, rows in split_rows.items()}
    official_assigned = len(all_ids)
    effective_assigned = sum(effective_counts.values())
    corrupt_exclusions = (
        known_corrupt_sample_ids(manifest) & excluded_sample_ids
    )
    missing_assigned_exclusions = (
        known_missing_assigned_sample_ids(manifest) & excluded_sample_ids
    )
    valid_archive_npz = int(manifest["npz_entries"]) - len(corrupt_exclusions)
    valid_unassigned = valid_archive_npz - effective_assigned
    if excluded_sample_ids:
        expected_effective = manifest["known_mirror_effective_counts"]
        observed_effective = {
            **effective_counts,
            "assigned": effective_assigned,
            "valid_unassigned_archive_npz": valid_unassigned,
        }
        if observed_effective != expected_effective:
            raise RuntimeError(
                "Effective split counts differ from the manifest: "
                f"{observed_effective} != {expected_effective}"
            )

    return {
        "official_counts": {
            **official_counts,
            "assigned": official_assigned,
        },
        "effective_counts": {
            **effective_counts,
            "assigned": effective_assigned,
        },
        "excluded_sample_ids": sorted(excluded_sample_ids),
        "corrupt_excluded_sample_ids": sorted(corrupt_exclusions),
        "missing_assigned_sample_ids": sorted(missing_assigned_exclusions),
        "exclusions_by_split": exclusions_by_split,
        "archive_npz": int(manifest["npz_entries"]),
        "valid_archive_npz": valid_archive_npz,
        "valid_unassigned_archive_npz": valid_unassigned,
        "removed_stale_links": removed_stale_links,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    project = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project / "manifests" / "hardvs_compact_archive.json",
    )
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument(
        "--allow-dirty-code",
        action="store_true",
        help="Record but explicitly allow a dirty tracked source tree.",
    )
    parser.add_argument(
        "--allow-known-corrupt-mirror",
        action="store_true",
        help=(
            "Explicitly accept only the pinned known-deviant mirror after a "
            "complete CRC and label/archive audit, quarantine its six corrupt "
            "members, and exclude all 13 unavailable assigned rows."
        ),
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    preparation_git = git_state(project)
    if preparation_git["git_dirty"] and not args.allow_dirty_code:
        raise RuntimeError(
            "Refusing dataset preparation from a dirty tracked source tree; "
            "commit the preparation code or pass --allow-dirty-code explicitly."
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    download = args.download.resolve()
    hardvs_root = args.data_root.resolve() / "HARDVS"
    archive = download / manifest["archive_name"]
    archive_validation = validate_archive(
        manifest,
        archive,
        acquisition_record_path=download / "acquisition_record.json",
        allow_known_corrupt_mirror=args.allow_known_corrupt_mirror,
        manifest_sha256=file_digest(args.manifest.resolve(), "sha256"),
    )

    extraction_dir = hardvs_root / "extracted"
    extracted_root = extraction_dir / "MINIHARDVS_EVENT_files"
    extraction_record = extract(
        archive,
        extraction_dir,
        infos=archive_validation["infos"],
        integrity_mode=archive_validation["integrity_mode"],
        archive_md5=archive_validation["archive_md5"],
        excluded_members=set(archive_validation["excluded_members"]),
        quarantine=hardvs_root / "quarantine",
    )
    counts = materialize_splits(
        manifest,
        download / "labels",
        extracted_root,
        hardvs_root / "raw",
        excluded_sample_ids=set(archive_validation["excluded_sample_ids"]),
    )
    record = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "integrity_mode": archive_validation["integrity_mode"],
        "source_deviation": (
            None
            if archive_validation["integrity_mode"] == "official_exact"
            else (
                "The pinned mirror is not byte-identical to the official archive. "
                "Six corrupt assigned members and seven assigned rows absent "
                "from the archive were excluded exactly as declared in the "
                "integrity manifest."
            )
        ),
        "preparation_runtime": {
            **preparation_git,
            "python": sys.version,
            "script": str(Path(__file__).resolve()),
            "script_sha256": file_digest(Path(__file__).resolve(), "sha256"),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_digest(args.manifest.resolve(), "sha256"),
            "argv": sys.argv,
        },
        "archive": {
            "name": archive.name,
            "size": archive.stat().st_size,
            "md5": archive_validation["archive_md5"],
            "sha256": archive_validation["archive_sha256"],
            "complete_crc_member_count": int(manifest["npz_entries"]),
            "complete_crc_failure_count": len(archive_validation["excluded_members"]),
            "missing_assigned_count": len(
                archive_validation["acquisition_record"][
                    "label_reconciliation"
                ]["missing_assigned_sample_ids"]
            ),
            "excluded_assigned_count": len(
                archive_validation["excluded_sample_ids"]
            ),
            "acquisition_record_hash": archive_validation["acquisition_record"]["record_hash"],
        },
        "extraction": extraction_record,
        "splits": counts,
    }
    record["record_hash"] = stable_hash(record)
    record_path = hardvs_root / "integrity_record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    if not args.keep_archive:
        archive.unlink()
        print(f"removed verified source archive after extraction: {archive}")


if __name__ == "__main__":
    main()
