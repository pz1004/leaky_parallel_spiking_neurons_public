#!/usr/bin/env python3
"""Extract HARDVS compact events and materialize the official split layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


def file_digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


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


def validate_archive(manifest: dict[str, Any], archive: Path) -> None:
    if archive.stat().st_size != int(manifest["archive_size"]):
        raise RuntimeError(f"archive-size mismatch: {archive.stat().st_size}")
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
    if len(infos) != int(manifest["archive_entries"]) or npz_count != int(manifest["npz_entries"]):
        raise RuntimeError(f"archive central-directory mismatch: entries={len(infos)}, npz={npz_count}")


def extract(archive: Path, destination: Path) -> None:
    marker = destination / ".extraction_complete"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", str(archive), "-d", str(destination)], check=True)
    marker.write_text("unzip completed with CRC checks\n", encoding="utf-8")


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
) -> dict[str, Any]:
    expected_label_meta = {row["name"]: row for row in manifest["labels"]}
    split_rows: dict[str, list[tuple[str, int]]] = {}
    all_ids: set[str] = set()
    for split in ("train", "val", "test"):
        label_path = labels_dir / f"{split}_label.txt"
        meta = expected_label_meta[label_path.name]
        if label_path.stat().st_size != int(meta["size"]):
            raise RuntimeError(f"label-size mismatch: {label_path}")
        if file_digest(label_path, "md5") != meta["md5"]:
            raise RuntimeError(f"label MD5 mismatch: {label_path}")
        rows = read_label_rows(label_path)
        if len(rows) != int(meta["samples"]):
            raise RuntimeError(f"label count mismatch: {label_path}")
        ids = {sample_id for sample_id, _label in rows}
        if len(ids) != len(rows) or ids & all_ids:
            raise RuntimeError(f"duplicate or cross-split identifiers in {label_path}")
        all_ids.update(ids)
        split_rows[split] = rows

    for split, rows in split_rows.items():
        split_root = raw_root / split
        for class_index in range(300):
            (split_root / f"action_{class_index + 1:03d}").mkdir(parents=True, exist_ok=True)
        for sample_id, _label in rows:
            class_name, sample_name = sample_id.split("/")
            source = locate_source(extracted_root, sample_id)
            target = split_root / class_name / f"{sample_name}.npz"
            relative_source = Path(os.path.relpath(source, target.parent))
            if target.is_symlink():
                if target.readlink() != relative_source:
                    raise RuntimeError(f"existing split link has wrong target: {target}")
            elif target.exists():
                raise RuntimeError(f"refusing to replace non-symlink split file: {target}")
            else:
                target.symlink_to(relative_source)

    return {
        "train": len(split_rows["train"]),
        "val": len(split_rows["val"]),
        "test": len(split_rows["test"]),
        "assigned": len(all_ids),
        "archive_npz": int(manifest["npz_entries"]),
        "unassigned_archive_npz": int(manifest["npz_entries"]) - len(all_ids),
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
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    download = args.download.resolve()
    hardvs_root = args.data_root.resolve() / "HARDVS"
    archive = download / manifest["archive_name"]
    validate_archive(manifest, archive)

    extraction_dir = hardvs_root / "extracted"
    extracted_root = extraction_dir / "MINIHARDVS_EVENT_files"
    extract(archive, extraction_dir)
    counts = materialize_splits(
        manifest,
        download / "labels",
        extracted_root,
        hardvs_root / "raw",
    )
    record = {
        "schema_version": 1,
        "archive": {
            "name": archive.name,
            "size": archive.stat().st_size,
            "part_sha256_verified": True,
            "zip_crc_verified_during_extraction": True,
        },
        "splits": counts,
    }
    record_path = hardvs_root / "integrity_record.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    if not args.keep_archive:
        archive.unlink()
        print(f"removed verified source archive after extraction: {archive}")


if __name__ == "__main__":
    main()
