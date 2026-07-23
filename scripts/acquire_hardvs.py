#!/usr/bin/env python3
"""Acquire and cryptographically verify the HARDVS compact-event archive.

The official project links a single 244.899 GB Baidu object. A pinned public
mirror stores a same-size object as 23 ordered binary chunks, but that combined
object is not byte-identical to the official archive, contains six corrupt
assigned NPZ members, and omits seven additional assigned label rows. Exact
official bytes are required by default. The known mirror can be used only with
an explicit opt-in that verifies its combined MD5/SHA-256, reproduces the
complete six-member CRC failure inventory, and reconciles every label row with
the archive central directory.

The download is resumable: partial HTTP downloads and a boundary-aligned
combined archive are reused.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

BUFFER_SIZE = 8 * 1024 * 1024
USER_AGENT = "LP-SN reproducibility artifact HARDVS acquisition/1.0"


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(BUFFER_SIZE):
            value.update(block)
    return value.hexdigest()


def combined_digests(path: Path, algorithms: tuple[str, ...]) -> dict[str, str]:
    values = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as handle:
        while block := handle.read(BUFFER_SIZE):
            for value in values.values():
                value.update(block)
    return {name: value.hexdigest() for name, value in values.items()}


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def request(url: str, start: int = 0):
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def download_one(
    part: dict[str, Any],
    chunks_dir: Path,
    base_url: str,
    attempts: int,
) -> Path:
    target = chunks_dir / part["name"]
    partial = target.with_suffix(target.suffix + ".partial")
    expected_size = int(part["size"])
    expected_hash = str(part["sha256"])
    if target.exists():
        if target.stat().st_size == expected_size and digest(target) == expected_hash:
            print(f"[verified] {target.name}", flush=True)
            return target
        target.rename(partial)

    for attempt in range(1, attempts + 1):
        current = partial.stat().st_size if partial.exists() else 0
        if current > expected_size:
            partial.unlink()
            current = 0
        url = f"{base_url}/{part['name']}?download=true"
        try:
            with request(url, current) as response:
                status = getattr(response, "status", response.getcode())
                if current and status != 206:
                    partial.unlink()
                    current = 0
                    raise RuntimeError("server did not honor the resume Range request")
                mode = "ab" if current else "wb"
                with partial.open(mode) as handle:
                    while block := response.read(BUFFER_SIZE):
                        handle.write(block)
            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(f"size {actual_size} != {expected_size}")
            actual_hash = digest(partial)
            if actual_hash != expected_hash:
                partial.unlink()
                raise RuntimeError(f"SHA-256 {actual_hash} != {expected_hash}")
            partial.rename(target)
            print(f"[downloaded+verified] {target.name}", flush=True)
            return target
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            if attempt == attempts:
                raise
            delay = min(60, 2**attempt)
            print(
                f"[retry {attempt}/{attempts}] {part['name']}: {error}; "
                f"resuming in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def valid_combined_parts(archive: Path, parts: list[dict[str, Any]]) -> int:
    if not archive.exists():
        return 0
    size = archive.stat().st_size
    cumulative = 0
    if size == 0:
        return 0
    for count, part in enumerate(parts, start=1):
        cumulative += int(part["size"])
        if size == cumulative:
            return count
        if size < cumulative:
            raise RuntimeError(
                f"{archive} ends inside chunk {part['index']} ({size} bytes). "
                "Move the incomplete combined archive aside and rerun."
            )
    raise RuntimeError(f"{archive} is larger than the declared archive ({size} bytes)")


def concatenate(
    archive: Path,
    chunks_dir: Path,
    parts: list[dict[str, Any]],
    keep_chunks: bool,
) -> None:
    completed = valid_combined_parts(archive, parts)
    print(f"[combine] resuming after {completed}/{len(parts)} chunks", flush=True)
    with archive.open("ab") as output:
        for part in parts[completed:]:
            source = chunks_dir / part["name"]
            if not source.is_file():
                raise FileNotFoundError(source)
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, output, length=BUFFER_SIZE)
            output.flush()
            os.fsync(output.fileno())
            print(f"[combined] {source.name}", flush=True)
            if not keep_chunks:
                source.unlink()
    expected = sum(int(part["size"]) for part in parts)
    if archive.stat().st_size != expected:
        raise RuntimeError(f"combined size mismatch: {archive.stat().st_size} != {expected}")


def download_labels(
    manifest: dict[str, Any], output: Path, attempts: int
) -> list[dict[str, Any]]:
    repo = manifest["mirror_repository"]
    revision = manifest["mirror_revision"]
    base = f"https://huggingface.co/datasets/{repo}/resolve/{revision}"
    verified: list[dict[str, Any]] = []
    for label in manifest["labels"]:
        part = {
            "name": label["name"],
            "size": label["size"],
            "sha256": label["sha256"],
        }
        path = download_one(part, output, base, attempts)
        actual_md5 = digest(path, "md5")
        if actual_md5 != label["md5"]:
            raise RuntimeError(f"MD5 mismatch for {path}: {actual_md5} != {label['md5']}")
        verified.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "md5": actual_md5,
                "sha256": digest(path),
                "samples": int(label["samples"]),
            }
        )
    return verified


def read_assigned_labels(labels_dir: Path) -> dict[str, str]:
    assigned: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_label.txt"
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if len(fields) != 3:
                raise RuntimeError(
                    f"{path}:{number}: expected three label fields"
                )
            sample_id = fields[0]
            if sample_id in assigned:
                raise RuntimeError(f"Duplicate assigned sample ID: {sample_id}")
            assigned[sample_id] = split
    return assigned


def archive_sample_id(member: str) -> str:
    path = PurePosixPath(member)
    if (
        len(path.parts) != 4
        or path.parts[0] != "MINIHARDVS_EVENT_files"
        or path.parts[3] != f"{path.parts[2]}.npz"
    ):
        raise RuntimeError(f"Unexpected HARDVS NPZ member layout: {member}")
    return f"{path.parts[1]}/{path.parts[2]}"


def verify_zip(
    manifest: dict[str, Any],
    archive: Path,
    *,
    labels_dir: Path,
    allow_known_corrupt_mirror: bool = False,
) -> dict[str, Any]:
    if archive.stat().st_size != int(manifest["archive_size"]):
        raise RuntimeError(
            f"archive-size mismatch: {archive.stat().st_size} "
            f"!= {manifest['archive_size']}"
        )
    archive_hashes = combined_digests(archive, ("md5", "sha256"))
    official_md5 = str(manifest["official_archive_md5"])
    mirror_md5 = str(manifest["known_mirror_archive_md5"])
    if archive_hashes["md5"] == official_md5:
        integrity_mode = "official_exact"
        expected_bad: set[str] = set()
        expected_missing_assigned: set[str] = set()
        expected_unassigned_count = int(manifest["official_unassigned_archive_npz"])
    elif archive_hashes["md5"] == mirror_md5:
        if not allow_known_corrupt_mirror:
            raise RuntimeError(
                "The pinned mirror is not byte-identical to the official HARDVS "
                f"archive (mirror MD5 {mirror_md5}; official MD5 {official_md5}) "
                "and leaves 13 assigned samples unavailable (six corrupt and "
                "seven absent). Supply an exact "
                "official archive, or rerun with --allow-known-corrupt-mirror "
                "to use the explicit exclusion protocol."
            )
        expected_sha256 = str(manifest["known_mirror_archive_sha256"])
        if archive_hashes["sha256"] != expected_sha256:
            raise RuntimeError(
                "Known-mirror SHA-256 mismatch: "
                f"{archive_hashes['sha256']} != {expected_sha256}"
            )
        integrity_mode = "known_mirror_exclusion"
        expected_bad = {
            str(row["member"]) for row in manifest["known_mirror_corrupt_samples"]
        }
        expected_missing_assigned = {
            str(row["sample_id"])
            for row in manifest["known_mirror_missing_assigned_samples"]
        }
        expected_unassigned_count = int(
            manifest["known_mirror_effective_counts"][
                "valid_unassigned_archive_npz"
            ]
        )
    else:
        raise RuntimeError(
            "Archive MD5 matches neither the exact official object nor the "
            f"pinned known mirror: {archive_hashes['md5']}"
        )

    bad: list[dict[str, str]] = []
    started = time.time()
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        npz_infos = [
            info for info in infos if not info.is_dir() and info.filename.lower().endswith(".npz")
        ]
        for index, info in enumerate(npz_infos, start=1):
            try:
                with handle.open(info) as source:
                    while source.read(BUFFER_SIZE):
                        pass
            except Exception as error:  # exact names are checked below
                row = {
                    "member": info.filename,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                bad.append(row)
                print(
                    f"[bad ZIP member] {info.filename}: "
                    f"{row['error_type']}: {row['error']}",
                    file=sys.stderr,
                    flush=True,
                )
            if index % 5000 == 0:
                print(
                    f"[full CRC] {index}/{len(npz_infos)} bad={len(bad)}",
                    flush=True,
                )
    npz_count = len(npz_infos)
    uncompressed = sum(info.file_size for info in infos)
    if len(infos) != int(manifest["archive_entries"]):
        raise RuntimeError(f"ZIP entry count mismatch: {len(infos)}")
    if npz_count != int(manifest["npz_entries"]):
        raise RuntimeError(f"ZIP NPZ count mismatch: {npz_count}")
    if uncompressed != int(manifest["uncompressed_bytes"]):
        raise RuntimeError(f"ZIP uncompressed-byte mismatch: {uncompressed}")
    actual_bad = {row["member"] for row in bad}
    if actual_bad != expected_bad:
        missing = sorted(expected_bad - actual_bad)
        unexpected = sorted(actual_bad - expected_bad)
        raise RuntimeError(
            "Full ZIP CRC inventory mismatch: "
            f"missing expected failures={missing}; unexpected failures={unexpected}"
        )
    assigned = read_assigned_labels(labels_dir)
    if len(assigned) != int(manifest["official_assigned_samples"]):
        raise RuntimeError(f"Assigned-label count mismatch: {len(assigned)}")
    archive_ids: dict[str, str] = {}
    for info in npz_infos:
        sample_id = archive_sample_id(info.filename)
        if sample_id in archive_ids:
            raise RuntimeError(f"Duplicate archive sample ID: {sample_id}")
        archive_ids[sample_id] = info.filename
    missing_assigned = set(assigned) - set(archive_ids)
    unassigned_archive = set(archive_ids) - set(assigned)
    if missing_assigned != expected_missing_assigned:
        raise RuntimeError(
            "Assigned-label/archive reconciliation mismatch: "
            f"missing expected={sorted(expected_missing_assigned - missing_assigned)}; "
            f"unexpected missing={sorted(missing_assigned - expected_missing_assigned)}"
        )
    bad_sample_ids = {archive_sample_id(member) for member in actual_bad}
    if not bad_sample_ids <= set(assigned):
        raise RuntimeError(
            f"Corrupt archive members are not assigned rows: {sorted(bad_sample_ids - set(assigned))}"
        )
    valid_unassigned = unassigned_archive - bad_sample_ids
    if len(valid_unassigned) != expected_unassigned_count:
        raise RuntimeError(
            "Valid unassigned archive count mismatch: "
            f"{len(valid_unassigned)} != {expected_unassigned_count}"
        )
    excluded_sample_ids = missing_assigned | bad_sample_ids
    effective_counts = {
        split: sum(
            assigned[sample_id] == split and sample_id not in excluded_sample_ids
            for sample_id in assigned
        )
        for split in ("train", "val", "test")
    }
    effective_counts["assigned"] = sum(effective_counts.values())
    if integrity_mode == "known_mirror_exclusion":
        expected_effective = {
            key: int(value)
            for key, value in manifest["known_mirror_effective_counts"].items()
            if key != "valid_unassigned_archive_npz"
        }
        if effective_counts != expected_effective:
            raise RuntimeError(
                f"Effective assigned counts mismatch: {effective_counts} != {expected_effective}"
            )
    elif effective_counts["assigned"] != int(manifest["official_assigned_samples"]):
        raise RuntimeError("Exact official archive does not preserve all assigned rows")
    print(
        f"[zip verified] mode={integrity_mode} entries={len(infos)} "
        f"npz={npz_count} bad={len(bad)} missing_assigned={len(missing_assigned)} "
        f"unassigned={len(valid_unassigned)} uncompressed={uncompressed}",
        flush=True,
    )
    return {
        "integrity_mode": integrity_mode,
        "archive_size": archive.stat().st_size,
        "archive_md5": archive_hashes["md5"],
        "archive_sha256": archive_hashes["sha256"],
        "archive_entries": len(infos),
        "npz_entries": npz_count,
        "uncompressed_bytes": uncompressed,
        "full_crc_members": len(npz_infos),
        "full_crc_failures": bad,
        "full_crc_seconds": time.time() - started,
        "label_reconciliation": {
            "assigned_label_rows": len(assigned),
            "archive_sample_ids": len(archive_ids),
            "missing_assigned_sample_ids": sorted(missing_assigned),
            "corrupt_assigned_sample_ids": sorted(bad_sample_ids),
            "excluded_assigned_sample_ids": sorted(excluded_sample_ids),
            "effective_counts": effective_counts,
            "valid_unassigned_archive_count": len(valid_unassigned),
            "valid_unassigned_archive_sample_ids": sorted(valid_unassigned),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_manifest = Path(__file__).resolve().parents[1] / "manifests" / "hardvs_compact_archive.json"
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--keep-chunks", action="store_true")
    parser.add_argument(
        "--allow-known-corrupt-mirror",
        action="store_true",
        help=(
            "Explicitly accept the pinned non-official mirror only after its "
            "combined hashes, complete six-member CRC failure set, and full "
            "label/archive reconciliation match the manifest. Preparation "
            "will exclude all 13 unavailable assigned samples."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    parts = sorted(manifest["parts"], key=lambda row: int(row["index"]))
    if [int(row["index"]) for row in parts] != list(range(len(parts))):
        raise RuntimeError("chunk indices are not contiguous")
    if sum(int(row["size"]) for row in parts) != int(manifest["archive_size"]):
        raise RuntimeError("chunk sizes do not sum to the declared archive size")

    output = args.output.resolve()
    chunks_dir = output / "chunks"
    labels_dir = output / "labels"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    revision = manifest["mirror_revision"]
    repo = manifest["mirror_repository"]
    base_url = (
        f"https://huggingface.co/datasets/{repo}/resolve/{revision}/HARDVS_Event"
    )
    archive = output / manifest["archive_name"]
    already_combined = valid_combined_parts(archive, parts)
    pending = parts[already_combined:]
    free = shutil.disk_usage(output).free
    required = sum(int(row["size"]) for row in pending) + 16 * 1024**3
    if free < required:
        raise RuntimeError(f"insufficient free space: {free} bytes; require at least {required}")
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(download_one, part, chunks_dir, base_url, args.attempts)
                for part in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        concatenate(archive, chunks_dir, parts, args.keep_chunks)
    label_records = download_labels(manifest, labels_dir, args.attempts)
    verification = verify_zip(
        manifest,
        archive,
        labels_dir=labels_dir,
        allow_known_corrupt_mirror=args.allow_known_corrupt_mirror,
    )
    record = {
        "schema_version": 3,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": digest(args.manifest.resolve()),
        "mirror_repository": repo,
        "mirror_revision": revision,
        "pending_chunk_count_at_start": len(pending),
        "pending_chunk_sha256_verification": (
            "each pending chunk was verified before concatenation"
            if pending
            else "not applicable: a complete combined archive was already present"
        ),
        "combined_archive_hashes_verified": True,
        "preexisting_combined_archive": not bool(pending),
        "labels": label_records,
        **verification,
    }
    record["record_hash"] = stable_hash(record)
    record_path = output / "acquisition_record.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[record] {record_path}", flush=True)
    print(f"[complete] {archive}", flush=True)


if __name__ == "__main__":
    main()
