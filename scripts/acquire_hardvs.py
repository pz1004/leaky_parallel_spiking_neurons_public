#!/usr/bin/env python3
"""Acquire and verify the full HARDVS compact-event archive.

The official project links a single 244.899 GB Baidu object. A public mirror
stores that exact-size object as 23 ordered binary chunks. This script downloads
the pinned chunks concurrently, verifies every SHA-256 digest, concatenates them
in order, and verifies the resulting ZIP central directory. It is resumable:
partial HTTP downloads and a boundary-aligned combined archive are reused.
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
from pathlib import Path
from typing import Any

BUFFER_SIZE = 8 * 1024 * 1024
USER_AGENT = "LP-SN reproducibility artifact HARDVS acquisition/1.0"


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(BUFFER_SIZE):
            value.update(block)
    return value.hexdigest()


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


def download_labels(manifest: dict[str, Any], output: Path, attempts: int) -> None:
    repo = manifest["mirror_repository"]
    base = f"https://huggingface.co/datasets/{repo}/resolve/main"
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


def verify_zip(manifest: dict[str, Any], archive: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
    npz_count = sum(info.filename.lower().endswith(".npz") for info in infos)
    uncompressed = sum(info.file_size for info in infos)
    if len(infos) != int(manifest["archive_entries"]):
        raise RuntimeError(f"ZIP entry count mismatch: {len(infos)}")
    if npz_count != int(manifest["npz_entries"]):
        raise RuntimeError(f"ZIP NPZ count mismatch: {npz_count}")
    if uncompressed != int(manifest["uncompressed_bytes"]):
        raise RuntimeError(f"ZIP uncompressed-byte mismatch: {uncompressed}")
    print(
        f"[zip verified] entries={len(infos)} npz={npz_count} "
        f"uncompressed={uncompressed}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    default_manifest = Path(__file__).resolve().parents[1] / "manifests" / "hardvs_compact_archive.json"
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--keep-chunks", action="store_true")
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
    free = shutil.disk_usage(output).free
    required = int(manifest["archive_size"]) + 16 * 1024**3
    if free < required:
        raise RuntimeError(f"insufficient free space: {free} bytes; require at least {required}")

    revision = manifest["mirror_revision"]
    repo = manifest["mirror_repository"]
    base_url = (
        f"https://huggingface.co/datasets/{repo}/resolve/{revision}/HARDVS_Event"
    )
    archive = output / manifest["archive_name"]
    already_combined = valid_combined_parts(archive, parts)
    pending = parts[already_combined:]
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(download_one, part, chunks_dir, base_url, args.attempts)
                for part in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        concatenate(archive, chunks_dir, parts, args.keep_chunks)
    download_labels(manifest, labels_dir, args.attempts)
    verify_zip(manifest, archive)
    print(f"[complete] {archive}", flush=True)


if __name__ == "__main__":
    main()
