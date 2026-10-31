#!/usr/bin/env python3
"""Lightweight environment check for the LP-SN reference runtime."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    expected: str
    actual: str

    @property
    def ok(self) -> bool:
        return self.actual.startswith(self.expected)


EXPECTED = {
    "python": "3.12",
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "cuda": "12.8",
    "cudnn": "91002",
}
EXPECTED_SPIKINGJELLY_COMMIT = "9b87d38"
EXPECTED_SPIKINGJELLY_ARCHIVE_SHA256 = "156a166fe3cf37785930207408789d9d9e80f980043323fbb082693c1688b7e2"


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"missing ({exc})"
    return str(getattr(module, "__version__", "unknown"))


def spikingjelly_revision() -> tuple[str, str]:
    """Return the installed SpikingJelly commit and archive hash.

    A PEP 610 ``direct_url.json`` record stores VCS metadata for a git
    install and archive metadata for an immutable source-archive install.
    The latter has no ``.git`` directory, so recover its full commit from the
    pinned archive URL and verify the recorded source hash separately.
    """
    try:
        dist = metadata.distribution("spikingjelly")
        for file in dist.files or []:
            if str(file).endswith("direct_url.json"):
                direct_url = json.loads(dist.locate_file(file).read_text(encoding="utf-8"))
                commit = direct_url.get("vcs_info", {}).get("commit_id")
                if commit:
                    return str(commit), "vcs"
                url = str(direct_url.get("url", ""))
                archive_hashes = direct_url.get("archive_info", {}).get("hashes", {})
                archive_hash = str(archive_hashes.get("sha256", "unknown"))
                marker = "/archive/"
                if marker in url:
                    commit = url.split(marker, 1)[1].split(".", 1)[0]
                    return commit, archive_hash
    except Exception:
        pass

    module = importlib.import_module("spikingjelly")
    module_file = Path(getattr(module, "__file__", "")).resolve()
    for parent in (module_file.parent, *module_file.parents):
        if (parent / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except Exception as exc:
                return f"unknown (git probe failed: {exc})", "unknown"
            commit = result.stdout.strip()
            if result.returncode == 0 and commit:
                return commit, "vcs"
            return f"unknown (git probe failed: {result.stderr.strip()})", "unknown"
    return f"unknown (file {module_file})", "unknown"


def collect_checks() -> list[Check]:
    checks = [
        Check("python", EXPECTED["python"], platform.python_version()),
        Check("torch", EXPECTED["torch"], module_version("torch")),
        Check("torchvision", EXPECTED["torchvision"], module_version("torchvision")),
    ]

    try:
        import torch

        checks.append(Check("cuda", EXPECTED["cuda"], str(torch.version.cuda)))
    except Exception as exc:
        checks.append(Check("cuda", EXPECTED["cuda"], f"unavailable ({exc})"))

    try:
        import torch

        checks.append(Check("cudnn", EXPECTED["cudnn"], str(torch.backends.cudnn.version())))
    except Exception as exc:
        checks.append(Check("cudnn", EXPECTED["cudnn"], f"unavailable ({exc})"))

    try:
        spikingjelly_commit, spikingjelly_archive_hash = spikingjelly_revision()
        checks.append(Check("spikingjelly_commit", EXPECTED_SPIKINGJELLY_COMMIT, spikingjelly_commit))
        if spikingjelly_archive_hash != "vcs":
            checks.append(
                Check(
                    "spikingjelly_archive_sha256",
                    EXPECTED_SPIKINGJELLY_ARCHIVE_SHA256,
                    spikingjelly_archive_hash,
                )
            )
    except Exception as exc:
        checks.append(Check("spikingjelly_commit", EXPECTED_SPIKINGJELLY_COMMIT, f"missing ({exc})"))

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any recorded runtime field differs.",
    )
    args = parser.parse_args(argv)

    checks = collect_checks()
    failures = [check for check in checks if not check.ok]
    for check in checks:
        status = "OK" if check.ok else "MISMATCH"
        print(f"{status:8s} {check.name}: expected {check.expected}, actual {check.actual}")

    try:
        import spikingjelly

        sj_file = getattr(spikingjelly, "__file__", "unknown")
        sj_version = getattr(spikingjelly, "__version__", "unknown")
        print(f"INFO     spikingjelly: version {sj_version}, file {sj_file}")
    except Exception:
        pass

    if failures and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
