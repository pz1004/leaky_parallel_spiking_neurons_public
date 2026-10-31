"""Portable locations for new reproductions, separate from historical records.

Importing this module does not create directories or open datasets. Commands
are normally run from the repository root. Set LPSN_EVIDENCE_ROOT to place new
measurement and analysis outputs elsewhere. Historical records keep their
original paths and hashes; they are never rewritten by this helper.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL = PROJECT  # The retained baseline and extension share one checkout.


def resolve_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    """Resolve a recorded path without changing its provenance or searching.

    Absolute paths are respected. Relative paths use the public repository
    root unless a caller supplies an explicit base. A missing input remains
    missing, rather than silently selecting a different checkpoint or record.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(base if base is not None else PROJECT) / path


EVIDENCE = resolve_path(os.environ.get("LPSN_EVIDENCE_ROOT", "reproduction_outputs"))


def portable_path(value: str | Path) -> str:
    """Use repository-relative provenance when possible; retain external roots."""
    path = resolve_path(value).resolve()
    try:
        return path.relative_to(PROJECT).as_posix()
    except ValueError:
        return str(path)
