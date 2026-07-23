"""Event-dataset indexing, split manifests, temporal binning, and corruptions."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import ConditionConfig, ExperimentConfig, SCHEMA_VERSION, stable_hash, utc_now, write_json

DATASET_META = {
    "synthetic": {"height": 16, "width": 16, "classes": 4},
    "cifar10_dvs": {"height": 128, "width": 128, "classes": 10},
    "dvs_gesture": {"height": 128, "width": 128, "classes": 11},
    "hardvs": {"height": 260, "width": 346, "classes": 300},
}
PERTURBATION_NAMESPACE_SEED = 20240723
FRAME_CACHE_SCHEMA = 2


def _record(sample_id: str, path: Path | None, label: int, group: str = "") -> dict[str, Any]:
    return {"id": sample_id, "path": None if path is None else str(path), "label": int(label), "group": group}


def _class_from_parent(path: Path) -> int:
    name = path.parent.name
    digits = "".join(character for character in name if character.isdigit())
    if digits:
        value = int(digits)
        return value - 1 if name.startswith("action_") else value
    classes = sorted(item.name for item in path.parent.parent.iterdir() if item.is_dir())
    return classes.index(name)


def _stratified_train_val(records: list[dict[str, Any]], fraction: float, seed: int) -> tuple[list, list]:
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[int(record["label"])].append(record)
    rng = np.random.default_rng(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for label in sorted(by_class):
        group = sorted(by_class[label], key=lambda row: row["id"])
        order = rng.permutation(len(group))
        val_count = max(1, int(round(len(group) * fraction)))
        val_indices = set(int(index) for index in order[:val_count])
        val.extend(row for index, row in enumerate(group) if index in val_indices)
        train.extend(row for index, row in enumerate(group) if index not in val_indices)
    return sorted(train, key=lambda row: row["id"]), sorted(val, key=lambda row: row["id"])


def _subject_train_val(records: list[dict[str, Any]], fraction: float, seed: int) -> tuple[list, list]:
    subjects = sorted({str(record["group"]) for record in records})
    if len(subjects) < 2:
        raise ValueError("DVS-Gesture subject-disjoint validation requires at least two training subjects")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subjects))
    count = max(1, int(round(len(subjects) * fraction)))
    val_subjects = {subjects[int(index)] for index in order[:count]}
    train = [record for record in records if record["group"] not in val_subjects]
    val = [record for record in records if record["group"] in val_subjects]
    expected = {int(record["label"]) for record in records}
    if {int(record["label"]) for record in train} != expected or {int(record["label"]) for record in val} != expected:
        raise ValueError("Subject split lost class coverage; choose a different split seed")
    return sorted(train, key=lambda row: row["id"]), sorted(val, key=lambda row: row["id"])


def _scan_cifar(root: Path, seed: int) -> dict[str, list[dict[str, Any]]]:
    base = root / "CIFAR10DVS" / "events_np"
    classes = sorted(path for path in base.iterdir() if path.is_dir() and path.name not in {"train", "test"})
    train_full: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label, class_dir in enumerate(classes):
        files = sorted(class_dir.glob("*.npz"))
        for path in files:
            suffix = path.stem.rsplit("_", 1)[-1]
            index = int(suffix)
            record = _record(f"{class_dir.name}/{path.name}", path.relative_to(root), label)
            (test if index < 100 else train_full).append(record)
    train, val = _stratified_train_val(train_full, 0.10, seed)
    return {"train": train, "val": val, "test": sorted(test, key=lambda row: row["id"])}


def _scan_gesture(root: Path, seed: int) -> dict[str, list[dict[str, Any]]]:
    base = root / "DVSGesture" / "download"
    train_full: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for split_dir, target in ((base / "ibmGestureTrain", train_full), (base / "ibmGestureTest", test)):
        for path in sorted(split_dir.glob("*/*.npy")):
            label = int(path.stem)
            subject = path.parent.name.split("_", 1)[0]
            target.append(_record(str(path.relative_to(base)), path.relative_to(root), label, subject))
    train, val = _subject_train_val(train_full, 0.20, seed)
    return {"train": train, "val": val, "test": sorted(test, key=lambda row: row["id"])}


def _find_hardvs_split(root: Path, split: str) -> Path:
    candidates = (
        root / "HARDVS" / "raw" / split,
        root / "HARDVS" / split,
        root / "hardvs" / "raw" / split,
        root / "hardvs" / split,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Missing HARDVS {split!r} directory. Expected an official split under one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _scan_hardvs(root: Path) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        split_root = _find_hardvs_split(root, split)
        rows = []
        for path in sorted(split_root.glob("*/*.npz")):
            rows.append(
                _record(
                    f"{split}/{path.parent.name}/{path.name}",
                    path.relative_to(root),
                    _class_from_parent(path),
                )
            )
        if not rows:
            raise FileNotFoundError(f"No HARDVS event files found below {split_root}")
        splits[split] = rows
    return splits


def _synthetic_splits(seed: int) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, count in (("train", 64), ("val", 24), ("test", 24)):
        splits[split] = [
            _record(f"synthetic/{split}/{index:04d}", None, index % DATASET_META["synthetic"]["classes"])
            for index in range(count)
        ]
    return splits


def prepare_split_manifest(dataset: str, data_root: Path, output: Path, *, seed: int = 20240723) -> dict[str, Any]:
    dataset_integrity: dict[str, Any] | None = None
    if dataset == "synthetic":
        splits = _synthetic_splits(seed)
    elif dataset == "cifar10_dvs":
        splits = _scan_cifar(data_root, seed)
    elif dataset == "dvs_gesture":
        splits = _scan_gesture(data_root, seed)
    elif dataset == "hardvs":
        splits = _scan_hardvs(data_root)
        integrity_path = data_root / "HARDVS" / "integrity_record.json"
        if not integrity_path.is_file():
            raise FileNotFoundError(
                "HARDVS requires the hash-protected integrity_record.json "
                "created by scripts/prepare_hardvs.py"
            )
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        claimed_hash = integrity.pop("record_hash", None)
        if claimed_hash != stable_hash(integrity):
            raise ValueError(f"HARDVS integrity-record hash mismatch: {integrity_path}")
        observed_counts = {name: len(rows) for name, rows in splits.items()}
        observed_with_total = {
            **observed_counts,
            "assigned": sum(observed_counts.values()),
        }
        if observed_with_total != integrity["splits"]["effective_counts"]:
            raise ValueError(
                "HARDVS materialized split counts disagree with the integrity record"
            )
        dataset_integrity = {
            "record_path": str(integrity_path.relative_to(data_root)),
            "record_hash": claimed_hash,
            "integrity_mode": integrity["integrity_mode"],
            "source_deviation": integrity["source_deviation"],
            "archive_md5": integrity["archive"]["md5"],
            "archive_sha256": integrity["archive"]["sha256"],
            "effective_counts": integrity["splits"]["effective_counts"],
            "excluded_sample_ids": integrity["splits"]["excluded_sample_ids"],
        }
    else:
        raise ValueError(dataset)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "data_root": ".",
        "data_root_note": "Paths are relative to ExperimentConfig.data_root; datasets are not redistributed.",
        "split_seed": seed,
        "created_at": utc_now(),
        "splits": splits,
    }
    if dataset_integrity is not None:
        manifest["dataset_integrity"] = dataset_integrity
    validate_split_manifest(manifest)
    hash_payload = {"dataset": dataset, "seed": seed, "splits": splits}
    if dataset_integrity is not None:
        hash_payload["dataset_integrity"] = dataset_integrity
    manifest["split_hash"] = stable_hash(hash_payload)
    write_json(output, manifest)
    return manifest


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    if set(manifest.get("splits", {})) != {"train", "val", "test"}:
        raise ValueError("Split manifest must contain exactly train, val, and test")
    sets = {name: {row["id"] for row in rows} for name, rows in manifest["splits"].items()}
    if any(not values for values in sets.values()):
        raise ValueError("Every split must be non-empty")
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Train, validation, and test sample identifiers must be disjoint")
    dataset = manifest["dataset"]
    expected = set(range(DATASET_META[dataset]["classes"]))
    for split, rows in manifest["splits"].items():
        labels = {int(row["label"]) for row in rows}
        if labels != expected:
            raise ValueError(f"{dataset}/{split} class coverage mismatch: expected {len(expected)}, got {len(labels)}")
    if dataset == "dvs_gesture":
        train_groups = {row["group"] for row in manifest["splits"]["train"]}
        val_groups = {row["group"] for row in manifest["splits"]["val"]}
        if train_groups & val_groups:
            raise ValueError("DVS-Gesture train and validation subjects overlap")
    if dataset == "hardvs":
        integrity = manifest.get("dataset_integrity")
        if not isinstance(integrity, dict):
            raise ValueError("HARDVS split manifests require dataset_integrity")
        if integrity.get("integrity_mode") not in {
            "official_exact",
            "known_corrupt_mirror_exclusion",
        }:
            raise ValueError("Unknown HARDVS integrity mode")
        counts = {
            name: len(manifest["splits"][name])
            for name in ("train", "val", "test")
        }
        observed = {**counts, "assigned": sum(counts.values())}
        if observed != integrity.get("effective_counts"):
            raise ValueError("HARDVS split counts disagree with dataset_integrity")
        if (
            integrity["integrity_mode"] == "known_corrupt_mirror_exclusion"
            and len(integrity.get("excluded_sample_ids", [])) != 6
        ):
            raise ValueError("Known-mirror HARDVS mode must record six exclusions")


def load_split_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    hash_payload = {
        "dataset": manifest["dataset"],
        "seed": manifest["split_seed"],
        "splits": manifest["splits"],
    }
    if "dataset_integrity" in manifest:
        hash_payload["dataset_integrity"] = manifest["dataset_integrity"]
    expected = stable_hash(hash_payload)
    if manifest.get("split_hash") != expected:
        raise ValueError(f"Split-manifest hash mismatch: {path}")
    return manifest


def _seed_for(sample_id: str, condition: ConditionConfig, base_seed: int) -> int:
    # Input perturbations are a fixed part of the evaluation/training cell and
    # are shared across model-initialization seeds. Randomness is determined
    # only by sample, condition, severity, and declared draw. This preserves a
    # paired design rather than silently assigning every trained checkpoint a
    # different corruption realization.
    _ = base_seed  # retained in the call signature for compatibility
    # The renormalized-thinning arm must use the *same retained events* as the
    # unrenormalized arm; only the post-binning amplitude multiplier may differ.
    # Using the literal condition name here would confound that control with a
    # second random subset.
    randomization_name = "event_thin" if condition.name == "event_thin_renorm" else condition.name
    value = (
        f"{PERTURBATION_NAMESPACE_SEED}|{sample_id}|{randomization_name}|"
        f"{condition.severity:.12g}|"
        f"{condition.draw}"
    )
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def _synthetic_seed_for(sample_id: str, base_seed: int) -> int:
    # The synthetic fixture is a dataset, not model-initialization noise.
    _ = base_seed
    value = f"{sample_id}|synthetic_source"
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def normalize_events(raw: Any, suffix: str = ".npz") -> dict[str, np.ndarray]:
    if suffix == ".npy":
        values = np.asarray(raw)
        if values.ndim != 2 or values.shape[1] < 4:
            raise ValueError(f"Expected [N,4] event array, got {values.shape}")
        return {
            "x": values[:, 0].astype(np.int64),
            "y": values[:, 1].astype(np.int64),
            "p": values[:, 2].astype(np.int64),
            "t": values[:, 3].astype(np.float64),
        }
    return {
        "x": np.asarray(raw["x"], dtype=np.int64),
        "y": np.asarray(raw["y"], dtype=np.int64),
        "p": np.asarray(raw["p"], dtype=np.int64),
        "t": np.asarray(raw["t"], dtype=np.float64),
    }


def corrupt_events(
    events: dict[str, np.ndarray], condition: ConditionConfig, rng: np.random.Generator, height: int, width: int
) -> tuple[dict[str, np.ndarray], float]:
    name = condition.name
    severity = float(condition.severity)
    copied = {key: np.asarray(value).copy() for key, value in events.items()}
    count = len(copied["t"])
    amplitude_scale = 1.0
    if name in {"clean", "temporal_sparse", "amplitude_scale", "reverse", "permute", "causal_prefix"}:
        return copied, severity if name == "amplitude_scale" else amplitude_scale
    if name in {"event_thin", "event_thin_renorm"}:
        keep_probability = severity
        keep = rng.random(count) < keep_probability
        copied = {key: value[keep] for key, value in copied.items()}
        if name == "event_thin_renorm" and keep_probability > 0:
            amplitude_scale = 1.0 / keep_probability
        return copied, amplitude_scale
    if name == "event_dropout":
        keep = rng.random(count) >= severity
        return {key: value[keep] for key, value in copied.items()}, amplitude_scale
    if count == 0:
        return copied, amplitude_scale
    t_min, t_max = float(copied["t"].min()), float(copied["t"].max())
    duration = max(t_max - t_min, 1.0)
    if name == "background_activity":
        add = int(round(count * severity))
        copied["x"] = np.concatenate((copied["x"], rng.integers(0, width, add)))
        copied["y"] = np.concatenate((copied["y"], rng.integers(0, height, add)))
        copied["p"] = np.concatenate((copied["p"], rng.integers(0, 2, add)))
        copied["t"] = np.concatenate((copied["t"], rng.uniform(t_min, t_max, add)))
    elif name == "polarity_flip":
        flip = rng.random(count) < severity
        copied["p"][flip] = 1 - copied["p"][flip]
    elif name == "temporal_jitter":
        copied["t"] = np.clip(copied["t"] + rng.normal(0.0, severity * duration, count), t_min, t_max)
    elif name == "packet_loss":
        span = severity * duration
        start = rng.uniform(t_min, max(t_min, t_max - span))
        keep = (copied["t"] < start) | (copied["t"] > start + span)
        copied = {key: value[keep] for key, value in copied.items()}
    elif name == "hot_pixels":
        add = int(round(count * severity))
        hot_count = max(1, int(round(height * width * min(severity, 1.0) * 0.001)))
        hot_x = rng.integers(0, width, hot_count)
        hot_y = rng.integers(0, height, hot_count)
        selected = rng.integers(0, hot_count, add)
        copied["x"] = np.concatenate((copied["x"], hot_x[selected]))
        copied["y"] = np.concatenate((copied["y"], hot_y[selected]))
        copied["p"] = np.concatenate((copied["p"], rng.integers(0, 2, add)))
        copied["t"] = np.concatenate((copied["t"], rng.uniform(t_min, t_max, add)))
    else:
        raise ValueError(f"Unknown event condition: {name}")
    order = np.argsort(copied["t"], kind="stable")
    return {key: value[order] for key, value in copied.items()}, amplitude_scale


def events_to_frames(
    events: dict[str, np.ndarray],
    time_steps: int,
    height: int,
    width: int,
    *,
    time_bounds: tuple[float, float] | None = None,
) -> torch.Tensor:
    frames = np.zeros((time_steps, 2, height, width), dtype=np.float32)
    if len(events["t"]) == 0:
        return torch.from_numpy(frames)
    x = np.clip(events["x"].astype(np.int64), 0, width - 1)
    y = np.clip(events["y"].astype(np.int64), 0, height - 1)
    p = np.clip(events["p"].astype(np.int64), 0, 1)
    timestamps = events["t"].astype(np.float64)
    low, high = (
        (float(timestamps.min()), float(timestamps.max()))
        if time_bounds is None
        else (float(time_bounds[0]), float(time_bounds[1]))
    )
    if high <= low:
        bins = np.zeros_like(x)
    else:
        bins = np.floor((timestamps - low) / (high - low + 1e-12) * time_steps).astype(np.int64)
        bins = np.clip(bins, 0, time_steps - 1)
    np.add.at(frames, (bins, p, y, x), 1.0)
    return torch.from_numpy(frames)


def letterbox(frames: torch.Tensor, size: int) -> torch.Tensor:
    height, width = frames.shape[-2:]
    scale = min(size / height, size / width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    resized = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left
    return F.pad(resized, (left, right, top, bottom))


def frame_condition(frames: torch.Tensor, condition: ConditionConfig, rng: np.random.Generator) -> torch.Tensor:
    output = frames.clone()
    name, severity = condition.name, float(condition.severity)
    if name == "temporal_sparse":
        keep = torch.from_numpy(rng.random(output.shape[0]) < severity)
        output[~keep] = 0
    elif name == "amplitude_scale":
        output.mul_(severity)
    elif name == "reverse":
        output = output.flip(0)
    elif name == "permute":
        output = output[torch.from_numpy(rng.permutation(output.shape[0])).long()]
    elif name == "causal_prefix":
        kept = max(1, int(round(output.shape[0] * severity)))
        output[kept:] = 0
    return output


def cache_source_condition(condition: ConditionConfig) -> ConditionConfig | None:
    """Return the raw-event transform whose binned frames can be reused.

    Frame-only controls are derived losslessly from clean cached frames.
    Renormalized thinning reuses the exact unscaled frames from its matched
    unrenormalized thinning arm. Raw-event sensor corruptions use their own
    condition/draw namespace so an expensive event file is decoded and
    corrupted once, then reused identically across matched models.
    """

    if condition.name in {
        "clean",
        "amplitude_scale",
        "temporal_sparse",
        "reverse",
        "permute",
        "causal_prefix",
    }:
        return ConditionConfig(name="clean", severity=0.0, draw=0)
    if condition.name in {"event_thin", "event_thin_renorm"}:
        return ConditionConfig(
            name="event_thin",
            severity=condition.severity,
            draw=condition.draw,
        )
    if condition.name in {
        "event_dropout",
        "background_activity",
        "polarity_flip",
        "temporal_jitter",
        "packet_loss",
        "hot_pixels",
    }:
        return condition
    return None


def frame_cache_namespace(
    config: ExperimentConfig, condition: ConditionConfig | None = None
) -> Path | None:
    selected = config.condition if condition is None else condition
    source = cache_source_condition(selected)
    if not config.cache_frames or source is None:
        return None
    seed = 0
    condition = f"{source.name}_{source.severity:.12g}_draw{source.draw}_seed{seed}"
    return (
        Path(str(config.frame_cache_root))
        / f"schema_v{FRAME_CACHE_SCHEMA}"
        / config.dataset
        / f"T{config.time_steps}_S{config.image_size}"
        / condition
    )


def _cache_key(config: ExperimentConfig, source: ConditionConfig, sample_id: str) -> Path:
    namespace = frame_cache_namespace(config, source)
    if namespace is None:
        raise RuntimeError("frame cache key requested for an uncacheable condition")
    identifier_hash = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return (
        namespace
        / identifier_hash[:2]
        / f"{identifier_hash}.npz"
    )


def _read_frame_cache(path: Path, expected_shape: tuple[int, ...]) -> tuple[torch.Tensor, float, float]:
    with np.load(path, allow_pickle=False) as cached:
        frames = np.asarray(cached["frames"], dtype=np.float32)
        event_count = float(cached["event_count"])
        duration = float(cached["duration"])
    if tuple(frames.shape) != expected_shape:
        raise ValueError(f"Cached-frame shape mismatch in {path}: {frames.shape} != {expected_shape}")
    return torch.from_numpy(frames), event_count, duration


def _write_frame_cache(path: Path, frames: torch.Tensor, event_count: float, duration: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            frames=frames.numpy().astype(np.float32, copy=False),
            event_count=np.asarray(event_count, dtype=np.float64),
            duration=np.asarray(duration, dtype=np.float64),
        )
    os.replace(temporary, path)


class EventManifestDataset(Dataset):
    def __init__(
        self,
        config: ExperimentConfig,
        manifest: dict[str, Any],
        split: str,
        *,
        condition: ConditionConfig | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.config = config
        self.root = Path(config.data_root)
        self.records = list(manifest["splits"][split])
        if config.smoke_samples > 0:
            self.records = self.records[: config.smoke_samples]
        self.condition = config.condition if condition is None else condition
        self.meta = DATASET_META[config.dataset]

    def __len__(self) -> int:
        return len(self.records)

    def _synthetic_events(self, record: dict[str, Any], rng: np.random.Generator) -> dict[str, np.ndarray]:
        label = int(record["label"])
        count = 96 + label * 8
        t = np.sort(rng.uniform(0, 1_000_000, count))
        center_x = 3 + label * 3
        center_y = 4 + label * 2
        x = np.clip(np.rint(rng.normal(center_x, 1.5, count)), 0, self.meta["width"] - 1).astype(np.int64)
        y = np.clip(np.rint(rng.normal(center_y, 1.5, count)), 0, self.meta["height"] - 1).astype(np.int64)
        p = ((np.arange(count) + label) % 2).astype(np.int64)
        return {"x": x, "y": y, "p": p, "t": t}

    def __getitem__(self, index: int):
        record = self.records[index]
        source_condition = cache_source_condition(self.condition)
        cache_path = (
            _cache_key(self.config, source_condition, record["id"])
            if self.config.cache_frames and source_condition is not None
            else None
        )
        expected_shape = (self.config.time_steps, 2, self.config.image_size, self.config.image_size)
        if cache_path is not None and cache_path.is_file():
            frames, event_count, duration = _read_frame_cache(cache_path, expected_shape)
        else:
            event_condition = self.condition if source_condition is None else source_condition
            rng = np.random.default_rng(_seed_for(record["id"], event_condition, self.config.seed))
            if self.config.dataset == "synthetic":
                events = self._synthetic_events(
                    record, np.random.default_rng(_synthetic_seed_for(record["id"], self.config.seed))
                )
            else:
                path = self.root / record["path"]
                raw = np.load(path, mmap_mode="r" if path.suffix == ".npy" else None)
                try:
                    events = normalize_events(raw, path.suffix)
                finally:
                    close = getattr(raw, "close", None)
                    if callable(close):
                        close()
            original_bounds = (
                (float(events["t"].min()), float(events["t"].max()))
                if len(events["t"])
                else (0.0, 0.0)
            )
            events, _source_scale = corrupt_events(
                events,
                event_condition,
                rng,
                int(self.meta["height"]),
                int(self.meta["width"]),
            )
            event_count = float(len(events["t"]))
            duration = original_bounds[1] - original_bounds[0]
            frames = events_to_frames(
                events,
                self.config.time_steps,
                int(self.meta["height"]),
                int(self.meta["width"]),
                time_bounds=original_bounds,
            )
            frames = letterbox(frames, self.config.image_size)
            if cache_path is not None:
                _write_frame_cache(cache_path, frames, event_count, duration)

        rng = np.random.default_rng(_seed_for(record["id"], self.condition, self.config.seed))
        if self.condition.name == "event_thin_renorm" and self.condition.severity > 0:
            frames.mul_(1.0 / self.condition.severity)
        frames = frame_condition(frames, self.condition, rng)
        statistics = {
            "event_count": float(event_count),
            "active_bin_fraction": float((frames != 0).float().mean().item()),
            "event_mass": float(frames.sum().item()),
            "duration": duration,
        }
        return frames, int(record["label"]), record["id"], statistics


def build_dataset(config: ExperimentConfig, split: str, condition: ConditionConfig | None = None) -> tuple[Dataset, dict]:
    manifest = load_split_manifest(Path(config.split_manifest))
    if manifest["dataset"] != config.dataset:
        raise ValueError(f"Config dataset {config.dataset!r} does not match split manifest {manifest['dataset']!r}")
    return EventManifestDataset(config, manifest, split, condition=condition), manifest
