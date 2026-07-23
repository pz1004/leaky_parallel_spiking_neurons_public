"""Versioned configuration and manifest schemas.

The training and locked-test commands deliberately use different entry points.
This makes accidental test-set checkpoint selection structurally difficult and
leaves an auditable marker when the test set has been evaluated.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

SCHEMA_VERSION = 1
SOURCE_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DATASETS = {"synthetic", "cifar10_dvs", "dvs_gesture", "hardvs"}
ALLOWED_MODELS = {
    "lif",
    "psn",
    "masked_psn",
    "sliding_psn",
    "rpsu",
    "lpsn_a1_lif_init",
    "lpsn_a2_soft",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ConditionConfig:
    """Input condition used for matched retraining or evaluation."""

    name: str = "clean"
    severity: float = 0.0
    draw: int = 0
    renormalize: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ConditionConfig":
        return cls(**({} if value is None else dict(value)))


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    dataset: str
    model: str
    seed: int
    time_steps: int
    data_root: str
    split_manifest: str
    output_root: str = "runs_revision"
    image_size: int = 128
    architecture: str = "vgg_small"
    epochs: int = 200
    patience: int = 15
    batch_size: int = 16
    effective_batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    amp: bool = False
    device: str = "cuda:0"
    selection_metric: str = "macro_f1"
    condition: ConditionConfig = field(default_factory=ConditionConfig)
    model_params: dict[str, Any] = field(default_factory=dict)
    smoke_samples: int = 0
    cache_frames: bool = False
    frame_cache_root: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        values = dict(raw)
        values["condition"] = ConditionConfig.from_mapping(values.get("condition"))
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Expected schema_version={SCHEMA_VERSION}, got {self.schema_version}")
        if self.dataset not in ALLOWED_DATASETS:
            raise ValueError(f"Unsupported dataset {self.dataset!r}; choose from {sorted(ALLOWED_DATASETS)}")
        if self.model not in ALLOWED_MODELS:
            raise ValueError(f"Unsupported model {self.model!r}; choose from {sorted(ALLOWED_MODELS)}")
        if self.time_steps not in {8, 16, 32, 64}:
            raise ValueError("time_steps must be one of 8, 16, 32, or 64")
        if self.image_size <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("image_size, epochs, and batch_size must be positive")
        if self.effective_batch_size < self.batch_size or self.effective_batch_size % self.batch_size:
            raise ValueError("effective_batch_size must be an integer multiple of batch_size")
        if self.selection_metric not in {"macro_f1", "accuracy"}:
            raise ValueError("selection_metric must be macro_f1 or accuracy")
        if not 0.0 <= self.condition.severity <= 1.0:
            raise ValueError("condition.severity must be in [0, 1]")
        if self.cache_frames and not self.frame_cache_root:
            raise ValueError("frame_cache_root is required when cache_frames is enabled")

    @property
    def accumulation_steps(self) -> int:
        return self.effective_batch_size // self.batch_size

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.experiment_id / self.dataset / self.model / f"T{self.time_steps}" / f"seed_{self.seed}"

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    created_at: str
    phase: str
    experiment_id: str
    config_hash: str
    split_hash: str
    git_commit: str
    git_dirty: bool
    python: str
    torch: str
    cuda: str | None
    device: str
    dataset: str
    model: str
    seed: int
    time_steps: int
    condition: dict[str, Any]
    checkpoint_epoch: int | None = None
    test_evaluated_at: str | None = None


@dataclass(frozen=True)
class MetricsRecord:
    schema_version: int
    split: str
    dataset: str
    model: str
    seed: int
    time_steps: int
    condition: str
    severity: float
    draw: int
    epoch: int | None
    samples: int
    loss: float
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    balanced_accuracy: float
    firing_rate: float
    seconds: float
    throughput: float
    peak_cuda_memory_mb: float
    mean_event_count: float
    mean_active_bin_fraction: float
    mean_event_mass: float
    mean_duration: float
    layer_firing_rates: dict[str, float] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return ExperimentConfig.from_mapping(raw)


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SOURCE_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=SOURCE_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.returncode != 0 or bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return True


def runtime_manifest(config: ExperimentConfig, *, phase: str, split_hash: str) -> RunManifest:
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        created_at=utc_now(),
        phase=phase,
        experiment_id=config.experiment_id,
        config_hash=config.config_hash,
        split_hash=split_hash,
        git_commit=git_commit(),
        git_dirty=git_dirty(),
        python=sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=config.device,
        dataset=config.dataset,
        model=config.model,
        seed=config.seed,
        time_steps=config.time_steps,
        condition=asdict(config.condition),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def environment_summary() -> dict[str, Any]:
    try:
        cudnn: Any = torch.backends.cudnn.version()
    except RuntimeError as exc:
        cudnn = {"error": str(exc)}
    devices: list[Any] = []
    for index in range(torch.cuda.device_count()):
        try:
            devices.append(torch.cuda.get_device_name(index))
        except RuntimeError as exc:
            devices.append({"index": index, "error": str(exc)})
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": cudnn,
        "cuda_available": torch.cuda.is_available(),
        "devices": devices,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
