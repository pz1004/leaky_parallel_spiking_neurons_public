"""Expand a declarative experiment matrix into YAML configurations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _slug(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def expand(spec: dict[str, Any], data_root: Path, manifests_root: Path, output: Path) -> int:
    defaults = dict(spec["defaults"])
    count = 0
    for study_name, study in spec["studies"].items():
        conditions = study.get("conditions", [{"name": "clean", "severity": 0.0}])
        for dataset in study["datasets"]:
            dataset_values = dict(spec.get("dataset_overrides", {}).get(dataset, {}))
            for model in study["models"]:
                for time_steps in study.get("time_steps_by_dataset", {}).get(dataset, study["time_steps"]):
                    for seed in study["seeds"]:
                        for condition in conditions:
                            condition_name = condition["name"]
                            severity = float(condition.get("severity", 0.0))
                            experiment_id = study_name
                            if condition_name != "clean":
                                experiment_id += f"_{condition_name}_{_slug(severity)}"
                            values = {
                                **defaults,
                                **dataset_values,
                                "schema_version": 1,
                                "experiment_id": experiment_id,
                                "dataset": dataset,
                                "model": model,
                                "seed": seed,
                                "time_steps": time_steps,
                                "data_root": str(data_root),
                                "split_manifest": str(manifests_root / f"{dataset}.json"),
                                "condition": {
                                    "name": condition_name,
                                    "severity": severity,
                                    "draw": 0,
                                    "renormalize": condition_name == "event_thin_renorm",
                                },
                            }
                            if values.get("cache_frames"):
                                values["frame_cache_root"] = str(data_root / ".lpsn_frame_cache")
                            target = output / experiment_id / dataset / model / f"T{time_steps}" / f"seed_{seed}.yml"
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(yaml.safe_dump(values, sort_keys=True), encoding="utf-8")
                            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifests-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    count = expand(spec, args.data_root.resolve(), args.manifests_root.resolve(), args.output)
    print(f"wrote {count} configurations below {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
