"""Normalize saved layer counters; no dataset reads or model inference.

Require all 45 validation checkpoint records. Normalize within each layer and
seed before equally averaging the five convolutional layers, then average
across the five training seeds. Hidden and output layers remain separate.
"""
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.release_paths import EVIDENCE, portable_path

DATASETS = ('cifar10_dvs', 'dvs_gesture', 'hardvs')
SEEDS = {42, 123, 256, 512, 1024}
LAYERS = ['network.2', 'network.6', 'network.10', 'network.14', 'network.18', 'network.23', 'network.26']


def write_csv(name, rows):
    path = EVIDENCE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_layers():
    per_layer = []
    per_seed = []
    provenance = []
    seen = set()
    for path in sorted((EVIDENCE / "diagnostics").glob("layers_*.json")):
        record = json.loads(path.read_text())
        identity = record["dataset"], record["model"], record["seed"]
        assert identity not in seen
        seen.add(identity)
        assert record["time_steps"] == 16 and record["split"] == "validation"
        assert set(record["layers"]) == set(LAYERS)
        normalized = []
        for name in LAYERS:
            counts = record["layers"][name]
            n, units = counts["elements"], counts["units"]
            if units <= 0 or n <= 0 or counts['consecutive_pairs'] != 15 * units:
                raise ValueError(f'Invalid or empty layer denominators: {path.name}/{name}')
            if len(counts['histogram']) != 18 or any(value < 0 for value in counts['histogram']):
                raise ValueError(f'Invalid membrane histogram: {path.name}/{name}')
            assert n == 16 * units and sum(counts["histogram"]) == n
            row = dict(dataset=record["dataset"], model=record["model"], seed=record["seed"], layer=name,
                       firing=counts["spikes"] / n,
                       duty_half=counts["sustained_half"] / units,
                       duty_nine_tenths=counts["sustained_nine_tenths"] / units,
                       adjacent_spiking=counts["consecutive_spikes"] / counts["consecutive_pairs"],
                       mean_margin=counts["margin_sum"] / n,
                       mean_correction=counts["correction_sum"] / n,
                       margin_le_minus8=counts["histogram"][0] / n,
                       margin_gt8=counts["histogram"][-1] / n)
            normalized.append(row)
            per_layer.append(row)
        summary = dict(dataset=record["dataset"], model=record["model"], seed=record["seed"])
        for metric in ["firing", "duty_half", "duty_nine_tenths", "adjacent_spiking", "mean_margin", "mean_correction"]:
            summary["conv_" + metric] = mean(row[metric] for row in normalized[:5])
            summary["hidden_" + metric] = normalized[5][metric]
            summary["output_" + metric] = normalized[6][metric]
        per_seed.append(summary)
        provenance.append(dict(path=portable_path(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    expected = {(dataset, model, seed) for dataset in DATASETS
                for model in ["psn", "lpsn_a1_lif_init", "lpsn_a2_soft"] for seed in SEEDS}
    assert seen == expected, f"Incomplete layer evidence: {expected - seen}"
    groups = defaultdict(list)
    for row in per_seed:
        groups[row["dataset"], row["model"]].append(row)
    summaries = []
    for dataset in DATASETS:
        for model in ["psn", "lpsn_a1_lif_init", "lpsn_a2_soft"]:
            rows = groups[dataset, model]
            out = dict(dataset=dataset, model=model, seeds=len(rows))
            for metric in rows[0]:
                if metric not in {"dataset", "model", "seed"}:
                    out[metric] = mean(row[metric] for row in rows)
                    out[metric + "_sd"] = stdev(row[metric] for row in rows)
            summaries.append(out)
    write_csv("diagnostics/layer_normalized_per_seed.csv", per_layer)
    write_csv("diagnostics/layer_seed_summary.csv", per_seed)
    write_csv("diagnostics/layer_summary.csv", summaries)
    (EVIDENCE / "diagnostics/layer_summary_sources.json").write_text(json.dumps({
        "checkpoints": len(seen), "layer_records": len(per_layer), "sources": provenance,
        "aggregation": "Normalize within each layer and seed; equal-weight the five convolutional layers; average five seeds. Hidden and output layers remain separate.",
        "inference": "Descriptive frozen-validation summaries, without causal or independent-example inference."
    }, indent=2) + "\n")
    return summaries


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    rows = summarize_layers()
    print(json.dumps({'checkpoint_records': 45, 'layer_records': 315,
                      'model_dataset_summaries': len(rows),
                      'output': str(EVIDENCE / 'diagnostics')}))


if __name__ == '__main__':
    main()
