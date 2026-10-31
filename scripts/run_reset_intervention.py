"""Full-validation fixed-weight reset intervention; no training or test reads.

The design and source/checkpoint hashes are saved before dataset inference.
Outputs are new evidence; the training checkouts and checkpoints stay unchanged.
"""
from __future__ import annotations
import argparse
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from dataclasses import replace
import numpy as np
import torch
from spikingjelly.activation_based import functional

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.config import file_sha256, stable_hash, utc_now, environment_summary
from experiments.cli import _load_checkpoint
from experiments.data import build_dataset, DATASET_META
from experiments.engine import make_loader, seed_everything, confusion_metrics
from scripts.measure_gpu_energy import config_for, check_no_training, source_snapshot
from experiments.release_paths import EVIDENCE
from lpsn_neuron import LPSpikingNeuron
from lpsn_variants import build_lpsn_neuron

OUT = EVIDENCE / "reset_intervention"
SEEDS = [42, 123, 256, 512, 1024]
DATASETS = ["cifar10_dvs", "dvs_gesture", "hardvs"]
MODELS = ["lpsn_a1_lif_init", "lpsn_a2_soft"]

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def set_reset(model, enabled):
    count = 0
    for node in model.modules():
        if isinstance(node, LPSpikingNeuron):
            assert not any(p.requires_grad for p in node.parameters())
            assert not torch.count_nonzero(node.delta_W)
            assert torch.all(node.bias == -node.v_threshold)
            if enabled and not hasattr(node, "beta"):
                node.register_buffer("beta", node.alpha.new_tensor(node.v_threshold))
            node.soft_reset_strategy = "exact_detached" if enabled else "none"
            node.soft_reset = enabled
            count += 1
    return count

def synthetic_checks():
    records = []
    for t in [8, 16, 32]:
        for pattern in ["mixed", "nonnegative", "subthreshold"]:
            torch.manual_seed(832 + t)
            x = torch.randn(t, 2, 7, dtype=torch.float64)
            x = {"mixed": x * 1.7 + 1.2, "nonnegative": x.abs() * 2.5,
                 "subthreshold": x * .1}[pattern]
            for source_name, target_name, enabled in [
                (MODELS[0], MODELS[1], True), (MODELS[1], MODELS[0], False)
            ]:
                source = build_lpsn_neuron(T=t, variant=source_name).double().eval()
                target = build_lpsn_neuron(T=t, variant=target_name).double().eval()
                original = source(x).detach().clone()
                switched = copy.deepcopy(source); set_reset(switched, enabled)
                with torch.no_grad():
                    y = switched(x); ref = target(x)
                    logits = switched.forward_logits(x); ref_logits = target.forward_logits(x)
                set_reset(switched, not enabled)
                records.append({"T": t, "pattern": pattern, "source": source_name,
                    "target": target_name, "output_exact": bool(torch.equal(y, ref)),
                    "logits_exact": bool(torch.equal(logits, ref_logits)),
                    "roundtrip_output_exact": bool(torch.equal(switched(x), original))})
    assert all(r["output_exact"] and r["logits_exact"] and r["roundtrip_output_exact"] for r in records)
    save("synthetic_checks.json", {"records": records, "all_pass": True})

def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Outputs: LPSN_EVIDENCE_ROOT/reset_intervention. This command performs full-validation GPU inference when invoked.')
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise RuntimeError("Refusing to overwrite a started intervention; inspect existing evidence")
    lock = (OUT.parent / "gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_no_training(); torch.set_num_threads(8); seed_everything(20261030)
    source = source_snapshot(OUT, __file__)
    configs = {}
    sources = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for model in MODELS:
                cfg, path = config_for(dataset, model, seed, t=16)
                cfg = replace(cfg, cache_frames=False,
                              num_workers=4, batch_size=16, device="cuda:0")
                checkpoint = cfg.run_dir / "checkpoint_best.pt"
                sources.append({"dataset": dataset, "seed": seed, "trained_model": model,
                    "config_path": str(path), "config_sha256": file_sha256(path),
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)})
                configs[dataset, seed, model] = cfg
    protocol = {"created_at_before_inference": utc_now(), "script_sha256": file_sha256(Path(__file__)), "source_manifest": source,
        "environment": environment_summary(), "gpu_snapshot": subprocess.check_output(["nvidia-smi"], text=True),
        "datasets": DATASETS, "seeds": SEEDS, "trained_models": MODELS, "T": 16,
        "split": "validation", "sampling": "complete frozen validation split in manifest order",
        "batch_size": 16, "models_in_eval_mode": True, "gradient_enabled": False,
        "intervention": "Switch reset at all seven LP-SN layers: A1 off to exact soft reset on; A2 on to off. All existing parameters and normalization buffers unchanged.",
        "baseline": "Unmodified selected checkpoint, evaluated on the same validation inputs",
        "estimand": "Within-checkpoint validation performance effect of an inference-time reset switch, conditional on the already fitted spatial weights and normalization state",
        "limits": "Fixed-weight inference sensitivity does not isolate training with reset or explain dataset-specific performance differences; validation examples were previously used for checkpoint selection",
        "analysis": "All per-seed accuracy and macro-F1 results and paired differences; five-seed means and ranges; descriptive summaries without significance tests",
        "protocol_timing_scope": "This invocation records its protocol before inference. The record does not establish when earlier training or test outcomes were inspected.",
        "training": "none", "checkpoint_selection": "unchanged existing validation-selected checkpoints",
        "locked_test_forward_passes": 0, "sources": sources}
    save("protocol.json", protocol)
    shutil.copy2(__file__, OUT / "executed_source.py")
    synthetic_checks()
    all_records = []
    started = time.perf_counter()
    for dataset in DATASETS:
        cfg0 = configs[dataset, SEEDS[0], MODELS[0]]
        ds, split = build_dataset(cfg0, "val")
        expected_ids = [r["id"] for r in ds.records]
        expected_labels = [int(r["label"]) for r in ds.records]
        test_ids = {r["id"] for r in split["splits"]["test"]}
        assert not set(expected_ids) & test_ids
        save(f"inputs_{dataset}.json", {"dataset": dataset, "split": "val", "split_hash": split["split_hash"],
             "sample_ids": expected_ids, "labels": expected_labels, "sample_ids_sha256": stable_hash(expected_ids),
             "sample_count": len(expected_ids), "disjoint_from_locked_test": True})
        batches = []
        input_hash = hashlib.sha256()
        seen = []
        for frames, labels, ids, stats in make_loader(ds, cfg0, shuffle=False):
            frames = frames.contiguous()
            input_hash.update(frames.numpy().tobytes())
            batches.append((frames, labels.clone(), list(ids)))
            seen.extend(ids)
        assert seen == expected_ids
        print(json.dumps({"stage": "inputs_loaded", "dataset": dataset, "samples": len(seen),
                          "seconds_since_start": time.perf_counter() - started}), flush=True)
        for seed in SEEDS:
            for model_name in MODELS:
                cfg = configs[dataset, seed, model_name]
                seed_everything(seed)
                base, checkpoint, checkpoint_path = _load_checkpoint(cfg)
                base.eval()
                toggled = copy.deepcopy(base).eval()
                assert set_reset(toggled, model_name == MODELS[0]) == 7
                before = [state_hash(base), state_hash(toggled)]
                predictions = [[], []]; firing = [[], []]
                with torch.inference_mode():
                    for frames, labels, ids in batches:
                        x = frames.to(cfg.device)
                        for index, model in enumerate([base, toggled]):
                            output = model(x)
                            scores = output.mean(0)
                            predictions[index].extend(scores.argmax(-1).cpu().tolist())
                            firing[index].extend(output.mean((0, 2)).cpu().tolist())
                            functional.reset_net(model)
                        del x, output, scores
                after = [state_hash(base), state_hash(toggled)]
                assert before == after
                assert file_sha256(checkpoint_path) == next(r["checkpoint_sha256"] for r in sources
                    if r["dataset"] == dataset and r["seed"] == seed and r["trained_model"] == model_name)
                n_classes = int(DATASET_META[dataset]["classes"])
                metrics = []
                for pred in predictions:
                    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
                    np.add.at(cm, (np.array(expected_labels), np.array(pred)), 1)
                    metrics.append(confusion_metrics(cm))
                filename = f"predictions_{dataset}_{model_name}_seed{seed}.jsonl"
                with (OUT / filename).open("w") as handle:
                    for i, sample_id in enumerate(expected_ids):
                        handle.write(json.dumps({"sample_id": sample_id, "label": expected_labels[i],
                            "baseline_prediction": predictions[0][i], "switched_prediction": predictions[1][i],
                            "baseline_output_firing": firing[0][i], "switched_output_firing": firing[1][i]}) + "\n")
                result = {"dataset": dataset, "trained_model": model_name, "seed": seed, "T": 16,
                    "samples": len(seen), "baseline_reset": model_name == MODELS[1],
                    "intervention_reset": model_name == MODELS[0], "baseline": metrics[0], "switched": metrics[1],
                    "delta_accuracy": metrics[1]["accuracy"] - metrics[0]["accuracy"],
                    "delta_macro_f1": metrics[1]["macro_f1"] - metrics[0]["macro_f1"],
                    "baseline_output_firing": float(np.mean(firing[0])), "switched_output_firing": float(np.mean(firing[1])),
                    "prediction_changed_fraction": float(np.mean(np.array(predictions[0]) != np.array(predictions[1]))),
                    "input_tensor_sha256": input_hash.hexdigest(), "checkpoint_sha256": file_sha256(checkpoint_path),
                    "model_state_before": before, "model_state_after": after, "model_states_unchanged": before == after,
                    "predictions_file": filename, "predictions_sha256": file_sha256(OUT / filename),
                    "created_at": utc_now()}
                all_records.append(result)
                save("per_checkpoint.json", {"records": all_records})
                print(json.dumps({"stage": "complete_checkpoint_pair", "dataset": dataset, "model": model_name,
                    "seed": seed, "baseline_accuracy": metrics[0]["accuracy"], "switched_accuracy": metrics[1]["accuracy"],
                    "seconds_since_start": time.perf_counter() - started}), flush=True)
                del base, toggled, checkpoint
                torch.cuda.empty_cache()
        del batches, ds
    assert len(all_records) == 30
    unchanged = all(file_sha256(Path(r["checkpoint_path"])) == r["checkpoint_sha256"] and
                    file_sha256(Path(r["config_path"])) == r["config_sha256"] for r in sources)
    assert unchanged
    save("completion.json", {"completed_at": utc_now(), "checkpoint_pairs": len(all_records), "evaluations": 60,
         "all_checkpoint_and_config_hashes_unchanged": unchanged, "source_sha256": file_sha256(Path(__file__)),
         "total_seconds": time.perf_counter() - started})

if __name__ == "__main__":
    main()
