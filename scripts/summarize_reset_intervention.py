"""Verify and summarize the full-validation reset intervention without GPU work."""
from pathlib import Path
import argparse
import csv
import sys
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.release_paths import EVIDENCE, resolve_path
OUT = EVIDENCE / "reset_intervention"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def write_csv(name, rows):
    with (OUT / name).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def metrics(labels, predictions, classes):
    true = [0] * classes; predicted = [0] * classes; correct = [0] * classes
    for y, p in zip(labels, predictions):
        true[y] += 1; predicted[p] += 1
        correct[y] += int(y == p)
    f1 = [2*c/(t+p) if t+p else 0 for c,t,p in zip(correct,true,predicted)]
    return sum(correct)/len(labels), sum(f1)/classes

def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Reads saved predictions and writes CSV, integrity, and LaTeX reports under LPSN_EVIDENCE_ROOT/reset_intervention.')
    parser.parse_args()
    protocol = json.loads((OUT / "protocol.json").read_text())
    completion = json.loads((OUT / "completion.json").read_text())
    records = json.loads((OUT / "per_checkpoint.json").read_text())["records"]
    synthetic = json.loads((OUT / "synthetic_checks.json").read_text())
    assert protocol["script_sha256"] == completion["source_sha256"] == sha(OUT / "executed_source.py")
    assert len(records) == completion["checkpoint_pairs"] == 30
    assert completion["evaluations"] == 60
    assert len(synthetic["records"]) == 18 and synthetic["all_pass"]
    assert all(r["output_exact"] and r["logits_exact"] and r["roundtrip_output_exact"] for r in synthetic["records"])
    sources = {(r["dataset"],r["trained_model"],r["seed"]):r for r in protocol["sources"]}
    inputs_by_dataset={dataset:json.loads((OUT/f'inputs_{dataset}.json').read_text())
                       for dataset in protocol['datasets']}
    expected = {(d,m,s) for d in protocol["datasets"] for m in protocol["trained_models"] for s in protocol["seeds"]}
    assert set(sources) == expected == {(r["dataset"],r["trained_model"],r["seed"]) for r in records}
    rows=[]; checks=[]; source_hashes={}; class_counts={"cifar10_dvs":10,"dvs_gesture":11,"hardvs":300}
    for r in records:
        key=(r["dataset"],r["trained_model"],r["seed"]); source=sources[key]
        cp=resolve_path(source["checkpoint_path"]); cfg=resolve_path(source["config_path"])
        assert sha(cp)==source["checkpoint_sha256"]==r["checkpoint_sha256"]
        assert sha(cfg)==source["config_sha256"]
        assert r["model_states_unchanged"] and r["model_state_before"]==r["model_state_after"]
        predictions=OUT/r["predictions_file"]
        assert sha(predictions)==r["predictions_sha256"]
        data=[json.loads(line) for line in predictions.read_text().splitlines() if line]
        inputs=inputs_by_dataset[r['dataset']]
        assert len(data)==r["samples"]==inputs["sample_count"]
        assert inputs["disjoint_from_locked_test"]
        assert [v["sample_id"] for v in data]==inputs["sample_ids"]
        assert [v["label"] for v in data]==inputs["labels"]
        for condition in ["baseline","switched"]:
            acc,f1=metrics(inputs["labels"],[v[condition+"_prediction"] for v in data],class_counts[r["dataset"]])
            assert abs(acc-r[condition]["accuracy"])<1e-12
            assert abs(f1-r[condition]["macro_f1"])<1e-12
        assert abs(r["delta_accuracy"]-(r["switched"]["accuracy"]-r["baseline"]["accuracy"]))<1e-12
        assert abs(r["delta_macro_f1"]-(r["switched"]["macro_f1"]-r["baseline"]["macro_f1"]))<1e-12
        selection_path=cp.parent/"train_result.json"
        selection=json.loads(selection_path.read_text())
        source_hashes[str(selection_path)]=sha(selection_path)
        check={"dataset":r["dataset"],"trained_model":r["trained_model"],"seed":r["seed"],
               "accuracy_difference_from_selected_validation":r["baseline"]["accuracy"]-selection["best_validation_accuracy"],
               "macro_f1_difference_from_selected_validation":r["baseline"]["macro_f1"]-selection["best_validation_value"]}
        check["matches_selected_validation_metrics"]=abs(check["accuracy_difference_from_selected_validation"])<1e-12 and abs(check["macro_f1_difference_from_selected_validation"])<1e-12
        checks.append(check)
        rows.append({"dataset":r["dataset"],"trained_model":r["trained_model"],"seed":r["seed"],
            "n_samples":r["samples"],"baseline_reset":r["baseline_reset"],"intervention_reset":r["intervention_reset"],
            "baseline_accuracy_pct":100*r["baseline"]["accuracy"],"switched_accuracy_pct":100*r["switched"]["accuracy"],
            "delta_accuracy_pp":100*r["delta_accuracy"],"baseline_macro_f1_pct":100*r["baseline"]["macro_f1"],
            "switched_macro_f1_pct":100*r["switched"]["macro_f1"],"delta_macro_f1_pp":100*r["delta_macro_f1"],
            "baseline_output_firing_pct":100*r["baseline_output_firing"],"switched_output_firing_pct":100*r["switched_output_firing"],
            "prediction_changed_pct":100*r["prediction_changed_fraction"]})
    write_csv("per_seed.csv",rows)
    summaries=[]
    for dataset in protocol["datasets"]:
        for model in protocol["trained_models"]:
            group=[r for r in rows if r["dataset"]==dataset and r["trained_model"]==model]
            assert len(group)==5 and len({r["n_samples"] for r in group})==1
            summary={"dataset":dataset,"trained_model":model,"baseline_reset":group[0]["baseline_reset"],
                "intervention_reset":group[0]["intervention_reset"],"n_seeds":5,"n_samples":group[0]["n_samples"]}
            for field in ["baseline_accuracy_pct","switched_accuracy_pct","delta_accuracy_pp","baseline_macro_f1_pct",
                          "switched_macro_f1_pct","delta_macro_f1_pp","baseline_output_firing_pct","switched_output_firing_pct","prediction_changed_pct"]:
                summary[field]=statistics.mean(r[field] for r in group)
            for metric in ["accuracy","macro_f1"]:
                vals=[r[f"delta_{metric}_pp"] for r in group]
                summary[f"delta_{metric}_min_pp"]=min(vals);summary[f"delta_{metric}_max_pp"]=max(vals)
                summary[f"delta_{metric}_negative_seeds"]=sum(v<0 for v in vals)
                summary[f"delta_{metric}_positive_seeds"]=sum(v>0 for v in vals)
            summaries.append(summary)
    write_csv("summary.csv",summaries)
    name={"cifar10_dvs":"CIFAR10-DVS","dvs_gesture":"DVS-Gesture","hardvs":"HARDVS"}
    modelname={"lpsn_a1_lif_init":"A1: add reset","lpsn_a2_soft":"A2: remove reset"}
    sample_description=', '.join(f'{inputs_by_dataset[dataset]["sample_count"]:,} {name[dataset]}'
                                 for dataset in protocol['datasets'])
    method_text=(
        f'This fixed-weight inference-time intervention compares {len(records)} validation-selected A1/A2 checkpoints at common $T=16$: '
        f'{len(protocol["trained_models"])} trained variants, {len(protocol["seeds"])} seeds, and {len(protocol["datasets"])} datasets. '
        f'Each checkpoint and its reset-switched copy use the complete validation split ({sample_description} examples). '
        'The switch enables exact soft reset in all seven A1 neuron layers or disables it in all seven A2 layers. '
        'Spatial weights, kernel parameters, thresholds, biases, and learned normalization statistics remain fixed, with both networks in evaluation mode. '
        'The runner records its protocol before inference; this record does not establish when earlier training or test outcomes were inspected. '
        'Because these validation examples previously served checkpoint selection, the results characterize sensitivity of the fitted networks rather than independent holdout performance. '
        'Switching reset also changes the activation distribution presented to fixed downstream layers, so performance differences can include normalization mismatch. '
        'The intervention performs no retraining, checkpoint reselection, or test-set inference.')
    mainlines=[r"\subsubsection{Inference-time reset intervention}\label{subsec:reset_intervention}","",
        method_text,"",
        r"\begin{table*}[t]\centering\footnotesize",
        r"\caption{Full-validation reset interventions at $T=16$: five-seed means. Before/after values are percentages; deltas are switched minus baseline in percentage points, computed before rounding. The trained variant identifies the source checkpoint, not a retrained counterfactual. All spatial weights and normalization state remain fixed.}\label{tab:reset_intervention}",
        r"\begin{tabular}{llrrrrrr}\toprule Dataset & Switch & Acc. before & Acc. after & $\Delta$ Acc. & F1 before & F1 after & $\Delta$ F1\\\midrule"]
    for r in summaries:
        mainlines.append(f'{name[r["dataset"]]} & {modelname[r["trained_model"]]} & {r["baseline_accuracy_pct"]:.2f} & {r["switched_accuracy_pct"]:.2f} & ${r["delta_accuracy_pp"]:+.2f}$ & {r["baseline_macro_f1_pct"]:.2f} & {r["switched_macro_f1_pct"]:.2f} & ${r["delta_macro_f1_pp"]:+.2f}$'+r"\\")
    mainlines.extend([r"\bottomrule\end{tabular}\end{table*}",""])
    negative=sum(r["delta_macro_f1_pp"]<0 for r in rows)
    positive=sum(r["delta_macro_f1_pp"]>0 for r in rows)
    ties=len(rows)-negative-positive
    mainlines.append(f'The reset switch lowers validation macro-F1 in {negative} of {len(rows)} checkpoint comparisons, raises it in {positive}, and leaves it unchanged in {ties}. ' +
        'The table reports the observed directions for each dataset and trained variant. These fixed-weight interventions do not isolate the effect of training with reset or establish population-level reliability.')
    matches=sum(r["matches_selected_validation_metrics"] for r in checks)
    assert matches == 30, "Baseline replay differs from selected-validation records; inspect before reporting"
    (OUT/"reset_methods.tex").write_text(method_text+"\n")
    (OUT/"reset_summary.tex").write_text("\n".join(mainlines)+"\n")
    per_seed_report=[r"\section{Full-validation fixed-weight reset intervention}",
        method_text,
        r"At all seven neuron layers, A1's reset is enabled using the exact detached soft-reset strategy and a fixed reset amplitude equal to threshold; A2's reset is disabled. The baseline and switched network have identical spatial parameters, kernel values, biases, thresholds, and normalization state. Batch normalization and dropout remain in evaluation mode. Thus, the switch changes inference dynamics conditional on fitted weights, including the activation distribution seen by fixed downstream normalization. These measurements do not isolate the effect of training with reset or explain dataset-specific performance differences. Summaries are descriptive and include no significance tests.",
        r"Predictions, sample identifiers, input tensor hashes, configuration/checkpoint hashes, and per-network state hashes are retained. Eighteen synthetic checks span $T=8,16,32$, mixed-sign/nonnegative/subthreshold inputs, and both switch directions; outputs and logits match the target operator exactly, and switching back restores the original output. All 30 baseline/switched pairs retain their parameter and buffer states. Checkpoint and configuration bytes are unchanged.",
        f'Baseline replay agrees with the saved selected-validation accuracy and macro-F1 for {matches} of {len(records)} checkpoints; individual differences are retained in the integrity record. The following table gives every seed, with macro-F1 before and after the switch and both accuracy and macro-F1 changes. Values are percentages or percentage points; no interval or population-level reliability claim is implied.',
        r"\begingroup\footnotesize",
        r"\begin{longtable}{llrrrrr}\toprule Dataset & Trained & Seed & F1 before & F1 after & $\Delta$ Acc. & $\Delta$ F1\\\midrule\endhead"]
    for r in rows:
        m="A1" if r["trained_model"]=="lpsn_a1_lif_init" else "A2"
        per_seed_report.append(f'{name[r["dataset"]]} & {m} & {r["seed"]} & {r["baseline_macro_f1_pct"]:.2f} & {r["switched_macro_f1_pct"]:.2f} & ${r["delta_accuracy_pp"]:+.2f}$ & ${r["delta_macro_f1_pp"]:+.2f}$'+r"\\")
    per_seed_report.extend([r"\bottomrule\end{longtable}\endgroup",r"A1 rows add reset; A2 rows remove reset. Seed-level metrics are recomputed from the saved paired per-example predictions with all classes included in macro-F1. Summary rows average the five seed-level values without pooling seeds as independent examples."])
    (OUT/"reset_per_seed.tex").write_text("\n\n".join(per_seed_report)+"\n")
    artifacts={str(p.relative_to(OUT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!="integrity.json"}
    integrity={"passed":True,"checkpoint_pairs":30,"validation_evaluations":60,"synthetic_contrasts":18,
        "all_predictions_metrics_recomputed":True,"checkpoint_and_config_bytes_unchanged":True,
        "all_network_states_unchanged":True,"baseline_replay_matches":matches,"baseline_replay_checks":checks,
        "selection_source_sha256":source_hashes,"artifact_sha256":artifacts,
        "summary_script_sha256":sha(Path(__file__)),"independent_holdout":False,
        "interpretation":"Fixed-weight inference reset sensitivity conditional on selected checkpoints and reused validation examples; these measurements do not isolate effects of training with reset"}
    (OUT/"integrity.json").write_text(json.dumps(integrity,indent=2)+"\n")
    print(json.dumps({"passed":True,"baseline_replay_matches":matches,"summaries":summaries},indent=2))

if __name__=="__main__":main()
