"""Verify saved execution measurements and write CSV/TeX summaries; no model execution."""
from __future__ import annotations
import argparse
import csv
import sys
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.release_paths import EVIDENCE, resolve_path
OUT=EVIDENCE/"execution_comparison"
LABELS={"a2_original":"A2, matrix correction","a2_retained":"A2 retained correction",
        "lif_soft_detached":"LIF soft reset, detached","diagonal_dense":"Diagonal GEMM",
        "diagonal_elementwise":"Diagonal elementwise","a1_dense":"A1 GEMM"}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)


def main():
    parser=argparse.ArgumentParser(description=__doc__, epilog='Reads saved measurements and writes CSV, integrity, and LaTeX reports under LPSN_EVIDENCE_ROOT/execution_comparison.')
    parser.parse_args()
    protocol=json.loads((OUT/"protocol.json").read_text())
    exact=json.loads((OUT/"exactness.json").read_text())
    full=json.loads((OUT/"full_model_checks.json").read_text())
    raw=[json.loads(s) for s in (OUT/"raw_blocks.jsonl").read_text().splitlines()]
    errors=[];groups=defaultdict(list)
    expected={(m["family"],m["T"],imp,b,block) for m in protocol["families"]
              for imp in m["implementations"] for b in [1,16] for block in range(protocol["blocks"])}
    keys=[(r["family"],r["T"],r["implementation"],r["batch"],r["block"]) for r in raw]
    if len(raw)!=len(expected) or len(set(keys))!=len(expected) or set(keys)!=expected:errors.append("Raw blocks do not match the declared unique blocks")
    if "completed_at" not in protocol:errors.append("Benchmark completion marker absent")
    if digest(OUT/"benchmark_source_executed.py")!=protocol["script_sha256"]:errors.append("Executed benchmark source hash mismatch")
    if full.get("check_script_sha256")!=digest(OUT/full["check_source_file"]):errors.append("Full-model check source snapshot hash mismatch")
    for m in protocol["families"]:
        if m["checkpoint_sha256_before"]!=m.get("checkpoint_sha256_after") or digest(resolve_path(m["checkpoint"]))!=m["checkpoint_sha256_before"]:
            errors.append("Checkpoint hash mismatch: "+m["checkpoint"])
        if digest(resolve_path(m["config"]))!=m["config_sha256"]:errors.append("Config hash mismatch: "+m["config"])
    for r in raw:
        inf=r["inference"];n=inf["calls"];b=r["batch"]
        checks=[close(inf["joules_per_sample"],(inf["end_mj"]-inf["start_mj"])/1000/(n*b)),
                close(inf["latency_ms"],inf["seconds"]*1000/n),
                inf["seconds"]>=protocol["inference_min_seconds"],n>=protocol["inference_min_calls"],
                r["learned_parameters_unchanged"],r["all_state_restored"],
                len(r["adam_zero_lr_steps"])==protocol["workload_steps_per_block"]]
        for s in r["adam_zero_lr_steps"]:
            checks.append(close(s["total_step_ms"],sum(s[f] for f in ["forward_loss_ms","backward_ms","optimizer_reset_ms"])))
        meta=next(m for m in protocol["families"] if m["family"]==r["family"] and m["T"]==r["T"])
        checks.append(r["input_batch_sha256"]==meta["input_batch_sha256"][r["block"]])
        if not all(checks):errors.append("Raw arithmetic/state/protocol failure: "+str((r["family"],r["T"],r["implementation"],b,r["block"])))
        groups[r["family"],r["T"],r["implementation"],b].append(r)
    if not exact["all_pass"] or len(exact["records"])!=27:errors.append("Float64 exactness checks failed or missing")
    if len(full["records"])!=5:errors.append("Expected five full-model implementation contrasts")
    expected_full={(m['family'],m['T'],m['implementations'][0],implementation)
                   for m in protocol['families'] for implementation in m['implementations'][1:]}
    actual_full={(r['family'],r['T'],r['reference'],r['implementation']) for r in full['records']}
    if actual_full!=expected_full:errors.append('Full-model contrasts do not match the declared implementations')
    if exact.get('tolerance')!={'atol':1e-11,'rtol':1e-10} or full.get('atol')!=1e-4 or full.get('rtol')!=1e-3:
        errors.append('Numerical-check tolerances differ from the declared protocol')
    if not all(r['output_exact'] and r['gradient_pass'] for r in full['records']):
        errors.append('One or more full-model output or gradient checks failed')
    if not all(r['output_exact'] and r['gradient_pass'] for r in exact['records']):
        errors.append('One or more float64 output or gradient checks failed')
    integrity={"passed":not errors,"errors":errors,"raw_blocks":len(raw),"conditions":len(groups),
        "expected_raw_blocks":len(expected),"float64_all_pass":exact["all_pass"],
        "full_model_output_exact_count":sum(r["output_exact"] for r in full["records"]),
        "full_model_gradient_pass_count":sum(r["gradient_pass"] for r in full["records"]),
        "full_model_contrasts":len(full["records"]),"all_parameters_unchanged":all(r["learned_parameters_unchanged"] for r in raw),
        "all_mutable_state_restored":all(r["all_state_restored"] for r in raw),
        "sources_sha256":{p.name:digest(p) for p in [OUT/"protocol.json",OUT/"raw_blocks.jsonl",OUT/"exactness.json",OUT/"full_model_checks.json",OUT/"benchmark_source_executed.py",ROOT/"scripts/summarize_execution_comparison.py"]}}
    (OUT/"integrity.json").write_text(json.dumps(integrity,indent=2)+"\n")
    if errors:raise RuntimeError("; ".join(errors))
    rows=[]
    for (family,t,implementation,batch),rr in sorted(groups.items()):
        row=dict(family=family,T=t,implementation=implementation,batch=batch,blocks=len(rr))
        vals={"inference_ms":[r["inference"]["latency_ms"] for r in rr],
              "inference_joules_per_sample":[r["inference"]["joules_per_sample"] for r in rr]}
        for phase in ["forward_loss_ms","backward_ms","optimizer_reset_ms","total_step_ms"]:
            vals[phase]=[statistics.mean(s[phase] for s in r["adam_zero_lr_steps"]) for r in rr]
        for k,v in vals.items():row[k+"_mean"]=statistics.mean(v);row[k+"_sd"]=statistics.stdev(v)
        row["inference_peak_mib_max"]=max(r["inference"]["peak_allocated_mib"] for r in rr)
        row["workload_peak_mib_max"]=max(r["training_peak_allocated_mib"] for r in rr)
        rows.append(row)
    with (OUT/"summary.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    methods = [
        f"Saved execution protocol: {protocol['blocks']} validation batches/blocks, at least "
        f"{protocol['inference_min_seconds']:g} seconds and {protocol['inference_min_calls']} calls per inference block, "
        f"{protocol['inference_warmups']} inference warmups, {protocol['workload_warmups']} workload warmups, "
        f"and {protocol['workload_steps_per_block']} timed Adam steps per block at zero learning rate.",
        "Inference and gradient workloads use fixed validation inputs and checkpoint parameters. "
        "Gross GPU board energy includes idle/background activity and excludes acquisition, preprocessing, host transfer, and CPU energy. "
        "Block variability does not measure uncertainty across independent training seeds."
    ]
    (OUT/"execution_methods.tex").write_text("\n\n".join(methods)+"\n")
    summary_table = [r'\begin{table*}[t]\centering\footnotesize',
        r'\caption{Saved implementation measurements at $T=32$, batch 16. Latency is in milliseconds; memory is maximum allocated MiB; energy is gross GPU joules per sample.}',
        r'\begin{tabular}{lrrrrrr}\toprule Implementation & Inference & Backward & Total step & Infer. MiB & Step MiB & J/sample\\\midrule']
    for r in rows:
        if r['T']==32 and r['batch']==16:
            summary_table.append(f'{LABELS[r["implementation"]]} & {r["inference_ms_mean"]:.2f} & {r["backward_ms_mean"]:.2f} & {r["total_step_ms_mean"]:.2f} & {r["inference_peak_mib_max"]:.0f} & {r["workload_peak_mib_max"]:.0f} & {r["inference_joules_per_sample_mean"]:.3f}'+r'\\')
    summary_table.append(r'\bottomrule\end{tabular}\end{table*}')
    (OUT/'execution_summary.tex').write_text('\n'.join(summary_table)+'\n')
    details=[r'\section{Implementation benchmarks and numerical verification}\label{sec:execution_checks}',
       methods[0] + r' Implementations are interleaved within each family using fixed validation batches. Each zero-learning-rate Adam step computes gradients without changing learned parameters; mutable state is restored after each block. Values are means and sample standard deviations of block means, not training-seed confidence intervals. Inference energy is gross GPU board energy. The torch recurrent reference uses matched soft-reset dynamics, reset detachment, and ATan surrogate. These workloads do not establish epoch speed, learning convergence, or time to target quality.',
       r'\begingroup\scriptsize\setlength{\tabcolsep}{2pt}',
       r'\begin{longtable}{p{1.25in}lrrrrrr}',
       r'\caption{All implementation conditions. Inference and total zero-update step latency in milliseconds (mean $\pm$ block SD). Memory maxima in MiB.}\label{tab:execution_all}\\',
       r'\toprule Implementation & $T$/B & Inference & Backward & Total step & Infer. memory & Step memory & J/sample\\\midrule\endfirsthead',
       r'\toprule Implementation & $T$/B & Inference & Backward & Total step & Infer. memory & Step memory & J/sample\\\midrule\endhead']
    for r in rows:
        details.append(f'{LABELS[r["implementation"]]} & {r["T"]}/{r["batch"]} & '+
           f'${r["inference_ms_mean"]:.2f}\\pm{r["inference_ms_sd"]:.2f}$ & ${r["backward_ms_mean"]:.2f}\\pm{r["backward_ms_sd"]:.2f}$ & '+
           f'${r["total_step_ms_mean"]:.2f}\\pm{r["total_step_ms_sd"]:.2f}$ & {r["inference_peak_mib_max"]:.0f} & {r["workload_peak_mib_max"]:.0f} & '+
           f'${r["inference_joules_per_sample_mean"]:.3f}\\pm{r["inference_joules_per_sample_sd"]:.3f}$'+r'\\')
    details += [r'\bottomrule\end{longtable}',r'\begin{longtable}{p{1.4in}lrrrr}',
       r'\caption{Synchronized zero-update Adam phase timings, milliseconds (mean $\pm$ block SD). Forward includes loss calculation; optimizer includes state reset. Total includes the synchronization boundaries.}\label{tab:execution_phases}\\',
       r'\toprule Implementation & $T$/B & Forward/loss & Backward & Optimizer/reset & Total\\\midrule\endfirsthead',
       r'\toprule Implementation & $T$/B & Forward/loss & Backward & Optimizer/reset & Total\\\midrule\endhead']
    for r in rows:
        phase_values=[f'${r[k+"_mean"]:.2f}\\pm{r[k+"_sd"]:.2f}$' for k in ["forward_loss_ms","backward_ms","optimizer_reset_ms","total_step_ms"]]
        details.append(f'{LABELS[r["implementation"]]} & {r["T"]}/{r["batch"]} & '+ ' & '.join(phase_values)+r'\\')
    details += [r'\bottomrule\end{longtable}',r'\clearpage',r'\begin{longtable}{p{1.65in}lrrrr}',
       r'\caption{Full-model float32 checks with batch one, neuron training paths, and frozen normalization/dropout. Gradient relative errors are $L_2$ relative errors over all trainable spatial parameters.}\label{tab:execution_checks}\\',
       r'\toprule Comparison & $T$ & Different output spikes & Max gradient error & Relative gradient error & Gradient tolerance pass\\\midrule\endfirsthead',
       r'\toprule Comparison & $T$ & Different output spikes & Max gradient error & Relative gradient error & Gradient tolerance pass\\\midrule\endhead']
    for r in full["records"]:
        details.append(f'{LABELS[r["implementation"]]} vs {LABELS[r["reference"]]} & {r["T"]} & {r["output"]["different_elements"]}/{r["output"]["elements"]} & {r["parameter_gradients"]["max_abs"]:.3g} & {r["parameter_gradients"]["relative_l2"]:.3g} & {str(r["gradient_pass"])}'+r'\\')
    details += [r'\bottomrule\end{longtable}\endgroup',r'The 27 float64 operator contrasts cover mixed-sign, nonnegative, and subthreshold inputs at $T=8,16,32$. They pass exact output equality and gradient tolerances $10^{-11}$ absolute/$10^{-10}$ relative, with maximum observed gradient errors retained in the raw records. Synthetic checks include nonzero spike and nonzero gradient cases. The five full-network float32 contrasts use one validation example per selected checkpoint at temporal lengths of 16 or 32, with batch size one. They reproduce identical output spikes and meet parameter-gradient tolerances of $10^{-4}$ absolute and $10^{-3}$ relative. These finite-precision checks do not imply identical retraining trajectories or accuracy. The full-model harness explicitly disables both PyTorch and SpikingJelly dropout and freezes batch normalization while keeping neuron training paths active. Repeated checks retain their preceding records under the rechecks directory. Frozen checkpoint and configuration hashes, input indices and hashes, raw NVML counters, phase timings, and state-restoration assertions are supplied in the execution evidence package.']
    (OUT/"execution_details.tex").write_text("\n\n".join(details)+"\n")
    integrity["latex_reports_sha256"] = {name:digest(OUT/name) for name in ["execution_methods.tex", "execution_summary.tex", "execution_details.tex"]}
    (OUT/"integrity.json").write_text(json.dumps(integrity,indent=2)+"\n")
    print(json.dumps({"conditions":len(rows),"blocks":len(raw),"integrity_passed":True,
        "full_output_exact":integrity["full_model_output_exact_count"],"full_gradient_pass":integrity["full_model_gradient_pass_count"]}))


if __name__=="__main__":main()
