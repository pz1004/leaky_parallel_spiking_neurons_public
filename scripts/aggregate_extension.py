"""Aggregate saved extension artifacts; never train or run model inference.

Summaries report the supplied configurations, results, and completed checks.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from experiments.config import load_config,file_sha256,stable_hash,utc_now
from experiments.statistics import crossed_interval
from experiments.cli import _exact_signed_rank,_holm_adjust
from scripts.audit_extension import audit_run,lines
from scripts.measure_gpu_energy import config_for
from scripts.diagnose_checkpoints import sealed,write_csv

from experiments.release_paths import EVIDENCE, resolve_path, portable_path
LABELS={'psn':'PSN','lpsn_a1_lif_init':'A1','lpsn_a2_soft':'A2','lif':'LIF',
        'fixed_diagonal':'Diagonal','fixed_causal_uniform':'Uniform','lpsn_a3_kernel':'A3',
        'lpsn_fixed_kernel_residual':'Fixed + residual','lpsn_a4a_frozen_bias':'Learned + residual'}
DATASETS={'cifar10_dvs':'CIFAR10-DVS','dvs_gesture':'DVS-Gesture','hardvs':'HARDVS'}
ARCHITECTURES={'vgg_small':'VGG','vgg_small_half':'Half VGG','spiking_resnet18':'ResNet-18'}

def table(caption,label,header,rows,wide=False):
    kind='table*' if wide else 'table'
    return '\n'.join([r'\begin{'+kind+'}[t]',r'\centering\footnotesize',r'\caption{'+caption+'}',
        r'\label{'+label+'}',r'\begin{tabular}{'+'l'*len(header)+'}',r'\toprule',
        ' & '.join(header)+r'\\\midrule',*[' & '.join(map(str,row))+r'\\' for row in rows],
        r'\bottomrule',r'\end{tabular}',r'\end{'+kind+'}'])


def baseline_reuse(protocol):
    accepted=[]
    excluded={'experiment_id','architecture','model','seed','output_root','frame_cache_root'}
    for dataset in ['cifar10_dvs','dvs_gesture']:
        exemplar=load_config(resolve_path(next(r['config'] for r in protocol['runs'] if r['dataset']==dataset)))
        for model in ['psn','lpsn_a1_lif_init','lpsn_a2_soft']:
            for seed in [42,123,256,512,1024]:
                cfg,path=config_for(dataset,model,seed)
                original=load_config(path)
                left={k:v for k,v in asdict(exemplar).items() if k not in excluded}
                right={k:v for k,v in asdict(original).items() if k not in excluded}
                assert left==right, {k:(left[k],right[k]) for k in left if left[k]!=right[k]}
                metric,_,source=sealed(cfg)
                result=json.loads((resolve_path(cfg.run_dir)/'train_result.json').read_text())
                checkpoint=torch.load(resolve_path(cfg.run_dir)/'checkpoint_best.pt',map_location='cpu',weights_only=False)
                assert metric['training_config_hash']==original.config_hash==checkpoint['config_hash']
                if 'config_hash' in result:
                    assert result['config_hash']==original.config_hash
                assert checkpoint['config']==asdict(original)
                assert checkpoint['epoch']==result['best_epoch']
                benchmark=json.loads((resolve_path(cfg.run_dir)/'benchmark.json').read_text())
                assert metric['selection_metric']=='macro_f1' and metric['selection_split']=='validation'
                accepted.append({'dataset':dataset,'architecture':'vgg_small','model':model,'seed':seed,
                                 'accuracy':metric['accuracy'],'macro_f1':metric['macro_f1'],
                                 'parameter_count':result['parameter_count'], 'reused':True,
                                 'latency_ms_per_batch':benchmark['latency_ms_per_batch'],
                                 'peak_memory_mb':benchmark['peak_cuda_memory_mb'],
                                 'source_config':portable_path(path),'checkpoint_sha256':metric['checkpoint_sha256'],
                                 'prediction_source':source})
    (EVIDENCE/'baseline_reuse.json').write_text(json.dumps(accepted,indent=2)+'\n')
    return accepted


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    protocol=json.loads((EVIDENCE/'protocol.json').read_text())
    if stable_hash({k:v for k,v in protocol.items() if k!='protocol_hash'}) != protocol['protocol_hash']:
        raise RuntimeError('Protocol hash mismatch')
    if len(protocol['runs']) != 110:
        raise RuntimeError('Expected the complete 110-condition extension protocol')
    for entry in protocol['runs']:
        if file_sha256(resolve_path(entry['config'])) != entry['sha256']:
            raise RuntimeError('Frozen config changed: '+entry['config'])
    sections=[]; values={'generated_at':utc_now()}; source_files=[]
    layer_paths=list((EVIDENCE/'diagnostics').glob('layers_*.json'))
    values['layer_records_present']=len(layer_paths)
    reused=baseline_reuse(protocol)
    common_path=EVIDENCE/'diagnostics/common_t16_per_seed.csv'
    if common_path.exists():
        rows=list(csv.DictReader(common_path.open()));grouped=defaultdict(list)
        for r in rows: grouped[r['dataset'],r['model']].append(r)
        out=[];vals=[]
        for (dataset,model),rs in grouped.items():
            assert len(rs)==5
            v=dict(dataset=dataset,model=model)
            for metric in ['accuracy','macro_f1']:
                v[metric]=float(np.mean([float(r[metric]) for r in rs]))
                v[metric+'_ci']=crossed_interval(rs,metric)
            vals.append(v)
            out.append([DATASETS[dataset],LABELS[model],f'{100*v["accuracy"]:.2f}',f'{100*v["macro_f1"]:.2f}'])
        values['common_t16']=vals
        sections += [r'\subsection{Common-length and paired dataset diagnostics}\label{subsec:checkpoint_diagnostics}',
                     'Table~\\ref{tab:common_t16} compares all three datasets at $T=16$, using sealed per-example predictions and existing checkpoints. The common length permits comparison of the observed model ordering without confounding it with different temporal lengths. This summary reads saved results without training or model inference.',
                     table('Existing common-$T=16$ results: means over five matched seeds (percent). Per-seed values and paired differences are retained in the evidence package.','tab:common_t16',['Dataset','Model','Accuracy','Macro-F1'],out)]
        ass=list(csv.DictReader((EVIDENCE/'diagnostics/paired_associations.csv').open()))
        a=[r for r in ass if r['dataset']=='hardvs' and r['left']=='lpsn_a1_lif_init' and r['right']=='psn']
        axes=['duration','event_count','event_mass','active_bin_fraction']
        expected={(axis,seed) for axis in axes for seed in [42,123,256,512,1024]}
        selected=[r for r in a if r['axis'] in axes]
        if len(selected)!=len(expected) or {(r['axis'],int(r['seed'])) for r in selected}!=expected:
            raise ValueError('Incomplete or duplicate HARDVS within-class association records')
        if any(r['within_class_correlation'] and not np.isfinite(float(r['within_class_correlation'])) for r in selected):
            raise ValueError('Nonfinite HARDVS correlation; undefined values must be recorded as missing')
        correlations={axis:[float(r['within_class_correlation']) for r in selected
                            if r['axis']==axis and r['within_class_correlation']] for axis in axes}
        values['hardvs_within_class_correlations']=correlations
        arr=[x for xs in correlations.values() for x in xs]
        if arr:
            correlation_text=f'For HARDVS A1 minus PSN, defined within-class correlations between paired correctness differences and duration, event count, framed event mass, or occupancy range from {min(arr):.3f} to {max(arr):.3f}, with {len(arr)} of 20 seed--variable pairs defined.'
        else:
            correlation_text='For HARDVS A1 minus PSN, none of the 20 within-class seed--variable correlations is defined; no correlation range is reported.'
        sections.append(correlation_text+' Undefined correlations are retained as missing values when either class-centered quantity has negligible variance. Duration, counts, and mass use $\\log(1+x)$; occupancy is untransformed. Centering both quantities within each class reduces class-composition confounding but does not control all differences between examples. Class-support associations are reported separately; support has no within-class variation. Per-seed global and within-class quartile tables retain empty/tied-bin limitations. These descriptive associations do not establish that accumulation or reset sensitivity causes any observed HARDVS performance difference.')
        sections.append('Layerwise validation diagnostics are retained separately; '+str(len(layer_paths))+' of 45 expected checkpoint records are present. Their counters are verified by the layer-summary command. Firing, sustained activity, threshold-relative membrane distributions, and reset corrections describe the frozen models and cannot establish architecture or temporal-length generalization on HARDVS.')
    energy=[];energy_keys=set()
    expected_energy={(dataset,model,batch) for dataset in DATASETS
                     for model in ['psn','lpsn_a1_lif_init','lpsn_a2_soft']+(['lif'] if dataset=='hardvs' else [])
                     for batch in [1,16]}
    for path in sorted((EVIDENCE/'energy').glob('*_seed42_B*.json')):
        r=json.loads(path.read_text());assert len(r['blocks'])==10
        identity=r['dataset'],r['model'],r['batch_size']
        if identity not in expected_energy or identity in energy_keys:
            raise ValueError('Unexpected or duplicate energy condition: '+str(identity))
        energy_keys.add(identity)
        assert r['warmup']==100 and {b['block'] for b in r['blocks']}==set(range(10))
        for b in r['blocks']:
            assert b['seconds']>=5 and b['forward_calls']>=100
            assert b['end_mj']>b['start_mj']
            assert np.isclose(b['joules_per_sample'],(b['end_mj']-b['start_mj'])/1000/(b['forward_calls']*r['batch_size']),rtol=1e-10,atol=1e-10)
            assert np.isclose(b['latency_ms_per_batch'],b['seconds']*1000/b['forward_calls'],rtol=1e-10,atol=1e-10)
        cfg,cfg_path=config_for(r['dataset'],r['model']);metric,_,_=sealed(cfg)
        assert r['seed']==42 and r['time_steps']==cfg.time_steps
        assert r['checkpoint_sha256']==metric['checkpoint_sha256']
        assert r['config_sha256']==file_sha256(cfg_path)
        energy.append({**{k:r[k] for k in ['dataset','model','batch_size','seed','time_steps']},
            'accuracy':metric['accuracy'],
            **{m:float(np.mean([b[m] for b in r['blocks']])) for m in ['joules_per_sample','latency_ms_per_batch','throughput','peak_memory_mb']},
            'energy_block_sd':float(np.std([b['joules_per_sample'] for b in r['blocks']],ddof=1)),
            'latency_block_sd':float(np.std([b['latency_ms_per_batch'] for b in r['blocks']],ddof=1))})
        source_files.append({'path':portable_path(path),'sha256':file_sha256(path)})
    values['energy_measurements_verified']=len(energy)
    if energy_keys==expected_energy:
        write_csv(EVIDENCE/'energy_summary.csv',energy);values['energy']=energy
        sections += [r'\subsection{Measured GPU energy and inference cost}\label{subsec:gpu_energy}',
            'Table~\\ref{tab:energy_measured} measures gross GPU energy using the cumulative NVML counter in millijoules, divided by 1,000 and by the number of processed samples. Seed 42 was fixed before measurement rather than chosen for performance. The first 16 frozen validation inputs are identical across models within each dataset. Batch sizes 1 and 16 use 100 warm-up iterations and ten blocks, each lasting at least five seconds and containing at least 100 whole-model forward calls, with synchronization and neuron-state reset between calls. Inputs reside on the device; event acquisition, decoding, framing, host transfer, and CPU energy are excluded. Measurement blocks must run without concurrent training; their runtime provenance is recorded separately. These measurements concern dense GPU board energy, including background graphics activity; they are not neuromorphic or whole-system energy.',
            table('Measured gross GPU energy and latency (mean over ten blocks); energy variability is one block standard deviation. Accuracy is the selected seed-42 checkpoint, not a five-seed mean. B denotes batch size; latency is per batch.','tab:energy_measured',
                  ['Dataset','Model','B','Acc. (\\%)','J/sample','SD','ms/batch','MiB'],
                  [[DATASETS[r['dataset']],LABELS[r['model']],r['batch_size'],f'{r["accuracy"]*100:.2f}',f'{r["joules_per_sample"]:.3f}',f'{r["energy_block_sd"]:.3f}',f'{r["latency_ms_per_batch"]:.2f}',f'{r["peak_memory_mb"]:.0f}'] for r in energy],wide=True)]
        idle=json.loads((EVIDENCE/'energy/protocol.json').read_text())['idle_blocks']
        sections.append(f'Idle power, recorded separately in ten one-second blocks, averaged {np.mean([r["watts"] for r in idle]):.1f} W. It was not subtracted from gross energy. Block variation measures repeatability for these inputs and checkpoint, not uncertainty across all examples or training seeds. Throughput and per-layer A2 component timings are included in the evidence package. The component timings separately include temporal multiplication, sequential history construction, and reset-correction multiplication; isolated microbenchmarks are not an additive decomposition of whole-model latency.')
    else:
        sections += [r'\subsection{GPU energy evidence status}\label{subsec:gpu_energy}',f'Energy collection is incomplete ({len(energy)} of 20 model--dataset--batch measurements). The combined energy table requires all 20 measurements.']
    admitted=[];pending=[]
    for entry in protocol['runs']:
        cfg=load_config(resolve_path(entry['config']))
        if not (resolve_path(cfg.run_dir)/'test_evaluation_ledger.jsonl').exists():pending.append(entry);continue
        audit_run(cfg)
        metric=lines(resolve_path(cfg.run_dir)/'locked_test_metrics.jsonl')[0];result=json.loads((resolve_path(cfg.run_dir)/'train_result.json').read_text())
        benchmark=json.loads((resolve_path(cfg.run_dir)/'benchmark.json').read_text())
        admitted.append({'dataset':cfg.dataset,'architecture':cfg.architecture,'model':cfg.model,'seed':cfg.seed,
                         'accuracy':metric['accuracy'],'macro_f1':metric['macro_f1'],
                         'parameter_count':result['parameter_count'],'reused':False,
                         'latency_ms_per_batch':benchmark['latency_ms_per_batch'],
                         'peak_memory_mb':benchmark['peak_cuda_memory_mb'],
                         'source_config':entry['config'],'checkpoint_sha256':metric['checkpoint_sha256']})
    values['extension_runs_verified']=len(admitted);values['extension_runs_expected']=110
    sections += [r'\subsection{Architecture and component extension}\label{subsec:extension_results}']
    if pending:
        sections.append(f'The 110-run matrix is incomplete: {len(admitted)} extension runs have passed the acceptance audit. Paired comparison tables are produced after all 110 runs pass validation. Partial outcomes remain in their source records.')
    else:
        all_runs=reused+admitted;lookup={(r['dataset'],r['architecture'],r['model'],r['seed']):r for r in all_runs}
        pairs=[];deltas=[]
        for c in protocol['contrasts']:
            rs=[]
            for seed in [42,123,256,512,1024]:
                left=lookup[c['dataset'],c['architecture'],c['left'],seed];right=lookup[c['dataset'],c['architecture'],c['right'],seed]
                rs.append({'seed':seed,'delta':left['macro_f1']-right['macro_f1']})
                deltas.append({**c,**rs[-1],'accuracy_delta':left['accuracy']-right['accuracy']})
            test=_exact_signed_rank([r['delta'] for r in rs]);ci=crossed_interval(rs,'delta')
            pairs.append({**c,'test_family':c['family'],'macro_f1_delta':float(np.mean([r['delta'] for r in rs])),
                'ci_low':ci[0],'ci_high':ci[1],'p_two_sided':test['wilcoxon_p_two_sided_raw']})
        _holm_adjust(pairs,'p_two_sided','p_holm')
        write_csv(EVIDENCE/'extension_per_seed.csv',all_runs);write_csv(EVIDENCE/'extension_paired_tests.csv',pairs)
        write_csv(EVIDENCE/'extension_seed_differences.csv',deltas)
        values['extension_contrasts']=pairs
        groups=defaultdict(list)
        for row in all_runs:groups[row['dataset'],row['architecture'],row['model']].append(row)
        out=[]
        for (ds,arch,model),rs in groups.items():
            out.append([DATASETS[ds],ARCHITECTURES[arch],LABELS[model],rs[0]['parameter_count'],
                        f'{np.mean([r["accuracy"] for r in rs])*100:.2f}',f'{np.mean([r["macro_f1"] for r in rs])*100:.2f}',
                        f'{np.mean([r["latency_ms_per_batch"] for r in rs]):.2f}',f'{np.mean([r["peak_memory_mb"] for r in rs]):.0f}'])
        sections += [table('Audited architecture and component results; five-seed means in percent. VGG baselines are reused after compatibility and hash checks. Latency and memory use the whole-model zero-input benchmark at batch 16. All extension conditions are included.','tab:extension_metrics',['Dataset','Backbone','Condition','Parameters','Acc.','F1','ms','MiB'],out,wide=True),
                     'All 110 extension runs passed the single-test and validation-selection audit. Seed-level differences, percentile intervals, and both raw and Holm-adjusted two-sided p-values are supplied in the evidence package. With five pairs, no exact two-sided result can establish significance at 0.05.']
    (EVIDENCE/'extension_tables.tex').write_text('\n\n'.join(sections)+'\n')
    (EVIDENCE/'analysis_values.json').write_text(json.dumps(values,indent=2)+'\n')
    (EVIDENCE/'extension_sources.json').write_text(json.dumps(source_files,indent=2)+'\n')
    status={'extension_runs':{'verified':len(admitted),'expected':110,'pending':len(pending)},
            'energy_measurements':{'verified':len(energy),'expected':20},
            'layer_records':{'present':len(layer_paths),'expected':45,
                             'counter_integrity':'not assessed by this command'},
            'integrity':{'protocol':'passed','baseline_reuse':'passed',
                         'verified_extension_runs':'passed' if admitted else 'none available',
                         'present_energy_records':'passed' if energy else 'none available'},
            'extension_comparisons_written':not pending,'updated_at':utc_now()}
    (EVIDENCE/'analysis_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))

if __name__=='__main__':main()
