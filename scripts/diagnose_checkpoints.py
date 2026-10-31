"""Read sealed predictions; collect frozen-validation neuronal diagnostics.

No training and no test-set forward passes. Associations are descriptive and
conditional on classes, examples, and the five existing optimization seeds.
"""
import argparse
import csv
import fcntl
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from spikingjelly.activation_based import functional

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.config import file_sha256, stable_hash, utc_now
from experiments.cli import _load_checkpoint
from experiments.data import build_dataset
from experiments.engine import make_loader, seed_everything, _is_spiking
from scripts.measure_gpu_energy import config_for, check_no_training, source_snapshot
from lpsn_neuron import LPSpikingNeuron, soft_reset_correction
from experiments.release_paths import EVIDENCE, resolve_path

OUTPUT = EVIDENCE / 'diagnostics'
SEEDS = [42, 123, 256, 512, 1024]
MODELS = ['psn', 'lpsn_a1_lif_init', 'lpsn_a2_soft']


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sealed(cfg):
    metrics = [json.loads(l) for l in (cfg.run_dir/'locked_test_metrics.jsonl').read_text().splitlines() if l]
    clean = [r for r in metrics if r['condition']=='clean' and r['draw']==0 and r['severity']==0]
    if len(clean) != 1: raise RuntimeError(f'Clean record count {cfg.run_dir}: {len(clean)}')
    row = clean[0]
    manifest_path = resolve_path(row['evaluation_manifest_path'])
    manifest = json.loads(manifest_path.read_text())
    if stable_hash({k:v for k,v in manifest.items() if k!='manifest_hash'}) != manifest['manifest_hash']:
        raise RuntimeError(f'Manifest hash mismatch: {manifest_path}')
    metrics_digest = manifest.get('artifacts', {}).get('metrics_payload_sha256')
    if not metrics_digest:
        raise RuntimeError(f'Metrics payload hash missing from evaluation manifest: {manifest_path}')
    if stable_hash(row) != metrics_digest:
        raise RuntimeError(f'Metrics payload hash mismatch: {cfg.run_dir}/locked_test_metrics.jsonl (manifest: {manifest_path})')
    path = resolve_path(row['per_sample_path'])
    if file_sha256(path) != manifest['artifacts']['per_sample_sha256']:
        raise RuntimeError(f'Prediction hash mismatch: {path}')
    if file_sha256(cfg.run_dir/'checkpoint_best.pt') != row['checkpoint_sha256']:
        raise RuntimeError(f'Checkpoint hash mismatch: {cfg.run_dir}')
    return row, [json.loads(l) for l in path.read_text().splitlines() if l], {'path':str(path), 'sha256':file_sha256(path)}


def correlation(x, y):
    if np.std(x)<1e-12 or np.std(y)<1e-12: return None
    return float(np.corrcoef(x,y)[0,1])


def predictions():
    summaries, strata, associations, sources = [], [], [], []
    for dataset in ['cifar10_dvs','dvs_gesture','hardvs']:
        for seed in SEEDS:
            records = {}
            for model in MODELS:
                cfg,_ = config_for(dataset,model,seed,t=16)
                metric, samples, source = sealed(cfg)
                sources.append(source)
                records[model] = {r['sample_id']:r for r in samples}
                summaries.append(dict(dataset=dataset, model=model, seed=seed, time_steps=16,
                                      accuracy=metric['accuracy'], macro_f1=metric['macro_f1'],
                                      checkpoint_sha256=metric['checkpoint_sha256'], split_hash=metric['split_hash']))
            ids = sorted(records['psn'])
            base = records['psn']
            labels = np.array([base[i]['label'] for i in ids])
            support = Counter(labels)
            for left,right in [('lpsn_a1_lif_init','psn'),('lpsn_a2_soft','psn'),('lpsn_a2_soft','lpsn_a1_lif_init')]:
                assert set(records[left])==set(records[right])==set(ids)
                for i in ids:
                    for key in ['label','duration','event_count','event_mass','active_bin_fraction']:
                        assert records[left][i][key]==records[right][i][key], (i,key)
                delta = np.array([int(records[left][i]['prediction']==base[i]['label']) - int(records[right][i]['prediction']==base[i]['label']) for i in ids])
                for axis in ['duration','event_count','event_mass','active_bin_fraction','class_support']:
                    x = np.array([support[base[i]['label']] if axis=='class_support' else base[i][axis] for i in ids],float)
                    xt = x if axis=='active_bin_fraction' else np.log1p(x)
                    xc, dc = xt.copy(), delta.astype(float).copy()
                    for label in support:
                        mask = labels==label
                        xc[mask] -= xc[mask].mean(); dc[mask] -= dc[mask].mean()
                    associations.append(dict(dataset=dataset, seed=seed, left=left, right=right, axis=axis,
                        transform='identity' if axis=='active_bin_fraction' else 'log1p',
                        marginal_correlation=correlation(xt,delta), within_class_correlation=correlation(xc,dc),
                        note='descriptive paired correctness difference; no causal interpretation'))
                    for label in [-1]+sorted(support):
                        selected = np.ones(len(ids),bool) if label==-1 else labels==label
                        edges = np.quantile(x[selected],[.25,.5,.75])
                        bins = np.searchsorted(edges,x,side='right')+1
                        for quartile in range(1,5):
                            mask=selected & (bins==quartile)
                            if mask.any():
                                strata.append(dict(dataset=dataset, seed=seed, left=left, right=right, axis=axis,
                                    class_index=int(label), quartile=quartile, support=int(mask.sum()),
                                    paired_accuracy_delta=float(delta[mask].mean()),
                                    q25=float(edges[0]),q50=float(edges[1]),q75=float(edges[2])))
    write_csv(OUTPUT/'common_t16_per_seed.csv',summaries)
    write_csv(OUTPUT/'paired_strata_per_seed_class.csv',strata)
    write_csv(OUTPUT/'paired_associations.csv',associations)
    (OUTPUT/'sealed_sources.json').write_text(json.dumps(sources,indent=2)+'\n')


class LayerDiagnostics:
    def __init__(self, model):
        self.rows = {}; self.hooks=[]
        for name, module in model.named_modules():
            if _is_spiking(module):
                self.hooks.append(module.register_forward_hook(self.hook(name)))

    def hook(self,name):
        def collect(node, args, spikes):
            x=args[0]; flat=x.flatten(1)
            correction=torch.zeros((),device=x.device)
            if isinstance(node,LPSpikingNeuron):
                membrane=(node.effective_weight()@flat).view_as(x)
                if node.soft_reset_strategy=='exact_detached':
                    history=node._exact_detached_soft_reset_spikes(membrane)
                    correction=soft_reset_correction(history,node.log_tau,node.beta)
                margin=membrane-correction+node._bias_view(membrane)
            elif node.__class__.__name__=='PSN':
                margin=(node.weight@flat+node.bias).view_as(x)
            else:
                return
            row=self.rows.setdefault(name,dict(elements=0,spikes=0.,units=0,sustained_half=0.,sustained_nine_tenths=0.,
                consecutive_spikes=0.,consecutive_pairs=0,margin_sum=0.,margin_square_sum=0.,
                correction_sum=0.,correction_square_sum=0.,histogram=[0]*18))
            duty=spikes.float().mean(0)
            # Histogram includes tails: (-inf,-8], unit bins, and (8,inf).
            bins=torch.bucketize(margin,torch.arange(-8,9,device=x.device,dtype=x.dtype))
            hist=torch.bincount(bins.flatten(),minlength=18).cpu().tolist()
            row['histogram']=[a+b for a,b in zip(row['histogram'],hist)]
            row['elements']+=margin.numel(); row['spikes']+=spikes.sum().item()
            row['units']+=duty.numel(); row['sustained_half']+=(duty>=.5).sum().item()
            row['sustained_nine_tenths']+=(duty>=.9).sum().item()
            row['consecutive_spikes']+=(spikes[1:]*spikes[:-1]).sum().item()
            row['consecutive_pairs']+=spikes[1:].numel()
            row['margin_sum']+=margin.sum().item(); row['margin_square_sum']+=margin.square().sum().item()
            row['correction_sum']+=correction.sum().item(); row['correction_square_sum']+=correction.square().sum().item()
        return collect

    def close(self):
        for h in self.hooks: h.remove()


def layers():
    lock=(OUTPUT.parent/'gpu.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_no_training()
    for dataset in ['cifar10_dvs','dvs_gesture','hardvs']:
        for seed in SEEDS:
            for model_name in MODELS:
                target=OUTPUT/f'layers_{dataset}_{model_name}_T16_seed{seed}.json'
                if target.exists(): continue
                cfg,path=config_for(dataset,model_name,seed,t=16)
                cfg=replace(cfg,cache_frames=False,num_workers=4)
                seed_everything(seed)
                ds,split=build_dataset(cfg,'val')
                model,checkpoint,ckpath=_load_checkpoint(cfg)
                collector=LayerDiagnostics(model); seen=[]
                with torch.inference_mode():
                    for frames,labels,ids,stats in make_loader(ds,cfg,shuffle=False):
                        model(frames.to(cfg.device)); functional.reset_net(model); seen.extend(ids)
                collector.close()
                result=dict(created_at=utc_now(),dataset=dataset,model=model_name,seed=seed,time_steps=16,
                            split='validation',split_hash=split['split_hash'],samples=len(seen),
                            sample_ids_sha256=stable_hash(seen),checkpoint_sha256=file_sha256(ckpath),
                            layers=collector.rows,
                            interpretation='threshold-relative membrane (PSN logits); global histogram, no causal inference')
                target.write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({'complete':target.name,'samples':len(seen)}),flush=True)
                del model,collector; torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser(description=__doc__, epilog='Outputs: LPSN_EVIDENCE_ROOT/diagnostics. The default reads saved predictions only; --layers performs validation GPU inference.')
    parser.add_argument('--layers',action='store_true',help='Collect layer statistics by running the selected checkpoints on validation data.')
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    mode = 'layers' if args.layers else 'predictions'
    existing = list(OUTPUT.glob('layers_*.json')) if args.layers else [p for p in (OUTPUT/'common_t16_per_seed.csv', OUTPUT/'sealed_sources.json') if p.exists()]
    if existing:
        raise RuntimeError(f'{mode} evidence already exists; choose a fresh LPSN_EVIDENCE_ROOT')
    source_dir = OUTPUT / (mode + '_invocation')
    source_dir.mkdir(exist_ok=True)
    source_snapshot(source_dir, __file__)
    if args.layers: layers()
    else: predictions()

if __name__=='__main__':main()
