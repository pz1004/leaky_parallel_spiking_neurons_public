"""Verify configuration, checkpoint selection, and evaluation integrity for extension runs."""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from experiments.config import file_sha256,stable_hash,load_config,utc_now
from experiments.data import load_split_manifest
from experiments.release_paths import EVIDENCE, resolve_path


def lines(path):return [json.loads(l) for l in path.read_text().splitlines() if l]
def resolve(path):return resolve_path(path)


def audit_run(cfg):
    folder=resolve(cfg.run_dir)
    result=json.loads((folder/'train_result.json').read_text())
    split=load_split_manifest(resolve(cfg.split_manifest))
    history=lines(folder/'metrics.jsonl')
    ledger=lines(folder/'test_evaluation_ledger.jsonl')
    metrics=lines(folder/'locked_test_metrics.jsonl')
    assert len(ledger)==len(metrics)==1, 'Exactly one clean test evaluation required'
    metric=metrics[0]
    assert metric['condition']=='clean' and metric['severity']==0 and metric['draw']==0
    assert result['selection_split']=='validation' and result['selection_metric']=='macro_f1'
    assert result['architecture']==metric['architecture']==cfg.architecture
    assert result['config_hash']==metric['config_hash']==cfg.config_hash
    assert result['split_hash']==metric['split_hash']==split['split_hash']
    assert [r['epoch'] for r in history]==list(range(1,result['executed_epochs']+1))
    best_primary=best_accuracy=-math.inf;best=0
    for row in history:
        val=row['validation'];primary=val['macro_f1'];accuracy=val['accuracy']
        improved=primary>best_primary+1e-12 or (math.isclose(primary,best_primary,abs_tol=1e-12) and accuracy>best_accuracy+1e-12)
        assert row['improved']==improved
        if improved:best_primary,best_accuracy,best=primary,accuracy,row['epoch']
    assert best==result['best_epoch']==metric['checkpoint_epoch']
    assert result['executed_epochs']==cfg.epochs or result['executed_epochs']==best+cfg.patience
    checkpoint_path=folder/'checkpoint_best.pt';digest=file_sha256(checkpoint_path)
    assert digest==result['checkpoint_sha256']==metric['checkpoint_sha256']==ledger[0]['checkpoint_sha256']
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==best and checkpoint['config_hash']==cfg.config_hash
    assert checkpoint['selection_split']=='validation' and checkpoint['selection_metric']=='macro_f1'
    samples=lines(resolve(metric['per_sample_path']))
    expected={r['id']:r['label'] for r in split['splits']['test']}
    assert len(samples)==len(expected)==metric['samples']
    assert {r['sample_id']:r['label'] for r in samples}==expected
    manifest=json.loads(resolve(metric['evaluation_manifest_path']).read_text())
    assert stable_hash({k:v for k,v in manifest.items() if k!='manifest_hash'})==manifest['manifest_hash']
    assert manifest['artifacts']['metrics_payload_sha256']==stable_hash(metric)
    assert manifest['artifacts']['per_sample_sha256']==file_sha256(resolve(metric['per_sample_path']))
    confusion_path=resolve(manifest['artifacts']['confusion_path'])
    assert manifest['artifacts']['confusion_sha256']==file_sha256(confusion_path)
    confusion=np.load(confusion_path);check=np.zeros_like(confusion)
    for row in samples:check[row['label'],row['prediction']]+=1
    np.testing.assert_array_equal(confusion,check)
    # Git status is retained as provenance. Regenerated tracked split manifests
    # may legitimately make the checkout dirty; the queue freezes source content.
    for name in ['run_manifest.json', 'environment.json']:
        if 'git_dirty' not in json.loads((folder/name).read_text()):
            raise AssertionError(f'Missing source provenance: {name}')
    return dict(status='complete',checked_at=utc_now(),config_hash=cfg.config_hash,checkpoint_sha256=digest,
                split_hash=split['split_hash'],test_evaluations=1,samples=len(samples),selected_epoch=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=EVIDENCE / 'run_audit.json')
    args = parser.parse_args()
    args.output = resolve(args.output)
    evidence=EVIDENCE
    protocol=json.loads((evidence/'protocol.json').read_text());complete=[];pending=[];errors=[]
    if stable_hash({k:v for k,v in protocol.items() if k!='protocol_hash'}) != protocol['protocol_hash']:
        raise RuntimeError('Protocol hash mismatch')
    if len(protocol['runs']) != 110 or len({entry['run_dir'] for entry in protocol['runs']}) != 110:
        raise RuntimeError('Expected 110 distinct extension declarations')
    for entry in protocol['runs']:
        config_path = resolve(entry['config'])
        if file_sha256(config_path) != entry['sha256']:
            errors.append({'config': entry['config'], 'error': 'Frozen config hash mismatch'});continue
        cfg=load_config(config_path)
        if not (resolve(cfg.run_dir)/'test_evaluation_ledger.jsonl').exists():pending.append(entry['config']);continue
        try:complete.append({'config':entry['config'],**audit_run(cfg)})
        except Exception as e:errors.append({'config':entry['config'],'error':repr(e)})
    report=dict(status='complete' if len(complete)==110 and not errors else 'incomplete',complete=complete,pending=pending,errors=errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'complete':len(complete),'pending':len(pending),'errors':errors}))
    return 0 if report['status']=='complete' else 2

if __name__=='__main__':raise SystemExit(main())
