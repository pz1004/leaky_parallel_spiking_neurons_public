"""Preview or explicitly execute the frozen 110-run extension queue."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.config import file_sha256, stable_hash, utc_now, load_config
from experiments.release_paths import EVIDENCE, resolve_path
from scripts.audit_extension import audit_run


def verify_freeze(protocol: dict, freeze: dict) -> None:
    if stable_hash({k: v for k, v in protocol.items() if k != 'protocol_hash'}) != protocol['protocol_hash']:
        raise RuntimeError('Protocol hash mismatch')
    if stable_hash({k: v for k, v in freeze.items() if k != 'manifest_hash'}) != freeze['manifest_hash']:
        raise RuntimeError('Implementation-freeze hash mismatch')
    if freeze['protocol_hash'] != protocol['protocol_hash']:
        raise RuntimeError('Implementation freeze belongs to another protocol')
    if len(protocol['runs']) != 110 or len({row['run_dir'] for row in protocol['runs']}) != 110:
        raise RuntimeError('Expected 110 distinct extension declarations')
    for source, digest in freeze['source_files'].items():
        if file_sha256(resolve_path(source)) != digest:
            raise RuntimeError(f'Frozen implementation changed: {source}')
    for source in protocol['source_configs']:
        source_path = resolve_path(source['path'])
        if file_sha256(source_path) != source['sha256']:
            raise RuntimeError(f'Frozen baseline config changed: {source_path}')
        baseline = load_config(source_path)
        if file_sha256(resolve_path(baseline.split_manifest)) != source['split_sha256']:
            raise RuntimeError(f'Frozen split manifest changed: {baseline.split_manifest}')
    for entry in protocol['runs']:
        if file_sha256(resolve_path(entry['config'])) != entry['sha256']:
            raise RuntimeError(f'Frozen config changed: {entry["config"]}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Run training, locked-test evaluation, and synthetic-input timing')
    args = parser.parse_args()
    protocol = json.loads((EVIDENCE / 'protocol.json').read_text())
    freeze = json.loads((EVIDENCE / 'implementation_freeze.json').read_text())
    verify_freeze(protocol, freeze)
    runs = sorted(protocol['runs'], key=lambda row: (row['dataset'] != 'dvs_gesture', row['study'], row['architecture'], row['model'], row['seed']))
    if not args.execute:
        print(json.dumps({'runs': len(runs), 'evidence': str(EVIDENCE), 'execution': 'disabled; add --execute to train/evaluate/benchmark'}, indent=2))
        return 0
    status = {'started_at': utc_now(), 'pid': os.getpid(), 'expected_runs': len(runs), 'complete': [], 'state': 'waiting_for_gpu_lock'}

    def save() -> None:
        status['updated_at'] = utc_now()
        temporary = EVIDENCE / 'execution_status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(EVIDENCE / 'execution_status.json')

    environment = os.environ.copy()
    environment.setdefault('OMP_NUM_THREADS', '4')
    environment.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    environment.pop('LPSN_READONLY_FRAME_CACHE', None)
    # The caller controls CUDA_VISIBLE_DEVICES; no physical GPU index is forced.
    # Energy/layer diagnostics are independent stages, not training prerequisites.
    save()
    with (EVIDENCE / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            for entry in runs:
                verify_freeze(protocol, freeze)
                config_path = resolve_path(entry['config'])
                cfg = load_config(config_path)
                folder = resolve_path(cfg.run_dir)
                status.update(state='running', active=entry)
                save()
                log = EVIDENCE / 'training_logs' / f'{cfg.experiment_id}_{cfg.dataset}_{cfg.architecture}_{cfg.model}_{cfg.seed}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('a') as stream:
                    for phase in ('train', 'evaluate'):
                        marker = folder / ('train_result.json' if phase == 'train' else 'test_evaluation_ledger.jsonl')
                        if marker.exists():
                            continue
                        if phase == 'train' and (folder / 'metrics.jsonl').exists():
                            raise RuntimeError(f'Partial run preserved; explicit recovery needed: {folder}')
                        subprocess.run([sys.executable, '-u', '-m', 'experiments.cli', phase, '--config', str(config_path)], cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                    if not (folder / 'benchmark.json').exists():
                        subprocess.run([sys.executable, '-u', '-m', 'experiments.cli', 'benchmark', '--config', str(config_path), '--output', str(folder / 'benchmark.json')], cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
                audit = audit_run(cfg)
                (folder / 'extension_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
                status['complete'].append(entry['config'])
                save()
            status.update(state='training_complete', active=None)
            save()
        except Exception as error:
            status.update(state='failed', error=repr(error))
            save()
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
