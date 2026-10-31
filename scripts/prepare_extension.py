"""Record configurations and source hashes for architecture and component studies.

This command prepares the extension before its training stage. The recorded
timestamp does not establish when results from other studies were inspected.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.config import ExperimentConfig, load_config, stable_hash, file_sha256, utc_now
from experiments.data import load_split_manifest
from experiments.release_paths import EVIDENCE, resolve_path

SEEDS = [42, 123, 256, 512, 1024]
DATASETS = ['cifar10_dvs', 'dvs_gesture']
MODELS = ['psn', 'lpsn_a1_lif_init', 'lpsn_a2_soft']
COMPONENTS = ['fixed_diagonal', 'fixed_causal_uniform', 'lpsn_a3_kernel',
              'lpsn_fixed_kernel_residual', 'lpsn_a4a_frozen_bias']
EXPERIMENT_IDS = {'architecture': 'architecture_study', 'components': 'component_study'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Write fresh configuration and protocol freezes; does not train or evaluate')
    parser.add_argument('--baseline-configs', type=Path, default=PROJECT / 'generated_configs')
    parser.add_argument('--configs-root', type=Path, default=PROJECT / 'generated_configs_extension')
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'runs_extension')
    parser.add_argument('--frame-cache-root', type=Path, default=PROJECT / 'frame_cache')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'training_runs': 110, 'architecture_runs': 60, 'component_runs': 50, 'evidence': str(EVIDENCE), 'action': 'preview only; add --execute to freeze source/configurations'}))
        return 0
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    protocol_path = EVIDENCE / 'protocol.json'
    freeze_path = EVIDENCE / 'implementation_freeze.json'
    if protocol_path.exists() or freeze_path.exists():
        raise RuntimeError('Protocol or implementation freeze already exists; retain it and select a new evidence root and separate configuration/run directories for a new reproduction')
    declarations = []
    source_configs = []
    for dataset in DATASETS:
        for seed in SEEDS:
            source = resolve_path(args.baseline_configs) / f'temporal_native/{dataset}/psn/T32/seed_{seed}.yml'
            base = load_config(source)
            manifest = load_split_manifest(resolve_path(base.split_manifest))
            source_configs.append({'path': str(source), 'sha256': file_sha256(source),
                                   'split_sha256': file_sha256(resolve_path(base.split_manifest)),
                                   'split_hash': manifest['split_hash']})
            cells = [('architecture', a, m) for a in ['vgg_small_half', 'spiking_resnet18'] for m in MODELS]
            cells += [('components', 'vgg_small', m) for m in COMPONENTS]
            for study, architecture, model in cells:
                template_path = PROJECT / f'configs/extension/{study}/{dataset}/{architecture}/{model}/seed_{seed}.yml'
                template = yaml.safe_load(template_path.read_text())
                if (template['dataset'], template['architecture'], template['model'], template['seed']) != (dataset, architecture, model, seed):
                    raise RuntimeError(f'Template metadata differs: {template_path}')
                if template['experiment_id'] != EXPERIMENT_IDS[study]:
                    raise RuntimeError(f'Template study identifier differs: {template_path}')
                raw = {**asdict(base), 'experiment_id': EXPERIMENT_IDS[study],
                       'architecture': architecture, 'model': model,
                       'frame_cache_root': str(resolve_path(args.frame_cache_root)),
                       'output_root': str(resolve_path(args.output_root))}
                candidate = ExperimentConfig.from_mapping(raw)
                if any((resolve_path(candidate.run_dir) / name).exists() for name in ('metrics.jsonl', 'train_result.json', 'checkpoint_best.pt', 'test_evaluation_ledger.jsonl')):
                    raise RuntimeError(f'Fresh preparation requires unused run directories; choose a separate --output-root: {candidate.run_dir}')
                config_path = resolve_path(args.configs_root) / f'{study}/{dataset}/{architecture}/{model}/seed_{seed}.yml'
                config_path.parent.mkdir(parents=True, exist_ok=True)
                value = yaml.safe_dump(raw, sort_keys=True)
                if config_path.exists() and config_path.read_text() != value:
                    raise RuntimeError(f'Refusing to change frozen config: {config_path}')
                config_path.write_text(value)
                config = load_config(config_path)
                declarations.append({'config': str(config_path), 'sha256': file_sha256(config_path),
                                     'config_hash': config.config_hash, 'run_dir': str(config.run_dir),
                                     'study': study, 'dataset': dataset, 'architecture': architecture,
                                     'model': model, 'seed': seed})
    contrasts = []
    for dataset in DATASETS:
        for architecture in ['vgg_small', 'vgg_small_half', 'spiking_resnet18']:
            for left, right in [('lpsn_a1_lif_init', 'psn'), ('lpsn_a2_soft', 'psn'), ('lpsn_a2_soft', 'lpsn_a1_lif_init')]:
                contrasts.append(dict(family='architecture_macro_f1', dataset=dataset, architecture=architecture, left=left, right=right))
        for left, right in [('lpsn_a1_lif_init', 'fixed_diagonal'), ('lpsn_a1_lif_init', 'fixed_causal_uniform'),
                            ('lpsn_a3_kernel', 'lpsn_a1_lif_init'), ('lpsn_fixed_kernel_residual', 'lpsn_a1_lif_init'),
                            ('lpsn_a4a_frozen_bias', 'lpsn_a3_kernel'), ('lpsn_a4a_frozen_bias', 'lpsn_fixed_kernel_residual')]:
            contrasts.append(dict(family='components_macro_f1', dataset=dataset, architecture='vgg_small', left=left, right=right))
    protocol = {'schema_version': 1, 'created_at': utc_now(), 'runs': declarations,
                'new_training_runs': len(declarations), 'source_configs': source_configs,
                'baseline_configs_root': str(resolve_path(args.baseline_configs)),
                'contrasts': contrasts, 'primary_metric': 'macro_f1', 'accuracy': 'descriptive',
                'tests': 'paired exact two-sided signed-rank on five seed differences; Holm within each declared family',
                'bootstrap': {'replicates': 2000, 'rng_seed': 20240723, 'percentiles': [2.5, 97.5],
                              'units': 'crossed training seed and shared corruption draw; no sample resampling'},
                'energy': {'checkpoint_seed': 42, 'selection': 'fixed seed, not selected by observed performance',
                           'batches': [1, 16], 'warmup': 100, 'blocks': 10, 'minimum_seconds': 5,
                           'minimum_forward_calls': 100, 'inputs': 'first 16 frozen validation examples',
                           'counter_units': 'millijoules', 'reported_units': 'gross GPU joules/sample'},
                'hardvs_training_runs': 0, 'original_test_families': 'unchanged',
                'cache_policy': 'separate extension cache; baseline cache is unchanged',
                'reproduction_scope': 'fresh reproduction; historical evidence and numerical results are not regenerated by preparation',
                'architecture_specification': {'vgg_small_half': '32/64/128/128/128, hidden 256; unchanged blocks',
                    'spiking_resnet18': '2-channel 7x7 stride-2 stem, 3x3 stride-2 maxpool; stages 64/128/256/512, 2 post-addition spiking basic blocks each; 1x1 projection+BN at stage changes; global average pool, spiking classifier, engine time average'}}
    assert len(declarations) == 110 and len({r['run_dir'] for r in declarations}) == 110
    protocol['protocol_hash'] = stable_hash(protocol)
    sources = sorted(set(PROJECT.glob('*.py')) | set((PROJECT / 'experiments').rglob('*.py')) | set((PROJECT / 'scripts').rglob('*.py')))
    sources += [PROJECT / name for name in ('requirements.txt', 'environment.yml')]
    freeze = {'schema_version': 1, 'created_at': utc_now(),
              'protocol_hash': protocol['protocol_hash'],
              'source_files': {str(path.relative_to(PROJECT)): file_sha256(path) for path in sources}}
    freeze['manifest_hash'] = stable_hash(freeze)
    freeze_path.write_text(json.dumps(freeze, indent=2) + '\n')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'runs': 110, 'families': {'architecture_macro_f1':18, 'components_macro_f1':12}}))

if __name__ == '__main__':
    main()
