"""Plot saved crossed-bootstrap intervals and record source/output hashes.

Install requirements-plotting.txt before invoking this optional command.
Input is the CSV produced by experiments.cli analyze; no inference is run.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.release_paths import EVIDENCE, portable_path, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=PROJECT / 'aggregates/results.csv')
    parser.add_argument('--output-dir', type=Path, default=EVIDENCE / 'figures')
    args = parser.parse_args()
    source, output = resolve_path(args.input), resolve_path(args.output_dir)
    source_bytes = source.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    analysis_manifest = source.with_suffix('.manifest.json')
    manifest_bytes = analysis_manifest.read_bytes()
    metadata = json.loads(manifest_bytes)
    if 'crossed' not in str(metadata.get('bootstrap', '')).lower():
        raise ValueError('Plot input must be accompanied by a crossed-bootstrap analysis manifest')
    manifest_payload = {key: value for key, value in metadata.items() if key != 'manifest_hash'}
    manifest_digest = hashlib.sha256(json.dumps(manifest_payload, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()
    if metadata.get('manifest_hash') != manifest_digest:
        raise ValueError('Analysis manifest hash mismatch')
    artifacts = metadata.get('artifacts', {})
    summary = artifacts.get('summary_csv') if isinstance(artifacts, dict) else None
    if not isinstance(summary, dict) or not summary.get('sha256'):
        raise ValueError('Analysis manifest lacks a summary CSV hash. Generate new aggregates from saved records with experiments.cli analyze before plotting.')
    if summary['sha256'] != source_digest:
        raise ValueError('Summary CSV hash mismatch: the CSV does not match its analysis manifest')
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode('utf-8'))))
    if metadata.get('groups') != len(rows):
        raise ValueError('Analysis manifest group count does not match the supplied CSV')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    models = ['psn', 'lpsn_a1_lif_init', 'lpsn_a2_soft']
    names = ['PSN', 'A1', 'A2']
    colors = ['#555555', '#0072B2', '#D55E00']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    paths = []
    families = [
        ('sensor_corruptions', ['event_dropout', 'background_activity', 'polarity_flip',
                                'temporal_jitter', 'packet_loss', 'hot_pixels']),
        ('temporal_diagnostics', ['reverse', 'permute', 'causal_prefix', 'temporal_sparse']),
    ]
    for dataset in ['cifar10_dvs', 'dvs_gesture']:
        for family, conditions in families:
            fig, axes = plt.subplots(2, 3 if len(conditions) == 6 else 2,
                                     figsize=(12, 6.6), sharey=True, layout='constrained')
            for ax, condition in zip(axes.flat, conditions):
                for model, label, color in zip(models, names, colors):
                    selected = sorted(
                        [r for r in rows if r['experiment_id'] == 'temporal_native'
                         and r['dataset'] == dataset and r['model'] == model
                         and r.get('architecture', 'vgg_small') == 'vgg_small'
                         and r['condition'] == condition and r['time_steps'] == '32'],
                        key=lambda r: float(r['severity']))
                    if not selected:
                        raise ValueError(f'Missing plotted condition: {dataset}/{model}/{condition}')
                    x = np.array([float(r['severity']) for r in selected])
                    y = 100 * np.array([float(r['accuracy_mean']) for r in selected])
                    lo = 100 * np.array([float(r['accuracy_ci_low']) for r in selected])
                    hi = 100 * np.array([float(r['accuracy_ci_high']) for r in selected])
                    if len(set(x)) != len(x) or not all(np.isfinite(values).all() for values in [x, y, lo, hi]):
                        raise ValueError(f'Duplicate severity or nonfinite plotted value: {dataset}/{model}/{condition}')
                    if np.any(lo > hi) or any(np.any((values < 0) | (values > 100)) for values in [y, lo, hi]):
                        raise ValueError(f'Invalid accuracy or interval endpoints: {dataset}/{model}/{condition}')
                    ax.plot(x, y, color=color, marker='o', label=label, linewidth=1.5)
                    # Percentile intervals need not contain the observed mean.
                    # Draw their endpoints directly, without negative error sizes.
                    ax.vlines(x, lo, hi, color=color, linewidth=1.2)
                    ax.plot(x, lo, '_', color=color, markersize=6)
                    ax.plot(x, hi, '_', color=color, markersize=6)
                ax.set_title(condition.replace('_', ' '))
                ax.set_xlabel('Retention' if condition in ['causal_prefix', 'temporal_sparse'] else 'Severity')
                ax.set_ylim(0, 100)
                ax.grid(alpha=.2)
            for ax in axes[:, 0]:
                ax.set_ylabel('Accuracy (%)')
            axes.flat[0].legend(loc='best', frameon=False)
            title = 'CIFAR10-DVS' if dataset == 'cifar10_dvs' else 'DVS-Gesture'
            fig.suptitle(title + ' — crossed seed × shared-draw bootstrap (95%)')
            path = output / f'{dataset}_{family}.png'
            fig.savefig(path, dpi=220)
            plt.close(fig)
            paths.append(path)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'figure_manifest.json').write_text(json.dumps({
        'input': portable_path(source), 'input_sha256': source_digest,
        'analysis_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'outputs': {portable_path(path): digest(path) for path in paths},
        'resampling': 'crossed seed x shared draw',
    }, indent=2) + '\n')
    print(json.dumps({'figures': len(paths), 'output_dir': str(output)}))


if __name__ == '__main__':
    main()
