"""Summarize existing validation trajectories and the recorded scope reduction.

Read-only with respect to all training artifacts. This script does not train,
select a different checkpoint, or evaluate test data.
"""
import argparse
import math
import os
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Keep path handling local: this saved-log summary must not import the
# experiments package, whose initializer imports the training dependencies.
def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def portable_path(value):
    path = resolve_path(value).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


EVIDENCE = resolve_path(os.environ.get('LPSN_EVIDENCE_ROOT', 'reproduction_outputs'))
OUT = EVIDENCE / 'optimization'
DATASETS = ('cifar10_dvs', 'dvs_gesture')
ARCHITECTURES = ('vgg_small_half', 'spiking_resnet18')
MODELS = ('psn', 'lpsn_a1_lif_init', 'lpsn_a2_soft')
SEEDS = (42, 123, 256, 512, 1024)


def source(path):
    path = path.resolve()
    return {'path': portable_path(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_hash(value):
    """Match experiments.config.stable_hash without importing model code."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def integer(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {name}: {value!r}")
    return number


def sample_sd(values):
    """A sample SD is undefined for fewer than two observations."""
    return statistics.stdev(values) if len(values) >= 2 else None


def summarize_run(result_path):
    config_path = result_path.parent / 'config.yml'
    metrics_path = result_path.parent / 'metrics.jsonl'
    config = yaml.safe_load(config_path.read_text())
    result = json.loads(result_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Frozen configuration must be a mapping: {config_path}")
    config_hash = canonical_hash(config)
    if result.get('config_hash') != config_hash:
        raise ValueError(f"Frozen configuration/result hash mismatch: {config_path}")
    for key in ('schema_version', 'experiment_id', 'dataset', 'architecture', 'model',
                'seed', 'time_steps', 'condition', 'selection_metric'):
        if key not in config or key not in result or config[key] != result[key]:
            raise ValueError(f"Frozen configuration/result identity mismatch for {key}: {result_path}")
    if result.get('selection_split') != 'validation' or not result.get('completed_at'):
        raise ValueError(f"Missing completed validation-selected training record: {result_path}")
    selection_metric = config['selection_metric']
    if selection_metric not in ('macro_f1', 'accuracy'):
        raise ValueError(f"Unsupported recorded selection metric: {selection_metric!r}")
    epochs = integer(config['epochs'], 'configured epochs', minimum=1)
    patience = integer(config['patience'], 'configured patience')
    executed = integer(result['executed_epochs'], 'executed epochs', minimum=1)
    best = integer(result['best_epoch'], 'selected epoch', minimum=1)
    if not 1 <= best <= executed <= epochs:
        raise ValueError(f"Selected/executed epochs exceed completed-run bounds: {result_path}")
    trajectory = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if len(trajectory) != executed or [row['epoch'] for row in trajectory] != list(range(1, executed + 1)):
        raise ValueError(f"History must contain each completed epoch exactly once: {metrics_path}")

    # Verify the recorded selection and first stopping opportunity using the
    # train loop's metric tolerance and accuracy tie-break. No checkpoint is changed.
    best_primary = best_accuracy = -math.inf
    verified_best = stale = 0
    for epoch, row in enumerate(trajectory, start=1):
        if row.get('selection_metric') != selection_metric:
            raise ValueError(f"History selection metric differs at epoch {epoch}: {metrics_path}")
        primary = finite(row['validation'][selection_metric], 'validation selection metric')
        accuracy = finite(row['validation']['accuracy'], 'validation accuracy')
        finite(row['validation']['macro_f1'], 'validation macro-F1')
        improved = primary > best_primary + 1e-12 or (
            math.isclose(primary, best_primary, abs_tol=1e-12)
            and accuracy > best_accuracy + 1e-12
        )
        if row.get('improved') is not improved:
            raise ValueError(f"Recorded improvement differs at epoch {epoch}: {metrics_path}")
        if improved:
            best_primary, best_accuracy, verified_best, stale = primary, accuracy, epoch, 0
        else:
            stale += 1
        if patience > 0 and stale >= patience and epoch != executed:
            raise ValueError(f"History continues after configured patience was exhausted: {metrics_path}")
    if verified_best != best:
        raise ValueError(f"Recorded selected epoch differs from the logged selection rule: {result_path}")
    if not math.isclose(best_primary, finite(result['best_validation_value'], 'selected value'), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"Selected validation value differs from history: {result_path}")
    if not math.isclose(best_accuracy, finite(result['best_validation_accuracy'], 'selected accuracy'), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"Selected validation accuracy differs from history: {result_path}")

    hit_cap = executed == epochs
    patience_exhausted = patience > 0 and stale >= patience
    if not (hit_cap or patience_exhausted):
        raise ValueError(f"Completed record ends before either configured stopping condition: {result_path}")
    trigger = ('epoch_cap_and_patience' if hit_cap and patience_exhausted
               else 'epoch_cap' if hit_cap else 'patience')
    selected = trajectory[best - 1]
    trailing = [finite(row['validation']['macro_f1'], 'trailing validation macro-F1')
                for row in trajectory[-5:]]
    record = {
        'dataset': config['dataset'], 'architecture': config['architecture'],
        'model': config['model'], 'seed': config['seed'], 'time_steps': config['time_steps'],
        'selection_metric': selection_metric,
        'configured_epochs': epochs, 'configured_patience': patience,
        'patience_enabled': patience > 0,
        'selected_epoch': best, 'executed_epochs': executed,
        'epochs_after_selected': executed - best,
        'hit_epoch_cap': hit_cap, 'patience_exhausted': patience_exhausted,
        'completion_trigger': trigger,
        'selected_validation_macro_f1': selected['validation']['macro_f1'],
        'selected_validation_accuracy': selected['validation']['accuracy'],
        'online_training_accuracy_at_selected_epoch': finite(selected['train']['accuracy'], 'selected-epoch online training accuracy'),
        'online_training_minus_validation_accuracy': float(selected['train']['accuracy']) - float(selected['validation']['accuracy']),
        'last_validation_macro_f1': trajectory[-1]['validation']['macro_f1'],
        'last5_validation_macro_f1_n_epochs': len(trailing),
        'last5_validation_macro_f1_sd': sample_sd(trailing),
        'config_hash': config_hash, 'config_path': portable_path(config_path),
        'train_result_path': portable_path(result_path), 'metrics_path': portable_path(metrics_path),
    }
    return record, [source(config_path), source(result_path), source(metrics_path)]


def stopping_settings(records):
    counts = defaultdict(int)
    for row in records:
        counts[row['configured_epochs'], row['configured_patience']] += 1
    return [{'epochs': epochs, 'patience': patience, 'patience_enabled': patience > 0, 'runs': count}
            for (epochs, patience), count in sorted(counts.items())]


def stopping_description(records):
    settings = []
    for setting in stopping_settings(records):
        patience_text = (f"patience {setting['patience']}" if setting['patience_enabled']
                         else f"patience disabled (configured value {setting['patience']})")
        settings.append(f"{setting['runs']} runs with epoch cap {setting['epochs']} and {patience_text}")
    counts = {trigger: sum(row['completion_trigger'] == trigger for row in records)
              for trigger in ('patience', 'epoch_cap', 'epoch_cap_and_patience')}
    return ('Configured stopping settings: ' + '; '.join(settings) + '. '
            f"{counts['patience']} runs ended when patience was exhausted before the epoch cap; "
            f"{counts['epoch_cap']} reached their epoch cap without exhausting patience; "
            f"{counts['epoch_cap_and_patience']} met both conditions at the final epoch. "
            'Both conditions are recorded for a tie; nonpositive patience disables early stopping. ')


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Writes CSV, provenance, and LaTeX summaries under LPSN_EVIDENCE_ROOT/optimization.')
    parser.add_argument('--runs-root', type=Path, help='Optional architecture-run root; otherwise use the frozen protocol run paths.')
    parser.add_argument('--scope-record-dir', type=Path, help='Optional directory of original historical scope records; these are audited separately from the fresh run summaries.')
    args = parser.parse_args()
    selection_sources = []
    if args.runs_root:
        result_paths = sorted(resolve_path(args.runs_root).rglob('train_result.json'))
    else:
        protocol_path = EVIDENCE / 'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        if canonical_hash({key: value for key, value in protocol.items() if key != 'protocol_hash'}) != protocol.get('protocol_hash'):
            raise ValueError(f"Frozen protocol hash mismatch: {protocol_path}")
        result_paths = sorted(resolve_path(row['run_dir']) / 'train_result.json'
                              for row in protocol['runs'] if row['study'] == 'architecture')
        selection_sources.append(source(protocol_path))
    records, sources = [], selection_sources
    for result_path in result_paths:
        record, run_sources = summarize_run(result_path)
        records.append(record)
        sources.extend(run_sources)
    expected = {(dataset, architecture, model, seed)
                for dataset in DATASETS for architecture in ARCHITECTURES
                for model in MODELS for seed in SEEDS}
    identities = [(row['dataset'], row['architecture'], row['model'], row['seed']) for row in records]
    if len(records) != 60 or len(set(identities)) != 60 or set(identities) != expected:
        raise ValueError('Expected exactly the 60 distinct architecture-study runs across both datasets, both architectures, three models, and five seeds')
    groups = defaultdict(list)
    for row in records:
        groups[row['dataset'], row['architecture'], row['model']].append(row)
    metrics = ['selected_epoch', 'executed_epochs', 'selected_validation_macro_f1',
               'selected_validation_accuracy', 'online_training_accuracy_at_selected_epoch',
               'online_training_minus_validation_accuracy', 'last_validation_macro_f1',
               'last5_validation_macro_f1_n_epochs', 'last5_validation_macro_f1_sd']
    summaries = []
    for (dataset, architecture, model), rows in sorted(groups.items()):
        summary = {
            'dataset': dataset, 'architecture': architecture, 'model': model,
            'n_seeds': len(rows),
            'configured_epochs_values': json.dumps(sorted({row['configured_epochs'] for row in rows})),
            'configured_patience_values': json.dumps(sorted({row['configured_patience'] for row in rows})),
            'selection_metrics': json.dumps(sorted({row['selection_metric'] for row in rows})),
            'patience_disabled_count': sum(not row['patience_enabled'] for row in rows),
            'hit_epoch_cap_count': sum(row['hit_epoch_cap'] for row in rows),
            'patience_exhausted_count': sum(row['patience_exhausted'] for row in rows),
            'epoch_cap_only_count': sum(row['completion_trigger'] == 'epoch_cap' for row in rows),
            'patience_only_count': sum(row['completion_trigger'] == 'patience' for row in rows),
            'epoch_cap_and_patience_count': sum(row['completion_trigger'] == 'epoch_cap_and_patience' for row in rows),
        }
        for metric in metrics:
            values = [row[metric] for row in rows if row[metric] is not None]
            summary[metric + '_n_available'] = len(values)
            summary[metric + '_mean'] = statistics.mean(values) if values else None
            summary[metric + '_sd'] = sample_sd(values)
        summaries.append(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'per_seed.csv', records)
    write_csv(OUT / 'summary.csv', summaries)
    provenance = {
        'schema_version': 2,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'script': source(Path(__file__)), 'sources': sources,
        'training_runs_read': len(records), 'new_training': False, 'new_test_evaluation': False,
        'configuration_validation': 'Each frozen config.yml is hashed using the same canonical JSON SHA256 as experiments.config.stable_hash and matched to train_result.json identity/config_hash. Logged selection and first stopping opportunity are verified against that configuration.',
        'configured_stopping_settings': stopping_settings(records),
        'selection_metrics': sorted({row['selection_metric'] for row in records}),
        'stopping_rule': 'After each epoch, enabled patience is checked before the loop can terminate at its epoch cap. Both hit_epoch_cap and patience_exhausted are recorded when they coincide. Nonpositive patience disables early stopping.',
        'interpretation': 'Descriptive summary of the supplied validation-selected checkpoints and executed stopping rule; no checkpoint reselection.',
        'training_accuracy_caveat': 'Training accuracy is collected online while weights change within the selected epoch. Validation accuracy is measured after that epoch. Their difference is not a same-checkpoint train-versus-validation evaluation.',
        'stopping_caveat': 'Patience exhaustion and absence of epoch-cap hits do not demonstrate convergence to an optimum or equal optimization quality across neurons or architectures.',
        'units': 'Accuracy and macro-F1 are fractions in CSV; LaTeX tables display percentages. Across-seed SDs are sample SDs over available values, with n_available recorded. Each last5 SD uses the final min(5, executed_epochs) epochs, with n_epochs recorded. A sample SD is undefined with fewer than two observations; undefined CSV values are blank and excluded from group aggregates.',
    }
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

    if args.scope_record_dir:
        scope_root = resolve_path(args.scope_record_dir)
        record_path = scope_root / 'scope_reduction_record.md'
        continuation_path = scope_root / 'continuation_status.pre_scope_reduction.json'
        audit_path = scope_root / 'audit_index.pre_scope_reduction.jsonl'
        removed_path = scope_root / 'declarations_removed.txt'
        observed_path = scope_root / 'dropped_group_observed_results.txt'
        record_text = record_path.read_text()
        assert 'Date: 2026-09-15' in record_text and '562.6' in record_text and '326 GB' in record_text
        audit_rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
        scope = {
            'scope': 'historical scope provenance; not a statement about fresh reproduction completion',
            'sources': [source(path) for path in [record_path, continuation_path, audit_path, removed_path, observed_path]],
            'record_date': '2026-09-15',
            'continuation_started_at': json.loads(continuation_path.read_text())['started_at'],
            'prescope_audit_entries': len(audit_rows),
            'prescope_first_audited_at': min(row['audited_at'] for row in audit_rows),
            'prescope_last_audited_at': max(row['audited_at'] for row in audit_rows),
            'declared_configurations': 695, 'retained_configurations': 410,
            'removed_configurations': len(removed_path.read_text().splitlines()),
            'retained_physical_runs': 335, 'retained_deterministic_aliases': 75,
            'recorded_resource_rationale': {
                'status': 'Values reported in the dated record; the underlying runtime and storage estimates are not independently remeasured here.',
                'completed_hardvs_runs': 42, 'continuous_span_hours': 562.6, 'concurrent_workers': 2,
                'reported_blended_hours_per_run': 13.7, 'reported_t32_hours_per_run_approx': 22,
                'remaining_physical_hardvs_runs': 220,
                'estimated_full_remainder_gpu_hours': [4400, 4900],
                'estimated_full_corruption_cache_tb': 3.2, 'recorded_available_disk_gb': 326,
            },
            'observed_removed_group': {'condition': 'amplitude_scale_0.25', 'dataset': 'hardvs', 'time_steps': 32,
                                       'a1_completed_and_test_evaluated': 5, 'a1_mean_accuracy': 0.1658,
                                       'a2_completed_not_test_evaluated': 2, 'psn_completed': 0},
            'interpretation': 'The record documents a resource rationale after some outcomes had been observed. It does not demonstrate outcome-independent selection. The full-grid cost estimate does not prove that a smaller targeted completion was infeasible.',
            'unsupported_original_inference': 'The original record describes exclusion as conservative because A1 exceeds its own clean T16 accuracy. This compares different conditions and temporal lengths without complete comparators; it does not establish the direction of selection bias.',
        }
        assert scope['removed_configurations'] == 285
        (OUT / 'scope_record_audit.json').write_text(json.dumps(scope, indent=2) + '\n')

    datasets = {'cifar10_dvs': 'CIFAR10-DVS', 'dvs_gesture': 'DVS-Gesture'}
    architectures = {'spiking_resnet18': 'ResNet-18', 'vgg_small_half': 'Half VGG'}
    models = {'psn': 'PSN', 'lpsn_a1_lif_init': 'A1', 'lpsn_a2_soft': 'A2'}
    lookup = {(row['dataset'], row['architecture'], row['model']): row for row in summaries}
    order = ['psn', 'lpsn_a1_lif_init', 'lpsn_a2_soft']
    def cifar_values(architecture, metric):
        return '/'.join(f'{100*lookup["cifar10_dvs", architecture, model][metric + "_mean"]:.2f}' for model in order)
    selected_scores = (
        r'In PSN/A1/A2 order, selected CIFAR10-DVS validation macro-F1 means are '
        + cifar_values('spiking_resnet18', 'selected_validation_macro_f1') + r'\% on ResNet-18 and '
        + cifar_values('vgg_small_half', 'selected_validation_macro_f1') + r'\% on half-width VGG. On ResNet-18, online training accuracy during the selected epoch is '
        + cifar_values('spiking_resnet18', 'online_training_accuracy_at_selected_epoch') + r'\%, whereas end-epoch validation accuracy is '
        + cifar_values('spiking_resnet18', 'selected_validation_accuracy') + r'\%.'
    )
    tex = [r'\subsection*{Validation trajectories}',
           f'This summary reads {len(records)} saved architecture-run logs without retraining, checkpoint reselection, or test evaluation. '
           + stopping_description(records) +
           'Stopping behavior does not establish optimality or equally effective optimization. Values below are means over five seeds. The recorded selection metric for each run is retained in the CSV files.',
           r'\small\begin{longtable}{llllrrr}', r'\toprule',
           r'Dataset & Backbone & Neuron & Val. F1 (\%) & Best epoch & Executed & Gap (pp)\\\midrule\endhead']
    for row in summaries:
        tex.append(' & '.join([datasets[row['dataset']], architectures[row['architecture']], models[row['model']],
                               f'{100*row["selected_validation_macro_f1_mean"]:.2f}',
                               f'{row["selected_epoch_mean"]:.1f}', f'{row["executed_epochs_mean"]:.1f}',
                               f'{100*row["online_training_minus_validation_accuracy_mean"]:.2f}']) + r'\\')
    tex.extend([r'\bottomrule\end{longtable}\normalsize', selected_scores,
                r'The last column is a difference in percentage points between online training accuracy collected while weights change during the selected epoch and validation accuracy measured after that epoch. It is not a same-checkpoint training-set evaluation. These descriptive gaps do not identify a unique explanation of performance. Individual seeds, configured stopping settings, final validation scores, and variability over up to the final five epochs are retained in CSV files with source hashes. Trailing-window counts are explicit; a sample standard deviation is undefined for a single epoch and is left blank rather than reported as zero.'])
    if args.scope_record_dir:
        tex.append(r'A separate scope-record audit preserves the historical 15 September 2026 decision and its observed-outcome limitations. Historical records are not relabeled as fresh reproduction results.')
    (OUT/'optimization_summary.tex').write_text('\n\n'.join(tex)+'\n')
    print(f'Summarized {len(records)} saved architecture trajectories (historical scope audit only if explicitly requested) in {OUT}')


if __name__ == '__main__':
    main()
