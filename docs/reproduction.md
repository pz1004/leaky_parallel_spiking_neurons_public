# Reproduction Workflow

This guide covers dataset preparation, baseline and extension studies,
validation diagnostics, GPU measurements, aggregation, and evidence integrity.
See the [project README](../README.md) for installation and minimal usage.

The recorded environment uses Python 3.12, PyTorch 2.10.0, torchvision 0.25.0,
CUDA 12.8, and SpikingJelly commit `9b87d38`.

Run all commands below from the repository root, not from `docs/`. Choose the visible GPU with
`CUDA_VISIBLE_DEVICES` before execution; schedulers preserve that setting.
Dataset acquisition, training, evaluation, and measurements are separate stages.
`run_study_matrix.py`, `run_diagnostics.py`, `prepare_extension.py`,
and `run_extension.py` preview their selection unless `--execute` is supplied.
Dataset acquisition/preparation commands act immediately.
The underlying `experiments.cli` train/evaluate/benchmark commands execute
immediately. Optional tests include model forward passes and are not part of
a source-only release check.

## Contents

- `lpsn_neuron.py` and `lpsn_variants.py`: reusable LP-SN implementation and named A1--A5 variants.
- `experiments/`: dataset, model-adapter, training, evaluation, corruption, temporal-influence, and analysis code.
- `configs/studies.yml`: 410 retained declarations (335 physical runs and 75 deterministic aliases).
- `configs/historical/studies_695.yml`: the original superseded specification, preserved for provenance.
- `configs/extension/`: 110 portable templates (60 architecture and 50 component runs).
- `configs/reproduction.yml`: named end-to-end reproduction targets.
- `manifests/`: deterministic CIFAR10-DVS and subject-disjoint DVS-Gesture split manifests.
- `tests/`: split, corruption, native-length, and model-adapter tests.
- `requirements.txt` and `environment.yml`: pinned runtime definitions.

## Platform and Hardware

The experiment workflow targets **Linux** and uses POSIX file locks and symlinks.
Install Git, Conda, and the system `unzip` utility in addition to the pinned
Python environment. The full experiment configurations use `cuda:0` and require
an NVIDIA GPU with a driver compatible with the CUDA 12.8 PyTorch wheels.
The minimal neuron example and synthetic smoke configuration use the CPU.

The reference worker budget targets an NVIDIA A100. On a GPU with less memory,
reduce `--workers` and `--temporal-worker-budget` when scheduling runs.
`measure_gpu_energy.py` explicitly requires an A100 and the cumulative NVML
energy counter. GPU measurement scripts also require `nvidia-smi` and
`libnvidia-ml.so.1` from the NVIDIA driver installation. Timing and energy values
depend on the GPU, driver, workload, and background activity. Implementation
benchmarks record the measured NVML device name and available UUID in their
protocol; they do not assume that every supported GPU is an A100.

HARDVS needs space for its approximately 245 GB archive and 257 GB extracted
payload during preparation, plus any generated float32 frame caches. These sizes
do not include checkpoints or other datasets.

Optional figure generation uses a separate dependency file:

```bash
python -m pip install -r requirements-plotting.txt
```

## Dataset Preparation

Prepare event arrays under `../data` before generating split manifests.
The loaders expect CIFAR10-DVS NPZ files with separate `x`, `y`, `p`, and `t`
arrays, and DVS-Gesture NPY files whose first four columns are **`[x, y, p, t]`**.
The `prepare` command scans these prepared files; it does not convert the
original recordings into these formats.

See [Dataset sources and prepared formats](datasets.md) for paths, sample
naming, split rules, and the HARDVS acquisition/integrity commands. The reported
HARDVS population uses the documented 13-exclusion mirror protocol; the exact
official archive has a different sample population.

Generate split manifests:

```bash
python -m experiments.cli prepare --dataset cifar10_dvs --data-root ../data \
  --output manifests/cifar10_dvs.json
python -m experiments.cli prepare --dataset dvs_gesture --data-root ../data \
  --output manifests/dvs_gesture.json
python -m experiments.cli prepare --dataset hardvs --data-root ../data \
  --output manifests/hardvs.json
```

Each manifest contains sample identifiers, labels, split membership, and a
content hash. Dataset paths remain relative and the datasets themselves are not
included.

## Locked Evaluation Protocol

Training never constructs a test dataset. It selects checkpoints using
validation macro-F1, with validation accuracy as the prespecified tie-breaker:

```bash
python -m experiments.cli train --config generated_configs/<study>/<dataset>/<model>/T<steps>/seed_<seed>.yml
```

The CLI enables strict PyTorch deterministic algorithms, deterministic cuDNN,
and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; each environment manifest records these
settings. If an operation has no deterministic CUDA implementation, training
fails rather than silently continuing with a warning.

The separately invoked evaluation command loads only the selected checkpoint.
Every `(condition, severity, draw)` is recorded in an append-only ledger and a
duplicate locked-test evaluation is rejected. A persistent reservation is written
before reading test data and survives interruption. Interrupted conditions require
explicit review; `--skip-completed` skips only completed ledger entries:

```bash
python -m experiments.cli evaluate --config <config.yml>
```

The training manifest is never overwritten by evaluation. Each test opening
has its own hash-bearing manifest with evaluation-time git state, checkpoint
epoch and SHA-256, selection rule, split hash, and hashes of its per-sample and
confusion artifacts. Per-layer firing activity, macro metrics, latency,
throughput, and peak allocated CUDA memory are machine-readable.

For a completed run created before evaluation manifests were introduced,
backfill hashes the already sealed checkpoint, per-sample rows, and confusion
matrix without recomputing predictions, and labels the result explicitly:

```bash
python scripts/backfill_evaluation_manifests.py \
  --runs-root runs_baseline --dry-run
python scripts/backfill_evaluation_manifests.py \
  --runs-root runs_baseline \
  --output aggregates/evaluation_manifest_backfill.json
```

## Baseline Studies

Expand the retained matrix after preparing all manifests. Use a new or empty
`generated_configs` directory: expansion does not delete old files. The historical
695-condition specification includes discontinued HARDVS work and must not be
expanded into the active directory. Both baseline schedulers validate generated
declarations against `configs/studies.yml` and refuse stale or duplicate files
without deleting them. For a separate historical/custom study, supply its
`--spec` explicitly and use a separate `--configs-root`.

```bash
python -m experiments.expand_matrix \
  --spec configs/studies.yml \
  --data-root ../data \
  --manifests-root manifests \
  --output generated_configs
```

The matrix specifies:

- HARDVS at `T=16` for LIF, PSN, masked PSN, sliding PSN, RPSU, LP-SN A1, and LP-SN A2.
- Native retraining at `T = 8, 16, 32, 64` on CIFAR10-DVS and DVS-Gesture; HARDVS retains only `T=16`.
- CIFAR10-DVS matched retraining for raw-event density, temporal sparsity, amplitude-only controls, and mass-renormalized thinning.
- The matched seeds `42, 123, 256, 512, 1024`.

Preview the retained schedule, then explicitly start it when datasets, storage,
and the selected GPU are ready:

```bash
python scripts/run_study_matrix.py --workers 4
python scripts/run_study_matrix.py --workers 4 --execute
```

The scheduler precomputes lossless float32 frame caches once per raw-event
condition, runs matched models against the same cache, removes a cache only
after its whole group succeeds, and resumes from completed run ledgers.
Severity-1 thinning, renormalized thinning, amplitude scaling, and temporal
sparsity are mathematical identity conditions. Those cells, plus duplicate
clean declarations shared by two studies, reuse one deterministic physical
run with an `equivalence_provenance.json` record; they are not treated as
independent replications.

All event recordings are divided into equal-duration bins over their complete
native duration. A model is reconstructed for each native `T`; fixed-length
zero padding is not used as a sequence-length experiment.

## Robustness and Temporal Analyses

Generate the shared sensor-motivated corruption grid and evaluate ten draws per
non-clean condition:

```bash
python -m experiments.cli corrupt --output manifests/corruptions.json
python -m experiments.cli evaluate-grid --config <clean-T32-config.yml> \
  --manifest manifests/corruptions.json --skip-completed
```

The committed named grids can be used instead of the generated JSON manifest,
for example `--manifest configs/evaluation_grids.yml --grid robustness`.

The grid covers event dropout, background activity, polarity flips, timestamp
jitter, contiguous packet loss, and hot-pixel bursts at three severities. A
perturbation is determined by sample, condition, severity, and declared draw,
so the identical realization is shared across every model and initialization
seed. These are explicitly sensor-motivated synthetic corruptions, not a
calibrated physical-sensor model.

For clean-selected `T=32` diagnostics on CIFAR10-DVS and DVS-Gesture (all four
grids and temporal input gradients), run the scheduler after baseline training.
Its full benchmark scope also includes the seven retained HARDVS `T=16` models:

```bash
python scripts/run_diagnostics.py --workers 3 --execute
```

It caches each raw-event corruption once, evaluates matched run directories
sequentially while allowing different models to run concurrently, and removes
each cache only after its group succeeds. Benchmarks are always serialized so
their timings cannot interfere with one another.

Individual temporal input influence and fixed benchmark commands remain
available:

```bash
python -m experiments.cli temporal-influence --config <config.yml> \
  --output <influence.json>
python -m experiments.cli benchmark --config <config.yml> \
  --warmup 100 --iterations 500 --output <benchmark.json>
```

The benchmark uses a synchronized, all-zero tensor already resident on the
recorded device. It measures model-forward latency only, not event decoding,
host transfer, end-to-end application latency, energy, or neuromorphic
hardware efficiency.

Use `evaluate --condition causal_prefix --severity 0.25`, `0.50`, `0.75`, and
`1.00` for causal-prefix curves, and `reverse` or `permute` for temporal-order
controls.

## Aggregation and Named Reproduction Targets

```bash
python -m experiments.cli analyze --runs-root runs_baseline \
  --output aggregates/results.csv
python scripts/aggregate_diagnostics.py
python scripts/audit_empirical_package.py --require-diagnostics
python -m experiments.cli reproduce --matrix configs/reproduction.yml \
  --target smoke --execute
```

Aggregation uses the crossed bootstrap: sample training seeds and shared
corruption-draw indices independently, then use their Cartesian product.
The same sampled draw stays shared across sampled seeds. Clean-relative density results
use the matched `temporal_native` clean cell when their own experiment has no
duplicate clean declaration, and the output records that reference explicitly.
The analysis manifest records the summary CSV's SHA-256 digest so plotting can
verify the exact input bytes against their analysis record.
The default audit checks all 410 retained declarations, sealed evaluation artifacts, hashes,
sample/confusion counts, alias provenance, and diagnostic coverage. The generated
CSV tables retain the source and uncertainty information needed to trace results
back to their evaluation records.

## Fresh and Historical HARDVS Audits

The default empirical audit validates the selected source's content hashes,
counts, exclusions, and local integrity record. It also requires the preparation
script and source-integrity specification to match the current release byte hashes.
Fresh download timestamps, paths, and preparation commits may differ from the original experiment. To verify the
preserved original acquisition/preparation records instead, use
`audit_empirical_package.py --historical-source`; this option deliberately
requires the historical provenance. Auditing the superseded full matrix also
requires `--spec configs/historical/studies_695.yml --expected-configs 695`
and its separate configuration directory. The official exact source and the documented
13-exclusion mirror protocol have different sample populations; only the latter
matches the reported HARDVS population.

The `smoke` named target executes synthetic training and evaluation. Its output
is single-use: a second invocation refuses a run whose test ledger already
exists. Use a separate smoke output configuration for another execution. Do not
remove sealed experiment outputs to bypass the test lock.

## Architecture and Component Studies

After generating baseline configurations, freeze a fresh extension protocol and
the current source hashes. Preparation creates configuration files and metadata;
it does not train or open the test set. The committed 110 templates remain
unchanged. Outputs default to `reproduction_outputs/`, or the directory selected
with `LPSN_EVIDENCE_ROOT`. Changing that variable relocates metadata and
measurements, not training runs. Use a fresh checkout for an independent
reproduction, or provide separate baseline, extension configuration, and training
output roots. Preparation refuses extension run directories containing earlier
training artifacts. Generated extension configurations default to
`generated_configs_extension/`.
Keep baseline splits and the shared recipe unchanged for the 30 reused baselines.

```bash
python scripts/prepare_extension.py
python scripts/prepare_extension.py --execute
python scripts/run_extension.py
python scripts/run_extension.py --execute
python scripts/audit_extension.py
```

The extension queue performs 110 training runs, one locked clean-test evaluation
per selected checkpoint, and the synthetic-input forward benchmark. It preserves
partial runs for explicit recovery, verifies the frozen source/configuration
hashes, and honors the selected GPU. Energy or layer diagnostics are independent
stages; their absence does not block training. Finish baseline training before
running the following diagnostics, which require its selected checkpoints.

Run GPU stages serially with no concurrent training or other GPU workload.
Baseline schedulers, the extension queue, and measurement scripts share
`$LPSN_EVIDENCE_ROOT/gpu.lock` (default `reproduction_outputs/gpu.lock`).
Use the same evidence root for all stages; scheduler workers run concurrently
within their owning stage:

```bash
python scripts/measure_gpu_energy.py
python scripts/diagnose_checkpoints.py
python scripts/diagnose_checkpoints.py --layers
python scripts/benchmark_implementations.py
python scripts/run_reset_intervention.py
```

The energy, layer, implementation, and reset commands execute GPU measurements.
`diagnose_checkpoints.py` without `--layers` reads saved predictions.
The energy stage covers 20 model/dataset/batch conditions; layer diagnostics
cover 45 checkpoints. Implementation comparisons use matched weights and three
validation batches. The reset
intervention evaluates 30 selected checkpoints with reset on/off on validation
examples previously used for checkpoint selection; it does not retrain models.
Measurements depend on hardware, inputs, and the reproduced checkpoints.

After all required saved records exist, produce analysis tables and figures:

```bash
python scripts/aggregate_extension.py
python scripts/summarize_layer_diagnostics.py
python scripts/summarize_execution_comparison.py
python scripts/summarize_reset_intervention.py
python scripts/summarize_optimization.py
python scripts/plot_bootstrap_intervals.py
```

`aggregate_extension.py` writes `extension_tables.tex`, `analysis_values.json`,
and `analysis_status.json`. Execution, reset, and optimization summaries write
`execution_methods.tex`, `execution_summary.tex`, `execution_details.tex`,
`reset_methods.tex`, `reset_summary.tex`, `reset_per_seed.tex`, and
`optimization_summary.tex` in their corresponding output directories.
Optimization summaries read each run's saved epoch cap and patience settings;
when both stopping conditions coincide, the report records both.

Aggregation and summary commands consume saved records; they do not launch
training or measurement. Plotting defaults to `aggregates/results.csv`; use
`--input` for a different analysis CSV. Figure outputs follow
`LPSN_EVIDENCE_ROOT`. Plotting verifies both the analysis manifest and its CSV
digest, and requires `requirements-plotting.txt`. For older aggregates without
a CSV digest, generate a new analysis output from the saved run records using
`experiments.cli analyze`; keep the historical files unchanged.
Historical scope-reduction records are provenance, not instructions to complete
the removed experiments. The optimization summary accepts a separate optional
`--scope-record-dir` when those original records are available.

## Controlled Comparator Scope

PSN, masked PSN, and sliding PSN use the pinned SpikingJelly implementations.
RPSU is an independent implementation of the published equations, with the
official source revision recorded as
`Brain-Cog-Lab/PSU@89fd724365cd4970cf3cf3802122c9fc22285a71`.
The retained baseline comparators use VGG-Small and the shared optimization
protocol. The extension adds half-width VGG-Small and spiking ResNet-18 for
PSN/A1/A2, plus five VGG-Small kernel/residual controls. Architecture changes
are explicit in configuration and run-directory names.
Full sequence architectures such as P-SpikeSSM and SpikingSSMs provide related
architectural context; they are outside the implemented neuron-comparator set.

## Release Scope

The release contains source, portable configurations, split/corruption manifests,
and verification code. Licensed datasets, trained checkpoints, raw measurements,
and local generated outputs are excluded. Reproducing numerical results requires
data acquisition and execution of the experimental stages above. Source and
configuration checks alone do not verify numerical agreement with the reported
results.

## Existing Artifacts

The active default names apply to new runs. They do not rename existing run
folders or rewrite frozen configuration/protocol paths and hashes. To inspect
older evidence, use its explicit configuration and run paths with the matching
historical source checkout. Keep original records unchanged; legacy output
folders remain ignored by Git.
