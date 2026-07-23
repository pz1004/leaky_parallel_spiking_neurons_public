# Leaky-Parallel Spiking Neurons (LP-SN)

This artifact contains the LP-SN implementation and the complete, config-driven
pipeline for the reviewer-remediation experiments. Training, validation-based
selection, locked-test evaluation, perturbation generation, temporal analysis,
benchmarking, and result aggregation are all executable from this repository.

Datasets, trained checkpoints, and bulky raw run logs are not redistributed.
The small split manifests and declarative study specification are included.

## Contents

- `lpsn_neuron.py` and `lpsn_variants.py`: reusable LP-SN implementation and named A1--A5 variants.
- `experiments/`: dataset, model-adapter, training, evaluation, corruption, temporal-influence, and analysis code.
- `configs/studies.yml`: the complete 695-run training matrix for the new studies.
- `configs/reproduction.yml`: named end-to-end reproduction targets.
- `manifests/`: deterministic CIFAR10-DVS and subject-disjoint DVS-Gesture split manifests.
- `tests/`: split, corruption, native-length, and model-adapter tests.
- `requirements.txt` and `environment.yml`: pinned runtime definitions.

## Environment

The recorded environment uses Python 3.12, PyTorch 2.10.0, torchvision 0.25.0,
CUDA 12.8, and SpikingJelly commit `9b87d38`.

```bash
conda env create -f environment.yml
conda activate lpsn-manuscript
python check_reproducibility.py
python -m unittest discover -v tests
```

Run all commands below from this directory.

## Dataset Preparation

Place the datasets below the chosen data root without changing their official
test assignments:

```text
data/
  CIFAR10DVS/events_np/<class>/*.npz
  DVSGesture/download/ibmGestureTrain/<subject_condition>/*.npy
  DVSGesture/download/ibmGestureTest/<subject_condition>/*.npy
  HARDVS/raw/{train,val,test}/action_<id>/*.npz
```

HARDVS is not bundled and must be acquired under its own terms. The default
protocol requires the exact 244,899,036,166-byte official archive (MD5
`9c4cc0d9ba043faa17f6f1a9e9aff982`) and preserves every published train,
validation, and test row. Put that archive at
`../data/HARDVS/download/MINIHARDVS_EVENT_files.zip`, then run:

```bash
python scripts/acquire_hardvs.py --output ../data/HARDVS/download
python scripts/prepare_hardvs.py --download ../data/HARDVS/download \
  --data-root ../data
```

The pinned chunked mirror is the same size but is not byte-identical to the
official object: its combined MD5 is `ef1099ca11f647958870ada2ef29e2d9`, and a
complete read of all 107,710 NPZ members finds six corrupt assigned samples
(three train and three test). It is therefore rejected by default. If the
official provider is inaccessible, the only supported fallback is an explicit
six-sample exclusion protocol:

```bash
python scripts/acquire_hardvs.py --output ../data/HARDVS/download --workers 8 \
  --allow-known-corrupt-mirror
python scripts/prepare_hardvs.py --download ../data/HARDVS/download \
  --data-root ../data --allow-known-corrupt-mirror
```

That opt-in verifies every downloaded chunk, the combined archive MD5 and
SHA-256, all label digests, and the complete six-member failure inventory. It
then quarantines/excludes exactly the declared failures. Effective counts are
64,523 train, 10,734 validation, and 32,383 test (107,640 assigned samples);
the generated integrity record labels this as a source deviation. It must not
be reported as an exact replication of the official 107,646-row split.

The downloader does not redistribute data from this repository. Users remain
responsible for the dataset terms and should prefer the original provider when
its client is available. Acquisition performs a complete CRC read and writes a
hash-protected record; preparation refuses to proceed without that matching
record, validates the label digests and split disjointness, creates only
symlinks in `HARDVS/raw/{train,val,test}`, and removes the source archive after
successful extraction unless `--keep-archive` is given.

CIFAR10-DVS keeps the TEBN test assignment and takes a deterministic
class-stratified validation subset from training. DVS-Gesture keeps the
official test assignment and uses a subject-disjoint validation subset from
training.

Generate or verify manifests:

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
duplicate locked-test evaluation is rejected:

```bash
python -m experiments.cli evaluate --config <config.yml>
```

The training manifest is never overwritten by evaluation. Each test opening
has its own hash-bearing manifest with evaluation-time git state, checkpoint
epoch and SHA-256, selection rule, split hash, and hashes of its per-sample and
confusion artifacts. Per-layer firing activity, macro metrics, latency,
throughput, and peak allocated CUDA memory are machine-readable.

## Full Study Matrix

Expand the committed declarative matrix after preparing all manifests:

```bash
python -m experiments.expand_matrix \
  --spec configs/studies.yml \
  --data-root ../data \
  --manifests-root manifests \
  --output generated_configs
```

The matrix specifies:

- Full HARDVS at `T=16` for LIF, PSN, masked PSN, sliding PSN, RPSU, LP-SN A1, and LP-SN A2.
- Native retraining at `T = 8, 16, 32, 64` on CIFAR10-DVS, DVS-Gesture, and HARDVS.
- Matched retraining for raw-event density, temporal sparsity, amplitude-only controls, and mass-renormalized thinning.
- The matched seeds `42, 123, 256, 512, 1024`.

For the full matrix, use the resumable cache-aware scheduler rather than
manually invoking 695 commands:

```bash
python scripts/run_study_matrix.py --workers 4
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

For the complete clean-selected `T=32` diagnostic package (all four committed
grids, temporal input gradients, and non-overlapping fixed benchmarks), use
the resumable scheduler:

```bash
python scripts/run_diagnostics.py --workers 3
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
python -m experiments.cli analyze --runs-root runs_revision \
  --output aggregates/results.csv
python scripts/aggregate_diagnostics.py
python scripts/audit_empirical_package.py --require-diagnostics
python -m experiments.cli reproduce --matrix configs/reproduction.yml \
  --target smoke --execute
```

Aggregation reports hierarchical bootstrap intervals by resampling training
seeds and then corruption draws within seeds. Clean-relative density results
use the matched `temporal_native` clean cell when their own experiment has no
duplicate clean declaration, and the output records that reference explicitly.
The audit checks all 695 declarations, sealed evaluation artifacts, hashes,
sample/confusion counts, alias provenance, and diagnostic coverage. Manuscript
tables should be built only from these audited aggregates, never by manual
transcription.

## Controlled Comparator Scope

PSN, masked PSN, and sliding PSN use the pinned SpikingJelly implementations.
RPSU is an independent implementation of the published equations, with the
official source revision recorded as
`Brain-Cog-Lab/PSU@89fd724365cd4970cf3cf3802122c9fc22285a71`.
All comparators use the same VGG-Small backbone and optimization protocol.
Full sequence architectures such as P-SpikeSSM and SpikingSSMs are contextual
references, not mislabeled drop-in neuron comparisons.

## Minimal LP-SN Usage

```python
import torch
from lpsn_variants import build_lpsn_neuron

neuron = build_lpsn_neuron(T=16, variant="a2")
x_seq = torch.randn(16, 32, 128)  # [time, batch, ...]
s_seq = neuron(x_seq)
```

## Release Scope

The release package includes complete code, configurations, split/corruption
manifests, and tests, and it generates lightweight aggregate-table inputs. It
intentionally excludes licensed datasets, checkpoints, and bulky raw logs.
Consequently it is code-complete but not a pretrained-artifact archive; the
manuscript and response letter must describe that distinction exactly.

## Citation

Add the final paper citation and immutable release DOI before publication.
