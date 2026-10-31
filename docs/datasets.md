# Dataset Sources and Prepared Formats

Dataset files are obtained separately under their providers' terms. The default
commands use a `data` directory beside the repository, addressed as `../data`.

## CIFAR10-DVS and DVS-Gesture

Source information is available from the [CIFAR10-DVS dataset authors](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671)
and [IBM's DVS-Gesture publication and dataset page](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system).
The [TEBN repository](https://github.com/ChaotengDuan/TEBN) describes its
CIFAR10-DVS preprocessing background. A separate [prepared DVS-Gesture NPY release](https://zenodo.org/records/8060604)
is documented by its maintainers; check its files against the format and sample
identifiers required below before using it.

This repository consumes prepared event arrays. It does not provide a converter
from the original CIFAR10-DVS or DVS-Gesture recordings. Conversion must preserve
event coordinates, polarities, timestamps, sample identifiers, and official
DVS-Gesture train/test membership. `experiments.cli prepare` creates a split
manifest from existing arrays; it does not perform that conversion.

```text
data/
  CIFAR10DVS/events_np/<class>/<original_name>_<index>.npz
  DVSGesture/download/ibmGestureTrain/<subject_condition>/<label>.npy
  DVSGesture/download/ibmGestureTest/<subject_condition>/<label>.npy
  HARDVS/raw/{train,val,test}/action_<id>/*.npz
```

| Dataset | Required event representation | Naming and split rules |
| --- | --- | --- |
| CIFAR10-DVS | NPZ with separate, equally sized arrays named `x`, `y`, `p`, and `t` | Preserve class directories and integer filename suffixes. Suffixes below 100 define the test set in each class. A deterministic, class-stratified 10% validation subset is taken from the remaining training examples. |
| DVS-Gesture | A two-dimensional NPY array with event rows; the first four columns are **`x`, `y`, `p`, `t`**, in that order | Filenames contain integer class labels `0`–`10`. The directory prefix before the first underscore identifies the subject. Preserve the `ibmGestureTrain`/`ibmGestureTest` assignment; 20% of training subjects form the deterministic validation subset. |

Coordinates and polarity are loaded as integers; timestamps are loaded as
floating-point values. Polarity should use `0` and `1`. DVS-Gesture arrays with
`[t, x, y, p]` order are incompatible with this loader. Keep all events and their
original timing so that framing covers each recording's complete duration.
The authoritative loader and split logic are in [experiments/data.py](../experiments/data.py).
Compare prepared sample identifiers and assignments with the committed
[split manifests](../manifests/) before training.

Once the prepared files are available, generate the two split manifests:

```bash
python -m experiments.cli prepare --dataset cifar10_dvs --data-root ../data \
  --output manifests/cifar10_dvs.json
python -m experiments.cli prepare --dataset dvs_gesture --data-root ../data \
  --output manifests/dvs_gesture.json
```

## HARDVS

The [official HARDVS project](https://github.com/Event-AHU/HARDVS) provides the
dataset source information. The pinned archive, label inventories, and source
hashes are specified in [hardvs_compact_archive.json](../manifests/hardvs_compact_archive.json).

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
(three train and three test). Full label-to-central-directory reconciliation
also finds seven consecutive assigned `action_006` train rows with no archive
member. The mirror is therefore rejected by default. If the official provider
is inaccessible, the only supported fallback is an explicit 13-sample
exclusion protocol:

```bash
python scripts/acquire_hardvs.py --output ../data/HARDVS/download --workers 8 \
  --allow-known-corrupt-mirror
python scripts/prepare_hardvs.py --download ../data/HARDVS/download \
  --data-root ../data --allow-known-corrupt-mirror
```

That opt-in verifies every downloaded chunk, the combined archive MD5 and
SHA-256, all label digests, the complete six-member CRC failure inventory, and all
107,646 assigned label identifiers. It then quarantines the six unreadable
members and excludes those plus the seven absent rows. Effective counts are
64,516 train, 10,734 validation, and 32,383 test (107,633 assigned samples);
the generated integrity record labels this as a source deviation. It must not
be reported as an exact replication of the official 107,646-row split.
The committed `manifests/hardvs_source_audit.json` records the completed
archive/member/label reconciliation, all 13 exclusions, extracted-payload CRC
coverage, and the hashes of the local acquisition and preparation records.

The downloader does not redistribute data from this repository. Users remain
responsible for the dataset terms and should prefer the original provider when
its client is available. Acquisition performs a complete CRC read and writes a
hash-protected record; preparation refuses to proceed without that matching
record, validates the label digests and split disjointness, creates only
symlinks in `HARDVS/raw/{train,val,test}`, and removes the source archive after
successful extraction unless `--keep-archive` is given. It refuses a dirty
tracked source tree by default and records the preparation commit plus script
and manifest hashes in the dataset integrity record.

Generate the HARDVS split manifest after successful archive preparation:

```bash
python -m experiments.cli prepare --dataset hardvs --data-root ../data \
  --output manifests/hardvs.json
```

The default audit checks a fresh preparation against the current script and
source-specification hashes. `--historical-source` instead checks the preserved
original acquisition/preparation provenance. See the [audit workflow](reproduction.md#fresh-and-historical-hardvs-audits).
