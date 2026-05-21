# Leaky-Parallel Spiking Neurons (LP-SN)

This folder contains only the reusable LP-SN logic intended for public
disclosure. It excludes manuscript files, run outputs, datasets, checkpoints,
and analysis scripts.

## Contents

- `lpsn_neuron.py`: the `LPSpikingNeuron` module and kernel/reset helpers.
- `lpsn_variants.py`: named A1-A5 variant configurations and a construction helper.
- `__init__.py`: package exports for convenient imports.
- `requirements.txt`: manuscript runtime dependency pins.
- `environment.yml`: conda environment matching the recorded manuscript profile.
- `check_reproducibility.py`: lightweight runtime-version checker.

## Installation

The manuscript runs used Python 3.12, PyTorch 2.10.0, torchvision 0.25.0,
CUDA 12.8, cuDNN 91002, and SpikingJelly from commit `9b87d38`.

For a conda-style setup:

```bash
conda env create -f environment.yml
conda activate lpsn-manuscript
```

For a pip-style setup, install a CUDA-compatible PyTorch wheel for your
platform if the default wheel is not suitable, then install:

```bash
pip install -r requirements.txt
```

Check the active runtime with:

```bash
python check_reproducibility.py
```

## Minimal Usage

```python
import torch
from lpsn_neuron import LPSpikingNeuron

T = 16
neuron = LPSpikingNeuron(
    T=T,
    init_tau=2.0,
    learnable_kernel=False,
    learnable_residual=False,
    soft_reset_strategy="exact_detached",
)

x_seq = torch.randn(T, 32, 128)  # [time, batch, ...]
s_seq = neuron(x_seq)
```

LP-SN expects multi-step input with time as the leading dimension:
`[T, batch, ...]`. When SpikingJelly is installed, the module is configured
with `step_mode = "m"` for activation-based multi-step networks.

## Named Variants

```python
from lpsn_variants import build_lpsn_neuron

neuron = build_lpsn_neuron(T=16, variant="a2")
```

Core manuscript variants:

| Variant | Description |
| --- | --- |
| `lpsn_a1_lif_init` | Frozen LIF-equivalent causal kernel, no residual. |
| `lpsn_a2_soft` | A1 plus detached soft-reset correction. |
| `lpsn_a3_kernel` | Learnable kernel scalars, no residual. |
| `lpsn_a4` | Learnable kernel, causal residual, and temporal bias. |
| `lpsn_a5_psn` | Dense residual PSN-recovery control. |

By default, `available_variants()` returns only the manuscript A1-A5 variants.
Pass `include_diagnostics=True` to include the A4a/A4b diagnostic controls.
Each variant also records a `temporal_profile`: A2 uses
`exact_detached_scan`, while no-reset and PSN-recovery variants use
`dense_gemm`.

Diagnostic variants:

| Variant | Description |
| --- | --- |
| `lpsn_a4a_frozen_bias` | A4 with frozen temporal bias. |
| `lpsn_a4b_frozen_kernel` | A4 with frozen kernel scalars. |

## Training Note

For variants with a learnable residual, add the optional residual penalty to
your task loss:

```python
reg = sum(
    module.regularization_loss()
    for module in model.modules()
    if hasattr(module, "regularization_loss")
)
loss = task_loss + reg
```

## Citation

If this public release accompanies a paper, add the final citation here before
publishing.
