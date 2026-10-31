# Leaky-Parallel Spiking Neurons (LP-SN)

## Abstract

LP-SN is a PyTorch/SpikingJelly neuron family that combines temporal mixing
with the exponential dynamics of leaky integrate-and-fire neurons. It provides
fixed and learnable temporal kernels, optional residual weights, and exact
soft-reset dynamics for experimenting with spiking neural networks.

## Environment

The Linux reference environment uses **Python 3.12**, **PyTorch 2.10.0 with
CUDA 12.8 wheels**, and **SpikingJelly at commit `9b87d38`**. Versions are pinned in
[environment.yml](environment.yml). The example below runs on CPU; GPU use
requires a compatible NVIDIA driver.

## Installation

```bash
git clone https://github.com/pz1004/leaky_parallel_spiking_neurons_public.git
cd leaky_parallel_spiking_neurons_public
conda env create -f environment.yml
conda activate lpsn
```

Run the example from the repository directory. No dataset or checkpoint is needed.

## Usage

```python
import torch
from lpsn_variants import build_lpsn_neuron

T, batch, features = 16, 2, 64
neuron = build_lpsn_neuron(T=T, variant="a1")
x = torch.randn(T, batch, features, requires_grad=True)

spikes = neuron(x)
spikes.mean().backward()  # Surrogate gradients flow to x and upstream layers.
print(spikes.shape)      # torch.Size([16, 2, 64])
```

Inputs use **`[time, batch, ...]`** order; outputs have the same shape.
The time dimension must match `T` at construction. Only multi-step mode is
supported. Each call processes an independent sequence; membrane and reset
state are not carried between calls. Keep the neuron and input on the same
device and use matching dtypes.

| Variant | Temporal computation |
| --- | --- |
| `a1` | Fixed exponential kernel, without reset |
| `a2` | Fixed exponential kernel with exact soft reset; reset spike history is detached from gradients |
| `a3` | Learnable exponential kernel |
| `a4` | Learnable kernel, causal residual, and bias |
| `a5` | Dense, noncausal temporal mixing with parallel spiking neuron (PSN) initialization |

A1 and A2 have no trainable neuron parameters but pass gradients to upstream
layers. **A2 constructs its reset history sequentially.** For residual variants,
add `neuron.regularization_loss()` to your task loss; it includes the configured
penalty weight. See [lpsn_variants.py](lpsn_variants.py) for configuration options.

## Main Results

Reported **test accuracy (%)**, averaged over five matched seeds with a shared
VGG-Small backbone. PSN is the unconstrained temporal-mixing baseline.

| Dataset | Timesteps | PSN | LP-SN A1 | LP-SN A2 |
| --- | ---: | ---: | ---: | ---: |
| CIFAR10-DVS | 32 | 65.38 | 71.84 | 71.66 |
| DVS-Gesture | 32 | 62.73 | 73.11 | 74.92 |
| HARDVS | 16 | 11.97 | 9.90 | 15.98 |

Performance depends on dataset and architecture. These five-seed comparisons
did not reach statistical significance after correction for multiple comparisons.
HARDVS uses the [documented mirror-exclusion split](docs/datasets.md#hardvs).

In the VGG-Small zero-input GPU benchmark, A1 had similar inference latency to
PSN, while A2 was slower. Neither variant reduced measured gross GPU energy
relative to PSN on the evaluated validation inputs.

[Full experiment guide](docs/reproduction.md) · [Citation](CITATION.cff) · [MIT license](LICENSE)
