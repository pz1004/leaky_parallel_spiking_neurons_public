"""Crossed bootstrap for corruption draws shared across training seeds.

Resample seed indices and draw indices independently, then take their Cartesian
product. A sampled draw is shared by every sampled seed, preserving pairing.
Rows passed for model contrasts must already contain paired differences.
"""
import numpy as np


def crossed_estimates(rows, metric, seed=20240723, replicates=2000):
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError('replicates must be a positive integer')
    seeds = sorted({int(row['seed']) for row in rows})
    draws = sorted({int(row.get('draw', 0)) for row in rows})
    cells = {}
    for row in rows:
        key = (int(row['seed']), int(row.get('draw', 0)))
        if key in cells:
            raise ValueError(f'Duplicate seed/draw cell: {key}')
        cells[key] = float(row[metric])
    if not seeds or len(cells) != len(seeds) * len(draws):
        raise ValueError('Crossed bootstrap requires a complete rectangular seed-by-draw grid')
    values = np.array([[cells[s, d] for d in draws] for s in seeds])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite bootstrap input')
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for i in range(replicates):
        si = rng.integers(len(seeds), size=len(seeds))
        di = rng.integers(len(draws), size=len(draws))
        estimates[i] = values[np.ix_(si, di)].mean()
    return estimates


def crossed_interval(rows, metric, seed=20240723, replicates=2000):
    return tuple(float(v) for v in np.quantile(crossed_estimates(rows, metric, seed, replicates), [0.025, 0.975]))
