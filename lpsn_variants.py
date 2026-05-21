"""Named LP-SN variants and a small construction helper.

This file intentionally contains only reusable configuration logic. It does
not depend on the benchmark harness used to generate manuscript runs.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from .lpsn_neuron import LPSpikingNeuron
except ImportError:  # pragma: no cover - supports direct script-style imports
    from lpsn_neuron import LPSpikingNeuron


TEMPORAL_PROFILE_BY_SOFT_RESET_STRATEGY: dict[str, str] = {
    "none": "dense_gemm",
    "tentative": "tentative_parallel",
    "exact_detached": "exact_detached_scan",
}


LPSN_VARIANTS: dict[str, dict[str, Any]] = {
    "lpsn_a1_lif_init": {
        "variant": "lpsn_a1_lif_init",
        "init_tau": 2.0,
        "init_alpha": None,
        "learnable_kernel": False,
        "learnable_residual": False,
        "residual_mask": "causal",
        "learnable_bias": False,
        "soft_reset_strategy": "none",
        "temporal_profile": "dense_gemm",
        "learnable_beta": False,
        "residual_l2": 0.0,
        "psn_init": False,
    },
    "lpsn_a2_soft": {
        "variant": "lpsn_a2_soft",
        "init_tau": 2.0,
        "init_alpha": None,
        "learnable_kernel": False,
        "learnable_residual": False,
        "residual_mask": "causal",
        "learnable_bias": False,
        # Manuscript A2 prioritizes Prop. 2a exactness. The optional
        # "tentative" strategy in LPSpikingNeuron is a fully parallel
        # approximation and is not the A2 condition used in reported tables.
        "soft_reset_strategy": "exact_detached",
        "temporal_profile": "exact_detached_scan",
        "learnable_beta": False,
        "residual_l2": 0.0,
        "psn_init": False,
    },
    "lpsn_a3_kernel": {
        "variant": "lpsn_a3_kernel",
        "init_tau": 2.0,
        "init_alpha": None,
        "learnable_kernel": True,
        "learnable_residual": False,
        "residual_mask": "causal",
        "learnable_bias": False,
        "soft_reset_strategy": "none",
        "temporal_profile": "dense_gemm",
        "learnable_beta": False,
        "residual_l2": 0.0,
        "psn_init": False,
    },
    "lpsn_a4": {
        "variant": "lpsn_a4",
        "init_tau": 2.0,
        "init_alpha": None,
        "learnable_kernel": True,
        "learnable_residual": True,
        "residual_mask": "causal",
        "learnable_bias": True,
        "soft_reset_strategy": "none",
        "temporal_profile": "dense_gemm",
        "learnable_beta": False,
        "residual_l2": 1e-3,
        "psn_init": False,
    },
    "lpsn_a4a_frozen_bias": {
        "variant": "lpsn_a4a_frozen_bias",
        "init_tau": 2.0,
        "init_alpha": None,
        "learnable_kernel": True,
        "learnable_residual": True,
        "residual_mask": "causal",
        "learnable_bias": False,
        "soft_reset_strategy": "none",
        "temporal_profile": "dense_gemm",
        "learnable_beta": False,
        "residual_l2": 1e-3,
        "psn_init": False,
    },
    "lpsn_a4b_frozen_kernel": {
        "variant": "lpsn_a4b_frozen_kernel",
        "init_tau": 2.0,
        "init_alpha": None,
        "learnable_kernel": False,
        "learnable_residual": True,
        "residual_mask": "causal",
        "learnable_bias": True,
        "soft_reset_strategy": "none",
        "temporal_profile": "dense_gemm",
        "learnable_beta": False,
        "residual_l2": 1e-3,
        "psn_init": False,
    },
    "lpsn_a5_psn": {
        "variant": "lpsn_a5_psn",
        "init_tau": 2.0,
        "init_alpha": 0.0,
        "learnable_kernel": False,
        "learnable_residual": True,
        "residual_mask": "dense",
        "learnable_bias": True,
        "soft_reset_strategy": "none",
        "temporal_profile": "dense_gemm",
        "learnable_beta": False,
        "residual_l2": 0.0,
        "psn_init": True,
    },
}

CORE_LPSN_VARIANTS: tuple[str, ...] = (
    "lpsn_a1_lif_init",
    "lpsn_a2_soft",
    "lpsn_a3_kernel",
    "lpsn_a4",
    "lpsn_a5_psn",
)

DIAGNOSTIC_LPSN_VARIANTS: tuple[str, ...] = (
    "lpsn_a4a_frozen_bias",
    "lpsn_a4b_frozen_kernel",
)


LPSN_ALIASES: dict[str, str] = {
    "lpsn_a1": "lpsn_a1_lif_init",
    "lpsn-a1": "lpsn_a1_lif_init",
    "a1": "lpsn_a1_lif_init",
    "lpsn_a1_lif_init": "lpsn_a1_lif_init",
    "lpsn-a1-lif-init": "lpsn_a1_lif_init",
    "lpsn_a2": "lpsn_a2_soft",
    "lpsn-a2": "lpsn_a2_soft",
    "a2": "lpsn_a2_soft",
    "lpsn_a2_soft": "lpsn_a2_soft",
    "lpsn-a2-soft": "lpsn_a2_soft",
    "lpsn_a3": "lpsn_a3_kernel",
    "lpsn-a3": "lpsn_a3_kernel",
    "a3": "lpsn_a3_kernel",
    "lpsn_a3_kernel": "lpsn_a3_kernel",
    "lpsn-a3-kernel": "lpsn_a3_kernel",
    "lpsn_a4": "lpsn_a4",
    "lpsn-a4": "lpsn_a4",
    "a4": "lpsn_a4",
    "lpsn_a4a": "lpsn_a4a_frozen_bias",
    "lpsn-a4a": "lpsn_a4a_frozen_bias",
    "a4a": "lpsn_a4a_frozen_bias",
    "lpsn_a4a_frozen_bias": "lpsn_a4a_frozen_bias",
    "lpsn-a4a-frozen-bias": "lpsn_a4a_frozen_bias",
    "lpsn_a4b": "lpsn_a4b_frozen_kernel",
    "lpsn-a4b": "lpsn_a4b_frozen_kernel",
    "a4b": "lpsn_a4b_frozen_kernel",
    "lpsn_a4b_frozen_kernel": "lpsn_a4b_frozen_kernel",
    "lpsn-a4b-frozen-kernel": "lpsn_a4b_frozen_kernel",
    "lpsn_a5": "lpsn_a5_psn",
    "lpsn-a5": "lpsn_a5_psn",
    "a5": "lpsn_a5_psn",
    "lpsn_a5_psn": "lpsn_a5_psn",
    "lpsn-a5-psn": "lpsn_a5_psn",
}


DEFAULT_VARIANT = "lpsn_a2_soft"

_CONSTRUCTOR_KEYS = {
    "T",
    "init_tau",
    "init_alpha",
    "v_threshold",
    "learnable_kernel",
    "learnable_residual",
    "residual_init",
    "residual_mask",
    "learnable_bias",
    "bias_init",
    "soft_reset",
    "soft_reset_strategy",
    "learnable_beta",
    "surrogate_function",
    "residual_l2",
}


def canonical_variant(name: str) -> str:
    """Return the canonical variant name for an alias."""
    key = name.strip().lower()
    canonical = LPSN_ALIASES.get(key, key)
    if canonical not in LPSN_VARIANTS:
        choices = ", ".join(available_variants(include_diagnostics=True))
        raise ValueError(f"Unknown LP-SN variant {name!r}. Available variants: {choices}.")
    return canonical


def temporal_profile_for_strategy(soft_reset_strategy: str) -> str:
    """Return the public metadata label for a reset strategy."""
    try:
        return TEMPORAL_PROFILE_BY_SOFT_RESET_STRATEGY[soft_reset_strategy]
    except KeyError as exc:
        choices = ", ".join(sorted(TEMPORAL_PROFILE_BY_SOFT_RESET_STRATEGY))
        raise ValueError(
            f"Unknown soft_reset_strategy {soft_reset_strategy!r}. "
            f"Expected one of: {choices}."
        ) from exc


def available_variants(*, include_diagnostics: bool = False) -> tuple[str, ...]:
    """Return canonical LP-SN variant names.

    By default this returns the manuscript A1-A5 variants. Pass
    ``include_diagnostics=True`` to include residual-rescue controls.
    """
    if include_diagnostics:
        return (*CORE_LPSN_VARIANTS, *DIAGNOSTIC_LPSN_VARIANTS)
    return CORE_LPSN_VARIANTS


def variant_config(name: str = DEFAULT_VARIANT, *, T: int | None = None) -> dict[str, Any]:
    """Return a copy of a named variant configuration."""
    config = deepcopy(LPSN_VARIANTS[canonical_variant(name)])
    if T is not None:
        config["T"] = T
    config["temporal_profile"] = temporal_profile_for_strategy(config["soft_reset_strategy"])
    return config


def build_lpsn_neuron(
    *,
    T: int,
    variant: str = DEFAULT_VARIANT,
    **overrides: Any,
) -> LPSpikingNeuron:
    """Construct an :class:`LPSpikingNeuron` from a named variant.

    Args:
        T: number of timesteps in the input sequence.
        variant: canonical variant name or alias, such as ``"a1"`` or
            ``"lpsn_a2_soft"``.
        **overrides: optional ``LPSpikingNeuron`` constructor parameters.
            ``psn_init`` may also be overridden for the dense PSN-recovery
            variant.
    """
    config = variant_config(variant, T=T)
    psn_init = bool(config.pop("psn_init"))
    config.pop("variant", None)

    for key, value in overrides.items():
        if key == "psn_init":
            psn_init = bool(value)
        elif key in _CONSTRUCTOR_KEYS:
            config[key] = value
        else:
            raise ValueError(f"Unknown LP-SN override {key!r}.")

    params = {key: value for key, value in config.items() if key in _CONSTRUCTOR_KEYS}
    node = LPSpikingNeuron(**params)
    if psn_init:
        node.reset_psn_parameters()
    return node
