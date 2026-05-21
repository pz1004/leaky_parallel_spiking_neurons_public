"""Public LP-SN utilities."""

from .hard_reset_diagnostics import (
    hard_reset_lif_reference,
    prop2b_hard_reset_reference_bound,
    prop2b_hard_reset_reference_stats,
)
from .lpsn_neuron import LPSpikingNeuron, build_causal_exp_kernel, soft_reset_correction
from .lpsn_variants import (
    CORE_LPSN_VARIANTS,
    DEFAULT_VARIANT,
    DIAGNOSTIC_LPSN_VARIANTS,
    LPSN_ALIASES,
    LPSN_VARIANTS,
    TEMPORAL_PROFILE_BY_SOFT_RESET_STRATEGY,
    available_variants,
    build_lpsn_neuron,
    canonical_variant,
    temporal_profile_for_strategy,
    variant_config,
)

__all__ = [
    "DEFAULT_VARIANT",
    "CORE_LPSN_VARIANTS",
    "DIAGNOSTIC_LPSN_VARIANTS",
    "LPSN_ALIASES",
    "LPSN_VARIANTS",
    "LPSpikingNeuron",
    "TEMPORAL_PROFILE_BY_SOFT_RESET_STRATEGY",
    "available_variants",
    "build_causal_exp_kernel",
    "build_lpsn_neuron",
    "canonical_variant",
    "hard_reset_lif_reference",
    "prop2b_hard_reset_reference_bound",
    "prop2b_hard_reset_reference_stats",
    "soft_reset_correction",
    "temporal_profile_for_strategy",
    "variant_config",
]
