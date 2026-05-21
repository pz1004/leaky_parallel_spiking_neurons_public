"""
Leaky-Parallel Spiking Neuron (LP-SN) for SpikingJelly.

Reference:
    Leaky-Parallel Spiking Neurons With LIF-Anchored Temporal Mixing.

The LP-SN reparameterizes PSN's free T x T temporal mixing matrix as
        W = K_{tau, alpha} + Delta W
where K_{tau, alpha} is a causal exponential Toeplitz kernel parameterized
by two learnable scalars (tau, alpha), and Delta W is a causally-masked
learnable residual.

Theoretical properties:
  - At init (Delta W = 0, alpha = 1/tau, lambda = 1 - 1/tau): exactly
    equivalent to no-reset LIF (Proposition 1).
  - With soft-reset correction (beta = V_th, detached spike history):
    exactly equivalent to soft-reset LIF (Proposition 2a).
  - As alpha -> 0 with Delta W unrestricted: recovers PSN (Proposition 3).

Usage:
    >>> import torch
    >>> from lpsn_neuron import LPSpikingNeuron
    >>> neuron = LPSpikingNeuron(T=16, learnable_kernel=False,
    ...                          learnable_residual=False,
    ...                          soft_reset=True)
    >>> x_seq = torch.randn(16, 64, 128, 7, 7)  # [T, B, C, H, W]
    >>> s_seq = neuron(x_seq)  # [T, B, C, H, W]

Compatible with SpikingJelly's activation_based API. Use with
step_mode='m' (multi-step) for the parallel computation benefit.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

# SpikingJelly imports. We use the activation_based API since LP-SN's
# parallel structure aligns with multi-step computation.
try:
    from spikingjelly.activation_based import base, surrogate
    _SJ_AVAILABLE = True
except ImportError:
    # Allow the module to import without SpikingJelly for testing.
    # In that case the user must provide their own base class and
    # surrogate function. We fall back to torch.nn.Module.
    _SJ_AVAILABLE = False
    base = None
    surrogate = None


# ---------------------------------------------------------------------
# Helper: causal exponential Toeplitz kernel construction.
# ---------------------------------------------------------------------
def _effective_tau(log_tau: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return the stable effective tau used by LP-SN kernels."""
    return torch.exp(log_tau).clamp(min=1.0 + eps)


def build_causal_exp_kernel(
    T: int,
    log_tau: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a T x T lower-triangular Toeplitz kernel with geometric decay.

    K[t, s] = alpha * lambda^(t - s)  for t >= s,  0 otherwise,
    where lambda = 1 - 1/tau with
    tau = max(exp(log_tau), 1 + eps).

    Note: We parameterize tau through log_tau to ensure tau > 0 under
    unconstrained gradient updates, then clamp the effective tau to the
    manuscript's stable tau > 1 regime. The geometric form lambda^(t-s)
    (rather than exp(-(t-s)/tau)) matches LIF's discrete-time recurrence
    exactly under the closed-form no-reset LIF equivalence derivation.

    Args:
        T: number of timesteps.
        log_tau: scalar tensor, log of membrane time constant.
        alpha: scalar tensor, input gain.
        eps: small constant for numerical safety when lambda -> 1.

    Returns:
        kernel: tensor of shape [T, T], lower-triangular with geometric
                entries; upper triangle is zero.
    """
    tau = _effective_tau(log_tau, eps=eps)
    lam = 1.0 - 1.0 / tau  # in (0, 1) since tau > 1

    # Time-difference matrix: D[t, s] = t - s (negative above diagonal).
    t_idx = torch.arange(T, device=log_tau.device, dtype=log_tau.dtype)
    diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)  # [T, T]

    # Mask: 1 if t >= s, else 0 (causal lower-triangular).
    causal_mask = (diff >= 0).to(log_tau.dtype)

    # K[t, s] = alpha * lambda^(t - s) for t >= s.
    # Use max(diff, 0) inside the power to keep gradients well-defined
    # in the upper triangle (which is masked out anyway).
    safe_diff = diff.clamp(min=0)
    kernel = alpha * (lam ** safe_diff) * causal_mask
    return kernel


def _is_lif_equivalent_alpha(init_alpha: Optional[float], init_tau: float) -> bool:
    return init_alpha is None or math.isclose(
        float(init_alpha),
        1.0 / float(init_tau),
        rel_tol=0.0,
        abs_tol=1e-7,
    )


def _validate_exact_detached_scope(
    *,
    init_tau: float,
    init_alpha: Optional[float],
    learnable_kernel: bool,
    learnable_residual: bool,
    residual_init: float,
    residual_mask: str,
    learnable_bias: bool,
    bias_init: Optional[float],
    learnable_beta: bool,
    v_threshold: float,
) -> None:
    """Guard the theorem-level A2 soft-reset equivalence scope."""
    errors = []
    if learnable_kernel:
        errors.append("learnable_kernel=False")
    if not _is_lif_equivalent_alpha(init_alpha, init_tau):
        errors.append("init_alpha=None or init_alpha=1/init_tau")
    if learnable_residual:
        errors.append("learnable_residual=False")
    if not math.isclose(float(residual_init), 0.0, rel_tol=0.0, abs_tol=1e-12):
        errors.append("residual_init=0.0")
    if residual_mask != "causal":
        errors.append("residual_mask='causal'")
    if learnable_bias:
        errors.append("learnable_bias=False")
    if bias_init is not None and not math.isclose(
        float(bias_init), -float(v_threshold), rel_tol=0.0, abs_tol=1e-7
    ):
        errors.append("bias_init=None or bias_init=-v_threshold")
    if learnable_beta:
        errors.append("learnable_beta=False")
    if errors:
        expected = ", ".join(errors)
        raise ValueError(
            "soft_reset_strategy='exact_detached' is reserved for the "
            "Proposition 2a/A2 theorem scope and requires "
            f"{expected}. Use soft_reset_strategy='tentative' for approximate "
            "or learnable reset ablations."
        )


# ---------------------------------------------------------------------
# Helper: soft-reset spike-history correction.
# ---------------------------------------------------------------------
def soft_reset_correction(
    s_detached: torch.Tensor,
    log_tau: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the soft-reset correction term from a detached spike history.

    For each timestep t, returns
        correction[t] = beta * sum_{s' < t, S[s'] = 1} lambda^(t - s'),
    consistent with Proposition 2a of the LP-SN manuscript.

    The spike history s_detached is treated as a constant (no gradient).

    Args:
        s_detached: spike tensor of shape [T, B, ...], detached.
        log_tau: scalar tensor, log of membrane time constant.
        beta: scalar tensor; set to V_th for exact soft-reset LIF.
        eps: numerical safety.

    Returns:
        correction: tensor of shape [T, B, ...] matching s_detached.
    """
    T = s_detached.shape[0]
    tau = _effective_tau(log_tau, eps=eps)
    lam = 1.0 - 1.0 / tau

    # Build the strictly-lower-triangular correction kernel:
    # C[t, s'] = lambda^(t - s')  for s' < t,  0 otherwise.
    t_idx = torch.arange(T, device=log_tau.device, dtype=log_tau.dtype)
    diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)
    strict_lower = (diff > 0).to(log_tau.dtype)
    safe_diff = diff.clamp(min=0)
    corr_kernel = (lam ** safe_diff) * strict_lower  # [T, T]

    # Apply the kernel along the time axis. Use einsum to handle
    # arbitrary trailing spatial dimensions.
    # s_detached: [T, B, ...] -> flatten trailing dims for matmul,
    # then reshape back.
    T_, B = s_detached.shape[0], s_detached.shape[1]
    flat = s_detached.reshape(T_, -1)             # [T, B * spatial]
    correction_flat = corr_kernel @ flat           # [T, B * spatial]
    correction = correction_flat.reshape(s_detached.shape)  # [T, B, ...]
    return beta * correction


# ---------------------------------------------------------------------
# The LP-SN module.
# ---------------------------------------------------------------------
_BaseClass = base.MemoryModule if _SJ_AVAILABLE else nn.Module


class LPSpikingNeuron(_BaseClass):
    """Leaky-Parallel Spiking Neuron.

    Forward computation:
        W      = K(tau, alpha) + mask * Delta W
        H      = einsum('tu,ub...->tb...', W, x_seq)
        logits = H + b
        spikes = surrogate(logits)

    The temporal bias ``b`` is PSN-style: the default value is
    ``-V_th``, so thresholding ``H`` at ``V_th`` is equivalent to
    applying the surrogate to ``H + b``. ``forward_logits(...,
    include_bias=False)`` returns the bias-free membrane-like quantity
    used in the equivalence checks.

    Args:
        T: number of timesteps. Must be set at construction; the kernel
           is built lazily once T is known.
        init_tau: initial membrane time constant. Default 2.0 matches
                  SpikingJelly's LIF default and yields LIF-equivalent
                  initialization (Proposition 1).
        init_alpha: initial input gain. If None, defaults to 1/init_tau
                    (Proposition 1 exact-equivalence setting).
        v_threshold: spike threshold V_th. Default 1.0.
        learnable_kernel: if True, (tau, alpha) are learnable. If False,
                          they remain frozen at init values (ablation A1).
        learnable_residual: if True, Delta W is a learnable T x T parameter.
                            If False, Delta W = 0 (frozen).
        residual_init: initial value for Delta W elements. Default 0.0.
        residual_mask: ``"causal"`` masks Delta W to the lower triangle;
                       ``"dense"`` leaves Delta W unrestricted. Dense mode
                       is intended for the PSN-recovery control.
        learnable_bias: if True, learn the PSN-style temporal bias b.
                        If False, keep it frozen.
        bias_init: scalar initialization for b. Defaults to ``-V_th``.
        soft_reset: if True, apply the theorem-level detached soft-reset
                    correction using ``"exact_detached"``. Pass
                    ``soft_reset_strategy="tentative"`` explicitly to use the
                    fully parallel approximation.
        soft_reset_strategy: one of ``"none"``, ``"tentative"``, or
                             ``"exact_detached"``. The exact strategy
                             constructs the detached soft-reset spike
                             history sequentially and is used for the
                             proposition-level verification; tentative
                             mode keeps the fully parallel approximation.
        learnable_beta: if True and soft_reset, beta is learnable; else
                        fixed at v_threshold.
        surrogate_function: a SpikingJelly surrogate function. Default
                            ATan, matching the original PSN paper.
        residual_l2: optional L2 regularization weight for Delta W. If
                     non-zero, this term is added to the module's
                     `.regularization_loss()` property for the trainer
                     to include in the total loss.
    """

    def __init__(
        self,
        T: int,
        init_tau: float = 2.0,
        init_alpha: Optional[float] = None,
        v_threshold: float = 1.0,
        learnable_kernel: bool = True,
        learnable_residual: bool = True,
        residual_init: float = 0.0,
        residual_mask: str = "causal",
        learnable_bias: bool = False,
        bias_init: Optional[float] = None,
        soft_reset: Optional[bool] = None,
        soft_reset_strategy: str = "none",
        learnable_beta: bool = False,
        surrogate_function=None,
        residual_l2: float = 1e-3,
    ) -> None:
        super().__init__()

        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}.")
        if init_tau <= 1.0:
            raise ValueError(
                f"init_tau must be > 1.0 for stable LIF dynamics "
                f"(lambda = 1 - 1/tau must lie in (0, 1)). Got {init_tau}."
            )
        if residual_mask not in {"causal", "dense"}:
            raise ValueError(
                f"residual_mask must be 'causal' or 'dense', got {residual_mask!r}."
            )
        if soft_reset is not None:
            soft_reset_strategy = "exact_detached" if soft_reset else "none"
        if soft_reset_strategy not in {"none", "tentative", "exact_detached"}:
            raise ValueError(
                "soft_reset_strategy must be one of "
                f"'none', 'tentative', 'exact_detached'; got {soft_reset_strategy!r}."
            )
        if soft_reset_strategy == "exact_detached":
            _validate_exact_detached_scope(
                init_tau=init_tau,
                init_alpha=init_alpha,
                learnable_kernel=learnable_kernel,
                learnable_residual=learnable_residual,
                residual_init=residual_init,
                residual_mask=residual_mask,
                learnable_bias=learnable_bias,
                bias_init=bias_init,
                learnable_beta=learnable_beta,
                v_threshold=v_threshold,
            )

        self.T = T
        self.v_threshold = v_threshold
        self.residual_l2 = residual_l2
        self.residual_mask = residual_mask
        self.soft_reset_strategy = soft_reset_strategy
        self.soft_reset = soft_reset_strategy != "none"

        # Learnable kernel parameters (per-layer scalars). We store log(tau)
        # to keep tau positive; the effective tau helper clamps to tau > 1
        # under unconstrained updates, matching the manuscript's stable regime.
        log_tau_init = math.log(init_tau)
        alpha_init = (1.0 / init_tau) if init_alpha is None else init_alpha

        if learnable_kernel:
            self.log_tau = nn.Parameter(torch.tensor(log_tau_init))
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer("log_tau", torch.tensor(log_tau_init))
            self.register_buffer("alpha", torch.tensor(alpha_init))

        # Learnable residual Delta W (T x T, masked at forward).
        if learnable_residual:
            self.delta_W = nn.Parameter(
                torch.full((T, T), residual_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "delta_W", torch.zeros(T, T, dtype=torch.float32)
            )

        # Residual mask. Keep causal_mask as a compatibility alias for
        # existing checks and scripts that inspect the module state.
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.float32))
        residual_mask_tensor = (
            causal_mask if residual_mask == "causal" else torch.ones(T, T, dtype=torch.float32)
        )
        self.register_buffer("causal_mask", causal_mask)
        self.register_buffer("residual_mask_tensor", residual_mask_tensor)

        # PSN-style temporal bias. With the default -V_th value,
        # surrogate(WX + b) is equivalent to thresholding WX at V_th.
        bias_value = -float(v_threshold) if bias_init is None else float(bias_init)
        bias_tensor = torch.full((T, 1), bias_value, dtype=torch.float32)
        if learnable_bias:
            self.bias = nn.Parameter(bias_tensor)
        else:
            self.register_buffer("bias", bias_tensor)

        # Soft-reset coefficient beta.
        if self.soft_reset:
            if learnable_beta:
                self.beta = nn.Parameter(torch.tensor(float(v_threshold)))
            else:
                self.register_buffer(
                    "beta", torch.tensor(float(v_threshold))
                )

        # Surrogate gradient. Default to ATan with alpha=2.0, matching
        # SpikingJelly's PSN default for clean fixed-harness comparison.
        if surrogate_function is None:
            if _SJ_AVAILABLE:
                surrogate_function = surrogate.ATan(alpha=2.0)
            else:
                raise RuntimeError(
                    "SpikingJelly not available; you must provide a "
                    "surrogate_function explicitly."
                )
        self.surrogate_function = surrogate_function

        # Mark as multi-step compatible. SpikingJelly uses this to route
        # batched temporal input directly to forward() rather than
        # unrolling timestep-by-timestep.
        if _SJ_AVAILABLE:
            self.step_mode = "m"

    # -----------------------------------------------------------------
    def kernel_weight(self) -> torch.Tensor:
        """Return the structured causal exponential kernel K."""
        return build_causal_exp_kernel(self.T, self.log_tau, self.alpha)

    # -----------------------------------------------------------------
    def residual_weight(self, masked: bool = True) -> torch.Tensor:
        """Return Delta W, optionally after applying the configured mask."""
        if masked:
            return self.delta_W * self.residual_mask_tensor
        return self.delta_W

    # -----------------------------------------------------------------
    def effective_weight(self) -> torch.Tensor:
        """Construct the effective mixing matrix W = K + mask * Delta W."""
        return self.kernel_weight() + self.residual_weight(masked=True)

    # -----------------------------------------------------------------
    def effective_bias(self) -> torch.Tensor:
        """Return the temporal bias b with shape [T, 1]."""
        return self.bias

    # -----------------------------------------------------------------
    def temporal_profile(self) -> str:
        """Return the temporal-computation profile used by this instance.

        ``exact_detached_scan`` is theorem-level exact for A2, but it
        constructs the detached spike history sequentially. ``tentative_parallel``
        keeps a fully parallel reset approximation. All no-reset variants use
        the dense GEMM profile shared with PSN in the training harness.
        """
        if self.soft_reset_strategy == "exact_detached":
            return "exact_detached_scan"
        if self.soft_reset_strategy == "tentative":
            return "tentative_parallel"
        return "dense_gemm"

    # -----------------------------------------------------------------
    def _build_W(self) -> torch.Tensor:
        """Backward-compatible alias for effective_weight()."""
        return self.effective_weight()

    # -----------------------------------------------------------------
    def _bias_view(self, target: torch.Tensor) -> torch.Tensor:
        shape = (self.T,) + (1,) * (target.dim() - 1)
        return self.effective_bias().to(device=target.device, dtype=target.dtype).view(shape)

    # -----------------------------------------------------------------
    def _exact_detached_soft_reset_spikes(self, membrane: torch.Tensor) -> torch.Tensor:
        """Sequentially construct the exact detached soft-reset spike history.

        This is intentionally used for proposition verification and A2-style
        ablations. The correction itself remains detached from the spike
        pattern, matching the theorem statement.
        """
        tau = _effective_tau(self.log_tau)
        lam = (1.0 - 1.0 / tau).to(device=membrane.device, dtype=membrane.dtype)
        beta = self.beta.to(device=membrane.device, dtype=membrane.dtype)
        correction_state = torch.zeros_like(membrane[0])
        spikes = []
        for t in range(self.T):
            corrected = membrane[t] - correction_state
            spike = self.surrogate_function(corrected - self.v_threshold).detach()
            spikes.append(spike)
            correction_state = lam * (correction_state + beta * spike)
        return torch.stack(spikes, dim=0)

    # -----------------------------------------------------------------
    def _apply_soft_reset(
        self,
        membrane: torch.Tensor,
        spike_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.soft_reset_strategy == "none":
            return membrane
        if self.soft_reset_strategy == "tentative":
            if spike_history is None:
                tentative_logits = membrane + self._bias_view(membrane)
                spike_history = self.surrogate_function(tentative_logits).detach()
        elif self.soft_reset_strategy == "exact_detached":
            if spike_history is None:
                spike_history = self._exact_detached_soft_reset_spikes(membrane)
            else:
                spike_history = spike_history.detach()
        else:  # pragma: no cover - guarded in __init__
            raise RuntimeError(self.soft_reset_strategy)

        correction = soft_reset_correction(
            spike_history.detach(), self.log_tau, self.beta
        )
        return membrane - correction

    # -----------------------------------------------------------------
    def forward_logits(
        self,
        x_seq: torch.Tensor,
        *,
        spike_history: Optional[torch.Tensor] = None,
        include_bias: bool = True,
    ) -> torch.Tensor:
        """Return LP-SN logits along the temporal axis.

        Args:
            x_seq: input tensor of shape [T, B, ...].
            spike_history: optional detached spike history for exact
                soft-reset checks.
            include_bias: if True, include the PSN-style temporal bias.
                If False, return the membrane-like pre-threshold value.
        """
        if x_seq.shape[0] != self.T:
            raise ValueError(
                f"Input time dimension {x_seq.shape[0]} does not match "
                f"configured T={self.T}."
            )

        W = self.effective_weight()  # [T, T]
        T_ = x_seq.shape[0]
        flat = x_seq.reshape(T_, -1)
        membrane_flat = W @ flat
        membrane = membrane_flat.reshape(x_seq.shape)
        membrane = self._apply_soft_reset(membrane, spike_history=spike_history)

        if include_bias:
            return membrane + self._bias_view(membrane)
        return membrane

    # -----------------------------------------------------------------
    def reset_psn_parameters(self) -> None:
        """Initialize dense Delta W and bias like SpikingJelly's PSN."""
        nn.init.kaiming_uniform_(self.delta_W, a=math.sqrt(5))
        with torch.no_grad():
            self.bias.fill_(-float(self.v_threshold))

    # -----------------------------------------------------------------
    def load_psn_parameters(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """Copy a SpikingJelly PSN weight/bias pair into this module."""
        if tuple(weight.shape) != (self.T, self.T):
            raise ValueError(f"Expected weight shape {(self.T, self.T)}, got {tuple(weight.shape)}")
        if bias.numel() != self.T:
            raise ValueError(f"Expected bias with {self.T} elements, got {bias.numel()}")
        with torch.no_grad():
            self.delta_W.copy_(weight.to(device=self.delta_W.device, dtype=self.delta_W.dtype))
            self.bias.copy_(bias.reshape(self.T, 1).to(device=self.bias.device, dtype=self.bias.dtype))

    # -----------------------------------------------------------------
    def diagnostics(self) -> dict[str, Any]:
        """Return detached tensors and scalars for heatmap/lag diagnostics."""
        K = build_causal_exp_kernel(self.T, self.log_tau, self.alpha)
        dW = self.residual_weight(masked=True)
        W = K + dW
        return {
            "T": self.T,
            "tau": float(_effective_tau(self.log_tau.detach()).cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "residual_mask": self.residual_mask,
            "soft_reset_strategy": self.soft_reset_strategy,
            "temporal_profile": self.temporal_profile(),
            "residual_l2": float(self.residual_l2),
            "K": K.detach().cpu(),
            "delta_W": dW.detach().cpu(),
            "W": W.detach().cpu(),
            "bias": self.effective_bias().detach().cpu(),
        }

    # -----------------------------------------------------------------
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_seq: input tensor of shape [T, B, ...], where the leading
                   dimension is time. Must have T equal to self.T.

        Returns:
            s_seq: spike tensor of shape [T, B, ...].
        """
        logits = self.forward_logits(x_seq, include_bias=True)
        return self.surrogate_function(logits)

    # -----------------------------------------------------------------
    def regularization_loss(self) -> torch.Tensor:
        """L2 penalty on the residual Delta W.

        Add this to the training loss to encourage banded / sparse
        structure in the learned residual (consistent with the
        renewal-process prediction in the manuscript).

        Returns:
            scalar tensor: residual_l2 * ||Delta W||_F^2,
            or 0 if delta_W is not learnable.
        """
        if self.residual_l2 <= 0:
            return torch.zeros((), device=self.delta_W.device)
        if isinstance(self.delta_W, nn.Parameter):
            return self.residual_l2 * self.residual_weight(masked=True).pow(2).sum()
        return torch.zeros((), device=self.delta_W.device)

    # -----------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"T={self.T}, tau={float(_effective_tau(self.log_tau.detach()).cpu()):.3f}, "
            f"alpha={self.alpha.item():.3f}, "
            f"residual_mask={self.residual_mask}, "
            f"soft_reset_strategy={self.soft_reset_strategy}, "
            f"temporal_profile={self.temporal_profile()}, "
            f"residual_l2={self.residual_l2}"
        )


# ---------------------------------------------------------------------
# Sanity checks.
# ---------------------------------------------------------------------
def _proposition_1_check(verbose: bool = True) -> bool:
    """Verify Proposition 1: LP-SN at the LIF-equivalent init produces
    the same pre-threshold logits as no-reset LIF on random input.

    Returns True if the equivalence holds within tolerance.
    """
    torch.manual_seed(0)
    T, B, C = 16, 4, 8
    tau_m = 2.0
    x_seq = torch.randn(T, B, C)

    # No-reset LIF: linear recurrence V[t] = lam * V[t-1] + g * X[t].
    lam = 1.0 - 1.0 / tau_m
    g = 1.0 / tau_m
    V = torch.zeros(B, C)
    lif_logits = []
    for t in range(T):
        V = lam * V + g * x_seq[t]
        lif_logits.append(V.clone())
    lif_logits = torch.stack(lif_logits, dim=0)  # [T, B, C]

    # LP-SN at the equivalence init (no soft reset, frozen residual).
    # We bypass the SpikingJelly dependency for this test by importing
    # only the build_causal_exp_kernel helper.
    log_tau = torch.tensor(math.log(tau_m))
    alpha = torch.tensor(g)
    K = build_causal_exp_kernel(T, log_tau, alpha)
    flat = x_seq.reshape(T, -1)
    lpsn_logits = (K @ flat).reshape(x_seq.shape)

    diff = (lpsn_logits - lif_logits).abs().max().item()
    if verbose:
        print(f"[Prop. 1 check] max |LP-SN - LIF| = {diff:.2e}")
    return diff < 1e-5


if __name__ == "__main__":
    # Quick self-test of the theoretical equivalence at construction.
    ok = _proposition_1_check(verbose=True)
    print(f"Proposition 1 sanity check: {'PASS' if ok else 'FAIL'}")
