from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array


class ResidualScaleTracker:
    """Lagged robust channel scales that never receive model gradients."""

    def __init__(self, channels: int, *, decay: float = 0.95, epsilon: float = 1.0e-8) -> None:
        if channels <= 0 or not 0.0 <= decay < 1.0 or epsilon <= 0:
            raise ValueError("Invalid residual scale tracker configuration")
        self.decay = decay
        self.epsilon = epsilon
        self._scales = jnp.ones((channels,))
        self._initialized = False

    @property
    def scales(self) -> Array:
        return jax.lax.stop_gradient(self._scales)

    def update(self, residuals: Array) -> Array:
        if residuals.ndim != 2 or residuals.shape[1] != self._scales.shape[0]:
            raise ValueError("Residual array shape does not match channel scales")
        robust = jnp.median(jnp.abs(jax.lax.stop_gradient(residuals)), axis=0)
        robust = jnp.maximum(robust, self.epsilon)
        self._scales = (
            robust if not self._initialized else self.decay * self._scales + (1 - self.decay) * robust
        )
        self._initialized = True
        return self.scales

    def normalize(self, residuals: Array) -> Array:
        return residuals / (self.scales + self.epsilon)


def bounded_residual_score(normalized_residuals: Array) -> Array:
    squared = normalized_residuals**2
    return squared / (1.0 + squared)


def empirical_cvar(values: Array, q: float = 0.95) -> Array:
    """Empirical upper-tail CVaR using the finite-sample quantile definition."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError("CVaR values must be a non-empty vector")
    if not 0.0 <= q < 1.0:
        raise ValueError("q must be in [0, 1)")
    sorted_values = jnp.sort(values)
    start = math.floor(q * values.size)
    return jnp.mean(sorted_values[min(start, values.size - 1) :])


def fixed_schedule_cvar_upper(
    empirical: Array,
    *,
    n: int,
    q: float,
    delta: float,
    planned_audits: int,
    channels: int,
) -> Array:
    """Conservative union-corrected upper bound for bounded-score CVaR.

    The correction combines a bounded empirical-process deviation with the
    1/(1-q) CVaR Lipschitz factor. It is valid only for the predeclared finite
    audit schedule under independent draws. It is not an anytime confidence
    sequence and is not a solution-error certificate.
    """
    if n <= 0 or planned_audits <= 0 or channels <= 0:
        raise ValueError("n, planned_audits, and channels must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    corrected_delta = delta / (planned_audits * channels)
    deviation = math.sqrt(math.log(2.0 / corrected_delta) / (2.0 * n))
    return jnp.minimum(1.0, empirical + deviation / max(1.0 - q, 1.0e-12))


def smooth_max(values: Array, temperature: float = 0.1) -> Array:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return temperature * jax.scipy.special.logsumexp(values / temperature)


@dataclass(frozen=True)
class AuditRisk:
    channel_mean: Array
    channel_cvar: Array
    channel_upper: Array
    aggregate: Array
    empirical_aggregate: Array
    upper_aggregate: Array
    empirical_only: bool
    bound_scope: str

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_mean": [float(value) for value in self.channel_mean],
            "channel_cvar": [float(value) for value in self.channel_cvar],
            "channel_upper": [float(value) for value in self.channel_upper],
            "aggregate": float(self.aggregate),
            "empirical_aggregate": float(self.empirical_aggregate),
            "upper_aggregate": float(self.upper_aggregate),
            "empirical_only": self.empirical_only,
            "bound_scope": self.bound_scope,
        }


def estimate_audit_risk(
    normalized_residuals: Array,
    *,
    q: float = 0.95,
    delta: float = 0.05,
    planned_audits: int = 100,
    certified_schedule: bool = True,
    temperature: float = 0.1,
) -> AuditRisk:
    scores = bounded_residual_score(normalized_residuals)
    means = jnp.mean(scores, axis=0)
    cvars = jnp.stack(tuple(empirical_cvar(scores[:, index], q) for index in range(scores.shape[1])))
    if certified_schedule:
        upper = fixed_schedule_cvar_upper(
            cvars,
            n=scores.shape[0],
            q=q,
            delta=delta,
            planned_audits=planned_audits,
            channels=scores.shape[1],
        )
        scope = "fixed finite audit schedule, iid bounded scores, union correction"
    else:
        upper = cvars
        scope = "empirical tail metric; no confidence guarantee"
    empirical_aggregate = smooth_max(cvars, temperature)
    upper_aggregate = smooth_max(upper, temperature)
    return AuditRisk(
        channel_mean=means,
        channel_cvar=cvars,
        channel_upper=upper,
        aggregate=upper_aggregate if certified_schedule else empirical_aggregate,
        empirical_aggregate=empirical_aggregate,
        upper_aggregate=upper_aggregate,
        empirical_only=not certified_schedule,
        bound_scope=scope,
    )
