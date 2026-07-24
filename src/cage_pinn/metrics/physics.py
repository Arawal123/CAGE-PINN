from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def relative_l2(prediction: Array, reference: Array, epsilon: float = 1.0e-12) -> Array:
    if prediction.shape != reference.shape:
        raise ValueError("Prediction/reference shape mismatch")
    return jnp.linalg.norm(prediction - reference) / (
        jnp.linalg.norm(reference) + epsilon
    )


def dense_residual_metrics(residuals: Array) -> dict[str, Array]:
    if residuals.ndim != 2 or residuals.shape[0] == 0:
        raise ValueError("Residuals must be a non-empty point-by-channel array")
    return {
        "channel_mean_abs": jnp.mean(jnp.abs(residuals), axis=0),
        "channel_rms": jnp.sqrt(jnp.mean(residuals**2, axis=0)),
        "channel_max_abs": jnp.max(jnp.abs(residuals), axis=0),
    }

