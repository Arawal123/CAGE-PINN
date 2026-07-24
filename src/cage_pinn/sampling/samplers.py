from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from cage_pinn.geometry import Box
from cage_pinn.pdes import PDEProblem


@dataclass(frozen=True)
class UniformSampler:
    geometry: Box

    def sample(self, key: PRNGKeyArray, n: int, **context: Any) -> Array:
        del context
        return self.geometry.sample(key, n)


@dataclass(frozen=True)
class ResidualAdaptiveSampler:
    problem: PDEProblem
    oversample: int = 4
    exploration: float = 0.1

    def sample(
        self,
        key: PRNGKeyArray,
        n: int,
        **context: Any,
    ) -> Array:
        model = context.get("model")
        scales = context.get("scales")
        if model is None or scales is None:
            raise ValueError("ResidualAdaptiveSampler requires model and scales")
        candidate = self.problem.sample_interior(key, max(self.oversample * n, n))
        residual = self.problem.residuals(model, candidate) / (
            jnp.asarray(scales) + 1.0e-8
        )
        score = jnp.max(jnp.abs(residual), axis=1)
        probability = score + jnp.mean(score) * self.exploration + 1.0e-12
        probability /= jnp.sum(probability)
        indices = jax.random.choice(
            key, candidate.shape[0], (n,), replace=False, p=probability
        )
        return candidate[indices]

