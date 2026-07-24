from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


@dataclass
class ReLoBRaLoState:
    """Reference clean-room relative-loss balancing primitive."""

    losses_at_start: jax.Array
    losses_previous: jax.Array
    weights: jax.Array
    temperature: float = 0.1
    ema: float = 0.99

    @classmethod
    def initialize(cls, losses: jax.Array) -> ReLoBRaLoState:
        if losses.ndim != 1 or jnp.any(losses <= 0):
            raise ValueError("Initial losses must be a positive vector")
        return cls(losses, losses, jnp.ones_like(losses))

    def update(self, losses: jax.Array, *, lookback_to_start: bool) -> jax.Array:
        reference = self.losses_at_start if lookback_to_start else self.losses_previous
        logits = losses / (reference + 1.0e-12) / self.temperature
        target = losses.size * jax.nn.softmax(logits)
        self.weights = self.ema * self.weights + (1.0 - self.ema) * target
        self.losses_previous = losses
        return self.weights


def config_two_gradients(first: Any, second: Any, epsilon: float = 1.0e-12) -> Any:
    """Conflict-free equal-rate direction for two gradient pytrees."""
    flat_first, unravel = ravel_pytree(first)
    flat_second, _ = ravel_pytree(second)
    first_unit = flat_first / (jnp.linalg.norm(flat_first) + epsilon)
    second_unit = flat_second / (jnp.linalg.norm(flat_second) + epsilon)
    direction = first_unit + second_unit
    scale = 2.0 / (
        1.0 / (jnp.linalg.norm(flat_first) + epsilon)
        + 1.0 / (jnp.linalg.norm(flat_second) + epsilon)
    )
    return unravel(scale * direction / (jnp.linalg.norm(direction) + epsilon))
