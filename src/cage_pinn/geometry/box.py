from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


@dataclass(frozen=True)
class Box:
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lower) != len(self.upper):
            raise ValueError("lower and upper must have the same dimension")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise ValueError("Every lower bound must be below its upper bound")

    @property
    def dimension(self) -> int:
        return len(self.lower)

    @property
    def volume(self) -> float:
        value = 1.0
        for lo, hi in zip(self.lower, self.upper, strict=True):
            value *= hi - lo
        return value

    def sample(self, key: PRNGKeyArray, n: int) -> Array:
        if n <= 0:
            raise ValueError("n must be positive")
        lo = jnp.asarray(self.lower)
        hi = jnp.asarray(self.upper)
        return jax.random.uniform(key, (n, self.dimension), minval=lo, maxval=hi)

    def clip(self, points: Array) -> Array:
        return jnp.clip(points, jnp.asarray(self.lower), jnp.asarray(self.upper))

