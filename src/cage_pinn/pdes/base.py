from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from cage_pinn.geometry import Box

ModelFn = Callable[[Array], Array]


class PDEProblem(ABC):
    """Typed PDE contract used by training, audit, and reference verification."""

    name: str
    geometry: Box
    output_dim: int
    residual_channel_names: tuple[str, ...]
    residual_derivative_order: int
    smooth: bool = True
    weak_form_valid: bool = True
    reference_available: bool = True
    reference_kind: str = "analytic"

    def sample_interior(self, key: PRNGKeyArray, n: int) -> Array:
        return self.geometry.sample(key, n)

    @abstractmethod
    def residual_point(self, model: ModelFn, z: Array) -> Array:
        """Return one nondimensionalized residual value per channel."""

    def residuals(self, model: ModelFn, points: Array) -> Array:
        values = jax.vmap(lambda z: self.residual_point(model, z))(points)
        if values.ndim != 2 or values.shape[1] != len(self.residual_channel_names):
            raise ValueError(
                f"{self.name} residual shape {values.shape} does not match "
                f"{len(self.residual_channel_names)} channels"
            )
        return values

    @abstractmethod
    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        """Return scalar mean-square boundary/initial/interface loss."""

    @abstractmethod
    def reference(self, points: Array) -> Array:
        """Trusted analytic reference used only for post-training evaluation."""

    def relative_l2(self, model: ModelFn, points: Array, epsilon: float = 1.0e-12) -> Array:
        prediction = jax.vmap(model)(points)
        truth = self.reference(points)
        return jnp.linalg.norm(prediction - truth) / (jnp.linalg.norm(truth) + epsilon)

    def reference_residual_max(self, points: Array) -> Array:
        return jnp.max(jnp.abs(self.residuals(self.reference_point, points)))

    def reference_point(self, z: Array) -> Array:
        return self.reference(z[None, :])[0]
