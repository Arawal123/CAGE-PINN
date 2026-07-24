from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from cage_pinn.budgets.ledger import CostModel
from cage_pinn.pdes import PDEProblem


@dataclass(frozen=True)
class AtomEvaluation:
    name: str
    loss: Array
    tokens: int
    provenance: dict[str, Any]
    applicable: bool = True
    reason: str = ""


class EnforcementAtom(ABC):
    name: str

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        problem: PDEProblem,
        points: Array,
        scales: Array,
        key: PRNGKeyArray,
        cost_model: CostModel,
    ) -> AtomEvaluation:
        pass

    def applicable(self, problem: PDEProblem) -> tuple[bool, str]:
        return True, ""


class FreshStrongAtom(EnforcementAtom):
    name = "S"

    def evaluate(
        self,
        model: Any,
        problem: PDEProblem,
        points: Array,
        scales: Array,
        key: PRNGKeyArray,
        cost_model: CostModel,
    ) -> AtomEvaluation:
        del key
        residuals = problem.residuals(model, points)
        normalized = residuals / (jax.lax.stop_gradient(scales) + 1.0e-8)
        loss = jnp.mean(normalized**2)
        tokens = round(
            cost_model.residual(
                len(points), problem.residual_derivative_order, backward=True
            )
        )
        return AtomEvaluation(
            self.name,
            loss,
            tokens,
            {
                "points": len(points),
                "freshness_managed_by_sampler": True,
                "point_type": "interior",
            },
        )


class ResidualSlopeAtom(EnforcementAtom):
    name = "J"

    def applicable(self, problem: PDEProblem) -> tuple[bool, str]:
        if not problem.smooth:
            return False, "Residual-input slopes are disabled for non-smooth problems"
        if problem.residual_derivative_order >= 3:
            return False, "Configured derivative order would make J prohibitively high-order"
        return True, ""

    def evaluate(
        self,
        model: Any,
        problem: PDEProblem,
        points: Array,
        scales: Array,
        key: PRNGKeyArray,
        cost_model: CostModel,
    ) -> AtomEvaluation:
        del key
        valid, reason = self.applicable(problem)
        if not valid:
            return AtomEvaluation(self.name, jnp.asarray(0.0), 0, {}, False, reason)
        fixed_scales = jax.lax.stop_gradient(scales)

        def normalized_point(z: Array) -> Array:
            return problem.residual_point(model, z) / (fixed_scales + 1.0e-8)

        jacobians = jax.vmap(jax.jacfwd(normalized_point))(points)
        clipped = jnp.clip(jacobians, -1.0e3, 1.0e3)
        loss = jnp.mean(jnp.sum(clipped**2, axis=-1))
        tokens = round(
            cost_model.residual(
                len(points), problem.residual_derivative_order + 1, backward=True
            )
        )
        return AtomEvaluation(
            self.name,
            loss,
            tokens,
            {
                "points": len(points),
                "coordinate_order": 1,
                "normalization_stop_gradient": True,
            },
        )


class WeakWitnessAtom(EnforcementAtom):
    name = "W"

    def __init__(
        self,
        *,
        scales: tuple[float, ...] = (0.05, 0.15, 0.30),
        quadrature_points: int = 6,
        witnesses: int = 8,
    ) -> None:
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("Weak scales must be positive")
        if quadrature_points <= 1 or witnesses <= 0:
            raise ValueError("Weak quadrature/witness count is invalid")
        self.scales = scales
        self.quadrature_points = quadrature_points
        self.witnesses = witnesses

    def applicable(self, problem: PDEProblem) -> tuple[bool, str]:
        if not problem.weak_form_valid:
            return False, "No mathematically justified generic weak form for this problem"
        return True, ""

    @staticmethod
    def bump(offset: Array) -> Array:
        radius_sq = jnp.sum(offset**2, axis=-1)
        return jnp.where(radius_sq < 1.0, (1.0 - radius_sq) ** 2, 0.0)

    def evaluate(
        self,
        model: Any,
        problem: PDEProblem,
        points: Array,
        scales: Array,
        key: PRNGKeyArray,
        cost_model: CostModel,
    ) -> AtomEvaluation:
        valid, reason = self.applicable(problem)
        if not valid:
            return AtomEvaluation(self.name, jnp.asarray(0.0), 0, {}, False, reason)
        center_key, offset_key, scale_key = jax.random.split(key, 3)
        center_indices = jax.random.randint(
            center_key, (self.witnesses,), 0, points.shape[0]
        )
        centers = points[center_indices]
        scale_indices = jax.random.randint(
            scale_key, (self.witnesses,), 0, len(self.scales)
        )
        radii = jnp.asarray(self.scales)[scale_indices]
        raw_offsets = jax.random.uniform(
            offset_key,
            (self.witnesses, self.quadrature_points, points.shape[1]),
            minval=-1.0,
            maxval=1.0,
        )
        quadrature = centers[:, None, :] + radii[:, None, None] * raw_offsets
        quadrature = problem.geometry.clip(quadrature)
        flat = quadrature.reshape((-1, points.shape[1]))
        normalized = problem.residuals(model, flat) / (
            jax.lax.stop_gradient(scales) + 1.0e-8
        )
        normalized = normalized.reshape(
            (self.witnesses, self.quadrature_points, -1)
        )
        phi = self.bump(raw_offsets)
        moments = jnp.mean(normalized * phi[..., None], axis=1)
        volumes = (2.0 * radii) ** points.shape[1]
        moments = moments * volumes[:, None]
        loss = jnp.mean(moments**2)
        tokens = round(
            cost_model.weak(
                self.witnesses,
                self.quadrature_points,
                problem.residual_derivative_order,
            )
            * cost_model.backward_multiplier
        )
        return AtomEvaluation(
            self.name,
            loss,
            tokens,
            {
                "witnesses": self.witnesses,
                "quadrature_points": self.quadrature_points,
                "scales": list(self.scales),
                "basis": "compact_c2_bump",
                "randomized": True,
            },
        )


def default_atom_bank(*, weak_quadrature_points: int = 6) -> dict[str, EnforcementAtom]:
    return {
        "S": FreshStrongAtom(),
        "J": ResidualSlopeAtom(),
        "W": WeakWitnessAtom(quadrature_points=weak_quadrature_points),
    }
