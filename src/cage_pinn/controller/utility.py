from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PRNGKeyArray


@dataclass(frozen=True)
class UtilityEstimate:
    name: str
    alignment: float
    gradient_norm: float
    calibrated_cost: float
    utility: float
    positive: bool
    mode: str


def _flatten_gradient(gradient: Any) -> Array:
    vector, _ = ravel_pytree(gradient)
    return jnp.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def estimate_utilities(
    audit_gradient: Any,
    atom_gradients: dict[str, Any],
    costs: dict[str, float],
    *,
    key: PRNGKeyArray,
    exact: bool = True,
    sketch_dim: int = 64,
    preconditioner: Array | None = None,
    epsilon: float = 1.0e-12,
) -> dict[str, UtilityEstimate]:
    audit = _flatten_gradient(audit_gradient)
    if preconditioner is None:
        preconditioner = jnp.ones_like(audit)
    if preconditioner.shape != audit.shape:
        raise ValueError("Preconditioner shape does not match gradient")
    conditioned_audit = preconditioner * audit
    results: dict[str, UtilityEstimate] = {}
    atom_keys = jax.random.split(key, max(1, len(atom_gradients)))
    for index, (name, gradient) in enumerate(atom_gradients.items()):
        atom = _flatten_gradient(gradient)
        if atom.shape != audit.shape:
            raise ValueError(f"Gradient shape mismatch for atom {name}")
        conditioned_atom = preconditioner * atom
        norm = jnp.sqrt(jnp.vdot(atom, conditioned_atom).real + epsilon)
        if exact:
            alignment = jnp.vdot(conditioned_audit, atom).real
            mode = "exact"
        else:
            if sketch_dim <= 0:
                raise ValueError("sketch_dim must be positive")
            projection = jax.random.rademacher(
                atom_keys[index], (sketch_dim, atom.size), dtype=atom.dtype
            ) / jnp.sqrt(float(sketch_dim))
            audit_sketch = projection @ conditioned_audit
            atom_sketch = projection @ atom
            alignment = jnp.vdot(audit_sketch, atom_sketch).real
            mode = f"rademacher_{sketch_dim}"
        alignment = jnp.nan_to_num(alignment, nan=0.0, posinf=0.0, neginf=0.0)
        cost = max(float(costs[name]), epsilon)
        utility = jnp.maximum(alignment, 0.0) / (cost * (norm + epsilon))
        results[name] = UtilityEstimate(
            name=name,
            alignment=float(alignment),
            gradient_norm=float(norm),
            calibrated_cost=cost,
            utility=float(utility),
            positive=bool(alignment > 0),
            mode=mode,
        )
    return results

