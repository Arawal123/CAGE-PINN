from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from cage_pinn.backbones import create_backbone
from cage_pinn.budgets.ledger import CostModel
from cage_pinn.core.schemas import capture_environment, stable_hash
from cage_pinn.enforcement import default_atom_bank
from cage_pinn.pdes import create_problem


@dataclass(frozen=True)
class CalibrationEntry:
    atom: str
    symbolic_tokens: int
    median_seconds: float
    seconds_per_token: float
    repeats: int
    warmups: int


def _block_tree(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def calibrate_costs(
    *,
    problem_name: str = "poisson_1d",
    backbone_name: str = "vanilla",
    points: int = 32,
    repeats: int = 5,
    warmups: int = 2,
    precision: str = "float64",
    output_directory: str | Path = "results/calibration",
) -> tuple[Path, dict[str, Any]]:
    if repeats <= 0 or warmups < 0 or points <= 0:
        raise ValueError("Invalid calibration sizes")
    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
    problem = create_problem(problem_name)
    model = create_backbone(
        backbone_name,
        input_dim=problem.geometry.dimension,
        output_dim=problem.output_dim,
        width=12,
        depth=1,
        key=jax.random.PRNGKey(17),
        lower=float(problem.geometry.lower[0]),
        upper=float(problem.geometry.upper[0]),
    )
    sample = problem.sample_interior(jax.random.PRNGKey(18), points)
    residual = problem.residuals(model, sample)
    scales = jnp.maximum(jnp.median(jnp.abs(residual), axis=0), 1.0e-8)
    cost_model = CostModel()
    atoms = default_atom_bank(weak_quadrature_points=4)
    entries = []
    for index, (name, atom) in enumerate(atoms.items()):
        if not atom.applicable(problem)[0]:
            continue
        atom_key = jax.random.fold_in(jax.random.PRNGKey(19), index)

        def loss_function(
            candidate: Any,
            fixed_atom: Any = atom,
            fixed_key: jax.Array = atom_key,
        ) -> jax.Array:
            return fixed_atom.evaluate(
                candidate, problem, sample, scales, fixed_key, cost_model
            ).loss

        measured_tokens = atom.evaluate(
            model, problem, sample, scales, atom_key, cost_model
        ).tokens
        for _ in range(warmups):
            _block_tree(eqx.filter_value_and_grad(loss_function)(model))
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            _block_tree(eqx.filter_value_and_grad(loss_function)(model))
            timings.append(time.perf_counter() - start)
        median = statistics.median(timings)
        entries.append(
            CalibrationEntry(
                atom=name,
                symbolic_tokens=measured_tokens,
                median_seconds=median,
                seconds_per_token=median / max(measured_tokens, 1),
                repeats=repeats,
                warmups=warmups,
            )
        )
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "problem": problem_name,
        "backbone": backbone_name,
        "points": points,
        "precision": precision,
        "hardware": capture_environment(),
        "entries": [asdict(entry) for entry in entries],
        "scope": (
            "Device-specific diagnostic calibration; use identical warm-up and "
            "exclusive hardware for comparisons."
        ),
    }
    calibration_id = stable_hash(payload)[:16]
    target_dir = Path(output_directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"calibration-{calibration_id}.json"
    if target.exists():
        raise FileExistsError(f"Calibration record is immutable: {target}")
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target, payload
