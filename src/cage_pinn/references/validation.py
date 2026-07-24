from __future__ import annotations

from dataclasses import asdict, dataclass

import jax

from cage_pinn.pdes import create_problem, problem_registry


@dataclass(frozen=True)
class ReferenceCheck:
    problem: str
    points: int
    max_abs_residual: float
    tolerance: float
    passed: bool
    reference_kind: str = "analytic"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_analytic_references(
    *, points: int = 32, tolerance: float = 1.0e-9, seed: int = 711
) -> list[ReferenceCheck]:
    jax.config.update("jax_enable_x64", True)
    checks = []
    for index, name in enumerate(problem_registry()):
        problem = create_problem(name)
        if not problem.reference_available:
            continue
        sample = problem.sample_interior(jax.random.PRNGKey(seed + index), points)
        maximum = float(problem.reference_residual_max(sample))
        checks.append(
            ReferenceCheck(
                problem=name,
                points=points,
                max_abs_residual=maximum,
                tolerance=tolerance,
                passed=maximum <= tolerance,
            )
        )
    return checks
