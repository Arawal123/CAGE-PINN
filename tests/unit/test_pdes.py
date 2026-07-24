import jax
import jax.numpy as jnp
import pytest

from cage_pinn.pdes import create_problem, problem_registry


@pytest.mark.parametrize("name", sorted(problem_registry()))
def test_analytic_reference_has_near_zero_residual(name: str) -> None:
    jax.config.update("jax_enable_x64", True)
    problem = create_problem(name)
    if not problem.reference_available:
        pytest.skip(f"{name} requires an external validated reference")
    points = problem.sample_interior(jax.random.PRNGKey(13), 12)
    residual = problem.residuals(problem.reference_point, points)
    assert residual.shape == (12, len(problem.residual_channel_names))
    assert float(jnp.max(jnp.abs(residual))) < 1.0e-9


@pytest.mark.parametrize("name", sorted(problem_registry()))
def test_reference_shape(name: str) -> None:
    problem = create_problem(name)
    if not problem.reference_available:
        pytest.skip(f"{name} requires an external validated reference")
    points = problem.sample_interior(jax.random.PRNGKey(19), 7)
    assert problem.reference(points).shape == (7, problem.output_dim)
