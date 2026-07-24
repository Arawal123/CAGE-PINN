import jax
import jax.numpy as jnp
import pytest

from cage_pinn.audit import ResidualScaleTracker
from cage_pinn.backbones import create_backbone, parameter_count
from cage_pinn.budgets import CostModel
from cage_pinn.enforcement import default_atom_bank
from cage_pinn.pdes import create_problem


@pytest.mark.parametrize("name", ["vanilla", "xpinn", "ab_pinn"])
def test_backbone_shapes_and_finite_interface_loss(name: str) -> None:
    model = create_backbone(
        name,
        input_dim=1,
        output_dim=1,
        width=12,
        depth=1,
        key=jax.random.PRNGKey(2),
        lower=0.0,
        upper=1.0,
    )
    assert model(jnp.asarray([0.3])).shape == (1,)
    assert parameter_count(model) > 0
    assert jnp.isfinite(model.interface_loss())


@pytest.mark.parametrize("atom_name", ["S", "J", "W"])
def test_atom_losses_are_finite(atom_name: str) -> None:
    problem = create_problem("poisson_1d")
    model = create_backbone(
        "vanilla",
        input_dim=1,
        output_dim=1,
        width=8,
        depth=1,
        key=jax.random.PRNGKey(3),
        lower=0.0,
        upper=1.0,
    )
    points = problem.sample_interior(jax.random.PRNGKey(4), 8)
    residual = problem.residuals(model, points)
    tracker = ResidualScaleTracker(1, decay=0.0)
    tracker.update(residual)
    atom = default_atom_bank(weak_quadrature_points=4)[atom_name]
    evaluation = atom.evaluate(
        model, problem, points, tracker.scales, jax.random.PRNGKey(5), CostModel()
    )
    assert evaluation.applicable
    assert jnp.isfinite(evaluation.loss)
    assert evaluation.tokens > 0
