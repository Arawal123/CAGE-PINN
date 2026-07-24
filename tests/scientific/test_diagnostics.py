import jax
import jax.numpy as jnp
import pytest

from cage_pinn.budgets import CostModel
from cage_pinn.controller import estimate_utilities
from cage_pinn.enforcement import ResidualSlopeAtom, WeakWitnessAtom
from cage_pinn.pdes import create_problem

pytestmark = pytest.mark.scientific


def test_collocation_polynomial_exposes_train_audit_gap() -> None:
    train = jnp.linspace(0.0, 1.0, 8)
    audit = jnp.linspace(0.03, 0.97, 31)

    def residual(points: jax.Array) -> jax.Array:
        differences = points[:, None] - train[None, :]
        return jnp.prod(differences, axis=1)

    train_risk = jnp.mean(residual(train) ** 2)
    audit_risk = jnp.mean(residual(audit) ** 2)
    assert train_risk < 1.0e-20
    assert audit_risk > train_risk + 1.0e-12


def test_j_and_w_detect_off_grid_oscillation() -> None:
    jax.config.update("jax_enable_x64", True)
    problem = create_problem("poisson_1d")

    def exact(z: jax.Array) -> jax.Array:
        return jnp.asarray([jnp.sin(jnp.pi * z[0])])

    def oscillatory(z: jax.Array) -> jax.Array:
        return jnp.asarray(
            [jnp.sin(jnp.pi * z[0]) + 0.02 * jnp.sin(12.0 * jnp.pi * z[0])]
        )

    points = problem.sample_interior(jax.random.PRNGKey(31), 24)
    scales = jnp.ones((1,))
    cost = CostModel()
    slope = ResidualSlopeAtom()
    weak = WeakWitnessAtom(scales=(0.08, 0.2), quadrature_points=12, witnesses=12)
    exact_j = slope.evaluate(
        exact, problem, points, scales, jax.random.PRNGKey(32), cost
    ).loss
    oscillatory_j = slope.evaluate(
        oscillatory, problem, points, scales, jax.random.PRNGKey(32), cost
    ).loss
    exact_w = weak.evaluate(
        exact, problem, points, scales, jax.random.PRNGKey(33), cost
    ).loss
    oscillatory_w = weak.evaluate(
        oscillatory, problem, points, scales, jax.random.PRNGKey(33), cost
    ).loss
    assert oscillatory_j > exact_j + 1.0e-8
    assert oscillatory_w > exact_w + 1.0e-10


def test_positive_utility_predicts_quadratic_audit_decrease() -> None:
    theta = jnp.asarray([1.0, -2.0])
    audit_gradient = {"theta": theta}
    atom_gradients = {
        "aligned": {"theta": theta},
        "opposed": {"theta": -theta},
    }
    utilities = estimate_utilities(
        audit_gradient,
        atom_gradients,
        {"aligned": 1.0, "opposed": 1.0},
        key=jax.random.PRNGKey(34),
        exact=True,
    )
    selected = max(utilities, key=lambda name: utilities[name].utility)
    update = atom_gradients[selected]["theta"]
    before = 0.5 * jnp.sum(theta**2)
    after = 0.5 * jnp.sum((theta - 0.1 * update) ** 2)
    assert selected == "aligned"
    assert after < before
