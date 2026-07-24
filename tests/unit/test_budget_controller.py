import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cage_pinn.budgets import BudgetExceeded, BudgetLedger
from cage_pinn.controller.allocation import BudgetAllocator, project_bounded_simplex
from cage_pinn.controller.utility import estimate_utilities


def test_bounded_simplex_projection() -> None:
    result = project_bounded_simplex(
        np.asarray([3.0, -1.0, 0.2]),
        np.asarray([0.1, 0.05, 0.05]),
        np.asarray([0.8, 0.6, 0.5]),
    )
    assert np.isclose(result.sum(), 1.0)
    assert np.all(result >= np.asarray([0.1, 0.05, 0.05]) - 1.0e-10)
    assert np.all(result <= np.asarray([0.8, 0.6, 0.5]) + 1.0e-10)


def test_allocator_exact_token_conservation_and_fallback() -> None:
    allocator = BudgetAllocator()
    result = allocator.allocate(
        {"S": 0.0, "J": 0.0, "W": 0.0},
        applicable={"S": True, "J": True, "W": True},
        tokens=101,
    )
    assert sum(result.realized_tokens.values()) == 101
    assert result.fallback
    assert np.isclose(sum(result.requested.values()), 1.0)


def test_budget_ledger_refuses_overspend() -> None:
    ledger = BudgetLedger(10)
    ledger.charge("x", 7, step=0)
    with pytest.raises(BudgetExceeded):
        ledger.charge("y", 4, step=1)
    assert ledger.spent_tokens == 7


def test_utility_sign_convention() -> None:
    audit = {"x": jnp.asarray([1.0, 0.0])}
    atoms = {
        "aligned": {"x": jnp.asarray([2.0, 0.0])},
        "opposed": {"x": jnp.asarray([-2.0, 0.0])},
    }
    estimates = estimate_utilities(
        audit,
        atoms,
        {"aligned": 1.0, "opposed": 1.0},
        key=jax.random.PRNGKey(0),
        exact=True,
    )
    assert estimates["aligned"].utility > 0
    assert estimates["opposed"].utility == 0


def test_sketch_tracks_exact_alignment() -> None:
    audit = {"x": jnp.linspace(-1.0, 1.0, 32)}
    atoms = {"same": {"x": jnp.linspace(-1.0, 1.0, 32)}}
    exact = estimate_utilities(
        audit, atoms, {"same": 1.0}, key=jax.random.PRNGKey(1), exact=True
    )
    sketch = estimate_utilities(
        audit,
        atoms,
        {"same": 1.0},
        key=jax.random.PRNGKey(1),
        exact=False,
        sketch_dim=2048,
    )
    assert exact["same"].alignment > 0
    assert sketch["same"].alignment > 0
    assert abs(sketch["same"].alignment / exact["same"].alignment - 1.0) < 0.2

