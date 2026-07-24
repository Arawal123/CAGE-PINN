import jax
import jax.numpy as jnp
import numpy as np

from cage_pinn.audit import (
    AuditFoldManager,
    ResidualScaleTracker,
    bounded_residual_score,
    empirical_cvar,
    estimate_audit_risk,
)
from cage_pinn.geometry import Box


def test_bounded_score_and_cvar() -> None:
    residual = jnp.asarray([[-10.0], [0.0], [1.0], [2.0]])
    score = bounded_residual_score(residual)[:, 0]
    assert jnp.all(score >= 0)
    assert jnp.all(score <= 1)
    assert np.isclose(float(empirical_cvar(score, 0.5)), float(jnp.mean(jnp.sort(score)[2:])))


def test_union_corrected_bound_dominates_empirical() -> None:
    residual = jnp.linspace(-2, 2, 200)[:, None]
    risk = estimate_audit_risk(
        residual, q=0.9, delta=0.05, planned_audits=10, certified_schedule=True
    )
    assert jnp.all(risk.channel_upper >= risk.channel_cvar)
    assert jnp.all(risk.channel_upper <= 1.0)
    assert not risk.empirical_only


def test_scale_is_stop_gradient_and_scaling_invariant() -> None:
    tracker = ResidualScaleTracker(2, decay=0.0)
    raw = jnp.asarray([[1.0, 20.0], [-1.0, -20.0], [2.0, 40.0]])
    tracker.update(raw)
    normalized = tracker.normalize(raw)
    scaled_tracker = ResidualScaleTracker(2, decay=0.0)
    scaled = raw * jnp.asarray([10.0, 0.1])
    scaled_tracker.update(scaled)
    assert jnp.allclose(jnp.abs(normalized), jnp.abs(scaled_tracker.normalize(scaled)))
    gradient = jax.grad(lambda x: jnp.sum(x / tracker.scales))(jnp.ones((2,)))
    assert jnp.all(jnp.isfinite(gradient))


def test_fold_roles_rotate_refresh_and_do_not_overlap() -> None:
    manager = AuditFoldManager(
        Box((0.0,), (1.0,)),
        jax.random.PRNGKey(5),
        fold_size=16,
        rotate_interval=1,
        refresh_after_selections=1,
        prohibited_window=2,
    )
    initial_roles = manager.roles
    learner = jnp.linspace(0.001, 0.999, 15)[:, None]
    manager.register_learner(learner, step=0)
    manager.on_control(1)
    assert manager.roles != initial_roles
    assert manager.leakage_report().passed
    assert len(manager.provenance()) >= 3 * 16 + 15

