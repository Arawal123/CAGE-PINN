from pathlib import Path

import pytest

from cage_pinn.core import RunConfig
from cage_pinn.training import run_training


@pytest.mark.integration
@pytest.mark.parametrize(
    ("method", "backbone"),
    [
        ("vanilla", "vanilla"),
        ("cage", "vanilla"),
        ("cage", "xpinn"),
        ("cage", "ab_pinn"),
    ],
)
def test_one_step_training_writes_auditable_result(
    tmp_path: Path, method: str, backbone: str
) -> None:
    outcome = run_training(
        RunConfig(
            problem="poisson_1d",
            backbone=backbone,
            method=method,
            steps=1,
            width=6,
            depth=1,
            learner_points=6,
            boundary_points=4,
            audit_points=6,
            control_interval=1,
            total_ad_tokens=50_000,
            output=str(tmp_path),
        )
    )
    assert outcome.path is not None and outcome.path.exists()
    assert outcome.result.status == "completed"
    assert outcome.result.metrics["leakage"]["passed"]
    assert outcome.result.metrics["reference_used_during_training"] is False
    assert outcome.result.ledger["spent_tokens"] <= outcome.result.ledger["total_tokens"]
    assert "weak_audit" in outcome.result.history[0]["monitor"]
    if method == "cage":
        assert outcome.result.history[0]["calibration"] is not None
        assert len(
            outcome.result.history[0]["monitor"]["weak_audit"]["seed_words"]
        ) == 2
