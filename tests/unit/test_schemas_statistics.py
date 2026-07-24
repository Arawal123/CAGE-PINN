from pathlib import Path

import numpy as np
import pytest

from cage_pinn.core import ExperimentManifest, ResultRecord, RunConfig, SeedHierarchy
from cage_pinn.statistics import analyze_paired, holm_adjust


def test_seed_hierarchy_is_deterministic_and_separated() -> None:
    first = SeedHierarchy.derive(7)
    second = SeedHierarchy.derive(7)
    assert first == second
    assert len({first.model, first.learner, first.audit, first.weak, first.controller}) == 5


def test_result_record_is_immutable(tmp_path: Path) -> None:
    record = ResultRecord.begin(RunConfig(steps=1))
    record.finish()
    target = record.write_immutable(tmp_path)
    assert target.exists()
    with pytest.raises(FileExistsError):
        record.write_immutable(tmp_path)


def test_manifest_validation() -> None:
    manifest = ExperimentManifest.from_yaml("experiments/manifests/discovery.yaml")
    assert manifest.study == "discovery"
    assert not manifest.frozen


def test_paired_statistics_and_holm() -> None:
    cage = np.asarray([0.8, 0.9, 1.0, 0.7, 0.95])
    base = np.ones(5)
    result = analyze_paired(cage, base, bootstrap_seed=3)
    assert result.pairs == 5
    assert result.geometric_mean_ratio < 1.0
    adjusted = holm_adjust([0.01, 0.04, 0.2])
    assert all(0 <= value <= 1 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]

