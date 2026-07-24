from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def load_generator() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "benchmarks" / "generate_1d_references.py"
    spec = importlib.util.spec_from_file_location("generate_1d_references", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import reference generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.scientific
def test_burgers_reference_solver_reaches_final_time() -> None:
    generator = load_generator()
    x, t, values = generator.solve("burgers_1d", 128, 5, 1.0e-6, 1.0e-8)

    assert x.shape == (128,)
    assert t[-1] == pytest.approx(1.0)
    assert values.shape == (128, 5, 1)
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values[[0, -1], :, :])) == pytest.approx(0.0)
    assert np.max(np.abs(values)) <= 1.01
