import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cage_pinn.references import load_external_reference, validate_external_metadata


def test_external_reference_checksum_schema_and_interpolation(tmp_path: Path) -> None:
    x = np.linspace(0.0, 1.0, 5)
    t = np.linspace(0.0, 1.0, 4)
    mesh_x, mesh_t = np.meshgrid(x, t, indexing="ij")
    values = (mesh_x + 2.0 * mesh_t)[..., None]
    data_path = tmp_path / "reference.npz"
    np.savez_compressed(data_path, x=x, t=t, values=values)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    metadata = {
        "problem": "diagnostic",
        "primary_file": data_path.name,
        "primary_sha256": digest,
        "resolutions": [3, 5],
        "solver": "analytic test fixture",
        "tolerances": {"rtol": 0.0, "atol": 0.0},
        "convergence": {"passed": True},
        "diagnostics": {"finite": {"passed": True}},
        "axes_order": ["x", "t"],
        "values_key": "values",
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert validate_external_metadata(metadata_path)["passed"]
    reference = load_external_reference(metadata_path)
    prediction = reference.evaluate(jnp.asarray([[0.25, 0.5]]))
    assert jnp.allclose(prediction, jnp.asarray([[1.25]]))

