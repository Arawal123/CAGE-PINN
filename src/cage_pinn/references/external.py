from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExternalGridReference:
    problem: str
    axes: tuple[np.ndarray, ...]
    values: np.ndarray
    metadata: dict[str, Any]

    def evaluate(self, points: jnp.ndarray) -> jnp.ndarray:
        query = np.asarray(points, dtype=np.float64)
        channels = self.values.shape[-1]
        output = []
        for channel in range(channels):
            interpolator = RegularGridInterpolator(
                self.axes,
                self.values[..., channel],
                method="linear",
                bounds_error=True,
            )
            output.append(interpolator(query))
        return jnp.asarray(np.stack(output, axis=1))


def validate_external_metadata(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "problem",
        "primary_file",
        "primary_sha256",
        "resolutions",
        "solver",
        "tolerances",
        "convergence",
        "diagnostics",
        "axes_order",
        "values_key",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Reference metadata missing fields: {missing}")
    if len(metadata["resolutions"]) < 2:
        raise ValueError("Reference validation requires at least two resolutions")
    primary = (path.parent / metadata["primary_file"]).resolve()
    if not primary.exists():
        raise FileNotFoundError(primary)
    actual_hash = file_sha256(primary)
    if actual_hash != metadata["primary_sha256"]:
        raise ValueError("Reference checksum mismatch")
    convergence = metadata["convergence"]
    if not bool(convergence.get("passed", False)):
        raise ValueError("Reference convergence check is not marked passed")
    diagnostics = metadata["diagnostics"]
    failed = [name for name, value in diagnostics.items() if not bool(value.get("passed", False))]
    if failed:
        raise ValueError(f"Reference diagnostics failed: {failed}")
    with np.load(primary) as data:
        for axis in metadata["axes_order"]:
            if axis not in data:
                raise ValueError(f"Reference file missing axis {axis!r}")
        values = np.asarray(data[metadata["values_key"]])
        expected = tuple(len(data[axis]) for axis in metadata["axes_order"])
        if values.shape[:-1] != expected:
            raise ValueError(
                f"Reference values shape {values.shape} does not match axes {expected}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Reference contains non-finite values")
    return {
        "problem": metadata["problem"],
        "metadata": str(path),
        "primary": str(primary),
        "sha256": actual_hash,
        "resolutions": metadata["resolutions"],
        "convergence": convergence,
        "diagnostics": diagnostics,
        "passed": True,
    }


def load_external_reference(metadata_path: str | Path) -> ExternalGridReference:
    validation = validate_external_metadata(metadata_path)
    path = Path(metadata_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    primary = path.parent / metadata["primary_file"]
    with np.load(primary) as data:
        axes = tuple(
            np.asarray(data[name], dtype=np.float64) for name in metadata["axes_order"]
        )
        values = np.asarray(data[metadata["values_key"]], dtype=np.float64)
    return ExternalGridReference(
        problem=str(validation["problem"]),
        axes=axes,
        values=values,
        metadata=metadata,
    )


def find_external_reference(
    problem: str, root: str | Path = "references/generated"
) -> ExternalGridReference | None:
    metadata = Path(root) / problem / "metadata.json"
    if not metadata.exists():
        return None
    reference = load_external_reference(metadata)
    if reference.problem != problem:
        raise ValueError(
            f"Reference problem mismatch: expected {problem}, got {reference.problem}"
        )
    return reference

