"""Generate converged Burgers or Allen-Cahn references.

This is intentionally a standalone numerical experiment. Use at least two
resolutions, inspect the convergence diagnostics, then run the repository's
metadata validator. It does not run during package installation or smoke tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def burgers_rhs(viscosity: float, dx: float) -> Callable[[float, np.ndarray], np.ndarray]:
    def rhs(time: float, interior: np.ndarray) -> np.ndarray:
        del time
        full = np.pad(interior, (1, 1))
        ux = (full[2:] - full[:-2]) / (2.0 * dx)
        uxx = (full[2:] - 2.0 * full[1:-1] + full[:-2]) / dx**2
        return -interior * ux + viscosity * uxx

    return rhs


def allen_cahn_rhs(
    diffusivity: float, reaction: float, dx: float
) -> Callable[[float, np.ndarray], np.ndarray]:
    def rhs(time: float, values: np.ndarray) -> np.ndarray:
        del time
        uxx = (np.roll(values, -1) - 2.0 * values + np.roll(values, 1)) / dx**2
        return diffusivity * uxx - reaction * (values**3 - values)

    return rhs


def solve(problem: str, nx: int, nt: int, rtol: float, atol: float) -> tuple[np.ndarray, ...]:
    x = np.linspace(-1.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    dx = x[1] - x[0]
    if problem == "burgers_1d":
        initial = -np.sin(np.pi * x)
        result = solve_ivp(
            burgers_rhs(0.01 / np.pi, dx),
            (0.0, 1.0),
            initial[1:-1],
            method="BDF",
            t_eval=t,
            rtol=rtol,
            atol=atol,
        )
        if not result.success:
            raise RuntimeError(result.message)
        values = np.zeros((nx, nt, 1))
        values[1:-1, :, 0] = result.y
    elif problem == "allen_cahn":
        initial = x**2 * np.cos(np.pi * x)
        result = solve_ivp(
            allen_cahn_rhs(1.0e-4, 5.0, dx),
            (0.0, 1.0),
            initial,
            method="BDF",
            t_eval=t,
            rtol=rtol,
            atol=atol,
        )
        if not result.success:
            raise RuntimeError(result.message)
        values = result.y[:, :, None]
    else:
        raise KeyError(problem)
    return x, t, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", choices=("burgers_1d", "allen_cahn"))
    parser.add_argument("--resolutions", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--time-points", type=int, default=201)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--convergence-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--output-root", default="references/generated")
    args = parser.parse_args()
    resolutions = sorted(set(args.resolutions))
    if len(resolutions) < 2:
        raise ValueError("At least two distinct resolutions are required")
    output = Path(args.output_root) / args.problem
    output.mkdir(parents=True, exist_ok=True)
    solutions = []
    files = []
    for nx in resolutions:
        x, t, values = solve(args.problem, nx, args.time_points, args.rtol, args.atol)
        path = output / f"{args.problem}_nx{nx}_nt{args.time_points}.npz"
        np.savez_compressed(path, x=x, t=t, values=values)
        solutions.append((x, t, values))
        files.append({"file": path.name, "sha256": sha256(path), "nx": nx, "nt": len(t)})
    coarse_x, coarse_t, coarse_values = solutions[-2]
    fine_x, fine_t, fine_values = solutions[-1]
    interpolator = RegularGridInterpolator(
        (fine_x, fine_t), fine_values[..., 0], bounds_error=True
    )
    mesh = np.meshgrid(coarse_x, coarse_t, indexing="ij")
    query = np.stack((mesh[0].ravel(), mesh[1].ravel()), axis=1)
    fine_on_coarse = interpolator(query).reshape(coarse_values.shape[:-1])
    relative_difference = float(
        np.linalg.norm(fine_on_coarse - coarse_values[..., 0])
        / (np.linalg.norm(fine_on_coarse) + 1.0e-15)
    )
    if args.problem == "burgers_1d":
        boundary_error = float(
            max(np.max(np.abs(fine_values[0])), np.max(np.abs(fine_values[-1])))
        )
    else:
        boundary_error = float(
            np.max(np.abs(fine_values[0] - fine_values[-1]))
        )
    metadata = {
        "problem": args.problem,
        "primary_file": files[-1]["file"],
        "primary_sha256": files[-1]["sha256"],
        "files": files,
        "resolutions": resolutions,
        "solver": "scipy.solve_ivp BDF; second-order centered method of lines",
        "tolerances": {"rtol": args.rtol, "atol": args.atol},
        "convergence": {
            "relative_fine_coarse_difference": relative_difference,
            "tolerance": args.convergence_tolerance,
            "passed": relative_difference <= args.convergence_tolerance,
        },
        "diagnostics": {
            "boundary_or_periodicity": {
                "value": boundary_error,
                "tolerance": 1.0e-5,
                "passed": boundary_error <= 1.0e-5,
            },
            "finite_values": {"passed": bool(np.all(np.isfinite(fine_values)))},
        },
        "axes_order": ["x", "t"],
        "values_key": "values",
        "notes": "Reference generation is an experiment; inspect before approving metadata.",
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metadata": str(metadata_path), **metadata}, indent=2))
    if not metadata["convergence"]["passed"]:
        raise SystemExit("Convergence tolerance not met; increase resolutions")


if __name__ == "__main__":
    main()
