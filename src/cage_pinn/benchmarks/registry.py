from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    stage: str
    family: str
    executable: bool
    reference_kind: str
    failure_mode: str
    reference_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_registry() -> dict[str, BenchmarkSpec]:
    specs = (
        BenchmarkSpec("poisson_1d", "smoke", "elliptic", True, "analytic", "coverage", "verified"),
        BenchmarkSpec("heat_1d", "smoke", "parabolic", True, "analytic", "space-time", "verified"),
        BenchmarkSpec("coupled_ode", "smoke", "system", True, "analytic", "multi-channel", "verified"),
        BenchmarkSpec("advection_1d", "discovery", "hyperbolic", True, "analytic", "propagation", "verified"),
        BenchmarkSpec("burgers_1d", "discovery", "nonlinear", True, "spectral/grid", "sharp gradient", "generator required"),
        BenchmarkSpec("allen_cahn", "discovery", "reaction-diffusion", True, "spectral/grid", "stiffness", "generator required"),
        BenchmarkSpec("helmholtz_2d", "discovery", "elliptic", True, "manufactured", "frequency bias", "verified"),
        BenchmarkSpec("kovasznay", "discovery", "incompressible flow", True, "analytic", "coupled channels", "verified"),
        BenchmarkSpec("weak_sharp", "discovery", "conservation/weak", True, "analytic discontinuous transport", "weak witness", "verified away from jump"),
        BenchmarkSpec("heterogeneous_poisson", "confirmatory", "elliptic/interface", True, "analytic interface", "interfaces", "verified away from interface"),
        BenchmarkSpec("high_frequency_helmholtz", "confirmatory", "elliptic", True, "manufactured", "frequency bias", "verified"),
        BenchmarkSpec("advection_diffusion", "confirmatory", "transport", True, "manufactured", "boundary layer", "verified"),
        BenchmarkSpec("wave_1d", "confirmatory", "hyperbolic", True, "analytic", "long-time propagation", "verified"),
        BenchmarkSpec("taylor_green", "confirmatory", "unsteady flow", True, "analytic", "vector/divergence", "verified"),
        BenchmarkSpec("cavity_re100", "confirmatory", "incompressible flow", True, "grid-converged CFD", "recirculation", "blocked on reference"),
        BenchmarkSpec("cavity_re1000", "confirmatory", "incompressible flow", True, "grid-converged CFD", "thin layers", "blocked on reference"),
        BenchmarkSpec("cavity_re3200", "confirmatory-extension", "incompressible flow", True, "grid-converged CFD", "stability", "optional"),
        BenchmarkSpec("poisson_10d", "confirmatory", "high-dimensional", True, "analytic", "dimension", "verified"),
    )
    return {spec.name: spec for spec in specs}
