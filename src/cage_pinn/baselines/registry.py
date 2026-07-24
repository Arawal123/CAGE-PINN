from __future__ import annotations

import importlib.util
from dataclasses import dataclass


class ExternalBaselineUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    tier: str
    implementation: str
    parent: str
    source_url: str
    license: str
    executable: bool
    notes: str
    required_module: str | None = None


def baseline_registry() -> dict[str, BaselineSpec]:
    specs = (
        BaselineSpec(
            "vanilla",
            "A",
            "native",
            "none",
            "https://doi.org/10.1016/j.jcp.2018.10.045",
            "clean-room MIT",
            True,
            "Standard strong-residual PINN.",
        ),
        BaselineSpec(
            "uniform",
            "A",
            "native",
            "vanilla",
            "https://doi.org/10.1016/j.jcp.2018.10.045",
            "clean-room MIT",
            True,
            "Fresh uniform collocation every step.",
        ),
        BaselineSpec(
            "rar_d",
            "A",
            "native",
            "vanilla",
            "https://doi.org/10.1016/j.cma.2022.115671",
            "clean-room MIT",
            True,
            "Residual-proportional candidate resampling; deviations require fidelity audit.",
        ),
        BaselineSpec(
            "gpinn",
            "A",
            "native",
            "vanilla",
            "https://arxiv.org/abs/2111.02801",
            "clean-room MIT",
            True,
            "Static strong plus residual-gradient schedule.",
        ),
        BaselineSpec(
            "static_sjw",
            "ablation",
            "native",
            "vanilla",
            "internal ablation",
            "MIT",
            True,
            "Matched atom bank with no audit controller.",
        ),
        BaselineSpec(
            "weak_pinn",
            "B",
            "native",
            "vanilla",
            "https://arxiv.org/abs/2605.03542",
            "clean-room MIT",
            True,
            "Generic randomized compact-support witness; not claimed paper-identical.",
        ),
        BaselineSpec(
            "relobralo",
            "A",
            "native-component",
            "vanilla",
            "https://doi.org/10.1016/j.softx.2024.101616",
            "clean-room MIT",
            True,
            "Executable clean-room diagnostic; full paper-fidelity study remains gated.",
        ),
        BaselineSpec(
            "sa_pinn",
            "A",
            "native-component",
            "vanilla",
            "https://doi.org/10.1016/j.jcp.2021.110768",
            "clean-room MIT",
            True,
            "Executable point-weight diagnostic; no publication comparison until fidelity check.",
        ),
        BaselineSpec(
            "config",
            "A",
            "native-component",
            "vanilla",
            "https://proceedings.iclr.cc/paper_files/paper/2025/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html",
            "clean-room MIT",
            True,
            "Executable two-gradient diagnostic; momentum schedule pending fidelity audit.",
        ),
        BaselineSpec(
            "ropinn",
            "A",
            "official-adapter",
            "vanilla",
            "https://github.com/thuml/RoPINN",
            "external repository license",
            False,
            "Pinned external environment required; never silently replaced.",
            "ropinn",
        ),
        BaselineSpec(
            "pinnacle_point_selection",
            "B",
            "official-adapter",
            "vanilla",
            "https://openreview.net/forum?id=G9Sw3QZBvL",
            "external repository license",
            False,
            "Official adapter required for confirmatory use.",
            "pinnacle",
        ),
        BaselineSpec(
            "ab_pinn_official",
            "C",
            "official-adapter",
            "ab_pinn",
            "https://github.com/merlresearch/ab-pinns",
            "external repository license",
            False,
            "PyTorch official repository is isolated from the JAX clean-room parent.",
            "ab_pinns",
        ),
    )
    return {spec.name: spec for spec in specs}


def require_external_baseline(name: str) -> BaselineSpec:
    try:
        spec = baseline_registry()[name]
    except KeyError as exc:
        raise KeyError(f"Unknown baseline {name!r}") from exc
    if spec.required_module is None:
        if not spec.executable:
            raise ExternalBaselineUnavailable(
                f"{name} is intentionally gated: {spec.notes}"
            )
        return spec
    if importlib.util.find_spec(spec.required_module) is None:
        raise ExternalBaselineUnavailable(
            f"{name} requires isolated official module {spec.required_module!r}. "
            f"Follow docs/baseline_fidelity.md; no fallback was used."
        )
    return spec
