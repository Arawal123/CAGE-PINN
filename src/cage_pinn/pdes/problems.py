from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from cage_pinn.geometry import Box
from cage_pinn.pdes.base import ModelFn, PDEProblem
from cage_pinn.pdes.extended import (
    Advection1D,
    AdvectionDiffusion1D,
    AllenCahn1D,
    Burgers1D,
    Helmholtz2D,
    HeterogeneousPoisson1D,
    HighFrequencyHelmholtz2D,
    KovasznayFlow,
    LidDrivenCavity,
    LidDrivenCavityRe1000,
    LidDrivenCavityRe3200,
    Poisson10D,
    TaylorGreenVortex,
    Wave1D,
    WeakSharpAdvection,
)


@dataclass(frozen=True)
class Poisson1D(PDEProblem):
    """-u_xx = pi^2 sin(pi x), u(0)=u(1)=0."""

    name: str = "poisson_1d"
    geometry: Box = field(default_factory=lambda: Box((0.0,), (1.0,)))
    output_dim: int = 1
    residual_channel_names: tuple[str, ...] = ("poisson",)
    residual_derivative_order: int = 2

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        def scalar(x: Array) -> Array:
            return model(jnp.asarray([x]))[0]

        u_xx = jax.grad(jax.grad(scalar))(z[0])
        forcing = jnp.pi**2 * jnp.sin(jnp.pi * z[0])
        return jnp.asarray([-u_xx - forcing])

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        del key, n
        values = jnp.stack((model(jnp.asarray([0.0]))[0], model(jnp.asarray([1.0]))[0]))
        return jnp.mean(values**2)

    def reference(self, points: Array) -> Array:
        return jnp.sin(jnp.pi * points[:, :1])


@dataclass(frozen=True)
class Heat1D(PDEProblem):
    """u_t-u_xx=0 on [0,1]x[0,1], u=0 at x boundaries."""

    name: str = "heat_1d"
    geometry: Box = field(default_factory=lambda: Box((0.0, 0.0), (1.0, 1.0)))
    output_dim: int = 1
    residual_channel_names: tuple[str, ...] = ("heat",)
    residual_derivative_order: int = 2

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        def scalar(point: Array) -> Array:
            return model(point)[0]

        gradient = jax.grad(scalar)(z)
        hessian = jax.hessian(scalar)(z)
        return jnp.asarray([gradient[1] - hessian[0, 0]])

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        kx, kt = jax.random.split(key)
        x = jax.random.uniform(kx, (n,))
        t = jax.random.uniform(kt, (n,))
        left = jax.vmap(lambda time: model(jnp.asarray([0.0, time]))[0])(t)
        right = jax.vmap(lambda time: model(jnp.asarray([1.0, time]))[0])(t)
        initial = jax.vmap(lambda pos: model(jnp.asarray([pos, 0.0]))[0])(x)
        target = jnp.sin(jnp.pi * x)
        return jnp.mean(left**2) + jnp.mean(right**2) + jnp.mean((initial - target) ** 2)

    def reference(self, points: Array) -> Array:
        x = points[:, 0:1]
        t = points[:, 1:2]
        return jnp.exp(-(jnp.pi**2) * t) * jnp.sin(jnp.pi * x)


@dataclass(frozen=True)
class CoupledOscillator(PDEProblem):
    """Two-channel first-order system y'=v, v'=-y."""

    name: str = "coupled_ode"
    geometry: Box = field(default_factory=lambda: Box((0.0,), (2.0 * math.pi,)))
    output_dim: int = 2
    residual_channel_names: tuple[str, ...] = ("position", "velocity")
    residual_derivative_order: int = 1

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        jacobian = jax.jacfwd(model)(z)
        value = model(z)
        return jnp.asarray([jacobian[0, 0] - value[1], jacobian[1, 0] + value[0]])

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        del key, n
        initial = model(jnp.asarray([0.0]))
        return jnp.mean((initial - jnp.asarray([0.0, 1.0])) ** 2)

    def reference(self, points: Array) -> Array:
        t = points[:, 0]
        return jnp.stack((jnp.sin(t), jnp.cos(t)), axis=1)


def problem_registry() -> dict[str, type[PDEProblem]]:
    return {
        "poisson_1d": Poisson1D,
        "heat_1d": Heat1D,
        "coupled_ode": CoupledOscillator,
        "advection_1d": Advection1D,
        "burgers_1d": Burgers1D,
        "allen_cahn": AllenCahn1D,
        "helmholtz_2d": Helmholtz2D,
        "high_frequency_helmholtz": HighFrequencyHelmholtz2D,
        "advection_diffusion": AdvectionDiffusion1D,
        "wave_1d": Wave1D,
        "heterogeneous_poisson": HeterogeneousPoisson1D,
        "poisson_10d": Poisson10D,
        "weak_sharp": WeakSharpAdvection,
        "kovasznay": KovasznayFlow,
        "taylor_green": TaylorGreenVortex,
        "cavity_re100": LidDrivenCavity,
        "cavity_re1000": LidDrivenCavityRe1000,
        "cavity_re3200": LidDrivenCavityRe3200,
    }


def create_problem(name: str) -> PDEProblem:
    registry = problem_registry()
    try:
        factory: Callable[[], PDEProblem] = registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown executable problem {name!r}; available: {available}") from exc
    return factory()
