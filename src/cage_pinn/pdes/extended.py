from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from cage_pinn.geometry import Box
from cage_pinn.pdes.base import ModelFn, PDEProblem


def _scalar_derivatives(model: ModelFn, z: Array) -> tuple[Array, Array, Array]:
    def scalar(point: Array) -> Array:
        return model(point)[0]

    return scalar(z), jax.grad(scalar)(z), jax.hessian(scalar)(z)


def _sample_rectangle_boundary(
    key: PRNGKeyArray, n: int, lower: tuple[float, float], upper: tuple[float, float]
) -> Array:
    side_key, coordinate_key = jax.random.split(key)
    side = jax.random.randint(side_key, (n,), 0, 4)
    s = jax.random.uniform(coordinate_key, (n,))
    x = lower[0] + s * (upper[0] - lower[0])
    y = lower[1] + s * (upper[1] - lower[1])
    x = jnp.where(side == 0, lower[0], jnp.where(side == 1, upper[0], x))
    y = jnp.where(side == 2, lower[1], jnp.where(side == 3, upper[1], y))
    return jnp.stack((x, y), axis=1)


@dataclass(frozen=True)
class Advection1D(PDEProblem):
    speed: float = 10.0
    name: str = "advection_1d"
    geometry: Box = field(default_factory=lambda: Box((0.0, 0.0), (1.0, 1.0)))
    output_dim: int = 1
    residual_channel_names: tuple[str, ...] = ("advection",)
    residual_derivative_order: int = 1

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        _, gradient, _ = _scalar_derivatives(model, z)
        return jnp.asarray([gradient[1] + self.speed * gradient[0]])

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        x_key, t_key = jax.random.split(key)
        x = jax.random.uniform(x_key, (n,))
        t = jax.random.uniform(t_key, (n,))
        initial = jax.vmap(lambda pos: model(jnp.asarray([pos, 0.0]))[0])(x)
        periodic = jax.vmap(
            lambda time: model(jnp.asarray([0.0, time]))[0]
            - model(jnp.asarray([1.0, time]))[0]
        )(t)
        return jnp.mean((initial - jnp.sin(2.0 * jnp.pi * x)) ** 2) + jnp.mean(
            periodic**2
        )

    def reference(self, points: Array) -> Array:
        phase = points[:, 0:1] - self.speed * points[:, 1:2]
        return jnp.sin(2.0 * jnp.pi * phase)


@dataclass(frozen=True)
class Helmholtz2D(PDEProblem):
    frequency: int = 4
    name: str = "helmholtz_2d"
    geometry: Box = field(default_factory=lambda: Box((0.0, 0.0), (1.0, 1.0)))
    output_dim: int = 1
    residual_channel_names: tuple[str, ...] = ("helmholtz",)
    residual_derivative_order: int = 2

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        value, _, hessian = _scalar_derivatives(model, z)
        wave_number_sq = 2.0 * (self.frequency * jnp.pi) ** 2
        return jnp.asarray([jnp.trace(hessian) + wave_number_sq * value])

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        points = _sample_rectangle_boundary(key, n, (0.0, 0.0), (1.0, 1.0))
        return jnp.mean(jax.vmap(lambda point: model(point)[0] ** 2)(points))

    def reference(self, points: Array) -> Array:
        k = self.frequency * jnp.pi
        return jnp.sin(k * points[:, 0:1]) * jnp.sin(k * points[:, 1:2])


@dataclass(frozen=True)
class HighFrequencyHelmholtz2D(Helmholtz2D):
    frequency: int = 10
    name: str = "high_frequency_helmholtz"


@dataclass(frozen=True)
class AdvectionDiffusion1D(PDEProblem):
    speed: float = 4.0
    diffusivity: float = 0.01
    name: str = "advection_diffusion"
    geometry: Box = field(default_factory=lambda: Box((0.0, 0.0), (1.0, 1.0)))
    output_dim: int = 1
    residual_channel_names: tuple[str, ...] = ("advection_diffusion",)
    residual_derivative_order: int = 2

    def _forcing(self, z: Array) -> Array:
        x, t = z
        value = jnp.exp(-t) * jnp.sin(jnp.pi * x)
        return (
            -value
            + self.speed * jnp.pi * jnp.exp(-t) * jnp.cos(jnp.pi * x)
            + self.diffusivity * jnp.pi**2 * value
        )

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        _, gradient, hessian = _scalar_derivatives(model, z)
        residual = (
            gradient[1]
            + self.speed * gradient[0]
            - self.diffusivity * hessian[0, 0]
            - self._forcing(z)
        )
        return jnp.asarray([residual])

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        x_key, t_key = jax.random.split(key)
        x = jax.random.uniform(x_key, (n,))
        t = jax.random.uniform(t_key, (n,))
        initial = jax.vmap(lambda pos: model(jnp.asarray([pos, 0.0]))[0])(x)
        left = jax.vmap(lambda time: model(jnp.asarray([0.0, time]))[0])(t)
        right = jax.vmap(lambda time: model(jnp.asarray([1.0, time]))[0])(t)
        return (
            jnp.mean((initial - jnp.sin(jnp.pi * x)) ** 2)
            + jnp.mean(left**2)
            + jnp.mean(right**2)
        )

    def reference(self, points: Array) -> Array:
        return jnp.exp(-points[:, 1:2]) * jnp.sin(jnp.pi * points[:, 0:1])


@dataclass(frozen=True)
class Wave1D(PDEProblem):
    wave_speed: float = 1.0
    name: str = "wave_1d"
    geometry: Box = field(default_factory=lambda: Box((0.0, 0.0), (1.0, 1.0)))
    output_dim: int = 1
    residual_channel_names: tuple[str, ...] = ("wave",)
    residual_derivative_order: int = 2

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        _, _, hessian = _scalar_derivatives(model, z)
        return jnp.asarray([hessian[1, 1] - self.wave_speed**2 * hessian[0, 0]])

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        x_key, t_key = jax.random.split(key)
        x = jax.random.uniform(x_key, (n,))
        t = jax.random.uniform(t_key, (n,))
        left = jax.vmap(lambda time: model(jnp.asarray([0.0, time]))[0])(t)
        right = jax.vmap(lambda time: model(jnp.asarray([1.0, time]))[0])(t)
        initial = jax.vmap(lambda pos: model(jnp.asarray([pos, 0.0]))[0])(x)

        def initial_velocity(pos: Array) -> Array:
            return jax.grad(lambda time: model(jnp.asarray([pos, time]))[0])(0.0)

        velocity = jax.vmap(initial_velocity)(x)
        return (
            jnp.mean(left**2)
            + jnp.mean(right**2)
            + jnp.mean((initial - jnp.sin(jnp.pi * x)) ** 2)
            + jnp.mean(velocity**2)
        )

    def reference(self, points: Array) -> Array:
        return jnp.sin(jnp.pi * points[:, 0:1]) * jnp.cos(
            self.wave_speed * jnp.pi * points[:, 1:2]
        )


@dataclass(frozen=True)
class HeterogeneousPoisson1D(PDEProblem):
    name: str = "heterogeneous_poisson"
    geometry: Box = field(default_factory=lambda: Box((0.0,), (1.0,)))
    output_dim: int = 1
    residual_channel_names: tuple[str, ...] = ("heterogeneous_poisson",)
    residual_derivative_order: int = 2

    @staticmethod
    def coefficient(x: Array) -> Array:
        return jnp.where(x < 0.5, 1.0, 2.0)

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        def flux(x: Array) -> Array:
            derivative = jax.grad(lambda position: model(jnp.asarray([position]))[0])(x)
            return self.coefficient(x) * derivative

        return jnp.asarray([-jax.grad(flux)(z[0]) - 1.0])

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        del key, n
        values = jnp.asarray([model(jnp.asarray([0.0]))[0], model(jnp.asarray([1.0]))[0]])
        eps = 1.0e-5
        left_flux = jax.grad(lambda x: model(jnp.asarray([x]))[0])(0.5 - eps)
        right_flux = 2.0 * jax.grad(lambda x: model(jnp.asarray([x]))[0])(0.5 + eps)
        return jnp.mean(values**2) + (left_flux - right_flux) ** 2

    def reference(self, points: Array) -> Array:
        x = points[:, 0:1]
        left = -0.5 * x**2 + (5.0 / 12.0) * x
        right = -0.25 * x**2 + (5.0 / 24.0) * x + 1.0 / 24.0
        return jnp.where(x < 0.5, left, right)


@dataclass(frozen=True)
class Poisson10D(PDEProblem):
    name: str = "poisson_10d"
    geometry: Box = field(
        default_factory=lambda: Box((0.0,) * 10, (1.0,) * 10)
    )
    output_dim: int = 1
    residual_channel_names: tuple[str, ...] = ("poisson_10d",)
    residual_derivative_order: int = 2

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        value, _, hessian = _scalar_derivatives(model, z)
        return jnp.asarray(
            [-jnp.trace(hessian) - self.geometry.dimension * jnp.pi**2 * value]
        )

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        point_key, dimension_key, side_key = jax.random.split(key, 3)
        points = self.geometry.sample(point_key, n)
        dimensions = jax.random.randint(
            dimension_key, (n,), 0, self.geometry.dimension
        )
        sides = jax.random.bernoulli(side_key, shape=(n,)).astype(points.dtype)
        points = points.at[jnp.arange(n), dimensions].set(sides)
        return jnp.mean(jax.vmap(lambda point: model(point)[0] ** 2)(points))

    def reference(self, points: Array) -> Array:
        return jnp.prod(jnp.sin(jnp.pi * points), axis=1, keepdims=True)


@dataclass(frozen=True)
class Burgers1D(PDEProblem):
    viscosity: float = 0.01 / math.pi
    name: str = "burgers_1d"
    geometry: Box = field(default_factory=lambda: Box((-1.0, 0.0), (1.0, 1.0)))
    output_dim: int = 1
    residual_channel_names: tuple[str, ...] = ("burgers",)
    residual_derivative_order: int = 2
    reference_available: bool = False
    reference_kind: str = "external grid-converged"

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        value, gradient, hessian = _scalar_derivatives(model, z)
        return jnp.asarray(
            [gradient[1] + value * gradient[0] - self.viscosity * hessian[0, 0]]
        )

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        x_key, t_key = jax.random.split(key)
        x = jax.random.uniform(x_key, (n,), minval=-1.0, maxval=1.0)
        t = jax.random.uniform(t_key, (n,))
        initial = jax.vmap(lambda pos: model(jnp.asarray([pos, 0.0]))[0])(x)
        left = jax.vmap(lambda time: model(jnp.asarray([-1.0, time]))[0])(t)
        right = jax.vmap(lambda time: model(jnp.asarray([1.0, time]))[0])(t)
        return (
            jnp.mean((initial + jnp.sin(jnp.pi * x)) ** 2)
            + jnp.mean(left**2)
            + jnp.mean(right**2)
        )

    def reference(self, points: Array) -> Array:
        del points
        raise RuntimeError("Burgers reference must be generated and checksum-validated")


@dataclass(frozen=True)
class AllenCahn1D(PDEProblem):
    diffusivity: float = 1.0e-4
    reaction: float = 5.0
    name: str = "allen_cahn"
    geometry: Box = field(default_factory=lambda: Box((-1.0, 0.0), (1.0, 1.0)))
    output_dim: int = 1
    residual_channel_names: tuple[str, ...] = ("allen_cahn",)
    residual_derivative_order: int = 2
    reference_available: bool = False
    reference_kind: str = "external grid-converged"

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        value, gradient, hessian = _scalar_derivatives(model, z)
        return jnp.asarray(
            [
                gradient[1]
                - self.diffusivity * hessian[0, 0]
                + self.reaction * (value**3 - value)
            ]
        )

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        x_key, t_key = jax.random.split(key)
        x = jax.random.uniform(x_key, (n,), minval=-1.0, maxval=1.0)
        t = jax.random.uniform(t_key, (n,))
        target = x**2 * jnp.cos(jnp.pi * x)
        initial = jax.vmap(lambda pos: model(jnp.asarray([pos, 0.0]))[0])(x)
        periodic_value = jax.vmap(
            lambda time: model(jnp.asarray([-1.0, time]))[0]
            - model(jnp.asarray([1.0, time]))[0]
        )(t)

        def periodic_gradient(time: Array) -> Array:
            left = jax.grad(lambda pos: model(jnp.asarray([pos, time]))[0])(-1.0)
            right = jax.grad(lambda pos: model(jnp.asarray([pos, time]))[0])(1.0)
            return left - right

        return (
            jnp.mean((initial - target) ** 2)
            + jnp.mean(periodic_value**2)
            + jnp.mean(jax.vmap(periodic_gradient)(t) ** 2)
        )

    def reference(self, points: Array) -> Array:
        del points
        raise RuntimeError("Allen-Cahn reference must be generated and checksum-validated")


@dataclass(frozen=True)
class WeakSharpAdvection(Advection1D):
    speed: float = 1.0
    name: str = "weak_sharp"
    smooth: bool = False

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        x_key, t_key = jax.random.split(key)
        x = jax.random.uniform(x_key, (n,))
        t = jax.random.uniform(t_key, (n,))
        target = jnp.where(x < 0.5, 1.0, -1.0)
        initial = jax.vmap(lambda pos: model(jnp.asarray([pos, 0.0]))[0])(x)
        periodic = jax.vmap(
            lambda time: model(jnp.asarray([0.0, time]))[0]
            - model(jnp.asarray([1.0, time]))[0]
        )(t)
        return jnp.mean((initial - target) ** 2) + jnp.mean(periodic**2)

    def reference(self, points: Array) -> Array:
        phase = jnp.mod(points[:, 0:1] - self.speed * points[:, 1:2], 1.0)
        return jnp.where(phase < 0.5, 1.0, -1.0)


def _steady_navier_stokes_residual(
    model: ModelFn, z: Array, viscosity: float
) -> Array:
    value = model(z)
    jacobian = jax.jacfwd(model)(z)
    hessians = jax.jacfwd(jax.jacrev(model))(z)
    u, v = value[0], value[1]
    momentum_x = (
        u * jacobian[0, 0]
        + v * jacobian[0, 1]
        + jacobian[2, 0]
        - viscosity * (hessians[0, 0, 0] + hessians[0, 1, 1])
    )
    momentum_y = (
        u * jacobian[1, 0]
        + v * jacobian[1, 1]
        + jacobian[2, 1]
        - viscosity * (hessians[1, 0, 0] + hessians[1, 1, 1])
    )
    continuity = jacobian[0, 0] + jacobian[1, 1]
    return jnp.asarray([momentum_x, momentum_y, continuity])


@dataclass(frozen=True)
class KovasznayFlow(PDEProblem):
    reynolds: float = 40.0
    name: str = "kovasznay"
    geometry: Box = field(default_factory=lambda: Box((-0.5, -0.5), (1.0, 1.5)))
    output_dim: int = 3
    residual_channel_names: tuple[str, ...] = ("momentum_x", "momentum_y", "continuity")
    residual_derivative_order: int = 2

    @property
    def lam(self) -> float:
        return self.reynolds / 2.0 - math.sqrt(self.reynolds**2 / 4.0 + 4.0 * math.pi**2)

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        return _steady_navier_stokes_residual(model, z, 1.0 / self.reynolds)

    def _solution(self, points: Array) -> Array:
        x, y = points[:, 0], points[:, 1]
        exponential = jnp.exp(self.lam * x)
        u = 1.0 - exponential * jnp.cos(2.0 * jnp.pi * y)
        v = self.lam / (2.0 * jnp.pi) * exponential * jnp.sin(2.0 * jnp.pi * y)
        p = 0.5 * (1.0 - jnp.exp(2.0 * self.lam * x))
        return jnp.stack((u, v, p), axis=1)

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        points = _sample_rectangle_boundary(key, n, (-0.5, -0.5), (1.0, 1.5))
        target = self._solution(points)
        prediction = jax.vmap(model)(points)
        return jnp.mean((prediction - target) ** 2)

    def reference(self, points: Array) -> Array:
        return self._solution(points)


@dataclass(frozen=True)
class TaylorGreenVortex(PDEProblem):
    reynolds: float = 100.0
    name: str = "taylor_green"
    geometry: Box = field(
        default_factory=lambda: Box((0.0, 0.0, 0.0), (2.0 * math.pi, 2.0 * math.pi, 1.0))
    )
    output_dim: int = 3
    residual_channel_names: tuple[str, ...] = ("momentum_x", "momentum_y", "continuity")
    residual_derivative_order: int = 2

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        value = model(z)
        jacobian = jax.jacfwd(model)(z)
        hessians = jax.jacfwd(jax.jacrev(model))(z)
        u, v = value[0], value[1]
        viscosity = 1.0 / self.reynolds
        momentum_x = (
            jacobian[0, 2]
            + u * jacobian[0, 0]
            + v * jacobian[0, 1]
            + jacobian[2, 0]
            - viscosity * (hessians[0, 0, 0] + hessians[0, 1, 1])
        )
        momentum_y = (
            jacobian[1, 2]
            + u * jacobian[1, 0]
            + v * jacobian[1, 1]
            + jacobian[2, 1]
            - viscosity * (hessians[1, 0, 0] + hessians[1, 1, 1])
        )
        return jnp.asarray([momentum_x, momentum_y, jacobian[0, 0] + jacobian[1, 1]])

    def _solution(self, points: Array) -> Array:
        x, y, t = points[:, 0], points[:, 1], points[:, 2]
        viscosity = 1.0 / self.reynolds
        velocity_decay = jnp.exp(-2.0 * viscosity * t)
        pressure_decay = jnp.exp(-4.0 * viscosity * t)
        u = -jnp.cos(x) * jnp.sin(y) * velocity_decay
        v = jnp.sin(x) * jnp.cos(y) * velocity_decay
        p = -0.25 * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y)) * pressure_decay
        return jnp.stack((u, v, p), axis=1)

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        x_key, y_key, t_key = jax.random.split(key, 3)
        x = jax.random.uniform(x_key, (n,), maxval=2.0 * jnp.pi)
        y = jax.random.uniform(y_key, (n,), maxval=2.0 * jnp.pi)
        t = jax.random.uniform(t_key, (n,))
        initial_points = jnp.stack((x, y, jnp.zeros_like(x)), axis=1)
        initial = jnp.mean((jax.vmap(model)(initial_points) - self._solution(initial_points)) ** 2)
        periodic_x = jax.vmap(
            lambda yy, tt: model(jnp.asarray([0.0, yy, tt]))
            - model(jnp.asarray([2.0 * jnp.pi, yy, tt]))
        )(y, t)
        periodic_y = jax.vmap(
            lambda xx, tt: model(jnp.asarray([xx, 0.0, tt]))
            - model(jnp.asarray([xx, 2.0 * jnp.pi, tt]))
        )(x, t)
        return initial + jnp.mean(periodic_x**2) + jnp.mean(periodic_y**2)

    def reference(self, points: Array) -> Array:
        return self._solution(points)


@dataclass(frozen=True)
class LidDrivenCavity(PDEProblem):
    reynolds: float = 100.0
    name: str = "cavity_re100"
    geometry: Box = field(default_factory=lambda: Box((0.0, 0.0), (1.0, 1.0)))
    output_dim: int = 3
    residual_channel_names: tuple[str, ...] = ("momentum_x", "momentum_y", "continuity")
    residual_derivative_order: int = 2
    reference_available: bool = False
    reference_kind: str = "external grid-converged CFD"

    def residual_point(self, model: ModelFn, z: Array) -> Array:
        return _steady_navier_stokes_residual(model, z, 1.0 / self.reynolds)

    def boundary_loss(self, model: ModelFn, key: PRNGKeyArray, n: int) -> Array:
        points = _sample_rectangle_boundary(key, n, (0.0, 0.0), (1.0, 1.0))
        prediction = jax.vmap(model)(points)
        target_u = jnp.where(jnp.isclose(points[:, 1], 1.0), 1.0, 0.0)
        velocity_target = jnp.stack((target_u, jnp.zeros_like(target_u)), axis=1)
        return jnp.mean((prediction[:, :2] - velocity_target) ** 2) + prediction[0, 2] ** 2

    def reference(self, points: Array) -> Array:
        del points
        raise RuntimeError("Cavity reference must be generated and checksum-validated")


@dataclass(frozen=True)
class LidDrivenCavityRe1000(LidDrivenCavity):
    reynolds: float = 1000.0
    name: str = "cavity_re1000"


@dataclass(frozen=True)
class LidDrivenCavityRe3200(LidDrivenCavity):
    reynolds: float = 3200.0
    name: str = "cavity_re3200"

