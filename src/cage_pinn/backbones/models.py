from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class MLP(eqx.Module):
    network: eqx.nn.MLP

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
    ) -> None:
        self.network = eqx.nn.MLP(
            in_size=input_dim,
            out_size=output_dim,
            width_size=width,
            depth=depth,
            activation=jnp.tanh,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, z: Array) -> Array:
        return self.network(z)


class VanillaBackbone(eqx.Module):
    model: MLP
    name: str = eqx.field(static=True, default="vanilla")

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
    ) -> None:
        self.model = MLP(input_dim, output_dim, width, depth, key)

    def __call__(self, z: Array) -> Array:
        return self.model(z)

    def interface_loss(self) -> Array:
        return jnp.asarray(0.0)

    def audit_strata(self, points: Array) -> dict[str, Array]:
        return {"global": points}


class XPINNBackbone(eqx.Module):
    models: tuple[MLP, ...]
    cutpoints: tuple[float, ...] = eqx.field(static=True)
    name: str = eqx.field(static=True, default="xpinn")

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        *,
        lower: float,
        upper: float,
        subdomains: int = 2,
    ) -> None:
        if subdomains < 2:
            raise ValueError("XPINN requires at least two subdomains")
        keys = jax.random.split(key, subdomains)
        self.models = tuple(
            MLP(input_dim, output_dim, width, depth, subkey) for subkey in keys
        )
        self.cutpoints = tuple(
            float(value)
            for value in jnp.linspace(lower, upper, subdomains + 1)[1:-1].tolist()
        )

    def subdomain_index(self, z: Array) -> Array:
        return jnp.searchsorted(jnp.asarray(self.cutpoints), z[0], side="right")

    def value_in(self, index: int, z: Array) -> Array:
        return self.models[index](z)

    def __call__(self, z: Array) -> Array:
        branches = tuple(
            (lambda point, local_model=local_model: local_model(point))
            for local_model in self.models
        )
        return jax.lax.switch(self.subdomain_index(z), branches, z)

    def interface_loss(self) -> Array:
        losses = []
        input_dim = self.models[0].network.in_size
        for index, cut in enumerate(self.cutpoints):
            z = jnp.zeros((input_dim,)).at[0].set(cut)
            left_model = self.models[index]
            right_model = self.models[index + 1]
            value_jump = left_model(z) - right_model(z)
            left_flux = jax.jacfwd(left_model)(z)[:, 0]
            right_flux = jax.jacfwd(right_model)(z)[:, 0]
            losses.append(jnp.mean(value_jump**2) + jnp.mean((left_flux - right_flux) ** 2))
        return jnp.mean(jnp.stack(losses))

    def audit_strata(self, points: Array) -> dict[str, Array]:
        strata: dict[str, Array] = {"global": points}
        indices = jax.vmap(self.subdomain_index)(points)
        for index in range(len(self.models)):
            strata[f"subdomain_{index}"] = points[indices == index]
        return strata


class ABPINNBackbone(eqx.Module):
    models: tuple[MLP, ...]
    centers: Array
    log_widths: Array
    name: str = eqx.field(static=True, default="ab_pinn")
    addition_schedule: tuple[int, ...] = eqx.field(static=True, default=())

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        *,
        lower: float,
        upper: float,
        windows: int = 3,
        addition_schedule: tuple[int, ...] = (),
    ) -> None:
        if windows < 2:
            raise ValueError("AB-PINN requires at least two local windows")
        keys = jax.random.split(key, windows)
        self.models = tuple(
            MLP(input_dim, output_dim, width, depth, subkey) for subkey in keys
        )
        self.centers = jnp.linspace(lower, upper, windows)
        initial_width = 1.5 * (upper - lower) / max(windows - 1, 1)
        self.log_widths = jnp.full((windows,), jnp.log(initial_width))
        self.addition_schedule = addition_schedule

    def window_weights(self, z: Array) -> Array:
        widths = jax.nn.softplus(self.log_widths) + 1.0e-4
        raw = jnp.exp(-0.5 * ((z[0] - self.centers) / widths) ** 2)
        return raw / (jnp.sum(raw) + 1.0e-12)

    def __call__(self, z: Array) -> Array:
        local = jnp.stack(tuple(model(z) for model in self.models))
        return jnp.tensordot(self.window_weights(z), local, axes=1)

    def interface_loss(self) -> Array:
        return jnp.asarray(0.0)

    def audit_strata(self, points: Array) -> dict[str, Array]:
        weights = jax.vmap(self.window_weights)(points)
        assignment = jnp.argmax(weights, axis=1)
        strata: dict[str, Array] = {"global": points}
        for index in range(len(self.models)):
            strata[f"window_{index}"] = points[assignment == index]
        return strata


Backbone = VanillaBackbone | XPINNBackbone | ABPINNBackbone


def parameter_count(model: Any) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return int(sum(leaf.size for leaf in leaves))


def create_backbone(
    name: str,
    *,
    input_dim: int,
    output_dim: int,
    width: int,
    depth: int,
    key: PRNGKeyArray,
    lower: float,
    upper: float,
) -> Backbone:
    factories: dict[str, Callable[[], Backbone]] = {
        "vanilla": lambda: VanillaBackbone(input_dim, output_dim, width, depth, key),
        "xpinn": lambda: XPINNBackbone(
            input_dim,
            output_dim,
            max(4, int(width / jnp.sqrt(2.0))),
            depth,
            key,
            lower=lower,
            upper=upper,
            subdomains=2,
        ),
        "ab_pinn": lambda: ABPINNBackbone(
            input_dim,
            output_dim,
            max(4, int(width / jnp.sqrt(3.0))),
            depth,
            key,
            lower=lower,
            upper=upper,
            windows=3,
        ),
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown backbone {name!r}; choices: {sorted(factories)}") from exc

