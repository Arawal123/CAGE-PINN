from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from jaxtyping import Array, PRNGKeyArray


@dataclass(frozen=True)
class ResidualChannel:
    name: str
    physical_unit: str
    nondimensional_scale: float | None
    derivative_order: int


@dataclass(frozen=True)
class BoundaryCondition:
    name: str
    kind: str
    component: int
    physical_unit: str


@runtime_checkable
class Geometry(Protocol):
    @property
    def dimension(self) -> int: ...

    def sample(self, key: PRNGKeyArray, n: int) -> Array: ...

    def clip(self, points: Array) -> Array: ...


@runtime_checkable
class ReferenceSolution(Protocol):
    def evaluate(self, points: Array) -> Array: ...


@runtime_checkable
class Sampler(Protocol):
    def sample(self, key: PRNGKeyArray, n: int, **context: Any) -> Array: ...


@runtime_checkable
class AuditRiskEstimator(Protocol):
    def __call__(self, normalized_residuals: Array, **options: Any) -> Any: ...


@runtime_checkable
class ComputeCostModel(Protocol):
    def residual(
        self, points: int, derivative_order: int, *, backward: bool = False
    ) -> float: ...


@runtime_checkable
class Backbone(Protocol):
    name: str

    def __call__(self, point: Array) -> Array: ...

    def interface_loss(self) -> Array: ...

    def audit_strata(self, points: Array) -> dict[str, Array]: ...


@runtime_checkable
class Metric(Protocol):
    name: str

    def __call__(self, prediction: Array, target: Array) -> Array: ...


@runtime_checkable
class Controller(Protocol):
    def allocate(
        self,
        utilities: dict[str, float],
        *,
        applicable: dict[str, bool],
        tokens: int,
    ) -> Any: ...


@runtime_checkable
class Trainer(Protocol):
    def __call__(self, config: Any) -> Any: ...


LossFunction = Callable[[Any], Array]

