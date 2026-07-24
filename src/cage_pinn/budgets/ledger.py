from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CostModel:
    residual_order_weights: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    backward_multiplier: float = 2.0
    weak_quadrature_weight: float = 1.0
    audit_backward_multiplier: float = 2.0

    def residual(self, points: int, derivative_order: int, *, backward: bool = False) -> float:
        if points < 0 or derivative_order < 0:
            raise ValueError("points and derivative_order must be non-negative")
        index = min(derivative_order, len(self.residual_order_weights) - 1)
        value = points * self.residual_order_weights[index]
        return value * (self.backward_multiplier if backward else 1.0)

    def weak(self, witnesses: int, quadrature_points: int, derivative_order: int) -> float:
        return (
            witnesses
            * quadrature_points
            * self.weak_quadrature_weight
            * self.residual_order_weights[min(derivative_order, len(self.residual_order_weights) - 1)]
        )


@dataclass
class BudgetLedger:
    total_tokens: int
    tolerance: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def spent_tokens(self) -> int:
        return sum(int(entry["tokens"]) for entry in self.entries)

    @property
    def remaining_tokens(self) -> int:
        return self.total_tokens - self.spent_tokens

    def charge(
        self,
        category: str,
        tokens: int,
        *,
        step: int,
        atom: str | None = None,
        measured_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if tokens < 0:
            raise ValueError("Cannot charge negative compute")
        if self.spent_tokens + tokens > self.total_tokens + self.tolerance:
            raise BudgetExceeded(
                f"Charge of {tokens} exceeds remaining budget {self.remaining_tokens}"
            )
        self.entries.append(
            {
                "category": category,
                "tokens": int(tokens),
                "step": int(step),
                "atom": atom,
                "measured_seconds": measured_seconds,
                "metadata": metadata or {},
            }
        )

    def totals_by_category(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for entry in self.entries:
            category = str(entry["category"])
            totals[category] = totals.get(category, 0) + int(entry["tokens"])
        return totals

    def totals_by_atom(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for entry in self.entries:
            if entry["atom"] is not None:
                atom = str(entry["atom"])
                totals[atom] = totals.get(atom, 0) + int(entry["tokens"])
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "spent_tokens": self.spent_tokens,
            "remaining_tokens": self.remaining_tokens,
            "tolerance": self.tolerance,
            "totals_by_category": self.totals_by_category(),
            "totals_by_atom": self.totals_by_atom(),
            "wall_seconds": time.perf_counter() - self.started_at,
            "entries": list(self.entries),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BudgetLedger:
        ledger = cls(total_tokens=int(value["total_tokens"]), tolerance=int(value["tolerance"]))
        ledger.entries = [dict(entry) for entry in value.get("entries", [])]
        return ledger


def serialize_cost_model(model: CostModel) -> dict[str, Any]:
    return asdict(model)

