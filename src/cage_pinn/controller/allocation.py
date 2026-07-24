from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def project_bounded_simplex(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    if values.ndim != 1 or lower.shape != values.shape or upper.shape != values.shape:
        raise ValueError("Bounded simplex inputs must be equal-sized vectors")
    if np.any(lower < 0) or np.any(upper > 1) or np.any(lower > upper):
        raise ValueError("Invalid simplex bounds")
    if lower.sum() > 1.0 + 1.0e-12 or upper.sum() < 1.0 - 1.0e-12:
        raise ValueError("Infeasible simplex bounds")
    lo = float(np.min(values - upper)) - 1.0
    hi = float(np.max(values - lower)) + 1.0
    for _ in range(100):
        midpoint = 0.5 * (lo + hi)
        projected = np.clip(values - midpoint, lower, upper)
        if projected.sum() > 1.0:
            lo = midpoint
        else:
            hi = midpoint
    result = np.clip(values - 0.5 * (lo + hi), lower, upper)
    result /= result.sum()
    return result


@dataclass(frozen=True)
class AllocationResult:
    requested: dict[str, float]
    realized_tokens: dict[str, int]
    realized: dict[str, float]
    utilities: dict[str, float]
    fallback: bool
    reason: str


class BudgetAllocator:
    def __init__(
        self,
        atom_names: tuple[str, ...] = ("S", "J", "W"),
        *,
        floors: dict[str, float] | None = None,
        caps: dict[str, float] | None = None,
        entropy: float = 0.15,
        turnover_penalty: float = 0.1,
        max_change: float = 0.25,
    ) -> None:
        self.atom_names = atom_names
        self.floors = floors or {"S": 0.10, "J": 0.05, "W": 0.05}
        self.caps = caps or {"S": 0.90, "J": 0.45, "W": 0.45}
        self.entropy = entropy
        self.turnover_penalty = turnover_penalty
        self.max_change = max_change
        initial = np.ones(len(atom_names), dtype=float) / len(atom_names)
        self.previous = project_bounded_simplex(
            initial,
            np.asarray([self.floors[name] for name in atom_names]),
            np.asarray([self.caps[name] for name in atom_names]),
        )

    def allocate(
        self,
        utilities: dict[str, float],
        *,
        applicable: dict[str, bool],
        tokens: int,
    ) -> AllocationResult:
        if tokens <= 0:
            raise ValueError("Allocation tokens must be positive")
        names = self.atom_names
        mask = np.asarray([bool(applicable.get(name, False)) for name in names])
        if not np.any(mask):
            raise ValueError("At least one enforcement atom must be applicable")
        lower = np.asarray([self.floors[name] if mask[i] else 0.0 for i, name in enumerate(names)])
        upper = np.asarray([self.caps[name] if mask[i] else 0.0 for i, name in enumerate(names)])
        if lower.sum() > 1.0 or upper.sum() < 1.0:
            only = np.flatnonzero(mask)
            lower = np.zeros_like(lower)
            upper = np.zeros_like(upper)
            upper[only] = 1.0
            lower[only] = min(0.02, 1.0 / len(only))
        rate_lower = np.maximum(lower, self.previous - self.max_change)
        rate_upper = np.minimum(upper, self.previous + self.max_change)
        if rate_lower.sum() > 1.0 or rate_upper.sum() < 1.0:
            rate_lower, rate_upper = lower, upper
        values = np.asarray([max(0.0, float(utilities.get(name, 0.0))) for name in names])
        fallback = not np.any(values > 0)
        reason = "all utilities non-positive; exploration/previous allocation used" if fallback else ""
        allocation = project_bounded_simplex(self.previous, rate_lower, rate_upper)
        if not fallback:
            for _ in range(80):
                gradient = (
                    values
                    - 2.0 * self.turnover_penalty * (allocation - self.previous)
                    - self.entropy * (np.log(np.maximum(allocation, 1.0e-12)) + 1.0)
                )
                allocation = project_bounded_simplex(
                    allocation + 0.05 * gradient, rate_lower, rate_upper
                )
        realized_tokens = self._largest_remainder(allocation, tokens)
        realized = np.asarray(realized_tokens, dtype=float) / tokens
        self.previous = allocation
        return AllocationResult(
            requested=dict(zip(names, allocation.tolist(), strict=True)),
            realized_tokens=dict(zip(names, realized_tokens, strict=True)),
            realized=dict(zip(names, realized.tolist(), strict=True)),
            utilities=dict(zip(names, values.tolist(), strict=True)),
            fallback=fallback,
            reason=reason,
        )

    @staticmethod
    def _largest_remainder(shares: np.ndarray, tokens: int) -> list[int]:
        exact = shares * tokens
        base = np.floor(exact).astype(int)
        remaining = tokens - int(base.sum())
        order = np.argsort(-(exact - base), kind="stable")
        for index in order[:remaining]:
            base[index] += 1
        if int(base.sum()) != tokens:
            raise AssertionError("Token rounding drifted from exact budget")
        return base.tolist()

