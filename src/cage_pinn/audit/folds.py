from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from cage_pinn.geometry import Box


def coordinate_hash(point: Array) -> str:
    normalized = np.asarray(point, dtype=np.float64)
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def batch_hash(points: Array) -> str:
    hashes = sorted(coordinate_hash(point) for point in np.asarray(points))
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


@dataclass(frozen=True)
class FoldRoles:
    selector: int
    monitor: int
    calibration: int


@dataclass(frozen=True)
class LeakageReport:
    passed: bool
    learner_unique: bool
    audit_unique: bool
    learner_audit_overlap: tuple[str, ...]
    audit_fold_overlap: tuple[str, ...]
    prohibited_window: int


class AuditFoldManager:
    """Owns cross-fitted coordinates and makes role transitions explicit."""

    def __init__(
        self,
        geometry: Box,
        key: PRNGKeyArray,
        *,
        fold_size: int,
        folds: int = 3,
        rotate_interval: int = 2,
        refresh_after_selections: int = 3,
        prohibited_window: int = 2,
    ) -> None:
        if folds < 3:
            raise ValueError("CAGE requires at least selector, monitor, and calibration folds")
        if fold_size <= 0 or rotate_interval <= 0 or refresh_after_selections <= 0:
            raise ValueError("Fold sizes and intervals must be positive")
        self.geometry = geometry
        self.fold_size = fold_size
        self.folds_count = folds
        self.rotate_interval = rotate_interval
        self.refresh_after_selections = refresh_after_selections
        self.prohibited_window = prohibited_window
        self._master_key = key
        keys = jax.random.split(key, folds + 1)
        self._master_key = keys[0]
        self._folds = [geometry.sample(keys[index + 1], fold_size) for index in range(folds)]
        self._generations = [0 for _ in range(folds)]
        self._selection_counts = [0 for _ in range(folds)]
        self._role_offset = 0
        self._last_rotation_control = 0
        self._learner_history: deque[set[str]] = deque(maxlen=max(1, prohibited_window))
        self._provenance: list[dict[str, Any]] = []
        for fold_id, points in enumerate(self._folds):
            self._record_points(points, source="audit", fold_id=fold_id, generation=0)
        self.assert_no_leakage()

    @property
    def roles(self) -> FoldRoles:
        return FoldRoles(
            selector=self._role_offset % self.folds_count,
            monitor=(self._role_offset + 1) % self.folds_count,
            calibration=(self._role_offset + 2) % self.folds_count,
        )

    def points(self, role: str) -> Array:
        roles = self.roles
        try:
            fold_id = getattr(roles, role)
        except AttributeError as exc:
            raise KeyError(f"Unknown audit role {role!r}") from exc
        return self._folds[fold_id]

    def register_learner(self, points: Array, *, step: int) -> None:
        hashes = {coordinate_hash(point) for point in np.asarray(points)}
        if len(hashes) != len(points):
            raise ValueError("Learner batch contains duplicate coordinates")
        self._learner_history.append(hashes)
        self._record_points(points, source="learner", step=step)
        self.assert_no_leakage()

    def on_control(self, control_index: int) -> None:
        selector = self.roles.selector
        self._selection_counts[selector] += 1
        if self._selection_counts[selector] >= self.refresh_after_selections:
            self.refresh_fold(selector)
        if control_index - self._last_rotation_control >= self.rotate_interval:
            self._role_offset = (self._role_offset + 1) % self.folds_count
            self._last_rotation_control = control_index

    def refresh_fold(self, fold_id: int) -> None:
        self._master_key, refresh_key = jax.random.split(self._master_key)
        candidate = self.geometry.sample(refresh_key, self.fold_size)
        forbidden = set().union(*self._learner_history) if self._learner_history else set()
        other_audit: set[str] = set()
        for index, points in enumerate(self._folds):
            if index != fold_id:
                other_audit.update(coordinate_hash(point) for point in np.asarray(points))
        candidate_hashes = {coordinate_hash(point) for point in np.asarray(candidate)}
        if candidate_hashes & (forbidden | other_audit):
            raise RuntimeError("Audit refresh produced prohibited coordinate overlap")
        self._generations[fold_id] += 1
        self._selection_counts[fold_id] = 0
        self._folds[fold_id] = candidate
        self._record_points(
            candidate,
            source="audit",
            fold_id=fold_id,
            generation=self._generations[fold_id],
        )
        self.assert_no_leakage()

    def leakage_report(self) -> LeakageReport:
        learner = set().union(*self._learner_history) if self._learner_history else set()
        audit_sets = [
            {coordinate_hash(point) for point in np.asarray(points)} for points in self._folds
        ]
        audit_union = set().union(*audit_sets)
        audit_overlap: set[str] = set()
        for index, current in enumerate(audit_sets):
            for other in audit_sets[index + 1 :]:
                audit_overlap.update(current & other)
        learner_overlap = learner & audit_union
        # register_learner checks uniqueness within every batch. Reuse across
        # learner steps is permitted for fixed-collocation parent baselines.
        learner_unique = True
        audit_unique = all(len(values) == self.fold_size for values in audit_sets)
        passed = not learner_overlap and not audit_overlap and learner_unique and audit_unique
        return LeakageReport(
            passed=passed,
            learner_unique=learner_unique,
            audit_unique=audit_unique,
            learner_audit_overlap=tuple(sorted(learner_overlap)),
            audit_fold_overlap=tuple(sorted(audit_overlap)),
            prohibited_window=self.prohibited_window,
        )

    def assert_no_leakage(self) -> None:
        report = self.leakage_report()
        if not report.passed:
            raise RuntimeError(f"Learner/audit leakage detected: {report}")

    def sample_hashes(self) -> dict[str, str]:
        roles = self.roles
        return {
            role: batch_hash(self._folds[getattr(roles, role)])
            for role in ("selector", "monitor", "calibration")
        }

    def provenance(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._provenance)

    def _record_points(self, points: Array, *, source: str, **metadata: Any) -> None:
        for local_id, point in enumerate(np.asarray(points)):
            self._provenance.append(
                {
                    "source": source,
                    "coordinate_hash": coordinate_hash(point),
                    "local_id": local_id,
                    **metadata,
                }
            )
