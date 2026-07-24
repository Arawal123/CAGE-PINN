from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback as traceback_module
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import yaml


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def dependency_fingerprint(project_root: Path | None = None) -> str:
    root = project_root or Path.cwd()
    candidates = [root / "uv.lock", root / "pyproject.toml", root / "requirements-colab.txt"]
    digest = hashlib.sha256()
    found = False
    for path in candidates:
        if path.exists():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
            found = True
    return digest.hexdigest() if found else "unavailable"


@dataclass(frozen=True)
class SeedHierarchy:
    root: int
    model: int
    learner: int
    audit: int
    weak: int
    controller: int

    @classmethod
    def derive(cls, root: int) -> SeedHierarchy:
        values = []
        for label in ("model", "learner", "audit", "weak", "controller"):
            digest = hashlib.sha256(f"{root}:{label}".encode()).digest()
            values.append(int.from_bytes(digest[:4], "little") & 0x7FFFFFFF)
        return cls(root, *values)


@dataclass(frozen=True)
class RunConfig:
    problem: str = "poisson_1d"
    backbone: str = "vanilla"
    method: str = "cage"
    seed: int = 0
    steps: int = 8
    learning_rate: float = 1.0e-3
    width: int = 24
    depth: int = 2
    learner_points: int = 32
    boundary_points: int = 16
    audit_points: int = 32
    control_interval: int = 2
    rotate_interval: int = 2
    refresh_after_selections: int = 3
    weak_quadrature_points: int = 6
    precision: str = "float64"
    total_ad_tokens: int = 100_000
    output: str = "results/raw"
    exact_utility: bool = True
    sketch_dim: int = 32

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.control_interval <= 0:
            raise ValueError("control_interval must be positive")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        if self.total_ad_tokens <= 0:
            raise ValueError("total_ad_tokens must be positive")


@dataclass(frozen=True)
class ExperimentManifest:
    study: str
    frozen: bool
    created_at: str
    seeds: tuple[int, ...]
    problems: tuple[str, ...]
    methods: tuple[str, ...]
    backbones: tuple[str, ...]
    method_backbone_pairs: tuple[tuple[str, str], ...]
    budget: dict[str, Any]
    tuning_policy: dict[str, Any]
    reference_policy: dict[str, Any]
    manifest_hash: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentManifest:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest {path} must contain a mapping")
        manifest = cls(
            study=str(raw["study"]),
            frozen=bool(raw.get("frozen", False)),
            created_at=str(raw.get("created_at", "")),
            seeds=tuple(int(v) for v in raw["seeds"]),
            problems=tuple(str(v) for v in raw["problems"]),
            methods=tuple(str(v) for v in raw["methods"]),
            backbones=tuple(str(v) for v in raw["backbones"]),
            method_backbone_pairs=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in raw.get("method_backbone_pairs", [])
            ),
            budget=dict(raw["budget"]),
            tuning_policy=dict(raw.get("tuning_policy", {})),
            reference_policy=dict(raw.get("reference_policy", {})),
            manifest_hash=str(raw.get("manifest_hash", "")),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not self.seeds or not self.problems or not self.methods or not self.backbones:
            raise ValueError("Manifest experiment axes cannot be empty")
        if not self.method_backbone_pairs:
            raise ValueError("Manifest must declare explicit method_backbone_pairs")
        invalid_pairs = [
            pair
            for pair in self.method_backbone_pairs
            if pair[0] not in self.methods or pair[1] not in self.backbones
        ]
        if invalid_pairs:
            raise ValueError(f"Manifest has invalid method/backbone pairs: {invalid_pairs}")
        if int(self.budget.get("total", 0)) <= 0:
            raise ValueError("Manifest budget.total must be positive")
        if self.frozen and not self.manifest_hash:
            raise ValueError("Frozen manifest must include manifest_hash")

    def payload_for_hash(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("manifest_hash", None)
        return payload


@dataclass
class ResultRecord:
    run_id: str
    config: dict[str, Any]
    seeds: dict[str, int]
    status: str = "running"
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    git_commit: str = field(default_factory=git_commit)
    dependency_hash: str = field(default_factory=dependency_fingerprint)
    hardware: dict[str, Any] = field(default_factory=dict)
    precision: str = "float64"
    compile_seconds: float = 0.0
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    optimizer_steps: int = 0
    parameter_count: int = 0
    sample_hashes: dict[str, str] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    traceback: str | None = None

    @classmethod
    def begin(cls, config: RunConfig) -> ResultRecord:
        config.validate()
        seeds = SeedHierarchy.derive(config.seed)
        run_id = stable_hash({"config": asdict(config), "started_at": utc_now()})[:16]
        return cls(
            run_id=run_id,
            config=asdict(config),
            seeds=asdict(seeds),
            precision=config.precision,
            hardware=capture_environment(),
        )

    def finish(self, status: str = "completed") -> None:
        self.status = status
        self.ended_at = utc_now()
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.ended_at)
        self.total_seconds = (end - start).total_seconds()

    def fail(self) -> None:
        self.status = "failed"
        self.traceback = traceback_module.format_exc()
        self.finish("failed")

    def write_immutable(self, directory: str | Path) -> Path:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{self.run_id}.json"
        if target.exists():
            raise FileExistsError(f"Raw result is immutable: {target}")
        target.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return target


def capture_environment() -> dict[str, Any]:
    devices = []
    try:
        devices = [str(device) for device in jax.devices()]
    except RuntimeError as exc:
        devices = [f"JAX unavailable: {exc}"]
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "devices": devices,
        "hostname": platform.node(),
        "pid": os.getpid(),
    }
