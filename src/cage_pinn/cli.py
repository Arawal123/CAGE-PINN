from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cage_pinn.baselines import baseline_registry, require_external_baseline
from cage_pinn.benchmarks import benchmark_registry
from cage_pinn.budgets import calibrate_costs
from cage_pinn.core import ExperimentManifest, RunConfig
from cage_pinn.core.schemas import stable_hash
from cage_pinn.literature import build_corpus
from cage_pinn.references import (
    find_external_reference,
    validate_external_metadata,
    verify_analytic_references,
)
from cage_pinn.training import run_training


def _print(value: Any) -> None:
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--problem", default="poisson_1d")
    parser.add_argument("--backbone", choices=("vanilla", "xpinn", "ab_pinn"), default="vanilla")
    parser.add_argument(
        "--method",
        choices=(
            "cage",
            "vanilla",
            "uniform",
            "rar_d",
            "gpinn",
            "weak_pinn",
            "static_sjw",
            "relobralo",
            "sa_pinn",
            "config",
        ),
        default="cage",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learner-points", type=int, default=32)
    parser.add_argument("--boundary-points", type=int, default=16)
    parser.add_argument("--audit-points", type=int, default=32)
    parser.add_argument("--control-interval", type=int, default=2)
    parser.add_argument("--total-ad-tokens", type=int, default=100_000)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", default="results/raw")
    parser.add_argument("--sketch", action="store_true", help="Use Rademacher utility sketches")
    parser.add_argument("--sketch-dim", type=int, default=32)


def _run_config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        problem=args.problem,
        backbone=args.backbone,
        method=args.method,
        seed=args.seed,
        steps=args.steps,
        learning_rate=args.learning_rate,
        width=args.width,
        depth=args.depth,
        learner_points=args.learner_points,
        boundary_points=args.boundary_points,
        audit_points=args.audit_points,
        control_interval=args.control_interval,
        total_ad_tokens=args.total_ad_tokens,
        precision=args.precision,
        output=args.output,
        exact_utility=not args.sketch,
        sketch_dim=args.sketch_dim,
    )


def command_literature(args: argparse.Namespace) -> int:
    if args.literature_command == "build":
        _print(build_corpus(args.directory))
        return 0
    raise AssertionError("unreachable")


def command_novelty(args: argparse.Namespace) -> int:
    matrix = Path("research/novelty/collision_matrix.md")
    decision = Path("research/novelty/decision.md")
    if not matrix.exists() or not decision.exists():
        _print({"passed": False, "reason": "novelty documents are missing"})
        return 2
    corpus_csv = Path("research/literature/papers.csv")
    records = len(pd.read_csv(corpus_csv)) if corpus_csv.exists() else 0
    result = {
        "passed_for_implementation": True,
        "publication_claim_approved": False,
        "records": records,
        "closest_collision_level": 2,
        "reason": "No Level 0/1 in seed screen; required snowball/full-text review incomplete.",
        "decision": str(decision),
        "matrix": str(matrix),
    }
    _print(result)
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    if args.benchmark_command == "list":
        _print([spec.to_dict() for spec in benchmark_registry().values()])
        return 0
    if args.benchmark_command == "verify-references":
        checks = verify_analytic_references(points=args.points, tolerance=args.tolerance)
        payload = [check.to_dict() for check in checks]
        _print(payload)
        return 0 if all(check.passed for check in checks) else 2
    if args.benchmark_command == "validate-external":
        _print(validate_external_metadata(args.metadata))
        return 0
    raise AssertionError("unreachable")


def command_baseline(args: argparse.Namespace) -> int:
    if args.baseline_command == "list":
        _print([asdict(spec) for spec in baseline_registry().values()])
        return 0
    if args.baseline_command == "require":
        _print(asdict(require_external_baseline(args.name)))
        return 0
    raise AssertionError("unreachable")


def command_budget(args: argparse.Namespace) -> int:
    if args.budget_command == "calibrate":
        path, payload = calibrate_costs(
            problem_name=args.problem,
            backbone_name=args.backbone,
            points=args.points,
            repeats=args.repeats,
            warmups=args.warmups,
            precision=args.precision,
            output_directory=args.output,
        )
        _print({"path": str(path), **payload})
        return 0
    raise AssertionError("unreachable")


def freeze_manifest(path: Path, output_directory: Path) -> Path:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Manifest must contain a mapping")
    raw.pop("manifest_hash", None)
    raw["frozen"] = True
    digest = stable_hash(raw)
    raw["manifest_hash"] = digest
    output_directory.mkdir(parents=True, exist_ok=True)
    target = output_directory / f"{raw['study']}-{digest[:12]}.yaml"
    serialized = yaml.safe_dump(raw, sort_keys=False)
    if target.exists() and target.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Frozen manifest collision: {target}")
    if not target.exists():
        target.write_text(serialized, encoding="utf-8")
    ExperimentManifest.from_yaml(target)
    return target


def _manifest_plan(manifest: ExperimentManifest) -> dict[str, Any]:
    registry = benchmark_registry()
    missing = [name for name in manifest.problems if name not in registry]
    blocked_code = [
        {
            "problem": name,
            "reference_status": registry[name].reference_status,
            "reason": "executable PDE/reference adapter not yet enabled",
        }
        for name in manifest.problems
        if name in registry and not registry[name].executable
    ]
    requires_reference = bool(
        manifest.reference_policy.get("require_validated_reference", False)
    )
    accepted_reference_statuses = {
        "verified",
        "verified away from jump",
        "verified away from interface",
    }
    external_reference_ready = {
        name: find_external_reference(name) is not None
        for name in manifest.problems
        if name in registry
        and registry[name].reference_status not in accepted_reference_statuses
    }
    blocked_references = [
        {
            "problem": name,
            "reference_status": registry[name].reference_status,
            "reason": "validated numerical reference is required before this study",
        }
        for name in manifest.problems
        if (
            name in registry
            and registry[name].executable
            and requires_reference
            and registry[name].reference_status not in accepted_reference_statuses
            and not external_reference_ready.get(name, False)
        )
    ]
    runs = (
        len(manifest.seeds)
        * len(manifest.problems)
        * len(manifest.method_backbone_pairs)
    )
    return {
        "study": manifest.study,
        "runs": runs,
        "budget_per_run": manifest.budget,
        "frozen": manifest.frozen,
        "manifest_hash": manifest.manifest_hash,
        "unknown_problems": missing,
        "blocked_code": blocked_code,
        "blocked_references": blocked_references,
        "executable": not missing and not blocked_code and not blocked_references,
    }


def command_study(args: argparse.Namespace) -> int:
    if args.study_command == "smoke":
        outcome = run_training(_run_config_from_args(args))
        _print(
            {
                "status": outcome.result.status,
                "run_id": outcome.result.run_id,
                "result": str(outcome.path),
                "metrics": outcome.result.metrics,
                "spent_tokens": outcome.result.ledger["spent_tokens"],
            }
        )
        return 0
    if args.study_command == "freeze":
        target = freeze_manifest(Path(args.manifest), Path(args.output_directory))
        _print({"frozen_manifest": str(target)})
        return 0
    if args.study_command in {"discovery", "confirmatory"}:
        manifest = ExperimentManifest.from_yaml(args.manifest)
        if manifest.study != args.study_command:
            raise ValueError(
                f"Manifest study {manifest.study!r} does not match command {args.study_command!r}"
            )
        plan = _manifest_plan(manifest)
        if not args.execute:
            plan["mode"] = "dry-run"
            _print(plan)
            return 0
        if not manifest.frozen:
            _print(
                {
                    "started": False,
                    "reason": "Execution requires a frozen manifest. Run `study freeze` first.",
                }
            )
            return 2
        if not plan["executable"]:
            plan["started"] = False
            plan["reason"] = (
                "Scientific guard: unresolved PDE/reference adapters block execution; "
                "no partial confirmatory campaign was silently run."
            )
            _print(plan)
            return 2
        completed = []
        for problem in manifest.problems:
            for method, backbone in manifest.method_backbone_pairs:
                for seed in manifest.seeds:
                    run_config = RunConfig(
                        problem=problem,
                        backbone=backbone,
                        method=method,
                        seed=seed,
                        steps=int(manifest.budget.get("steps", 1000)),
                        total_ad_tokens=int(manifest.budget["total"]),
                        output=f"experiments/{manifest.study}",
                    )
                    outcome = run_training(run_config)
                    completed.append(outcome.result.run_id)
        _print({"completed_runs": len(completed), "run_ids": completed})
        return 0
    raise AssertionError("unreachable")


def command_analyze(args: argparse.Namespace) -> int:
    from cage_pinn.reporting import load_raw_records
    from cage_pinn.statistics import analyze_paired

    records = load_raw_records(args.raw)
    rows = []
    for record in records:
        error = record.get("metrics", {}).get("relative_l2_post_training")
        if record.get("status") != "completed" or error is None:
            continue
        config = record["config"]
        rows.append(
            {
                "problem": config["problem"],
                "backbone": config["backbone"],
                "seed": config["seed"],
                "steps": config["steps"],
                "total_ad_tokens": config["total_ad_tokens"],
                "method": config["method"],
                "error": float(error),
            }
        )
    frame = pd.DataFrame(rows)
    analyses = []
    if not frame.empty and "cage" in set(frame["method"]):
        key_columns = ["problem", "backbone", "seed", "steps", "total_ad_tokens"]
        cage = frame.loc[
            frame["method"] == "cage", [*key_columns, "error"]
        ].rename(
            columns={"error": "cage_error"}
        )
        for baseline_name in sorted(set(frame["method"]) - {"cage"}):
            baseline = frame.loc[
                frame["method"] == baseline_name, [*key_columns, "error"]
            ].rename(columns={"error": "baseline_error"})
            paired = cage.merge(baseline, on=key_columns, how="inner")
            if len(paired) >= 3:
                result = analyze_paired(
                    paired["cage_error"].to_numpy(),
                    paired["baseline_error"].to_numpy(),
                    bootstrap_seed=0,
                ).to_dict()
                analyses.append({"baseline": baseline_name, **result})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "study": args.study_name,
        "raw_records": len(records),
        "completed": sum(record.get("status") == "completed" for record in records),
        "paired_analyses": analyses,
        "status": (
            "Paired analyses emitted from raw records."
            if analyses
            else "No inferential analysis emitted without at least three complete paired runs."
        ),
    }
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["output"] = str(output)
    _print(summary)
    return 0


def command_report(args: argparse.Namespace) -> int:
    from cage_pinn.reporting import build_report

    _print(build_report(args.raw, args.processed, args.figures))
    return 0


def command_audit(args: argparse.Namespace) -> int:
    value = json.loads(Path(args.path).read_text(encoding="utf-8"))
    required = {
        "run_id",
        "config",
        "status",
        "git_commit",
        "dependency_hash",
        "hardware",
        "precision",
        "ledger",
        "metrics",
    }
    missing = sorted(required - value.keys())
    leakage = value.get("metrics", {}).get("leakage", {})
    result = {
        "path": args.path,
        "schema_complete": not missing,
        "missing": missing,
        "leakage_passed": leakage.get("passed", False),
        "reference_used_during_training": value.get("metrics", {}).get(
            "reference_used_during_training", "unknown"
        ),
        "budget_within_limit": value.get("ledger", {}).get("remaining_tokens", -1) >= 0,
    }
    result["passed"] = (
        result["schema_complete"]
        and result["leakage_passed"]
        and result["reference_used_during_training"] is False
        and result["budget_within_limit"]
    )
    _print(result)
    return 0 if result["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cage-pinn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    literature = subparsers.add_parser("literature")
    literature_sub = literature.add_subparsers(dest="literature_command", required=True)
    literature_build = literature_sub.add_parser("build")
    literature_build.add_argument("--directory", default="research/literature")

    novelty = subparsers.add_parser("novelty")
    novelty.add_subparsers(dest="novelty_command", required=True).add_parser("audit")

    benchmark = subparsers.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_sub.add_parser("list")
    verify = benchmark_sub.add_parser("verify-references")
    verify.add_argument("--points", type=int, default=32)
    verify.add_argument("--tolerance", type=float, default=1.0e-9)
    validate_external = benchmark_sub.add_parser("validate-external")
    validate_external.add_argument("metadata")

    baseline = subparsers.add_parser("baseline")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_sub.add_parser("list")
    require = baseline_sub.add_parser("require")
    require.add_argument("name")

    budget = subparsers.add_parser("budget")
    budget_sub = budget.add_subparsers(dest="budget_command", required=True)
    calibrate = budget_sub.add_parser("calibrate")
    calibrate.add_argument("--problem", default="poisson_1d")
    calibrate.add_argument(
        "--backbone", choices=("vanilla", "xpinn", "ab_pinn"), default="vanilla"
    )
    calibrate.add_argument("--points", type=int, default=32)
    calibrate.add_argument("--repeats", type=int, default=5)
    calibrate.add_argument("--warmups", type=int, default=2)
    calibrate.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    calibrate.add_argument("--output", default="results/calibration")

    study = subparsers.add_parser("study")
    study_sub = study.add_subparsers(dest="study_command", required=True)
    smoke = study_sub.add_parser("smoke")
    _add_common_run_arguments(smoke)
    freeze = study_sub.add_parser("freeze")
    freeze.add_argument("--manifest", required=True)
    freeze.add_argument("--output-directory", default="research/preregistration/frozen")
    for name in ("discovery", "confirmatory"):
        experiment = study_sub.add_parser(name)
        experiment.add_argument("--manifest", required=True)
        experiment.add_argument("--execute", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("study_name")
    analyze.add_argument("--raw", default="results/raw")
    analyze.add_argument("--output", default="results/processed/paired_analysis.json")

    report = subparsers.add_parser("report")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    report_build = report_sub.add_parser("build")
    report_build.add_argument("--raw", default="results/raw")
    report_build.add_argument("--processed", default="results/processed")
    report_build.add_argument("--figures", default="results/figures")

    audit = subparsers.add_parser("audit")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    audit_run = audit_sub.add_parser("run")
    audit_run.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "literature": command_literature,
        "novelty": command_novelty,
        "benchmark": command_benchmark,
        "baseline": command_baseline,
        "budget": command_budget,
        "study": command_study,
        "analyze": command_analyze,
        "report": command_report,
        "audit": command_audit,
    }
    try:
        return handlers[args.command](args)
    except Exception as exc:
        _print({"error": type(exc).__name__, "message": str(exc)})
        return 2


if __name__ == "__main__":
    sys.exit(main())
