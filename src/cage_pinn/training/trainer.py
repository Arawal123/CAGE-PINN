from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from cage_pinn.audit import (
    AuditFoldManager,
    ResidualScaleTracker,
    estimate_audit_risk,
)
from cage_pinn.backbones import Backbone, create_backbone, parameter_count
from cage_pinn.baselines import ReLoBRaLoState, config_two_gradients
from cage_pinn.budgets import BudgetExceeded, BudgetLedger, CostModel
from cage_pinn.controller import AllocationResult, BudgetAllocator, estimate_utilities
from cage_pinn.core import ResultRecord, RunConfig, SeedHierarchy
from cage_pinn.enforcement import AtomEvaluation, EnforcementAtom, default_atom_bank
from cage_pinn.pdes import PDEProblem, create_problem
from cage_pinn.references import find_external_reference
from cage_pinn.sampling import ResidualAdaptiveSampler


@dataclass(frozen=True)
class TrainingOutcome:
    model: Backbone
    result: ResultRecord
    path: Path | None


def _combine_gradients(
    first: Any, second: Any, *, first_weight: float = 1.0, second_weight: float = 1.0
) -> Any:
    def combine(left: Any, right: Any) -> Any:
        if left is None and right is None:
            return None
        if left is None:
            return second_weight * right
        if right is None:
            return first_weight * left
        return first_weight * left + second_weight * right

    return jax.tree_util.tree_map(
        combine, first, second, is_leaf=lambda value: value is None
    )


def _allocation_for_method(
    method: str, allocator: BudgetAllocator, applicable: dict[str, bool]
) -> AllocationResult:
    if method in {
        "vanilla",
        "uniform",
        "rar_d",
        "relobralo",
        "sa_pinn",
        "config",
    }:
        utilities = {"S": 1.0e6, "J": 0.0, "W": 0.0}
    elif method == "gpinn":
        utilities = {"S": 1.0, "J": 1.0, "W": 0.0}
    elif method == "weak_pinn":
        utilities = {"S": 1.0, "J": 0.0, "W": 1.0}
    elif method == "static_sjw":
        utilities = {"S": 1.0, "J": 1.0, "W": 1.0}
    else:
        utilities = {"S": 0.0, "J": 0.0, "W": 0.0}
    method_applicable = dict(applicable)
    if method in {
        "vanilla",
        "uniform",
        "rar_d",
        "relobralo",
        "sa_pinn",
        "config",
    }:
        method_applicable.update({"J": False, "W": False})
    elif method == "gpinn":
        method_applicable["W"] = False
    elif method == "weak_pinn":
        method_applicable["J"] = False
    return allocator.allocate(utilities, applicable=method_applicable, tokens=10_000)


def _choose_atom_by_compute_fair_queue(
    shares: dict[str, float],
    spent: dict[str, int],
    applicable: dict[str, bool],
) -> str:
    choices = []
    for name, share in shares.items():
        if applicable.get(name, False) and share > 0:
            virtual_finish = spent.get(name, 0) / share
            choices.append((virtual_finish, name))
    if not choices:
        raise RuntimeError("No applicable atom has positive allocation")
    return min(choices)[1]


def _audit_objective(
    model: Backbone,
    problem: PDEProblem,
    points: jax.Array,
    scales: jax.Array,
) -> jax.Array:
    normalized = problem.residuals(model, points) / (
        jax.lax.stop_gradient(scales) + 1.0e-8
    )
    risk = estimate_audit_risk(
        normalized,
        q=0.95,
        delta=0.05,
        planned_audits=100,
        certified_schedule=False,
    )
    return risk.aggregate


def _evaluate_atom_gradient(
    atom: EnforcementAtom,
    model: Backbone,
    problem: PDEProblem,
    points: jax.Array,
    scales: jax.Array,
    key: jax.Array,
    cost_model: CostModel,
) -> tuple[AtomEvaluation, Any]:
    def loss_function(candidate: Backbone) -> jax.Array:
        return atom.evaluate(candidate, problem, points, scales, key, cost_model).loss

    loss, gradient = eqx.filter_value_and_grad(loss_function)(model)
    evaluation = atom.evaluate(model, problem, points, scales, key, cost_model)
    if not jnp.allclose(loss, evaluation.loss, rtol=1.0e-6, atol=1.0e-8):
        raise RuntimeError(f"Non-deterministic loss evaluation in atom {atom.name}")
    return evaluation, gradient


def _evaluate_self_adaptive_strong(
    model: Backbone,
    problem: PDEProblem,
    points: jax.Array,
    scales: jax.Array,
    weights: jax.Array,
    cost_model: CostModel,
) -> tuple[AtomEvaluation, Any]:
    fixed_scales = jax.lax.stop_gradient(scales)
    fixed_weights = jax.lax.stop_gradient(weights)

    def loss_function(candidate: Backbone) -> jax.Array:
        normalized = problem.residuals(candidate, points) / (fixed_scales + 1.0e-8)
        point_losses = jnp.mean(normalized**2, axis=1)
        return jnp.mean(fixed_weights * point_losses)

    loss, gradient = eqx.filter_value_and_grad(loss_function)(model)
    tokens = round(
        cost_model.residual(
            len(points), problem.residual_derivative_order, backward=True
        )
    )
    return (
        AtomEvaluation(
            name="S",
            loss=loss,
            tokens=tokens,
            provenance={
                "points": len(points),
                "self_adaptive_point_weights": True,
                "weight_min": float(jnp.min(weights)),
                "weight_max": float(jnp.max(weights)),
            },
        ),
        gradient,
    )


def _risk_snapshot(
    model: Backbone,
    problem: PDEProblem,
    points: jax.Array,
    scales: jax.Array,
    *,
    certified: bool,
    weak_atom: EnforcementAtom | None = None,
    weak_key: jax.Array | None = None,
    cost_model: CostModel | None = None,
) -> dict[str, Any]:
    residual = problem.residuals(model, points)
    normalized = residual / (scales + 1.0e-8)
    risk = estimate_audit_risk(normalized, certified_schedule=certified)
    snapshot = risk.to_dict()
    snapshot["raw_abs_mean"] = [
        float(value) for value in jnp.mean(jnp.abs(residual), axis=0)
    ]
    stratified: dict[str, float] = {}
    stratified_extra_points = 0
    for name, stratum in model.audit_strata(points).items():
        if len(stratum) == 0:
            continue
        if name == "global":
            stratified[name] = float(risk.empirical_aggregate)
            continue
        stratum_residual = problem.residuals(model, stratum) / (scales + 1.0e-8)
        stratified_extra_points += len(stratum)
        stratum_risk = estimate_audit_risk(
            stratum_residual, certified_schedule=False
        )
        stratified[name] = float(stratum_risk.empirical_aggregate)
    snapshot["stratified_empirical_aggregate"] = stratified
    snapshot["worst_stratum_empirical"] = (
        max(stratified.values()) if stratified else None
    )
    snapshot["stratified_extra_points"] = stratified_extra_points
    if weak_atom is not None:
        if weak_key is None or cost_model is None:
            raise ValueError("Weak audit snapshot requires key and cost model")
        weak_evaluation = weak_atom.evaluate(
            model, problem, points, scales, weak_key, cost_model
        )
        snapshot["weak_audit"] = {
            "loss": float(weak_evaluation.loss),
            "applicable": weak_evaluation.applicable,
            "reason": weak_evaluation.reason,
            "provenance": weak_evaluation.provenance,
            "training_equivalent_tokens": weak_evaluation.tokens,
            "seed_words": [
                int(value) for value in jax.random.key_data(weak_key).tolist()
            ],
        }
    return snapshot


def run_training(config: RunConfig, *, write_result: bool = True) -> TrainingOutcome:
    config.validate()
    if config.precision == "float64":
        jax.config.update("jax_enable_x64", True)
    record = ResultRecord.begin(config)
    seeds = SeedHierarchy.derive(config.seed)
    problem = create_problem(config.problem)
    model_key = jax.random.PRNGKey(seeds.model)
    model = create_backbone(
        config.backbone,
        input_dim=problem.geometry.dimension,
        output_dim=problem.output_dim,
        width=config.width,
        depth=config.depth,
        key=model_key,
        lower=float(problem.geometry.lower[0]),
        upper=float(problem.geometry.upper[0]),
    )
    record.parameter_count = parameter_count(model)
    audit_manager = AuditFoldManager(
        problem.geometry,
        jax.random.PRNGKey(seeds.audit),
        fold_size=config.audit_points,
        rotate_interval=config.rotate_interval,
        refresh_after_selections=config.refresh_after_selections,
    )
    tracker = ResidualScaleTracker(len(problem.residual_channel_names))
    atoms = default_atom_bank(weak_quadrature_points=config.weak_quadrature_points)
    applicable = {name: atom.applicable(problem)[0] for name, atom in atoms.items()}
    cost_model = CostModel()
    ledger = BudgetLedger(config.total_ad_tokens)
    allocator = BudgetAllocator()
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    learner_key = jax.random.PRNGKey(seeds.learner)
    weak_key = jax.random.PRNGKey(seeds.weak)
    controller_key = jax.random.PRNGKey(seeds.controller)
    boundary_key = jax.random.PRNGKey(seeds.learner ^ 0x5A5A5A5A)
    current_allocation = _allocation_for_method(config.method, allocator, applicable)
    training_spent = {name: 0 for name in atoms}
    control_index = 0
    relobralo_state: ReLoBRaLoState | None = None
    self_adaptive_weights = jnp.ones((config.learner_points,))
    compile_start = time.perf_counter()
    learner_key, fixed_sample_key = jax.random.split(learner_key)
    fixed_learner_points = problem.sample_interior(
        fixed_sample_key, config.learner_points
    )
    residual_adaptive_sampler = ResidualAdaptiveSampler(problem)
    try:
        for step in range(config.steps):
            learner_key, sample_key = jax.random.split(learner_key)
            if config.method == "rar_d" and step > 0:
                learner_points = residual_adaptive_sampler.sample(
                    sample_key,
                    config.learner_points,
                    model=model,
                    scales=tracker.scales,
                )
                ledger.charge(
                    "adaptive_sampling_candidates",
                    round(
                        cost_model.residual(
                            4 * config.learner_points,
                            problem.residual_derivative_order,
                            backward=False,
                        )
                    ),
                    step=step,
                    metadata={"method": "rar_d"},
                )
            elif config.method in {
                "vanilla",
                "gpinn",
                "relobralo",
                "sa_pinn",
                "config",
            }:
                learner_points = fixed_learner_points
            else:
                learner_points = problem.sample_interior(sample_key, config.learner_points)
            audit_manager.register_learner(learner_points, step=step)
            raw_train_residual = problem.residuals(model, learner_points)
            ledger.charge(
                "normalization_scale_forward",
                round(
                    cost_model.residual(
                        len(learner_points),
                        problem.residual_derivative_order,
                        backward=False,
                    )
                ),
                step=step,
            )
            scales = tracker.update(raw_train_residual)
            is_control = config.method == "cage" and step % config.control_interval == 0
            selector_before = None
            utility_payload: dict[str, object] = {}
            calibration_snapshot = None
            if is_control:
                control_index += 1
                selector_points = audit_manager.points("selector")
                selector_weak_key = jax.random.fold_in(
                    jax.random.PRNGKey(seeds.audit), 10_000 + control_index
                )

                def selector_objective(
                    candidate: Backbone,
                    fixed_points: jax.Array = selector_points,
                    fixed_scales: jax.Array = scales,
                    fixed_key: jax.Array = selector_weak_key,
                ) -> jax.Array:
                    strong = _audit_objective(
                        candidate, problem, fixed_points, fixed_scales
                    )
                    weak = atoms["W"].evaluate(
                        candidate,
                        problem,
                        fixed_points,
                        fixed_scales,
                        fixed_key,
                        cost_model,
                    )
                    return strong + jnp.where(
                        weak.applicable,
                        0.1 * jnp.sqrt(weak.loss + 1.0e-12),
                        0.0,
                    )

                selector_before, audit_gradient = eqx.filter_value_and_grad(
                    selector_objective
                )(model)
                audit_cost = round(
                    cost_model.residual(
                        len(selector_points),
                        problem.residual_derivative_order,
                        backward=True,
                    )
                )
                ledger.charge(
                    "selector_audit_gradient",
                    audit_cost,
                    step=step,
                    metadata={"role": "selector", "direct_training_loss": False},
                )
                selector_weak = atoms["W"].evaluate(
                    model,
                    problem,
                    selector_points,
                    scales,
                    selector_weak_key,
                    cost_model,
                )
                if selector_weak.applicable:
                    ledger.charge(
                        "selector_weak_audit_gradient",
                        selector_weak.tokens,
                        step=step,
                        atom="W",
                        metadata={
                            "role": "selector",
                            "direct_parameter_update": False,
                        },
                    )
                candidate_gradients: dict[str, Any] = {}
                candidate_costs: dict[str, float] = {}
                for atom_index, (name, atom) in enumerate(atoms.items()):
                    if not applicable[name]:
                        continue
                    weak_key, candidate_key = jax.random.split(weak_key)
                    evaluation, gradient = _evaluate_atom_gradient(
                        atom,
                        model,
                        problem,
                        learner_points,
                        scales,
                        jax.random.fold_in(candidate_key, atom_index),
                        cost_model,
                    )
                    candidate_gradients[name] = gradient
                    candidate_costs[name] = max(evaluation.tokens, 1)
                    ledger.charge(
                        "controller_candidate_gradient",
                        evaluation.tokens,
                        step=step,
                        atom=name,
                        metadata={"direct_parameter_update": False},
                    )
                controller_key, utility_key = jax.random.split(controller_key)
                utility_estimates = estimate_utilities(
                    audit_gradient,
                    candidate_gradients,
                    candidate_costs,
                    key=utility_key,
                    exact=config.exact_utility,
                    sketch_dim=config.sketch_dim,
                )
                utility_values = {
                    name: utility_estimates[name].utility if name in utility_estimates else 0.0
                    for name in atoms
                }
                current_allocation = allocator.allocate(
                    utility_values, applicable=applicable, tokens=10_000
                )
                utility_payload = {
                    name: asdict(estimate) for name, estimate in utility_estimates.items()
                }

            selected_atom_name = _choose_atom_by_compute_fair_queue(
                current_allocation.requested, training_spent, applicable
            )
            selected_atom = atoms[selected_atom_name]
            weak_key, atom_key = jax.random.split(weak_key)
            if config.method == "sa_pinn":
                normalized_for_weights = problem.residuals(
                    model, learner_points
                ) / (scales + 1.0e-8)
                point_losses = jnp.mean(normalized_for_weights**2, axis=1)
                self_adaptive_weights = jnp.clip(
                    self_adaptive_weights
                    + 0.01 * jax.lax.stop_gradient(point_losses),
                    1.0e-3,
                    100.0,
                )
                self_adaptive_weights *= (
                    self_adaptive_weights.size
                    / jnp.sum(self_adaptive_weights)
                )
                ledger.charge(
                    "self_adaptive_weight_forward",
                    round(
                        cost_model.residual(
                            len(learner_points),
                            problem.residual_derivative_order,
                            backward=False,
                        )
                    ),
                    step=step,
                    metadata={"method": "sa_pinn"},
                )
                atom_evaluation, atom_gradient = _evaluate_self_adaptive_strong(
                    model,
                    problem,
                    learner_points,
                    scales,
                    self_adaptive_weights,
                    cost_model,
                )
            else:
                atom_evaluation, atom_gradient = _evaluate_atom_gradient(
                    selected_atom,
                    model,
                    problem,
                    learner_points,
                    scales,
                    atom_key,
                    cost_model,
                )
            boundary_key, constraint_key = jax.random.split(boundary_key)

            def constraint_loss(
                candidate: Backbone, fixed_key: jax.Array = constraint_key
            ) -> jax.Array:
                return problem.boundary_loss(
                    candidate, fixed_key, config.boundary_points
                ) + candidate.interface_loss()

            constraints, constraint_gradient = eqx.filter_value_and_grad(constraint_loss)(model)
            atom_weight = 1.0
            constraint_weight = 1.0
            if config.method == "relobralo":
                losses = jnp.asarray([atom_evaluation.loss, constraints]) + 1.0e-12
                if relobralo_state is None:
                    relobralo_state = ReLoBRaLoState.initialize(losses)
                weights = relobralo_state.update(
                    losses, lookback_to_start=step % 10 == 0
                )
                atom_weight, constraint_weight = map(float, weights)
            if config.method == "config":
                combined_gradient = config_two_gradients(
                    atom_gradient, constraint_gradient
                )
            else:
                combined_gradient = _combine_gradients(
                    atom_gradient,
                    constraint_gradient,
                    first_weight=atom_weight,
                    second_weight=constraint_weight,
                )
            updates, opt_state = optimizer.update(
                combined_gradient, opt_state, eqx.filter(model, eqx.is_inexact_array)
            )
            model = eqx.apply_updates(model, updates)
            ledger.charge(
                "training_atom",
                atom_evaluation.tokens,
                step=step,
                atom=selected_atom_name,
                metadata=atom_evaluation.provenance,
            )
            boundary_tokens = round(
                cost_model.residual(config.boundary_points, 0, backward=True)
            )
            ledger.charge("boundary_initial_interface", boundary_tokens, step=step)
            training_spent[selected_atom_name] += atom_evaluation.tokens
            total_training_atom_tokens = sum(training_spent.values())
            realized_training_shares = {
                name: (
                    training_spent[name] / total_training_atom_tokens
                    if total_training_atom_tokens
                    else 0.0
                )
                for name in atoms
            }

            monitor_points = audit_manager.points("monitor")
            monitor_weak_key = jax.random.fold_in(
                jax.random.PRNGKey(seeds.audit), 20_000 + step
            )
            monitor_snapshot = _risk_snapshot(
                model,
                problem,
                monitor_points,
                scales,
                certified=True,
                weak_atom=atoms["W"],
                weak_key=monitor_weak_key,
                cost_model=cost_model,
            )
            monitor_cost = round(
                cost_model.residual(
                    len(monitor_points), problem.residual_derivative_order, backward=False
                )
            )
            ledger.charge(
                "monitor_audit_forward",
                monitor_cost,
                step=step,
                metadata={"role": "monitor", "influences_update": False},
            )
            monitor_stratified_points = int(
                monitor_snapshot["stratified_extra_points"]
            )
            if monitor_stratified_points:
                ledger.charge(
                    "monitor_stratified_audit_forward",
                    round(
                        cost_model.residual(
                            monitor_stratified_points,
                            problem.residual_derivative_order,
                            backward=False,
                        )
                    ),
                    step=step,
                    metadata={"role": "monitor", "influences_update": False},
                )
            monitor_weak_tokens = round(
                monitor_snapshot["weak_audit"]["training_equivalent_tokens"]
                / cost_model.backward_multiplier
            )
            ledger.charge(
                "monitor_weak_audit_forward",
                monitor_weak_tokens,
                step=step,
                atom="W",
                metadata={"role": "monitor", "influences_update": False},
            )
            if is_control:
                calibration_points = audit_manager.points("calibration")
                calibration_snapshot = _risk_snapshot(
                    model,
                    problem,
                    calibration_points,
                    scales,
                    certified=True,
                )
                calibration_cost = round(
                    cost_model.residual(
                        len(calibration_points),
                        problem.residual_derivative_order,
                        backward=False,
                    )
                )
                ledger.charge(
                    "calibration_audit_forward",
                    calibration_cost,
                    step=step,
                    metadata={
                        "role": "calibration",
                        "influences_update": False,
                    },
                )
                calibration_stratified_points = int(
                    calibration_snapshot["stratified_extra_points"]
                )
                if calibration_stratified_points:
                    ledger.charge(
                        "calibration_stratified_audit_forward",
                        round(
                            cost_model.residual(
                                calibration_stratified_points,
                                problem.residual_derivative_order,
                                backward=False,
                            )
                        ),
                        step=step,
                        metadata={
                            "role": "calibration",
                            "influences_update": False,
                        },
                    )
            train_snapshot = _risk_snapshot(
                model, problem, learner_points, scales, certified=False
            )
            train_diagnostic_points = len(learner_points) + int(
                train_snapshot["stratified_extra_points"]
            )
            ledger.charge(
                "learner_diagnostic_forward",
                round(
                    cost_model.residual(
                        train_diagnostic_points,
                        problem.residual_derivative_order,
                        backward=False,
                    )
                ),
                step=step,
                metadata={"influences_update": False},
            )
            monitor_value = float(monitor_snapshot["empirical_aggregate"])
            monitor_upper = float(monitor_snapshot["upper_aggregate"])
            train_value = float(train_snapshot["empirical_aggregate"])
            gap = math.log((monitor_value + 1.0e-12) / (train_value + 1.0e-12))
            bound_gap = math.log(
                (monitor_upper + 1.0e-12) / (train_value + 1.0e-12)
            )
            record.history.append(
                {
                    "step": step,
                    "selected_atom": selected_atom_name,
                    "atom_loss": float(atom_evaluation.loss),
                    "constraint_loss": float(constraints),
                    "loss_weights": {
                        "physics": atom_weight,
                        "constraints": constraint_weight,
                    },
                    "scales": [float(value) for value in scales],
                    "allocation_requested": current_allocation.requested,
                    "allocation_realized": current_allocation.realized,
                    "cumulative_training_compute_shares": realized_training_shares,
                    "allocation_fallback": current_allocation.fallback,
                    "utilities": utility_payload,
                    "selector_objective_before_update": (
                        float(selector_before) if selector_before is not None else None
                    ),
                    "monitor": monitor_snapshot,
                    "calibration": calibration_snapshot,
                    "train": train_snapshot,
                    "generalization_gap": gap,
                    "generalization_gap_bound": bound_gap,
                    "roles": asdict(audit_manager.roles),
                    "budget_spent": ledger.spent_tokens,
                }
            )
            if is_control:
                audit_manager.on_control(control_index)
            if step == 0:
                record.compile_seconds = time.perf_counter() - compile_start
        record.training_seconds = time.perf_counter() - compile_start
        relative_l2 = None
        external_reference = None
        if not problem.reference_available:
            external_reference = find_external_reference(problem.name)
        if problem.reference_available:
            evaluation_key = jax.random.PRNGKey(config.seed ^ 0x13579BDF)
            evaluation_points = problem.sample_interior(evaluation_key, 256)
            relative_l2 = float(problem.relative_l2(model, evaluation_points))
        elif external_reference is not None:
            evaluation_key = jax.random.PRNGKey(config.seed ^ 0x13579BDF)
            evaluation_points = problem.sample_interior(evaluation_key, 256)
            prediction = jax.vmap(model)(evaluation_points)
            truth = external_reference.evaluate(evaluation_points)
            relative_l2 = float(
                jnp.linalg.norm(prediction - truth)
                / (jnp.linalg.norm(truth) + 1.0e-12)
            )
        record.metrics = {
            "relative_l2_post_training": relative_l2,
            "reference_available": problem.reference_available
            or external_reference is not None,
            "reference_kind": (
                problem.reference_kind
                if external_reference is None
                else "validated external grid"
            ),
            "reference_used_during_training": False,
            "final_monitor": record.history[-1]["monitor"],
            "final_train": record.history[-1]["train"],
            "final_generalization_gap": record.history[-1]["generalization_gap"],
            "final_generalization_gap_bound": record.history[-1][
                "generalization_gap_bound"
            ],
            "leakage": asdict(audit_manager.leakage_report()),
            "provenance_records": len(audit_manager.provenance()),
            "purpose": "smoke/sanity only; not a scientific result",
        }
        record.sample_hashes = audit_manager.sample_hashes()
        record.optimizer_steps = config.steps
        record.ledger = ledger.to_dict()
        record.finish("completed")
    except BudgetExceeded:
        record.ledger = ledger.to_dict()
        record.fail()
        record.status = "budget_exceeded"
        path = record.write_immutable(config.output) if write_result else None
        raise
    except Exception:
        record.ledger = ledger.to_dict()
        record.fail()
        path = record.write_immutable(config.output) if write_result else None
        raise
    path = record.write_immutable(config.output) if write_result else None
    return TrainingOutcome(model=model, result=record, path=path)
