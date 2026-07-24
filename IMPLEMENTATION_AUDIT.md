# Implementation audit — 2026-07-24

## Delivered

- JAX/Equinox/Optax implementation of vanilla, fixed-partition XPINN, and
  adaptive-basis partition-of-unity parents.
- Cross-fitted selector/monitor/calibration folds with rotation, refresh,
  coordinate hashes, provenance, and leakage assertions.
- Stop-gradient robust channel scales; bounded residual scores; empirical CVaR;
  fixed-schedule union-corrected upper reporting bound.
- Strong (S), residual-slope (J), and seeded multi-scale weak (W) atoms.
- Exact and Rademacher-sketched selector-gradient utility per calibrated cost.
- Bounded-simplex allocation, exploration/caps/rate limits, exact token rounding,
  and measured-compute fair queuing.
- Full cost ledger, immutable run records, device calibration command, raw-only
  analysis/report generation, and claim gate.
- Executable clean-room diagnostics for vanilla, uniform resampling, RAR-D-style
  sampling, gPINN, ReLoBRaLo-style weighting, self-adaptive point weighting,
  two-gradient ConFIG-style composition, weak-PINN, and static S/J/W.
- Strict adapters/gates for official RoPINN, PINNACLE, and AB-PINN environments.
- 18 benchmark declarations and 18 training adapters. Thirteen analytic or
  manufactured references are derivative-verified. Burgers, Allen–Cahn, and
  three cavity cases require external validated references.
- Literature CSV/Parquet corpus, search log, collision matrix, novelty decision,
  preregistered ablation/sensitivity files, manuscript scaffold, CI, Dockerfile,
  uv lock, and Colab commands.

## Verification actually performed

```text
ruff check .                         PASS
mypy src/cage_pinn                   PASS (46 source files)
pytest -q                            PASS: 54 passed, 10 skipped
uv lock --check                      PASS: 52 packages resolved
benchmark verify-references          PASS: 13 analytic/manufactured PDEs
literature build                     PASS: 24 records, CSV and Parquet
novelty audit                        provisional implementation pass
budget calibrate (tiny CPU check)    PASS: S/J/W records serialized
paper claim check                    correctly BLOCKS publication build
Docker build                         NOT RUN: local Docker daemon unavailable
```

The ten skips are two reference tests for each of Allen–Cahn, Burgers, and
cavity \(Re=100,1000,3200\). They are intentional scientific gates.

Persistent lightweight raw diagnostics completed locally:

- vanilla+CAGE (two steps, including the final weak/stratified audit path);
- XPINN+CAGE (one step);
- adaptive-basis+CAGE (one step);
- ReLoBRaLo-style, self-adaptive, and ConFIG-style diagnostics.

The final vanilla+CAGE record spent 2,896 symbolic tokens, passed its schema and
leakage audit, and logged no training-time reference access. These are wiring
checks, not accuracy evidence.

No published baseline reproduction and no discovery/confirmatory campaign was
claimed as completed.

## Novelty decision

The implementation gate provisionally passes because the seed screen found no
Level 0/1 exact combination. Closest Level-2 collisions include residual-
overfitting/double-backpropagation work, gPINN, stochastic variational PINNs,
RoPINN, PINNACLE point selection, ConFIG/gradient alignment, and residual
certification. Publication novelty remains unapproved until full-text extraction
and two post-saturation snowball rounds are completed.

## Experiment readiness

The discovery plan contains 420 explicit problem/seed/method-backbone runs and is
blocked only on validated Burgers and Allen–Cahn references.

The confirmatory plan contains 1,820 explicit runs and is blocked on validated
Burgers, Allen–Cahn, cavity \(Re=100\), and cavity \(Re=1000\) references. At the
manifest wall cap this is an upper bound of 7,280 accelerator-hours and
9.1 billion symbolic AD tokens. Actual estimates must be replaced by device
calibration and pilot timing before resource allocation.

## Currently supported claims

- The tested core enforces hashed learner/audit coordinate separation.
- The tested scheduler and ledger conserve declared symbolic token budgets.
- Thirteen analytic/manufactured references satisfy their coded residuals within
  the tested float64 tolerance.
- The repository blocks unsupported manuscript results and novelty language.

No claim of improved solution accuracy, controller utility, broad robustness,
statistical superiority, or publication novelty is supported yet.

