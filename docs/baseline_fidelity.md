# Baseline fidelity ledger

Native clean-room executable baselines:

| Name | Parent reduction | Current status |
|---|---|---|
| vanilla | direct | executable |
| uniform resampling | vanilla with fresh points | executable |
| RAR-D-style | vanilla with residual-proportional candidate sampling | executable diagnostic; paper-fidelity reproduction pending |
| gPINN | static S/J schedule | executable diagnostic |
| static S/J/W | no-controller CAGE atom bank | executable ablation |
| weak-PINN | static S/W schedule | executable diagnostic |

Tested primitives, not yet approved as paper-identical baselines:
ReLoBRaLo-style relative weighting, SA-style point weights, and two-loss
ConFIG-style gradient composition.

Official adapters are intentionally unavailable until their isolated repositories
are installed and pinned:

- RoPINN: <https://github.com/thuml/RoPINN>
- AB-PINN: <https://github.com/merlresearch/ab-pinns>
- PINNACLE point selection: official source linked from the ICLR paper.

`require_external_baseline` raises if an official dependency is absent. It never
substitutes the native approximation. Before publication, record commit, license,
environment, published benchmark tolerance, deviations, hyperparameter search,
and equal-compute charges for every baseline.

The native adaptive-basis backbone is a parity/composability test vehicle. It is
not labeled an official AB-PINN reproduction.

