# Audit leakage contract

The learner, selector, monitor, and calibration roles use independently derived
random keys and exact coordinate hashes.

Forbidden:

- selector or monitor coordinates in the learner loss;
- selector loss directly contributing to a parameter update;
- monitor metrics changing allocation, learning rate, stopping, checkpoint
  selection, or hyperparameters;
- reuse of one selector forever while calling it independent;
- hidden access to reference values from training modules;
- witness seeds/quadrature patches shared across learner and audit roles inside
  the prohibited window.

`AuditFoldManager.assert_no_leakage()` runs at construction, after every learner
registration, and after refresh. The monitor is still statistically influenced
by prior decisions once its fold has served as selector; rotation history and
generation must therefore be considered in analysis.

