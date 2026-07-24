# Equal-compute protocol

Primary publication comparisons use equal end-to-end wall time on an exclusive,
identical accelerator. Calibrated AD tokens are co-primary/key secondary, and
equal optimizer steps are labeled secondary.

The symbolic ledger counts:

- residual evaluations by declared derivative order;
- residual-input Jacobians;
- weak quadrature residuals;
- selector forward/backward passes;
- all candidate-atom gradients used by the controller;
- monitor/calibration forwards;
- boundary, initial, and interface gradients;
- method-specific preprocessing and compilation.

The `CostModel` weights must be calibrated on the target device before a real
study. Calibration records symbolic counts and median device time after symmetric
warm-up. A Colab GPU type may change between sessions; runs from different SKUs
must not be pooled as equal-wall-clock comparisons.

Candidate-gradient work is never free. Failed, timed-out, divergent, and OOM
runs remain outcomes. Raw ledgers are immutable.

