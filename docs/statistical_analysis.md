# Statistical analysis plan

Runs are paired by problem instance, severity, backbone, seed, precision, and
budget. For error \(E\),

\[
d_{p,s}=\log\frac{E_{\mathrm{CAGE},p,s}+\epsilon}
{E_{\mathrm{baseline},p,s}+\epsilon}.
\]

Negative values favor CAGE. The preregistered report includes paired median and
geometric-mean ratios, BCa bootstrap intervals, Wilcoxon signed-rank tests when
their assumptions are reasonable, Holm adjustment within declared hypothesis
families, probability of superiority, and win/tie/loss using a ±5% ROPE.

A hierarchical model with PDE-level random effects is optional until sufficient
PDE/seed coverage exists; its prior and convergence diagnostics must be frozen
before confirmatory outcomes are inspected. Failure/OOM/timeout rates are
reported separately and never dropped.

Discovery uses at least five paired seeds. Confirmatory manifests request ten.
Any lower count for an expensive case must be justified in a signed manifest
before results exist.

