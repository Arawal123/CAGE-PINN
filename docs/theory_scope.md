# Theory scope

Proven in code/tests:

- bounded residual score lies in \([0,1]\);
- fixed-schedule union correction is applied to the declared number of audits
  and channels;
- exact token rounding and bounded-simplex feasibility;
- first-order sign convention for positive gradient alignment;
- stop-gradient residual scales.

Standard assumptions, not proven by this repository:

- independent audit draws from the target coordinate distribution;
- differentiability/smoothness needed by the PDE and J atom;
- a positive PDE stability or coercivity constant connecting residual and
  boundary norms to solution error;
- local smoothness of the selector objective for one-step Taylor prediction.

Empirical hypotheses:

- monitor audit risk tracks solution error closely enough to be useful;
- selector-gradient utility predicts short-horizon monitor improvement;
- dynamic S/J/W compute allocation beats a matched static mixture.

No solution-error guarantee is claimed for shocks, entropy selection, nonlinear
non-coercive systems, chaotic long-time dynamics, or an unvalidated weak witness.

