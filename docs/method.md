# Formal method specification

## 1. PDE and residual channels

Let \(u_\theta:\Omega\to\mathbb R^d\) be a differentiable backbone and let

\[
r_m(z;\theta)=\mathcal N_m[u_\theta](z)-f_m(z),\quad m=1,\ldots,M.
\]

Boundary, initial, and interface channels are \(b_j(z;\theta)\). Every
`PDEProblem` declares channel names, derivative order, domain, smoothness, and
weak-form applicability. Dimensional channels must be nondimensionalized by the
problem definition before CAGE sees them; CAGE does not average raw quantities
with incompatible units.

Reference solutions are exposed only by the post-training metric path. Training,
sampling, scale tracking, audit, allocation, and checkpoint diagnostics receive
only the problem operators and coordinates.

## 2. Stop-gradient robust normalization

For learner residuals in control interval \(t\), define

\[
\widehat m_{m,t}=\operatorname{median}_{z\in T_t}|r_m(z;\theta_t)|,\qquad
s_{m,t}=\beta s_{m,t-1}+(1-\beta)\max(\widehat m_{m,t},\epsilon).
\]

The first update uses \(s_{m,0}=\max(\widehat m_{m,0},\epsilon)\).
The implementation applies `stop_gradient` both before the median and when
normalizing:

\[
\widetilde r_m(z;\theta)=
\frac{r_m(z;\theta)}
{\operatorname{sg}(s_{m,t})+\epsilon}.
\]

Thus \(\partial s_{m,t}/\partial\theta=0\). Under residual-unit scaling
\(r'_m=c_m r_m\), the median/EMA scales by \(|c_m|\), so
\(\widetilde r'_m=\operatorname{sign}(c_m)\widetilde r_m\) after scale
initialization; all squared scores are invariant. Lag and \(\epsilon\) cause a
documented transient approximation rather than exact instantaneous invariance.

Raw and normalized channel statistics are logged.

## 3. Cross-fitted audit roles

At least three independently sampled coordinate sets exist:

- selector \(A_t^{(s)}\): may change the next atom allocation but is never a
  direct loss term;
- monitor \(A_t^{(m)}\): reporting only; it changes neither allocation nor
  parameters;
- calibration \(A_t^{(c)}\): confidence calibration or an untouched check.

Roles rotate on a fixed control schedule. A fold is refreshed after a fixed
number of selector uses. Exact float64 coordinate hashes are checked against
all learner batches in a prohibited rolling window and against all other audit
folds. Random witnesses use independent derived seeds. Each coordinate has a
source, fold, generation, local ID, step where relevant, and SHA-256 hash.

Rotation does not make a previously selected fold instantly independent. Logs
therefore retain its influence count/generation; monitor analyses must stratify
by whether a fold has influenced earlier decisions.

## 4. Tail audit risk

The bounded score

\[
\ell_m(z;\theta)=\frac{\widetilde r_m(z;\theta)^2}
{1+\widetilde r_m(z;\theta)^2}\in[0,1]
\]

prevents a single unbounded AD residual from invalidating bounded-loss
concentration. For \(n\) audit scores sorted increasingly, empirical upper-tail
CVaR is the mean of scores from index \(\lfloor qn\rfloor\) through \(n-1\).
The differentiable selector uses the empirical quantity.

For reporting at a preregistered finite set of \(T\) audit times and \(M\)
channels, the code uses

\[
U_{m,t}=\min\left\{1,\widehat{\mathrm{CVaR}}_{q,m,t}
+\frac{1}{1-q}\sqrt{\frac{\log(2TM/\delta)}{2n}}\right\}.
\]

This is intentionally conservative. Its scope is independent bounded scores and
the fixed finite schedule with a union correction. It is **not** an anytime
confidence sequence, a distribution-free solution certificate, or valid after
data-dependent changes to audit sampling.

Channel risks are aggregated with

\[
A_t(\theta)=\tau_r\log\sum_m \exp(U_{m,t}/\tau_r),
\]

which approaches a worst-channel maximum rather than hiding one equation in an
average. The logged gap is

\[
G_t=\log\frac{A_t^{\mathrm{audit}}+\epsilon}
{A_t^{\mathrm{train}}+\epsilon}.
\]

## 5. Enforcement atoms

### S: fresh strong coverage

Fresh independently keyed learner points minimize

\[
L_S=\frac1{|T_t|M}\sum_{z,m}\widetilde r_m(z;\theta)^2.
\]

The current executable sampler is global uniform exploration. Benchmark-specific
strata can be added, but audit hashes are always excluded.

### J: residual-input slope

For smooth problems with an acceptable AD order,

\[
L_J=\frac1{|T_t|M}\sum_{z,m}
\|D_z\widetilde r_m(z;\theta)\|_2^2.
\]

This is a known gPINN/Double-PINN-family primitive. The scale is detached. The
implementation rejects non-smooth problems, caps extreme numerical Jacobians,
and reports the added derivative order. Shock problems require a separate,
mathematically justified policy.

### W: stochastic multi-scale weak witnesses

For center \(c\), radius \(h\), compact bump
\(\phi(\xi)=(1-\|\xi\|^2)_+^2\), and seeded quadrature,

\[
L_W=\mathbb E_{c,h,\phi}\left\|
|B_h|\frac1Q\sum_{q=1}^Q
\widetilde r(c+h\xi_q;\theta)\phi(\xi_q)
\right\|_2^2.
\]

The generic implementation does not claim integration-by-parts lowering,
entropy admissibility, or exact \(H^{-1}\) equivalence. PDE-specific weak atoms
must supply boundary terms and proofs before those claims are enabled.

## 6. Audit-gradient utility

At a control time, compute selector gradient \(g_A=\nabla_\theta A_t\) and
candidate learner-atom gradient \(g_c=\nabla_\theta L_c\). Candidate gradients
are controller overhead and do not update parameters. With positive diagonal
preconditioner \(P_t\) and calibrated cost \(\kappa_c\),

\[
v_c=\frac{[\langle g_A,P_tg_c\rangle]_+}
{\kappa_c(\|P_t^{1/2}g_c\|_2+\epsilon)}.
\]

For the descent update \(\theta^+=\theta-\eta P_tg_c\),

\[
A_t(\theta^+)=A_t(\theta)
-\eta\langle g_A,P_tg_c\rangle+O(\eta^2\|P_tg_c\|^2).
\]

The positive inner product therefore predicts a first-order decrease. The code
supports exact flattened products and a seeded Rademacher sketch. NaN/Inf
components become zero. When all utilities are non-positive, the allocator
retains exploration/previous shares.

## 7. Allocation and compute scheduling

CAGE approximately solves

\[
\max_{a\in\Delta}\;
\sum_c a_cv_c+\tau H(a)-\rho\|a-a_t\|_2^2
\]

under applicability masks, floors, caps, and rate limits. Projected ascent uses
a bisection projection onto the bounded simplex. Largest-remainder rounding
conserves every requested control token exactly. Training uses weighted fair
queuing on accumulated **atom compute**, so shares control differentiation cost
rather than only loss coefficients.

Selector gradients, all candidate gradients, monitor forwards, weak quadrature,
boundaries/interfaces, and active-atom training are charged. If the ledger would
exceed its absolute budget, the run fails and records the outcome.

## 8. Backbones

The vanilla parent is an Equinox MLP. XPINN uses fixed subdomain MLPs with
solution and normal-flux jump losses. The adaptive-basis parent is a fixed-count
partition of unity of movable Gaussian windows. CAGE never changes partition,
window count/addition schedule, or parameter budget relative to the paired
parent.

The current adaptive-basis implementation is a clean-room research parent, not
a paper-identical reproduction of the MERL PyTorch AB-PINN. Publication use
requires the fidelity work in `docs/baseline_fidelity.md`.

