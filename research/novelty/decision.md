# Novelty-gate decision

Decision date: 2026-07-24  
Verdict: **provisional pass for implementation; publication claim withheld**

No Level 0 or Level 1 collision was found in the completed seed and conceptual
screen. Several Level 2 collisions are substantial:

- *PINNs Failure Modes are Overfitting* directly targets collocation overfitting
  and uses residual-gradient regularization.
- gPINN/Double-PINN establish the J primitive.
- stochastic variational and weak PINNs establish the W primitive.
- RoPINN targets generalization through region sampling.
- PINNACLE dynamically allocates among training point types.
- residual certification uses unseen residual samples and confidence/error bounds.
- ConFIG and second-order PINN work establish gradient-alignment/preconditioning
  ideas.

Accordingly, the repository may test the combined controlling question, but may
not claim that S, J, W, off-grid diagnostics, CVaR, cross-validation, gradient
alignment, or cost-aware optimization are individually new.

## Approved wording

> To the best of the systematic search completed at submission time, we did not
> identify a prior PINN method combining rotating cross-fitted physics audits,
> a non-influential monitor fold, and measured-cost allocation among strong,
> residual-slope, and weak enforcement forms. The present repository's search is
> currently incomplete, so this sentence is not yet approved for a manuscript.

## Gate conditions before submission

- Finish two post-saturation snowball rounds.
- Review every Level-2 paper in full with page/section pointers.
- Search adjacent adaptive regularization, bilevel optimization, validation-
  gradient, algorithm-selection, and sequential-testing literature outside PINNs.
- Adjudicate VARA-PINN if supplied.
- Obtain independent reviewer sign-off.

If any Level 0/1 source is found, stop the flagship claim and execute the
three-pivot collision protocol before confirmatory runs.

