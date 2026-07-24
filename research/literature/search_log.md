# Literature search log

Timestamp: 2026-07-24 (Asia/Calcutta)  
Scope status: **non-exhaustive seed and collision screen; not a completed systematic review**

## Sources actually queried in this implementation session

- Web search across indexed scholarly pages using exact and conceptual queries.
- arXiv abstract/metadata pages, including 2404.07662, 2405.14369,
  2408.11104, 2410.13228, 2603.06761, 2603.19165, 2604.01835,
  2605.03542, 2605.30910, and 2607.15560.
- Official NeurIPS proceedings pages for RoPINN and PINNacle.
- Official ICLR proceedings pages for PINNACLE point selection and ConFIG.
- PMLR page for the residual-based error-bound paper.
- MERL project page/GitHub link for AB-PINNs.

The environment did not provide authenticated Scopus or Web of Science access.
No claim of exhaustiveness is made. Metadata-only screening is never treated as
full-text verification.

## Exact/concept queries

1. `PINN holdout residual validation set cross fitting audit residual generalization controller`
2. `physics informed neural network train test residual gap overfitting holdout physics residual`
3. `PINN validation gradient select loss objective strong weak residual cost aware allocation`
4. `2026 PINN adaptive residual weak form cross fitted audit controller`
5. `RoPINN Region Optimized Physics Informed Neural Networks`
6. `PINNACLE PINN adaptive point selection NeurIPS 2024`
7. `ConFIG physics informed neural networks gradient conflict`
8. `AB-PINN adaptive basis physics informed neural networks`
9. Exact searches for gPINN, R3, ReLoBRaLo, stochastic variational PINNs,
   residual certification, and the prompt-supplied 2026 arXiv identifiers.

## Saturation status

The prompt requires two snowball rounds after the last new near-neighbor family.
That threshold has **not** been completed. The current corpus is a reproducible
seed for the next review stage. Before a paper claim is approved:

1. query Crossref, OpenAlex, Semantic Scholar, arXiv, OpenReview, and publisher
   indexes programmatically;
2. perform two recorded backward/forward snowball rounds;
3. extract page/section pointers from every Level-2-or-closer full text;
4. deduplicate by DOI, preprint ID, normalized title, and author/year;
5. log inaccessible full texts separately; and
6. have a second reviewer adjudicate collision levels.

## VARA-PINN workspace check

The Git workspace contained only `.git` at inspection time. No local VARA-PINN
repository, manuscript, supplement, or experiment package was available. CAGE's
allocator therefore explicitly bans rollback, regional transfer, and
region-by-variable multipliers, but a full non-overlap audit remains pending if
VARA material is later supplied.

