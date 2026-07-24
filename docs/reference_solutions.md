# Reference-solution policy

Analytic smoke references are differentiated independently and checked against
their PDE residuals. They are imported only by post-training metrics and the
`benchmark verify-references` command.

Numerical references must include:

- solver name/version and complete input;
- at least two grid/time resolutions;
- convergence table and tolerance;
- boundary/conservation checks;
- raw file checksum and generation timestamp;
- a loader that validates shape, domain, units, and checksum.

The checked-in registry marks unresolved references as blocked. Cavity runs need
validated fields and Ghia-style centerlines; a coarse web image or digitized plot
is not a field reference. Burgers/Allen–Cahn discovery runs need independently
converged spectral or finite-volume data before accuracy metrics are enabled.

Reference arrays may not be imported from `training`, `sampling`, `audit`,
`controller`, or `enforcement`. The source audit test searches those modules for
forbidden imports and keys.

