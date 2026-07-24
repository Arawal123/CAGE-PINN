# Troubleshooting

## JAX has no GPU in Colab

Run `import jax; print(jax.devices())`. Do not force-install a CPU-only `jaxlib`
over Colab's preinstalled accelerator build. Restart the runtime after changing
JAX packages.

## Float64 is unexpectedly disabled

Launch with `JAX_ENABLE_X64=True` or use a `float64` run config before creating
JAX arrays. Precision is recorded in every result.

## Budget exceeded

This is a recorded outcome, not a retry hint. Increase the preregistered budget
only in discovery, or reduce atom/audit sizes before freezing confirmatory work.

## Leakage assertion

Inspect coordinate hashes and seed derivation. Never bypass the assertion.
Refreshing a fold with a new generation is the safe response.

## Official baseline unavailable

Follow `docs/baseline_fidelity.md`, create an isolated environment, and pin the
official commit. Do not rename a native approximation as the official method.

## Compilation dominates a tiny smoke run

Expected for JAX. Smoke verifies wiring only. Wall-clock comparisons need
symmetric warm-up and compilation accounting on the target device.

