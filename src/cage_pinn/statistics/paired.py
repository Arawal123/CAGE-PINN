from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


def bca_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.median,
    *,
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("BCa requires a one-dimensional sample with at least 3 values")
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("Invalid bootstrap configuration")
    observed = float(statistic(values))
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(resamples, values.size))]
    boot = np.asarray([statistic(sample) for sample in samples], dtype=float)
    proportion = np.clip(np.mean(boot < observed), 1.0 / (2 * resamples), 1 - 1.0 / (2 * resamples))
    bias = stats.norm.ppf(proportion)
    jackknife = np.asarray(
        [statistic(np.delete(values, index)) for index in range(values.size)], dtype=float
    )
    jack_mean = jackknife.mean()
    numerator = np.sum((jack_mean - jackknife) ** 3)
    denominator = 6.0 * np.sum((jack_mean - jackknife) ** 2) ** 1.5
    acceleration = numerator / denominator if denominator > 0 else 0.0
    alpha = (1.0 - confidence) / 2.0

    def corrected_probability(probability: float) -> float:
        z = stats.norm.ppf(probability)
        adjusted = stats.norm.cdf(
            bias + (bias + z) / (1.0 - acceleration * (bias + z))
        )
        return float(np.clip(adjusted, 0.0, 1.0))

    low_q = corrected_probability(alpha)
    high_q = corrected_probability(1.0 - alpha)
    return float(np.quantile(boot, low_q)), float(np.quantile(boot, high_q))


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    raw = np.asarray(p_values, dtype=float)
    order = np.argsort(raw)
    adjusted_sorted = np.maximum.accumulate(
        (len(raw) - np.arange(len(raw))) * raw[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


@dataclass(frozen=True)
class PairedAnalysis:
    pairs: int
    median_log_ratio: float
    median_ratio: float
    geometric_mean_ratio: float
    bca_low_log: float
    bca_high_log: float
    wilcoxon_statistic: float
    wilcoxon_p: float
    probability_cage_better: float
    wins: int
    ties: int
    losses: int
    rope: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def analyze_paired(
    cage_error: np.ndarray,
    baseline_error: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
    rope: float = 0.05,
    bootstrap_seed: int = 0,
) -> PairedAnalysis:
    cage = np.asarray(cage_error, dtype=float)
    baseline = np.asarray(baseline_error, dtype=float)
    if cage.shape != baseline.shape or cage.ndim != 1 or cage.size < 3:
        raise ValueError("Paired analysis requires equal vectors with at least 3 pairs")
    if np.any(cage < 0) or np.any(baseline < 0):
        raise ValueError("Errors cannot be negative")
    log_ratio = np.log((cage + epsilon) / (baseline + epsilon))
    low, high = bca_interval(log_ratio, seed=bootstrap_seed)
    try:
        wilcoxon = stats.wilcoxon(log_ratio, zero_method="pratt")
        statistic, p_value = float(wilcoxon.statistic), float(wilcoxon.pvalue)
    except ValueError:
        statistic, p_value = 0.0, 1.0
    ratio = np.exp(log_ratio)
    lower = 1.0 - rope
    upper = 1.0 + rope
    return PairedAnalysis(
        pairs=int(cage.size),
        median_log_ratio=float(np.median(log_ratio)),
        median_ratio=float(np.exp(np.median(log_ratio))),
        geometric_mean_ratio=float(np.exp(np.mean(log_ratio))),
        bca_low_log=low,
        bca_high_log=high,
        wilcoxon_statistic=statistic,
        wilcoxon_p=p_value,
        probability_cage_better=float(np.mean(log_ratio < 0)),
        wins=int(np.sum(ratio < lower)),
        ties=int(np.sum((ratio >= lower) & (ratio <= upper))),
        losses=int(np.sum(ratio > upper)),
        rope=rope,
    )
