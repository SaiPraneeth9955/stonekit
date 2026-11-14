"""Split-conformal intervals for segmentation-derived stone measurements (master plan v2 §G).

All intervals are built on the log scale, so a residual of log(V / V_hat) means a relative error and
intervals are multiplicative: [V_hat * exp(lo), V_hat * exp(hi)].

Methods (U-numbers as in the plan):
- U1 global: one quantile of |log(V/V_hat)| over all calibration stones.
- U2 Mondrian: the same, separately within predicted-size strata, with the pre-specified merge rule for
  strata below a minimum calibration count.
- U4/U5 normalised: score |log(V/V_hat)| / sigma, with sigma a per-stone signal (TTA spread, threshold
  sensitivity) floored at a fixed value chosen on development data.
- U3 CQR: conformalised quantile regression (Romano et al. 2019) of log(V/V_hat) on features.
- U0 Bland-Altman on the log scale: mean +/- z * SD of log ratios (not a conformal method; comparator).

Finite-sample quantile: the ceil((n+1)(1-alpha))-th smallest score. If that rank exceeds n the
interval is unbounded (np.inf), which is a correct conformal answer for a too-small calibration set,
never silently replaced.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score, or inf if rank > n."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    return float(scores[rank - 1]) if rank <= n else math.inf


def log_ratio(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """log(reference / predicted); both must be positive."""
    reference, predicted = np.asarray(reference, float), np.asarray(predicted, float)
    if (reference <= 0).any() or (predicted <= 0).any():
        raise ValueError("log-ratio needs positive values")
    return np.log(reference / predicted)


@dataclass(frozen=True)
class Strata:
    """Predicted-size strata after the merge rule. `edges` are the inner cut points (mm)."""

    edges: tuple[float, ...]
    quantiles: tuple[float, ...]
    counts: tuple[int, ...]
    merged_from: tuple[float, ...]

    def index(self, size: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self.edges), np.asarray(size, float), side="right")


def merge_strata(size: np.ndarray, edges: tuple[float, ...], min_count: int) -> tuple[float, ...]:
    """Pre-specified merge rule (plan §G): while some stratum has fewer than `min_count` calibration
    stones, merge the smallest such stratum into its larger neighbour (the neighbour with more stones;
    the upper neighbour on a tie). Uses calibration sizes only, never outcomes."""
    edges = list(edges)
    size = np.asarray(size, float)
    while edges:
        counts = np.bincount(np.searchsorted(edges, size, side="right"), minlength=len(edges) + 1)
        small = [i for i, c in enumerate(counts) if c < min_count]
        if not small:
            break
        i = min(small, key=lambda j: (counts[j], j))
        if i == 0:
            edges.pop(0)
        elif i == len(counts) - 1:
            edges.pop(-1)
        else:
            edges.pop(i if counts[i + 1] >= counts[i - 1] else i - 1)
    return tuple(edges)


def fit_mondrian(scores: np.ndarray, size: np.ndarray, edges: tuple[float, ...], alpha: float, min_count: int) -> Strata:
    final = merge_strata(size, edges, min_count)
    groups = np.searchsorted(final, np.asarray(size, float), side="right")
    scores = np.asarray(scores, float)
    quantiles = tuple(conformal_quantile(scores[groups == g], alpha) for g in range(len(final) + 1))
    counts = tuple(int((groups == g).sum()) for g in range(len(final) + 1))
    return Strata(final, quantiles, counts, tuple(edges))


def symmetric_interval(predicted: np.ndarray, q: np.ndarray | float, sigma: np.ndarray | float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """[V_hat * exp(-q*sigma), V_hat * exp(q*sigma)]; q may be inf (unbounded above, zero below)."""
    width = np.asarray(q, float) * np.asarray(sigma, float)
    predicted = np.asarray(predicted, float)
    with np.errstate(over="ignore"):
        return predicted * np.exp(-width), predicted * np.exp(width)


def bland_altman_log(log_ratios: np.ndarray, z: float) -> tuple[float, float]:
    """U0 comparator: mean +/- z*SD of calibration log ratios (lower, upper)."""
    r = np.asarray(log_ratios, float)
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    return mean - z * sd, mean + z * sd


def fit_cqr(features: np.ndarray, log_ratios: np.ndarray, alpha: float, fit_index: np.ndarray, calibration_index: np.ndarray):
    """U3: linear quantile regression of the log ratio at alpha/2 and 1-alpha/2 on `features`
    (fitted on `fit_index`), conformalised on `calibration_index`. Returns a predict function giving
    (lower, upper) log-ratio bounds for new feature rows."""
    import statsmodels.api as sm

    x = sm.add_constant(np.asarray(features, float), has_constant="add")
    y = np.asarray(log_ratios, float)
    lo_model = sm.QuantReg(y[fit_index], x[fit_index]).fit(q=alpha / 2)
    hi_model = sm.QuantReg(y[fit_index], x[fit_index]).fit(q=1 - alpha / 2)
    lo, hi = lo_model.predict(x[calibration_index]), hi_model.predict(x[calibration_index])
    q = conformal_quantile(np.maximum(lo - y[calibration_index], y[calibration_index] - hi), alpha)

    def predict(new_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xn = sm.add_constant(np.asarray(new_features, float), has_constant="add")
        return lo_model.predict(xn) - q, hi_model.predict(xn) + q

    return predict


def clopper_pearson(covered: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial CI for an observed coverage proportion."""
    if n == 0:
        return (math.nan, math.nan)
    ci = stats.binomtest(covered, n).proportion_ci(confidence_level=level, method="exact")
    return float(ci.low), float(ci.high)


def cluster_bootstrap_coverage(covered: np.ndarray, cluster: np.ndarray, resamples: int, seed: int, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI for coverage, resampling clusters (scans) with replacement."""
    covered, cluster = np.asarray(covered, bool), np.asarray(cluster)
    ids, inverse = np.unique(cluster, return_inverse=True)
    hits = np.bincount(inverse, weights=covered.astype(float), minlength=len(ids))
    totals = np.bincount(inverse, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(resamples, len(ids)))
    values = hits[draws].sum(axis=1) / np.maximum(totals[draws].sum(axis=1), 1.0)
    tail = (1 - level) / 2
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))
