"""Case-level bootstrap (Section 11.9; method fixed in docs/STAGE2_PROTOCOL.md).

TODO(Stage 7): patient-level cluster bootstrap for MSWAL.
"""

from dataclasses import dataclass

import numpy as np

UNRELIABLE_DEGENERATE_FRACTION = 0.01


@dataclass(frozen=True)
class PercentileInterval:
    lower: float
    upper: float
    degenerate: int
    resamples: int

    @property
    def unreliable(self) -> bool:
        return self.degenerate > UNRELIABLE_DEGENERATE_FRACTION * self.resamples


def resample_indices(n: int, resamples: int, seed: int) -> np.ndarray:
    """Return a (resamples, n) array of case indices drawn with replacement."""
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def percentile_interval(values: np.ndarray) -> PercentileInterval:
    """2.5th and 97.5th percentiles. Non-finite values are counted and left out, not replaced."""
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    degenerate = int(values.size - finite.size)
    if finite.size == 0:
        return PercentileInterval(float("nan"), float("nan"), degenerate, values.size)
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return PercentileInterval(float(lower), float(upper), degenerate, values.size)
