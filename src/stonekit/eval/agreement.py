"""Bland-Altman, ICC (Section 11.9; definitions fixed in docs/STAGE2_PROTOCOL.md)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pingouin as pg

LIMIT_FACTOR = 1.96


@dataclass(frozen=True)
class BlandAltman:
    bias: float
    sd: float
    lower_limit: float
    upper_limit: float


def icc_a1(measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """ICC(A,1) for two raters. Accepts (n,) or (resamples, n); undefined values are NaN.

    Two-way random effects, absolute agreement, single measurement.
    """
    data = np.stack([np.asarray(measured, float), np.asarray(reference, float)], axis=-1)
    n = data.shape[-2]
    k = data.shape[-1]
    grand = data.mean(axis=(-2, -1), keepdims=True)
    ss_targets = k * ((data.mean(axis=-1, keepdims=True) - grand) ** 2).sum(axis=(-2, -1))
    ss_raters = n * ((data.mean(axis=-2, keepdims=True) - grand) ** 2).sum(axis=(-2, -1))
    ss_total = ((data - grand) ** 2).sum(axis=(-2, -1))
    ms_targets = ss_targets / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = (ss_total - ss_targets - ss_raters) / ((n - 1) * (k - 1))

    numerator = ms_targets - ms_error
    denominator = ms_targets + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def pingouin_icc_a1(measured: np.ndarray, reference: np.ndarray) -> tuple[float, str]:
    """Return pingouin's ICC(A,1) and its interval, which it rounds to 2 decimals."""
    n = len(measured)
    long = pd.DataFrame(
        {
            "stone": np.tile(np.arange(n), 2),
            "rater": ["method"] * n + ["reference"] * n,
            "value": np.concatenate([measured, reference]),
        }
    )
    table = pg.intraclass_corr(data=long, targets="stone", raters="rater", ratings="value")
    row = table.set_index("Type").loc["ICC(A,1)"]
    return float(row["ICC"]), str(row["CI95"])


def bland_altman(measured: np.ndarray, reference: np.ndarray) -> BlandAltman:
    """Differences are measured minus reference. SD uses n - 1."""
    difference = np.asarray(measured, float) - np.asarray(reference, float)
    bias = float(difference.mean())
    sd = float(difference.std(ddof=1))
    return BlandAltman(bias, sd, bias - LIMIT_FACTOR * sd, bias + LIMIT_FACTOR * sd)
