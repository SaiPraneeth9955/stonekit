"""Secondary split-conformal ranges and global limits (preregistration, section 13).

Ranges are for the supplied reference. Marginal coverage is targeted only if stones are
exchangeable. No size-conditional or physical-truth coverage is provided or claimed.
"""

from dataclasses import dataclass

import numpy as np

from stonekit.reliability.evaluation import Rows, design_matrix
from stonekit.reliability.model import ScaleFit, fit_scale_model, predict_scale

N_OUTER_FOLDS = 10
LOG_LIMIT_Z = 1.645
ML_LIMIT_Z = 1.96
ALPHA = 0.10
TARGET_NUMERATOR = 9
TARGET_DENOMINATOR = 10


def conformal_order(n_calibration: int) -> int | None:
    """The 1-based order statistic ceil((n + 1) * 0.90). None if it exceeds the sample."""
    order = -(-(n_calibration + 1) * TARGET_NUMERATOR // TARGET_DENOMINATOR)
    return order if order <= n_calibration else None


def conformal_quantile(ratios: np.ndarray) -> float | None:
    order = conformal_order(len(ratios))
    return None if order is None else float(np.sort(ratios)[order - 1])


def interval_score(lower: np.ndarray, upper: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Interval score with alpha = 0.10, on ln R."""
    below = np.maximum(0, lower - target)
    above = np.maximum(0, target - upper)
    return (upper - lower) + (2 / ALPHA) * (below + above)


def log_limits(
    volume: np.ndarray, reference: np.ndarray, new_volume: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nominal 90% limits for ln R from ln(V/R) in training stones. Not a conformal guarantee."""
    ratios = np.log(volume / reference)
    mean, sd = ratios.mean(), ratios.std(ddof=1)
    centre = np.log(new_volume) - mean
    return centre - LOG_LIMIT_Z * sd, centre + LOG_LIMIT_Z * sd


def ml_limits(
    volume: np.ndarray, reference: np.ndarray, new_volume: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Conventional nominal 95% limits for R, in mL. Non-positive lower limits are not clipped."""
    difference = volume - reference
    bias, sd = difference.mean(), difference.std(ddof=1)
    centre = new_volume - bias
    return centre - ML_LIMIT_Z * sd, centre + ML_LIMIT_Z * sd


@dataclass(frozen=True)
class CalibratedScale:
    fit: ScaleFit
    q: float


def calibrate_scale(
    predictors: np.ndarray, errors: np.ndarray, proper: np.ndarray, calibration: np.ndarray
) -> CalibratedScale | None:
    """Fit on the proper subset, and take q from a separate calibration subset. None if invalid."""
    fit = fit_scale_model(predictors[proper], errors[proper])
    if not fit.converged:
        return None
    q = conformal_quantile(errors[calibration] / predict_scale(fit, predictors[calibration]))
    return None if q is None else CalibratedScale(fit, q)


def split_groups(
    stone_id: np.ndarray, training: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Assign one third of unique training stones to calibration, keeping copies together.

    For outer training sets of 90 or 91 stones this gives exactly 30 calibration stones.
    """
    groups = np.unique(stone_id[training])
    n_calibration = round(len(groups) / 3)
    chosen = rng.permutation(groups)[:n_calibration]
    calibration = training & np.isin(stone_id, chosen)
    return training & ~calibration, calibration


def outer_folds(stone_id: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Unstratified random outer folds on unique stones, mapped back to every copy."""
    unique = np.unique(stone_id)
    folds = rng.permutation(len(unique)) % N_OUTER_FOLDS
    return folds[np.searchsorted(unique, stone_id)]


def range_partition(
    rows: Rows,
    assumptions: tuple[str, ...],
    full_predictors: tuple[str, ...],
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray]] | None:
    """One random 10-fold repeat. Returns held-out bounds by method. None if any step fails."""
    stone_id = rows.stone_id
    fold = outer_folds(stone_id, rng)
    full = design_matrix(rows, full_predictors)
    size = design_matrix(rows, ("x1",))
    reference = rows.columns["reference_volume_ml"]
    bounds = {}
    for assumption in assumptions:
        for method in ("full", "size_scale", "log_limits", "ml_limits"):
            bounds[f"{method}_{assumption}"] = {
                "lower": np.empty(len(rows)),
                "upper": np.empty(len(rows)),
            }
    for current in np.unique(fold):
        held_out = fold == current
        training = ~held_out
        proper, calibration = split_groups(stone_id, training, rng)
        for assumption in assumptions:
            errors = rows.columns[f"E_{assumption}"]
            volume = rows.columns[f"volume_{assumption}"]
            for method, predictors in (("full", full), ("size_scale", size)):
                calibrated = calibrate_scale(predictors, errors, proper, calibration)
                if calibrated is None:
                    return None
                half_width = calibrated.q * predict_scale(calibrated.fit, predictors[held_out])
                centre = np.log(volume[held_out])
                target = bounds[f"{method}_{assumption}"]
                target["lower"][held_out] = centre - half_width
                target["upper"][held_out] = centre + half_width
            for method, limits in (("log_limits", log_limits), ("ml_limits", ml_limits)):
                lower, upper = limits(volume[training], reference[training], volume[held_out])
                target = bounds[f"{method}_{assumption}"]
                target["lower"][held_out] = lower
                target["upper"][held_out] = upper
    return bounds


def summarize_bounds(
    bounds: dict[str, np.ndarray], reference: np.ndarray, method: str, subset=None
) -> dict:
    """Coverage and width, always together. Interval score applies to log-scale ranges only."""
    subset = np.ones(len(reference), dtype=bool) if subset is None else subset
    lower, upper = bounds["lower"][subset], bounds["upper"][subset]
    if method.startswith("ml_limits"):
        target = reference[subset]
        return {
            "coverage": float(((lower <= target) & (target <= upper)).mean()),
            "median_width": float(np.median(upper - lower)),
            "width_unit": "mL",
            "mean_interval_score": float("nan"),
            "nonpositive_lower_limits": int((lower <= 0).sum()),
        }
    target = np.log(reference[subset])
    return {
        "coverage": float(((lower <= target) & (target <= upper)).mean()),
        "median_width": float(np.median(upper - lower)),
        "width_unit": "ln mL",
        "mean_interval_score": float(interval_score(lower, upper, target).mean()),
        "nonpositive_lower_limits": 0,
    }
