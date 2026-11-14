"""Log-link quasi-likelihood mean model with working variance proportional to mu^2.

Only the conditional mean and this working variance are specified (preregistration, section 8).
The error is not assumed to follow any named probability distribution.
"""

from dataclasses import dataclass

import numpy as np

MAX_ITERATIONS = 100
TOLERANCE = 1e-8
MAX_HALVINGS = 60
# A step that changes the objective by less than rounding error does not count as increasing it.
ROUNDING = 1e-12


@dataclass(frozen=True)
class ScaleFit:
    coefficients: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    lows: np.ndarray
    highs: np.ndarray
    converged: bool
    iterations: int
    reason: str = ""


def _objective(design: np.ndarray, errors: np.ndarray, coefficients: np.ndarray) -> float:
    linear = design @ coefficients
    return float((linear + errors * np.exp(-linear)).sum())


def fit_scale_model(predictors: np.ndarray, errors: np.ndarray) -> ScaleFit:
    """Solve sum x_i (E_i / mu_i - 1) = 0 by Fisher scoring with step halving, unpenalized."""
    predictors = np.asarray(predictors, float)
    errors = np.asarray(errors, float)
    means = predictors.mean(axis=0)
    sds = predictors.std(axis=0, ddof=1)
    lows = predictors.min(axis=0)
    highs = predictors.max(axis=0)

    def failed(reason: str, iterations: int = 0) -> ScaleFit:
        empty = np.full(predictors.shape[1] + 1, np.nan)
        return ScaleFit(empty, means, sds, lows, highs, False, iterations, reason)

    if not (np.isfinite(predictors).all() and np.isfinite(errors).all()):
        return failed("non-finite input")
    if (sds == 0).any():
        return failed("zero-SD predictor")
    if errors.mean() <= 0:
        return failed("all errors are zero")

    design = np.column_stack([np.ones(len(errors)), (predictors - means) / sds])
    information = design.T @ design
    if np.linalg.matrix_rank(information) < design.shape[1]:
        return failed("rank-deficient predictors")

    coefficients = np.zeros(design.shape[1])
    coefficients[0] = np.log(errors.mean())
    with np.errstate(over="ignore", invalid="ignore"):
        current = _objective(design, errors, coefficients)
        for iteration in range(1, MAX_ITERATIONS + 1):
            mean = np.exp(design @ coefficients)
            step = np.linalg.solve(information, design.T @ (errors / mean - 1))
            allowance = ROUNDING * max(abs(current), 1.0)
            for _ in range(MAX_HALVINGS):
                candidate = _objective(design, errors, coefficients + step)
                if np.isfinite(candidate) and candidate <= current + allowance:
                    break
                step = step / 2
            else:
                return failed("no acceptable step", iteration)
            coefficients = coefficients + step
            current = candidate
            if np.abs(step).max() < TOLERANCE:
                return ScaleFit(coefficients, means, sds, lows, highs, True, iteration)
    return failed("iteration limit", MAX_ITERATIONS)


def predict_scale(fit: ScaleFit, predictors: np.ndarray) -> np.ndarray:
    """Predicted mean error. Values are clipped to the fitting-set range before standardizing."""
    clipped = np.clip(np.asarray(predictors, float), fit.lows, fit.highs)
    design = np.column_stack([np.ones(len(clipped)), (clipped - fit.means) / fit.sds])
    return np.exp(design @ fit.coefficients)
