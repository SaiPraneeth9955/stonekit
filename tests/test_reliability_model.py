"""The quasi-likelihood mean model on hand-made values."""

import numpy as np
import pytest

from stonekit.reliability.model import fit_scale_model, predict_scale

X = np.array(
    [[1.0, 4.0], [2.0, 1.0], [3.0, 7.0], [4.0, 2.0], [5.0, 9.0], [6.0, 3.0], [7.0, 8.0], [8.0, 5.0]]
)


def test_recovers_an_exact_mean():
    errors = np.exp(0.2 + 0.3 * X[:, 0] - 0.1 * X[:, 1])
    fit = fit_scale_model(X, errors)
    assert fit.converged
    assert predict_scale(fit, X) == pytest.approx(errors, rel=1e-6)


def test_solution_satisfies_the_estimating_equation():
    errors = np.array([0.1, 0.7, 0.2, 1.4, 0.05, 0.9, 0.3, 0.6])
    fit = fit_scale_model(X, errors)
    assert fit.converged
    design = np.column_stack([np.ones(len(X)), (X - fit.means) / fit.sds])
    score = design.T @ (errors / predict_scale(fit, X) - 1)
    assert np.abs(score).max() < 1e-6


def test_standardizing_uses_sample_sd():
    fit = fit_scale_model(X, np.arange(1.0, 9.0))
    assert fit.sds == pytest.approx(X.std(axis=0, ddof=1))


def test_zero_errors_are_allowed_without_a_floor():
    errors = np.array([0.0, 0.7, 0.2, 1.4, 0.0, 0.9, 0.3, 0.6])
    assert fit_scale_model(X, errors).converged


def test_invalid_inputs_do_not_converge():
    errors = np.arange(1.0, 9.0)
    constant = X.copy()
    constant[:, 1] = 3.0
    assert fit_scale_model(constant, errors).reason == "zero-SD predictor"
    assert fit_scale_model(X, np.zeros(8)).reason == "all errors are zero"
    bad = X.copy()
    bad[0, 0] = np.nan
    assert fit_scale_model(bad, errors).reason == "non-finite input"
    collinear = np.column_stack([X[:, 0], 2 * X[:, 0]])
    assert fit_scale_model(collinear, errors).reason == "rank-deficient predictors"


def test_prediction_is_positive_and_clips_to_the_fitting_range():
    fit = fit_scale_model(X, np.array([0.1, 0.7, 0.2, 1.4, 0.05, 0.9, 0.3, 0.6]))
    far = np.array([[1000.0, -1000.0]])
    edge = np.array([[X[:, 0].max(), X[:, 1].min()]])
    assert predict_scale(fit, far) == pytest.approx(predict_scale(fit, edge))
    assert (predict_scale(fit, X) > 0).all()
