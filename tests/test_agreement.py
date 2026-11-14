"""ICC and Bland-Altman. The numbers here are hand-made code fixtures."""

import numpy as np
import pytest

from stonekit.eval.agreement import bland_altman, icc_a1, pingouin_icc_a1


def fixture_values():
    rng = np.random.default_rng(0)
    reference = rng.normal(10, 3, 30)
    return reference + rng.normal(0.4, 0.5, 30), reference


def test_formula_matches_pingouin():
    measured, reference = fixture_values()
    value, _ = pingouin_icc_a1(measured, reference)
    assert float(icc_a1(measured, reference)) == pytest.approx(value, abs=1e-6)


def test_perfect_agreement_is_one():
    values = np.arange(10.0)
    assert float(icc_a1(values, values)) == pytest.approx(1.0)


def test_absolute_agreement_penalises_a_constant_offset():
    values = np.arange(10.0)
    assert float(icc_a1(values + 5, values)) < 0.8


def test_identical_constant_values_are_undefined():
    assert np.isnan(icc_a1(np.ones(5), np.ones(5)))


def test_resample_array_matches_single_calls():
    measured, reference = fixture_values()
    indices = np.array([[0, 1, 2, 3, 4, 5], [5, 5, 4, 3, 2, 1]])
    batch = icc_a1(measured[indices], reference[indices])
    assert batch[1] == pytest.approx(float(icc_a1(measured[indices[1]], reference[indices[1]])))


def test_repeated_targets_are_treated_as_separate_targets():
    measured, reference = fixture_values()
    indices = np.array([0, 0, 1, 2, 3, 3, 4, 5])
    value, _ = pingouin_icc_a1(measured[indices], reference[indices])
    assert float(icc_a1(measured[indices], reference[indices])) == pytest.approx(value, abs=1e-6)


def test_bland_altman_values_and_sign():
    stats = bland_altman(np.array([2.0, 4.0, 6.0]), np.array([1.0, 2.0, 3.0]))
    assert stats.bias == pytest.approx(2.0)
    assert stats.sd == pytest.approx(1.0)
    assert stats.lower_limit == pytest.approx(2.0 - 1.96)
    assert stats.upper_limit == pytest.approx(2.0 + 1.96)
