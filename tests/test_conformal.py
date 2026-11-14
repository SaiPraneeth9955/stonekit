"""Tests for stonekit.uq.conformal. Random numbers here are test inputs for the statistics code,
not patient data."""

import math

import numpy as np
import pytest

from stonekit.uq import conformal as cf


def test_quantile_rank_and_unbounded():
    scores = np.arange(1, 10)  # n = 9
    assert cf.conformal_quantile(scores, 0.2) == 8  # ceil(10 * 0.8) = 8th smallest
    assert math.isinf(cf.conformal_quantile(scores, 0.05))  # ceil(10 * 0.95) = 10 > 9


def test_marginal_coverage_guarantee_on_exchangeable_data():
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(400):
        cal, test = np.abs(rng.standard_normal(30)), np.abs(rng.standard_normal(1))
        hits.append(test[0] <= cf.conformal_quantile(cal, 0.1))
    assert 0.86 <= np.mean(hits) <= 0.97  # guarantee: >= 0.9 in expectation, <= 0.9 + 1/31


def test_log_ratio_rejects_non_positive():
    with pytest.raises(ValueError):
        cf.log_ratio([1.0, 0.0], [1.0, 1.0])


def test_merge_rule():
    size = np.array([1, 2, 6, 7, 8, 9, 12, 13, 14, 15, 16])  # strata <5: 2, 5-10: 4, >=10: 5
    assert cf.merge_strata(size, (5.0, 10.0), 2) == (5.0, 10.0)
    assert cf.merge_strata(size, (5.0, 10.0), 3) == (10.0,)  # <5 (2 stones) merges upward
    # min 6: <5 (2) merges up -> <10 has 6, >=10 has 5 -> >=10 merges down -> one stratum
    assert cf.merge_strata(size, (5.0, 10.0), 6) == ()
    assert cf.merge_strata(size, (5.0, 10.0), 20) == ()


def test_mondrian_fits_per_stratum_and_intervals():
    size = np.array([2.0] * 20 + [12.0] * 20)
    scores = np.array([1.0] * 20 + [0.1] * 20)
    strata = cf.fit_mondrian(scores, size, (5.0,), 0.1, 10)
    assert strata.counts == (20, 20) and strata.quantiles == (1.0, 0.1)
    lo, hi = cf.symmetric_interval(np.array([100.0]), strata.quantiles[1])
    assert lo[0] == pytest.approx(100 * math.exp(-0.1)) and hi[0] == pytest.approx(100 * math.exp(0.1))


def test_cqr_returns_bounds_that_widen_where_noise_is_larger():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 600)
    y = rng.standard_normal(600) * (0.1 + x)
    fit, cal = np.arange(0, 300), np.arange(300, 600)
    predict = cf.fit_cqr(x[:, None], y, 0.1, fit, cal)
    lo, hi = predict(np.array([[0.05], [0.95]]))
    assert (hi - lo)[1] > 2 * (hi - lo)[0]


def test_coverage_intervals():
    low, high = cf.clopper_pearson(45, 50)
    assert low < 0.9 < high
    covered = np.array([1, 1, 0, 1, 1, 1], bool)
    cluster = np.array([0, 0, 1, 1, 2, 2])
    lo, hi = cf.cluster_bootstrap_coverage(covered, cluster, 2000, 3)
    assert 0 <= lo <= covered.mean() <= hi <= 1
