"""Case-level bootstrap and degenerate resamples.

TODO(Stage 7): add tests for the patient-level cluster bootstrap.
"""

import numpy as np

from stonekit.eval.bootstrap import percentile_interval, resample_indices


def test_resamples_are_fixed_by_the_seed():
    first = resample_indices(10, 5, seed=1)
    assert first.shape == (5, 10)
    assert np.array_equal(first, resample_indices(10, 5, seed=1))
    assert not np.array_equal(first, resample_indices(10, 5, seed=2))


def test_percentile_interval_uses_2_5_and_97_5():
    interval = percentile_interval(np.arange(1001.0))
    assert (interval.lower, interval.upper) == (25.0, 975.0)
    assert interval.degenerate == 0


def test_degenerate_values_are_left_out_and_counted_not_replaced():
    values = np.concatenate([np.arange(100.0), [np.nan, np.inf, -np.inf]])
    interval = percentile_interval(values)
    clean = percentile_interval(np.arange(100.0))
    assert interval.degenerate == 3
    assert (interval.lower, interval.upper) == (clean.lower, clean.upper)


def test_more_than_1_percent_degenerate_is_unreliable():
    assert not percentile_interval(np.r_[np.ones(1980), np.full(20, np.nan)]).unreliable
    assert percentile_interval(np.r_[np.ones(1979), np.full(21, np.nan)]).unreliable


def test_all_degenerate_gives_no_interval():
    interval = percentile_interval(np.full(5, np.nan))
    assert np.isnan(interval.lower)
    assert interval.degenerate == 5
