"""Split-conformal ranges and global limits on hand-made values."""

import math

import numpy as np
import pytest

from stonekit.reliability.evaluation import Rows
from stonekit.reliability.ranges import (
    calibrate_scale,
    conformal_order,
    conformal_quantile,
    interval_score,
    log_limits,
    ml_limits,
    outer_folds,
    range_partition,
    split_groups,
    summarize_bounds,
)


def make_rows(n=60) -> Rows:
    grid = np.arange(n, dtype=float)
    reference = 0.5 + grid / 10
    volume_a = reference * np.exp(0.3 * np.sin(grid))
    volume_b = reference * np.exp(0.3 * np.sin(grid) + 0.01)
    return Rows(
        np.arange(1, n + 1),
        np.zeros(n, dtype=int),
        {
            "x1": np.log(volume_a * 100),
            "S_G": np.sin(2 * grid),
            "M": np.cos(3 * grid),
            "reference_volume_ml": reference,
            "volume_A": volume_a,
            "volume_B": volume_b,
            "E_A": np.abs(np.log(volume_a / reference)),
            "E_B": np.abs(np.log(volume_b / reference)),
        },
    )


def test_order_statistic_is_exact():
    assert conformal_order(30) == 28
    assert conformal_order(9) == 9
    assert conformal_order(8) is None
    assert conformal_order(29) == 27


def test_conformal_quantile_takes_the_28th_of_30():
    assert conformal_quantile(np.arange(30.0, 0.0, -1.0)) == 28.0
    assert conformal_quantile(np.ones(5)) is None


def test_interval_score_is_hand_checked():
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])
    target = np.array([0.5, 2.0, -1.0])
    assert interval_score(lower, upper, target) == pytest.approx([1.0, 21.0, 21.0])


def test_log_limits_follow_the_frozen_formula():
    volume = np.array([2.0, 3.0, 5.0, 7.0])
    reference = np.array([1.0, 3.0, 4.0, 9.0])
    ratios = np.log(volume / reference)
    mean, sd = ratios.mean(), ratios.std(ddof=1)
    lower, upper = log_limits(volume, reference, np.array([4.0]))
    assert lower[0] == pytest.approx(math.log(4.0) - mean - 1.645 * sd)
    assert upper[0] == pytest.approx(math.log(4.0) - mean + 1.645 * sd)


def test_ml_limits_are_not_clipped():
    volume = np.array([0.1, 0.2, 10.0])
    reference = np.array([0.1, 0.5, 5.0])
    lower, _ = ml_limits(volume, reference, np.array([0.1]))
    assert lower[0] < 0


def test_outer_training_sets_of_101_stones_give_30_calibration_stones():
    stone_id = np.arange(1, 102)
    rng = np.random.default_rng(0)
    fold = outer_folds(stone_id, rng)
    assert sorted(np.bincount(fold)) == [10] * 9 + [11]
    for current in range(10):
        training = fold != current
        proper, calibration = split_groups(stone_id, training, rng)
        assert calibration.sum() == 30
        assert not (proper & calibration).any()
        assert ((proper | calibration) == training).all()


def test_copies_stay_in_one_calibration_group():
    stone_id = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    proper, calibration = split_groups(
        stone_id, np.ones(12, dtype=bool), np.random.default_rng(1)
    )
    for stone in range(1, 7):
        assert len(set(calibration[stone_id == stone])) == 1
    assert (proper ^ calibration).all()


def test_calibration_subset_does_not_fit_the_model():
    rows = make_rows()
    predictors = np.column_stack([rows.columns["x1"], rows.columns["S_G"]])
    errors = rows.columns["E_A"]
    proper = np.arange(60) < 40
    calibration = ~proper
    changed = errors.copy()
    changed[calibration] = changed[calibration] + 5
    first = calibrate_scale(predictors, errors, proper, calibration)
    second = calibrate_scale(predictors, changed, proper, calibration)
    assert first.fit.coefficients == pytest.approx(second.fit.coefficients)
    assert first.q != pytest.approx(second.q)


def test_held_out_outcome_does_not_change_that_stones_range():
    rows = make_rows()
    first = range_partition(rows, ("A",), ("x1", "S_G", "M"), np.random.default_rng(2))
    errors = rows.columns["E_A"].copy()
    errors[10] = 7.0
    columns = {**rows.columns, "E_A": errors}
    changed = Rows(rows.stone_id, rows.copy, columns)
    second = range_partition(changed, ("A",), ("x1", "S_G", "M"), np.random.default_rng(2))
    for bound in ("lower", "upper"):
        assert first["full_A"][bound][10] == pytest.approx(second["full_A"][bound][10])
    assert not np.allclose(first["full_A"]["upper"], second["full_A"]["upper"])


def test_range_partition_returns_every_method_for_every_stone():
    rows = make_rows()
    bounds = range_partition(rows, ("A", "B"), ("x1", "S_G", "M"), np.random.default_rng(3))
    for method in ("full", "size_scale", "log_limits", "ml_limits"):
        for assumption in ("A", "B"):
            values = bounds[f"{method}_{assumption}"]
            assert np.isfinite(values["lower"]).all()
            assert (values["upper"] >= values["lower"]).all()


def test_too_few_calibration_stones_fail_rather_than_invent_a_range():
    rows = make_rows(12)
    assert range_partition(rows, ("A",), ("x1", "S_G", "M"), np.random.default_rng(4)) is None


def test_summaries_report_coverage_and_width_together():
    reference = np.array([1.0, np.e, np.e**2])
    bounds = {"lower": np.array([-0.5, 0.5, 3.0]), "upper": np.array([0.5, 1.5, 4.0])}
    summary = summarize_bounds(bounds, reference, "full_A")
    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["median_width"] == pytest.approx(1.0)
    ml = summarize_bounds(
        {"lower": np.array([-1.0, 0.0, 1.0]), "upper": np.array([2.0, 3.0, 8.0])},
        reference,
        "ml_limits_A",
    )
    assert ml["nonpositive_lower_limits"] == 2
    assert math.isnan(ml["mean_interval_score"])
