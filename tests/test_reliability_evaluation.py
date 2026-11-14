"""Folds, flags, the primary estimand and leakage, on hand-made values."""

import numpy as np
import pytest

from stonekit.reliability.evaluation import (
    Comparison,
    ModelSpec,
    Rows,
    accepted_mean,
    copy_numbers,
    delta_by_k,
    evaluate_partition,
    fold_ranks,
    model_ranks,
    psi,
    row_folds,
    scale_dependent,
    section_11_reading,
    spacing_dependent,
    stratified_folds,
    trapezoid_weights,
)

KS = (2, 3, 4, 5, 6)


def make_rows(n=45) -> Rows:
    grid = np.arange(n, dtype=float)
    return Rows(
        np.arange(1, n + 1),
        np.zeros(n, dtype=int),
        {
            "x1": grid / 10,
            "S_G": np.sin(grid),
            "M": np.cos(2 * grid),
            "E_A": 0.1 + 0.5 * np.abs(np.sin(3 * grid)),
        },
    )


def test_weights_are_the_frozen_trapezoid():
    assert trapezoid_weights(KS) == pytest.approx([1 / 8, 1 / 4, 1 / 4, 1 / 4, 1 / 8])


def test_101_stones_give_folds_of_20_and_21_with_one_stone_per_block():
    x1 = np.arange(101, dtype=float)
    folds = stratified_folds(x1, np.arange(1, 102), np.random.default_rng(0))
    assert sorted(np.bincount(folds)) == [20, 20, 20, 20, 21]
    for start in range(0, 100, 5):
        assert sorted(folds[start : start + 5]) == [0, 1, 2, 3, 4]


def test_flagged_counts_give_10_to_30_stones_of_101():
    x1 = np.arange(101, dtype=float)
    stone_id = np.arange(1, 102)
    fold = stratified_folds(x1, stone_id, np.random.default_rng(0))
    ranks = fold_ranks(-x1, stone_id, np.zeros(101, dtype=int), fold)
    assert [int((ranks < k).sum()) for k in KS] == [10, 15, 20, 25, 30]


def test_sorting_ties_use_stone_id():
    x1 = np.zeros(10)
    ids = np.arange(10)[::-1].copy()
    first = stratified_folds(x1, ids, np.random.default_rng(1))
    second = stratified_folds(x1, ids, np.random.default_rng(1))
    assert first.tolist() == second.tolist()


def test_copies_of_a_stone_stay_in_one_fold():
    rows = make_rows()
    resampled = rows.take(np.random.default_rng(3).integers(0, len(rows), len(rows)))
    fold = row_folds(resampled, np.random.default_rng(4))
    for stone in np.unique(resampled.stone_id):
        assert len(set(fold[resampled.stone_id == stone])) == 1


def test_copy_numbers_count_repeats():
    assert copy_numbers(np.array([5, 7, 5, 5])).tolist() == [0, 0, 1, 2]


def test_ranking_ties_break_by_stone_id_then_copy():
    priority = np.ones(4)
    ranks = fold_ranks(
        priority, np.array([9, 3, 3, 1]), np.array([0, 1, 0, 0]), np.zeros(4, dtype=int)
    )
    assert ranks.tolist() == [3, 2, 1, 0]


def test_accepted_mean_and_delta_are_hand_checked():
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = np.array([0, 1, 2, 3])
    proposed = np.array([3, 2, 1, 0])
    assert accepted_mean(errors, baseline, 1) == pytest.approx(3.0)
    assert accepted_mean(errors, proposed, 1) == pytest.approx(2.0)
    assert delta_by_k(errors, baseline, proposed, (1,)) == pytest.approx([1.0])


def test_psi_of_a_constant_difference_is_that_difference():
    assert psi(np.full(5, 0.07), trapezoid_weights(KS)) == pytest.approx(0.07)


def test_changing_held_out_outcomes_does_not_change_their_flags_or_the_fit():
    rows = make_rows()
    fold = row_folds(rows, np.random.default_rng(5))
    spec = ModelSpec(("x1", "S_G", "M"), "E_A")
    before, fits_before = model_ranks(rows, fold, spec)

    changed_columns = dict(rows.columns)
    changed_errors = rows.columns["E_A"].copy()
    changed_errors[fold == 2] = [9.0 - i for i in range(int((fold == 2).sum()))]
    changed_columns["E_A"] = changed_errors
    changed = Rows(rows.stone_id, rows.copy, changed_columns)
    after, fits_after = model_ranks(changed, fold, spec)

    assert before[fold == 2].tolist() == after[fold == 2].tolist()
    assert fits_before[2].coefficients == pytest.approx(fits_after[2].coefficients)
    assert fits_before[0].coefficients != pytest.approx(fits_after[0].coefficients)


def test_fold_assignment_does_not_use_outcomes():
    rows = make_rows()
    other = Rows(rows.stone_id, rows.copy, {**rows.columns, "E_A": rows.columns["E_A"][::-1]})
    first = row_folds(rows, np.random.default_rng(6))
    assert first.tolist() == row_folds(other, np.random.default_rng(6)).tolist()


def test_both_rules_flag_the_same_number_in_each_fold():
    rows = make_rows()
    fold = row_folds(rows, np.random.default_rng(7))
    models = {"full_A": ModelSpec(("x1", "S_G", "M"), "E_A")}
    result = evaluate_partition(
        rows, fold, models, [Comparison("psi", "E_A", "size", "full_A")], (2,)
    )
    for current in range(5):
        chosen = fold == current
        assert (result.ranks["size"][chosen] < 2).sum() == 2
        assert (result.ranks["full_A"][chosen] < 2).sum() == 2


def test_failed_rule_and_small_folds_give_nan():
    rows = make_rows()
    rows.columns["constant"] = np.ones(len(rows))
    fold = row_folds(rows, np.random.default_rng(8))
    comparison = Comparison("psi", "E_A", "size", "broken")
    broken = {"broken": ModelSpec(("x1", "constant"), "E_A")}
    assert np.isnan(evaluate_partition(rows, fold, broken, [comparison], KS).deltas["psi"]).all()
    good = {"full": ModelSpec(("x1", "S_G"), "E_A")}
    small = Comparison("psi", "E_A", "size", "full")
    deltas = evaluate_partition(rows, fold, good, [small], (50,)).deltas["psi"]
    assert np.isnan(deltas).all()


def test_section_11_readings_are_not_pass_or_fail():
    assert "added measurable" in section_11_reading(0.01, 0.2)
    assert "does not show the signals add nothing" in section_11_reading(-0.1, 0.2)
    assert section_11_reading(-0.2, -0.1) == "Size-only performed better in this dataset"
    assert section_11_reading(float("nan"), 0.1) == "Interval unavailable"


def test_spacing_and_scale_rules():
    assert spacing_dependent(0.1, -0.1, (-0.2, 0.3), (-0.3, 0.2))
    assert spacing_dependent(0.1, 0.1, (0.01, 0.2), (-0.01, 0.2))
    assert not spacing_dependent(0.1, 0.2, (-0.2, 0.3), (-0.1, 0.4))
    assert scale_dependent(0.1, -1.0)
    assert not scale_dependent(0.1, 1.0)
