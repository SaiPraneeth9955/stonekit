"""Metrics on small hand-made masks with answers worked out by hand."""

import numpy as np
import pytest

from stonekit.eval import seg_metrics


def masks():
    # Reference pieces: 1 (4 voxels), 2 (2 voxels, missed). Predicted: 1 (overlaps 2 voxels), 2 (no overlap).
    reference = np.zeros((10, 10), dtype=np.uint16)
    reference[0, 0:4] = 1
    reference[5, 0:2] = 2
    predicted = np.zeros((10, 10), dtype=np.uint16)
    predicted[0, 2:6] = 1
    predicted[9, 9] = 2
    return predicted, reference


def test_dice_and_iou_by_hand():
    predicted, reference = masks()
    assert seg_metrics.dice(predicted > 0, reference > 0) == pytest.approx(2 * 2 / (5 + 6))
    assert seg_metrics.iou(predicted > 0, reference > 0) == pytest.approx(2 / 9)


def test_both_empty_is_undefined_not_perfect():
    empty = np.zeros((3, 3), dtype=bool)
    assert seg_metrics.dice(empty, empty) is None
    assert seg_metrics.iou(empty, empty) is None


def test_one_empty_scores_zero():
    full = np.ones((3, 3), dtype=bool)
    empty = np.zeros((3, 3), dtype=bool)
    assert seg_metrics.dice(empty, full) == 0
    assert seg_metrics.iou(full, empty) == 0


def test_detection_counts_by_hand():
    predicted, reference = masks()
    assert seg_metrics.detection_counts(predicted, reference) == {
        "reference_pieces": 2,
        "reference_pieces_detected": 1,
        "predicted_pieces": 2,
        "predicted_pieces_without_reference": 1,
    }


def test_best_match_takes_the_largest_overlap():
    reference = np.array([[1, 1, 1, 1, 0]], dtype=np.uint16)
    predicted = np.array([[2, 3, 3, 3, 0]], dtype=np.uint16)
    assert seg_metrics.best_matches(predicted, reference) == {1: 3}


def test_no_overlap_gives_no_matches():
    predicted, reference = masks()
    predicted[:] = 0
    assert seg_metrics.best_matches(predicted, reference) == {}
    assert seg_metrics.detection_counts(predicted, reference)["reference_pieces_detected"] == 0
