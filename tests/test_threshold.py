"""HU threshold baseline."""

import numpy as np

from stonekit.seg.threshold import threshold_mask


def test_threshold_is_strictly_greater():
    image = np.array([[[271.0, 272.0, 273.0]]])
    assert threshold_mask(image, 272).tolist() == [[[False, False, True]]]


def test_nan_is_not_included():
    assert not threshold_mask(np.array([[[np.nan]]]), 272).any()
