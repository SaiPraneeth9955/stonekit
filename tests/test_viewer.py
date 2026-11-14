"""Tests for the tri-planar viewer payload (stonekit.app.viewer). Arrays are synthetic test inputs."""

import base64
import json
import zlib

import numpy as np
import pytest

from stonekit.app.viewer import viewer_html, viewer_payload


def unpack(b64: str, dtype, shape):
    return np.frombuffer(zlib.decompress(base64.b64decode(b64)), dtype=dtype).reshape(shape)


def test_round_trip_keeps_values_and_c_order():
    ct = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6) - 100
    stones = np.zeros((4, 5, 6), np.int32)
    stones[1, 2, 3] = 7
    p = viewer_payload(ct, (0.7, 0.8, 2.5), kidney=ct > 0, stones=stones, selected=7)
    assert p["shape"] == [4, 5, 6] and p["spacing"] == [0.7, 0.8, 2.5]
    assert np.array_equal(unpack(p["ct"], np.int16, (4, 5, 6)), ct)
    assert unpack(p["stones"], np.uint16, (4, 5, 6))[1, 2, 3] == 7
    assert unpack(p["kidney"], np.uint8, (4, 5, 6)).sum() == int((ct > 0).sum())
    assert p["focus"] == [2, 2, 3]


def test_shape_mismatch_and_clipping():
    ct = np.zeros((3, 3, 3), np.float32)
    with pytest.raises(ValueError):
        viewer_payload(ct, (1, 1, 1), stones=np.zeros((3, 3, 2), np.int32))
    big = np.full((2, 2, 2), 40000.0)
    assert unpack(viewer_payload(big, (1, 1, 1))["ct"], np.int16, (2, 2, 2)).max() == 32767


def test_focus_is_clamped_and_html_embeds_payload():
    p = viewer_payload(np.zeros((3, 4, 5), np.int16), (1, 1, 1), focus_voxel=[10, -2, 4])
    assert p["focus"] == [2, 0, 4]
    html = viewer_html(p, 500)
    assert json.dumps(p) in html and "500px" in html and "__PAYLOAD__" not in html
    assert "http" not in html.split("<script>")[1]  # no network access from the component
