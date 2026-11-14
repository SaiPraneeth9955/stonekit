"""The external-check rebuild reproduces the T1 calibration exactly (generated unit-test FIXTURE tables only)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stonekit.t1.analysis import run_analysis
from stonekit.t1.external import load_quantities, rebuild_burden, stone_bounds
from stonekit.uq import confirmatory as cfm

ROOT = Path(__file__).resolve().parent


def fixtures():
    spec = importlib.util.spec_from_file_location("tc", ROOT / "test_confirmatory.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.fixture_tables


def test_rebuilt_calibration_reproduces_t1_intervals(tmp_path):
    ft = fixtures()
    fit, cal, test = ft(80, 3, "f"), ft(60, 4, "c"), ft(30, 5, "t")
    run_analysis(fit, cal, lambda: test, tmp_path / "a", "DRESS_REHEARSAL")
    q = load_quantities(tmp_path / "a")
    original = cfm.fit_burden(fit, cal, 0.10, filtered=True)
    rebuilt = rebuild_burden(q["burden"]["0.1|filtered=True"], fit)
    a, b = cfm.apply_burden(original, test), cfm.apply_burden(rebuilt, test)
    for m in a:
        pd.testing.assert_frame_equal(a[m], b[m], rtol=1e-9, atol=1e-6)
    mt = cfm.matched(test)
    lo, hi = stone_bounds(q["per_stone"], "U2", "volume_mm3", 0.1, mt.pred_max_diameter_3d_mm.to_numpy())
    lo2, hi2 = cfm.stone_intervals("U2", cfm.matched(cal), mt, "volume_mm3", 0.1)
    assert np.allclose(lo, lo2) and np.allclose(hi, hi2)


def test_tampered_quantities_are_refused(tmp_path):
    ft = fixtures()
    run_analysis(ft(80, 3, "f"), ft(60, 4, "c"), lambda: ft(30, 5, "t"), tmp_path / "a", "DRESS_REHEARSAL")
    p = tmp_path / "a" / "calibration_quantities.json"
    d = json.loads(p.read_text())
    d["burden"]["0.1|filtered=True"]["p2_a_t"] = 0.0
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="SHA-256"):
        load_quantities(tmp_path / "a")


def test_wrong_fit_table_is_refused(tmp_path):
    ft = fixtures()
    fit = ft(80, 3, "f")
    run_analysis(fit, ft(60, 4, "c"), lambda: ft(30, 5, "t"), tmp_path / "a", "DRESS_REHEARSAL")
    q = load_quantities(tmp_path / "a")
    with pytest.raises(ValueError, match="pool"):
        rebuild_burden(q["burden"]["0.1|filtered=True"], ft(80, 9, "f"))
