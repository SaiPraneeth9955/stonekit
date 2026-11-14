"""stonekit.t1.case_ranges must reproduce the frozen external-check pipeline exactly for a single scan.

Uses one stored MSWAL case of the §5.4 external check (its run-case folder, its TTA probabilities and its saved
predicted.csv). Skips when those local outputs are absent (they are not distributed)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stonekit.t1 import case_ranges as cr
from stonekit.t1.external import load_quantities
from stonekit.uq import confirmatory as cfm

RUN = Path("results/runs/mswal_external_check_20261113T132418Z")
NEEDED = [RUN / "predicted.csv", RUN / "scans.csv", cr.T1_RUN / "analysis/calibration_quantities.json", cr.FIT_TABLE / "predicted.csv"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="local external-check outputs not present")
COLS = ["volume_mm3", "max_diameter_3d_mm", "mean_hu", "distance_to_kidney_mm", "p_mean", "tta_spread", "threshold_sensitivity"]


def _case():
    pred = pd.read_csv(RUN / "predicted.csv")
    kept = pred[~pred.beyond_30mm.astype(bool)]
    case = min(kept.case.unique())  # fixed rule: first eligible case (ID order) with a kept piece
    folder = next(p for p in (RUN / "cases").iterdir() if p.name.startswith(case + "_"))
    return case, folder, pred[pred.case == case].sort_values("pred_id").reset_index(drop=True)


def test_pieces_match_the_external_check_table():
    case, folder, stored = _case()
    pieces, scan = cr.predicted_pieces(case, folder, RUN / "tta" / case / "probabilities.npz")
    pieces = pieces.sort_values("pred_id").reset_index(drop=True)
    assert list(pieces.pred_id) == list(stored.pred_id)
    np.testing.assert_allclose(pieces[COLS].to_numpy(float), stored[COLS].to_numpy(float), rtol=1e-6, atol=1e-6)
    s = pd.read_csv(RUN / "scans.csv").set_index("case").loc[case]
    assert scan["predicted_burden_filtered_mm3"] == pytest.approx(s.predicted_burden_filtered_mm3)


def test_burden_interval_equals_the_frozen_engine():
    case, folder, stored = _case()
    q = load_quantities(cr.T1_RUN / "analysis")
    fit = cfm.Tables(*(pd.read_csv(cr.FIT_TABLE / f) for f in ("annotated.csv", "predicted.csv", "scans.csv")))
    pieces, scan = cr.predicted_pieces(case, folder, RUN / "tta" / case / "probabilities.npz")
    out = cr.ranges(pieces, scan, q, fit)
    s = pd.read_csv(RUN / "scans.csv")
    s = s[s.case == case]
    empty = pd.DataFrame(columns=["case", "group", "ref_volume_mm3", "pred_volume_mm3", "pred_max_diameter_3d_mm"])
    ref = cfm.apply_burden(cr.rebuild_burden(q["burden"]["0.1|filtered=True"], fit), cfm.Tables(empty, stored, s))["P2-A decomposition"].iloc[0]
    assert out["burden"]["interval_mm3"] == pytest.approx([ref.lo, ref.hi], rel=1e-9)
    d = out["burden"]["decomposition"]
    assert d["calibrated_interval_mm3"] == pytest.approx(out["burden"]["interval_mm3"], rel=1e-12)  # same draws as the engine
    assert d["expected_after_false_detection_mm3"] <= d["measured_mm3"] + 1e-9
    ref_cqr = cfm.apply_burden(cr.rebuild_burden(q["burden"]["0.1|filtered=True"], fit), cfm.Tables(empty, stored, s))["P2-B-CQR"].iloc[0]
    # the CQR comparator is a continuous function of the piece features, which the stored CSV holds at ~1e-16 relative
    # precision; its interval agrees to ~1e-8, far below any displayed digit
    assert out["burden"]["comparators"]["P2-B-CQR"] == pytest.approx([ref_cqr.lo, ref_cqr.hi], rel=1e-6)
    for st in out["stones"]:
        lo, hi = st["volume_range_mm3"]
        assert lo < st["volume_mm3"] < hi
        assert 0.0 <= st["false_detection_probability"] <= 1.0
    json.dumps(out)  # serialisable
