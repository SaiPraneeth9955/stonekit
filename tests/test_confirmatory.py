"""Unit tests for the T1 interval engine (src/stonekit/uq/confirmatory.py). The tables here are small
generated unit-test FIXTURES that exercise code paths; they are never used as data or results."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stonekit.uq import confirmatory as cfm

ROOT = Path(__file__).resolve().parent.parent


def fixture_tables(n_scans: int, seed: int, prefix: str) -> cfm.Tables:
    rng = np.random.default_rng(seed)
    ann, pred, scans = [], [], []
    for i in range(n_scans):
        case = f"{prefix}{i:03d}"
        slice_mm = [1.25, 2.5, 0.62][i % 3]
        k = 0 if i % 6 == 0 else int(rng.integers(1, 4))
        b = bh = bh_all = 0.0
        pid = 0
        for j in range(k):
            ref_v = float(rng.lognormal(4, 1.2))
            ratio = float(np.exp(rng.normal(0, 0.3 + 0.4 / np.log(ref_v + 2))))
            d = float((6 * ref_v / np.pi) ** (1 / 3))
            b += ref_v
            if rng.random() < 0.15:
                ann.append({"case": case, "group": "missed", "ref_volume_mm3": ref_v, "pred_volume_mm3": np.nan, "pred_max_diameter_3d_mm": np.nan,
                            "ref_max_diameter_3d_mm": d, "pred_tta_spread": np.nan, "pred_threshold_sensitivity": np.nan})  # fmt: skip
                continue
            pid += 1
            pv = ref_v / ratio
            ann.append({"case": case, "group": "one_to_one", "ref_volume_mm3": ref_v, "pred_volume_mm3": pv, "pred_max_diameter_3d_mm": d / ratio ** (1 / 3),
                        "ref_max_diameter_3d_mm": d, "pred_tta_spread": float(rng.uniform(0.02, 0.4)), "pred_threshold_sensitivity": float(rng.uniform(0.05, 0.6))})  # fmt: skip
            pred.append({"case": case, "group": "one_to_one", "volume_mm3": pv, "max_diameter_3d_mm": d / ratio ** (1 / 3), "mean_hu": float(rng.uniform(200, 1200)),
                         "p_mean": float(rng.uniform(0.6, 1)), "tta_spread": float(rng.uniform(0.02, 0.4)), "threshold_sensitivity": float(rng.uniform(0.05, 0.6)),
                         "distance_to_kidney_mm": float(rng.uniform(0, 10)), "beyond_30mm": False})  # fmt: skip
            bh += pv
            bh_all += pv
        for _ in range(int(rng.integers(0, 3))):
            v = float(rng.lognormal(2.5, 1))
            far = bool(rng.random() < 0.4)
            pred.append({"case": case, "group": "false_positive", "volume_mm3": v, "max_diameter_3d_mm": float((6 * v / np.pi) ** (1 / 3)), "mean_hu": float(rng.uniform(150, 400)),
                         "p_mean": float(rng.uniform(0.5, 0.8)), "tta_spread": float(rng.uniform(0.2, 0.8)), "threshold_sensitivity": float(rng.uniform(0.3, 1.2)),
                         "distance_to_kidney_mm": float(rng.uniform(31, 60) if far else rng.uniform(0, 30)), "beyond_30mm": far})  # fmt: skip
            bh_all += v
            bh += 0 if far else v
        scans.append({"case": case, "slice_mm": slice_mm, "annotated_burden_mm3": b, "predicted_burden_filtered_mm3": bh, "predicted_burden_mm3": bh_all})
    return cfm.Tables(pd.DataFrame(ann), pd.DataFrame(pred), pd.DataFrame(scans))


def load_e6():
    spec = importlib.util.spec_from_file_location("e6", ROOT / "scripts" / "e6_uncertainty_design.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["U0", "U1", "U2", "U4", "U5"])
@pytest.mark.parametrize("target", ["volume_mm3", "max_diameter_3d_mm"])
def test_stone_intervals_identical_to_e6(method, target):
    e6 = load_e6()
    cal, test = cfm.matched(fixture_tables(60, 1, "c")), cfm.matched(fixture_tables(30, 2, "t"))
    lo1, hi1 = cfm.stone_intervals(method, cal, test, target, 0.10)
    lo2, hi2 = e6.intervals(method, cal, test, target, 0.10)
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)


def test_burden_intervals_valid_deterministic_and_order_free():
    fit, cal, test = fixture_tables(80, 3, "f"), fixture_tables(35, 4, "c"), fixture_tables(20, 5, "t")
    a = cfm.burden_intervals(fit, cal, test, 0.10)
    assert set(a) == {"P2-0 naive", "P2-B direct", "P2-B-CQR", "P2-A decomposition"}
    for d in a.values():
        assert len(d) == 20 and (d.lo <= d.hi).all() and (d.lo >= 0).all()
    b = cfm.burden_intervals(fit, cal, test, 0.10)
    for m in a:
        pd.testing.assert_frame_equal(a[m], b[m])
    shuffled = cfm.Tables(test.annotated, test.predicted, test.scans.sample(frac=1, random_state=0))
    c = cfm.burden_intervals(fit, cal, shuffled, 0.10)
    for m in a:
        pd.testing.assert_frame_equal(a[m].sort_values("case").reset_index(drop=True), c[m].sort_values("case").reset_index(drop=True))


def test_test_scans_never_affect_calibrated_quantities():
    """Changing the TEST annotations must not change any TEST interval (they are only evaluated)."""
    fit, cal, test = fixture_tables(80, 3, "f"), fixture_tables(35, 4, "c"), fixture_tables(20, 5, "t")
    altered = cfm.Tables(test.annotated.assign(ref_volume_mm3=test.annotated.ref_volume_mm3 * 3),
                         test.predicted, test.scans.assign(annotated_burden_mm3=test.scans.annotated_burden_mm3 * 3))  # fmt: skip
    a, b = cfm.burden_intervals(fit, cal, test, 0.10), cfm.burden_intervals(fit, cal, altered, 0.10)
    for m in a:
        np.testing.assert_allclose(a[m][["lo", "hi"]].to_numpy(), b[m][["lo", "hi"]].to_numpy())


def test_p2_formal_test_logic():
    cases = [f"x{i}" for i in range(40)]
    base = pd.DataFrame({"case": cases, "B": np.linspace(0, 1000, 40)})
    iv = {"P2-A decomposition": base.assign(lo=base.B * 0.8, hi=base.B * 1.2 + 10),
          "P2-B-CQR": base.assign(lo=base.B * 0.5, hi=base.B * 2.0 + 40),
          "P2-B direct": base.assign(lo=0.0, hi=base.B * 3 + 100), "P2-0 naive": base.assign(lo=base.B, hi=base.B)}  # fmt: skip
    r = cfm.p2_test(iv)
    assert r["width_ratio_A_over_BCQR"]["ci95"][1] < 1 and r["formal"]["P2_success"]
    iv["P2-A decomposition"] = base.assign(lo=base.B + 1, hi=base.B + 2)  # narrow but never covers
    r = cfm.p2_test(iv)
    assert not r["formal"]["coverage_check_cp_upper_ge_nominal"] and not r["formal"]["P2_success"]
