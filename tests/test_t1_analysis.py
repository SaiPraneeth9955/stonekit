"""T1 analysis-stage order and marker rules, on generated unit-test FIXTURE tables only (never data or results)."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from stonekit.t1.analysis import ITEM5, Exclusion, reader_confirmed_errors, run_analysis

ROOT = Path(__file__).resolve().parent


def fixtures():
    spec = importlib.util.spec_from_file_location("tc", ROOT / "test_confirmatory.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.fixture_tables


@pytest.mark.parametrize("mode", ["DRESS_REHEARSAL", "T1"])
def test_calibration_hashed_before_test_load_and_marker_rule(tmp_path, mode):
    ft = fixtures()
    fit, cal, test = ft(80, 3, "f"), ft(60, 4, "c"), ft(30, 5, "t")
    out = tmp_path / "run"
    seen = {}

    def load_test():
        seen["sha_present"] = (out / "calibration_quantities.sha256").exists()
        return test

    ev = run_analysis(fit, cal, load_test, out, mode, annotation_error_scans=frozenset() if mode == "T1" else None)
    assert seen["sha_present"]
    if mode == "T1":
        assert ev["sensitivity"][ITEM5]["result"] == "no scan removed; analysis identical to primary"
    else:
        assert "sensitivity" not in ev
    assert (out / "COMPLETED").exists() == (mode == "T1")
    assert set(ev["formal_P2"]) == {"ratio_ci_excludes_1_below", "coverage_check_cp_upper_ge_nominal", "P2_success"}
    assert ev["per_stone"]["volume_mm3|0.90"]["U2"]["role"] == "final"
    assert ev["per_stone"]["max_diameter_3d_mm|0.90"]["U4"]["role"] == "exploratory"


def test_t1_requires_item5_input(tmp_path):
    ft = fixtures()
    with pytest.raises(ValueError, match="annotation_error_scans"):
        run_analysis(ft(40, 3, "f"), ft(40, 4, "c"), lambda: ft(10, 5, "t"), tmp_path / "z", "T1")
    assert not (tmp_path / "z").exists()  # refused before anything is written


def test_item5_sensitivity_refits_before_test_load_and_leaves_primary_unchanged(tmp_path):
    ft = fixtures()
    fit, cal, test = ft(80, 3, "f"), ft(60, 4, "c"), ft(30, 5, "t")
    marked = frozenset({"c001", "c002", "t003", "f005"})  # f005: a development scan, outside item 5
    seen = {}

    def load_test():
        q = json.loads((tmp_path / "s" / "calibration_quantities.json").read_text())["sensitivity"][ITEM5]
        seen["item5_calibration_hashed_first"] = q["calibration_removed"] == ["c001", "c002"] and "per_stone" in q
        return test

    ev = run_analysis(fit, cal, load_test, tmp_path / "s", "DRESS_REHEARSAL", annotation_error_scans=marked)
    base = run_analysis(fit, cal, lambda: test, tmp_path / "p", "DRESS_REHEARSAL")
    assert seen["item5_calibration_hashed_first"]
    s = ev["sensitivity"][ITEM5]
    assert s["calibration_removed"] == ["c001", "c002"] and s["test_removed"] == ["t003"] and s["fit_removed"] == []
    assert s["evaluation"]["test_scans"] == 29 and "formal_P2" in s["evaluation"]
    for k in ("per_stone", "nb", "burden", "formal_P2", "test_scans"):
        assert json.dumps(ev[k], default=float, sort_keys=True) == json.dumps(base[k], default=float, sort_keys=True)  # primary unaffected


def test_named_cohorts_remove_fit_scans_and_item5_unavailable(tmp_path):
    ft = fixtures()
    fit, cal, test = ft(80, 3, "f"), ft(60, 4, "c"), ft(30, 5, "t")
    sens = {"S1_screening_exclusions": Exclusion(frozenset({"f001", "f002", "t004"}), reserved_only=False),
            "S3_repeat_pairs": Exclusion(frozenset({"f007", "c010"}), reserved_only=True)}  # fmt: skip
    ev = run_analysis(fit, cal, lambda: test, tmp_path / "n", "T1", sensitivity=sens, item5_unavailable="no qualified reader (OD-19)")
    s1, s3 = ev["sensitivity"]["S1_screening_exclusions"], ev["sensitivity"]["S3_repeat_pairs"]
    assert s1["fit_removed"] == ["f001", "f002"] and s1["test_removed"] == ["t004"] and s1["evaluation"]["test_scans"] == 29
    assert s3["fit_removed"] == [] and s3["calibration_removed"] == ["c010"]
    assert ev["sensitivity"][ITEM5]["result"] == "not performed: no qualified reader (OD-19)"
    q = json.loads((tmp_path / "n" / "calibration_quantities.json").read_text())
    assert q["sensitivity"]["S1_screening_exclusions"]["fit_scans_n"] == 78
    assert (tmp_path / "n" / "COMPLETED").exists()


def _sheet(tmp_path, rows):
    cols = ["adj_id", "case", "target", "location_type", "q1_finding", "q2_label_validity", "q3_scan_contrast", "blinded_first_pass"]
    p = tmp_path / "sheet.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(dict(zip(cols, r, strict=True)) for r in rows)
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def test_reader_confirmed_errors_rule_and_refusals(tmp_path):
    L, U = "labelled stone region", "unlabelled region"
    rows = [("A1", "s1", "1", L, "confirmed_non_stone", "invalid", "no", "yes"),            # Q2 invalid -> marked
            ("A2", "s2", "1", U, "confirmed_urinary_stone", "not_applicable", "no", "yes"),  # omission -> marked
            ("A3", "s3", "1", L, "confirmed_urinary_stone", "valid", "no", "yes"),           # valid label -> not marked
            ("A4", "s4", "1", U, "confirmed_non_stone", "not_applicable", "no", "yes"),      # device etc. -> not marked
            ("A5", "s5", "1", L, "indeterminate", "indeterminate", "indeterminate", "yes")]  # fmt: skip
    p, sha = _sheet(tmp_path, rows)
    assert reader_confirmed_errors(p, sha) == frozenset({"s1", "s2"})
    with pytest.raises(ValueError, match="SHA-256"):
        reader_confirmed_errors(p, "0" * 64)
    p2, sha2 = _sheet(tmp_path, [("A1", "s1", "1", L, "", "invalid", "no", "yes")])
    with pytest.raises(ValueError, match="incomplete"):
        reader_confirmed_errors(p2, sha2)
    p3, sha3 = _sheet(tmp_path, [("A1", "s1", "1", L, "confirmed_non_stone", "invalid", "no", "no")])
    with pytest.raises(ValueError, match="first pass"):
        reader_confirmed_errors(p3, sha3)


def test_bad_mode_and_existing_folder(tmp_path):
    ft = fixtures()
    with pytest.raises(ValueError):
        run_analysis(ft(40, 3, "f"), ft(40, 4, "c"), lambda: ft(10, 5, "t"), tmp_path / "x", "OTHER")
    (tmp_path / "y").mkdir()
    with pytest.raises(FileExistsError):
        run_analysis(ft(40, 3, "f"), ft(40, 4, "c"), lambda: ft(10, 5, "t"), tmp_path / "y", "DRESS_REHEARSAL")
