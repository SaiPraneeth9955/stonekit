"""Workstation API tests: read-only services, frozen-artefact integrity, evidence values and report rendering.

Study-level checks use the MSWAL demonstration run when it exists locally (not distributed); they skip otherwise."""

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from stonekit.workstation import services
from stonekit.workstation.server import app

client = TestClient(app)
DEMO = next(iter(sorted(services.RUNS.glob("MSWAL_0139_*"))), None) if services.RUNS.exists() else None
needs_demo = pytest.mark.skipif(DEMO is None or not (DEMO / services.RANGES).exists(), reason="local demonstration run not present")
needs_t1 = pytest.mark.skipif(not (services.T1 / "analysis/evaluation.json").exists(), reason="T1 aggregates not present")


def test_index_and_vendor_assets_are_served_locally():
    assert client.get("/").status_code == 200
    r = client.get("/vendor/plotly.min.js")
    assert r.status_code == 200 and len(r.content) > 1_000_000
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"


def test_unknown_study_and_path_escape_are_refused():
    assert client.get("/api/studies/does_not_exist").status_code == 404
    assert client.get("/api/studies/..~..~etc").status_code == 404


def test_upload_rejects_unsafe_names():
    assert client.put("/api/uploads/evil.sh", content=b"x").status_code == 400
    assert client.put("/api/uploads/scan.nii.gz", content=b"not gzip").status_code == 400


@needs_t1
def test_evidence_matches_the_stored_locked_test():
    ev = json.loads((services.T1 / "analysis/evaluation.json").read_text())
    got = client.get("/api/evidence").json()
    assert got["locked_test"]["ratio"]["median"] == pytest.approx(ev["burden"]["0.90|filtered=True"]["width_ratio_A_over_BCQR"]["median"])
    assert got["locked_test"]["formal"]["P2_success"] is True


@needs_t1
def test_frozen_artefacts_verify():
    """Every present frozen artefact matches the freeze manifest; absent ones (public clone) are NOT AVAILABLE, never PASS."""
    i = client.get("/api/integrity").json()
    assert i["mismatch"] == 0, [r for r in i["rows"] if r["status"] == "FAIL"]
    for r in i["rows"]:
        assert r["status"] == ("PASS" if r["present"] else "NOT AVAILABLE")
    assert i["status"] == ("PASS" if i["missing"] == 0 else "NOT AVAILABLE")


@needs_demo
def test_study_attaches_ranges_only_to_the_same_piece():
    sid = services.study_id(DEMO)
    s = client.get(f"/api/studies/{sid}").json()
    ranges = json.loads((DEMO / services.RANGES).read_text())
    by_id = {r["stone_id"]: r for r in ranges["stones"]}
    for st in s["stones"]:
        if st["range"]:
            assert st["range"]["voxel_count"] == st["voxel_count"] == by_id[st["stone_id"]]["voxel_count"]
    assert s["eligibility"]["status"] == "ELIGIBLE"
    b = s["ranges"]["burden"]
    assert b["decomposition"]["calibrated_interval_mm3"] == pytest.approx(b["interval_mm3"])


@needs_demo
def test_report_and_export_render_without_diagnostic_wording():
    sid = services.study_id(DEMO)
    html = client.get(f"/report/{sid}").text
    assert "Research use" in html and "frozen research" in html
    for word in ("diagnosis of", "recommend treatment", "you have", "not validated for clinical diagnosis"):
        assert word not in html.lower()
    exp = client.get(f"/export/{sid}.json")
    assert exp.status_code == 200 and json.loads(exp.text)["stones"]


@needs_demo
def test_volume_payload_is_consistent():
    p = client.get(f"/api/studies/{services.study_id(DEMO)}/volume").json()
    assert len(p["shape"]) == 3 and all(s > 0 for s in p["shape"])
    assert set(map(int, p["centres"])) == {s["stone_id"] for s in json.loads((DEMO / "report.json").read_text())["methods"]["model"]["stones"]}
    assert Path(DEMO).exists()
