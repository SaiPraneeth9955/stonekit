"""Read-only data services for the workstation. Nothing here calculates a measurement or an interval; values come
from report.json (the pipeline), calibrated_ranges.json (the frozen calibration add-on) and committed study
aggregates. Derived display fields (size band, location relative to the kidney segmentation, QC status) are
computed from those saved outputs and labelled as such."""

import base64
import csv
import hashlib
import json
import os
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

from stonekit.data.registry import find_project_root
from stonekit.io.nifti import load_volume
from stonekit.measure.mesh import read_obj

ROOT = find_project_root()
# STONEKIT_WORKSTATION_RUNS lets the viewer open another folder of run-case outputs (for example a demonstration set)
RUNS = Path(os.environ["STONEKIT_WORKSTATION_RUNS"]).resolve() if os.environ.get("STONEKIT_WORKSTATION_RUNS") else ROOT / "data" / "interim" / "stonekit_runs"
T1 = ROOT / "results/runs/T1_20261113T081425Z"
EXT = ROOT / "results/runs/mswal_external_check_20261113T132418Z/external_check.json"
REPLAY = ROOT / "results/runs/t1_posthoc_width_replay_20261114T081724Z/posthoc_width_replay.json"
FIGS = ROOT / "paper" / "figures"
FREEZE = ROOT / "docs" / "T1_FREEZE_MANIFEST.json"
RANGES = "calibrated_ranges.json"
CONTRAST_HU = 100.0
SIZE_BANDS = ((5.0, "<5 mm"), (10.0, "5–10 mm"), (float("inf"), "≥10 mm"))
KIDNEY_RIGHT, KIDNEY_LEFT = 2, 3  # TotalSegmentator labels kept by the pipeline
STAGE_NAMES = {
    "load_ct": "Load CT", "input_checks": "Input validation", "kidney_segmentation": "Kidney segmentation",
    "kidney_roi": "Kidney region of interest", "stone_segmentation_model": "Stone segmentation (nnU-Net)",
    "stones_measurements_qc_meshes_model": "Measurement, QC and meshes", "stone_segmentation_threshold": "Threshold comparator (272 HU)",
    "stones_measurements_qc_meshes_threshold": "Comparator measurement", "kidney_mesh": "Kidney surface",
}  # fmt: skip


# ----------------------------------------------------------------------------------------------- identifiers
def study_id(folder: Path) -> str:
    return folder.relative_to(RUNS).as_posix().replace("/", "~")


def study_folder(sid: str) -> Path:
    folder = (RUNS / sid.replace("~", "/")).resolve()
    if RUNS.resolve() not in folder.parents or not (folder / "report.json").exists():
        raise KeyError(sid)
    return folder


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------- formatting helpers
def size_band(d_mm: float) -> str:
    return next(name for edge, name in SIZE_BANDS if d_mm < edge)


def status_of_check(value: str) -> str:
    return {"pass": "PASS", "ok": "PASS", "warn": "WARNING", "warning": "WARNING", "fail": "FAIL", "failed": "FAIL"}.get(str(value).lower(), "REVIEW")


def eligibility(report: dict, ranges: dict | None) -> dict:
    """Whether the frozen calibration applies to this study, and why not otherwise."""
    if report.get("failure") or report.get("result_status") not in (None, "complete"):
        return {"status": "FAILED", "detail": "Processing did not complete; no stone result exists."}
    aorta = (report.get("contrast_check") or {}).get("aorta_median_hu")
    if aorta is None:
        return {"status": "NOT AVAILABLE", "detail": "Contrast phase could not be checked (no aorta found)."}
    if aorta >= CONTRAST_HU:
        return {"status": "NOT ELIGIBLE", "detail": f"Bright aorta (median {aorta:.0f} HU): possible contrast. Calibration covers non-contrast CT only."}
    if ranges is None:
        return {"status": "NO RANGES", "detail": "Eligible non-contrast study; calibrated ranges have not been computed for this run."}
    return {"status": "ELIGIBLE", "detail": f"Non-contrast by the aorta heuristic (median {aorta:.0f} HU); calibrated ranges available."}


# ----------------------------------------------------------------------------------------------- worklist
def discover() -> list[Path]:
    if not RUNS.exists():
        return []
    found = []
    for path in [*RUNS.glob("*"), *RUNS.glob("*/*")]:
        if path.is_dir() and (path / "report.json").exists():
            try:
                _read_json(path / "report.json")
                found.append(path)
            except (OSError, ValueError):
                continue
    return sorted(found, key=lambda p: p.name.rsplit("_", 1)[-1], reverse=True)


def _ranges(folder: Path) -> dict | None:
    try:
        return _read_json(folder / RANGES) if (folder / RANGES).exists() else None
    except (OSError, ValueError):
        return None


def worklist() -> list[dict]:
    rows = []
    for folder in discover():
        report = _read_json(folder / "report.json")
        ranges = _ranges(folder)
        method = (report.get("methods") or {}).get(report.get("primary_method") or "", {})
        stones = method.get("stones", [])
        checks = report.get("input_checks") or {}
        rows.append({
            "id": study_id(folder), "case_id": report.get("case_id"), "created_utc": report.get("created_utc"),
            "source": "Li 2022 (development)" if str(report.get("case_id", "")).startswith("wd") else ("MSWAL (external)" if "MSWAL" in str(report.get("case_id")) else "Imported CT"),
            "group": folder.parent.name if folder.parent != RUNS else "",
            "shape": checks.get("shape"), "spacing_mm": checks.get("spacing_mm"),
            "stones": len(stones), "total_volume_mm3": method.get("total_volume_mm3"),
            "burden_interval_mm3": (ranges or {}).get("burden", {}).get("interval_mm3"),
            "eligibility": eligibility(report, ranges), "qc": qc_overall(report, stones),
            "config_sha256": report.get("config_sha256"),
        })  # fmt: skip
    return rows


def qc_overall(report: dict, stones: list[dict]) -> str:
    failed = {"processing_failure", "kidney_segmentation_failed", "stone_segmentation_failed", "unexpected_error", "no_kidney_found"}
    flags = set(report.get("case_qc_flags") or [])
    if report.get("failure") or flags & failed:
        return "FAIL"
    if flags or any(s.get("qc_flags") for s in stones):
        return "REVIEW"
    return "PASS"


# ----------------------------------------------------------------------------------------------- one study
@lru_cache(maxsize=8)
def _label_volumes(folder_str: str, label_file: str | None, modified: float):
    folder = Path(folder_str)
    report = _read_json(folder / "report.json")
    box = tuple(slice(a, b) for a, b in zip(report["roi"]["box_start_voxel"], report["roi"]["box_stop_voxel"], strict=True))
    kidneys = load_volume(folder / "kidney_totalsegmentator.nii.gz")
    kid = np.where(np.isin(kidneys.data[box], [KIDNEY_RIGHT, KIDNEY_LEFT]), kidneys.data[box], 0).astype(np.uint8)
    stones = load_volume(folder / label_file).data[box].astype(np.uint16) if label_file and (folder / label_file).exists() else None
    return box, kid, stones, kidneys.spacing_mm


_POINTS: dict = {}


def _kidney_points(kid: np.ndarray, label: int) -> np.ndarray:
    """Voxel coordinates of one kidney label (surface voxels only, which is enough for a nearest-kidney search)."""
    key = (id(kid), label)
    if key not in _POINTS:
        m = kid == label
        surface = m & ~ndimage.binary_erosion(m)
        _POINTS[key] = np.argwhere(surface).astype(np.float32)
    return _POINTS[key]


def _location(stone: dict, box, kid: np.ndarray, stones: np.ndarray | None, spacing) -> dict:
    """Location of a stone relative to the saved kidney segmentation (descriptive geometry, not anatomy reading)."""
    if stones is None or not kid.any():
        return {"side": "not available", "position": "not available"}
    vox = np.argwhere(stones == stone["stone_id"])
    if not len(vox):
        return {"side": "not available", "position": "not available"}
    centre = vox.mean(axis=0)
    c = tuple(int(np.rint(v)) for v in centre)
    sp = np.asarray(spacing, float)
    best, label = np.inf, 0
    for lab in (KIDNEY_RIGHT, KIDNEY_LEFT):
        pts = _kidney_points(kid, lab)
        if len(pts):
            d = float(np.sqrt((((pts - centre) * sp) ** 2).sum(axis=1)).min())
            if d < best:
                best, label = d, lab
    side = {KIDNEY_RIGHT: "Right kidney", KIDNEY_LEFT: "Left kidney"}.get(label, "not available")
    zs = _kidney_points(kid, label)[:, 2] if label else np.array([])
    position = "not available"
    if len(zs):
        lo, hi = zs.min(), zs.max()
        frac = (centre[2] - lo) / max(hi - lo, 1)
        position = "upper third" if frac > 2 / 3 else ("middle third" if frac > 1 / 3 else "lower third")
        if frac > 1 or frac < 0:
            position = "beyond kidney craniocaudal extent"
    inside = bool(kid[c]) if all(0 <= c[i] < kid.shape[i] for i in range(3)) else False
    return {"side": side, "position": position, "within_kidney_mask": inside}


def study(sid: str) -> dict:
    folder = study_folder(sid)
    report = _read_json(folder / "report.json")
    ranges = _ranges(folder)
    primary = report.get("primary_method")
    method = (report.get("methods") or {}).get(primary or "", {})
    stones = method.get("stones", [])
    by_id = {s["stone_id"]: s for s in (ranges or {}).get("stones", [])}
    box = kid = labels = spacing = None
    if report.get("roi") and (folder / "kidney_totalsegmentator.nii.gz").exists():
        lf = method.get("label_file")
        box, kid, labels, spacing = _label_volumes(str(folder), lf, (folder / "report.json").stat().st_mtime)
    out_stones = []
    for s in stones:
        r = by_id.get(s["stone_id"])
        if r and r.get("voxel_count") not in (None, s.get("voxel_count")):
            r = None  # never attach a range to a different piece
        b3, b2 = size_band(s["max_diameter_3d_mm"]), size_band(s["max_diameter_axial_mm"])
        out_stones.append({**s, "size_band": b3, "size_band_2d": b2, "category_changes_2d": b2 != b3,
                           "range": r, "location": _location(s, box, kid, labels, spacing) if kid is not None else {"side": "not available", "position": "not available"},
                           "qc_status": "REVIEW" if s.get("qc_flags") else "PASS"})  # fmt: skip
    return {"id": sid, "folder": folder.name, "report": report, "ranges": ranges, "stones": out_stones,
            "eligibility": eligibility(report, ranges), "qc": qc_overall(report, stones), "qc_items": qc_items(report, method, ranges),
            "integrity": integrity()}  # fmt: skip


def qc_items(report: dict, method: dict, ranges: dict | None) -> list[dict]:
    """The audit trail: one row per processing step with PASS / REVIEW / WARNING / FAIL / NOT AVAILABLE."""
    items = []
    for c in (report.get("input_checks") or {}).get("checks", []):
        name = (c["name"].replace("_", " ").capitalize() + " ").replace(" hu ", " HU ").replace(" nan ", " NaN ").strip()
        items.append({"group": "Input validation", "item": name, "status": status_of_check(c["status"]), "detail": c.get("detail", "")})
    cc = report.get("contrast_check") or {}
    a = cc.get("aorta_median_hu")
    items.append({"group": "Input validation", "item": "Contrast phase (aorta heuristic)",
                  "status": "NOT AVAILABLE" if a is None else ("PASS" if a < CONTRAST_HU else "WARNING"),
                  "detail": f"Aorta median {a:.0f} HU (threshold {CONTRAST_HU:.0f} HU). {cc.get('note', '')}" if a is not None else cc.get("note", "")})  # fmt: skip
    for s in report.get("stages") or []:
        items.append({"group": "Processing", "item": STAGE_NAMES.get(s["stage"], s["stage"]), "status": status_of_check(s["status"]),
                      "detail": f"{s.get('seconds', 0):.1f} s. {s.get('detail', '')}".strip()})  # fmt: skip
    k = report.get("kidney") or {}
    both = bool(k.get("right_volume_ml")) and bool(k.get("left_volume_ml"))
    items.append({"group": "Kidneys", "item": "Both kidneys segmented", "status": "PASS" if both else "REVIEW",
                  "detail": f"Right {k.get('right_volume_ml', 0):.1f} mL, left {k.get('left_volume_ml', 0):.1f} mL (segmentation volume)."})  # fmt: skip
    kd = method.get("kidney_distance_filter") or {}
    removed = kd.get("removed_pieces") or []
    items.append({"group": "Stones", "item": f"Kidney-distance filter ({kd.get('max_distance_mm', 30)} mm)", "status": "REVIEW" if removed else "PASS",
                  "detail": f"{len(removed)} piece(s) removed: " + ", ".join(f"#{p['stone_id']} {p['volume_mm3']:.0f} mm³ at {p['distance_to_kidney_mm']:.0f} mm" for p in removed) if removed else "No piece removed."})  # fmt: skip
    items.append({"group": "Stones", "item": "Tiny pieces (< 3 voxels) dropped", "status": "PASS",
                  "detail": f"{method.get('dropped_tiny_pieces', 0)} piece(s), {method.get('dropped_tiny_voxels', 0)} voxel(s)."})  # fmt: skip
    flagged = [s for s in method.get("stones", []) if s.get("qc_flags")]
    items.append({"group": "Stones", "item": "Per-stone QC flags", "status": "REVIEW" if flagged else "PASS",
                  "detail": "; ".join(f"#{s['stone_id']}: {', '.join(s['qc_flags'])}" for s in flagged) or "No stone flagged."})  # fmt: skip
    items.append({"group": "Uncertainty", "item": "Calibrated ranges (frozen calibration)", "status": "PASS" if ranges else "NOT AVAILABLE",
                  "detail": f"Calibration SHA-256 {ranges['calibration_quantities_sha256'][:12]}…, nominal {ranges['nominal']:.0%}." if ranges else "Not computed for this run."})  # fmt: skip
    return items


def volume_payload(sid: str) -> dict:
    """ROI crop of the CT (int16 HU) with kidney (1 right, 2 left) and stone-ID label volumes, zlib + base64."""
    folder = study_folder(sid)
    report = _read_json(folder / "report.json")
    method = (report.get("methods") or {}).get(report.get("primary_method") or "", {})
    box, kid, stones, spacing = _label_volumes(str(folder), method.get("label_file"), (folder / "report.json").stat().st_mtime)
    ct = load_volume(folder / "roi_crop.nii.gz").data
    if ct.shape != kid.shape:
        raise ValueError("ROI crop and label crop differ in shape")
    kid2 = np.select([kid == KIDNEY_RIGHT, kid == KIDNEY_LEFT], [1, 2], 0).astype(np.uint8)

    def pack(a, dt):
        return base64.b64encode(zlib.compress(np.ascontiguousarray(a, dtype=dt).tobytes(), 6)).decode("ascii")

    centres = {}
    if stones is not None:
        for sid_, sl in enumerate(ndimage.find_objects(stones), start=1):
            if sl is not None:
                v = np.argwhere(stones[sl] == sid_)
                if len(v):
                    centres[sid_] = [int(np.rint(c + s.start)) for c, s in zip(v.mean(axis=0), sl, strict=True)]
    return {"shape": [int(s) for s in ct.shape], "spacing": [float(s) for s in spacing], "box_start": [s.start for s in box],
            "ct": pack(np.clip(ct, -32768, 32767), np.int16), "kidney": pack(kid2, np.uint8),
            "stones": pack(stones, np.uint16) if stones is not None else None, "centres": centres}  # fmt: skip


def meshes(sid: str) -> dict:
    folder = study_folder(sid)
    report = _read_json(folder / "report.json")
    method = (report.get("methods") or {}).get(report.get("primary_method") or "", {})
    out = {"kidneys": None, "stones": []}
    kf = report.get("kidney_mesh_file")
    if kf and (folder / kf).exists():
        v, f = read_obj(folder / kf)
        out["kidneys"] = {"v": np.round(v, 2).tolist(), "f": f.tolist()}
    for s in method.get("stones", [])[:60]:
        p = folder / s.get("mesh_file", "")
        if s.get("mesh_file") and p.exists():
            v, f = read_obj(p)
            out["stones"].append({"stone_id": s["stone_id"], "v": np.round(v, 2).tolist(), "f": f.tolist()})
    return out


# ----------------------------------------------------------------------------------------------- integrity and evidence
@lru_cache(maxsize=1)
def integrity() -> dict:
    """SHA-256 of the frozen artefacts the uncertainty layer depends on, against the T1 freeze manifest."""
    want = _read_json(FREEZE)["sha256"]
    keys = ["configs/system_pipeline_v1.yaml", "src/stonekit/uq/confirmatory.py", "src/stonekit/uq/conformal.py",
            "data/interim/oof_tables_20261112T082351Z/annotated.csv", "data/interim/oof_tables_20261112T082351Z/predicted.csv",
            "data/interim/oof_tables_20261112T082351Z/scans.csv", "docs/CONFIRMATORY_PROTOCOL_v3_DRAFT.md",
            "data/nnunet/nnUNet_results/Dataset501_StoneNCCT/nnUNetTrainer_100epochs__nnUNetPlans_4GB__3d_fullres/fold_0/checkpoint_final.pth"]  # fmt: skip
    rows = []
    for k in keys:
        p = ROOT / k
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        rows.append({"artefact": k.split("/")[-1], "path": k, "expected": want.get(k), "actual": got, "present": got is not None,
                     "ok": got is not None and got == want.get(k)})  # fmt: skip
    q = T1 / "analysis" / "calibration_quantities.json"
    rec = (T1 / "analysis" / "calibration_quantities.sha256").read_text().split()[0] if q.exists() else None
    got = hashlib.sha256(q.read_bytes()).hexdigest() if q.exists() else None
    rows.append({"artefact": "calibration_quantities.json", "path": str(q.relative_to(ROOT)), "expected": rec, "actual": got,
                 "present": got is not None, "ok": got is not None and got == rec})  # fmt: skip
    for r in rows:
        r["status"] = "PASS" if r["ok"] else ("FAIL" if r["present"] else "NOT AVAILABLE")
    n, ok = len(rows), sum(r["ok"] for r in rows)
    bad, missing = sum(r["status"] == "FAIL" for r in rows), sum(not r["present"] for r in rows)
    status = "FAIL" if bad else ("PASS" if ok == n else "NOT AVAILABLE")
    summary = (f"all {n} frozen artefacts match the T1 freeze manifest" if status == "PASS" else
               f"{bad} artefact(s) do not match the T1 freeze manifest" if bad else
               f"{ok} of {n} frozen artefacts verified; {missing} not present on this installation (restricted data or model weights)")
    return {"rows": rows, "all_ok": status == "PASS", "status": status, "verified": ok, "missing": missing, "mismatch": bad, "summary": summary,
            "protocol": "Confirmatory protocol v3 (frozen 13 November 2026)", "calibration_sha256": rec, "t1_run": T1.name}  # fmt: skip


def _csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def evidence() -> dict:
    """Committed study aggregates (locked test, sensitivity, size bands, external check, phantom)."""
    ev = _read_json(T1 / "analysis" / "evaluation.json")
    b = ev["burden"]["0.90|filtered=True"]
    names = {"P2-0 naive": "Naive sum of per-stone ranges", "P2-B direct": "Direct split conformal", "P2-B-CQR": "Conformalised quantile regression", "P2-A decomposition": "Detection-aware (StoneKit)"}
    table = [{"method": names[m], "key": m, "coverage": b[m]["coverage"], "n": b[m]["n"], "cp95": b[m]["cp95"],
              "median_width_mm3": b[m]["median_width_mm3"], "stone_free": b[m]["coverage_stone_free"]} for m in names]  # fmt: skip
    ext = _read_json(EXT)
    replay = _read_json(REPLAY) if REPLAY.exists() else None
    return {"locked_test": {"table": table, "ratio": b["width_ratio_A_over_BCQR"], "formal": ev["formal_P2"], "n": ev["test_scans"],
                            "paired_coverage_difference": b["paired_coverage_difference_A_minus_BCQR"]},
            "robustness": _csv(FIGS / "fig3_burden_robustness_data.csv"), "size_bands": _csv(FIGS / "fig4_size_bands_data.csv"),
            "phantom": _csv(FIGS / "fig2_phantom_data.csv"),
            "external": {"eligible": ext["eligible_scans"], "cases": ext["cases_with_stone_label"], "burden": ext["burden_0.90"],
                         "per_stone": {k: v["marginal"] | {"median_rel_width": v["median_rel_width"]} for k, v in ext["per_stone"].items()},
                         "detection": ext["detection"], "fp_per_scan": ext["false_positives_per_scan_filtered"]},
            "posthoc_stone_bearing": (replay or {}).get("0.90|filtered=True", {}).get("stone_bearing")}  # fmt: skip
