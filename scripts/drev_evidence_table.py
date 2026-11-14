"""Turn the D-REV technical audit (audit.json) into a one-row-per-location evidence table (CSV + JSON).

The indicator rules below were written BEFORE the per-location audit values were read (13 November 2026), and none
is tuned to the data. They describe objective image facts only; they are not adjudication answers, and they never
change the screening record, the blinded sheet or the cohort.

Indicators (each True/False/None):
- contrast_scan_indicator: the aorta median HU OR the kidney-parenchyma median HU of the scan is above the maximum
  of the unflagged development scans (an empirical reference, no absolute threshold assumed).
- metal_indicator: the nearest bright component contains any voxel >= 2500 HU (the project's metal-like QC limit).
- large_structure_indicator: the nearest bright component reaches the edge of the 80 mm cube (larger than any
  development stone; consistent with bone or vessel, NOT proof of either).
- outside_kidney_indicator: the point is > 10 mm from the TotalSegmentator kidney.
- label_without_dense_voxel: the point carries the stone label but no voxel >= 130 HU lies within 3 mm.
- dense_without_label: a bright component lies within 5 mm, has no stone-label voxel, and is not large_structure.
Objective resolvability: "objective" when an indicator alone settles the screening question asked of the reader
(e.g. metal for a device question); otherwise "needs reader". The final column is always for the human reader.
"""

import csv
import json
import sys
from pathlib import Path


def latest(pattern: str) -> Path:
    runs = sorted(Path("results/runs").glob(pattern))
    if not runs:
        sys.exit("no audit run found")
    return runs[-1]


def main() -> None:
    run = latest("drev_technical_audit_*")
    a = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    ref = a["reference_unflagged_development"]
    out = []
    for x in a["locations"]:
        row = {k: x[k] for k in ("adj_id", "case", "target", "location_type")}
        row["R"], row["A"], row["S"] = x["ras"]
        row["audit_status"] = x["status"]
        if not x["status"].startswith("AUDITED"):
            row["resolvability"] = "needs reader (reserved scan; no technical evidence computed)"
            out.append(row)
            continue
        s, p, L = x["scan"], x["scan_percentiles_vs_unflagged_dev"], x["location"]
        c = L.get("bright_component") or {}
        row.update({
            "aorta_median_hu": s["aorta_median_hu"], "aorta_pct_vs_dev": p["aorta_median_hu"],
            "parenchyma_median_hu": s["kidney_parenchyma_median_hu"], "parenchyma_pct_vs_dev": p["kidney_parenchyma_median_hu"],
            "bright_kidney_share_R": s["bright_kidney_share_right"], "bright_kidney_share_L": s["bright_kidney_share_left"],
            "in_bounds": L["in_bounds"], "hu_at_point": L.get("hu_at_point"), "max_hu_3mm": L.get("max_hu_3mm"),
            "max_hu_5mm": L.get("max_hu_5mm"), "background_median_hu": L.get("background_median_hu_10_15mm"),
            "label_at_point": L.get("annotation_label_at_point"), "labels_within_5mm": " ".join(map(str, L.get("labels_within_5mm", []))),
            "dist_ts_kidney_mm": L.get("dist_to_ts_kidney_mm"), "dist_annot_kidney_mm": L.get("dist_to_annotation_kidney_mm"),
            "dist_annot_stone_mm": L.get("dist_to_annotation_stone_mm"), "dist_aorta_mm": L.get("dist_to_aorta_mm"),
            "comp_volume_ml": c.get("volume_ml"), "comp_max_hu": c.get("max_hu"), "comp_median_hu": c.get("median_hu"),
            "comp_metal_voxels": c.get("metal_like_voxels"), "comp_reaches_cube_edge": c.get("reaches_80mm_cube_edge"),
            "comp_axis_major_mm": c.get("axis_major_mm"), "comp_elongation": c.get("elongation"),
            "comp_share_in_stone_label": c.get("share_in_annotation_stone"), "comp_share_in_ts_kidney": c.get("share_in_ts_kidney"),
        })  # fmt: skip

        def num(v):
            return v if isinstance(v, (int, float)) else None

        ao, pa = num(s["aorta_median_hu"]), num(s["kidney_parenchyma_median_hu"])
        row["contrast_scan_indicator"] = bool((ao is not None and ao > ref["aorta_median_hu"]["max"]) or (pa is not None and pa > ref["kidney_parenchyma_median_hu"]["max"]))
        row["metal_indicator"] = None if not c else bool(c["metal_like_voxels"] > 0)
        row["large_structure_indicator"] = None if not c else bool(c["reaches_80mm_cube_edge"])
        dk = L.get("dist_to_ts_kidney_mm")
        row["outside_kidney_indicator"] = True if dk == "> 60" else (None if dk is None else bool(dk > 10))
        row["label_without_dense_voxel"] = bool(L.get("annotation_label_at_point") == 1 and (L.get("max_hu_3mm") or 0) < 130)
        row["dense_without_label"] = bool(c and c["share_in_annotation_stone"] == 0 and not c["reaches_80mm_cube_edge"])
        out.append(row)
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(run / "drev_evidence_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    (run / "drev_evidence_table.json").write_text(json.dumps({"source": str(run / "audit.json"), "rules": __doc__, "rows": out}, indent=1, default=str), encoding="utf-8")
    for r in out:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("R", "A", "S", "target")})
    print("Saved", run)


if __name__ == "__main__":
    main()
