"""Merge the D-REV technical evidence into one row per location (CSV + JSON). Read-only; no image access.

Sources (latest run of each): drev_technical_audit (audit.json + drev_evidence_table.csv), drev_audit_supplement
(model labels at the point, burden percentiles), drev_component_trace (whole-volume structure).

"evidence_pattern" is a DESCRIPTIVE grouping written AFTER the trace values were seen (post hoc). It is not an
adjudication answer, not an A-F category, and it never changes the screening record, the blinded sheet or the
cohort. Rules:
- skeleton-connected: the structure belongs to the largest >= 130 HU component of the scan.
- long isolated tube: not skeleton-connected; craniocaudal extent >= 200 mm; volume <= 15 mL.
- large isolated mass: not skeleton-connected; volume >= 20 mL.
- small discrete focus: not skeleton-connected; volume < 1 mL.
- reserved: calibration/test scan, not audited.
"""

import csv
import json
from pathlib import Path


def latest(prefix: str) -> Path:
    return max(Path("results/runs").glob(prefix + "_2*"))


def pattern(t):
    if not t or "skeleton_connected" not in t:
        return "no dense structure within 5 mm"
    if t["skeleton_connected"]:
        return "skeleton-connected"
    if t["extent_mm_xyz"][2] >= 200 and t["volume_ml"] <= 15:
        return "long isolated tube"
    if t["volume_ml"] >= 20:
        return "large isolated mass"
    if t["volume_ml"] < 1:
        return "small discrete focus"
    return "other isolated structure"


def main() -> None:
    a_run, s_run, t_run = latest("drev_technical_audit"), latest("drev_audit_supplement"), latest("drev_component_trace")
    with open(a_run / "drev_evidence_table.csv", encoding="utf-8") as f:
        base = list(csv.DictReader(f))
    sup = json.loads((s_run / "supplement.json").read_text(encoding="utf-8"))
    tr = json.loads((t_run / "trace.json").read_text(encoding="utf-8"))
    model = {m["adj_id"]: m for m in sup["model_at_location"]}
    trace = {t["adj_id"]: t for t in tr["locations"]}
    rows = []
    for b in base:
        r = dict(b)
        aid = r["adj_id"]
        if not r["audit_status"].startswith("AUDITED"):
            r["evidence_pattern"] = "reserved (not audited)"
            rows.append(r)
            continue
        m, t = model.get(aid, {}), trace.get(aid, {})
        r["model_label_at_point"] = m.get("model_label_at_point")
        r["model_piece_volume_within_5mm_mm3"] = round(sum(m.get("model_piece_volumes_mm3", {}).values()), 1)
        bp = sup["burden_percentile"].get(r["case"], {})
        r["scan_annotated_burden_mm3"] = round(bp.get("annotated_burden_mm3", 0), 1)
        r["scan_burden_percentile_dev"] = round(bp.get("percentile_among_165_dev", 0), 3)
        for k in ("skeleton_connected", "volume_ml", "extent_mm_xyz", "reaches_image_edge", "median_hu", "max_hu", "share_in_stone_label", "share_in_ts_kidney"):
            v = t.get(k)
            r[f"trace_{k}"] = " x ".join(f"{x:.0f}" for x in v) if isinstance(v, list) else (round(v, 3) if isinstance(v, float) else v)
        r["evidence_pattern"] = pattern(t)
        rows.append(r)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    run = a_run
    with open(run / "drev_evidence_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (run / "drev_evidence_final.json").write_text(json.dumps({"sources": [str(a_run), str(s_run), str(t_run)], "rules": __doc__, "rows": rows}, indent=1, default=str), encoding="utf-8")
    for r in rows:
        print(r["adj_id"], r["case"], r["location_type"][:9], "|", r["evidence_pattern"], "| model", r.get("model_piece_volume_within_5mm_mm3"))
    print("Saved", run / "drev_evidence_final.csv")


if __name__ == "__main__":
    main()
