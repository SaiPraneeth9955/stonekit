"""Characterise MSWAL from its LABEL masks and dataset.json only (no image downloaded or read).

For protocol v3 §5.4 eligibility planning: which released cases contain kidney stones (label 2), how many stone
components (26-connected) and their sizes, voxel spacing, and whether dataset.json carries contrast-phase
information. Counts only; nothing is tuned and no model is run. Output: a new results/runs folder.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from stonekit.config import create_run_folder

ROOT = Path("data/external/mswal")
STONE = 2


def main() -> None:
    meta = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
    rows = []
    for f in sorted((ROOT / "labelsTr").glob("MSWAL_*.nii.gz")):
        img = nib.load(f)
        lab = np.asarray(img.dataobj)
        sp = [float(s) for s in img.header.get_zooms()[:3]]
        m = lab == STONE
        rec = {"case": f.name.replace(".nii.gz", ""), "spacing_mm": sp, "shape": list(lab.shape), "labels_present": sorted(int(x) for x in np.unique(lab))}
        if m.any():
            comp, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
            vols = ndimage.sum_labels(np.ones_like(comp, dtype=np.int8), comp, index=np.arange(1, n + 1)) * float(np.prod(sp))
            rec.update({"stone_components": int(n), "stone_volumes_mm3": sorted((float(v) for v in vols), reverse=True),
                        "stone_burden_mm3": float(vols.sum()), "largest_sphere_eq_diam_mm": float((6 * vols.max() / np.pi) ** (1 / 3))})  # fmt: skip
        else:
            rec["stone_components"] = 0
        rows.append(rec)
    stone = [r for r in rows if r["stone_components"] > 0]
    vols = np.array([v for r in stone for v in r["stone_volumes_mm3"]])
    d = (6 * vols / np.pi) ** (1 / 3) if len(vols) else np.array([])
    slice_mm = np.array([r["spacing_mm"][2] for r in stone])
    phase_keys = [k for k in meta if any(s in k.lower() for s in ("phase", "contrast", "modality", "channel"))]
    summary = {
        "released_cases": len(rows),
        "cases_with_kidney_stone_label": len(stone),
        "stone_components_total": len(vols),
        "components_by_sphere_eq_diameter": {"<3": int((d < 3).sum()), "3-<5": int(((d >= 3) & (d < 5)).sum()), "5-<10": int(((d >= 5) & (d < 10)).sum()), ">=10": int((d >= 10).sum())},
        "stone_case_slice_thickness_mm": {"min": float(slice_mm.min()) if len(slice_mm) else None, "median": float(np.median(slice_mm)) if len(slice_mm) else None, "max": float(slice_mm.max()) if len(slice_mm) else None},
        "dataset_json_keys": sorted(meta.keys()),
        "dataset_json_phase_related": {k: meta[k] for k in phase_keys},
        "dataset_json_labels": meta.get("labels"),
        "dataset_json_licence_fields": {k: meta[k] for k in meta if "licen" in k.lower() or "reference" in k.lower() or "release" in k.lower()},
        "per_case_phase_available": any("phase" in json.dumps(x).lower() for x in (meta.get("training") or [])[:5]),
    }  # fmt: skip
    run = create_run_folder(Path("results/runs"), "mswal_label_characterisation")
    (run / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (run / "cases.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("Saved", run)


if __name__ == "__main__":
    main()
