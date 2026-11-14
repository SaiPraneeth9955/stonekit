"""Protocol v3 §5.4 external check on MSWAL (secondary, exploratory; run ONCE after T1 has COMPLETED).

Specification is in the frozen protocol. Implementation rules, written before any MSWAL image is processed:
- Cases: the 77 released MSWAL cases with kidney-stone voxels (label 2). The stone reference is label 2 only.
- Eligibility (non-contrast): the pipeline's own contrast check in report.json, aorta median HU < 100 (the
  existing heuristic threshold). A case whose aorta median is missing cannot be shown to be non-contrast and is
  listed as ineligible ("phase unknown"). A case that fails processing has no contrast check: it is listed and
  excluded from estimation ("failed, eligibility unknown").
- Fewer than 10 eligible stone-bearing scans -> "insufficient for estimation"; counts only.
- The frozen system (config, fold-0 checkpoint, 30 mm filter, 8-flip TTA) is run unchanged. Intervals come from
  the T1 run's hashed calibration quantities (stonekit.t1.external); nothing is fitted on MSWAL.
- Endpoints at 0.90: detected share by size; false positives per scan (filtered); per-stone coverage and width
  for U0, U1, U2 and U2-asymmetric (volume and 3D diameter); burden coverage and width for P2-0, P2-B,
  P2-B-CQR and P2-A (the OD-1 statistics are reported descriptively; no formal test here).
"""

import argparse
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from stonekit.config import create_run_folder, load_yaml
from stonekit.data.li2022 import LiCase
from stonekit.io.nifti import load_volume
from stonekit.pipeline.run_case import run_case
from stonekit.t1.analysis import _summ
from stonekit.t1.external import load_quantities, rebuild_burden, stone_bounds
from stonekit.uq import confirmatory as cfm
from stonekit.uq import conformal as cf

ROOT = Path(".")
CONFIG = Path("configs/system_pipeline_v1.yaml")
MSWAL = Path("data/external/mswal")
STONE_LABEL = 2
CONTRAST_HU = 100.0
MIN_ELIGIBLE = 10


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class MswalData:
    """The interface build_oof_stone_table.process needs: load(case) -> image plus a stone-only mask (1 = stone)."""

    def paths(self, case: str) -> tuple[Path, Path]:
        return MSWAL / "imagesTr" / f"{case}_0000.nii.gz", MSWAL / "labelsTr" / f"{case}.nii.gz"

    def load(self, case: str) -> LiCase:
        img_p, lab_p = self.paths(case)
        image, lab = load_volume(img_p), load_volume(lab_p)
        if lab.data.shape != image.data.shape or not np.allclose(lab.affine, image.affine, atol=1e-3):
            raise ValueError(f"{case}: label and image geometry differ")
        return LiCase(case, image, replace(lab, data=(lab.data == STONE_LABEL).astype(np.uint8)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("t1_run", type=Path, help="results/runs/T1_<stamp> (must contain COMPLETED)")
    ap.add_argument("--fit-table", type=Path, default=Path("data/interim/oof_tables_20261112T082351Z"))
    ap.add_argument("--summarise-only", type=Path, default=None,
                    help="existing mswal_external_check run folder: recompute only the summary from its saved tables (no case is reprocessed)")  # fmt: skip
    args = ap.parse_args()
    if not (args.t1_run / "COMPLETED").exists():
        raise SystemExit("T1 has not completed; §5.4 runs only after T1")
    q = load_quantities(args.t1_run / "analysis")
    fit = cfm.Tables(*(pd.read_csv(args.fit_table / f) for f in ("annotated.csv", "predicted.csv", "scans.csv")))
    burden_cal = rebuild_burden(q["burden"]["0.1|filtered=True"], fit)
    if args.summarise_only is not None:
        run = args.summarise_only
        if (run / "external_check.json").exists():
            raise SystemExit("summary already written; not overwriting")
        t = cfm.Tables(*(pd.read_csv(run / f) for f in ("annotated.csv", "predicted.csv", "scans.csv")))
        status = pd.read_csv(run / "eligibility.csv").to_dict("records")
        summarise(args, q, burden_cal, run, t, status, n_cases=len(status))
        return
    subset = json.loads(Path(max(Path("results/runs").glob("mswal_label_characterisation_2*")), "stone_case_image_subset.json").read_text())
    cases = subset["stone_cases"]
    run = create_run_folder(Path("results/runs"), "mswal_external_check")
    config = load_yaml(ROOT / CONFIG)
    model, fold = dict(config["model"]), int(config["model"]["fold"])
    tta_mod, table_mod = _script("oof_tta"), _script("build_oof_stone_table")
    data = MswalData()
    ann, pred, scans, status = [], [], [], []
    for i, case in enumerate(cases):
        ct, _ = data.paths(case)
        folder = run_case(ct, run / "cases", config, ROOT, case, None, None, "gpu", ROOT / CONFIG)
        rep = json.loads((folder / "report.json").read_text())
        aorta = (rep.get("contrast_check") or {}).get("aorta_median_hu")
        if rep.get("result_status") != "complete":
            status.append({"case": case, "status": "failed, eligibility unknown", "aorta_median_hu": aorta})
            continue
        if aorta is None or aorta >= CONTRAST_HU:
            status.append({"case": case, "status": "ineligible: contrast" if aorta is not None else "ineligible: phase unknown", "aorta_median_hu": aorta})
            continue
        tta_out = run / "tta" / case
        tta_out.mkdir(parents=True, exist_ok=False)
        tta_mod.process_case(folder, model, fold, tta_out, ROOT)
        a, p, s = table_mod.process(case, folder, fold, run / "tta", data)
        ann += a
        pred += p
        scans.append(s)
        status.append({"case": case, "status": "eligible", "aorta_median_hu": aorta})
        print(i + 1, "/", len(cases), case, "eligible", round(aorta), flush=True)
    pd.DataFrame(status).to_csv(run / "eligibility.csv", index=False)
    t = cfm.Tables(pd.DataFrame(ann), pd.DataFrame(pred), pd.DataFrame(scans))
    for name, df in (("annotated", t.annotated), ("predicted", t.predicted), ("scans", t.scans)):
        df.to_csv(run / f"{name}.csv", index=False)
    summarise(args, q, burden_cal, run, t, status, n_cases=len(cases))


def summarise(args, q, burden_cal, run: Path, t: cfm.Tables, status: list, n_cases: int) -> None:
    scans = t.scans
    cases = range(n_cases)
    out = {"protocol": "v3 §5.4 (secondary, exploratory)", "t1_run": str(args.t1_run), "calibration_quantities_sha256": (args.t1_run / "analysis" / "calibration_quantities.sha256").read_text().split()[0],
           "cases_with_stone_label": len(cases), "status_counts": pd.DataFrame(status)["status"].value_counts().to_dict(), "eligible_scans": len(scans)}  # fmt: skip
    if len(scans) < MIN_ELIGIBLE:
        out["result"] = f"insufficient for estimation (< {MIN_ELIGIBLE} eligible stone-bearing scans)"
    else:
        a = t.annotated
        d = (6 * a["ref_volume_mm3"] / np.pi) ** (1 / 3)
        bins = pd.cut(d, [0, 3, 5, 10, np.inf], right=False, labels=["<3", "3-<5", "5-<10", ">=10"])
        det = a["group"] != "missed"
        out["detection"] = {"detected_share": float(det.mean()), "n_annotated": len(a),
                            "by_size": {str(k): {"n": int((bins == k).sum()), "detected": float(det[bins == k].mean()) if (bins == k).any() else None} for k in bins.cat.categories}}  # fmt: skip
        p = t.predicted
        fp = p[(p["group"] == "false_positive") & ~p["beyond_30mm"].astype(bool)].groupby("case").size()
        out["false_positives_per_scan_filtered"] = float(t.scans["case"].map(fp).fillna(0).mean())
        mt = cfm.matched(t)
        out["per_stone"] = {}
        for target in cfm_targets():
            r = np.asarray(cf.log_ratio(mt[f"ref_{target}"], mt[f"pred_{target}"]), float)
            for method in ("U0", "U1", "U2", "U2-asymmetric"):
                lo, hi = stone_bounds(q["per_stone"], method, target, 0.1, mt["pred_max_diameter_3d_mm"].to_numpy())
                cov = (r >= lo) & (r <= hi)
                out["per_stone"][f"{target}|{method}"] = {"marginal": _summ(cov, mt.case.to_numpy()), "median_rel_width": float(np.median(np.exp(hi) - np.exp(lo)))}
        out["burden_0.90"] = cfm.p2_test(cfm.apply_burden(burden_cal, t), nominal=0.90)
        out["burden_0.90"]["note"] = "descriptive only on external data; the OD-1 formal test applies to T1 only"
    (run / "external_check.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_stone"}, indent=1, default=float)[:4000])
    print("Saved", run)


def cfm_targets() -> tuple[str, str]:
    return ("volume_mm3", "max_diameter_3d_mm")


if __name__ == "__main__":
    main()
