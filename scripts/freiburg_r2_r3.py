"""R2 (per-stone error decomposition) and R3 (2D vs 3D against caliper) on Freiburg.

Protocol: docs/FREIBURG_R2_R3_PROTOCOL.md (pre-specified 11 November 2026, after rep1). Reuses rep1's
loaders and segmentation primitive (stonekit.replication.neubauer.stone_profile) with one parameter
value per method, so masks and diameters are computed exactly as in the replication. Parameters come
from the rep1 run folder (mean optima) or from the paper's Table 1 (sensitivity). CPU, Freiburg only.

Usage: python scripts/freiburg_r2_r3.py results/runs/rep1_neubauer_replication_<stamp>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from stonekit.config import create_run_folder, load_yaml
from stonekit.data.freiburg import CSV_NAME, MAT_NAME, load_mat_stones, load_table, match_stones
from stonekit.data.registry import find_project_root
from stonekit.replication import neubauer as nb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_neubauer_replication as rep1_runner

SEED = 20261111
RESAMPLES = 10000
SIZE_BINS = [(0, 10, "<10 mm"), (10, 20, "10-<20 mm"), (20, np.inf, ">=20 mm")]
PAPER = {"M1": {"P1": 272.0}, "M2": {"P1": 194.0, "P2": 0.29}, "M3": {"P1": 0.48}, "M4": {"P1": 0.43, "P2": 0.11}}
CONFIG_OF = {"P": "P_Z2_surface6_3d", "A": "A_Z2_surface6_3d"}
GEOMETRY_OF = {"P": "P_Z2", "A": "A_Z2"}


def median_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, float)
    if len(values) < 2:
        return [float("nan"), float("nan")]
    draws = rng.integers(0, len(values), size=(RESAMPLES, len(values)))
    meds = np.median(values[draws], axis=1)
    return [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]


def measure(stone, method: str, params: dict, geometry_name: str, geometries: dict, voxel_ml: float, edge: float, min_vox: int) -> dict:
    threshold = params["P1"] if method in ("M1", "M2") else params["P1"] * nb.rough_median_hu(stone.image)
    prof = nb.stone_profile(stone.image, np.array([threshold]), {geometry_name: geometries[geometry_name]}, edge, min_vox)
    voxels, surface = float(prof.voxels[0]), float(prof.surface_6[0])
    pv = params.get("P2", 0.0) if method in ("M2", "M4") else 0.0
    volume = float(nb.corrected_volume_ml(voxels, surface, pv, 1.0)) * voxel_ml
    return {"volume_ml": volume, "d3_paper_mm": float(prof.diameter_3d_mm[geometry_name][0]),
            "d2_axial_paper_mm": float(prof.diameter_axial_mm[geometry_name][0]),
            "d_volume_equivalent_mm": (6 * max(volume, 0.0) * 1000 / np.pi) ** (1 / 3)}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rep1_run", type=Path)
    args = parser.parse_args()
    root = find_project_root()
    rep1 = json.loads((args.rep1_run / "summary.json").read_text())
    config = load_yaml(root / "configs/rep1_neubauer_replication.yaml")
    raw = root / config["raw_folder"]
    matched = match_stones(load_table(raw / CSV_NAME), load_mat_stones(raw / MAT_NAME))
    ids = sorted(matched)
    stones = [matched[i] for i in ids]
    types = {r["stone_id"]: (r["stone_type_raw"] or "").strip() for r in load_table(raw / CSV_NAME)}
    rows = []
    for spacing_name in ("P", "A"):
        table1 = rep1["configurations"][CONFIG_OF[spacing_name]]["table1"]
        ours = {m: {k: table1[m][k]["mean"] for k in ("P1", "P2") if k in table1[m]} for m in PAPER}
        for i, stone in zip(ids, stones, strict=True):
            geometries, voxel_ml = rep1_runner.geometries_for(stone, config)
            g = GEOMETRY_OF[spacing_name]
            for source, params_all in (("rep1", ours), ("paper", PAPER)):
                for method in PAPER:
                    m = measure(stone, method, params_all[method], g, geometries, voxel_ml[g], config["paper_voxel_edge_mm"], config["min_component_voxels"])
                    rows.append({"stone": i, "spacing": spacing_name, "parameters": source, "method": method,
                                 "ref_volume_ml": stone.volume_ml, "ref_diameter_mm": stone.diameter_mm,
                                 "rough_median_hu": nb.rough_median_hu(stone.image), "stone_type": types.get(i, ""), **m})  # fmt: skip
        print("measured spacing", spacing_name, flush=True)
    df = pd.DataFrame(rows)
    df["rel_vol_err"] = (df.volume_ml - df.ref_volume_ml) / df.ref_volume_ml
    df["abs_rel_vol_err"] = df.rel_vol_err.abs()
    for d in ("d3_paper_mm", "d2_axial_paper_mm", "d_volume_equivalent_mm"):
        df[f"err_{d}"] = df[d] - df.ref_diameter_mm
    df["size_bin"] = pd.cut(df.ref_diameter_mm, [b[0] for b in SIZE_BINS] + [np.inf], right=False, labels=[b[2] for b in SIZE_BINS])
    df["hu_tertile"] = df.groupby(["spacing", "parameters", "method"])["rough_median_hu"].transform(lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"]).astype(str))
    counts = df[(df.spacing == "P") & (df.parameters == "rep1") & (df.method == "M1")].stone_type.replace("", "not recorded").value_counts()
    keep = {t for t, n in counts.items() if n >= 5}
    df["type_group"] = df.stone_type.replace("", "not recorded").map(lambda t: t if t in keep else "other")
    rng = np.random.default_rng(SEED)
    out = {"protocol": "docs/FREIBURG_R2_R3_PROTOCOL.md", "rep1_run": str(args.rep1_run), "stones": len(ids),
           "stone_type_counts": counts.to_dict(), "R2": {}, "R3": {}}  # fmt: skip
    for (sp, src, method), g in df.groupby(["spacing", "parameters", "method"]):
        key = f"{sp}|{src}|{method}"
        r2 = {}
        for col in ("size_bin", "hu_tertile", "type_group"):
            r2[col] = {str(k): {"n": len(s), "median_abs_rel_vol_err": float(s.abs_rel_vol_err.median()),
                                "ci95": median_ci(s.abs_rel_vol_err, rng), "median_d3_err_mm": float(s.err_d3_paper_mm.median()),
                                "d3_err_ci95": median_ci(s.err_d3_paper_mm, rng)}
                       for k, s in g.groupby(col, observed=True)}  # fmt: skip
        r2["all"] = {"n": len(g), "median_abs_rel_vol_err": float(g.abs_rel_vol_err.median()), "ci95": median_ci(g.abs_rel_vol_err, rng)}
        out["R2"][key] = r2
        r3 = {}
        for d in ("d3_paper_mm", "d2_axial_paper_mm", "d_volume_equivalent_mm"):
            e = g[f"err_{d}"]
            r3[d] = {"bias_mm": float(e.mean()), "mae_mm": float(e.abs().mean()),
                     "loa95_mm": [float(e.mean() - 1.96 * e.std(ddof=1)), float(e.mean() + 1.96 * e.std(ddof=1))],
                     "category_agreement_10mm": float(((g[d] >= 10) == (g.ref_diameter_mm >= 10)).mean())}  # fmt: skip
        diff = g.err_d2_axial_paper_mm.abs() - g.err_d3_paper_mm.abs()
        r3["wilcoxon_abs_err_2d_minus_3d"] = {"median_diff_mm": float(diff.median()), "p": float(stats.wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0}
        ok2 = (g.d2_axial_paper_mm >= 10) == (g.ref_diameter_mm >= 10)
        ok3 = (g.d3_paper_mm >= 10) == (g.ref_diameter_mm >= 10)
        b, c = int((ok2 & ~ok3).sum()), int((~ok2 & ok3).sum())
        r3["mcnemar_10mm"] = {"2d_right_3d_wrong": b, "2d_wrong_3d_right": c, "exact_p": float(stats.binomtest(min(b, c), b + c).pvalue) if b + c else 1.0}
        r3["caliper_below_5mm"] = int((g.ref_diameter_mm < 5).sum())
        out["R3"][key] = r3
    run = create_run_folder(root / "results/runs", "freiburg_r2_r3")
    df.to_csv(run / "per_stone.csv", index=False)
    (run / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    p = out["R3"]["P|rep1|M1"]
    print(json.dumps({"R2 all P|rep1|M1": out["R2"]["P|rep1|M1"]["all"], "R2 size": out["R2"]["P|rep1|M1"]["size_bin"], "R3": p}, indent=1, default=float))
    print("Saved", run)


if __name__ == "__main__":
    main()
