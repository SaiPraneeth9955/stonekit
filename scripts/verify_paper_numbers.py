"""Claim-evidence verification: every key number in paper/main.tex is recomputed from its saved run folder and must
appear verbatim (as formatted) in the manuscript. Read-only; writes results/runs/paper_number_check_<stamp>/.
A failure means the text and the evidence disagree, and must be fixed before submission."""

import json
from pathlib import Path

from stonekit.config import create_run_folder

R = Path("results/runs")
WORDS = dict(enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]))  # manuscript spells counts <= 10


def j(p):
    return json.loads((R / p / "summary.json").read_text())


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def main() -> None:
    tex = Path("paper/main.tex").read_text(encoding="utf-8")
    r23 = j("freiburg_r2_r3_20261111T074847Z")
    r2, r3 = r23["R2"]["P|rep1|M1"], r23["R3"]["P|rep1|M1"]
    e6 = j("e6_uncertainty_design_20261112T082354Z")["results"]
    e7 = j("e7_detection_uncertainty_20261112T082355Z")
    e8 = j("e8_burden_uncertainty_20261112T085235Z")["results"]
    nb = j("secondary_na_nb_development_20261112T103848Z")["results"]["volume_mm3|0.90"]
    fa = j("failure_analysis_development_20261112T110426Z")
    checks = [
        ("R2 all-stone median volume error", pct(r2["all"]["median_abs_rel_vol_err"]) + "\\%"),
        ("R2 CI", f"{pct(r2['all']['ci95'][0])}--{pct(r2['all']['ci95'][1])}\\%"),
        ("R2 <10 mm", pct(r2["size_bin"]["<10 mm"]["median_abs_rel_vol_err"]) + "\\%"),
        ("R2 10-20 mm", pct(r2["size_bin"]["10-<20 mm"]["median_abs_rel_vol_err"]) + "\\%"),
        ("R2 >=20 mm", pct(r2["size_bin"][">=20 mm"]["median_abs_rel_vol_err"]) + "\\%"),
        ("R2 uric acid", pct(r2["type_group"]["uric acid"]["median_abs_rel_vol_err"]) + "\\%"),
        ("R2 brushite", pct(r2["type_group"]["brushite"]["median_abs_rel_vol_err"]) + "\\%"),
        ("R3 3D bias", f"$+{r3['d3_paper_mm']['bias_mm']:.2f}$"),
        ("R3 2D bias", f"$-{abs(r3['d2_axial_paper_mm']['bias_mm']):.2f}$"),
        ("R3 3D LoA", f"$-{abs(r3['d3_paper_mm']['loa95_mm'][0]):.2f}$ to $+{r3['d3_paper_mm']['loa95_mm'][1]:.2f}$"),
        ("R3 2D LoA", f"$-{abs(r3['d2_axial_paper_mm']['loa95_mm'][0]):.2f}$ to $+{r3['d2_axial_paper_mm']['loa95_mm'][1]:.2f}$"),
        ("R3 McNemar p", f"$p={r3['mcnemar_10mm']['exact_p']:.3f}$"),
        ("R3 McNemar counts", f"{WORDS[r3['mcnemar_10mm']['2d_wrong_3d_right']]} stones were categorised correctly only by 3D"),
        ("E7 detection <3 mm", f"{round(100 * e7['detection_by_size_bin']['<3']['share'])}\\%"),
        ("E7 detection >=10 mm", f"{round(100 * e7['detection_by_size_bin']['>=10']['share'])}\\%"),
        ("E7 HU odds ratio", f"{e7['detection_gee_odds_ratios']['hu100']['or']:.1f}-fold"),
        ("E6 U1 <5 mm", f"{e6['volume_mm3|0.90']['U1']['pred_stratum']['<5']['coverage']:.2f} at nominal 0.90"),
        ("E6 U2 bands", ", ".join(f"{e6['volume_mm3|0.90']['U2']['pred_stratum'][b]['coverage']:.2f}" for b in ("<5", "5-<10")) + f" and {e6['volume_mm3|0.90']['U2']['pred_stratum']['>=10']['coverage']:.2f}"),
        ("E6 low-HU", f"({e6['volume_mm3|0.90']['U2']['hu_tertile']['low']['coverage']:.2f} in the lowest HU tertile)"),
        ("Failure misses", f"{fa['miss_direction']['under_estimate_ref_above_interval']} of {fa['misses']} misses"),
        ("N-B inflation", f"{nb['N-B_rule']['width_inflation']:.2f}-fold"),
        ("N-B joint", f"{nb['N-B']['joint_patient']['coverage']:.2f} joint coverage"),
        ("E8 naive", f"naive sum covered {e8['0.90']['P2-0 naive']['coverage']:.2f}"),
        ("E8 P2-A", f"decomposition interval covered {e8['0.90']['P2-A decomposition']['coverage']:.2f}"),
        ("E8 P2-A width", f"{e8['0.90']['P2-A decomposition']['median_width_mm3']:,.0f}".replace(",", "{,}")),
        ("Missed share", f"{pct(fa['burden_drivers']['total_missed_over_total_burden'])}\\% of total burden"),
    ]
    results = [{"check": n, "expected_text": s, "found": s in tex} for n, s in checks]
    run = create_run_folder(R, "paper_number_check")
    (run / "summary.json").write_text(json.dumps({"checks": results, "all_found": all(r["found"] for r in results)}, indent=2), encoding="utf-8")
    for r in results:
        print(("OK   " if r["found"] else "MISS ") + r["check"] + " -> " + r["expected_text"])
    print("ALL FOUND:", all(r["found"] for r in results), "| Saved", run)


if __name__ == "__main__":
    main()
