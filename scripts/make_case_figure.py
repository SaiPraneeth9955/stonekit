"""Fig. 5 of the paper: StoneKit outputs for one external scan (MSWAL_0139), as produced by the workstation.

Inputs (local, not distributed): the run folder written by scripts/demo_case.py for MSWAL_0139, i.e. the frozen
pipeline outputs (report.json, roi_crop.nii.gz, kidney and stone label maps) and calibrated_ranges.json (the frozen
T1 calibration applied without refitting). No annotation is read: the figure shows what the system reports for a
new scan, not its accuracy. The scan was chosen by a fixed rule: the first eligible non-contrast MSWAL case (ID order)
with at least two retained pieces. MSWAL is CC BY-NC 4.0 (Wu et al., MICCAI 2025).

Panels: (a) coronal maximum-intensity projection of the kidney region with kidney outlines and retained stones;
(b) per-stone volume with its 90% range, coloured by false-detection probability; (c) the patient burden: measured
volume, expected volume after false-detection modelling, raw Monte Carlo interval, calibrated detection-aware
interval and the conformalised quantile-regression comparator.
Writes paper/figures/fig5_case.{pdf,png} and fig5_case_data.csv (the plotted values).
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from stonekit.io.nifti import load_volume

OUT = Path("paper/figures")
# label offsets (points) for MSWAL_0139 so that neighbouring stone numbers do not overlap
OFFSET = {1: (-7, -6), 3: (-7, 4), 4: (7, -4), 5: (7, 5), 6: (-7, 3), 7: (7, 3)}
ACC, SOFT, CMP, INK, INK2, STONE, KR, KL = "#0E6A85", "#B7D3DC", "#9AA6B0", "#1A1A1A", "#4D4D4D", "#D9822B", "#3C9C9A", "#5F86B3"
mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.8, "axes.labelsize": 7.8, "axes.titlesize": 7.8,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.4, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42, "savefig.dpi": 300,
})  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="data/interim/stonekit_runs/MSWAL_0139_<stamp>")
    run = ap.parse_args().run
    rep = json.loads((run / "report.json").read_text())
    cr = json.loads((run / "calibrated_ranges.json").read_text())
    if not rep["case_id"].startswith("MSWAL"):
        raise SystemExit("this figure is defined for an external (MSWAL) scan only")
    box = tuple(slice(a, b) for a, b in zip(rep["roi"]["box_start_voxel"], rep["roi"]["box_stop_voxel"], strict=True))
    ct = load_volume(run / "roi_crop.nii.gz")
    sx, _sy, sz = ct.spacing_mm
    kid = load_volume(run / "kidney_totalsegmentator.nii.gz").data[box]
    lab = load_volume(run / rep["methods"]["model"]["label_file"]).data[box]
    stones = {s["stone_id"]: s for s in cr["stones"]}
    y = np.where(np.isin(kid, [2, 3]).any(axis=(0, 2)))[0]  # coronal slab = antero-posterior extent of the kidneys
    slab = slice(max(y.min() - 5, 0), y.max() + 6)
    mip = ct.data[:, slab, :].max(axis=1)  # (x, z)
    img = np.clip((mip - (500 - 1000)) / 2000, 0, 1)  # window W 2000 / L 500
    # radiological display: patient right on screen left (x index increases toward the patient's left in RAS arrays here
    # is handled exactly as in the workstation viewer: screen column = nx - 1 - x), superior up
    disp = np.flipud(img[::-1, :].T)

    fig = plt.figure(figsize=(7.16, 2.4))
    a = fig.add_axes([0.0, 0.04, 0.33, 0.84])
    a.set_anchor("N")  # top-align the image (and its title) with panels (b) and (c)
    nx, nz = mip.shape
    a.imshow(disp, cmap="gray", extent=[0, nx * sx, 0, nz * sz], interpolation="nearest")
    for lbl, col in ((2, KR), (3, KL)):
        m = np.flipud((kid[:, slab, :] == lbl).any(axis=1)[::-1, :].T)
        a.contour(np.linspace(0, nx * sx, nx), np.linspace(nz * sz, 0, nz), m, levels=[0.5], colors=col, linewidths=0.7)
    rows = []
    for sid, s in stones.items():
        m = (lab[:, slab, :] == sid).any(axis=1)
        if not m.any():
            continue
        xs, zs = np.nonzero(m)
        cx, cz = (nx - 1 - xs.mean() + 0.5) * sx, (zs.mean() + 0.5) * sz
        hi = s["false_detection_probability"] >= 0.5
        mm = np.flipud(m[::-1, :].T)
        a.contour(np.linspace(0, nx * sx, nx), np.linspace(nz * sz, 0, nz), ndimage.binary_dilation(mm, iterations=1), levels=[0.5],
                  colors=CMP if hi else STONE, linewidths=0.9)  # fmt: skip
        a.annotate(str(sid), (cx, cz), xytext=OFFSET.get(sid, (5, 3)), textcoords="offset points", color="white", fontsize=7.2,
                   fontweight="bold", ha="center", va="center")  # fmt: skip
        rows.append({"panel": "a", "stone_id": sid, "centre_x_mm": cx, "centre_z_mm": cz})
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    a.text(4, nz * sz - 6, "R", color="white", fontsize=7.5, va="top", bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.2, "alpha": 0.55})
    a.text(nx * sx - 4, nz * sz - 6, "L", color="white", fontsize=7.5, va="top", ha="right", bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.2, "alpha": 0.55})
    a.plot([nx * sx - 40, nx * sx - 20], [9, 9], color="white", lw=1.2)
    a.text(nx * sx - 30, 13, "20 mm", color="white", fontsize=6.8, ha="center", va="bottom")
    a.set_title("(a) Coronal MIP, kidney region", loc="left", pad=3)

    b = fig.add_axes([0.455, 0.19, 0.205, 0.69])
    order = sorted(stones.values(), key=lambda s: s["volume_mm3"], reverse=True)
    yy = np.arange(len(order))[::-1]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("pi", [ACC, "#C8CED3"])
    for yv, s in zip(yy, order, strict=True):
        lo, hi = s["volume_range_mm3"]
        col = cmap(s["false_detection_probability"])
        b.plot([lo, hi], [yv, yv], color=col, lw=2.2, solid_capstyle="butt")
        b.plot([s["volume_mm3"]], [yv], marker="|", color=INK, ms=9, mew=1.6)
        rows.append({"panel": "b", "stone_id": s["stone_id"], "volume_mm3": s["volume_mm3"], "range_lo_mm3": lo, "range_hi_mm3": hi,
                     "max_diameter_3d_mm": s["max_diameter_3d_mm"], "false_detection_probability": s["false_detection_probability"]})  # fmt: skip
    b.set_xscale("log")
    b.set_xlim(0.5, 600)
    b.set_yticks(yy)
    b.set_yticklabels([f"Stone {s['stone_id']}  ($\\pi$ {s['false_detection_probability']:.2f})" for s in order])
    b.tick_params(axis="y", length=0)
    b.set_xlabel("Volume (mm$^3$), bar: 90% range")
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    b.set_title("(b) Per-stone ranges", loc="left", pad=3)

    c = fig.add_axes([0.80, 0.19, 0.195, 0.69])
    bd = cr["burden"]
    d = bd["decomposition"]
    cq = bd["comparators"]["P2-B-CQR"]
    labels = ["CQR comparator", "Detection-aware", "Raw Monte Carlo"]
    ivs = [(cq, CMP, 5), (d["calibrated_interval_mm3"], ACC, 6), (d["raw_interval_mm3"], SOFT, 5)]
    for k, (iv, col, lw) in enumerate(ivs):
        c.plot(iv, [k, k], color=col, lw=lw, solid_capstyle="butt")
        c.text(iv[1] + 40, k, f"{iv[0]:.0f}–{iv[1]:.0f}", va="center", fontsize=7.0, color=INK2)
    c.plot([bd["predicted_mm3"]], [3], marker="|", color=INK, ms=10, mew=1.8)
    c.plot([d["expected_after_false_detection_mm3"]], [3], marker="D", color=STONE, ms=4.2)
    c.text(bd["predicted_mm3"] + 40, 3, f"{bd['predicted_mm3']:.0f}", va="center", fontsize=7.0, color=INK2)
    c.text(d["expected_after_false_detection_mm3"], 3.45, f"$\\Sigma(1-\\pi)\\hat V$ = {d['expected_after_false_detection_mm3']:.0f}",
           ha="center", fontsize=7.0, color=STONE)  # fmt: skip
    c.set_yticks([0, 1, 2, 3])
    c.set_yticklabels([*labels, "Measured"])
    c.tick_params(axis="y", length=0)
    c.set_ylim(-0.6, 3.9)
    c.set_xlim(0, max(cq[1], d["calibrated_interval_mm3"][1]) * 1.42)
    c.set_xlabel("Stone burden (mm$^3$)")
    for sp in ("top", "right"):
        c.spines[sp].set_visible(False)
    c.set_title("(c) Patient burden (90%)", loc="left", pad=3)
    rows += [{"panel": "c", "quantity": n, "lo_mm3": iv[0], "hi_mm3": iv[1]} for n, (iv, _, _) in zip(labels, ivs, strict=True)]
    rows += [{"panel": "c", "quantity": "measured", "value_mm3": bd["predicted_mm3"]},
             {"panel": "c", "quantity": "expected_after_false_detection", "value_mm3": d["expected_after_false_detection_mm3"]},
             {"panel": "c", "quantity": "conformal_factor_t", "value": d["conformal_factor_t"]}]  # fmt: skip
    fig.savefig(OUT / "fig5_case.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig5_case.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    pd.DataFrame(rows).to_csv(OUT / "fig5_case_data.csv", index=False)
    print("wrote fig5_case; calibration", cr["calibration_quantities_sha256"][:12])


if __name__ == "__main__":
    main()
