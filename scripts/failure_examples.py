"""Representative examples of each outcome, chosen by a fixed rule (never by eye).

Rule: within each category, the annotated piece whose defining value is closest to that category's
median. Categories (100-epoch model, 30 mm filter, threshold 0.5, from the detection diagnosis rows):
- missed, small: missed pieces < 3 mm, median 3D diameter;
- missed, faint: missed pieces >= 5 mm, median peak HU;
- under-segmented: one-to-one pairs, median predicted/annotated volume ratio;
- detected, large: one-to-one pairs >= 10 mm, median volume ratio.
Each panel: axial CT slice through the piece's centre (window W 1500 / L 300, radiological orientation,
patient right on image left), annotation outline in green, model outline in orange, 5 mm scale bar.
The figure contains patient images, so it is written ONLY to data/interim/figures/ (git-ignored) and
must not be committed or shared outside the Li licence terms.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from stonekit.data.li2022 import LiDevelopmentData
from stonekit.io.nifti import load_volume
from stonekit.measure.stones import separate_stones
from stonekit.pipeline.report_html import axial_display


def pick(rows, key, transform=lambda v: v):
    values = np.array([transform(r) for r in rows])
    return rows[int(np.argmin(np.abs(values - np.median(values))))] if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=Path, required=True, help="detection diagnosis detail folder")
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (args.detail / "pieces.csv").open(newline="", encoding="utf-8") as f:
        rows = [{k: (float(v) if k not in ("case",) and not k.startswith("kind_") and v not in ("", "None") else v) for k, v in r.items()} for r in csv.DictReader(f)]
    missed = [r for r in rows if r["kind_t0.5"] == "missed"]
    pairs = [r for r in rows if r["kind_t0.5"] == "one_to_one"]
    ratio = lambda r: r["pred_mm3_t0.5"] / r["volume_mm3"]
    chosen = [
        ("Missed: small (<3 mm)", pick([r for r in missed if r["d3_mm"] < 3], "d3", lambda r: r["d3_mm"])),
        ("Missed: faint (>=5 mm)", pick([r for r in missed if r["d3_mm"] >= 5], "hu", lambda r: r["peak_hu"])),
        ("Detected, under-segmented (median ratio)", pick(pairs, "ratio", ratio)),
        ("Detected, >=10 mm (median ratio)", pick([r for r in pairs if r["d3_mm"] >= 10], "ratio", ratio)),
    ]
    runs = {json.loads((p / "report.json").read_text())["case_id"]: p for p in args.runs.iterdir() if (p / "report.json").exists()}
    data = LiDevelopmentData(Path.cwd())
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (title, row) in zip(axes, chosen, strict=True):
        ax.axis("off")
        if row is None:
            ax.set_title(f"{title}\n(no piece in this category)", fontsize=8)
            continue
        li = data.load(row["case"])
        reference = separate_stones(li.stone_reference, 3).labels == int(row["piece"])
        model = load_volume(runs[row["case"]] / "stone_labels_model.nii.gz").data > 0
        found = np.argwhere(reference)
        cx, cy, cz = (int(v) for v in found.mean(axis=0).round())
        half = [int(np.ceil(15 / s)) for s in li.image.spacing_mm[:2]]
        window = (slice(max(cx - half[0], 0), cx + half[0]), slice(max(cy - half[1], 0), cy + half[1]), cz)
        ct = axial_display(li.image.data[window].astype(float))
        ref = axial_display(reference[window])
        pred = axial_display(model[window])
        ax.imshow(np.clip((ct - (300 - 750)) / 1500, 0, 1), cmap="gray", origin="lower", interpolation="nearest")
        for mask, colour in ((ref, "#3fbf5a"), (pred, "#f28c28")):
            edge = mask & ~ndimage.binary_erosion(mask)
            if edge.any():
                ax.contour(mask.astype(float), levels=[0.5], colors=colour, linewidths=1.0, origin="lower")
        bar = 5 / li.image.spacing_mm[0]
        ax.plot([2, 2 + bar], [2, 2], color="white", lw=2)
        ax.text(2, 4, "5 mm", color="white", fontsize=7)
        extra = f"annot. {row['d3_mm']:.1f} mm, peak {row['peak_hu']:.0f} HU, max p {row['max_probability']:.2f}"
        if row["kind_t0.5"] == "one_to_one":
            extra += f"\npred/annot volume {ratio(row):.2f}"
        ax.set_title(f"{title}\n{extra}", fontsize=7.5)
    fig.suptitle("Representative outcomes (rule-selected; green = annotation, orange = model, 100 epochs + 30 mm filter)", fontsize=9)
    fig.tight_layout()
    out = Path("data/interim/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "failure_examples_rule_selected.png", dpi=170)
    print("Saved", out / "failure_examples_rule_selected.png")


if __name__ == "__main__":
    main()
