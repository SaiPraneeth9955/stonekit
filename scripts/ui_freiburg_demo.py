"""Write a standalone viewer page from PUBLIC Freiburg phantom data (CC BY 4.0) for visual UI checks.

Patient (Li) images are never used for screenshots (docs/decisions.md, 11 November 2026). The page shows
one phantom stone patch with its M1 mask (rep1 mean threshold under the paper-stated 0.7 mm spacing).
Output goes to a git-ignored folder.

Usage: python scripts/ui_freiburg_demo.py --stone 1 --out results/raw/ui_demo_freiburg.html
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from stonekit.app.viewer import viewer_html, viewer_payload
from stonekit.config import load_yaml
from stonekit.data.freiburg import CSV_NAME, MAT_NAME, load_mat_stones, load_table, match_stones
from stonekit.data.registry import find_project_root

THRESHOLD_HU = 194.665  # rep1 primary M1 mean optimum (results/runs/rep1_neubauer_replication_20261111T074337Z)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stone", type=int, default=None, help="Freiburg stone ID (default: the largest)")
    parser.add_argument("--out", type=Path, default=Path("results/raw/ui_demo_freiburg.html"))
    args = parser.parse_args()
    root = find_project_root()
    raw = root / load_yaml(root / "configs/rep1_neubauer_replication.yaml")["raw_folder"]
    stones = match_stones(load_table(raw / CSV_NAME), load_mat_stones(raw / MAT_NAME))
    sid = args.stone or max(stones, key=lambda i: stones[i].volume_ml)
    image = np.nan_to_num(stones[sid].image.astype(np.float32), nan=-1000.0)
    image = np.pad(image, 12, constant_values=float(np.nanmin(image)))  # margin so the stone is not at the edge
    labels, _ = ndimage.label(image > THRESHOLD_HU, structure=np.ones((3, 3, 3)))
    payload = viewer_payload(image, (0.7, 0.7, 0.7), None, labels, selected=1,
                             focus_voxel=[int(v) for v in ndimage.center_of_mass(labels > 0)],
                             title=f"Freiburg phantom stone {sid} (public data)")  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(viewer_html(payload, 640), encoding="utf-8")
    print("stone", sid, "shape", image.shape, "pieces", int(labels.max()), "->", args.out)


if __name__ == "__main__":
    main()
