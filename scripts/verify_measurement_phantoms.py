"""How much error does the measurement step add by itself, before any segmentation error?

Digital geometric shapes of known size (spheres and 2:1 prolate ellipsoids) are voxelised by
voxel-centre inclusion at three voxel spacings and measured with StoneKit's own code
(separate_stones + measure_stones). These are verification objects for the measurement code, not
patients and not a dataset; no result here describes a patient scan.

Fixed before running: diameters 2, 3, 4, 5, 7, 10, 15, 20 mm; spacings 0.7363 x 0.7363 x 0.7 mm
(Freiburg), 0.75 x 0.75 x 1.25 mm and 0.75 x 0.75 x 2.5 mm (Li-like thin and thick slices); 25
random sub-voxel centre offsets per shape and spacing, seed 20261105. True volume is analytic; true
diameter is the long axis. StoneKit's diameter is centre-to-centre between voxels, so it is expected
to fall short by about one voxel along the measured direction.
"""

import argparse
import json

import numpy as np

from stonekit.config import create_run_folder
from stonekit.measure.stones import measure_stones, separate_stones

SEED = 20261105
DIAMETERS = (2, 3, 4, 5, 7, 10, 15, 20)
SPACINGS = {"freiburg_0.74x0.74x0.7": (0.7363, 0.7363, 0.7), "thin_0.75x0.75x1.25": (0.75, 0.75, 1.25), "thick_0.75x0.75x2.5": (0.75, 0.75, 2.5)}
SHAPES = {"sphere": (1.0, 1.0, 1.0), "prolate_2to1_along_z": (0.5, 0.5, 1.0), "prolate_2to1_in_plane": (1.0, 0.5, 0.5)}


def voxelise(diameter, axes, spacing, offset):
    semi = np.array(axes) * diameter / 2
    half = np.ceil(semi / spacing).astype(int) + 2
    grids = [(np.arange(-h, h + 1) + o) * s for h, o, s in zip(half, offset, spacing, strict=True)]
    x, y, z = np.meshgrid(*grids, indexing="ij")
    return (x / semi[0]) ** 2 + (y / semi[1]) ** 2 + (z / semi[2]) ** 2 <= 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rng = np.random.default_rng(SEED)
    rows = []
    for spacing_name, spacing in SPACINGS.items():
        affine = np.diag([*spacing, 1.0])
        for shape_name, axes in SHAPES.items():
            for diameter in DIAMETERS:
                true_volume = np.pi / 6 * diameter**3 * np.prod(axes)
                for _ in range(25):
                    mask = voxelise(diameter, axes, np.array(spacing), rng.uniform(-0.5, 0.5, 3))
                    separation = separate_stones(mask, 1)
                    stones = measure_stones(separation, np.full(mask.shape, 1000.0, np.float32), affine, spacing)
                    found = len(stones)
                    stone = max(stones, key=lambda s: s["voxel_count"]) if stones else None
                    rows.append({
                        "spacing": spacing_name, "shape": shape_name, "diameter_mm": diameter, "pieces": found,
                        "volume_error_percent": 100 * (stone["volume_mm3"] - true_volume) / true_volume if stone else None,
                        "diameter_error_mm": stone["max_diameter_3d_mm"] - diameter if stone else None,
                    })  # fmt: skip
    summary = {}
    for spacing_name in SPACINGS:
        for shape_name in SHAPES:
            for diameter in DIAMETERS:
                group = [r for r in rows if (r["spacing"], r["shape"], r["diameter_mm"]) == (spacing_name, shape_name, diameter)]
                v = np.array([r["volume_error_percent"] for r in group if r["volume_error_percent"] is not None])
                d = np.array([r["diameter_error_mm"] for r in group if r["diameter_error_mm"] is not None])
                summary.setdefault(spacing_name, {}).setdefault(shape_name, {})[str(diameter)] = {
                    "n": len(group), "missing": len(group) - len(v), "not_one_piece": sum(r["pieces"] != 1 for r in group),
                    "volume_error_percent_mean": round(float(v.mean()), 2) if len(v) else None,
                    "volume_error_percent_range": [round(float(v.min()), 2), round(float(v.max()), 2)] if len(v) else None,
                    "diameter_error_mm_mean": round(float(d.mean()), 3) if len(d) else None,
                    "diameter_error_mm_range": [round(float(d.min()), 3), round(float(d.max()), 3)] if len(d) else None,
                }  # fmt: skip
    run = create_run_folder(__import__("pathlib").Path("results/runs"), "measurement_phantom_verification")
    (run / "summary.json").write_text(json.dumps({"definitions": __doc__, "summary": summary}, indent=2), encoding="utf-8")
    for spacing_name in SPACINGS:
        print(spacing_name)
        for shape_name in SHAPES:
            line = [f"{d}:{summary[spacing_name][shape_name][str(d)]['volume_error_percent_mean']}%/{summary[spacing_name][shape_name][str(d)]['diameter_error_mm_mean']}mm" for d in DIAMETERS]
            print(" ", shape_name, " ".join(line))
    print("Saved to", run)


if __name__ == "__main__":
    main()
