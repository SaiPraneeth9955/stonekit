"""Leakage audit: do suspected repeat scans of one patient straddle the split?

Li 2022 has no patient identifiers. Before the split (5 November 2026), the cohort audit saved, for
every one of the 260 scans, two normalised bone projections (data/interim/li2022_full_audit/proj/)
and, for the 30 most similar pairs and 40 random pairs, 4 mm bone volumes (vol4/). This script
recomputes exactly that audit's pair ranking (li_followup.py, part C) from the saved arrays - no
image is opened, so no reserved scan is loaded - and reports, for pairs whose 3D score exceeds
the random-pair maximum, which split groups they fall in. Counts only; no scan ID is printed or saved.
"""

import json
import random
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder
from stonekit.data import li_split

AUDIT = Path("data/interim/li2022_full_audit")
CANVAS = 160


def ncc3(a, b):
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float(np.fft.irfftn(np.fft.rfftn(a, s=(256,) * 3) * np.conj(np.fft.rfftn(b, s=(256,) * 3))).max()) / a.size


def validation_without(excluded: set) -> dict:
    """100-epoch filtered detection and false positives on validation scans, with and without `excluded`."""
    import csv

    detail = Path("data/interim/stone_analysis/20261107T184835Z")
    method = "model_within_30mm_of_kidney"
    with (detail / "annotated.csv").open(newline="", encoding="utf-8") as f:
        annotated = [r for r in csv.DictReader(f) if r["method"] == method]
    with (detail / "scans.csv").open(newline="", encoding="utf-8") as f:
        scans = [r for r in csv.DictReader(f) if r["method"] == method]

    def score(keep):
        a = [r for r in annotated if keep(r["case"])]
        s = [r for r in scans if keep(r["case"])]
        return {"scans": len(s), "annotated": len(a), "detected": sum(r["kind"] != "missed" for r in a),
                "detection": round(sum(r["kind"] != "missed" for r in a) / len(a), 4) if a else None,
                "fp_per_scan": round(sum(int(r["false_positive_pieces"]) for r in s) / len(s), 3)}  # fmt: skip

    return {"all": score(lambda c: True), "without": score(lambda c: c not in excluded)}


def main() -> None:
    manifest = li_split.load_manifest(Path("data/processed/li2022_working_split_v1.csv"))
    split = json.loads(Path("data/nnunet/li_stone_split.json").read_text(encoding="utf-8"))
    group = {c: ("train" if c in split["train"] else "validation" if c in split["validation"] else "development (unused)") if g == "development" else g
             for c, g in manifest.items()}  # fmt: skip
    ids = sorted(p.stem for p in (AUDIT / "proj").glob("wd*.npy"))
    spectra = {}
    for k in ids:
        p = np.load(AUDIT / "proj" / f"{k}.npy")
        spectra[k] = [np.fft.rfft2(np.pad((q - q.mean()) / (q.std() + 1e-6), (0, CANVAS)).astype(np.float32)).astype(np.complex64) for q in p]
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            cor = float(np.fft.irfft2(spectra[a][0] * np.conj(spectra[b][0])).max()) / CANVAS**2
            sag = float(np.fft.irfft2(spectra[a][1] * np.conj(spectra[b][1])).max()) / CANVAS**2
            pairs.append(((cor + sag) / 2, a, b))
    pairs.sort(reverse=True)
    top = pairs[1:31]
    random.seed(20261105)
    null = random.sample(pairs[100:], 40)
    volumes = {p.stem: np.load(p) for p in (AUDIT / "vol4").glob("wd*.npy")}
    null_scores = sorted(ncc3(volumes[a], volumes[b]) for _, a, b in null if a in volumes and b in volumes)
    threshold = max(null_scores)
    suspect = [(a, b) for _, a, b in top if a in volumes and b in volumes and ncc3(volumes[a], volumes[b]) > threshold]
    relation = {}
    for a, b in suspect:
        ga, gb = group.get(a, "excluded"), group.get(b, "excluded")
        key = " / ".join(sorted((ga, gb)))
        relation[key] = relation.get(key, 0) + 1
    scans = {c for pair in suspect for c in pair}
    development = {"train", "validation", "development (unused)"}

    def crossing(first: set, second: set) -> int:
        return sum(1 for a, b in suspect if (group.get(a, "excluded") in first and group.get(b, "excluded") in second)
                   or (group.get(b, "excluded") in first and group.get(a, "excluded") in second))  # fmt: skip

    # Sensitivity: validation results without the validation scans that resemble a training scan.
    twinned = {v for a, b in suspect for v, w in ((a, b), (b, a)) if group.get(v) == "validation" and group.get(w) == "train"}
    sensitivity = validation_without(twinned)
    out = {
        "definitions": __doc__, "scans_in_audit": len(ids), "random_pair_3d_max": round(threshold, 3),
        "random_pairs_scored": len(null_scores), "top_pairs_scored": len(top),
        "suspected_repeat_pairs_above_random_max": len(suspect), "scans_involved": len(scans),
        "pairs_by_split_relation": relation,
        "cross_boundary_pairs": {
            "train-validation": crossing({"train"}, {"validation"}),
            "development-reserved_test": crossing(development, {"reserved_test"}),
            "development-reserved_calibration": crossing(development, {"reserved_calibration"}),
            "reserved_calibration-reserved_test": crossing({"reserved_calibration"}, {"reserved_test"}),
        },
        "validation_scans_resembling_a_training_scan": len(twinned),
        "sensitivity_100_epochs_filtered_without_those_scans": sensitivity,
        "limitation": "Bone-projection similarity screens for repeat scans; it cannot prove or exclude that two scans share a patient. "
                      "Only the 30 most similar pairs were scored in 3D, so lower-ranked repeats are not examined.",
    }  # fmt: skip
    run = create_run_folder(Path("results/runs"), "leakage_repeat_pairs")
    (run / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "definitions"}, indent=1))
    print("Saved to", run)


if __name__ == "__main__":
    main()
