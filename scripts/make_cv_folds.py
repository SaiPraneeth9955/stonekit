"""Build the 5-fold development cross-validation split (experiment E-CV, master plan v2 §F).

Fold 0 is the existing frozen split (122 train / 32 validation, data/nnunet/li_stone_split.json)
and is copied unchanged. Folds 1-4 partition fold 0's 122 training scans into four validation
groups; each of those folds trains on the other 154 - |group| scans. Every one of the 154 trainable
development scans is therefore validated by exactly one fold, which gives out-of-fold (OOF)
predictions for all of them.

Leakage control: suspected repeat-scan pairs are recomputed exactly as in
scripts/leakage_repeat_pairs.py (same saved bone projections and 4 mm volumes, same seed and
threshold), restricted to pairs where BOTH scans are trainable development scans, and such pairs
are kept in the same validation group. Pairs that straddle fold 0's existing train/validation
boundary cannot be moved without changing the frozen fold 0; they are counted and reported.
Assignment balances the CSV's volume_group and spacing_group strata (seeded). Only development
scan IDs are written; no reserved scan is loaded or named.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leakage_repeat_pairs as lrp

SEED = 20261105
N_NEW_FOLDS = 4


def suspected_pairs() -> list[tuple[str, str]]:
    """The audit's suspected repeat pairs (the same computation as leakage_repeat_pairs.main)."""
    ids = sorted(p.stem for p in (lrp.AUDIT / "proj").glob("wd*.npy"))
    spectra = {}
    for k in ids:
        p = np.load(lrp.AUDIT / "proj" / f"{k}.npy")
        spectra[k] = [np.fft.rfft2(np.pad((q - q.mean()) / (q.std() + 1e-6), (0, lrp.CANVAS)).astype(np.float32)).astype(np.complex64) for q in p]
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            cor = float(np.fft.irfft2(spectra[a][0] * np.conj(spectra[b][0])).max()) / lrp.CANVAS**2
            sag = float(np.fft.irfft2(spectra[a][1] * np.conj(spectra[b][1])).max()) / lrp.CANVAS**2
            pairs.append(((cor + sag) / 2, a, b))
    pairs.sort(reverse=True)
    top = pairs[1:31]
    random.seed(20261105)
    null = random.sample(pairs[100:], 40)
    volumes = {p.stem: np.load(p) for p in (lrp.AUDIT / "vol4").glob("wd*.npy")}
    threshold = max(lrp.ncc3(volumes[a], volumes[b]) for _, a, b in null if a in volumes and b in volumes)
    return [(a, b) for _, a, b in top if a in volumes and b in volumes and lrp.ncc3(volumes[a], volumes[b]) > threshold]  # fmt: skip


def groups_of(scans: list[str], pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Connected components of the repeat-pair graph over `scans` (singletons included)."""
    parent = {s: s for s in scans}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    out = defaultdict(list)
    for s in scans:
        out[find(s)].append(s)
    return [sorted(g) for g in out.values()]


def assign(groups: list[list[str]], strata: dict[str, str], n: int, seed: int) -> list[list[str]]:
    """Greedy balanced assignment: largest groups first, each to the fold with the fewest scans of
    its stratum, ties broken by total size, then by a seeded random order."""
    rng = random.Random(seed)
    order = sorted(groups, key=lambda g: (-len(g), rng.random()))
    folds = [[] for _ in range(n)]
    for g in order:
        s = strata[g[0]]
        best = min(range(n), key=lambda f: (sum(strata[x] == s for x in folds[f]), len(folds[f]), rng.random()))
        folds[best].extend(g)
    return [sorted(f) for f in folds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/nnunet/li_stone_cv_split.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite (never overwrite a split)")
    fold0 = json.loads(Path("data/nnunet/li_stone_split.json").read_text(encoding="utf-8"))
    with open("data/processed/li2022_working_split_v1.csv", encoding="utf-8") as handle:
        rows = {r["case"]: r for r in csv.DictReader(handle)}
    trainable = set(fold0["train"]) | set(fold0["validation"])
    if any(rows[c]["assignment"] != "development" for c in trainable):
        raise SystemExit("non-development scan in the fold-0 split")
    pairs = [(a, b) for a, b in suspected_pairs() if a in trainable and b in trainable]
    straddle = sum((a in fold0["train"]) != (b in fold0["train"]) for a, b in pairs)
    strata = {c: f"{rows[c]['volume_group']}|{rows[c]['spacing_group']}" for c in fold0["train"]}
    groups = groups_of(sorted(fold0["train"]), pairs)
    new = assign(groups, strata, N_NEW_FOLDS, SEED)
    folds = [{"train": sorted(fold0["train"]), "val": sorted(fold0["validation"])}]
    for v in new:
        folds.append({"train": sorted(trainable - set(v)), "val": v})
    # Checks: each trainable scan validated exactly once; no train/val overlap within a fold.
    counted = Counter(c for f in folds for c in f["val"])
    assert set(counted) == trainable and set(counted.values()) == {1}
    assert all(not set(f["train"]) & set(f["val"]) for f in folds)
    record = {
        "status": "DEVELOPMENT_ONLY_NOT_FINAL",
        "seed": SEED,
        "fold0_source": "data/nnunet/li_stone_split.json",
        "fold0_source_sha256": hashlib.sha256(Path("data/nnunet/li_stone_split.json").read_bytes()).hexdigest(),
        "manifest_sha256": fold0.get("manifest_sha256"),
        "folds": folds,
        "fold_sizes": [(len(f["train"]), len(f["val"])) for f in folds],
        "within_development_repeat_pairs": len(pairs),
        "pairs_straddling_frozen_fold0_boundary": straddle,
        "pairs_kept_together_in_folds_1_4": sum(1 for a, b in pairs if a in fold0["train"] and b in fold0["train"]),
        "strata_per_new_fold": [dict(Counter(strata[c] for c in v)) for v in new],
        "definitions": __doc__,
    }
    args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("fold_sizes", "within_development_repeat_pairs", "pairs_straddling_frozen_fold0_boundary", "pairs_kept_together_in_folds_1_4")}))
    print("Saved", args.out, hashlib.sha256(args.out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
