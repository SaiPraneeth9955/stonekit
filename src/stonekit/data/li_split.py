"""Development-only working split of Li 2022, and the guard that protects reserved scans.

This is not the final scientific split (docs/decisions.md, 5 November 2026). Reserved scans
must not reach training, tuning or demonstration code.
"""

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from stonekit.config import make_rng

DEVELOPMENT = "development"
RESERVED = ("reserved_calibration", "reserved_test")
STATUS = "DEVELOPMENT_ONLY_NOT_FINAL"
SPACING_GROUPS = ("<=0.62", "1.25", "2.0", "2.5")
VOLUME_GROUPS = ("none", "<100", "100-1000", ">=1000")
FIELDS = ["case", "part", "spacing_mm", "spacing_group", "volume_group", "assignment", "status"]


class ProtectedCaseError(RuntimeError):
    """A reserved scan reached code that may only use development scans."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spacing_group(spacing_mm: float) -> str:
    """Same bands as the audit's feasibility count. Stored values are rounded in the paper."""
    if not 0 < spacing_mm <= 2.6:
        raise ValueError(f"Slice spacing {spacing_mm} is outside the audited range")
    if spacing_mm < 1:
        return "<=0.62"
    if spacing_mm < 1.5:
        return "1.25"
    return "2.0" if spacing_mm < 2.2 else "2.5"


def volume_group(stone_ml: float) -> str:
    """Total stone-label volume group, in mm3."""
    if stone_ml < 0:
        raise ValueError(f"Negative stone volume: {stone_ml}")
    volume_mm3 = stone_ml * 1000
    if volume_mm3 == 0:
        return "none"
    if volume_mm3 < 100:
        return "<100"
    return "100-1000" if volume_mm3 < 1000 else ">=1000"


def largest_remainder(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Allocate n items by fraction. Remainder ties go to the earlier group."""
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("Fractions must add up to 1")
    exact = {name: n * fraction for name, fraction in fractions.items()}
    counts = {name: int(value + 1e-9) for name, value in exact.items()}
    order = sorted(fractions, key=lambda name: -round(exact[name] - counts[name], 9))
    for name in order[: n - sum(counts.values())]:
        counts[name] += 1
    return counts


def build_assignments(
    rows: list[dict[str, str]], fractions: dict[str, float], excluded: Iterable[str], seed: int
) -> list[dict[str, str]]:
    """Stratify by spacing and volume group, shuffle each stratum, allocate one draw."""
    excluded = set(excluded)
    strata: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in sorted(rows, key=lambda r: r["case"]):
        if row["case"] in excluded:
            continue
        key = (spacing_group(float(row["dz"])), volume_group(float(row["stone_ml"])))
        strata.setdefault(key, []).append(row)

    rng = make_rng(seed)
    output = []
    for spacing in SPACING_GROUPS:
        for volume in VOLUME_GROUPS:
            members = strata.get((spacing, volume), [])
            order = rng.permutation(len(members))
            shuffled = [members[int(i)] for i in order]
            start = 0
            for name, count in largest_remainder(len(members), fractions).items():
                for row in shuffled[start : start + count]:
                    output.append(
                        {
                            "case": row["case"],
                            "part": row["part"],
                            "spacing_mm": row["dz"],
                            "spacing_group": spacing,
                            "volume_group": volume,
                            "assignment": name,
                            "status": STATUS,
                        }
                    )
                start += count
    return sorted(output, key=lambda r: r["case"])


def write_manifest(assignments: list[dict[str, str]], path: Path, metadata: dict) -> str:
    """Write the manifest and a record beside it. An existing manifest is never replaced."""
    record = path.with_suffix(".json")
    if path.exists() or record.exists():
        raise FileExistsError(f"{path} already exists. The working split is one draw only")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(assignments)
    digest = sha256_file(path)
    counts = Counter(row["assignment"] for row in assignments)
    full = {**metadata, "status": STATUS, "counts": dict(counts), "manifest_sha256": digest}
    with record.open("x", encoding="utf-8") as f:
        json.dump(full, f, indent=2)
    return digest


def load_manifest(path: Path, expected_sha256: str | None = None) -> dict[str, str]:
    """Return case to assignment. Refuses a manifest that fails its recorded hash."""
    if expected_sha256 is None:
        with path.with_suffix(".json").open(encoding="utf-8") as f:
            expected_sha256 = json.load(f)["manifest_sha256"]
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{path} does not match its recorded SHA-256")
    with path.open(newline="", encoding="utf-8") as f:
        return {row["case"]: row["assignment"] for row in csv.DictReader(f)}


def require_development(cases: Iterable[str], manifest: dict[str, str]) -> list[str]:
    """Return the cases if all are development scans. Anything else raises an error."""
    cases = list(cases)
    bad = [case for case in cases if manifest.get(case) != DEVELOPMENT]
    if bad:
        raise ProtectedCaseError(f"{len(bad)} case(s) are not development scans; refusing to use them")
    return cases


def unreviewed_flagged_cases(review_sheet: Path) -> set[str]:
    """Scans with at least one flagged location that has no saved finding."""
    with review_sheet.open(newline="", encoding="utf-8") as f:
        return {row["case"] for row in csv.DictReader(f) if not row["finding_code"].strip()}


def training_cases(manifest: dict[str, str], review_sheet: Path) -> list[str]:
    """Development scans without unreviewed flagged locations, which may be used for training."""
    held_back = unreviewed_flagged_cases(review_sheet)
    development = [case for case, group in manifest.items() if group == DEVELOPMENT]
    return sorted(case for case in development if case not in held_back)
