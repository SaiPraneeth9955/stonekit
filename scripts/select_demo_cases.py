"""Choose demonstration scans by a fixed rule, from audit values, before any model output exists.

Only scans held out from the model's training are candidates. The rule uses annotation-derived size
and piece counts from the audit, never a prediction. Ties break by case ID.

Rule, over validation scans, once each:
- ordinary: one stone piece, 100 to 1000 mm3, the median volume of that group
- small: under 100 mm3, the median volume of that group
- large: the largest total stone-label volume
- multiple: the most stone pieces
- no_stone: no stone label, the first by case ID
- suspicious: scans with finished review findings of B (device) or D (can't tell), first by case ID
"""

import argparse
import csv
import json
from pathlib import Path

from stonekit.config import create_run_folder


def median_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda r: (float(r["stone_ml"]), r["case"]))
    return ordered[(len(ordered) - 1) // 2]


def select(rows: list[dict], reviewed: dict[str, str]) -> dict[str, str | None]:
    available = list(rows)

    def take(row: dict | None) -> str | None:
        if row is None:
            return None
        available.remove(row)
        return row["case"]

    chosen: dict[str, str | None] = {}
    suspicious = [r for r in available if reviewed.get(r["case"]) in ("B", "D")]
    device = [r for r in suspicious if reviewed[r["case"]] == "B"]
    chosen["suspicious"] = take(min(device or suspicious, key=lambda r: r["case"]) if (device or suspicious) else None)
    chosen["large"] = take(max(available, key=lambda r: (float(r["stone_ml"]), r["case"])) if available else None)
    chosen["multiple"] = take(max(available, key=lambda r: (int(r["stone_components"]), r["case"])) if available else None)
    chosen["ordinary"] = take(
        median_row([r for r in available if int(r["stone_components"]) == 1 and 0.1 <= float(r["stone_ml"]) < 1])
    )
    chosen["small"] = take(median_row([r for r in available if 0 < float(r["stone_ml"]) < 0.1]))
    none = [r for r in available if float(r["stone_ml"]) == 0]
    chosen["no_stone"] = take(min(none, key=lambda r: r["case"]) if none else None)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root
    split = json.loads((root / "data" / "nnunet" / "li_stone_split.json").read_text(encoding="utf-8"))
    table = root / "data" / "interim" / "li2022_full_audit" / "li_full_cases.csv"
    with table.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["case"] in set(split["validation"])]
    sheet = root / "data" / "interim" / "li2022_full_audit" / "review" / "li_slicer_review_sheet.csv"
    with sheet.open(newline="", encoding="utf-8") as f:
        reviewed = {r["case"]: r["finding_code"].strip() for r in csv.DictReader(f) if r["finding_code"].strip()}
    chosen = select(rows, reviewed)
    run = create_run_folder(root / "results" / "runs", "demo_case_selection")
    record = {"rule": __doc__, "candidates": len(rows), "chosen": chosen}
    (run / "selection.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(chosen, indent=2))
    print("Saved to", run)


if __name__ == "__main__":
    main()
