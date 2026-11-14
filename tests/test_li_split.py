"""Tests for the development-only Li split. Fake rows only, except the optional real-data checks."""

from collections import Counter
from pathlib import Path

import pytest
import yaml

from stonekit.data import li_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = {"development": 0.65, "reserved_calibration": 0.15, "reserved_test": 0.20}


def fake_rows(n=120):
    spacings = ["0.62", "1.25", "2.0", "2.5"]
    volumes = ["0", "0.05", "0.5", "2.0"]
    return [
        {"case": f"wd{i:04d}", "part": "1", "dz": spacings[i % 4], "stone_ml": volumes[(i // 4) % 4]}
        for i in range(n)
    ]


@pytest.mark.parametrize(
    ("value", "group"), [(0.42, "<=0.62"), (0.62, "<=0.62"), (1.25, "1.25"), (2.0, "2.0"), (2.5, "2.5")]
)
def test_spacing_group(value, group):
    assert li_split.spacing_group(value) == group


def test_spacing_group_refuses_unaudited_value():
    with pytest.raises(ValueError):
        li_split.spacing_group(3.0)


@pytest.mark.parametrize(
    ("ml", "group"),
    [(0, "none"), (0.005, "<100"), (0.0999, "<100"), (0.1, "100-1000"), (0.999, "100-1000"), (1.0, ">=1000")],
)
def test_volume_group_boundaries(ml, group):
    assert li_split.volume_group(ml) == group


@pytest.mark.parametrize("n", [0, 1, 2, 4, 5, 8, 13, 27, 69, 253])
def test_largest_remainder_uses_every_item(n):
    assert sum(li_split.largest_remainder(n, FRACTIONS).values()) == n


def test_largest_remainder_known_values():
    # 5 gives exact shares 3.25, 0.75, 1.0, so the one left over goes to calibration.
    assert li_split.largest_remainder(5, FRACTIONS) == {
        "development": 3,
        "reserved_calibration": 1,
        "reserved_test": 1,
    }
    # 4 gives 2.6, 0.6, 0.8: test first, then the tie goes to development.
    assert li_split.largest_remainder(4, FRACTIONS) == {
        "development": 3,
        "reserved_calibration": 0,
        "reserved_test": 1,
    }


def test_assignment_is_deterministic_and_changes_with_seed():
    first = li_split.build_assignments(fake_rows(), FRACTIONS, [], 1)
    assert first == li_split.build_assignments(fake_rows(), FRACTIONS, [], 1)
    assert first != li_split.build_assignments(fake_rows(), FRACTIONS, [], 2)


def test_input_order_does_not_change_assignment():
    rows = fake_rows()
    assert li_split.build_assignments(rows, FRACTIONS, [], 1) == li_split.build_assignments(
        rows[::-1], FRACTIONS, [], 1
    )


def test_every_case_once_and_excluded_never_assigned():
    result = li_split.build_assignments(fake_rows(), FRACTIONS, ["wd0003"], 1)
    cases = [row["case"] for row in result]
    assert len(cases) == len(set(cases)) == 119
    assert "wd0003" not in cases


def test_each_row_is_labelled_not_final():
    result = li_split.build_assignments(fake_rows(), FRACTIONS, [], 1)
    assert {row["status"] for row in result} == {"DEVELOPMENT_ONLY_NOT_FINAL"}


def test_manifest_is_never_replaced_and_hash_is_checked(tmp_path):
    path = tmp_path / "split.csv"
    rows = li_split.build_assignments(fake_rows(), FRACTIONS, [], 1)
    digest = li_split.write_manifest(rows, path, {"seed": 1})
    assert li_split.sha256_file(path) == digest
    assert len(li_split.load_manifest(path)) == 120
    with pytest.raises(FileExistsError):
        li_split.write_manifest(rows, path, {"seed": 1})
    with path.open("a", encoding="utf-8") as f:
        f.write("changed\n")
    with pytest.raises(ValueError):
        li_split.load_manifest(path)


def test_guard_refuses_reserved_and_unknown_cases():
    manifest = {"a": "development", "b": "reserved_calibration", "c": "reserved_test"}
    assert li_split.require_development(["a"], manifest) == ["a"]
    for case in ("b", "c", "excluded_or_unknown"):
        with pytest.raises(li_split.ProtectedCaseError):
            li_split.require_development(["a", case], manifest)


def test_guard_error_does_not_name_cases():
    manifest = {"b": "reserved_test"}
    with pytest.raises(li_split.ProtectedCaseError) as error:
        li_split.require_development(["b"], manifest)
    assert "b" not in str(error.value).split()


def test_training_cases_hold_back_unreviewed_flagged_scans(tmp_path):
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "case,finding_code\ndone,B\nopen,\ntwo,D\ntwo,\n", encoding="utf-8"
    )
    manifest = {
        "done": "development",
        "open": "development",
        "two": "development",
        "plain": "development",
        "secret": "reserved_test",
    }
    assert li_split.training_cases(manifest, sheet) == ["done", "plain"]


def _real_config():
    path = PROJECT_ROOT / "configs" / "li_working_split.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_real_strata_match_the_approved_counts():
    config = _real_config()
    table = PROJECT_ROOT / config["cases_table"]
    if not table.exists():
        pytest.skip("Li audit table is not present")
    import csv

    with table.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = li_split.build_assignments(rows, config["fractions"], config["excluded"], config["seed"])
    assert len(result) == config["expected_cohort_size"]
    counts = Counter((row["spacing_group"], row["volume_group"]) for row in result)
    approved = {
        "<=0.62": (0, 0, 4, 4),
        "1.25": (8, 5, 15, 69),
        "2.0": (15, 8, 2, 6),
        "2.5": (28, 13, 27, 49),
    }
    for spacing, values in approved.items():
        for volume, value in zip(li_split.VOLUME_GROUPS, values, strict=True):
            assert counts[(spacing, volume)] == value


def test_real_manifest_matches_a_fresh_draw_when_present():
    config = _real_config()
    manifest = PROJECT_ROOT / config["manifest"]
    if not manifest.exists():
        pytest.skip("Working split has not been created")
    import csv

    with (PROJECT_ROOT / config["cases_table"]).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fresh = li_split.build_assignments(rows, config["fractions"], config["excluded"], config["seed"])
    assert li_split.load_manifest(manifest) == {row["case"]: row["assignment"] for row in fresh}
    assert not set(config["excluded"]) & set(li_split.load_manifest(manifest))
