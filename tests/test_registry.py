"""Tests for the dataset registry and `stonekit data status`.

Everything except the check of the real registry file runs in pytest's temporary folder
with fake entries. No test touches the real data/restricted/ folder or any MIMIC content.
"""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from stonekit.cli import app
from stonekit.data import registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALL_IDS = ["D1", "D2", "D3", "D4", "D5", "D6"]


@pytest.fixture
def project(tmp_path):
    """A fake project: six fake registry entries, all recorded as missing."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "configs").mkdir()
    entries = [
        {
            "id": dataset_id,
            "name": f"fake {dataset_id}",
            "access_mode": "fake",
            "role": "fake",
            "source_url": "fake",
            "licence": "VERIFY",
            "status": "missing",
        }
        for dataset_id in ALL_IDS
    ]
    write_registry(tmp_path, entries)
    write_paths(tmp_path, {f"D{i}": f"data/raw/d{i}" for i in range(1, 5)})
    return tmp_path


def write_registry(root, entries):
    text = yaml.safe_dump({"datasets": entries})
    (root / "data" / "registry.yaml").write_text(text, encoding="utf-8")


def write_paths(root, folders):
    text = yaml.safe_dump({"raw_datasets": folders})
    (root / "configs" / "paths.yaml").write_text(text, encoding="utf-8")


def status_of(root, dataset_id):
    return next(s for s in registry.dataset_statuses(root) if s.id == dataset_id).status


def add_file(root, relative_path):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not real data", encoding="utf-8")


def test_real_registry_lists_d1_to_d6_with_every_field():
    entries = registry.load_registry(PROJECT_ROOT)
    assert [e["id"] for e in entries] == ALL_IDS
    for entry in entries:
        assert all(entry[field] for field in registry.REQUIRED_FIELDS)
        assert entry["status"] in ("missing", "present")


def test_real_paths_config_lists_only_d1_to_d4():
    assert list(registry.load_raw_dataset_folders(PROJECT_ROOT)) == ["D1", "D2", "D3", "D4"]


def test_all_missing_when_folders_are_absent(project):
    assert {s.id: s.status for s in registry.dataset_statuses(project)} == dict.fromkeys(
        ALL_IDS, "missing"
    )


def test_empty_folder_is_missing(project):
    (project / "data" / "raw" / "d1").mkdir(parents=True)
    assert status_of(project, "D1") == "missing"


def test_folder_with_a_file_is_present(project):
    add_file(project, "data/raw/d1/nested/scan.txt")
    assert status_of(project, "D1") == "present"
    assert status_of(project, "D2") == "missing"


def test_restricted_folder_is_never_checked(project, monkeypatch):
    add_file(project, "data/restricted/mimiciv/fake.txt")
    checked = []
    real_check = registry._folder_has_file

    def recording_check(folder):
        checked.append(folder)
        return real_check(folder)

    monkeypatch.setattr(registry, "_folder_has_file", recording_check)
    statuses = {s.id: s for s in registry.dataset_statuses(project)}

    assert sorted(p.name for p in checked) == ["d1", "d2", "d3", "d4"]
    assert statuses["D5"].status == "missing"
    assert statuses["D5"].location == registry.NOT_INSPECTED


def test_d5_status_comes_from_the_registry(project):
    entries = registry.load_registry(project)
    entries[4]["status"] = "present"
    write_registry(project, entries)
    assert status_of(project, "D5") == "present"


def test_restricted_path_in_config_is_refused(project):
    write_paths(project, {"D5": "data/restricted/mimiciv"})
    with pytest.raises(ValueError, match="restricted"):
        registry.dataset_statuses(project)


def test_entry_missing_a_field_is_refused(project):
    entries = registry.load_registry(project)
    del entries[0]["licence"]
    write_registry(project, entries)
    with pytest.raises(ValueError, match="licence"):
        registry.load_registry(project)


def test_cli_status_prints_all_six_as_missing(project, monkeypatch):
    monkeypatch.chdir(project)
    result = CliRunner().invoke(app, ["data", "status"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for dataset_id in ALL_IDS:
        assert dataset_id in result.output
    assert "Missing: D1, D2, D3, D4, D5, D6" in result.output
    assert "Licence still to verify: D1, D2, D3, D4, D5, D6" in result.output


def test_cli_status_fails_outside_a_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["data", "status"], env={"COLUMNS": "200"})
    assert result.exit_code == 1
    assert "No StoneKit project found" in result.output
