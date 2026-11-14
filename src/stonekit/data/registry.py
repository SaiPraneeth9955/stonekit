"""Read data/registry.yaml and report the status of each dataset.

D1-D4: present if the folder named in configs/paths.yaml contains a file.
D5-D6 (MIMIC): the status comes only from registry.yaml. They have no folder in the
lookup, so this module never touches data/restricted/.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

REQUIRED_FIELDS = ("id", "name", "access_mode", "role", "source_url", "licence", "status")
NOT_INSPECTED = "owner-managed, not inspected"


@dataclass(frozen=True)
class DatasetStatus:
    id: str
    name: str
    access_mode: str
    licence: str
    status: str
    location: str


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` to the folder holding data/registry.yaml and pyproject.toml."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / "data" / "registry.yaml").is_file() and (folder / "pyproject.toml").is_file():
            return folder
    raise FileNotFoundError(f"No StoneKit project found at or above {here}")


def load_registry(root: Path) -> list[dict]:
    """Return the registry entries, checking that each has every required field."""
    with (root / "data" / "registry.yaml").open(encoding="utf-8") as f:
        entries = yaml.safe_load(f)["datasets"]
    for entry in entries:
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"Registry entry {entry.get('id', '?')} lacks fields: {missing}")
    return entries


def load_raw_dataset_folders(root: Path) -> dict[str, str]:
    """Return dataset ID -> folder from configs/paths.yaml, refusing any restricted path."""
    with (root / "configs" / "paths.yaml").open(encoding="utf-8") as f:
        folders = yaml.safe_load(f)["raw_datasets"]
    for dataset_id, folder in folders.items():
        if "restricted" in PurePosixPath(folder).parts:
            raise ValueError(f"{dataset_id}: status checks may not use a restricted path")
    return folders


def _folder_has_file(folder: Path) -> bool:
    return folder.is_dir() and any(p.is_file() for p in folder.rglob("*"))


def dataset_statuses(root: Path) -> list[DatasetStatus]:
    """Return the status of every dataset in the registry."""
    folders = load_raw_dataset_folders(root)
    statuses = []
    for entry in load_registry(root):
        folder = folders.get(entry["id"])
        if folder is None:
            status, location = entry["status"], NOT_INSPECTED
        else:
            status = "present" if _folder_has_file(root / folder) else "missing"
            location = folder
        statuses.append(
            DatasetStatus(
                id=entry["id"],
                name=entry["name"],
                access_mode=entry["access_mode"],
                licence=entry["licence"],
                status=status,
                location=location,
            )
        )
    return statuses
