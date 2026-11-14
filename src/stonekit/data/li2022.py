"""Li 2022 development loader. It refuses every scan that is not a development scan.

Label values are inferred from the data audit, not published: 0 background, 1 stone, 2 kidney.
Masks are annotation references, not physical truth. They are never edited.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stonekit.config import load_yaml
from stonekit.data import li_split
from stonekit.io.nifti import Volume, load_volume

STONE_LABEL = 1
KIDNEY_LABEL = 2
ALLOWED_LABELS = frozenset({0, STONE_LABEL, KIDNEY_LABEL})


class LiDataError(ValueError):
    """A Li file is not what the audit found."""


@dataclass(frozen=True)
class LiCase:
    case_id: str
    image: Volume
    mask: Volume

    @property
    def stone_reference(self) -> np.ndarray:
        return self.mask.data == STONE_LABEL

    @property
    def kidney_reference(self) -> np.ndarray:
        return self.mask.data == KIDNEY_LABEL


def validate_pair(image: Volume, mask: Volume) -> None:
    """Refuse a mask that does not fit its image, or holds labels the audit did not find."""
    if image.data.shape != mask.data.shape:
        raise LiDataError(f"Shapes differ: {image.data.shape} and {mask.data.shape}")
    if not np.allclose(image.affine, mask.affine, atol=1e-3):
        raise LiDataError("Image and mask affines differ")
    if not np.issubdtype(mask.data.dtype, np.integer):
        raise LiDataError(f"Mask type is {mask.data.dtype}, not an integer type")
    unknown = {int(v) for v in np.unique(mask.data)} - ALLOWED_LABELS
    if unknown:
        raise LiDataError(f"Unexpected mask labels: {sorted(unknown)}")


class LiDevelopmentData:
    """Access to development scans only. There is deliberately no way to load reserved scans."""

    def __init__(self, root: Path, config_path: Path | None = None) -> None:
        self.root = Path(root)
        config = load_yaml(config_path or self.root / "configs" / "li_working_split.yaml")
        manifest_path = self.root / config["manifest"]
        self.manifest = li_split.load_manifest(manifest_path)
        self.review_sheet = self.root / config["review_sheet"]
        with manifest_path.open(newline="", encoding="utf-8") as f:
            self._parts = {row["case"]: row["part"] for row in csv.DictReader(f)}

    def development_ids(self) -> list[str]:
        return sorted(c for c, group in self.manifest.items() if group == li_split.DEVELOPMENT)

    def training_ids(self) -> list[str]:
        return li_split.training_cases(self.manifest, self.review_sheet)

    def paths(self, case_id: str) -> tuple[Path, Path]:
        """Image and mask paths. The guard runs first, so a reserved scan gets no path."""
        li_split.require_development([case_id], self.manifest)
        folder = self.root / "data" / "interim" / f"li2022_part{self._parts[case_id]}"
        return folder / f"{case_id}_ps_image.nii.gz", folder / f"{case_id}_ps_mask.nii.gz"

    def load(self, case_id: str) -> LiCase:
        image_path, mask_path = self.paths(case_id)
        image = load_volume(image_path)
        mask = load_volume(mask_path)
        validate_pair(image, mask)
        return LiCase(case_id, image, mask)
