"""Gated Li data access for T1. Development scans behave exactly as in LiDevelopmentData. A reserved
(calibration or test) scan gets a path ONLY if a complete, all-passing list of T1 gate results was supplied;
otherwise it raises, just like the development loader. Nothing here bypasses the guard silently.
"""

from pathlib import Path

from stonekit.data import li_split
from stonekit.data.li2022 import LiDataError, LiDevelopmentData
from stonekit.t1 import gates

REQUIRED_GATES = {"protocol_frozen", "git_clean_at_frozen_commit", "owner_unlock", "split_unchanged", "single_run"}


class GatedLiData(LiDevelopmentData):
    def __init__(self, root: Path, gate_results: list[gates.GateResult] | None = None, config_path: Path | None = None) -> None:
        super().__init__(root, config_path)
        names = {g.name for g in gate_results or []}
        self.unlocked = bool(gate_results) and gates.all_passed(gate_results) and REQUIRED_GATES <= names

    def paths(self, case_id: str) -> tuple[Path, Path]:
        group = self.manifest.get(case_id)
        if group == li_split.DEVELOPMENT:
            return super().paths(case_id)
        if group is None:
            raise LiDataError(f"{case_id} is not in the manifest")
        if not self.unlocked:
            raise LiDataError(f"{case_id} is a reserved scan ({group}); T1 gates have not all passed")
        folder = self.root / "data" / "interim" / f"li2022_part{self._parts[case_id]}"
        return folder / f"{case_id}_ps_image.nii.gz", folder / f"{case_id}_ps_mask.nii.gz"
