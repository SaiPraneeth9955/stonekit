"""The gated loader refuses reserved scans unless every required gate passed. Uses the real manifest only for
case-group lookup (no image is loaded)."""

from pathlib import Path

import pytest

from stonekit.data import li_split
from stonekit.data.li2022 import LiDataError
from stonekit.t1 import gates
from stonekit.t1.access import REQUIRED_GATES, GatedLiData

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/processed/li2022_working_split_v1.csv"


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not present")
def test_reserved_refused_without_all_gates():
    data = GatedLiData(ROOT)
    reserved = next(c for c, g in data.manifest.items() if g != li_split.DEVELOPMENT)
    dev = next(c for c, g in data.manifest.items() if g == li_split.DEVELOPMENT)
    assert data.paths(dev)[0].name.startswith(dev)
    with pytest.raises(LiDataError):
        data.paths(reserved)
    partial = [gates.GateResult(n, True, "") for n in sorted(REQUIRED_GATES)[:-1]]
    with pytest.raises(LiDataError):
        GatedLiData(ROOT, partial).paths(reserved)
    failing = [gates.GateResult(n, n != "owner_unlock", "") for n in REQUIRED_GATES]
    with pytest.raises(LiDataError):
        GatedLiData(ROOT, failing).paths(reserved)
    with pytest.raises(LiDataError):
        data.paths("not_a_case")
