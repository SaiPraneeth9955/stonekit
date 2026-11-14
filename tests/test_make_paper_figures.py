"""Fig. 4 regenerates from public files alone: without the restricted E6 development table, the development markers
come from paper/figures/fig4_size_bands_data.csv, and the regenerated plotted values are byte-identical."""

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_paper_figures.py"
PUBLIC = ROOT / "paper" / "figures" / "fig4_size_bands_data.csv"
T1 = ROOT / "results/runs/T1_20261113T081425Z/analysis/evaluation.json"

pytestmark = pytest.mark.skipif(not (PUBLIC.exists() and T1.exists()), reason="paper figure data or T1 aggregates not present")


def load_script():
    spec = importlib.util.spec_from_file_location("make_paper_figures", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_fig4_regenerates_identically_without_restricted_data(tmp_path, monkeypatch):
    m = load_script()
    out = tmp_path / "figures"
    out.mkdir()
    shutil.copy2(PUBLIC, out / PUBLIC.name)
    monkeypatch.setattr(m, "OUT", out)
    monkeypatch.setattr(m, "FIG4_DATA", out / PUBLIC.name)
    monkeypatch.setattr(m, "E6", tmp_path / "absent_restricted_table.csv")
    m.fig_bands(m.check(m.T1))
    assert (out / "fig4_size_bands_data.csv").read_bytes() == PUBLIC.read_bytes()
    assert (out / "fig4_size_bands.pdf").stat().st_size > 0


def test_fig4_fallback_refuses_a_modified_public_file(tmp_path, monkeypatch):
    m = load_script()
    out = tmp_path / "figures"
    out.mkdir()
    (out / PUBLIC.name).write_text(PUBLIC.read_text().replace("0.782258064516129", "0.80"))
    monkeypatch.setattr(m, "OUT", out)
    monkeypatch.setattr(m, "FIG4_DATA", out / PUBLIC.name)
    monkeypatch.setattr(m, "E6", tmp_path / "absent_restricted_table.csv")
    with pytest.raises(SystemExit, match="SHA-256"):
        m.fig_bands(m.check(m.T1))
