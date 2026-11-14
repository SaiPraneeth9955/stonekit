"""Tests for the Freiburg loader.

The small files built here are hand-made code fixtures, not a dataset, and are used for no
result. The last section checks the real downloaded files and is skipped when they are absent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from stonekit.data.download import D1, file_md5
from stonekit.data.freiburg import (
    CSV_NAME,
    MAT_NAME,
    FreiburgDataError,
    build_table,
    write_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_FOLDER = PROJECT_ROOT / "data" / "raw" / "freiburg"
REAL_FILES_PRESENT = (REAL_FOLDER / CSV_NAME).is_file() and (REAL_FOLDER / MAT_NAME).is_file()
needs_real_files = pytest.mark.skipif(not REAL_FILES_PRESENT, reason="D1 files are not present")

CSV_TEXT = "ID;Volume;Diameter;StoneType\n1;0.01;5.71;\n2;0.13;5.97;Cystine\n3;1;15.2;cacliumoxalate-monohydrate\n"


def fixture_image():
    """A 5 x 5 x 5 patch with a 3 x 3 x 3 block and one enclosed zero voxel."""
    image = np.zeros((5, 5, 5))
    image[1:4, 1:4, 1:4] = 300.0
    image[2, 2, 2] = 0.0
    return image


def write_files(folder, diameters=(5.71, 15.2, 5.97), volumes=(0.0133, 1.0, 0.1333), voxvols=None):
    """Write fixture files with the MATLAB order deliberately different from the CSV."""
    (folder / CSV_NAME).write_text(CSV_TEXT, encoding="utf-8")
    voxvols = voxvols or (0.000379525, 0.000379525, 0.000344149)
    names = ("gt_diam", "gt_vol", "imgdata", "voxsiz", "voxvol_mL")
    stones = np.zeros((1, len(diameters)), dtype=[(name, object) for name in names])
    for i, (diameter, volume, voxvol) in enumerate(zip(diameters, volumes, voxvols)):
        stones[0, i] = (diameter, volume, fixture_image(), np.array([0.7363, 0.7363, 0.7]), voxvol)
    savemat(folder / MAT_NAME, {"stones": stones})


@pytest.fixture
def table(tmp_path):
    write_files(tmp_path)
    return build_table(tmp_path, expected_stones=3)


def test_matches_by_values_when_the_orders_differ(table):
    assert table["stone_id"].tolist() == [1, 2, 3]
    assert table["mat_index"].tolist() == [1, 3, 2]


def test_uses_the_precise_matlab_reference_and_keeps_the_csv_value(table):
    row = table.set_index("stone_id").loc[1]
    assert row["reference_volume_ml"] == pytest.approx(0.0133)
    assert row["csv_volume_ml"] == pytest.approx(0.01)


def test_spacing_fields_are_kept_unchanged_and_disagreement_is_flagged(table):
    row = table.set_index("stone_id").loc[2]
    assert row["voxsiz_x_mm"] == pytest.approx(0.7363)
    assert row["voxvol_stored_ml"] == pytest.approx(0.000344149)
    assert row["voxvol_relative_difference"] == pytest.approx(-0.0931, abs=1e-3)
    assert row["implied_inplane_mm"] == pytest.approx(0.70117, abs=1e-4)
    assert table["spacing_fields_disagree"].tolist() == [False, True, False]


def test_stone_type_keeps_the_original_and_normalises_spelling(table):
    assert table["stone_type_raw"].tolist()[1:] == ["Cystine", "cacliumoxalate-monohydrate"]
    assert table["stone_type"].tolist()[1:] == ["cystine", "calciumoxalate-monohydrate"]
    assert pd.isna(table["stone_type"].tolist()[0])


def test_image_summary_counts_stored_and_enclosed_voxels(table):
    row = table.iloc[0]
    assert row["stored_voxels"] == 26
    assert row["enclosed_zero_voxels"] == 1
    assert row["nan_voxels"] == 0
    assert row["min_stored_hu"] == 300.0


def test_reference_flags(table):
    by_id = table.set_index("stone_id")
    assert bool(by_id.loc[2, "volume_exceeds_diameter_sphere"])
    assert bool(by_id.loc[1, "volume_below_4mm_sphere"])
    assert not by_id.loc[3, "volume_exceeds_diameter_sphere"]


def test_no_rater_or_species_columns_are_invented(table):
    assert not [c for c in table.columns if "rater" in c or "species" in c]


def test_unmatched_stone_is_refused(tmp_path):
    write_files(tmp_path, diameters=(5.71, 15.2, 9.99))
    with pytest.raises(FreiburgDataError, match="MATLAB matches"):
        build_table(tmp_path, expected_stones=3)


def test_ambiguous_match_is_refused(tmp_path):
    write_files(tmp_path, diameters=(5.71, 5.71, 5.97), volumes=(0.0133, 0.0133, 0.1333))
    with pytest.raises(FreiburgDataError, match="MATLAB matches"):
        build_table(tmp_path, expected_stones=3)


def test_wrong_stone_count_is_refused(tmp_path):
    write_files(tmp_path)
    with pytest.raises(FreiburgDataError, match="Expected 101"):
        build_table(tmp_path)


def test_table_round_trips_through_parquet(table, tmp_path):
    path = tmp_path / "nested" / "freiburg_stones.parquet"
    write_table(table, path)
    assert pd.read_parquet(path)["stone_id"].tolist() == [1, 2, 3]


# Real downloaded files: these record what Stage 1 found.
@needs_real_files
def test_real_files_match_their_published_checksums():
    for expected in D1.files:
        assert file_md5(REAL_FOLDER / expected.name) == expected.md5


@needs_real_files
def test_real_table_has_101_matched_stones():
    real = build_table(REAL_FOLDER)
    assert real["stone_id"].tolist() == list(range(1, 102))
    assert sorted(real["mat_index"]) == list(range(1, 102))
    assert real[["reference_volume_ml", "reference_diameter_mm"]].notna().all().all()


@needs_real_files
def test_real_table_records_the_known_spacing_groups():
    real = build_table(REAL_FOLDER)
    assert int(real["spacing_fields_disagree"].sum()) == 21
    assert real["voxvol_stored_ml"].round(6).value_counts().to_dict() == {
        0.00038: 80,
        0.000396: 8,
        0.000404: 8,
        0.000344: 5,
    }
    assert set(real["voxsiz_x_mm"]) == {0.7363}
    assert real["nan_voxels"].sum() == 0
