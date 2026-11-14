"""Tests for the checksum-verified download. No test uses the network.

The file content is a tiny byte string, used as a code fixture only.
"""

import hashlib

import pytest
from typer.testing import CliRunner

from stonekit.cli import app
from stonekit.data.download import (
    D1,
    DatasetSpec,
    DownloadError,
    ExpectedFile,
    check_metadata,
    download_dataset,
)

CONTENT = b"not real data"
MD5 = hashlib.md5(CONTENT).hexdigest()
SPEC = DatasetSpec(
    article_url="fake://article",
    doi="fake-doi",
    version=1,
    licence="CC BY 4.0",
    files=(ExpectedFile("a.csv", len(CONTENT), MD5),),
)


def metadata(**changes):
    record = {
        "doi": "fake-doi",
        "version": 1,
        "license": {"name": "CC BY 4.0"},
        "files": [
            {"name": "a.csv", "size": len(CONTENT), "computed_md5": MD5, "download_url": "fake://a"}
        ],
    }
    return record | changes


def fetch_content(url):
    yield CONTENT


def fail_if_fetched(url):
    raise AssertionError("nothing should be fetched")


def test_downloads_a_missing_file(tmp_path):
    outcomes = download_dataset(SPEC, tmp_path, lambda url: metadata(), fetch_content)
    assert (tmp_path / "a.csv").read_bytes() == CONTENT
    assert outcomes == {"a.csv": "downloaded, checksum OK"}


def test_existing_good_file_is_not_fetched_again(tmp_path):
    (tmp_path / "a.csv").write_bytes(CONTENT)
    outcomes = download_dataset(SPEC, tmp_path, lambda url: metadata(), fail_if_fetched)
    assert outcomes == {"a.csv": "already present, checksum OK"}


def test_existing_bad_file_is_refused_and_left_in_place(tmp_path):
    (tmp_path / "a.csv").write_bytes(b"changed")
    with pytest.raises(DownloadError, match="left in place"):
        download_dataset(SPEC, tmp_path, lambda url: metadata(), fail_if_fetched)
    assert (tmp_path / "a.csv").read_bytes() == b"changed"


def test_bad_download_leaves_no_file(tmp_path):
    def fetch_wrong(url):
        yield b"wrong"

    with pytest.raises(DownloadError, match="Checksum mismatch"):
        download_dataset(SPEC, tmp_path, lambda url: metadata(), fetch_wrong)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"doi": "other"},
        {"version": 2},
        {"license": {"name": "other"}},
        {"files": []},
        {"files": [{"name": "a.csv", "size": 1, "computed_md5": MD5, "download_url": "fake://a"}]},
        {"files": [{"name": "a.csv", "size": 13, "computed_md5": "x", "download_url": "fake://a"}]},
    ],
)
def test_record_mismatch_stops_before_any_download(tmp_path, changes):
    with pytest.raises(DownloadError):
        download_dataset(SPEC, tmp_path, lambda url: metadata(**changes), fail_if_fetched)
    assert list(tmp_path.iterdir()) == []


def test_d1_spec_holds_the_published_values():
    assert D1.doi == "10.6084/m9.figshare.21992264.v1"
    assert {f.name: f.md5 for f in D1.files} == {
        "ct-uroliths_table.csv": "53f5df4a1d36e4d65c8c7b6eeec3e07f",
        "ct-uroliths_matlab.mat": "871b9e5af6d1dccb738a42acb5a25ab7",
    }


def test_check_metadata_returns_download_urls():
    assert check_metadata(SPEC, metadata()) == {"a.csv": "fake://a"}


@pytest.mark.parametrize("dataset_id", ["D2", "D5", "D6"])
def test_cli_refuses_other_datasets(dataset_id):
    result = CliRunner().invoke(app, ["data", "download", "--id", dataset_id])
    assert result.exit_code == 1
    assert "not available" in result.output
