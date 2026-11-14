"""Scripted download for open sources only.

Stage 1 covers D1 (CT-Uroliths-Freiburg). The public record is checked before any file is
fetched, and every file is checked against its published MD5. A file that already exists is
verified and never overwritten.
TODO(later stages): D3, and D2 only if its licence and a direct open link are confirmed.
"""

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

USER_AGENT = "StoneKit-research-download"
CHUNK_BYTES = 1024 * 1024


class DownloadError(RuntimeError):
    """The source record or a file did not match what was expected."""


@dataclass(frozen=True)
class ExpectedFile:
    name: str
    size: int
    md5: str


@dataclass(frozen=True)
class DatasetSpec:
    article_url: str
    doi: str
    version: int
    licence: str
    files: tuple[ExpectedFile, ...]


D1 = DatasetSpec(
    article_url="https://api.figshare.com/v2/articles/21992264",
    doi="10.6084/m9.figshare.21992264.v1",
    version=1,
    licence="CC BY 4.0",
    files=(
        ExpectedFile("ct-uroliths_table.csv", 2030, "53f5df4a1d36e4d65c8c7b6eeec3e07f"),
        ExpectedFile("ct-uroliths_matlab.mat", 18175283, "871b9e5af6d1dccb738a42acb5a25ab7"),
    ),
)


def _open(url: str):
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120)


def fetch_metadata(url: str) -> dict:
    with _open(url) as response:
        return json.load(response)


def fetch_chunks(url: str) -> Iterable[bytes]:
    with _open(url) as response:
        while chunk := response.read(CHUNK_BYTES):
            yield chunk


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def check_metadata(spec: DatasetSpec, metadata: dict) -> dict[str, str]:
    """Check the public record against `spec` and return each file's download URL."""
    licence = metadata.get("license", {}).get("name")
    actual = (metadata.get("doi"), metadata.get("version"), licence)
    if actual != (spec.doi, spec.version, spec.licence):
        raise DownloadError(f"Public record differs: {actual}")

    listed = {f["name"]: f for f in metadata.get("files", [])}
    if set(listed) != {f.name for f in spec.files}:
        raise DownloadError(f"Public record lists different files: {sorted(listed)}")
    for expected in spec.files:
        record = listed[expected.name]
        if record["size"] != expected.size or record["computed_md5"] != expected.md5:
            raise DownloadError(f"Public record differs for {expected.name}")
    return {name: record["download_url"] for name, record in listed.items()}


def _download_file(
    expected: ExpectedFile, url: str, path: Path, fetch: Callable[[str], Iterable[bytes]]
) -> None:
    partial = path.with_name(path.name + ".part")
    try:
        with partial.open("wb") as f:
            for chunk in fetch(url):
                f.write(chunk)
        if file_md5(partial) != expected.md5:
            raise DownloadError(f"Checksum mismatch for downloaded {expected.name}")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def download_dataset(
    spec: DatasetSpec,
    destination: Path,
    get_metadata: Callable[[str], dict] = fetch_metadata,
    fetch: Callable[[str], Iterable[bytes]] = fetch_chunks,
) -> dict[str, str]:
    """Verify the source, then fetch missing files. Return each file's outcome."""
    urls = check_metadata(spec, get_metadata(spec.article_url))
    destination.mkdir(parents=True, exist_ok=True)

    outcomes = {}
    for expected in spec.files:
        path = destination / expected.name
        if path.exists():
            if file_md5(path) != expected.md5:
                raise DownloadError(f"Existing {path} fails its checksum and was left in place")
            outcomes[expected.name] = "already present, checksum OK"
        else:
            _download_file(expected, urls[expected.name], path, fetch)
            outcomes[expected.name] = "downloaded, checksum OK"
    return outcomes
