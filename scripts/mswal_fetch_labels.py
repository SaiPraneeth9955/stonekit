"""Download MSWAL dataset.json and the label masks ONLY (owner approval OD-21 step 1, 13 November 2026).

Source: Hugging Face dataset zhaodongwu/MSWAL (CC BY-NC 4.0, per the dataset card; non-commercial research).
No image volume is downloaded. Files go to data/external/mswal/ (git-ignored); a SHA-256 list and the source
URL of every file are written next to them. Existing files are never overwritten.
"""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/datasets/zhaodongwu/MSWAL"
API = "https://huggingface.co/api/datasets/zhaodongwu/MSWAL/tree/main"
OUT = Path("data/external/mswal")


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "research-download-script"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if attempt == 3:
                raise
            print("retry", url, e, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def main() -> None:
    (OUT / "labelsTr").mkdir(parents=True, exist_ok=True)
    listing = json.loads(get(f"{API}/labelsTr"))
    files = ["dataset.json"] + [x["path"] for x in listing if x["type"] == "file"]
    if any("imagesTr" in f for f in files):
        sys.exit("refusing: an image path appeared in the label list")
    log = []
    for i, f in enumerate(files):
        dest = OUT / f
        if dest.exists():
            data = dest.read_bytes()
        else:
            data = get(f"{REPO}/resolve/main/{f}")
            dest.write_bytes(data)
        log.append({"file": f, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": f"{REPO}/resolve/main/{f}"})
        if i % 50 == 0:
            print(i, f, flush=True)
    (OUT / "download_log.json").write_text(json.dumps({"source": REPO, "licence": "CC BY-NC 4.0 (Hugging Face dataset card)", "files": log}, indent=1), encoding="utf-8")
    print("downloaded", len(log), "files,", round(sum(x["bytes"] for x in log) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
