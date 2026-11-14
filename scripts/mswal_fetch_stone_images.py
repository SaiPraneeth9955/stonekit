"""Download ONLY the MSWAL stone-case CT images (owner approval, 13 November 2026: 77 images, 6.65 GB).

The case list comes from the label characterisation run (stone_case_image_subset.json): cases with at least one
kidney-stone (label 2) voxel. No other image is downloaded. Files are streamed to data/external/mswal/imagesTr/
(git-ignored), never overwritten, and their sizes are checked against the Hugging Face listing; a SHA-256 log is
written. Licence: CC BY-NC 4.0.
"""

import glob
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/datasets/zhaodongwu/MSWAL/resolve/main/imagesTr"
OUT = Path("data/external/mswal/imagesTr")


def fetch(url: str, dest: Path, expected: int) -> str:
    tmp = dest.with_suffix(".part")
    for attempt in range(4):
        try:
            h = hashlib.sha256()
            req = urllib.request.Request(url, headers={"User-Agent": "research-download-script"})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
                    h.update(chunk)
            if tmp.stat().st_size != expected:
                raise OSError(f"size {tmp.stat().st_size} != listed {expected}")
            tmp.rename(dest)
            return h.hexdigest()
        except (OSError, urllib.error.URLError) as e:
            print("retry", dest.name, e, flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed: {dest.name}")


def main() -> None:
    run = max(glob.glob("results/runs/mswal_label_characterisation_*"))
    subset = json.loads(Path(run, "stone_case_image_subset.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    for i, case in enumerate(subset["stone_cases"]):
        name = f"{case}_0000.nii.gz"
        dest = OUT / name
        if dest.exists():
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        else:
            digest = fetch(f"{REPO}/{name}", dest, subset["bytes"][case])
        log.append({"file": f"imagesTr/{name}", "bytes": dest.stat().st_size, "sha256": digest})
        print(i + 1, "/", len(subset["stone_cases"]), name, flush=True)
    Path("data/external/mswal/download_log_images.json").write_text(json.dumps({"source": REPO, "licence": "CC BY-NC 4.0", "subset_from": run, "files": log}, indent=1), encoding="utf-8")
    print("done", len(log), "images,", round(sum(x["bytes"] for x in log) / 1e9, 2), "GB")


if __name__ == "__main__":
    main()
