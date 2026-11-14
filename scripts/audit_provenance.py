"""Reproducibility and provenance audit (read-only). Development and public data only; opens no image.

Checks:
1. Every results/runs/<name> folder cited in docs/results.md, docs/decisions.md, docs/experiments.md and
   paper/main.tex exists and holds a summary.json (or another result file).
2. Every scripts/*.py cited in those documents exists; its current SHA-256 is listed. Scripts that
   changed after a cited run cannot be proven unchanged without a commit, so this is reported, not hidden.
3. Environment: environment.lock.yml present; key package versions in the stonekit env.
4. Git: HEAD, the number of uncommitted paths, and whether the analysis code is tracked at all.
5. Locked data: the split manifest SHA matches the logged value.
Writes results/runs/provenance_audit_<stamp>/summary.json and prints a short report.
"""

import hashlib
import importlib.metadata as md
import json
import re
import subprocess
from pathlib import Path

from stonekit.config import create_run_folder

DOCS = [Path("docs/results.md"), Path("docs/decisions.md"), Path("docs/experiments.md"), Path("paper/main.tex")]
SPLIT = Path("data/processed/li2022_working_split_v1.csv")
SPLIT_SHA = "466ec9c86ae905326374d41d45d17f334e97a33e6e1ead993c533968d6c7e8cb"
PACKAGES = ["nnunetv2", "TotalSegmentator", "torch", "numpy", "scipy", "pandas", "statsmodels", "scikit-learn", "SimpleITK", "nibabel"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout


def main() -> None:
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in DOCS if p.exists())
    runs = sorted(set(re.findall(r"results/runs/([A-Za-z0-9_\-]+?_\d{8}T\d{6}Z)", text)))
    run_status = {}
    for r in runs:
        folder = Path("results/runs") / r
        files = sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []
        run_status[r] = "ok" if any(f.endswith((".json", ".csv")) for f in files) else ("missing" if not folder.exists() else "no result file")
    scripts = sorted(set(re.findall(r"scripts/([A-Za-z0-9_]+\.py)", text)))
    tracked = set(git("ls-files", "scripts", "src").split())
    script_status = {s: {"exists": (Path("scripts") / s).exists(), "sha256": sha(Path("scripts") / s) if (Path("scripts") / s).exists() else None,
                         "tracked_in_git": f"scripts/{s}" in tracked} for s in scripts}  # fmt: skip
    src = sorted(str(p) for p in Path("src/stonekit").rglob("*.py"))
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    porcelain = git("status", "--porcelain").splitlines()
    summary = {
        "status": "READ_ONLY_AUDIT",
        "cited_runs": len(runs), "runs_missing_or_empty": {r: s for r, s in run_status.items() if s != "ok"},
        "cited_scripts": len(scripts), "scripts_missing": [s for s, v in script_status.items() if not v["exists"]],
        "scripts_untracked": [s for s, v in script_status.items() if v["exists"] and not v["tracked_in_git"]],
        "src_files": len(src), "src_untracked": [f for f in src if f.replace("\\", "/") not in tracked],
        "script_sha256": {s: v["sha256"] for s, v in script_status.items()},
        "environment_lock_present": Path("environment.lock.yml").exists(),
        "environment_lock_sha256": sha(Path("environment.lock.yml")) if Path("environment.lock.yml").exists() else None,
        "package_versions": versions,
        "git_head": git("rev-parse", "HEAD").strip(), "uncommitted_paths": len(porcelain),
        "split_sha256_matches_log": sha(SPLIT) == SPLIT_SHA,
    }  # fmt: skip
    run = create_run_folder(Path("results/runs"), "provenance_audit")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    short = {k: (v if not isinstance(v, (list, dict)) or len(v) < 12 else f"{len(v)} items") for k, v in summary.items() if k != "script_sha256"}
    print(json.dumps(short, indent=1))
    print("Saved", run)


if __name__ == "__main__":
    main()
