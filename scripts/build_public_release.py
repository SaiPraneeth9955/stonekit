"""Assemble the public StoneKit release tree from an explicit ALLOWLIST and scan it before anything is published.

The public repository is built fresh from this tree (no private git history). Nothing is pushed here.
Output: public_release/ (git-ignored) plus public_release/RELEASE_MANIFEST.txt (SHA-256 of every file).
The scan refuses the release (exit code 1) on: image/volume/weight/archive formats, files over 5 MB,
local machine paths, credential-like strings, or e-mail addresses other than the three authors'.
"""

import fnmatch
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public_release"

ALLOW = [
    "README.md", "LICENSE", "pyproject.toml", "environment.yml", "environment.lock.yml", "Makefile", ".gitignore",
    "src/**/*", "tests/**/*", "configs/**/*", "scripts/**/*", "data/README.md", "data/registry.yaml",
    "paper/main.tex", "paper/references.bib", "paper/REFERENCES_AUDIT.md",
    "paper/figures/fig1_overview.*", "paper/figures/fig2_phantom.*", "paper/figures/fig3_burden_robustness.*",
    "paper/figures/fig4_size_bands.*", "paper/figures/fig1_overview_data.csv", "paper/figures/fig2_phantom_data.csv",
    "paper/figures/fig3_burden_robustness_data.csv", "paper/figures/fig4_size_bands_data.csv",
    "paper/figures/fig_freiburg_r2_volume_error_by_type_data.csv", "paper/figures/fig_freiburg_r3_diameter_agreement_data.csv",
    "docs/CONFIRMATORY_PROTOCOL_v3_DRAFT.md", "docs/T1_FREEZE_MANIFEST.json", "docs/results.md", "docs/data_registry.md",
    "docs/PAPER_STAGE_E_AUDIT.md", "docs/STAGE2_PROTOCOL.md", "docs/preregistration.md",
    "results/runs/T1_20261113T081425Z/inputs.json", "results/runs/T1_20261113T081425Z/COMPLETED",
    "results/runs/T1_20261113T081425Z/analysis/*",
    "results/runs/mswal_external_check_20261113T132418Z/external_check.json",
    "results/runs/mswal_label_characterisation_20261113T092648Z/summary.json",
    "results/runs/t1_dry_run_development_20261112T100623Z/summary.json",
    "results/runs/e4_li_annotation_2d3d_20261111T083202Z/summary.json",
    "results/runs/e6_uncertainty_design_20261112T082354Z/summary.json",
    "results/runs/e7_detection_uncertainty_20261112T082355Z/summary.json",
    "results/runs/e8_burden_uncertainty_20261112T085235Z/summary.json",
    "results/runs/ablation_nc_development_20261112T104213Z/summary.json",
    "results/runs/failure_analysis_development_20261112T110426Z/summary.json",
    "results/runs/secondary_na_nb_development_20261112T103848Z/summary.json",
    "results/runs/exploratory_asymmetric_u2_20261112T124900Z/summary.json",
    "results/runs/p1_power_simulation_20261112T084506Z/summary.json",
    "results/runs/freiburg_r2_r3_20261111T074847Z/*",
    "results/runs/rep1_neubauer_replication_20261111T074337Z/*",
    "results/runs/t1_posthoc_width_replay_20261114T081724Z/posthoc_width_replay.json",
]  # fmt: skip
# Machine-specific or operational files: they hold local paths or exist only for the owner's workstation.
DENY = [
    "scripts/ecv_resume_test.sh", "scripts/make_backup.sh", "scripts/post_training_chain.sh", "scripts/train_cv_folds.sh",
    "scripts/slicer_adjudication.py", "scripts/slicer_review.py", "scripts/slicer_review_hotfix.py",
    "scripts/verify_drev_package.py", "tests/test_radiology_adjudication.py", "tests/test_slicer_review.py",
    "**/__pycache__/**", "**/*.pyc", "**/*.egg-info/**", "**/.pytest_cache/**", "**/.ruff_cache/**",
]  # fmt: skip
BAD_EXT = (".nii", ".nii.gz", ".pth", ".pt", ".ckpt", ".npz", ".npy", ".nrrd", ".mha", ".dcm", ".zip", ".tar", ".gz", ".pkl", ".h5", ".parquet")
BAD_TEXT = [r"/home/[a-z]", r"[A-Za-z]:\\\\Users\\\\", r"[A-Za-z]:/Users/", r"wsl\.localhost", r"BEGIN [A-Z ]*PRIVATE KEY", r"\bghp_[A-Za-z0-9]{20,}",
            r"\bhf_[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}", r"(?i)(api[_-]?key|access[_-]?token|password)\s*[:=]\s*\S{6,}"]  # fmt: skip
ALLOWED_EMAILS = {"abhijithreddy2006@gmail.com", "praneethh777@gmail.com", "kolisettyjaswanth8@gmail.com"}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def tracked_or_present(pattern: str) -> list[Path]:
    hits = [p for p in ROOT.glob(pattern) if p.is_file()]
    return sorted(hits)


def denied(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, d) for d in DENY)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    files = []
    for pattern in ALLOW:
        found = tracked_or_present(pattern)
        if not found:
            sys.exit(f"allowlisted pattern matched nothing: {pattern}")
        files += [p for p in found if not denied(p.relative_to(ROOT).as_posix())]
    files = sorted(set(files))
    problems = []
    for src in files:
        rel = src.relative_to(ROOT).as_posix()
        if rel.lower().endswith(BAD_EXT):
            problems.append(f"forbidden format: {rel}")
        if src.stat().st_size > 5_000_000:
            problems.append(f"over 5 MB: {rel}")
        if src.suffix.lower() in (".png", ".pdf"):
            continue
        text = src.read_text(encoding="utf-8", errors="ignore")
        for pat in BAD_TEXT:
            if re.search(pat, text):
                problems.append(f"pattern {pat!r} in {rel}")
        for mail in set(EMAIL.findall(text)) - ALLOWED_EMAILS:
            problems.append(f"e-mail {mail} in {rel}")
    if problems:
        print("\n".join(problems))
        sys.exit("RELEASE REFUSED: fix or exclude the files above")
    manifest = []
    for src in files:
        rel = src.relative_to(ROOT).as_posix()
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest.append(f"{hashlib.sha256(src.read_bytes()).hexdigest()}  {rel}")
    (OUT / "RELEASE_MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True).stdout.strip()
    print(f"release tree: {OUT} ({len(files)} files; source HEAD {head} plus uncommitted working tree); scan passed")


if __name__ == "__main__":
    main()
