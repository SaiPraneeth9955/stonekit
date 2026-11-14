# StoneKit

**Detection-aware conformal uncertainty for automated kidney-stone measurement on non-contrast CT.**

StoneKit is research software. It segments kidney stones on non-contrast CT, measures each stone in 3D, and reports calibrated uncertainty intervals at two levels:

- for each stone's volume and 3D diameter;
- for the patient's total stone burden. This interval accounts for falsely detected pieces, per-stone measurement error and missed stones.

> **Research software only. It is not a medical device, and nothing in this repository supports diagnostic or clinical use.**

## Paper

This repository accompanies the manuscript:

> A. R. Kadari, S. P. Surepalli, and J. Kolisetty, "Detection-Aware Conformal Uncertainty for Automated Kidney-Stone Measurement on Non-Contrast CT," manuscript, 2026.

Authors: Department of Computer Science and Engineering, SRM University-AP, Andhra Pradesh, India. Corresponding author: Abhijith Reddy Kadari (abhijithreddy2006@gmail.com).

### Main findings

- **Locked confirmatory evaluation:** 53 held-out scans, frozen protocol, run once.
  - The detection-aware burden interval covered 50 of 53 scans at nominal 0.90.
  - Its median per-scan width ratio to a conformalised quantile-regression interval was 0.573 (95% CI 0.540–0.656), with the pre-specified 30 mm kidney-distance filter.
  - At nominal 0.90 without the filter, no width advantage was shown.
  - A post-hoc replay of the stored test outputs (descriptive) places the advantage in stone-bearing scans (ratio 0.554, 0.535–0.609); on stone-free scans the detection-aware interval was wider.
- **Small stones:** for stones predicted below 5 mm, size-stratified (Mondrian) per-stone diameter intervals covered 29 of 31 stones, against 19 of 31 for a global interval.
- **External check (MSWAL, 58 non-contrast scans):**
  - burden coverage held (56 of 58);
  - the width advantage did not transfer;
  - per-stone volume intervals under-covered (0.733).

Every number in the paper traces to a file listed under [Reproducing the reported results](#reproducing-the-reported-results).

## Repository structure

| Path | Contents |
|---|---|
| `src/stonekit/` | The Python package: measurement pipeline (`pipeline/`, `measure/`), conformal and burden interval engine (`uq/`), locked-evaluation gates and analysis (`t1/`), evaluation utilities (`eval/`), data loaders (`data/`). |
| `scripts/` | Analysis, training-orchestration and figure scripts, including `t1_run.py` (the guarded single-run evaluation) and `make_paper_figures.py`. |
| `configs/` | One YAML configuration per run type, including the frozen pipeline configuration `system_pipeline_v1.yaml`. |
| `tests/` | Unit and integration tests. Tests that need restricted data skip when it is absent. |
| `paper/` | Manuscript source (`main.tex`, `references.bib`), final figures with the exact plotted values as CSV (`figures/*_data.csv`), and the reference audit (`REFERENCES_AUDIT.md`). |
| `docs/` | Frozen confirmatory protocol (`CONFIRMATORY_PROTOCOL_v3_DRAFT.md`; the "DRAFT" in the file name is historical, and the file was frozen with its SHA-256 recorded in the freeze manifest), freeze manifest (`T1_FREEZE_MANIFEST.json`), results log (`results.md`), data registry (`data_registry.md`) and the final paper audit (`PAPER_STAGE_E_AUDIT.md`). |
| `results/runs/` | Aggregate result files (JSON or CSV summaries) behind every number in the paper. |

## Setup

```bash
conda env create -f environment.yml        # Python 3.11, PyTorch, nnU-Net v2, TotalSegmentator
conda activate stonekit
pip install -e .
pytest -rs                                 # data-dependent tests are skipped
```

`environment.lock.yml` records the exact versions used for the study. A CUDA GPU is needed only to run the segmentation pipeline; the analysis and figure scripts run on CPU.

## Datasets and access restrictions

No image data are included in this repository.

| Dataset | Role | Access and licence |
|---|---|---|
| Li et al. 2022, unenhanced abdominal CT with kidney and stone masks | Development (165 scans), calibration (35) and locked test (53) | Zenodo [10.5281/zenodo.6042410](https://doi.org/10.5281/zenodo.6042410). Access is restricted and granted by the original authors case by case. **Not redistributed.** |
| CT-Uroliths-Freiburg, 101 ex-vivo stones | Physical-reference phantom analysis | [figshare 21992264](https://figshare.com/articles/dataset/ct-uroliths-freiburg/21992264), CC BY 4.0 |
| MSWAL | External check (77 stone-bearing cases; 58 non-contrast) | [Hugging Face zhaodongwu/MSWAL](https://huggingface.co/datasets/zhaodongwu/MSWAL), CC BY-NC 4.0 |

To run the pipeline, obtain each dataset from its source under its own terms and place it as described in `configs/paths.yaml` (`data/raw/...`).

## What is included

- All source code, tests, configurations and scripts needed to run the pipeline and the analyses.
- The frozen confirmatory protocol and its hash manifest.
- Aggregate result files for:
  - the locked evaluation (`T1_20261113T081425Z`: calibration quantities with their SHA-256, and the evaluation);
  - the external check;
  - the development design analyses;
  - the phantom analyses.
- The manuscript source, the final figures, and the exact values plotted in every figure.

## What is excluded

- **Patient data:** CT images and masks from Li 2022, and any per-scan or per-stone tables derived from them.
- **External image data:** MSWAL images and labels, and per-case tables derived from them.
- **Model weights:** nnU-Net checkpoints trained on the patient data.
- **Run intermediates:** pipeline work folders, per-case predictions, probability maps and meshes.
- **Operational files:** internal working notes, machine-specific scripts (local training chains, backups, a 3D Slicer review tool), and any credentials or local paths.

## Reproducing the reported results

The confirmatory evaluation cannot be re-run without the restricted test scans. By design it was run once, and its outputs are stored and hashed. What can be reproduced from this repository:

1. **Every number in the paper, from the stored aggregates.**

   | Paper element | Source file |
   |---|---|
   | Table I, Fig. 3, locked-test text | `results/runs/T1_20261113T081425Z/analysis/evaluation.json` |
   | Table III, external rows of Fig. 3 | `results/runs/mswal_external_check_20261113T132418Z/external_check.json` |
   | Pre-unlock simulation (Methods) | `results/runs/t1_dry_run_development_20261112T100623Z/summary.json` |
   | Table II, development rows | `results/runs/e8_burden_uncertainty_20261112T085235Z/`, `ablation_nc_development_20261112T104213Z/`, `failure_analysis_development_20261112T110426Z/` |
   | Fig. 2, phantom text | `results/runs/freiburg_r2_r3_20261111T074847Z/` (summary and per-stone table) and `rep1_neubauer_replication_20261111T074337Z/` |
   | Post-hoc stone-bearing ratio (Section IV-B) | `results/runs/t1_posthoc_width_replay_20261114T081724Z/posthoc_width_replay.json` |

   The integrity of the locked-evaluation files can be checked with:

   ```bash
   cd results/runs/T1_20261113T081425Z/analysis && sha256sum -c calibration_quantities.sha256
   ```

2. **The figures.**
   - Every plotted value is in `paper/figures/*_data.csv`.
   - `python scripts/make_paper_figures.py` redraws Figs. 1–4 from the stored aggregates. Without the restricted development table, Fig. 4's development markers are read from `paper/figures/fig4_size_bands_data.csv` (SHA-256 checked), so the figure is the same.
3. **The manuscript:** `cd paper && latexmk -pdf main.tex`.
4. **The full study, with data access.** After obtaining the datasets:
   - run the pipeline (`stonekit run-case`, configuration `configs/system_pipeline_v1.yaml`);
   - run the development analyses (`scripts/`);
   - run the guarded locked evaluation with `scripts/t1_run.py`. It refuses to run unless the protocol hash, commit, split, configuration and checkpoint all match the freeze.

## Citation

```bibtex
@unpublished{kadari2026stonekit,
  author = {Kadari, Abhijith Reddy and Surepalli, Sai Praneeth and Kolisetty, Jaswanth},
  title  = {Detection-Aware Conformal Uncertainty for Automated Kidney-Stone Measurement on Non-Contrast CT},
  note   = {Manuscript},
  url    = {https://github.com/SaiPraneeth9955/stonekit},
  year   = {2026}
}
```

Please also cite the datasets you use (see the paper's references).

## Licence

Code: MIT (see `LICENSE`). Datasets keep their own licences and terms (see above).
