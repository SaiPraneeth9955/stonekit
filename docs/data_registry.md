# Data registry

Licences, sources and access dates. Machine-readable list: `data/registry.yaml`.

**D1 and D4 have been downloaded in full, and D2 in part** (updated 5 November 2026). D2 has 484 label files and 77 image files in `data/raw/mswal/`. D4 access was granted to the owner, and its record's access condition stays in force: never share it or put it in Git. "Seen" values come from public pages and APIs on 2 November 2026. They are not confirmed: every licence stays `VERIFY` until the owner confirms it.

| ID | Dataset | Source | Licence seen (owner to confirm) | Access seen | Citation | Access date | Downloaded by |
|---|---|---|---|---|---|---|---|
| D1 | CT-Uroliths-Freiburg | https://figshare.com/articles/dataset/ct-uroliths-freiburg/21992264 | CC BY 4.0 (figshare API) | Open | MASTER_PLAN.md Section 16, ref 1 | 2 November 2026 | Claude Code, owner-authorized |
| D2 | MSWAL | https://github.com/haochen-MBZUAI/MSWAL- | None found on GitHub or in Hugging Face tags | Hugging Face dataset, not gated | Section 16, ref 2 | Partial, by 3 November 2026 | Partial download for G1; G1 stays open and paused |
| D3 | CT-ORG | https://www.cancerimagingarchive.net/collection/ct-org/ | CC BY 3.0 (TCIA page) | Open | Section 16, ref 9; data DOI 10.7937/tcia.2019.tt7f4v7o | Not downloaded | - |
| D4 | Li et al. 2022 | https://zenodo.org/records/6042410 | None listed (Zenodo API) | Restricted; granted to the owner | Section 16, ref 5 | 4 to 5 November 2026 | Owner; verified by Claude Code |
| D5 | MIMIC-IV (hosp) | https://physionet.org/content/mimiciv/ | Not recorded yet | Credentialed; version 3.1 seen | Section 16, ref 11 | Not downloaded | - |
| D6 | MIMIC-IV-Note | https://physionet.org/content/mimic-iv-note/ | PhysioNet Credentialed Health Data License 1.5.0 | Credentialed; version 2.2 seen | Section 16, ref 10 | Not downloaded | - |

## Questions for later stages
- **D1 (Stage 1 file inspection):** the figshare API lists 2 files, 18,177,313 bytes (about 18 MB) in total. Its description says the `.mat` file holds CT data for all 101 stones. Check this when the file is opened. See "D1 public metadata" below.
- **D2 (Stage 3, gate G1):** the source URL in Section 5, `github.com/tiuxuxsh76075/MSWAL-`, redirects to the URL above. That repository's README points to `huggingface.co/datasets/zhaodongwu/MSWAL` and says "only training dataset has been released". Check this against the scan counts in Section 5. No licence has been found, and G1 requires one that allows research use.

## D1 public metadata (checked 2 November 2026, nothing downloaded)
Checked by Claude Code through the figshare API. Neither file has been opened, so their contents are as described by figshare, not verified.

| Item | Value |
|---|---|
| Dataset DOI | 10.6084/m9.figshare.21992264.v1 |
| Version | 1 (the only version listed), published 8 February 2023 |
| Licence shown | CC BY 4.0 (owner to confirm) |
| File 1 | `ct-uroliths_table.csv`, 2,030 bytes, MD5 `53f5df4a1d36e4d65c8c7b6eeec3e07f` |
| File 2 | `ct-uroliths_matlab.mat`, 18,175,283 bytes, MD5 `871b9e5af6d1dccb738a42acb5a25ab7` |

Described contents: the CSV has one row per stone with ID, Volume, Diameter and Stone Type. The `.mat` is a structure of the 101 stones with `gt_diam`, `gt_vol`, `imgdata`, `voxsiz` and `voxvol_mL`. Cavities in some stones are NaN in `imgdata`.

### Findings to resolve (Section 16, ref 1, targeted excerpts read by Claude Code)
1. **Rater values may be missing.** The paper says reference measurements of all raters are public, but the figshare description lists single Volume and Diameter fields. Section 11.5 needs the spread between the urologists to tune U3. Check at Stage 1.
2. **G3 could fail even with a faithful replication.** G3 asks for ICC above 0.99 for volume and diameter. The paper's absolute-threshold result is 0.988 (SD 0.033) for volume and 0.994 for diameter, averaged over validation splits. Agree how to read a result near that value before Stage 2. The gate is unchanged.
3. **Stone size and species may be linked** (Stage 1 update: species is not in the files). The paper says human stones above about 5-6 mm were not available, so larger sizes partly rely on the 34 dog stones. Species is not among the described fields, and composition was not assessed for 41 stones. Check at Stage 1 whether species can be identified, and how many stones of each species fall in each size bin.

## Facts to record at download (Section 5)
- **D1:** file format; licence; how stones are identified in images; rater table columns.
- **D2:** download route; licence; label integer for kidney stone; whether phase is recorded; whether ureteral stones are labelled.
- **D3:** current TCIA licence text; which scans are non-contrast.
- **D4:** recorded below ("D4 inspection").
- **D5:** version; table names; lab item for blood creatinine (look up by name in the dictionary table, never hard-code).
- **D6:** version; table and column names; how CT abdomen/pelvis reports are identified (exam name in `radiology_detail`).
- **TotalSegmentator:** command flags (`--roi_subset`, `--fast`), checked with `TotalSegmentator -h`.

## D1 Stage 1 inspection (2 November 2026)
Both files were downloaded to `data/raw/freiburg/`, and their MD5s match the table above. Full findings: `docs/STAGE1_FREIBURG_FEASIBILITY.md`.

- **File format:** CSV (ID, Volume, Diameter, StoneType) and MATLAB v5 struct (`gt_diam`, `gt_vol`, `imgdata`, `voxsiz`, `voxvol_mL`).
- **Stones:** 101 in each file. The MATLAB file has no ID, so the link to the CSV is inferred by a unique match on diameter and volume.
- **Voxel size:** `voxsiz` is 0.7363 x 0.7363 x 0.7 mm, against 0.7 x 0.7 mm in the paper. `voxvol_mL` has 4 values and disagrees with `voxsiz` for 21 stones, so the correct spacing is unknown.
- **Rater values and species:** not in the files.
- **Images:** masked; background is exactly 0. No NaN voxels, unlike the figshare description.

## D4 inspection (5 November 2026)
Zenodo DOI 10.5281/zenodo.6042410. Paper: Diagnostics 2022;12(8):1788 (PMC9330428). No licence is listed, so the licence stays `VERIFY`. The record's access condition is legitimate academic use, case by case.

| File | Bytes | MD5 | Cases |
|---|---|---|---|
| `part1.zip` | 3,474,049,918 | `e8088cc4befe709d1ad30823f385089e` | `wd0000`-`wd0049` |
| `part2.zip` | 3,460,375,210 | `af5d551a18a801ecbf2842b08fd78256` | `wd0050`-`wd0099` |
| `part3.zip` | 4,078,948,974 | `1ca99df42322bbec85fec11d4c646b07` | `wd0100`-`wd0149` |
| `part4.zip` | 3,513,579,510 | `7efc6e571f85d1593beb3e11ff98ed9c` | `wd0150`-`wd0199` |
| `part5.zip` | 4,487,791,087 | `d73e3c8cc49eb5cbb30deef6091c81f6` | `wd0200`-`wd0259` |

- **File format:** `wdNNNN_ps_image.nii.gz` (float32) and `wdNNNN_ps_mask.nii.gz` (int16), LPS.
- **Labels, inferred from HU and volume because the record does not state them:** 0 background, 1 stone, 2 kidney. Exceptions and mask problems: `docs/decisions.md`.
- **Identifiers:** no patient IDs; NIfTI text fields empty; no header extensions.
- **Location:** `data/raw/li2022/` (zips, unchanged), `data/interim/li2022_part1/` to `li2022_part5/` (unpacked), `data/interim/li2022_full_audit/` (audit). All git-ignored.
- **Backups:** must exclude all of these, including derived files.
