# Data folder

Nothing here is committed to git except this file and `registry.yaml`.

| ID | Dataset | Goes in | How obtained |
|---|---|---|---|
| D1 | CT-Uroliths-Freiburg | `data/raw/freiburg/` | Scripted download (Stage 1) |
| D2 | MSWAL | `data/raw/mswal/` | Owner obtains it (H2, Stage 3) |
| D3 | CT-ORG | `data/raw/ctorg/` | Optional, TCIA |
| D4 | Li et al. 2022 | `data/raw/li2022/` | Optional, only if the access request is granted (H4) |
| D5 | MIMIC-IV (hosp) | `data/restricted/mimiciv/` | Owner only, after PhysioNet credentialing (H3) |
| D6 | MIMIC-IV-Note | `data/restricted/mimiciv_note/` | Owner only, after PhysioNet credentialing (H3) |

## Rules
- `data/restricted/` is owner-only. Claude Code never reads it. Derived row-level MIMIC tables stay in `data/restricted/derived/`.
- `stonekit data status` does not check `data/restricted/`. For D5 and D6, the owner edits `status` in `registry.yaml` by hand.
- Freiburg stones are ex vivo and are never pooled with patient data.

## Files that do not exist yet
`data/processed/imaging_cases.parquet` and `stones.parquet` are built from real data in Stage 4. They are deliberately absent until then. `freiburg_stones.parquet` is built by `stonekit data build-freiburg`. See `docs/decisions.md`.
