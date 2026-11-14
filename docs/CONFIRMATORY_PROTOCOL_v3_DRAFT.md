# StoneKit comparative confirmatory protocol, version 3: FROZEN (owner approval, 13 November 2026)

- **Status:** FROZEN by the owner's explicit approval on 13 November 2026 ("Yes, all three now": freeze, first commit, unlock). The SHA-256 of this file is logged in `docs/decisions.md` under "PROTOCOL FROZEN". Any later change is a logged deviation. The file name keeps "_DRAFT" only because the guarded runner is configured with this path.
- **Drafted:** 12 November 2026 by Claude Code, at the owner's request; amended by owner decisions up to OD-21 before the freeze.
- **Replaces, once frozen:** `docs/CONFIRMATORY_PROTOCOL.md` (v1 draft) and `docs/CONFIRMATORY_PROTOCOL_v2_DRAFT.md` (v2 draft). Both are kept unchanged as history.
- **Authority:** owner decisions OD-1 to OD-21 (OD-19 to OD-21 on 13 November 2026) and the P1 decision (`docs/decisions.md`, 12 November 2026), master plan v2 (`docs/MASTER_RESEARCH_PLAN.md`), and the logged amendments 1–8. Where this draft and master plan v2 differ, the decisions log outranks the plan text; §12 lists the plan passages still to be reconciled.

**Status tags used throughout:**
- **[OWNER]:** an owner-approved decision, with its ID.
- **[PLAN]:** a requirement already in the approved plan or its logged amendments.
- **[AMEND]:** a logged plan amendment made by Claude Code under the owner's autonomous directive, not separately confirmed by the owner. Confirmation is requested at the freeze.
- **[PROPOSED]:** a new specification from Claude Code that needs owner approval before the freeze.
- **[OPEN]:** unresolved; it blocks the freeze.

**Standing scope [OWNER, 12 November 2026]:**
- This protocol governs one step, the single locked confirmatory evaluation (T1). It is not the end of the project.
- D-REV is a dependency of T1, not a stopping point.
- Work continues through the full research lifecycle to publication readiness. That includes novelty candidates compared against the frozen baseline, ablations, sensitivity and robustness analyses, the application, the reproducibility audit, figures and the manuscript.
- Any further method or novelty arm must be developed on development data only and added to this protocol **before** the freeze. The reserved scans are evaluated once, for all arms together [OWNER, 11 November test-set sequencing].

---

## 0. Preconditions: all must hold before any reserved scan is loaded

| # | Precondition | Status |
|---|---|---|
| 0.1 | ~~D-REV: radiology-trained, blinded adjudication completed and hashed~~ | **[OWNER OD-19] Superseded:** T1 proceeds without the radiology adjudication. Primary cohort 165/35/53; screening cohorts are sensitivity analyses (§5.1); the missing radiology check is a stated limitation |
| 0.2 | ~~If D-REV identifies ineligible development scans: rerun E6–E8~~ | **Not applicable** (OD-19: no adjudication, so no development scan is reclassified) |
| 0.3 | All arms frozen: method, configuration, checkpoint hash, code commit and endpoints written into §2 and §4 | [PLAN] |
| 0.4 | The code is committed (owner approval) and the commit hash is written here | **[OPEN]** owner approval |
| 0.5 | The full test suite passes and `ruff check` is clean on that commit; a smoke run on 2 development scans with the exact frozen configuration | [PLAN] |
| 0.6 | A guarded T1 runner exists that refuses to load reserved scans unless this protocol's frozen SHA-256 is supplied and matches `docs/decisions.md`. It writes a new run folder only | **Built** (`scripts/t1_run.py` T1 mode; 8 gates, including frozen config, checkpoint and screening-outcome hashes; verified to refuse on 13 November 2026) |
| 0.7 | The owner approves this file. Its SHA-256 is logged in `docs/decisions.md` (freeze) | **[OPEN]** |

---

## 1. Question

**Primary question:** on held-out scans from the same dataset, does a detection-aware patient-burden interval (P2-A) give narrower intervals than the strongest direct patient-level conformal comparator (P2-B-CQR), without evidence of under-coverage?

**Descriptive questions:**
- How well do the frozen per-stone measurement intervals cover the annotated volume and diameter (P1, descriptive)?
- How do detection, false positives and measurement error behave on held-out scans?

**Scope:**
- This is an internal held-out test: one dataset, one hospital. It is not external validation.
- The reference is the Li 2022 annotation, not physical truth.
- Research use only; no diagnostic or clinical claim.

---

## 2. Frozen system and arms

### 2.1 Segmentation and inference: identical for every arm [OWNER OD-2]

| Item | Value (SHA-256 at drafting; re-verified at the freeze) |
|---|---|
| Config | `configs/system_pipeline_v1.yaml`, `64bf721da9d46bc885b331b8a0ecf7a5608c17064c5778d5402cc3a12702b5ee` |
| Model | nnU-Net v2 2.8.1, `nnUNetTrainer_100epochs__nnUNetPlans_4GB__3d_fullres`, **fold 0**, `checkpoint_final.pth`, `2100ec09d2e2eadcfd11d928e9db7dbe487a1f1fd10bc17924cfff108a6d410c` [OWNER OD-4] |
| nnU-Net test-time augmentation | On (`disable_tta: false`), identical in all arms |
| Kidney step | TotalSegmentator 2.18.0 `--fast`, 600 s timeout, low-memory fallback (flagged and reported when used) |
| Post-processing | nnU-Net label output; 26-connected pieces of ≥ 3 voxels; **30 mm kidney-distance filter** on the model pieces; removed pieces listed |
| Measurement | `stonekit.measure.stones.measure_stones` (regression-checked against the 6 November baseline: identical on 154 stones; `results/runs/measurement_regression_20261112T100054Z/`) |
| Reliability signals for U4 and P2-A | Per-piece `p_mean`, TTA spread (8 flips), threshold sensitivity, as in `scripts/oof_tta.py` and `scripts/build_oof_stone_table.py`. **The segmentation mask is unchanged by this extraction** |
| Interval engine | `src/stonekit/uq/confirmatory.py`, `a736fd0c16cbb96dabd56086f0b716c021bf915243b6afadac6c45e6b72b577a` (draft; frozen with the commit) |
| Code | git commit ______ (no uncommitted changes) **[OPEN]** |

**Not used:**
- No 5-fold ensemble and no U6 in the confirmatory run [OWNER OD-4].
- `configs/baseline_pipeline.yaml` (identical to the system config except for the filter) is used **only** in the secondary filter ablation (§5.3) [OWNER OD-2, OD-3].

### 2.2 Arms: they differ only in the uncertainty methodology [OWNER OD-2]

| Arm | Per-stone intervals | Burden interval |
|---|---|---|
| **Baseline uncertainty arm** | U0 Bland–Altman (log) and U1 global split conformal, for volume and 3D diameter | P2-0 naive sum of per-stone bounds |
| **Final uncertainty arm** | **U2** Mondrian by predicted 3D diameter, for **volume** and for **3D diameter** [OWNER OD-12, amending OD-2]: the original pre-specified E6 rule (master plan v2 §G); no method qualified, so U2 is carried | **P2-A** decomposition [OWNER OD-11, amendment 2] (beats the best direct comparator on development; see §6) |
| **Comparators** (reported, not tested) | U3, U5; **U4** (normalised by TTA spread) **only as a secondary exploratory comparator**, never as a pre-specified selected method [OWNER OD-12]; **U2-asymmetric** (two-tailed Mondrian, α/2 per tail; `scripts/exploratory_asymmetric_u2.py`) as a secondary **exploratory** comparator, defined after development results (failure analysis), so any T1 advantage is reported as exploratory [Claude Code, 12 Nov; the owner may veto before the freeze] | **P2-B-CQR**, the fixed formal comparator [OWNER OD-1, OD-11, amendment 7]; P2-B direct |

Development evidence: the E6–E8 runs of 12 November 2026 and the dry run (§6). These selected the methods; they are not validation.

**Method-selection provenance [OWNER OD-12]:**
- **The rule applied:** the per-stone selection uses the **original** pre-specified rule (master plan v2 §G):
  - keep methods whose worst predicted-size-band Clopper–Pearson lower bound is ≥ nominal − 0.05;
  - take the smallest median relative width, with the 5% tie rule;
  - if none qualifies, carry U2.
- **The outcome:** applied to the saved final E6 statistics (no rerun), no method qualified for volume or 3D diameter, so U2 is carried for both.
- **The rejected proposal:** a post-hoc proposal (amendment 5: Clopper–Pearson upper bound ≥ nominal and point coverage ≥ nominal − 0.05) was written after viewing method-specific debug results on 29 fold-0 development scans that are part of the selection data. It is **not adopted**, and it is **not** pre-specified. Under it, U4 would have been chosen for diameter.
- **Scope of the correction:** no calibration or test data were involved. This corrects the selection provenance; it is not a change driven by test results.

---

## 3. Cohort

- **Authoritative design [OWNER OD-6]:** development 165, calibration 35, locked test **53** (`data/processed/li2022_working_split_v1.csv`, SHA-256 `466ec9c8…`). No Slicer-based exclusion is applied.
- **D-REV [OWNER OD-19, 13 November 2026]:**
  - No qualified radiology-trained reader is available in the project's time frame.
  - The **primary confirmatory cohort is 165/35/53**, against the dataset annotations as published.
  - The owner's non-radiologist screening (`data/processed/li2022_review_outcome_v1.csv`, SHA-256 `0d7bf638…`) defines the pre-specified sensitivity cohorts S1 and S2 (§5.1).
  - The radiology adjudication was not performed and is reported as a limitation. The blinded package stays available for future work.
- **Scans that fail processing** stay in the cohort. Their annotated pieces count as missed in the detection endpoints, and they are listed [PLAN, v1 §3].

---

## 4. Endpoints

### 4.1 P2: the single formal test [OWNER OD-1]

- **Test:** the paired median width ratio **P2-A / P2-B-CQR** over test patients at nominal 0.90.
  - Its 95% CI comes from a scan-level paired percentile bootstrap: 10,000 resamples, seed 20261105, the same resample indices for every paired quantity.
  - **Success requires both:**
    1. the upper CI limit is < 1;
    2. **the coverage validity check:** the Clopper–Pearson 95% **upper** bound of P2-A's patient-level coverage is ≥ 0.90.
- **This is not a non-inferiority claim.** It does not claim that coverage is proven to lie within a tight margin. It can only fail to detect under-coverage.
- **Reported descriptively for every burden method:**
  - coverage with its Clopper–Pearson CI;
  - the paired coverage difference P2-A − P2-B-CQR (bootstrap CI);
  - stone-free coverage;
  - coverage by burden tertile;
  - median width.
- **Multiplicity:** one formal test, so there is no Holm correction [OWNER OD-1].
- **Expected power (development evidence only, §6):** about 0.64.

### 4.2 P1: descriptive [OWNER, 12 November 2026, 09:00]

- **Measure:** marginal per-stone coverage of the Final arm's **volume** interval (U2) at nominal 0.90, over one-to-one matched test stones.
- **Reported:** the scan-cluster bootstrap 95% CI and the Clopper–Pearson CI. **No success criterion.**
- **Why descriptive:** the pre-specified power procedure (amendment 6) found no δ with ≥ 80% power.

### 4.3 Secondary and descriptive endpoints (estimates with CIs; no claims beyond them)

**Per-stone:**
- coverage by predicted-size stratum (< 5, 5–< 10, ≥ 10 mm, with the calibration merge rule [PLAN]), by annotated size (< 3, 3–< 5, 5–< 10, ≥ 10 mm), by HU tertile and by native slice group (≤ 1, 1–2, > 2 mm), each with the "not significantly below nominal" check;
- median relative width;
- per-patient "all stones covered" for U2;
- **N-B, patient-level joint-coverage interval** [OWNER OD-13; secondary; an established max-score mechanism (Andéol et al. 2025), never claimed as new]:
  - the share of test patients with all matched stones covered, with its CP CI, and the width inflation over U2, at 0.90 (0.80 descriptive);
  - calibration as in `scripts/secondary_na_nb_development.py`: size-band quantiles and τ come from the calibration scans, split by patient into two halves (seed 20261105);
  - development evidence: 0.934 joint coverage at 1.76× the U2 width (volume, 0.90);
- the **3D-diameter interval** of the Final arm (**U2**; OD-12) coverage and width **[OWNER OD-10, S-4: secondary only]**. U4 diameter is reported alongside, labelled as an exploratory comparator;
- the same summaries at **nominal 0.80** **[OWNER OD-10, S-5: 0.80 is descriptive only, for every endpoint]**.

**Detection and measurement:**
- detection by annotated size and HU;
- false positives per scan; stone-free scans with any false positive;
- calibration of the per-piece false-positive probability (Brier score, reliability table);
- bias and limits of agreement by size;
- Li-comparable Dice (E2 definitions).

**Baseline vs Final:** paired differences in coverage and width on the same stones and patients.

**Unit for per-stone coverage at T1 [OWNER OD-10, S-3]:**
- One-to-one matched stones are defined **after** the 30 mm filter, matching the deployed system.
- Disclosure: E6 on development used unfiltered matched stones. The effect is expected to be negligible: the filter removes mainly false positives.

---

## 5. Sensitivity, ablation and robustness (pre-specified)

### 5.1 Sensitivity analyses
All items are estimates only, never a second formal test. Each removes its scans, refits every arm with the identical procedure and seed, and has its calibration hashed together with the primary one **before** any test scan is loaded (`src/stonekit/t1/analysis.py`, `scripts/t1_run.py`).

**Pre-specified fragility analysis [OWNER, 13 November 2026]:**
- **The question:** does the P2 conclusion depend on the screened-flagged scans?
- **How it is answered:** the OD-1 statistics in S1 and S2 (width ratio with its 95% CI, the Clopper–Pearson coverage check, and the resulting pass/fail) are reported **side by side with the primary**.
- **Why it is pre-specified:** a development stand-in observation recorded before the freeze. In `t1_dress_rehearsal_20261113T071958Z` (one seeded split; not a result), removing the screened-flagged development scans flipped the formal P2 test from pass to fail.
- **Rule:** this analysis never replaces the primary result, never changes the locked cohort, and is reported whatever it shows.

1. **S1, screening exclusions [OWNER OD-19]:** remove every scan whose screening action is `exclude` (codes A or F). That is 3 test scans and 8 development scans; no calibration scan. They are removed from the fit, calibration and test sets. This is the screening-rule cohort (development 157 / calibration 35 / test 50).
2. **S2, all screened flags [OWNER OD-19]:** S1 plus every scan kept with a `sensitivity` mark (codes C or D). They are removed from all three sets.
   - **Selection bias stated in advance:** in the test set, S2 removes the 7 flagged scans, which hold 51% of test stone volume and 3 of 13 stones ≥ 20 mm (`docs/READINESS_ASSESSMENT.md`). S2 estimates are therefore not representative.
3. **S3, cross-split repeat pairs:** remove every reserved scan that is in a suspected repeat pair (the 7 November bone-projection audit, recomputed from the saved pre-split arrays) with a scan of another split group. They are removed from calibration and test. The list is computed only inside the T1 run, after every gate passes.
4. **Device subgroup:** no reserved scan is device-coded (the only one, wd0149, is development). Nothing to run; reported as such.
5. **Reader-confirmed annotation errors (secondary) [OWNER OD-17; NOT PERFORMED under OD-19]:** recorded in the T1 output as "not performed: no qualified radiology-trained reader". The specification below is kept for future work.
   - **Excluded scan:** any reserved scan where the blinded first-pass reader answer is Q2 = invalid, or Q1 = urinary stone at an unlabelled location (a label omission).
   - **Source of the answers:** the hashed adjudication sheet only. No technical-audit output and no screening code is used.
   - **Procedure:**
     - Calibration scans so marked are removed and every arm is refitted with the identical procedure and seed (`fit_burden`, U2 Mondrian, P2-B-CQR). Test scans so marked are removed from evaluation.
     - The OD-1 statistics (paired median width ratio P2-A/P2-B-CQR with its 95% bootstrap CI, and the Clopper–Pearson coverage bound) and the P1/secondary estimates are reported for the reduced sets.
   - **Status:** an estimate only, not a second formal test. **The primary analysis stays against the annotations as they are.**
   - **Reporting rule:** if no reserved scan is marked, report "no reader-confirmed annotation errors; analysis identical to primary".
   - **Relation to items 1–2:** independent of the owner's D-REV cohort decision (item 1). It overlaps item 2 only for the label-omission case.
   - **Limitation stated in advance:**
     - Only the 8 reserved scans with D-REV locations are reader-assessed. Annotation errors elsewhere in calibration or test cannot be detected (compare the unflagged development scan wd0252, OD-15).
     - With at most 1 calibration and 7 test scans affected, the change is expected to be small, and no conclusion is drawn from its size.

### 5.2 Robustness
- **Slice thickness:** observational only, by native slice group (0.42–2.5 mm). No artificial resampling [OWNER OD-5].
- **No claim** of robustness outside the native 0.42–2.5 mm range. **No weighted-conformal branch** [OWNER OD-11: amendment 8's conditional analysis dropped].
- **N-A, HU-conditional robustness** [OWNER OD-13; an established Mondrian method]:
  - per-stone coverage of a size-band × HU-tertile Mondrian interval is reported against U2, descriptively, by size × HU cell;
  - its pre-specified development rule was **not met** (`docs/results.md`, 12 November), so it is not a Final-arm method.
- **N-C, ablation** [OWNER OD-13]: a distance-only false-positive model in P2-A. This is development evidence only; it is not run at T1. On development it kept coverage but was 11% wider, so the full model stays.

### 5.3 Filter ablation (secondary) [OWNER OD-3]
- **Comparison:** filter on (deployed) vs filter off (the `baseline_pipeline.yaml` rule), using the same inference outputs (the filter-removed pieces are saved).
- **Reported:** false positives per scan, stone-free false-positive rate, per-stone coverage, and burden coverage and width for P2-0, P2-B, P2-B-CQR and P2-A.
- It never redefines the primary comparison.

### 5.4 External check on MSWAL (secondary, exploratory) [OWNER OD-21; specified before any MSWAL file is downloaded]
- **Data:**
  - MSWAL (MICCAI 2025; Hugging Face `zhaodongwu/MSWAL`; CC BY-NC 4.0, non-commercial research), released training cases only. Kidney stones are label 2.
  - **Eligible cases:** at least one label-2 voxel, and non-contrast. Non-contrast means the dataset's phase metadata says so where it is provided; otherwise the pipeline's existing contrast heuristic must not fire (aorta median < 100 HU, `report.json` `contrast_check`).
  - Cases that fail processing stay in, with their stones counted as missed.
- **What runs:** the frozen system exactly as at T1 (config, checkpoint and filter). **No retraining, tuning or recalibration on MSWAL.**
- **Intervals:** per-stone U2 and burden intervals use the **T1 calibration quantities** (fitted on the Li development set and calibrated on the 35 Li calibration scans). The external check therefore tests transfer of calibrated intervals to another hospital's data.
- **Run order:** once, after T1 has COMPLETED.
- **Endpoints (estimates with CIs; no formal test):**
  - detected share by size;
  - false positives per scan;
  - U2 per-stone coverage and width (volume and diameter, 0.90);
  - burden coverage and width for P2-0, P2-B, P2-B-CQR and P2-A (0.90).
- **Stated in advance:**
  - The annotation protocols differ (MSWAL masks vs Li labels).
  - The phase filter may leave few cases. Fewer than 10 eligible stone-bearing scans is reported as "insufficient for estimation", with descriptive counts only.
  - Overlap with the Li hospital is unknown.
  - Contrast scans are not used.

---

## 6. Calibration design and design evidence

### 6.1 What is fitted where

| Quantity | Fitted or calibrated on | Status |
|---|---|---|
| U0, U1, U2, U4 per-stone bounds | The **calibration** scans' one-to-one matched stones (35 scans, about 99 stones expected) | [PLAN] (T1: "calibration 35 → test 53"; master plan §G) |
| U2 strata | Predicted 3D diameter; merge rule (minimum 20 calibration stones) | [PLAN] |
| U4 signal floor | The calibration set's 5th percentile of TTA spread | [PLAN] (E6 fixed rule) |
| P2-0 per-stratum quantiles | **Calibration** matched stones, by predicted stratum, without merging (the E8 definition) | **[OWNER OD-10, S-1a]** |
| P2-A false-positive model | **Development** out-of-fold pieces (165 scans) | [PLAN] (master plan §F: "fitted on OOF") |
| P2-A residual pools and missed-burden pool | **Development** out-of-fold one-to-one residuals by predicted stratum; per-scan missed burden by slice group | [OWNER OD-11] (amendment 2) |
| P2-A scale factor t | **Calibration** scans (one log-scale factor) | [OWNER OD-11] (amendment 2; master plan §G) |
| P2-B quantile | **Calibration** scans | [PLAN] (master plan §G) |
| **P2-B-CQR quantile regression** | Fitted on **all development** scans; conformalised (CQR score) on the **calibration** scans | **[OWNER OD-10, S-2]** |
| Numerical settings | QuantReg max_iter 50,000; P2-A Monte Carlo draws seeded per scan from the case ID (order-independent); 2,000 draws | **[OWNER OD-10, S-1b]** |

**S-1 summary for the owner:** splitting component fitting (development) from conformal calibration (calibration scans) is **already approved** for P2-A and P2-B. Only S-1a (the source set for the P2-0 quantiles) and S-1b (numerical and seeding settings) are new.

**S-2 rationale:**
- Amendment 7 requires the strongest fair direct comparator.
- In E8, P2-B-CQR was fitted on half of the other folds and calibrated on the other half. Its median width was 8,702 mm³.
- Fitted on more development data, it is much narrower: about 2,300–5,300 mm³ depending on the split.
- Fitting it on all development scans is therefore the defensible T1 comparator.

### 6.2 Design evidence (development only; NOT confirmatory)

- **The E8 ratio is not an expected effect:** the E8 development ratio P2-A / P2-B-CQR of 0.234 (CI 0.192–0.295) was produced with the weaker half-split comparator.
- **The fairer projection is the T1 dry run on development** (`results/runs/t1_dry_run_development_20261112T100623Z/`):
  - 200 random splits into 35 / 53 / 77 scans;
  - median ratio about **0.64** (5th–95th percentile 0.09–1.73);
  - P(OD-1 success) about **0.64**;
  - the P2-A coverage check passes in 99.5% of splits.
- **Per-stone:** U2 volume coverage on 53 test scans is expected to be 0.79–0.97 (5th–95th percentile), with about 150 matched stones.
- These numbers include the 8 development scans that D-REV may make ineligible. **They are design evidence, not confirmatory evidence.**

---

## 7. Statistics

- **Unit:** the scan (patient). Stones are clustered within scans.
- **CIs:** a scan-cluster percentile bootstrap (10,000 resamples, seed 20261105), with Clopper–Pearson alongside. Driver models use GEE (independence working correlation). Subgroups with fewer than 5 scans are descriptive only.
- **One formal test (P2).** Everything else is an estimate with a CI.
- **Rule:** no test result feeds back into any choice.

---

## 8. Execution, firewall and integrity [PLAN]

- **Access:** the calibration and test IDs are opened only by the guarded T1 runner, after the freeze, against the frozen protocol SHA.
- **Order:**
  1. Run the frozen pipeline on the 35 calibration scans.
  2. Compute every calibration quantity in §6.1, then **hash and save them**.
  3. Run the 53 test scans.
  4. Apply the frozen calibration and evaluate.
- **One run, in a new timestamped folder.** Every report carries the commit, config, effective-config, checkpoint and protocol hashes.
- **No rerun after viewing results.** An engineering failure, such as a crash, may be rerun once, for the affected scan only, with the cause logged first. Every deviation is logged and reported.
- **Imaging data:** no reserved image is viewed by Claude Code, and no Li pixel leaves the machine.

---

## 9. Engineering acceptance before the run [PLAN]

- All tests pass and ruff is clean on the frozen commit.
- A smoke run on 2 development scans with the exact frozen configuration and engine.
- Memory and runtime checked (4 GB GPU; no other GPU application running).
- The measurement regression check and the engine–E6 equality tests pass.

---

## 10. Limitations stated in advance

- **Data and reference:**
  - Internal test only: same dataset and hospital.
  - The annotation is the reference; there is no physical truth.
  - Single fold-0 model, 100 epochs.
  - Annotation validity at the 24 flagged locations depends on D-REV. Labels not flagged are not verified by it.
- **Detection:** small and low-density stones are poorly detected (development d50 about 2.3 mm sphere-equivalent at median HU). Per-stone intervals cover detected (matched) stones only.
- **Uncertainty:**
  - Coverage is uneven by HU tertile on development.
  - Per-patient "all stones covered" is about 0.79 at nominal 0.90.
  - The false-positive model adds no discrimination beyond distance to the kidney (development AUROC 0.938 vs 0.944 for distance alone).
- **Statistical power:** P1 is descriptive. The formal P2 test has about two-thirds expected power. About 11 stone-free test scans are expected, so stone-free coverage is imprecise.
- **Leakage and acquisition:**
  - Suspected repeat pairs cross the development–test boundary (sensitivity analysis 3).
  - Within development, 3 suspected repeat pairs straddle the fold-0 boundary (amendment 4; development out-of-fold independence only).
  - Stones are clustered within patients. Per-stone conformal calibration therefore assumes exchangeable stones (see Dunn, Wasserman & Ramdas, JASA 2023, on hierarchical data). This is mitigated by patient-level splits and scan-cluster bootstrap CIs.
  - Fold-0 validation scans informed the 40 mm ROI margin and the 30 mm filter.
  - Seven test scans were viewed by a human reviewer for eligibility screening.
  - Claude Code saw pre-split descriptive statistics of all scans.
  - No acquisition outside 0.42–2.5 mm slices.

---

## 11. Decisions required before the freeze

| ID | Decision | Owner action needed |
|---|---|---|
| D-REV | Primary confirmatory cohort and sensitivity cohort | **DECIDED (OD-19):** primary 165/35/53; S1–S3 sensitivity; adjudication not performed |
| D-REV-dev | Whether E6–E8 must be rerun | **Not applicable (OD-19)** |
| External | MSWAL external check | **DECIDED (OD-21):** §5.4, secondary, after T1; the download needs owner approval of the files and size |
| S-1a | P2-0 per-stratum quantiles from the calibration matched stones (E8 definition, no merge) | **APPROVED (OD-10)** |
| S-1b | Numerical settings (QuantReg max_iter 50,000; per-scan seeding; 2,000 draws) | **APPROVED (OD-10)** |
| S-2 | P2-B-CQR fitted on all eligible development scans and conformalised on the calibration scans | **APPROVED (OD-10)** |
| S-3 | T1 per-stone coverage on matched stones after the 30 mm filter | **APPROVED (OD-10)** |
| S-4 | The diameter interval is reported as secondary only. The Final diameter method is U2 (OD-12); U4 is an exploratory comparator | **APPROVED (OD-10); method corrected by OD-12** |
| S-5 | Nominal 0.80 descriptive only | **APPROVED (OD-10)** |
| AMEND | Amendments 2, 3, 7 (comparator part) APPROVED (OD-11); 5 REJECTED (OD-12); 8's weighted-conformal branch DROPPED (OD-11); 1 and 6 superseded; 4 to be disclosed. The OD-11 text edits are applied (12 November, after OD-13) | **Settled** |
| Further arms | Any novelty arm developed later on development data must be added here before the freeze | As the work proceeds |
| Freeze | Approve this file; log its SHA | Yes |
| Commit | Approve the first commit; record the hash | Yes |

---

## 12. Conflicts still to reconcile (text only; the decisions log already outranks)

- **`docs/MASTER_RESEARCH_PLAN.md`:**
  - §C P1 (per-stratum hypothesis, m = 0.05) is superseded by amendments 1 and 6 and by the P1 descriptive decision;
  - §C P2 "≥ 0.90 − m" and "P2-A vs P2-B" are superseded by OD-1 and amendment 7;
  - §F "Deployed segmentation model" makes the ensemble an arm, and U6 is "locked-test arm only"; both are superseded by OD-4;
  - §G selection rule **stands** (OD-12; amendment 5 rejected as post hoc). The plan text needs no change here; the §G line about U6 is superseded by OD-4;
  - S6 resampling is superseded by amendment 8 and OD-5;
  - the T1 row ("incl. U6 ensemble", "Holm across the two") is superseded by OD-1 and OD-4;
  - the multiplicity line (Holm) is superseded by OD-1.
- **`paper/main.tex`:** line 63 (P1 non-inferiority, δ, Holm) and line 96 ("robustness (E10)") need rewording when the manuscript is updated.
- **`docs/CONFIRMATORY_PROTOCOL.md` (v1) and `docs/CONFIRMATORY_PROTOCOL_v2_DRAFT.md`:** superseded on this file's freeze. v2's §1 "50 test scans", §2 baseline config, §3 Holm and §9 D-REV default must not be used.
- **`docs/READINESS_ASSESSMENT.md`:** "7 inside stone labels, 2 unlabelled" for the test locations disagrees with the review sheet's location types (3 labelled, 6 unlabelled). Resolve during D-REV without reading test masks.
