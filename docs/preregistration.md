# Preregistration: reliability-aware measurement

## 1. Status, version, freeze date and hash
**Status: FROZEN, 3 November 2026. Approved by the owner.**

- **Version:** frozen version 1.
- **Freeze date:** 3 November 2026.
- **Hash:** a file cannot contain its own hash. The SHA-256 of this file is recorded in `docs/decisions.md`.
- **What this permits:** this fixes the study design. It does not by itself start implementation or any experiment. The prerequisites in section 21 must be decided first, and any later change is logged as a deviation (section 17).

This document covers the G5 study only. The original Stage 9 preregistration in `MASTER_PLAN.md` Section 10 stays unwritten.

## 2. Research question and scope
**Research question (owner-approved G5 wording):** determine whether physically motivated per-stone signals, primarily threshold sensitivity and density, can identify unreliable automated 3D kidney-stone volume measurements better than stone size alone, and whether conditional per-stone measurement ranges provide more informative uncertainty than global limits of agreement.

**Operational meaning:** "density" is represented by an attenuation-derived signal, not a direct physical density measurement (section 7). "Unreliable" means large disagreement with the supplied physical reference.

**Scope:**
- Automated 3D stone volume is the primary measurement target.
- Identifying unreliable measurements is the main question. Ranges and diameter are secondary.
- This is research software. It makes no diagnostic or treatment statement.

## 3. Prior exposure and leakage disclosure
This is an internal evaluation pre-specified after A0, not independent confirmation and not an untouched holdout. In this document, "pre-specified" means fixed before this study's per-stone analysis, not before A0.

| Already known before this study | Consequence |
|---|---|
| The original authors derived 272 HU from these same stones | A0's errors may be optimistic, and no fold structure undoes that |
| A0 and post-hoc diagnostics examined all 101 stones | Design knowledge cannot be removed by cross-validation |
| Post-hoc error patterns by peak attenuation and size | `M` is hypothesis-informed (section 7), and is not an independent discovery |
| Aggregate threshold sweep results | The `S_G` summary was written after they were seen (section 7) |
| Error differences by size, including a post-hoc 1 mL cut | That cut is not used in any rule or calibration here |

Cross-validation protects fitted parameters from held-out outcomes. It does not protect the design from prior knowledge.

**Per-stone exposure.** The a0 sweep generated per-stone masks and measurements in memory at each documented threshold, but saved only aggregate results. Per-stone `S_G`, `M` and `H` inside the 272 HU mask are not in saved outputs, and no relationship between `S_G` and error, or comparison between the full model and the size-only rule, has been analysed. Neubauer's supplements print a median HU for each stone, and they were seen during the literature search. Their numbering may not match this study's IDs, and they were not used as a data source.

| Category | Decisions |
|---|---|
| **A. Documented before the a0 run** | 272 HU; segmentation and hole-filling pipeline; spacing assumptions A and B; axes Z2, Z1 and Z0; D-paper; the threshold grid; flagged-reference IDs; float32 IDs; the spacing-agreement subset |
| **B. Made after aggregate A0 and post-hoc results, before this study's per-stone analysis** | `S_G` summary; `x1`; `H`, `T_HA` and `M`; log-ratio primary scale; model form; primary estimand and review range; fold and bootstrap designs; split-conformal procedure; use of the unreachable and 0.1 mL reference tiers |
| **C. Exploratory or post-hoc, and unable to replace primary results** | Local threshold elasticities; `ln H`; zero-parameter rank score; training-learned threshold; full risk-coverage curve; subgroup coverage; half-attenuation; mL limits; ICC |

## 4. Population and data provenance
- **Population:** all 101 Freiburg ex vivo stones, from one acquisition protocol. No patient population is studied.
- **Source:** the unchanged D1 files, checked against MD5s `871b9e5af6d1dccb738a42acb5a25ab7` (MAT) and `53f5df4a1d36e4d65c8c7b6eeec3e07f` (CSV) at run start.
- **Stone link:** `mat_index` and `stone_id` from `data/processed/freiburg_stones.parquet`. This inferred link is G2's recorded caveat.
- **Known data limits:** masked images with a 101–105 HU floor; no per-rater or per-stone species values; coarse reference volumes; 21 stones with disagreeing spacing fields; an unestablished slice axis; shared-scan structure unknown.

## 5. Frozen A0 baseline and recomputation exception
**Baseline:** A0 at 272 HU, from `results/runs/a0_replicate_freiburg_20261102T133018Z/`. It is a frozen baseline comparator only, under the ten G3 conditions in `docs/decisions.md`. It is not accepted as faithful replication or accurate measurement, is not independent validation, and is not clinically validated.

**Never:** rerun the A0 experiment, retune 272 HU, or modify, replace or overwrite any A0 output. `V_i^A`, `V_i^B` and saved D-paper values under Z2 are read from `a0_stones.csv`.

**Recomputation exception (approved by the owner, 3 November 2026).** A0 did not save masks, so a new run may apply the procedure in `docs/STAGE2_PROTOCOL.md` section 4 in memory. This is used only for the section 5 consistency check, `H`, `S_G`, the half-attenuation comparator and Z1/Z0 diameters. It is not an A0 rerun and creates no A0 result. Masks are not written to disk. At 272 HU, for all 101 stones:
- `mask_voxels`, `n_components` and `filled_voxels` must exactly match `a0_stones.csv`.
- `volume_A`, `volume_B`, `d_paper_A` and `d_paper_B` must match within a relative tolerance of 1e-9.
- The reused pipeline source files must match their hashes in A0's `run_manifest.json`.

Any mismatch stops the run, with no fallback. Recomputed values never replace A0's saved values.

## 6. Outcome definitions and reference limitations
Under spacing assumption `a` in {A, B}:

```text
E_i^a   = |ln(V_i^a / R_i)|                 primary error scale
APE_i^a = 100 * |V_i^a - R_i| / R_i          secondary error scale
```

- `V_i^a` is the saved A0 volume in mL under assumption `a` (`volume_A` or `volume_B`). `R_i` is the unrounded `reference_volume_ml` value in mL. All must be positive. Any non-positive value stops the run.
- **Why log-ratio is primary:** absolute percentage error caps underestimates at 100% but not overestimates, which can favour rules that accept under-measurement.
- **What log-ratio does not fix:** reference noise. If a 0.1 mL value is rounded, a perfect measurement could still show up to 50% percentage error (arithmetic, not a result).
- **Reading the scale:** `exp(E)` is the fold disagreement. Results describe disagreement with the supplied reference, not error against true physical volume.
- **Status of this choice.** This is a category B decision. Post-hoc A0 diagnostics had already examined log-ratio error by size and peak attenuation. No comparison of the study's rules has been computed on either scale.

## 7. Signal definitions
At most three signals. No predictor may be added without explicit owner approval. All use measured image data only, and the reference is never a predictor.

**Measured size.** `x1 = ln(mask_voxels)`. Voxel count does not depend on spacing, so the size-only flags are the same under A and B.

**Threshold-sensitivity signal, `S_G`.**
- **Documented grid.** `docs/STAGE2_PROTOCOL.md` section 9 and `configs/a0_replicate_freiburg.yaml` fixed the thresholds 101, 130, 150, 200, 250, 272, 300, 350 and 400 HU before the a0 run. The run manifest hashes both files.
- **Definition.** For each stone, run the frozen A0 pipeline in memory at every `t` in `G = {130, 150, 200, 250, 272, 300, 350, 400}` and record voxel count `N_t`. `S_G` is the negative of the ordinary least-squares slope of `ln(N_t + 1)` on `ln(t)`.
- **Why 101 HU is excluded.** The protocol recorded before the run that this value sits at the stored mask floor.
- **Zero and missing values.** `N_t = 0` remains valid and is flagged and counted. The +1 keeps `S_G` finite, so no value is missing or infinite, and no training-fold imputation is used.
- **Prior exposure.** The grid was documented before the run. The `S_G` summary was written on 3 November 2026, after aggregate sweep results had been seen, and before any per-stone threshold response was saved or analysed. The a0 sweep had generated per-stone values only transiently in memory. No relationship between `S_G` and error has been analysed. `S_G` is fixed at this study's freeze and is not described as preregistered before exposure.
- **Known weakness.** `S_G` is not purely local to 272 HU. Counts near 350–400 HU also depend on attenuation, so `S_G` may overlap with `M`.
- **No changes after results.** No grid value, exclusion or summary may change after the new study's results are seen.

**Attenuation-derived signal, `M`.**
- `H` is median stored attenuation in HU inside the recomputed 272 HU mask. It is not a direct physical density measurement.
- `T_HA = 0.5 * H` is the half-attenuation threshold, a measurement parameter used only in section 15.
- `M = |ln(T_HA / 272)|` is the model feature. It is physically motivated by the relationship between stone attenuation and threshold selection, and it assumes a background near 0 HU, which masked patches cannot verify.
- **Hypothesis-informed status.** `M` was motivated by patterns observed during earlier post-hoc analysis, so this is not an untouched feature-discovery study, and `M` is not an independent discovery. The study asks whether this pre-specified, hypothesis-informed signal adds information beyond size under the frozen analysis.

## 8. Comparator and model
**Comparator: size-only rule.** Rank stones by `x1`, smallest first, with ties by ascending `stone_id`, then copy number in bootstrap resamples, with no fitting. It stays primary because the question asks about information beyond size. Two limits are stated in advance:
- Measured size is mathematically part of the outcome, so over-measured stones look larger and are less likely to be flagged.
- Reference noise is expected to be relatively largest on small stones, though it cannot be measured here, so a size-only win may partly reflect the reference.

**Full reliability model (model family approved by the owner, 3 November 2026).**
- **Response:** `E_i^a`, non-negative and continuous.
- **Predictors:** `x1`, `x2 = S_G`, `x3 = M`. Each is standardized using the fitting-set mean and sample SD (`ddof = 1`). Held-out values are clipped to the fitting-set range before standardization.
- **Link:** `mu_i = exp(b0 + b1*x1 + b2*x2 + b3*x3)`, so `mu_i` is the expected `E`.
- **Model type:** log-link quasi-likelihood mean model with working variance proportional to `mu^2`. Only the conditional mean and this working variance are specified. `E` is not assumed to follow an exponential or any other named probability distribution. The coefficients estimate the conditional mean if the mean model is correct. If it is not, the fit remains a working ranking rule. No likelihood-based test, model standard error or goodness-of-fit result is used.
- **Estimation:** solve `sum x_i (E_i / mu_i - 1) = 0` by Fisher scoring, with no penalty: `b_new = b + (X'X)^(-1) X'(E/mu - 1)`. Start with `b0 = ln(mean E)` and all other coefficients 0. Halve any step that does not reduce `sum(x_i'b + E_i * exp(-x_i'b))`. Stop when the largest absolute coefficient change is below 1e-8, with at most 100 iterations. Non-convergence is counted and reported.
- **Zero values:** allowed. No floor or constant is added.
- **Extreme values:** no trimming or winsorizing. Their influence is examined only through section 14.
- **Scale:** `s_hat(x) = exp(x'b_hat)`.
- **Ranking:** larger `s_hat` is flagged earlier. Ties break by ascending `stone_id`, then copy number in bootstrap resamples.

## 9. Evaluation design and fold procedure
**Primary evaluation.** Repeated stratified 5-fold cross-validation, with 200 repeats, is the evaluation procedure for the primary reliability-ranking estimand only.
- **Fold assignment:** sort stones by ascending `x1`, with ties by `stone_id`. Starting from the smallest stone, form consecutive blocks of 5, and randomly assign one stone per block to each fold. The remaining largest stone goes to a random fold. Folds have 20–21 stones. Assignment uses no outcome. For repeat `r`, both rules and both spacing assumptions use the same folds.
- **Separation:** all fitting occurs on stones outside the held-out fold. Held-out outcomes do not enter fitting or flagging.
- **Flagging:** within each held-out fold, each rule flags its `k` highest-priority stones. Reference values are not used, and both rules review the same number of stones.
- **Why no nesting:** the model has no hyperparameters to select.
- **Why not leave-one-out or pooled ranking:** pooled ranks can let a stone's outcome affect its own flag through other folds' models.

**Secondary range evaluation.** Unstratified 10-fold outer splitting, with a separate training subset (60–61 stones) and calibration subset (30 stones) inside each outer training set, is used only for the secondary split-conformal range analysis (section 13).

**Separate roles.** These are not competing primary analyses. They answer different questions, and neither is selected, replaced or reinterpreted based on which looks better. Neither creates independent validation.

## 10. Primary estimand and uncertainty
There is one primary estimand, evaluated under the two unresolved spacing assumptions.

```text
For repeat r in 1..200, fold j in 1..5, and k in {2,3,4,5,6}:
  G(rule,r,j,k) = stones in fold j not among that rule's k flagged stones

  L(rule,a,r,k) = mean of E_i^a over the union across j of G(rule,r,j,k)

  Delta(a,r,k) = L(s,a,r,k) - L(f,a,r,k)

Weights (trapezoid area):
  w_2 = 1/8, w_3 = 1/4, w_4 = 1/4, w_5 = 1/4, w_6 = 1/8

Primary estimand:
  psi^a = (1/200) * sum over r, sum over k of [ w_k * Delta(a,r,k) ]
```

`s` is the size-only rule and `f` the full model, fitted under assumption `a`. `k` gives 10, 15, 20, 25 and 30 flagged stones out of 101, about 10–30%. This range is a disclosed convention, not a clinically derived workload. Positive `psi` favours the full model.

**Uncertainty.** A stone-level percentile bootstrap gives a 95% interval, used for description and not as a test.
- 2,000 resamples of 101 stones, paired across A and B, with the seed fixed in the config.
- The point estimate uses the 200 repeats on the original stones. Within each bootstrap resample, the same `psi` formula is averaged over 20 fold partitions, and this is used only for the interval. All fitting is repeated. Folds are built on unique stones, so all copies of a stone stay together. Each copy counts as one stone, `k` is unchanged and realized burdens may vary slightly. A fold with `k` or fewer stones makes that resample degenerate.
- Non-convergent resamples are left out of that percentile and counted. Over 1% labels the interval unreliable.
- Limits: fewer unique stones per resample may affect interval width, and shared-scan effects are ignored.

**Also reported:** Monte Carlo error across repeats, `psi` as a percentage of the size-only summary, and how many stones the two rules flag differently at each `k`.

## 11. Interpretation rules, including negative results
No result counts as passing or failing. There is no success threshold or practical-improvement anchor.

| Interval for `psi` | Pre-written reading |
|---|---|
| Above zero | The signals added measurable ranking information in this dataset, on supplied references. The size of the effect is the estimate shown |
| Contains zero | The data are compatible with no added information and with effects up to the interval's limits. This does not show the signals add nothing |
| Below zero | Size-only performed better in this dataset |

- **Spacing:** report `psi^A` and `psi^B` together. If they differ in sign, or only one interval excludes zero, the finding is called spacing-dependent. Neither assumption is selected.
- **Scale:** if `psi` on `E` and on percentage error differ in sign, the finding is called scale-dependent.
- **Negative results:** if the signals do not outperform size-only, that is the reported finding. The method will not be tuned until it wins, and the question or endpoint will not be changed to obtain a positive result.
- **A size-only win** may partly reflect reference noise and mathematical coupling (section 8).

## 12. Ordered secondary analyses and ablations
**Secondary estimands, in order:**
1. `Delta` averaged over repeats at `k` = 2, 4 and 6 (10, 20 and 30 flagged stones).
2. `psi` for the full model against the model without `S_G`, and against the model without `M`. An ablation cannot replace the full model, even if it performs better.
3. `psi` on `APE`.
4. Mean log-scale interval-score difference for the ranges, with coverage and width together (section 13).
5. Diameter: `psi` on `|ln(D-paper / reference diameter)|`, for axes Z2, Z1 and Z0 under A and B, using flags from the volume models. No axis is selected. Z1 and Z0 need in-memory recomputation.

All are descriptive, so no multiplicity correction applies.

**Exploratory and failure analyses, kept separate:**
- Full risk-coverage curve, and subgroup coverage.
- Half-attenuation comparator (section 15).
- Local threshold elasticities, which were selected after aggregate results were seen, and are exploratory and post-hoc: `|ln((N250 + 1)/(N300 + 1))| / ln(300/250)`, and the same form for 200 and 350 HU.
- `ln H` in place of `M`.
- Zero-parameter equal-weight rank score.
- A training-learned flag threshold, as a single-stone check. Realized burdens may differ between rules.
- Fitted size-only model, and the coefficient spread across folds.
- Sections 14 and 16.
- Conventional mL limits of agreement, and ICC as A0 context only. ICC is not an accuracy measure.

## 13. Split-conformal ranges and global limits
Ranges are secondary to the reliability question. This is not primarily a conformal uncertainty study.

**Procedure, repeated 200 times under A and B:**
1. Randomly partition all 101 stones into 10 outer folds. Each held-out fold is used only for evaluation.
2. In each outer training set, randomly assign 30 stones to calibration and the remaining 60–61 to a proper training subset.
3. Fit the scale model `s_hat` on the proper training subset only.
4. Compute `r_j = E_j / s_hat(x_j)` for calibration stones. `q` is the 28th smallest `r_j`, which is `ceil(31 * 0.90)`.
5. Freeze the model and `q`, then apply them to the held-out fold: `[V * exp(-q * s_hat), V * exp(+q * s_hat)]`.

No held-out outcome enters fitting or calibration.

**What it provides.** If stones are exchangeable, an assumption this study cannot verify, expected marginal coverage of the supplied reference is at least `28/31 = 90.3%`, and exactly that value with no tied `r_j`, over stones and calibration draws. For a single calibration set, coverage varies with an SD of about 5 percentage points (Beta(28, 3) arithmetic). This supports estimating average marginal coverage and width. It provides no size-conditional or physical-truth coverage guarantee, and no size-group calibration is performed.

**Comparators, fitted on the full outer training set unless stated:**
- **Log-scale limits, nominal 90% (main):** `[V * exp(-m - 1.645*SD), V * exp(-m + 1.645*SD)]`, where `m` and `SD` (`ddof = 1`) come from `ln(V/R)` in the outer training set. This nominal level assumes approximate normality, and it is not a conformal guarantee.
- **Size-only scale:** the same split-conformal steps, with a model containing only `x1`.
- **Conventional mL limits, nominal 95%:** bias ± 1.96 SD, reported for reference. Non-positive lower limits are counted, not clipped.

**Metrics.** Empirical coverage and median log-width, always together. Interval score with `alpha = 0.10`, on `ln R`:

`IS = (u - l) + (2/alpha) * [max(0, l - ln R) + max(0, ln R - u)]`

Secondary difference: mean `IS` for log-scale limits minus mean `IS` for the full-model ranges. Positive favours the full model. Uncertainty uses the grouped stone-level bootstrap, is approximate, and is descriptive only. In bootstrap resamples, calibration size can vary, so `q` uses order statistic `ceil((n_cal + 1) * 0.90)`, and the exact finite-sample statement above does not apply.

**Required language:** these are called **split-conformal ranges**, not "calibrated" ranges.
- **Marginal coverage:** targeted on average over stones and calibration draws, assuming stones are exchangeable.
- **Subgroup or size-group coverage:** observed empirically, never guaranteed. Size-conditional coverage is not claimed.
- **Coverage of the supplied reference:** the only coverage this study measures.
- **Coverage of true physical volume:** not measured, because the reference is noisy.

## 14. Spacing, axis, reference-problem and float32 handling
**Spacing.** A and B are both analysed and neither is treated as correct. Signals and size-only flags do not depend on spacing, but outcomes and model fits do. Sensitivity analysis: the 80 stones whose spacing fields agree. The paper's stated 0.7 × 0.7 mm spacing is not added, because A0 did not run it.

**Axis.** Volume is unaffected. Diameter is reported for Z2, Z1 and Z0, and no best axis is selected.

**Primary inclusion.** All 101 stones stay in the primary analysis. Reference-problem labels depend on the reference, which is part of the outcome, so excluding them up front would select on the outcome.

**Sensitivity analyses, removing stones and refitting:**
| Tier | Stones |
|---|---|
| Unreachable references | IDs 10, 19, 22, 46, 47, 49, 77, 99 (reference exceeds the stored region under B's voxel size) |
| Coarsest references | The stones with a stored reference of exactly 0.1 mL. Exactly 12 are expected, identified from reference values only. Any other count stops the run |
| Flagged references | IDs 1, 2, 3, 46, 62 |
| Union | All of the above |
| Float32 patches | IDs 46, 47, 56, 81, 83, 100. Data type marks possible provenance differences, not evidence of error |

Primary results stand whichever way these analyses turn out.

## 15. Half-attenuation comparator
An established comparator, not a contribution and not a predictor. Measure each stone at `T_HA = 0.5 * H` in one pass, with the frozen pipeline and no fitted factor. Because `H` exceeds 272 HU, `T_HA` stays above the mask floor. Report volume, `E` and `APE` against A0, across all 101 stones under A and B.

This asks whether better measuring reduces disagreement in this dataset. It does not alter `psi`. The reliability layer changes no volume value: it flags and gives ranges to measurements, and does not make them more accurate.

## 16. Failure analysis
Reported for every arm, whatever the primary result:
- Worst-error stones.
- High-error stones not flagged, and low-error stones flagged.
- Stones flagged by one rule but not the other, with their error.
- All reference-problem stones in section 14.
- Stones 74, 100 and 101, and the float32 patches.
- Stones with `N_t = 0` at any grid threshold, or non-monotonic counts.
- Composition where recorded. 42 stones are blank, and composition is entangled with attenuation, so no composition claim is made.

## 17. Stop rules and deviation logging
**Stop and report if:**
- any section 5 consistency check fails;
- source checksums fail;
- any `V` or `R` is non-positive, or `N_272 = 0`;
- any NaN or infinite predictor in the original stones, or a zero-SD predictor in any primary fitting set;
- the coarsest-reference count is not 12;
- the pipeline-integrity diagnostic triggers. This diagnostic is run before `psi` is computed on the real signals. For each spacing assumption, 100 permutations jointly shuffle `S_G` and `M` across stones, while `E` and `x1` stay fixed. Each permutation uses 20 fold partitions. The diagnostic triggers if mean permuted `psi` exceeds zero by more than 2 × (SD of permuted `psi` / √100). It is not a hypothesis test, success criterion or evidence about the signals. A trigger requires investigation and does not by itself prove a leak;
- model non-convergence in the primary analysis on the original stones.

**After a stop:** no threshold, signal, model, metric or exclusion may be changed to obtain a result. A true software fault is fixed with a test and disclosed, and all runs are reported.

**Deviations:** any change after freezing is logged in `docs/decisions.md` with its date and reason, and is reported in the results. The study has one analysis run in a new run folder. A further run is allowed only after a documented stop or software fault, and with the owner's approval. All runs are reported.

## 18. Permitted and forbidden claims
**Permitted, if results support it:**
> "In one ex vivo phantom dataset, in an internal evaluation pre-specified after A0 and using data previously examined post-hoc, adding threshold sensitivity and an attenuation-derived signal to measured size changed the summary of disagreement with the supplied reference among accepted stones by [psi^A] under spacing A (95% bootstrap interval [L, U]) and [psi^B] under spacing B ([L, U]), compared with a size-only rule. [Section 11 pre-written reading.]"

**Forbidden:**
- Novel or first method. G4 stays INSUFFICIENT EVIDENCE.
- Clinical usefulness, validity or diagnostic ability.
- Independent or external validation, generalization, or robustness to acquisition conditions.
- Conditional coverage by size, or coverage of true physical volume.
- That A0 is accurate or a faithful replication.
- That a high ICC proves accuracy.
- That `M` is an independent discovery, or that HU is direct physical density.
- That `S_G` was preregistered before exposure.
- Any attribution of disagreement to the method rather than the reference, or the reverse.
- That any category B or C decision was documented before A0.
- That exploratory or post-hoc results are preregistered findings.
- That the permutation diagnostic is evidence about the signals.
- That the ranges are "calibrated".

## 19. Reproducibility and required tests
**Records:** a new config and a new timestamped run folder. Fixed seeds. Nothing in the A0 run folder is overwritten, and the A0 folder's hash is recorded before and after the run. The manifest hashes source, tests, config, this document and the processed table. New work goes in new files, so A0's hashed files stay unchanged.

**Must exist and pass before any run:**
- Signal calculations, on hand-made shapes.
- Fold assignment, including grouped bootstrap copies.
- Leakage tests showing changed held-out outcomes do not alter fitted models, flags or calibration.
- The A0 reconciliation check.
- Range and calibration code, including the order statistic.
- Manifest and provenance for the new runner.

Test fixtures are hand-made unit-test arrays only, not datasets.

## 20. Future external validation, kept separate
External validation is not part of this study, and external data are not a prerequisite for it. Without external data, every claim stays limited to the Freiburg ex vivo phantom dataset and its acquisition and protocol.

If an independent physical-reference dataset is lawfully obtained later, its protocol is written separately, and the rule and ranges are fixed before that data is inspected. Scanning the same stones again does not count as independent stones.

## 21. Owner approval record
**Approved by the owner on 3 November 2026, including keeping threshold sensitivity in the primary model as `S_G`:**

| ID | Decision | Section |
|---|---|---|
| O1 | `S_G` over the documented grid; local pairs exploratory | 7, 12 |
| O2 | Size is `ln(mask_voxels)` | 7, 8 |
| O3 | `H`, `T_HA` and `M` kept separate; `M` hypothesis-informed; in-memory recomputation exception | 5, 7 |
| O4 | Partial risk-coverage primary estimand | 10 |
| O5 | Log-link quasi-likelihood mean model, as the owner's chosen model family | 8 |
| O6 | Estimate only, with no anchor | 11 |
| O7 | Repeated stratified 5-fold, 200 repeats, no nesting | 9 |
| O8 | Grouped bootstrap around the whole procedure | 10 |
| O9 | True split conformal, separate train and calibration subsets | 13 |
| O10 | Ranges secondary, with required language | 13 |
| O11 | One estimand under A and B; spacing-dependent wording | 11, 14 |
| O12 | Strict hierarchy | 10, 12 |
| O13 | Half-attenuation as a comparator only | 15 |
| O14 | All 101 in primary; tiered sensitivity | 14 |
| O15 | Float32 patches in primary; sensitivity analysis | 14 |
| O16 | Log-ratio primary, replacing the owner's earlier percentage-error endpoint with the owner's approval | 6 |

**Prerequisites before implementation, separate from the freeze (`MASTER_PLAN.md` A.7). Claude Code's recommendations await the owner's decision:**
| Item | Recommendation |
|---|---|
| Volume and diameter test grid incomplete | ACCEPT |
| Sphericity tested only at 0.5 mm | DEFER; not used here |
| Strict expected failure (r = 3 mm on 2.5 mm slices) | ACCEPT; keep strict, and revisit for thick-slice data |
| Whether to commit the uncommitted code | Owner's decision; a commit before implementation is recommended |

**Freeze approval:** given by the owner on 3 November 2026. The hash is in `docs/decisions.md`.
