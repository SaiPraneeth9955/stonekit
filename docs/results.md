# Results

Every number here must be copied from a file in a run folder, with its path (Section 12).
Stage 0 test counts are in `docs/status.md`; they are not research results.

## Li development loader check (5 November 2026; engineering check, not a result about stones)
`results/runs/li_development_loader_check_20261105T085858Z/summary.json`. All 165 development scans went through the guarded loader and input checks: 165 loaded, 0 load errors, 165 can be processed, 0 warnings, 0 failures, all LPS on disk. Stone volume, kidney volume, spacing, slice count and HU range differ from the audit table in 0 scans, within rounding tolerances. Load time median 5.11 s, maximum 25.33 s. Largest scan 232.0 million voxels. Whole-run peak memory 2,807,552 kB, for one scan at a time. The limits came from the audit of these same scans, so zero warnings is expected and doesn't test them on new data.

## Local training feasibility (5 November 2026; hardware timing only, not a result about stones)
`results/runs/phase_a_train_benchmark_20261105T082521Z/benchmark.json`. Random tensors on the RTX 3050, with no scan read. Plain 3D U-Net, mixed precision, SGD, batch size 2, without deep supervision or data loading.

| Patch (slices × in-plane) | Features (base, max) | Stages | GPU-only s/iteration | Training peak reserved (MiB) | Inference peak reserved (MiB) |
|---|---|---|---|---|---|
| 32 × 96 × 96 | 32, 320 | 5 | 0.288 | 1054 | 376 |
| 48 × 128 × 128 | 32, 320 | 5 | 0.733 | 3124 | 816 |
| 56 × 160 × 160 | 32, 320 | 6 | 9.977 | 5688 | 1392 |
| 64 × 192 × 192 | 32, 320 | 6 | 22.597 | 9290 | 2286 |
| 64 × 192 × 192 | 16, 160 | 6 | 1.573 | 4332 | 1066 |

Going past the 4096 MiB card did not raise an out-of-memory error. It got far slower instead, so a run can pass while being unusable. Data loading and augmentation are not measured, so times are lower bounds.

## Li 2022 data audit (5 November 2026; descriptive, no model or method evaluated)

Source: `data/interim/li2022_full_audit/` (git-ignored), mainly `li_full_summary.json`, with the two mismatched pairs from `mismatch/`. Thresholds are exploratory screens. Labels are annotation references, not physical truth. No CT slices were viewed by Claude Code.

**Engineering:** five zips match on size and MD5, CRC tests are clean, and 520 files cover 260 cases. 258 pairs agree on shape, spacing and affine. Two do not (`wd0097`, `wd0232`).

| Stone-label basis | With | Without | Other |
|---|---|---|---|
| Label 1 present in the file | 209 | 51 | Paper: 209 and 51 |
| After mask problems | 206 | 51 | 2 unknown (`wd0097`, `wd0218`), 1 duplicate removed |
| Complete label sets, one copy of the duplicate kept | 203 | 51 | 254 unique scans |
| Approved starting cohort, both duplicates excluded | 202 | 51 | 253 scans |

| Slice spacing (all 260) | Scans | Paper |
|---|---|---|
| 2.5 mm | 119 | 119 |
| 1.25 mm | 101 | 101 |
| 2.0 mm | 31 | 31 |
| 0.62 mm | 7 | 7 |
| 0.42 mm | 2 | 2 |

| Slice spacing (258 measured, by file) | With stone label | Scans |
|---|---|---|
| 2.0 mm | 16 | 31 |
| 2.5 mm | 91 | 119 |
| 1.25 mm | 92 | 100 |
| 0.62 mm | 7 | 7 |
| 0.42 mm | 1 | 1 |

- **Geometry of the 258:** 512×512; in-plane spacing 0.3711 to 0.9766 mm; 94 to 885 slices; 235 to 510 mm long; all LPS.
- **HU, all 260:** padding −3024 in 229 and −1024 in 31; maximum 1396 to 3071, with 55 reaching 3071.
- **HU, the 258:** whole-scan body median −87 to 52.
- **Kidney label:** median HU 11 to 62, median of medians 29; volume 56.662 to 1433.299 mL.
- **Alignment:** moving or flipping the kidney label never improved its soft-tissue share (median 0.977 in place, 0.826 at best when moved). The stone label fit best in place in 206 of 207, the exception being `wd0215`.
- **Stone regions:** 1,242 connected regions in the 207 files, including 2 from `wd0218` that look like kidney. Without those: under 3 mm 303, 3 to 5 mm 309, 5 to 10 mm 330, 10 mm or more 298. The duplicate pair is counted twice.
- **Repeat-scan screen:** one exact image duplicate, `wd0062` and `wd0192` (kidney Dice 0.88, stone Dice 0.274). Among 33,670 pairs, the top non-identical 2D score was 0.900. In 3D the top 30 pairs reached at most 0.751, 40 random pairs had median 0.476 and maximum 0.645, and a moved copy scored 0.998. No known real repeat was available as a control, so repeats are not excluded.

**Scientific:** the labels' correctness is not established. Unresolved items are in `docs/decisions.md`.

## r1: reliability-aware measurement, single approved run

Run folder: `results/runs/r1_reliability_freiburg_20261103T082418Z/`. Design: `docs/preregistration.md`, frozen version 1. Code: commit `68ab006`, clean tree. Seed 20261103. Deviations and open faults: `docs/decisions.md`. All numbers below are copied from that folder, except where marked as derived.

**Scope of every number here:** one ex vivo phantom dataset, an internal evaluation pre-specified after A0, on data previously examined post-hoc, against the supplied reference. Nothing here is independent validation or a clinical statement.

**Owner's adjudication:** inconclusive on the main question. See the r1 entry in `docs/decisions.md`.

**Engineering acceptance:** all six stages completed with no stop. Reconciliation with A0 had no mismatches. The integrity diagnostic did not trigger (`r1_integrity_diagnostic.csv`; 7 and 8 of 100 permutations failed, and the allowance is a logged deviation). The A0 folder is unchanged (`a0_unchanged` true; fingerprint `0ff0bb4f…457d`). **Limitation:** the ablation without `M` hit the preregistered 100-iteration limit in 22 (A) and 28 (B) of 200 repeats, always with stone 1 held out, so that comparison has no point estimate. A later diagnostic of this failure found the frozen iteration limit as its cause, and no false convergence among fits on the original 200 partitions. It did not audit other code or replay bootstrap or permutation failures (`results/runs/r1_diagnostic_no_m_20261103T093838Z/`; `docs/decisions.md`).

### Primary estimand (`r1_primary.csv`)

| Spacing | `psi` (E) | Monte Carlo SE | 95% bootstrap interval | Size-only summary | `psi` as % of size-only | Left-out resamples | Interval label |
|---|---|---|---|---|---|---|---|
| A | 0.015340 | 0.000678 | −0.020475 to 0.063244 | 0.372677 | 4.12% | 2 of 2,000 | none |
| B | 0.012681 | 0.000656 | −0.021020 to 0.062505 | 0.379701 | 3.34% | 21 of 2,000 (1.05%) | **unreliable** |

**Pre-written reading (Section 11):** the data are compatible with no added information and with effects up to the interval's limits. This does not show the signals add nothing. Not spacing-dependent and not scale-dependent under Section 11's definitions.

**Scientific acceptance:** no success threshold exists. This study did not show added ranking information beyond measured size. Relative to the size-only summary, the intervals run from about 5.5% worse to about 17% better (derived from this table), so neither no effect nor an effect of that size is excluded. Practical usefulness is not established in either direction, because no practical threshold was set.

### Ordered secondary estimates (`r1_secondary_estimates.csv`, `r1_flag_differences.csv`; descriptive)

| Estimate | A: estimate (interval) | B: estimate (interval), all labelled unreliable |
|---|---|---|
| `Delta` at k = 2 | 0.00426 (−0.01240, 0.02934) | 0.00272 (−0.01287, 0.02859) |
| `Delta` at k = 4 | 0.02424 (−0.02223, 0.06381) | 0.02083 (−0.02371, 0.06415) |
| `Delta` at k = 6 | 0.03388 (−0.03527, 0.09987) | 0.03319 (−0.03603, 0.09889) |
| Full vs without `S_G` | 0.01376 (−0.00575, 0.03785) | 0.01179 (−0.00658, 0.03647) |
| Full vs without `M` | no point estimate; 22 of 200 repeats failed; interval unreliable | no point estimate; 28 of 200 repeats failed |
| `psi` on `APE` (percentage points) | 3.554 (−4.291, 13.565) | 3.289 (−4.453, 14.095) |
| Diameter, Z2 | 0.00238 (−0.00262, 0.00455) | 0.00373 (−0.00166, 0.00546) |
| Diameter, Z1 | 0.00464 (0.00054, 0.00673) | 0.00622 (0.00184, 0.00821) |
| Diameter, Z0 | 0.00441 (−0.00083, 0.00671) | 0.00542 (0.00023, 0.00763) |

Stones flagged differently by the two rules, mean over repeats: at k = 2, 4 and 6, 4.60, 12.38 and 20.56 under A, and 4.26, 11.98 and 20.02 under B. The rules often flag different stones. The point estimates of the difference in accepted-stone error are small, but the intervals are too wide to conclude that the difference is small. No axis is selected. The diameter estimates are descriptive. An interval excluding zero does not make an effect useful: Z1's 0.00464 under A is about a 0.5% difference (derived).

### Split-conformal ranges (`r1_ranges_summary.csv`, `r1_ranges_interval_score.csv`; secondary)

Coverage is of the supplied reference only, and marginal.

| Spacing | Method | Mean coverage | Mean median width | Mean interval score |
|---|---|---|---|---|
| A | Full model | 0.9007 | 1.854 ln mL | 2.864 |
| A | Log-scale limits, nominal 90% | 0.9053 | 1.853 ln mL | 2.664 |
| A | Size-only scale | 0.9030 | 2.081 ln mL | 2.974 |
| A | mL limits, nominal 95% | 0.9506 | 1.207 mL | not applicable |
| B | Full model | 0.9005 | 1.940 ln mL | 2.940 |
| B | Log-scale limits, nominal 90% | 0.8995 | 1.885 ln mL | 2.730 |
| B | Size-only scale | 0.9027 | 2.118 ln mL | 3.047 |
| B | mL limits, nominal 95% | 0.9088 | 0.991 mL | not applicable |

Interval-score difference, log-scale limits minus full model (positive favours the full model): −0.200 (−1.970, 0.238) under A and −0.210 (−2.220, 0.192) under B. All interval-score intervals are labelled unreliable, as 80 of 2,000 resamples were left out. The point estimates favour the global log-scale limits, and the full-model ranges did not show more informative uncertainty. This is not evidence that conditional ranges are worse. Under Section 13, the log-scale limits subtract the training-set mean bias while the split-conformal ranges are centred on the measured volume, and these results do not separate centring from conditional width. The point estimates also sit far from the middle of their intervals, so the intervals are a rough description only. mL limits had non-positive lower limits for a mean of 73.66 stones (A) and 71.90 stones (B) out of 101.

### Exploratory, not findings of this study (`r1_exploratory_*.csv`)

These cannot replace the primary result, and they are unconfirmed. Section 14 says primary results stand whichever way sensitivity analyses turn out.

| Analysis | Stones | `psi` A (interval) | `psi` B (interval) |
|---|---|---|---|
| Without unreachable references | 93 | 0.0353 (−0.0163, 0.0924) | 0.0372 (−0.0149, 0.0948) |
| Without coarsest references | 89 | 0.0152 (−0.0273, 0.0630) | 0.0137 (−0.0273, 0.0614) |
| Without flagged references | 96 | 0.0571 (0.0047, 0.0974) | 0.0573 (0.0016, 0.0967) |
| Without union of tiers | 77 | 0.0569 (0.0069, 0.0979) | 0.0585 (0.0059, 0.0984) |
| Without float32 patches | 95 | 0.0323 (−0.0150, 0.0838) | 0.0340 (−0.0141, 0.0867) |
| Only spacing-agree stones | 80 | 0.0365 (−0.0208, 0.0880) | 0.0365 (−0.0208, 0.0880) |
| Local pair 250/300, post-hoc | 101 | 0.0149 (−0.0222, 0.0617) | 0.0122 (−0.0241, 0.0625) |
| Local pair 200/350, post-hoc | 101 | 0.0104 (−0.0239, 0.0647) | 0.0081 (−0.0246, 0.0639; unreliable) |
| `ln H` in place of `M` | 101 | 0.0110 (−0.0257, 0.0590) | 0.0093 (−0.0270, 0.0584) |
| Fitted size-only model | 101 | 0 | 0 |
| Zero-parameter rank score | 101 | 0.0222 (−0.0130, 0.0529) | 0.0215 (−0.0134, 0.0528) |

The intervals excluding zero come from removing stones chosen by their reference, which is part of the outcome. Section 14 excluded this selection from the primary analysis for that reason, and these rows must not be read as the study's result. Removing stone 46 also removes a high-error stone that size always flagged and the full model almost never did, so part of the rise may be mechanical. This has not been measured, and these rows do not show that reference noise hid a signal.

**Half-attenuation comparator (`r1_exploratory_half_attenuation.csv`).** Mean `E`: A0 0.3995, half-attenuation 0.2996 under A, difference 0.0999 (0.0693, 0.1373). A0 0.4069, half-attenuation 0.3071 under B, difference 0.0998 (0.0687, 0.1379). Median `E`: 0.3061 to 0.1778 under A, and 0.3061 to 0.1833 under B. Half-attenuation is an established method, not a contribution of this study. This comparison is exploratory on previously examined data, against the supplied reference. This compares two measurements across all stones, while `psi` compares two flagging rules, so the two numbers must not be compared for size. The idea was motivated by patterns already seen in these stones. Any follow-up needs its own new preregistration.

**Failure analysis (`r1_failure_analysis.csv`).** Largest `E` under A: stones 77 (1.997), 2 (1.783), 30 (1.438), 32 (1.406), 74 (1.260), 44 (1.248). At 4 stones per fold, stones 2, 44 and 62 were never flagged by either rule. Stone 46 (`E` 0.987) was always flagged by size and never by the full model. Composition is tabulated in `r1_exploratory_composition.csv` and is entangled with attenuation, so no composition claim is made.

### Section 18 statement

> In one ex vivo phantom dataset, in an internal evaluation pre-specified after A0 and using data previously examined post-hoc, adding threshold sensitivity and an attenuation-derived signal to measured size changed the summary of disagreement with the supplied reference among accepted stones by 0.0153 under spacing A (95% bootstrap interval −0.0205, 0.0632) and 0.0127 under spacing B (−0.0210, 0.0625, labelled unreliable at 1.05% left-out resamples), compared with a size-only rule. The data are compatible with no added information and with effects up to the interval's limits. This does not show the signals add nothing.

## a0: Freiburg replication, single approved run

Run folder: `results/runs/a0_replicate_freiburg_20261102T133018Z/`. Protocol: `docs/STAGE2_PROTOCOL.md`. Code: uncommitted tree on `91a2eaf`, with file hashes in `run_manifest.json`. Seed 20261102, 2,000 stone-level resamples, 0 degenerate resamples for any primary metric.

**Outcome: G3's numeric criterion is met, but stop rules 3 and 6 fired. G3 is not declared passed. This awaits the owner's review.** *(Superseded: the owner's G3 decision is in `docs/decisions.md`.)*

272 HU comes from the same authors and these same 101 stones, so nothing below is independent validation.

### Primary: 101 stones, 272 HU, axis assignment Z2 (`a0_primary.csv`)

Intervals are 95% percentile bootstrap. Bias is measured minus reference. Volume is in mL and diameter in mm.

| Assumption | Measure | Decides G3 | ICC(A,1) | ICC interval | Bias | SD | Limits of agreement |
|---|---|---|---|---|---|---|---|
| A | Volume | Yes | 0.99456 | 0.98780 to 0.99772 | +0.02495 | 0.30065 | -0.56433 to +0.61422 |
| A | D-paper | Yes | 0.99648 | 0.99349 to 0.99789 | +0.18330 | 0.67050 | -1.13088 to +1.49748 |
| A | D-centre | No | 0.99478 | 0.98988 to 0.99699 | -0.51670 | 0.67050 | -1.83088 to +0.79748 |
| B | Volume | Yes | 0.99626 | 0.98873 to 0.99808 | +0.03723 | 0.24982 | -0.45241 to +0.52687 |
| B | D-paper | Yes | 0.99688 | 0.99421 to 0.99818 | +0.17091 | 0.62971 | -1.06332 to +1.40515 |
| B | D-centre | No | 0.99503 | 0.99053 to 0.99715 | -0.52909 | 0.62971 | -1.76332 to +0.70515 |

The ICC formula matched pingouin's ICC(A,1) for all six rows.

### Stop rules (`a0_stop_rules.csv`)

| Rule | Fired | Finding |
|---|---|---|
| 1. G3 fails | No | All four deciding ICCs are above 0.99 |
| 2. Assumptions differ | Not on outcome | Both assumptions give the same outcome. The protocol sets no number for "materially different", so this needs the owner's judgement (see below) |
| 3. Borderline | **Yes** | Volume A is inside 0.985 to 0.995. The volume A and volume B intervals both contain 0.99 |
| 4. D-paper and D-centre differ | No | Both exceed 0.99 under both assumptions. D-centre under A is borderline, and its interval contains 0.99 |
| 5. Over 1% degenerate | No | Largest count is 0 |
| 6. Axis assignment matters | **Yes** | Z1 and Z0 under both assumptions move D-paper into the borderline band |

### Differences between assumptions A and B (`a0_stones.csv`)

- 21 stones differ in volume by more than 1%, by up to 10.3%. The other 80 agree.
- Stone 100: reference 12.33 mL, A 14.18 mL (+1.84), B 12.86 mL (+0.52). It is the largest volume error under A.
- Stone 101: reference 20.00 mL, A 19.44 mL (-0.56), B 20.69 mL (+0.69).
- Volume is inside the borderline band under A and outside it under B, though B's interval still contains 0.99.

### Axis sensitivity, D-paper (`a0_axis_sensitivity.csv`)

| Assignment | A: ICC (interval) | B: ICC (interval) |
|---|---|---|
| Z2 (primary) | 0.99648 (0.99349 to 0.99789) | 0.99688 (0.99421 to 0.99818) |
| Z1 | 0.99354 (0.98943 to 0.99574) | 0.99227 (0.98828 to 0.99478) |
| Z0 | 0.99360 (0.98946 to 0.99563) | 0.99238 (0.98823 to 0.99496) |

All six point estimates exceed 0.99. Z2 was fixed before the run, and its higher ICC is not evidence that axis 2 is the slice direction.

### Sensitivity analyses, none deciding G3 (`a0_sensitivity.csv`)

Results are for assumption A, then B.

| Analysis | n | Volume ICC | D-paper ICC |
|---|---|---|---|
| S1: without flagged IDs 1, 2, 3, 46, 62 | 96 | 0.99460, 0.99630 | 0.99684, 0.99724 |
| S2: without largest stones 100, 101 | 99 | 0.99197, 0.99181 | 0.99501, 0.99539 |
| **S3: only the 80 stones whose spacing fields agree** | 80 | **0.98976, 0.98976** | 0.99506, 0.99506 |
| S4: without float32 patches | 95 | 0.99597, 0.99592 | 0.99654, 0.99644 |
| S5: largest component instead of union | 101 | 0.99452, 0.99621 | 0.99648, 0.99688 |

**S3 volume is below 0.99** (interval 0.97475 to 0.99450). This subset is the one that does not depend on the spacing choice. The 21 stones it leaves out include the largest, 100 and 101. The cause has not been tested.

### Exploratory threshold sweep (`a0_sweep_exploratory.csv`)

**Exploratory only. No threshold is selected, and none of these ICCs is independent validation.** One stone has an empty mask at 400 HU and is kept with zero volume and diameter.

| HU | Volume A | Volume B | D-paper A | D-paper B | D-centre A | D-centre B |
|---|---|---|---|---|---|---|
| 101 | 0.98524 | 0.98535 | 0.97906 | 0.97899 | 0.98767 | 0.98754 |
| 130 | 0.98924 | 0.98929 | 0.99110 | 0.99117 | 0.99651 | 0.99652 |
| 150 | 0.99132 | 0.99142 | 0.99317 | 0.99357 | 0.99745 | 0.99775 |
| 200 | 0.99390 | 0.99463 | 0.99606 | 0.99648 | 0.99746 | 0.99776 |
| 250 | 0.99469 | 0.99606 | 0.99634 | 0.99679 | 0.99565 | 0.99596 |
| 272 | 0.99456 | 0.99626 | 0.99648 | 0.99688 | 0.99478 | 0.99503 |
| 300 | 0.99404 | 0.99608 | 0.99577 | 0.99621 | 0.99289 | 0.99316 |
| 350 | 0.99167 | 0.99431 | 0.99472 | 0.99498 | 0.98974 | 0.98986 |
| 400 | 0.98702 | 0.99021 | 0.98840 | 0.98864 | 0.98131 | 0.98138 |

### Segmentation details at 272 HU (`a0_stones.csv`)

- Holes were filled in 17 stones, 1,786 voxels in total.
- More than one component kept: ID 74 (4 components) and ID 100 (3). ID 94 keeps 1. Protocol section 4 lists ID 94 as having more than one component. The run files record only the components kept, so they cannot show why these differ; see `docs/A0_DIAGNOSTIC_REPORT.md`, section 2.
- No mask is empty at 272 HU.

### Descriptive check added after the run (not in the protocol)

The ICC is high across a wide range of volumes (0.01 to 20 mL, median about 0.19 mL), but per-stone volume agreement is much looser. Median absolute volume error relative to the reference is 30.2% under A and 31.7% under B. 55 stones under A and 56 under B differ by more than 25%. The limits of agreement (about +/-0.5 to 0.6 mL) are wider than the median stone's volume. This was computed from `a0_stones.csv` after seeing the results and does not change any gate.

### Limitations

- The reference values have no per-rater data here, so no inter-rater analysis is possible.
- The paper's Table 1 reports volume ICCs of 0.988 to 0.991 and its text says "ICC>0.99". How the paper set up that comparison may differ from this one, so the numbers are not directly comparable.
- Neither spacing assumption and no axis assignment has been established as correct.

## Phase C and D setup (5 November 2026, development scans only)

Engineering evidence only. Annotation labels are not physical truth, and these numbers are not validation.

Kidney segmentation (TotalSegmentator 2.18.0, `--fast`, kidneys and aorta, GPU): 164 of 165 development scans; median 40.0 s per scan. One long thin-slice scan (885 slices at 0.416 mm) failed on all three attempts. The first attempt was confirmed out of memory by the kernel log, after 2273 s. Two low-memory retries also failed, and their saved error text does not show the cause. It is a reported known failure and is in neither model split.

ROI coverage (`results/runs/li_roi_coverage_20261105T154719Z/`; 164 scans, 131 with stone labels, 745 label pieces):

| Box margin (mm) | Scans with all label inside | Pieces whole | Median crop (million voxels) | Largest crop (million voxels) |
|---|---|---|---|---|
| 0 | 117 | 727 | 1.8 | 5.7 |
| 10 | 126 | 737 | 3.0 | 9.7 |
| 20 | 130 | 744 | 4.4 | 14.6 |
| 30 | 130 | 744 | 6.3 | 20.9 |
| 40 | 131 | 745 | 8.4 | 28.6 |
| 50 | 131 | 745 | 11.1 | 36.5 |
| 75 | 131 | 745 | 19.7 | 57.7 |

Chosen box margin: 40 mm, the smallest tested that kept all labels. One scan alone caused the loss at 20 and 30 mm, and its kidney Dice is low (0.357); the cause was not investigated. Farthest stone label from the kidney mask (percentiles 50, 90, 95, 100): 6.2, 23.5, 28.8, 68.8 mm. Threshold comparator distance: 30 mm, with 125 of 131 scans keeping all labels. TotalSegmentator kidney Dice against Li labels (min, quartiles, max): 0.301, 0.842, 0.891, 0.912, 0.939. One scan had one kidney found. Aorta median HU per scan (min, 5th, median, 95th, max): 33, 36, 42, 47, 49, so the 100 HU contrast heuristic raised no flag on these scans.

Training crops (`results/runs/li_nnunet_dataset_20261105T161832Z/`): 122 training, 32 validation, 1 not written. 901,386 stone voxels with 0 outside the crop. Raw crops 2.3 GB.

Demo cases, chosen by the fixed rule before any model output (`results/runs/demo_case_selection_20261105T161832Z/`): ordinary wd0063, small wd0231, large wd0128, multiple wd0042, no stone wd0022, suspicious wd0149.

nnU-Net plan: 3d_fullres, patch 48x112x192, batch 2, spacing 2.0 x 0.756 x 0.756 mm. One-epoch trial (`results/runs/stone_baseline_training_nnUNetTrainer_1epoch_20261105T162347Z/`): 286.73 s per epoch, peak GPU memory 3888 MiB, peak system RAM used 6337 MiB, about 25 to 58 s per validation scan. Its Dice is a speed-trial output with no scientific meaning.

## Stone-level analysis of the 100-epoch baseline (7 November 2026; exploratory development evidence)

Source: saved outputs in `data/interim/stonekit_runs/validation_20261106T003707Z/`, which were not rerun. Run folder: `results/runs/stone_analysis_baseline_validation_20261107T032442Z/`. Definitions are in the docstring of `scripts/analyse_stone_validation.py`. Covers 32 fold-0 validation scans, 187 annotated pieces of at least 3 voxels (26-connected), compared with Li annotations, which are not physical truth. Intervals are 95% bootstraps over scans.

| Measure | nnU-Net 100 epochs | 272 HU threshold |
|---|---|---|
| Annotated pieces detected (any overlap) | 98/187 = 0.524 (0.407-0.796) | 122/187 = 0.652 (0.581-0.818) |
| Annotated stone volume in detected pieces | 0.980 (0.949-0.998) | 0.992 (0.982-0.999) |
| Predicted pieces overlapping an annotation | 105/154 = 0.682 (0.518-0.802) | 150/1130 = 0.133 (0.065-0.213) |
| False-positive pieces per scan | 1.53 | 30.63 |
| No-stone scans with any predicted piece | 4/6 | 6/6 |
| Groups (annotated side): one-to-one / split / complex / missed | 90 / 6 / 2 / 89 | 104 / 18 / 0 / 65 |

Detection by annotated 3D diameter (model; threshold): <3 mm 2/40 = 0.05; 5/40 = 0.125. 3-<5 mm 13/39 = 0.333; 21/39 = 0.539. 5-<10 mm 24/43 = 0.558; 33/43 = 0.767. >=10 mm 59/65 = 0.908; 63/65 = 0.969. Median diameter of missed pieces was 3.37 mm against 13.34 mm for detected pieces (model).

Detection by annotated peak attenuation (model): <400 HU 12/82 = 0.146; 400-<800 HU 21/38 = 0.553; >=800 HU 65/67 = 0.970. 82 of 187 annotated pieces peak below 400 HU. Attenuation is not composition.

Detection by slice spacing (model): 1.0-<1.5 mm 47/90 = 0.522; >=2.2 mm 42/87 = 0.483. The <1.0 and 1.5-<2.2 mm bands hold only 1 and 9 pieces, so no conclusion is drawn for them.

Agreement on one-to-one groups:

| Measure | Model, n=90 | Threshold, n=104 |
|---|---|---|
| Volume difference, mean % of pair mean (95% interval) | -33.1 (-39.6 to -24.5) | -73.0 (-86.1 to -51.2) |
| Limits of agreement | -94.1 to +27.9 | -164.4 to +18.4 |
| 3D diameter difference, mean mm (95% interval) | -1.18 (-1.52 to -0.76) | -2.23 (-2.99 to -1.38) |
| 3D diameter limits of agreement (mm) | -3.90 to +1.55 | -6.64 to +2.17 |
| Axial diameter difference, mean mm | -1.21 | -1.96 |

Model volume bias by size: 3-<5 mm -72.7% (n=13); 5-<10 mm -36.9% (n=23); >=10 mm -20.9% (n=52). Spearman correlation between size and absolute percentage error: rho -0.645 (n=90). Pieces within a scan are not independent, so this p-value is not reported as evidence.

Interpretation, kept apart from the numbers above:
- **Small stones:** the model's misses are concentrated in small, low-attenuation annotated pieces.
- **Volume:** detected pieces hold 98% of annotated volume, so total-volume summaries hide the small-stone weakness.
- **Diameter bias:** the model segments about 1.2 mm smaller in diameter than the annotations at every size. This is consistent with boundary under-segmentation, and it is a bias against the annotation, not a proven physical error.
- **Threshold comparator:** it detects more small pieces, but at 30.6 false pieces per scan it is not a usable detector.

## Measurement-code verification on digital shapes (7 November 2026)

Run: `results/runs/measurement_phantom_verification_20261107T033342Z/`; definitions are in the docstring of `scripts/verify_measurement_phantoms.py`. These are known-size spheres and 2:1 ellipsoids voxelised at three spacings, 25 sub-voxel offsets each. They verify StoneKit's measurement code, not patients or segmentation.

- Volume: mean error within about ±1% for spheres of 4 mm and larger at all three spacings. Small shapes on coarse grids are unreliable: a 2 mm sphere at 0.75 x 0.75 x 2.5 mm is +45% on average, and a 2 mm in-plane ellipsoid at that spacing is +183%. *(Added 8 November 2026.)* Single-position errors are much wider than these means when a stone spans few slices: on 2.5 mm slices a 2:1 in-plane ellipsoid ranges from -25% to +15% at 10 mm and from -83% to +85% at 4 mm. Averaged over positions, every shape is within ±3% from 7 mm. A rerun gave a byte-identical summary (`results/runs/measurement_phantom_verification_20261107T220928Z/`).
- 3D diameter, which is centre-to-centre: always short of the true long axis, by about one voxel along that axis. Spheres -0.02 to -0.74 mm. Ellipsoids with their long axis along z: about -0.7 mm at 0.7 mm slices, -1.1 mm at 1.25 mm slices and -2.0 mm at 2.5 mm slices.
- Meaning: part of any diameter difference against a physical reference comes from the diameter definition and the slice spacing, not from segmentation. Comparisons with Li annotations use the same definition on both sides, so that part cancels there.

## Post-processing sensitivity, 100-epoch model (7 November 2026, exploratory)

Run: `results/runs/postprocessing_sensitivity_model_20261107T033410Z/`. Based on 154 predicted pieces, 49 of them false positives, and 98 detected annotated pieces.

| Rule | False positives kept | True pieces removed | Annotated detected afterwards (approx.) |
|---|---|---|---|
| No rule | 49 | 0 | 98 |
| Volume >= 10 mm3 | 40 | 7 | 93 |
| Volume >= 20 mm3 | 36 | 17 | 85 |
| Volume >= 50 mm3 | 33 | 26 | 79 |
| Within 20 mm of kidney | 30 | 0 | 98 |
| Within 10 mm of kidney | 13 | 3 | 95 |
| Within 5 mm of kidney | 8 | 10 | 88 |

A minimum-size rule removes true small stones faster than false positives, so none is adopted. The distance rule was not chosen from this table; see the pre-specified ablation in docs/decisions.md.

## 100 vs 250 epochs, and the 30 mm kidney-distance filter (7 November 2026; development evidence, model selection)

Rules and endpoints were fixed before the results existed (`docs/decisions.md`). Same 32 fold-0 validation scans, same pipeline, same matching definitions (`scripts/analyse_stone_validation.py`). Compared with Li annotations, which are not physical truth.

Training of the 250-epoch model (`results/runs/stone_baseline_training_nnUNetTrainer_250epochs_20261107T032526Z/`):
- 03:25 to 18:05 UTC on 7 November, 202 to 230 s per epoch.
- nnU-Net's own fold-0 validation Dice: 0.592.
- Its EMA pseudo-Dice was 0.92 near the end, with losses flat over the last 50 epochs. Pseudo-Dice is computed on training-time patches and is not comparable with the pipeline numbers below.

Validation run: `data/interim/stonekit_runs/validation_250epochs_20261107T182634Z/`, summary `results/runs/stone_baseline_validation_250epochs_20261107T193928Z/`. 32 of 32 scans processed, 0 failures. Median 98.5 s per scan (maximum 219.1 s), with the kidney step reused from saved masks.

Stone-level analyses:
- 100 epochs: `results/runs/stone_analysis_baseline_validation_kd30_20261107T184831Z/`. It reproduces the earlier 100-epoch analysis exactly and adds the filtered variant.
- 250 epochs: `results/runs/stone_analysis_250epochs_validation_kd30_20261107T195842Z/`.

| Measure (187 annotated pieces, 32 scans) | 100 ep | 100 ep + filter | 250 ep | 250 ep + filter |
|---|---|---|---|---|
| Detected (any overlap) | 98 = 0.524 (0.407-0.796) | 98 | 112 = 0.599 (0.485-0.858) | 112 |
| <3 mm | 2/40 | 2/40 | 3/40 | 3/40 |
| 3-<5 mm | 13/39 | 13/39 | 20/39 | 20/39 |
| 5-<10 mm | 24/43 | 24/43 | 29/43 | 29/43 |
| >=10 mm | 59/65 | 59/65 | 60/65 | 60/65 |
| Peak <400 HU | 12/82 | 12/82 | 21/82 | 21/82 |
| Peak 400-<800 HU | 21/38 | 21/38 | 26/38 | 26/38 |
| Peak >=800 HU | 65/67 | 65/67 | 65/67 | 65/67 |
| Predicted pieces overlapping an annotation | 105/154 = 0.68 | 105/138 = 0.76 | 123/170 = 0.72 | 123/150 = 0.82 |
| False-positive pieces (per scan) | 49 (1.53) | 33 (1.03) | 47 (1.47) | 27 (0.84) |
| No-stone scans with any predicted piece | 4/6 | 2/6 | 5/6 | 2/6 |
| Mean Dice, 26 scans with stones | 0.694 | 0.733 | 0.707 | 0.765 |
| One-to-one pairs | 90 | 90 | 103 | 103 |
| Volume difference, mean % (95% interval) | -33.1 (-39.6 to -24.5) | same | -40.5 (-49.4 to -29.9) | same |
| Volume limits of agreement (%) | -94.1 to +27.9 | same | -114.4 to +33.4 | same |
| 3D diameter difference, mean mm (95% interval) | -1.18 (-1.52 to -0.76) | same | -1.22 (-1.49 to -0.89) | same |
| 3D diameter limits of agreement (mm) | -3.90 to +1.55 | same | -4.00 to +1.56 | same |

The 272 HU threshold comparator is unchanged: 122/187 detected, 30.6 false pieces per scan.

Paired differences, 250 minus 100 epochs, both unfiltered. Bootstrap over scans, same resamples for both (`results/runs/paired_comparison_schedule_100_vs_250_20261107T195846Z/`):
- Detection +0.075 (+0.039 to +0.110).
- False positives per scan −0.06 (−0.56 to +0.44).
- Dice +0.013 (−0.050 to +0.082).
- <3 mm detection +0.025 (0.000 to +0.083).
- 3-<5 mm detection +0.179 (+0.071 to +0.357).
- Volume bias −7.5 points (−13.0 to −2.6).

**Pre-specified schedule rule: the 100-epoch model is kept, because the volume-bias criterion failed (7.5 > 5 points).**

Post-hoc diagnostic, which does not change that outcome (`results/runs/common_pair_bias_100_vs_250_20261107T195920Z/`). On the 85 pieces matched one-to-one by both models:
- Volume bias −30.7% against −31.8%, difference −1.1 (−4.8 to +2.0).
- 3D diameter difference +0.02 mm (−0.10 to +0.12).
- 18 pieces are one-to-one only under 250 epochs: median annotated diameter 5.1 mm, mean volume difference −81.6%. The 5 pieces one-to-one only under 100 epochs had a median of 8.2 mm and −73.5%.
- The worse pooled bias reflects newly matched small stones, not worse measurement of the same stones.

Filter, paired differences against the same model unfiltered:
- 100 epochs: false positives per scan −0.50 (−0.78 to −0.25); Dice +0.039 (+0.004 to +0.102).
- 250 epochs: false positives per scan −0.63 (−0.97 to −0.31); Dice +0.058 (+0.013 to +0.113).
- In both cases detection, pairs and agreement are identical. **Pre-specified filter rule: adopted** (`configs/system_pipeline_v1.yaml`). The run folders are `results/runs/paired_comparison_filter_100epochs_20261107T184849Z/` and `results/runs/paired_comparison_filter_250epochs_20261107T195847Z/`.

Interpretation, kept apart from the numbers:
- Longer training mainly helps 3-<5 mm and low-attenuation pieces. Stones under 3 mm stay almost entirely missed under both schedules (2 and 3 of 40).
- Detected pieces hold about 98% of annotated volume under both models, so totals hide this weakness.
- Every interval here is over 32 development scans that were also used to choose between the models. They are not confirmatory. The reserved test set is the place for that, after the owner's Slicer review and approval.


## Capability verification (8 November 2026)

Corrected bootstrap intervals. Before 8 November, `bootstrap_mean` in `scripts/analyse_stone_validation.py` used only 2000 of its stated 10,000 resamples (`docs/decisions.md`). The intervals of mean agreement values above were recomputed from the saved per-piece rows with all 10,000 (`scripts/summarise_saved_analysis.py`). Every count matched the original summaries exactly. Point estimates, limits of agreement and detection intervals are unchanged.
- 100 epochs: `results/runs/stone_analysis_baseline_validation_kd30_corrected_bootstrap_20261107T222342Z/`
- 250 epochs: `results/runs/stone_analysis_250epochs_validation_kd30_corrected_bootstrap_20261107T222345Z/`

| Mean agreement (95% interval) | Before | Corrected |
|---|---|---|
| 100 epochs, volume % | -33.1 (-39.4 to -24.6) | -33.1 (-39.6 to -24.5) |
| 100 epochs, 3D diameter mm | -1.18 (-1.52 to -0.76) | -1.18 (-1.52 to -0.76) |
| 250 epochs, volume % | -40.5 (-49.2 to -29.8) | -40.5 (-49.4 to -29.9) |
| 250 epochs, 3D diameter mm | -1.22 (-1.49 to -0.86) | -1.22 (-1.49 to -0.89) |
| Threshold, volume % | -73.0 (-85.7 to -51.0) | -73.0 (-86.1 to -51.2) |
| Threshold, 3D diameter mm | -2.23 (-2.97 to -1.38) | -2.23 (-2.99 to -1.38) |

The three paired comparisons were rerun from the saved rows and reproduced their original summaries exactly, including the rule outcomes: the filter is selected for both models, and the 100-epoch model is kept (`results/runs/paired_comparison_*_recheck_20261107T2223*Z/`).

Real end-to-end check on validation scan wd0086 (not a demo case) with `configs/system_pipeline_v1.yaml`:
- **Saved kidney mask:** 146.9 s, peak RSS 2.2 GB, result status complete, 3 model pieces.
- **Fresh TotalSegmentator:** 181.7 s (kidney step 62.1 s, model step 67.3 s), peak RSS 4.6 GB, no process left.
- **Equivalence with the frozen-baseline validation run of the same scan** (same checkpoint): the filter removed exactly the one piece farther than 30 mm. The 3 kept pieces are identical on every measure. The saved label map equals the baseline map with that piece zeroed. The threshold comparator is identical and unfiltered.
- **Annotation Dice:** 0.720 for the baseline and 0.726 for the system configuration.
- **TotalSegmentator repeatability:** a fresh run differed from the saved kidney mask in 36 of 90,963,968 voxels, so it is not bit-reproducible.

wd0001 (system run of 7 November): wd0001 is a training scan whose annotation has no stone. Its 5 predicted pieces, the largest 798 mm³, are false positives relative to the annotation. The run is an engineering check, not evidence of detection.


## Scientific completion pass (8 November 2026)

All numbers, with sources, are in `docs/CHARACTERISATION_STUDY.md`. The key results:

| Result | Value | Source |
|---|---|---|
| Missed stones with stone probability < 0.05 | 84 of 89 | `results/runs/detection_diagnosis_20261107T234215Z/` |
| Threshold test (pre-specified; 0.4 to 0.1) | No candidate passes; detection gain at most 1 piece | same |
| 50% / 80% / 90% detection size, 100 epochs | 6.2 (3.3 to 8.2) / 11.6 (5.3 to 15.6) / 16.8 (6.8 to 23.2) mm | `results/runs/detection_characterisation_20261107T225115Z/` |
| Same, 250 epochs | 4.8 (2.7 to 6.3) / 9.4 (3.9 to 14.5) / 13.9 (4.9 to 24.1) mm | same |
| Detection by peak-HU tertile (<311, 311-<855, ≥855), 100 epochs | 5/63, 33/62, 60/62 | same |
| Volume bias by size (<5, 5-<10, 10-<20, ≥20 mm), 100 epochs | −69.3, −36.9, −31.3, −13.3% | same |
| Kidney repeat (31 scans): ROI identical; stones matched; largest measurement change | 31/31; 136/136; 0.68% of one stone; 0 category changes | `results/runs/kidney_repeatability_20261108T013217Z/` |
| Kidney Dice, TotalSegmentator against Li label 2 (164 scans) | median 0.891, minimum 0.301, 25 below 0.8 | `results/runs/filter_retention_20261107T225644Z/` |
| Low-memory kidney fallback against normal (6 scans) | Dice 0.948 to 0.988; ROI shift ≤ 3.3 mm; 0 stone voxels lost | `results/runs/kidney_lowmem_equivalence_20261108T010544Z/` |
| wd0031 end to end | Complete; Dice 0.932 against annotation; 355.5 s | `data/interim/stonekit_runs/wd0031_20261108T011522Z/` |
| Filter cost: development pieces wholly beyond 30 mm | 4 of 738 (2 scans) | `results/runs/stone_distance_pieces_20261107T235437Z/` |
| Conformal (leave-one-scan-out), Mondrian, 90% | 88.9% overall; 94/89/86% by size | `results/runs/conformal_exploratory_20261107T225745Z/` |
| Freiburg a0, median absolute volume error | 30.4% (ICC 0.995) | `results/runs/freiburg_agreement_metrics_20261107T230622Z/` |
| Repeat-scan pairs across splits (screen) | 3 train–validation, 7 development–test, 5 development–calibration | `results/runs/leakage_repeat_pairs_20261107T230505Z/` |


## E2: StoneKit against the published Li 2022 baseline (11 November 2026, development only)

Source: `results/runs/li2022_comparable_metrics_20261111T033725Z/` (`scripts/li2022_comparable_metrics.py`). Our side is 32 fold-0 validation scans, one training run. Li's side is 5-fold CV over all 260 scans (including our locked test scans), mean ± SD of 10 trainings. **Not a like-for-like comparison.** The paper does not say how Dice is aggregated, so every reading is shown.

| Metric | Li 2022 (3D U-Net / Res U-Net) | StoneKit 100 ep | 100 ep + 30 mm filter | 250 ep | 250 ep + filter |
|---|---|---|---|---|---|
| Stone Dice, pooled voxels | 0.800 / 0.806 | **0.872** | **0.886** | 0.891 | 0.906 |
| Stone Dice, mean per scan with stones | (same numbers, if per-scan) | 0.694 | 0.733 | 0.707 | 0.765 |
| Stone voxel sensitivity, pooled | 0.802 / 0.797 | 0.837 | 0.837 | 0.861 | 0.861 |
| Per-stone Dice 0–6 mm (missed = 0; n = 90) | 0.580 / 0.601 | **0.143** | 0.143 | 0.211 | 0.211 |
| Per-stone Dice 6–20 mm (n = 61) | 0.816 / 0.761 | 0.581 | 0.581 | 0.593 | 0.593 |
| Per-stone Dice >20 mm (n = 36) | 0.829 / 0.834 | 0.829 | 0.829 | 0.849 | 0.849 |
| Kidney Dice (ours: TotalSegmentator against Li label 2, pooled) | 0.961 / 0.965 (trained on Li labels) | 0.869 | — | — | — |

**Reading:**
- On volume-weighted metrics (pooled Dice, voxel sensitivity) and on large stones, StoneKit matches or exceeds the published numbers.
- On small stones, per-stone Dice is far below Li's 0–6 mm figure (0.14–0.21 against 0.58–0.60). That is consistent with our detection-size boundary.
- The size of that gap depends on Li's unstated per-size Dice definition, for example whether undetected stones count as 0. It cannot be resolved without their code.
- The filter changes pooled and per-scan Dice but not per-stone Dice (it removes only false positives).
- The kidney gap is expected: TotalSegmentator is off the shelf, and Li's networks were trained on Li's own kidney labels and definition.

**Claim class:** Observed (development). A reviewer-facing statement can say the baseline is comparable on volume-dominated metrics and weaker on small stones, with the definitional caveat.


## rep1: Neubauer 2023 replication on the released Freiburg data (11 November 2026, run once)

- **Run folder:** `results/runs/rep1_neubauer_replication_20261111T074337Z/`.
- **Protocol:** `docs/FREIBURG_REPLICATION_PROTOCOL.md`, SHA 50f5c585…; config `configs/rep1_neubauer_replication.yaml`, status `approved`.
- **Run:** CPU only, 44 min 47 s, peak RSS 657 MB. 1,000 random 80/21 splits, seed 20261111, the same splits for all methods.
- **Verdict rule:**
  - *Reproduced*: the paper's mean lies within our 1,000-split SD.
  - *Close*: within 2 SD.
  - *Not reproduced*: otherwise.

### Primary configuration (pre-specified): P = 0.7 mm isotropic (paper-stated), 6-neighbour surface, 3D diameter

| Method | Quantity | Paper mean (SD) | rep1 mean (SD) | Verdict |
|---|---|---|---|---|
| M1 | P1 threshold (HU) | 272 (5) | 194.7 (8.6) | not reproduced |
| M1 | Volume ICC / diff (mL) / SD (mL) | 0.988 / −0.036 / 0.23 | 0.987 / −0.018 / 0.316 | reproduced / reproduced / reproduced |
| M1 | Diameter ICC / diff (mm) / SD (mm) | 0.994 / 0.011 / 0.79 | 0.997 / 0.181 / 0.550 | close / close / not reproduced |
| M2 | P1 (HU) / P2 | 194 (17) / 0.29 (0.05) | 194.6 (8.5) / 0.002 (0.013) | reproduced / not reproduced |
| M2 | Volume ICC / diff / SD | 0.990 / 0.014 / 0.23 | 0.987 / −0.019 / 0.317 | reproduced (all three) |
| M2 | Diameter ICC / diff / SD | 0.995 / −0.098 / 0.69 | 0.997 / 0.181 / 0.550 | reproduced / reproduced / close |
| M3 | P1 (fraction of median) | 0.48 (0.02) | 0.368 (0.027) | not reproduced |
| M3 | Volume ICC / diff / SD | 0.991 / −0.020 / 0.24 | 0.987 / −0.048 / 0.341 | reproduced (all three) |
| M3 | Diameter ICC / diff / SD | 0.995 / −0.137 / 0.67 | 0.997 / 0.240 / 0.510 | reproduced / reproduced / close |
| M4 | P1 / P2 | 0.43 (0.04) / 0.11 (0.08) | 0.368 (0.027) / 0.0001 (0.002) | not reproduced / not reproduced |
| M4 | Volume ICC / diff / SD | 0.989 / 0.004 / 0.26 | 0.987 / −0.048 / 0.341 | reproduced (all three) |
| M4 | Diameter ICC / diff / SD | 0.995 / −0.137 / 0.69 | 0.997 / 0.240 / 0.510 | reproduced / reproduced / not reproduced |

**Bartlett test across the four methods (all stones):**
- Volume: p = 0.540 (paper 0.55).
- Diameter: p = 0.882 (paper 0.87).
- Both reproduced.
- The paired bootstrap of the SD range (added analysis, because Bartlett ignores pairing): volume 95% interval [0.002, 0.082] mL; diameter [0.002, 0.096] mm.

**All stones, in-sample with averaged parameters (Figures 3–4 procedure; added error metrics):**
- Volume ICC: 0.993 (M1, M2) and 0.991 (M3, M4).
- Diameter ICC: 0.998.
- **Median absolute relative volume error: 25.0% (M1, M2) and 23.4% (M3, M4).**
- Volume bias: −0.014 mL (M1) and −0.045 mL (M3).
- Diameter bias: +0.18 mm (M1) and +0.23 mm (M3).

### Sensitivity configurations (pre-specified; all reported)

| Configuration | M1 P1 (paper 272) | M2 P1 / P2 (paper 194 / 0.29) | M3 P1 (paper 0.48) | M4 P1 / P2 (paper 0.43 / 0.11) |
|---|---|---|---|---|
| A: released `voxsiz` (0.7363 mm in-plane) | **263.3, reproduced** | 244.7 / 0.113, not reproduced | 0.585, not reproduced | 0.585 / 0.000, not reproduced |
| B: released `voxvol_mL` | 279.7, not reproduced | 251.2 / **0.233, reproduced** | 0.590, not reproduced | 0.589 / 0.000, not reproduced |
| P with axes Z1 or Z0 | Identical to P_Z2 | Identical | Identical | Identical |
| P with 26-neighbour surface | 194.7 | 194.6 / 0.001 | 0.368 | 0.368 / 0.000 |
| P with 2D axial diameter | 119.2 | 106.8 / 0.400, close | 0.229 | 0.200 / 0.263 |

### Interpretation (class: Reproduced / Observed)

1. **Agreement statistics are reproduced, but the optimised parameters are not, and the parameters depend on the voxel-spacing interpretation.**
   - The paper's M1 threshold (272 HU) is reproduced only with the released `voxsiz` field (A), not with the 0.7 mm spacing stated in the paper's text (P).
   - This suggests the paper's analysis used the released voxel-size field. That is an inference, not a verified fact.
   - The partial-volume factor P2 is near zero in most configurations. With a free threshold, the surface correction is largely redundant: the threshold absorbs the partial-volume effect. Under P and A, M2 reduces to M1 and M4 to M3.
   - M3/M4 factors were not reproduced under any configuration.
   - Candidate explanations (not tuned away):
     - the pre-isolated patches are masked at about 100 HU, so the "rough >20 HU" region is the whole stored patch (a deviation stated in the protocol);
     - the optimiser is not stated in the paper;
     - the units of the cost function are inferred.
2. **ICC near 0.99 coexists with a median per-stone volume error of about 23–25%,** in-sample and against water displacement. This replicates the a0 lesson with the paper's own methods: ICC is dominated by the between-stone size range (about 0.01–9 mL).
3. **The method comparison is robust to the interpretation choices:** Bartlett p is reproduced, and the four methods do not differ in error variance.

**Not reproducible (stated in advance):**
- inter-rater analysis and the human-versus-animal subgroups (data not released);
- region growing (markers not released).


## R2 and R3: Freiburg per-stone error decomposition, and 2D vs 3D against caliper (11 November 2026)

- **Protocol:** `docs/FREIBURG_R2_R3_PROTOCOL.md`, SHA 3d364074…, pre-specified after rep1 and before this run.
- **Run:** `results/runs/freiburg_r2_r3_20261111T074847Z/`. CPU, 68 s, 673 MB.
- **Reference:** physical (water displacement; mean caliper of three urologists).
- **Unit:** the stone (101). Donor clustering is unknown, so stones are treated as independent; this is a limitation.
- **Parameters:** rep1 parameters are in-sample means. Paper parameters are Table 1 as published.

### R3 primary result (spacing P, rep1 parameters, M1; 101 stones)

| Diameter definition | Bias (mm) | MAE (mm) | 95% LoA (mm) | Agreement with caliper category at 10 mm |
|---|---|---|---|---|
| 3D maximum + 0.7 mm (paper definition) | +0.19 | 0.46 | −0.87 to +1.25 | 99.0% |
| 2D axial maximum + 0.7 mm | **−1.72** | 1.82 | −5.67 to +2.22 | 92.1% |
| Volume-equivalent sphere diameter | −3.59 | 3.59 | −8.68 to +1.50 | 72.3% |

- **Paired absolute error, 2D − 3D:** median +0.82 mm, Wilcoxon p = 6.5 × 10⁻¹⁴.
- **10 mm category (McNemar):** 2D wrong / 3D right in 7 stones; 2D right / 3D wrong in 0; exact p = 0.016.
- **5 mm cut-off:** no stone has a caliper diameter under 5 mm, so it is untestable here.

**Robustness across all 16 configurations** (spacing P/A × rep1/paper parameters × M1–M4):
- 3D bias stays between −0.13 and +0.51 mm.
- 2D bias stays between −1.16 and −2.02 mm.
- 3D category agreement is 98–100%; 2D is 88–94%.
- The McNemar discordant counts favour 3D in every configuration: 3D-right/2D-wrong 5–11, against 2D-right/3D-wrong 0–1. Exact p ranges from 0.001 to 0.125.

**Limitation:** stone orientation in the phantom is arbitrary relative to the scanner axes, so these are not *in vivo* orientations.

### R2 primary result (P, rep1 parameters, M1): median absolute relative volume error, with 95% bootstrap CI

| Stratum | n | Median abs relative volume error |
|---|---|---|
| All | 101 | 23.6% [17.1, 38.4] |
| Caliper < 10 mm | 44 | 27.7% [19.2, 41.0] |
| 10–< 20 mm | 40 | 39.4% [21.4, 76.5] |
| ≥ 20 mm | 17 | 6.9% [2.8, 11.2] |
| HU tertile low / mid / high | 34 / 33 / 34 | 25.5% / 10.3% / 52.1% |
| Brushite | 17 | 63.4% |
| Calcium oxalate dihydrate | 6 | 55.7% |
| Cystine | 23 | 23.6% |
| Uric acid | 8 | 9.9% |
| Not recorded | 42 | 16.8% |
| Other (pooled) | 5 | 23.4% |

**Consistent across configurations:**
- Error ≥ 20 mm is 3–12%, against 22–46% below 20 mm.
- Uric acid is lowest (6–12%) and brushite highest (33–65%) in every configuration.
- The adaptive methods (M3/M4) reduce the high-HU error relative to M1. Under P with rep1 parameters, the high-HU tertile falls from 52% to 36%.

### Interpretation (class: Validated, physical reference, this phantom set only)
1. **3D maximum diameter agrees with caliper; 2D axial underestimates it.** The 3D definition changes the 10 mm category for about 7% of stones, all in its favour. This is physical-reference evidence. It is not evidence about patient outcomes.
2. **Per-stone volume error is large and depends on size and composition, even though ICC is about 0.99.** Composition and attenuation are a source of error that a single global agreement figure hides. This supports, but does not yet establish, a density/composition-aware error model as a candidate contribution (master plan v2 §G, novelty matrix).
3. **Data notes:**
   - The `StoneType` labels contain spelling variants ("Cystine" vs "cystine", "cacliumoxalate-monohydrate"). They were kept as recorded, per the protocol, so two single stones fall into "other".
   - Composition is not recorded for 42 stones.


## E4 (S3) and E11 (S7): 2D vs 3D categories and largest-stone vs burden on Li development annotations (11 November 2026)

- **Run:** `results/runs/e4_li_annotation_2d3d_20261111T083202Z/`. Script `scripts/e4_li_annotation_2d3d.py`; the analysis is pre-specified in its docstring before the first run.
- **Data:** all 165 development scans (132 with stones, 741 annotated pieces).
- **Reference:** Li annotation, not physical truth. No model output is used. Class: Observed (development).

**S3, category disagreement** (research bins < 5, 5–< 10, ≥ 10 mm; centre-to-centre maxima):
- The D2 category differs from the D3 category for **11.3% of pieces** (84/741; scan-cluster 95% CI 9.4–13.6%).
- Every disagreement puts D2 in the lower category, as the definitions imply:

| D3 category | Pieces moved to a lower D2 category |
|---|---|
| 5–<10 mm | 46 to < 5 mm |
| ≥ 10 mm | 36 to 5–<10 mm; 2 to < 5 mm |

- **Share that differs, by D3 bin:** < 3 mm 0%; 3–< 5 mm 0% (no lower bin exists); 5–< 10 mm 25.0%; ≥ 10 mm 14.0%.
- **Median relative difference (D3 − D2)/D3:** 0.10 at 3–5 mm, 0.11 at 5–10 mm and 0.18 at ≥ 10 mm.
- **Drivers** (GEE logistic, independence working correlation, 741 pieces in 132 scans):

| Covariate | Odds ratio (95% CI) | p |
|---|---|---|
| Elongation (minor/major, 1 = round) | 0.025 (0.003–0.20) | 0.0006 |
| Major-axis angle to z, per 10° | 0.77 (0.72–0.83) | 2 × 10⁻¹² |
| Slice thickness, per mm | 0.82 (0.56–1.20) | 0.30 |
| log2 volume | 0.95 (0.85–1.06) | 0.36 |
| Mean HU, per 100 | 0.98 (0.81–1.17) | 0.79 |

- **Reading:** disagreement is driven by shape and orientation. Elongated pieces, and pieces whose long axis lies along the scanner z axis (through the slices), disagree. Size, attenuation and slice thickness add nothing detectable once shape is included.
- **Consistency with Freiburg:** this agrees with the R3 result (2D axial underestimates the caliper maximum by 1.7 mm). That is consistent evidence, not a confirmation.

**S7, largest stone vs total burden** (132 scans with stones):
- Spearman ρ between largest-piece D3 and total annotated volume: **0.96**.
- Yet among the **298 scan pairs whose largest-piece D3 values lie within 1 mm**, **25.5%** differ more than 2-fold in total volume.
- 78.0% of stone-bearing scans have ≥ 2 annotated pieces. These are pieces of the annotation, not necessarily separate stones: touching stones merge and one stone can split.
- Median share of total volume outside the largest piece: 14.0% (IQR 0.7–44.0%).
- **Reading:** the largest diameter ranks patients' burden well overall, but for a quarter of near-equal largest-stone pairs it misrepresents burden by more than 2×. This is descriptive. No outcome claim is made.


### rep1 addendum: difference sign verified from the full text (11 November 2026)

- Neubauer et al. define the difference as **reference minus estimate**. This was verified by an AI agent reading the full text; the owner has not yet checked it.
- The rep1 protocol could not know this, so `compare_with_paper` accepted either sign.
- `scripts/rep1_sign_verified.py` re-classifies every difference cell with the verified sign only. It reads rep1's saved per-split values, with no rerun and no change to the rep1 folder. Output: `results/runs/rep1_sign_verified_20261111T091921Z/`.
- **Primary configuration (P, 6-neighbour surface, 3D):** no verdict changes. With the paper's sign, the volume differences are M1 +0.018, M2 +0.019, M3 +0.048 and M4 +0.048 mL, against the paper's −0.036, +0.014, −0.020 and +0.004.
- **Sensitivity configurations:** two cells change from reproduced to close:
  - A: M3 vol_diff;
  - B: M1 diam_diff.
- The verdicts quoted in the paper use the verified sign.


### E4/S7 recomputed after the Slicer review exclusions (11 November 2026)

- **Run:** `results/runs/e4_li_annotation_2d3d_20261111T134815Z/`. It supersedes `..._083202Z`, which included 8 development scans now excluded.
- **Cohort:** 157 development scans, 126 with stones, 705 pieces.

| Measure | Result (157 scans) | Earlier result (165 scans) |
|---|---|---|
| 2D/3D category differs | **11.1% (95% CI 9.1–13.3%)** | 11.3% |
| Elongation, OR | **0.020 (0.002–0.18)**, p = 0.0006 | — |
| Major-axis angle to z, OR per 10° | **0.76 (0.70–0.82)**, p = 3 × 10⁻¹² | — |
| Slice thickness / size / HU | Not significant (p = 0.49 / 0.48 / 0.78) | — |
| Spearman ρ, largest D3 vs total volume (S7) | **0.956** | — |
| Near-equal largest-stone pairs differing > 2× in total volume | **25.9% of 274 pairs** | — |
| Scans with ≥ 2 pieces | 77.8% | — |
| Median share of volume outside the largest piece | 15.1% | — |

**Category moves between D3 and D2:**
- 41 pieces go from 5–<10 to < 5 mm.
- 35 go from ≥ 10 to 5–<10 mm.
- 2 go from ≥ 10 to < 5 mm.

The conclusions are unchanged.

**Correction (11 November 2026, 14:00 UTC; owner direction):** the authoritative E4/S7 result is the 165-scan run `results/runs/e4_li_annotation_2d3d_20261111T083202Z` (11.3%; ρ 0.96; 25.5% of 298 pairs). The 157-scan rerun above is a sensitivity result pending owner decision D-REV, not a replacement.


## 12 November 2026: full out-of-fold design results (development only; 165/35/53 design; no calibration or test data)

**Inputs.** Post-training chain finished 08:23 UTC, 0 errors. OOF table `data/interim/oof_tables_20261112T082351Z/`: 165 development scans, 0 failures, no review exclusions (`exclude_csv: null`), CV split SHA 63e7c9a7…. Full-case nnU-Net validation Dice per fold: 0.601 / 0.680 / 0.694 / 0.596 (folds 0–3; fold 4 not tabulated here). Reference everywhere = Li annotation, not physical truth. These are DESIGN results that select methods; they are not validation.

**E6 per-stone measurement intervals** (`results/runs/e6_uncertainty_design_20261112T082354Z/`; 465 one-to-one matched stones in 122 scans; cross-conformal by fold). Pre-registered selection rule (amendment 5) applied by the script; point-coverage criterion re-checked by hand.

| Target, nominal | Qualifying | Selected | Selected marginal coverage (CP95) | Median relative width |
|---|---|---|---|---|
| Volume, 0.90 | U2, U3 | **U2 Mondrian** | 0.901 (0.870–0.927) | 1.44 (U3 1.64) |
| Volume, 0.80 | U2 | **U2 Mondrian** | 0.802 (0.763–0.837) | 1.04 |
| 3D max diameter, 0.90 | U2, U3, U4 | **U4 TTA-normalised** | 0.899 (0.868–0.925) | 0.446 (U2 0.491) |
| 3D max diameter, 0.80 | U2, U4 | **U4 TTA-normalised** | 0.794 (0.754–0.829) | 0.319 (U2 0.372) |

- Global conformal (U1) and Bland–Altman (U0) under-cover stones predicted < 5 mm (volume 0.90: U1 0.734, U0 0.782); Mondrian U2 restores it (0.879). Size-conditional conformal is established in the literature (Jaubert ISBI 2025); this is a replication of a known effect, not a novelty.
- **Limitation (not part of the selection rule):** coverage is uneven by HU tertile. U2 volume 0.90: low-HU 0.832 (CP lower 0.764). U4 diameter 0.90: high-HU 0.858; U4 diameter 0.80: high-HU 0.658 (lower 0.578). No method is conditionally calibrated across HU.
- Per-patient "all stones covered" at nominal 0.90: 0.79 (U2 volume), 0.74 (U4 diameter). Per-stone intervals are not a per-patient guarantee.
- Native slice ≤ 1 mm has n = 15 stones; that stratum is uninformative.

**E7 detection** (`results/runs/e7_detection_uncertainty_20261112T082355Z/`; 741 annotated pieces, 165 scans). Detected share 0.660 (scan-cluster CI 0.590–0.738). By annotated diameter: < 3 mm 0.181, 3–< 5 mm 0.519, 5–< 10 mm 0.761, ≥ 10 mm 0.934. By HU tertile: low 0.251, mid 0.729, high 1.000. GEE ORs: per volume doubling 1.40 (1.10–1.77), per 100 HU 4.10 (2.73–6.15). d50 at median HU (269): 6.3 mm³ (≈ 2.3 mm sphere-equivalent). False positives per scan: 1.31 unfiltered, 0.78 with the 30 mm filter; stone-free scans with any FP: 19/33 unfiltered, 12/33 filtered. Cross-fitted FP-probability model: AUROC 0.938, Brier 0.065 (constant 0.209). **Honest reading:** distance to kidney alone has AUROC 0.944, so the uncertainty signals (TTA spread 0.79, p_mean 0.77, threshold sensitivity 0.72) add no discrimination beyond distance. Size and HU are confounded (partial volume), so the HU effect is not causal. Many < 3 mm annotated pieces may be annotation fragments (relevant to D-REV).

**E8 burden intervals** (`results/runs/e8_burden_uncertainty_20261112T082357Z/`; 165 scans, 33 stone-free).

| Nominal | Method | Coverage (CP95) | Median width mm³ | Stone-free coverage |
|---|---|---|---|---|
| 0.90 | P2-0 naive | 0.782 (0.711–0.842) | 1151 | 0.64 |
| 0.90 | P2-B direct | 0.903 (0.847–0.944) | 12106 | 0.67 |
| 0.90 | P2-B-CQR | 0.927 (0.876–0.962) | 8702 | 0.79 |
| 0.90 | **P2-A decomposition** | 0.915 (0.862–0.953) | **1530** | **0.91** |
| 0.90 | P2-A without missed term | 0.879 (0.819–0.924) | 2002 | 0.97 |
| 0.80 | P2-0 naive | 0.764 (0.691–0.826) | 852 | 0.64 |
| 0.80 | P2-B direct | 0.788 (0.718–0.848) | 1877 | 0.64 |
| 0.80 | P2-B-CQR | 0.830 (0.764–0.884) | 1760 | 0.73 |
| 0.80 | **P2-A decomposition** | 0.836 (0.771–0.889) | **890** | **0.91** |
| 0.80 | P2-A without missed term | 0.812 (0.744–0.869) | 864 | 0.91 |

- Paired median width ratio P2-A / P2-B-CQR: 0.23 (0.90), 0.50 (0.80). **No CI is computed for this ratio** (the script computes a bootstrap CI only for A / P2-B direct: 0.146 (0.126–0.179) at 0.90; 0.534 (0.440–0.701) at 0.80). The protocol's P2 success needs a CI against the best direct comparator, so this is a gap to fill before the freeze.
- P2-A coverage is even across burden tertiles (0.91 / 0.91 / 0.93 at 0.90); the direct methods under-cover low-burden and stone-free scans.
- Missed-term ablation is mixed: at 0.90 the missed term narrows intervals (1530 vs 2002 mm³) and raises coverage (0.915 vs 0.879, CIs overlap); at 0.80 it changes little (890 vs 864). The evidence that the explicit missed-stone term matters is weak on these data.
- **CQR convergence warning checked (read-only diagnostic, scratchpad only):** at 0.90, fold 4's upper-quantile QuantReg stopped at 1,000 iterations. Refitting with 50,000 iterations converges and leaves coverage (0.9273) and median width (8701.9 mm³) unchanged (the replication matched the saved numbers exactly); individual intervals move by at most 288 mm³. The comparator is not weakened by the warning.

**P1 power simulation** (pre-specified, amendment 6; selected method U2, volume, nominal 0.90; `results/runs/p1_power_simulation_20261112T084506Z/`): 53 test scans → power 0.466 / 0.633 / 0.781 for δ = 0.05 / 0.075 / 0.10. **No δ reaches 80% power, so under the pre-specified rule P1 is underpowered and is reported as descriptive.** Sensitivity (`…_20261112T084508Z/`, 39 scans): the script resamples only scans with ≥ 1 matched stone (122/165 = 74% in development), so 53 test scans are expected to give about 39 such scans; power is then 0.43 / 0.58 / 0.69. The 53-scan figure is therefore optimistic.

**P2-A / P2-B-CQR width-ratio CI (12 November 2026, owner-approved; `results/runs/e8_burden_uncertainty_20261112T085235Z/`).**
- Same scan-level paired bootstrap as A / P2-B (10,000 resamples, seed 20261105). All other E8 numbers are identical to `…_082357Z`.
- Nominal 0.90: median 0.234, 95% CI 0.192–0.295.
- Nominal 0.80: median 0.500, 95% CI 0.447–0.651.
- The CI treats each scan's cross-fitted interval as fixed. It does not include re-calibration variability.
- Development design evidence, not validation. P1 is descriptive by owner decision (docs/decisions.md, 12 November).


## 12 November 2026, 10:10 UTC: work that D-REV does not block (development only; no calibration or test data)

**Measurement regression check: PASSED** (`results/runs/measurement_regression_20261112T100054Z/`)
- **What was checked:** today's measurement code re-measured the 6 November fold-0 run's saved label maps.
- **Result:** 32 development scans, 154 stones. Volume, 3D and axial maximum diameter, mean HU and voxel count are identical to within 1e-6 relative, with 0 mismatches and 0 unmatched stones.
- **Meaning:** the shape descriptors added since 6 November did not change the frozen measurements.

**T1 interval engine:** `src/stonekit/uq/confirmatory.py`, tested by `tests/test_confirmatory.py` (13 tests passed).
- **Per-stone methods (U0, U1, U2, U4, U5):** identical to the E6 code; a test asserts exact equality.
- **Burden methods (P2-0, P2-B, P2-B-CQR, P2-A):** follow E8 and amendment 2.
- **Calibration split:**
  - fitted on the development scans: the P2-A components (false-positive model, residual pools, missed-burden pool) and the CQR quantile regression;
  - set on the calibration scans only: every conformal quantile or factor;
  - test scans are only evaluated (a test shows that test annotations cannot change any test interval).
- **Formal P2 test:** OD-1 is implemented in `p2_test`.
- **Check against E8, with fit = calibration = other folds:**
  - reproduces P2-0 (0.782; 1,150.8 mm³) and P2-B (0.903; 12,105.6 mm³) exactly;
  - P2-A is close (0.903; 1,427 mm³, against 0.915; 1,530 mm³; a random-draw difference).

**Finding: E8 overstated P2-A's advantage over P2-B-CQR.**
- **The cause:** E8 fitted the CQR comparator on only half of the other folds and calibrated it on the other half, giving a median width of 8,702 mm³.
- **Fitted on more development data, as at T1,** the comparator is much narrower: about 2,300–5,300 mm³ depending on the split.
- **Consequences:**
  - The development ratio of 0.234 (CI 0.192–0.295) **must not be quoted as the expected effect.**
  - P2-A still beats the best direct comparator on development, so the pre-specified selection of P2-A as the Final burden method stands. Its advantage is smaller and less certain.

**T1 dry run on development scans** (`results/runs/t1_dry_run_development_20261112T100623Z/`; label DEVELOPMENT_DRY_RUN_NOT_A_RESULT)
- **Design:** 200 seeded repetitions. Each repetition splits the 165 development scans into 35 pseudo-calibration, 53 pseudo-test and 77 fit scans. The fit set is conservative: 165 scans will be available at T1.
- **Per-stone results:** median (5th–95th percentile) over repetitions, about 150 matched stones per pseudo-test set.

| Arm, method | Coverage at 0.90 | < 5 mm stratum | Median relative width |
|---|---|---|---|
| Baseline U0, volume | 0.921 (0.851–0.960) | 0.778 (0.615–0.890) | 1.92 |
| Baseline U1, volume | 0.914 (0.820–0.967) | 0.750 (0.533–0.907) | 1.91 |
| **Final U2, volume** | 0.912 (0.791–0.969) | **0.927** (0.727–1.000) | 1.62 |
| **Final U4, 3D diameter** | 0.907 (0.830–0.974) | 0.884 (0.804–0.956) | 0.45 |

- **Burden results** (P2, nominal 0.90, filter on):

| Method | Coverage | Median width mm³ | Stone-free coverage |
|---|---|---|---|
| P2-0 naive | 0.792 (0.698–0.868) | 1,269 | 0.636 |
| P2-B direct | 0.925 (0.811–0.981) | 23,840 | 0.700 |
| P2-B-CQR | 0.925 (0.811–1.000) | 5,289 | 0.818 |
| **P2-A** | 0.925 (0.849–0.981) | **2,210** | **0.920** |

- **OD-1 formal P2 test:**
  - median width ratio P2-A / P2-B-CQR is 0.644 (0.088–1.729);
  - **P(success) = 0.64**;
  - P2-A's coverage check passes in 99.5% of repetitions.

  The expected power of the single formal P2 test is therefore about two-thirds. A negative T1 result is plausible and would be reported as such.
- **Spread of per-stone coverage:** U2 volume coverage on 53 test scans ranges 0.79–0.97 (5th–95th percentile), which supports P1 being descriptive.

**Filter ablation, OD-3 (secondary; development dry run):**
- **Without the 30 mm filter:**
  - P2-0 coverage falls from 0.792 to 0.717, and its stone-free coverage from 0.636 to 0.444;
  - P2-B direct width rises from 23,840 to 78,522 mm³;
  - P2-A is nearly unchanged (0.925; 2,402 mm³), because its false-positive term absorbs the extra pieces.
- E7 already showed false positives per scan of 1.31 without the filter and 0.78 with it.
- **Conclusion:** the filter mainly protects the naive and direct burden methods.

**Caveat:** all of this uses the 165 development scans, including the 8 scans that D-REV may make ineligible. It is cheap to rerun once D-REV is resolved.

**Reproducibility and provenance audit (12 November 2026, read-only; results/runs/provenance_audit_20261112T101342Z/):**
- **Provenance in place:** all 50 run folders cited in the results, decisions, experiments and paper exist and hold result files. All 42 cited scripts exist; their SHA-256 values are listed in the summary.
- **Environment:** environment.lock.yml matches the installed versions (nnunetv2 2.8.1, TotalSegmentator 2.18.0, torch 2.15.0+cu132, numpy 2.4.6, statsmodels 0.15.0, scikit-learn 1.9.1, pandas 3.1.0). The split SHA matches the log.
- **Gap: 40 of the 42 cited scripts and 12 of the 76 src files are untracked in git** (HEAD c54d867; 215 uncommitted paths). No result can be reproduced from version control until the owner-approved first commit. That commit is a precondition in protocol v3 §0.4.

**Correction (OD-12, 12 November 2026): the E6 per-stone selection.**
- **The rule:** the original pre-specified rule (master plan v2 §G, CP lower bound ≥ nominal − 0.05 in every predicted-size band) applies. Amendment 5 (the upper-bound rule) was a post-hoc proposal and is not adopted.
- **The outcome**, from the saved E6 statistics with no rerun: **no method qualified** for volume or 3D diameter, at 0.90 or 0.80, so **U2 is carried for both**.
- **What it supersedes:** the earlier statements above ("Selected … U4 TTA-normalised for 3D diameter"; the E6 table rows marked as "Selected U4"). U4 for diameter is an exploratory comparator only.
- **Values at 0.90 (development):**
  - U2 diameter: coverage 0.903 (CP 0.873–0.929), median relative width 0.491;
  - U4: coverage 0.899, width 0.446.
- **The dry run's diameter row (U4)** is an exploratory comparator. The dry run is not rerun.


## 12 November 2026, 10:45 UTC: secondary and ablation analyses N-A, N-B, N-C (OD-13; development only; rules fixed before running)

**N-A, robustness: Mondrian by predicted size × HU tertile** (`results/runs/secondary_na_nb_development_20261112T103848Z/`; same cross-conformal design as E6; 465 stones, 122 scans)
- **Pre-specified rule NOT met** at any target or level. Volume at 0.90:
  - marginal coverage 0.905 against 0.901 for U2;
  - median relative width 1.519 against 1.441 (+5%);
  - the worst size × HU cell falls from 0.727 (n = 11) to 0.333 (n = 3).
- **Reading:** the worst-cell criterion is dominated by a 3-stone cell (small, high-HU stones). It is unstable at this size, but the rule was fixed in advance and is **not changed**.
- **Descriptively,** N-A improves the low-HU cells:
  - 5–10 mm, low HU: 0.796 → 0.898;
  - ≥ 10 mm, low HU: 0.727 → 0.909;
  - < 5 mm, low HU: 0.863 → 0.874.
- **Classification:** robustness analysis with a negative pre-specified outcome. U2 remains the final method.

**N-B, secondary: patient-level joint coverage (max-score)** (same run)
- **Pre-specified rule MET** at every target and level.
- **Volume at 0.90:** all matched stones of a patient were covered for **0.934** of patients (CP 0.875–0.971), against 0.795 for per-stone U2, with intervals **1.76×** wider (median relative width).
- **Diameter at 0.90:** 0.918 (0.854–0.960) against 0.787, at 1.77× wider.
- **At 0.80:** 0.787 (volume) and 0.779 (diameter), at 1.34–1.36× wider.
- **Reading:** a patient-level "all stones covered" guarantee costs about 1.8× interval width at the 90% level. This is an established mechanism (Andéol et al. 2025; max-score conformal); it is reported as an application, never as a new method.

**N-C, ablation: distance-only false-positive model in P2-A** (`results/runs/ablation_nc_development_20261112T104213Z/`; the same 200 splits as the T1 dry run)
- **Coverage:** unchanged (median 0.925 for both).
- **Width:** the distance-only P2-A is **11% wider** (median relative change +0.111; 5th–95th percentile −0.14 to +2.67), with median widths of 2,798 against 2,210 mm³.
- **Pre-specified ±5% equivalence NOT met.** The full false-positive model stays.
- **Reading:** although the extra features add little discrimination (AUROC 0.938 vs 0.944 for distance alone), they make P2-A's intervals narrower. The likely reason is better-calibrated probabilities for ambiguous pieces; this is not tested.


## 12 November 2026, 11:05 UTC: failure analysis of the final-arm intervals (development only; `results/runs/failure_analysis_development_20261112T110426Z/`)

**U2 per-stone volume interval, nominal 0.90:** 46 misses among 465 matched stones.

**Direction: 45 of 46 misses are UNDER-estimates.** The annotated volume lies above the upper bound, meaning the model under-segments those stones. Only one miss is an over-estimate. The median |log error| of a miss is 0.98, about 2.7-fold, with a maximum of 2.62.

**Miss rate by subgroup** (Clopper–Pearson 95% CI):

| Subgroup | Miss rate (CI) |
|---|---|
| HU tertile, low | **16.8%** (11.3–23.6) |
| HU tertile, mid | 7.7% |
| HU tertile, high | 5.2% |
| Elongation tertile, low (compact stones) | **14.8%** (9.6–21.4) |
| Elongation tertile, mid | 7.7% |
| Elongation tertile, high | 7.1% |
| Predicted size < 5 mm | 12.1% |
| Predicted size 5–10 mm | 9.2% |
| Predicted size ≥ 10 mm | 9.0% |
| Slice ≤ 1 mm (n = 15; uninformative) | 20% |
| Slice 1–2 mm | 11.4% |
| Slice > 2 mm | 6.5% |
| Fold 1 | 5.5% |
| Fold 3 | 15.6% (the fold range) |

**Reading:**
- The error distribution is strongly one-sided (under-segmentation of faint, compact stones).
- U2's symmetric score |log(V/V̂)| therefore spends interval width on the side where errors almost never occur.
- An **asymmetric (signed-error) split conformal interval** is an established alternative (separate lower and upper quantiles).
- U2 remains the pre-specified final method (OD-12), and is **not** changed post hoc.
- **Status:** an asymmetric-U2 variant is a scientifically justified **exploratory comparator and limitation**, not a novelty and not a final-arm change.

**Patient-burden drivers (the same 165 scans):**
- **Missed stones:** **1.9%** of total annotated burden; median 0% per scan; 90th percentile 17.5%.
- **False positives:** the median false-positive share of predicted volume after the filter is 0%, but 12 of 33 stone-free scans keep false-positive volume.
- **Reading:** this agrees with Kataria 2026 (missed lesions add little volume). Burden-interval width and failures are driven mainly by measurement error and false positives on a minority of scans, not by missed stones. This is consistent with the mixed missed-term ablation in E8.


## 12 November 2026, 12:50 UTC: EXPLORATORY asymmetric U2 (development only; `results/runs/exploratory_asymmetric_u2_20261112T124900Z/`)

- **Motivation:** the failure analysis showed misses are one-sided (under-segmentation).
- **Method:** an established two-tailed split conformal interval per size band (α/2 per tail, signed log residual). It was designed **after** seeing development results, so it is exploratory and **not pre-specified**; U2 stays the final method (OD-12).

| Target, nominal | U2 coverage / median relative width / misses above–below | Asymmetric coverage / width / misses |
|---|---|---|
| Volume 0.90 | 0.901 / 1.441 / 45–1 | 0.912 / **1.337 (−7%)** / 20–21 |
| Volume 0.80 | 0.802 / 1.041 / 91–1 | 0.804 / **0.863 (−17%)** / 47–44 |
| Diameter 0.90 | 0.903 / 0.491 / 40–5 | 0.905 / **0.341 (−31%)** / 25–19 |
| Diameter 0.80 | 0.806 / 0.372 / 84–6 | 0.819 / **0.243 (−35%)** / 46–38 |

**Reading:**
- On development, the asymmetric interval keeps coverage and balances the misses. It is narrower, markedly so for diameter.
- Low-HU volume coverage is not improved (0.832 for both at 0.90): the HU-conditional weakness remains.
- Because it was chosen after viewing development results, it is **not** adopted as the final method; that would repeat the amendment-5 provenance problem.
- It is added to protocol v3 as a **secondary exploratory comparator at T1**. T1 data are untouched, so that comparison is clean, but any T1 advantage is reported as exploratory.


## D-REV technical evidence (13 November 2026; development scans only; not an adjudication)
- **Sources:** results/runs/drev_technical_audit_20261113T042306Z; drev_audit_supplement_20261113T044112Z; drev_component_trace_20261113T044621Z; drev_pair_check_20261113T044809Z; drev_unflagged_trace_20261113T045535Z. Report: docs/DREV_TECHNICAL_EVIDENCE_REPORT.md.
- **Reference (153 unflagged development scans):** aorta median HU median 42, p95 47.4, max 49; parenchyma median 31, p95 36, max 41. The 12 flagged development scans: aorta 33-45, parenchyma 24-41.
- **Long isolated tube (8 locations / 7 scans):** volume 3.2-10.0 mL; craniocaudal extent 207-300 mm; median 287-536 HU; max 1,023-2,068 HU; not skeleton-connected. The model has a 0.6-1.9 mL piece at 6 of 8.
- **Skeleton-connected (wd0131, 2 locations):** on the 1,249 mL largest dense component; model pieces 1.5 and 1.9 mL.
- **Large isolated masses:** wd0113, 62.4 mL (annotated 73.0 mL, predicted 12.6 mL); wd0239, 77.2 mL (annotated 84.1 mL, predicted 38.8 mL; its TS kidney on that side is 25 mL).
- **Small foci:** wd0215, 0.53 mL, 19.4 mm outside the kidney (model agrees); wd0144, 0.03 mL, model none. In wd0144 the annotation kidney label is on the left only, and the stone is on the right.
- **Unflagged wd0252:** unlabelled 72.4 mL mass, 77% in the TS right kidney (59 mL); model 63.9 mL. Not a duplicate of wd0239 (opposite side, different geometry).
- **Burden-gap concentration:** wd0252, wd0113 and wd0239 account for 20.3%, 19.2% and 14.4% of the summed absolute predicted-vs-annotated burden gap (314,300 mm3) over 165 development scans; 53.9% together.
- **Other unflagged scans with annotated burden 0 and predicted pieces (9):** no piece inside the TS kidney. 6 have pieces on the skeleton, 3 have isolated foci outside the kidney.

## MSWAL label characterisation (13 November 2026; labels and dataset.json only, no image; OD-21 step 1)
- **Run:** results/runs/mswal_label_characterisation_20261113T092648Z/ (scripts/mswal_fetch_labels.py, scripts/mswal_characterise_labels.py). Download: 485 files, 71.6 MB, with a per-file SHA-256 in data/external/mswal/download_log.json.
- **Licence:** CC BY-NC 4.0, stated in dataset.json and on the Hugging Face card. Non-commercial research use and publication with attribution are permitted.
- **Released training cases:** 484, of which **77 contain kidney-stone voxels (label 2)**.
- **Stone components (26-connected):** 300 in total. By sphere-equivalent diameter: < 3 mm 89; 3–< 5 mm 113; 5–< 10 mm 77; ≥ 10 mm 21.
- **Slice thickness in the stone cases:** 0.7–1.5 mm (median 1.25 mm).
- **Contrast phase:** **no per-case phase metadata.** Per protocol v3 §5.4, eligibility is decided by the pipeline's aorta check on the image. The dataset paper reports 38% non-contrast overall, but the phase mix of the stone cases is unknown until the images are processed.
- **Minimum image subset for §5.4:** the 77 stone-case images, 6.65 GB (all images: 34.87 GB). The list is in stone_case_image_subset.json.

## T1 CONFIRMATORY RESULT (13 November 2026; protocol v3 frozen, SHA 70737f5e...; run ONCE; reserved scans)
- **Run:** results/runs/T1_20261113T081425Z (commit f3e8b359; all 8 gates passed, listed in inputs.json). Cohort: fit 165 development, calibration 35, test 53.
  - calibration_quantities.json SHA-256 58c023ae0e68fb61bff2bc526631981b80dbeb0a8316bac43393ef2517b5cdff, hashed before the test load;
  - evaluation.json SHA-256 d8cf65da449a8bf8c65034b4e730c857945b904857d78ad4cc42697547468ed9.
- **FORMAL P2 (OD-1; nominal 0.90; 30 mm filter on): PASSED.**
  - Paired median width ratio P2-A / P2-B-CQR: **0.573, 95% bootstrap CI 0.540–0.656** (below 1).
  - P2-A coverage **50/53 = 0.943, Clopper–Pearson 95% 0.843–0.988** (upper bound ≥ 0.90).
- **Burden intervals at 0.90, filter on (53 test scans):**

  | Method | Coverage | CP 95% | Median width (mm³) | Stone-free coverage (11 scans) |
  |---|---|---|---|---|
  | P2-0 naive | 0.679 | 0.537–0.801 | 1,567 | 0.636 |
  | P2-B direct | 0.906 | 0.793–0.969 | 30,369 | 0.727 |
  | P2-B-CQR | 0.925 | 0.818–0.979 | 7,431 | 0.818 |
  | **P2-A decomposition** | **0.943** | 0.843–0.988 | **3,090** | 0.909 |

  - Paired coverage difference P2-A minus P2-B-CQR: +0.019 (95% CI −0.057 to +0.094); no evidence of a coverage difference.
- **Secondary arms (reported; not the formal test):**
  - **0.90, filter off:** ratio 0.951 (CI 0.893–1.194); the formal criteria are **not met**. The width advantage of P2-A depends on the pre-specified 30 mm filter.
  - **0.80, filter on:** ratio 0.430 (CI 0.373–0.792). Both P2-A and P2-B-CQR covered 0.830 (CP 0.702–0.919).
- **Per-stone, 0.90 (147 matched test stones in 35 scans; P1 is descriptive):**

  | Target | Method | Coverage (CP 95%) | Median relative width | < 5 mm (n 31) | 5–< 10 mm (n 43) | ≥ 10 mm (n 73) |
  |---|---|---|---|---|---|---|
  | Volume | U0 Bland–Altman | 0.980 (0.942–0.996) | 3.80 | 0.935 | 1.000 | 0.986 |
  | Volume | U1 global conformal | 0.952 (0.904–0.981) | 2.67 | 0.806 | 1.000 | 0.986 |
  | Volume | **U2 size-Mondrian (final)** | **0.966 (0.922–0.989)** | **2.32** | 0.935 | 0.977 | 0.973 |
  | Volume | U4 (exploratory) | 0.952 (0.904–0.981) | 1.58 | 0.968 | 0.977 | 0.932 |
  | Volume | U2-asymmetric (exploratory) | 0.912 (0.854–0.952) | 23.89 | 1.000 | 0.860 | 0.904 |
  | Diameter 3D | U0 | 0.939 (0.887–0.972) | 1.27 | 0.774 | 1.000 | 0.973 |
  | Diameter 3D | U1 | 0.898 (0.837–0.942) | 1.10 | **0.613** | 1.000 | 0.959 |
  | Diameter 3D | **U2 (final)** | **0.952 (0.904–0.981)** | **1.06** | **0.935** | 1.000 | 0.932 |
  | Diameter 3D | U4 (exploratory) | 0.939 (0.887–0.972) | 0.56 | 0.935 | 1.000 | 0.904 |
  | Diameter 3D | U2-asymmetric (exploratory) | 0.966 (0.922–0.989) | 2.32 | 1.000 | 1.000 | 0.932 |

  - Relative width is a ratio of the predicted size, as defined in the protocol (exp(hi) − exp(lo)).
  - For small stones (< 5 mm), the global conformal diameter interval U1 under-covered (19/31). The size-Mondrian U2 restored coverage (29/31), with a narrower median width than U1. The per-band n is small (31 stones in 16 scans), and the CP intervals are wide.
  - U2-asymmetric volume intervals are very wide (median 23.9) and under-cover in 5–< 10 mm (0.860); the exploratory concern from development is confirmed.
- **Patient-level joint (nb) coverage at 0.90 (35 scans with matched stones):** volume 1.00 (CP 0.900–1.00; median relative width 44.5); diameter 0.971 (CP 0.851–0.999; 4.21). These widths are too wide to be informative.
- **Pre-specified sensitivity and fragility analyses (§5.1; estimates only, not formal tests):**

  | Cohort | Removed fit / cal / test | Test scans | Ratio (CI) | P2-A coverage (CP 95%) | Formal criteria met |
  |---|---|---|---|---|---|
  | Primary | 0 / 0 / 0 | 53 | 0.573 (0.540–0.656) | 0.943 (0.843–0.988) | yes |
  | S1 screening exclusions | 8 / 0 / 3 | 50 | 0.622 (0.596–0.684) | 0.900 (0.782–0.967) | yes |
  | S2 all screened flags | 11 / 1 / 7 | 46 | 0.464 (0.427–0.557) | 0.935 (0.821–0.986) | yes |
  | S3 cross-split repeat pairs | 0 / 6 / 6 | 47 | 0.588 (0.543–0.656) | 0.936 (0.825–0.987) | yes |
  | Item 5 reader-confirmed errors | — | — | — | — | not performed (OD-19) |

  - **The development fragility did not reproduce at T1.** On a development stand-in split, removing flagged scans (S1/S2) made the formal criteria fail. On the reserved data, all three cohorts met both criteria.
  - Under S1, P2-A point coverage fell to 0.900 (45/50), exactly at nominal, and below P2-B-CQR (0.940). The paired difference was −0.04 (CI −0.12 to +0.04). The coverage margin is therefore thin.
  - **Filter-off, S2:** ratio 1.091 (CI 1.005–1.328). Without the filter, P2-A is wider than P2-B-CQR in this cohort.
- **Not established:**
  - Item 5 (reader-confirmed annotation errors) was not performed.
  - The reference standard is the dataset annotation, not adjudicated by a radiology-trained reader (OD-19).
  - The test set has 53 scans from one dataset, and the CP intervals reflect that.
  - Research software only; nothing here is a diagnostic claim.

## MSWAL EXTERNAL CHECK (13 November 2026; protocol v3 §5.4; secondary and exploratory; run once)
- **Run:** results/runs/mswal_external_check_20261113T132418Z. external_check.json SHA-256 6b57e1f666ad8a187b7cdfafe41342921b6ed7de63d97be7e2e50a1d398d3966.
  - The frozen system was applied unchanged.
  - Intervals were rebuilt from the hashed T1 calibration (58c023ae...). Nothing was fitted on MSWAL.
- **Code fix after processing, before any endpoint was computed:**
  - All 77 cases were processed, and the tables were saved.
  - The summary step then crashed on a type error (`.to_numpy()` on a numpy array).
  - The fix: one line, plus a `--summarise-only` option that reads the saved tables. No case was reprocessed (docs/decisions.md).
- **Eligibility:** 77 stone cases → **58 eligible** (aorta median HU 36.8 mean, max 97) and 19 contrast-enhanced (aorta median 172). No processing failures.
- **Detection (217 annotated stones, 58 scans):** detected 37.8% overall.

  | Size | Detected |
  |---|---|
  | < 3 mm | 2/60 (3%) |
  | 3–< 5 mm | 20/86 (23%) |
  | 5–< 10 mm | 42/52 (81%) |
  | ≥ 10 mm | 18/19 (95%) |

  - False positives per scan after the filter: 0.79.
- **Per-stone, 0.90 (75 matched stones in 43 scans):**

  | Target | U0 | U1 | U2 (final) | U2-asymmetric |
  |---|---|---|---|---|
  | Volume coverage (CP 95%) | 0.707 (0.590–0.806) | 0.520 (0.402–0.637) | **0.733 (0.619–0.829)** | 0.987 (0.928–1.000) |
  | Volume median rel. width | 3.80 | 2.67 | 6.69 | 43.5 |
  | Diameter coverage | 0.693 (0.576–0.795) | 0.587 (0.467–0.699) | **0.893 (0.801–0.953)** | 0.960 (0.888–0.992) |
  | Diameter median rel. width | 1.27 | 1.10 | 3.04 | 3.48 |

  - **Per-stone volume intervals under-cover on MSWAL.** The calibration does not transfer for volume.
  - Diameter U2 stays near nominal.
  - The U2 widths are larger than at T1 because more matched MSWAL stones are predicted below 5 mm: 38/75 (51%), against 31/147 (21%) at T1. The < 5 mm stratum has the widest intervals.
- **Burden, 0.90, filter on (58 scans; all stone-bearing):**

  | Method | Coverage (CP 95%) | Median width (mm³) |
  |---|---|---|
  | P2-0 naive | 0.448 (0.317–0.585) | 341 |
  | P2-B direct | 0.897 (0.788–0.961) | 2,947 |
  | P2-B-CQR | 0.879 (0.767–0.950) | 862 |
  | **P2-A** | **0.966 (0.881–0.996)** | **896** |

  - Width ratio P2-A/P2-B-CQR: 1.142 (95% CI 0.817–1.444). **The T1 width advantage did not transfer;** the widths are similar.
  - P2-A coverage held (56/58). Paired coverage difference +0.086 (CI 0.000–0.172).
  - Descriptive only; the OD-1 formal test applies to T1 alone.
- **Interpretation limits:**
  - MSWAL has different annotation conventions, scanners and a smaller stone-size mix.
  - Annotation differences and genuine model error cannot be separated here.
  - The low small-stone detection may partly reflect labelling of very small stones.

## 14 November 2026: T1 post-hoc width replay (review item 2e), a post-hoc width replay of the locked test (owner approved: "Yes, do 2e")

**Label:** POST HOC and DESCRIPTIVE; not part of frozen protocol v3. T1 was NOT rerun: no segmentation, TTA, refit or recalibration. No annotation mask was opened, and the guarded loader was not used.

**Method:**
- `scripts/t1_posthoc_width_replay.py` (SHA-256 53ccdeb49d330eb7391450f7bad189fa59e43475cd2faf9971c3a1800f90f839) rebuilds each test scan's predicted pieces. It uses the stored T1 run-case outputs (model mask, filter-removed mask, kidney mask and CT ROI crop) and the stored 8-flip probabilities, with the same rules as `build_oof_stone_table.py`.
- Every burden method is rebuilt from the hashed calibration quantities through `stonekit.t1.external.rebuild_burden`, with the development pools checked against their hashes.
- Stone-free status comes from the pre-split audit `data/interim/li2022_full_audit/li_full_cases.csv` (5 November 2026, written before the split and the lock). It gives 11 stone-free test scans, matching the T1 run.

**Reproduction check (acceptance condition):** in all four configurations (0.90 and 0.80, filter on and off), the replay reproduced the stored median widths of P2-0, P2-B, P2-B-CQR and P2-A, and the paired median width ratio with its bootstrap CI (same seed).

**Result (nominal 0.90, filter on):**
- **Stone-bearing scans (n = 42):** ratio 0.554 (bootstrap 95% CI 0.535–0.609). The detection-aware interval was narrower in 36 of 42 scans; median widths 8,209 against 16,293 mm³.
- **Stone-free scans (n = 11):** the detection-aware interval was wider in 8 of 11 scans; median widths 710 against 32 mm³, median ratio 19.3. It always adds a missed-burden draw.
- **Reading:** the 0.573 advantage comes from stone-bearing scans, and stone-free scans work against it. Stone-free coverage (stored) is 10/11 against 9/11 for the comparator.
- **Other configurations:**
  - 0.90 without the filter: stone-bearing ratio 0.919 (0.879–1.036);
  - 0.80 with the filter: 0.391 (0.333–0.462);
  - 0.80 without the filter: 0.445 (0.402–0.601).

**Output:** `results/runs/t1_posthoc_width_replay_20261114T081724Z/` (`posthoc_width_replay.json`, SHA-256 278eca1d602c5d2055e3c1880cf34d9d5d44bb7bf73063ae0a530027a850084a, plus per-scan width CSVs).

**Paper:**
- one Results sentence (Section IV-B), labelled post hoc;
- the Discussion notes the wider stone-free intervals;
- to keep exactly 6 pages, the Limitations sentence on stored aggregates was removed (the replay explains it), and the Conclusion and two other sentences were shortened.

**Build:** 6 pages, with 0 errors, 0 overfull boxes and 0 BibTeX warnings. Nothing committed.
