# Stage 2 protocol: replication against the Freiburg reference

**Status: approved by the owner on 2 November 2026, including all six points in section 14.** This protocol is fixed before the experiment runs. Any later change needs the owner's approval and must be logged.

G4 stays INSUFFICIENT EVIDENCE. Paper 1 stays provisional. Stage 3, the interval analysis, MSWAL, MIMIC, model training and Git commits are out of scope.

## 1. Purpose and interpretation
Stage 2 tests whether the plan's threshold measurement agrees with the supplied Freiburg reference, and evaluates G3.

**This is a replication, not independent validation.** The authors derived 272 HU from these same stones, so agreement at 272 HU cannot show performance on independent data.

## 2. Data
- **Stones:** all 101 from the unchanged D1 files.
- **Reference:** `gt_vol` (mL) and `gt_diam` (mm) from the MATLAB file, unrounded. These are single values, with no per-rater data.
- **Stone link:** `mat_index` and `stone_id` from `freiburg_stones.parquet`. This inferred link is G2's recorded caveat.

## 3. Primary threshold
**272 HU, locked before the experiment.** It will not change based on Freiburg results.

## 4. Segmentation and hole filling
Applied the same way to every stone and threshold, in this order:
1. A voxel is included if its value is greater than the threshold.
2. Connected components use 26-connectivity (Section 11.2).
3. Components under 3 voxels are dropped (Section 11.2 default).
4. Each remaining component has internal holes filled in 3D, using `scipy.ndimage.binary_fill_holes` with its default structure. The number of stones and voxels affected is reported.
5. **The stone mask is the union of all remaining components,** because each patch holds one stone. IDs 74, 94 and 100 have more than one component at 272 HU, and this is reported.

An empty mask stays in the analysis with volume 0 and both diameters 0, and is flagged. At 272 HU this should not occur, because every patch peaks at 408 HU or higher.

## 5. Voxel spacing assumptions and coordinates
Both assumptions use the identical procedure, and neither is treated as correct. Source values are not altered.

| | Spacing (mm), array axes 0, 1, 2 | Voxel volume (mL) |
|---|---|---|
| **A** | `voxsiz`: 0.7363, 0.7363, 0.7 | Product of `voxsiz` ÷ 1000 |
| **B** | s, s, 0.7, where s = √(`voxvol_mL` × 1000 ÷ 0.7) per stone | Stored `voxvol_mL` |

**Physical coordinates:** a voxel at array index (i, j, k) is at (i × s0, j × s1, k × s2) in mm. Distances are in mm. Volumes convert by 1 mL = 1000 mm³.

**Axis order: not established.** Nothing in the files or paper says which array axis is the slice direction.

| Status | Evidence |
|---|---|
| **Confirmed** | The MATLAB file holds only `stones`, with the fields `gt_diam`, `gt_vol`, `imgdata`, `voxsiz` and `voxvol_mL`. It has no orientation, affine or header field |
| **Confirmed** | `imgdata` is 3D, and `scipy.io.loadmat` keeps MATLAB's index order (fixture check) |
| **Confirmed** | figshare lists `voxsiz` as "x y z", and the paper says images were reconstructed axially. Neither maps these to array axes |
| **Inferred, for axis 2 as z** | `voxsiz` lists its different value third, which would match the third axis if the order carries over. The 9 non-cubic patches are shorter only on axis 2, which could reflect a scan end |
| **Inferred, against axis 2 as z** | **Axis 1 behaves differently from axes 0 and 2.** Inside stones, the median difference between neighbouring voxels (relative to mean HU) is 0.184 on axis 1, against 0.116 and 0.111. Neighbour correlation is 0.47, against 0.76 and 0.75. Near-round stones show the same pattern (n = 32), so stone shape does not explain it. A slice axis would usually be the odd one out |
| **Uninterpretable** | Stones are smallest along axis 1 in 67 of 101 patches. This could reflect stones resting flat, and gravity would act along an in-plane axis in a usual setup, which conflicts with the point above |

The evidence conflicts, so **axis 1 may be the slice direction.** These checks used stored image values only, with no reference values.

**What this affects**
- **Volume: not affected.** The product of the three spacings is the same in any order, under both assumptions. Segmentation works on array indices, so it is not affected either.
- **Diameter: affected, slightly.** Only the axis given the 0.7 mm spacing changes. The effect cannot exceed about 5.2% under A or 8.5% under B, and it is usually smaller. The fixed 0.7 mm in D-paper does not change.
- **Later work:** axial diameter (Section 11.3) depends directly on which axis is z. It is not part of G3 or this run, but this must be resolved or disclosed before any category analysis.

**Pre-specified axis assignments.** In-plane spacing is equal, so there are only three distinct cases.

| Assignment | 0.7 mm spacing on | Role |
|---|---|---|
| **Z2** | Axis 2, the stored order | **Primary**, fixed before any result, because it applies `voxsiz` in its listed order with no added rearrangement |
| Z1 | Axis 1 | Sensitivity, flagged by the evidence above |
| Z0 | Axis 0 | Sensitivity, for completeness |

Z2 being primary does not mean it is correct. **The assignment giving the best ICC is never selected.**

A code test must confirm that each axis uses its assigned spacing, using a hand-made shape with a known length along each axis.

## 6. Measurements
**Volume (mL):** number of mask voxels × voxel volume.

**Diameter: both calculations are always computed, under both spacing assumptions.**

| | Definition | Role |
|---|---|---|
| **D-paper** | Centre-to-centre maximum + fixed 0.7 mm | **Primary for G3** (recommended) |
| **D-centre** | Largest Euclidean distance between centres of mask voxels | Reported alongside it with equal prominence |

The largest distance is found through convex hull vertices. This gives the same result as all pairwise surface distances, because the farthest pair lies on the hull. If a hull cannot be formed (too few points, or points that are flat or in a line), all pairwise distances are used. A single voxel has D-centre 0.

### Basis for the diameter decision
**What the paper did (full text; it gives no equations for this):** it computed all pairwise distances between surface pixels and took the largest, then added "0.7 mm (the size of one voxel)" for voxel edges. Hole filling came first. The paper also gives spacing as 0.7 × 0.7 × 0.7 mm, so its 0.7 mm matches a voxel it describes as isotropic.

**What the plan says:** Stage 2 asks for the paper's convention by name for this replication. Section 11.3 asks to report both, and calls the added edge valid for isotropic voxels, with centre-to-centre primary for anisotropic spacing. So the two sections pull in different directions here.

**Is adding 0.7 mm appropriate?** It is supported as a reproduction of the paper. It is not an exact physical correction under either assumption.

| | Assumption A | Assumption B |
|---|---|---|
| In-plane voxel size | 0.7363 mm | 0.7012 to 0.7598 mm |
| Shortfall of 0.7 mm along an in-plane axis | 0.036 mm | 0.001 to 0.060 mm |
| As a share of the smallest stone (5.26 mm) | 0.7% | Up to 1.1% |

- **Why it applies to both:** a fixed value keeps the procedure identical under both assumptions and matches the paper. Scaling the edge by spacing would be a new method.
- **Limit on any spacing, even a truly isotropic one:** the real edge effect depends on direction. For box-shaped voxels, the distance from centre to outer corner across a pair adds up to about 1.21 to 1.28 mm along a diagonal, so a fixed 0.7 mm can fall short by up to about 0.58 mm. It is a convention, not a geometric truth.

### Primary metric recommendation
**D-paper for G3, with D-centre always reported beside it.** This was decided before any threshold diameter results exist, and not from any ICC.
1. **G3 is a replication gate.** Stage 2 defines this experiment using the paper's convention, and G3 compares against the paper's result.
2. **The reference measures outer extent.** A caliper spans the whole stone. D-centre runs between voxel centres, so it is expected to run short by construction, and absolute-agreement ICC would count that offset. That is a mismatch of definitions, not evidence about the stones.
3. **The anisotropy is small.** It is at most about 8.5% between in-plane and z (assumption B), while Section 11.3's concern is general.

**The trade-off:** this departs from Section 11.3's "centre-to-centre as primary". D-centre stays as the anisotropy-appropriate report, and any later non-replication work follows Section 11.3 unless the owner decides otherwise.

## 7. Agreement statistics
Measured against reference, on raw scales (mL, mm), for each spacing assumption and each diameter calculation. All 101 stones are in the main analysis.

**ICC**
- **Model:** two-way random effects, absolute agreement, single measurement. This is the plan's ICC2, or ICC(2,1), labelled `ICC(A,1)` in the installed `pingouin` 0.7.0.
- **Layout:** stones are targets, and the two "raters" are the method and the reference.
- **Reporting:** unrounded for the gate, and to 4 decimals in tables.

**Bland-Altman**
- **Difference:** measured − reference. Neubauer used the opposite sign, so comparisons with the paper must account for this.
- **Reported values:** mean difference (bias), SD with n − 1, and limits of agreement at bias ± 1.96 SD.
- **Plots:** difference against the mean of measured and reference, saved as vector PDF and PNG with units.

**Bootstrap confidence intervals**
- **Method:** percentile bootstrap, with 2,000 stone-level resamples of 101 stones with replacement (Section 11.9), using the 2.5th and 97.5th percentiles (numpy default linear interpolation).
- **Seed:** fixed in the config. The same resampled stones are used for every metric and assumption, so results stay paired.
- **Repeated stones:** each drawn copy is treated as a separate target in that resample.
- **ICC in resamples:** computed from the mean squares by the ICC(A,1) formula. A test must confirm it matches `pingouin` on fixture data.
- **`pingouin` interval:** listed for reference only, because it rounds to 2 decimals, too coarse near 0.99.
- **Applies to:** ICC, bias and limits of agreement.
- **Limits:** percentile intervals can be unreliable close to the ICC's upper bound of 1. Stone-level resampling also ignores any shared-scan effect, because scan sessions are unknown.

**Degenerate resamples**
- **Definition:** an ICC that is undefined or not finite, for example zero variance in a denominator.
- **Handling:** the resample is not replaced, redrawn or set to a value. It is left out of that metric's percentile only, and the count is reported.
- **Threshold:** if more than 1% (over 20 of 2,000) are degenerate for a metric, that interval is labelled unreliable, and this counts as a stop rule.
- **Expectation:** degenerate resamples should be very rare with 101 stones of varied size, but this is not assumed.

## 8. G3
**Unchanged: ICC must be greater than 0.99 for both volume and diameter.** Judged on all 101 stones at 272 HU, using the unrounded point estimate, with D-paper as the diameter (if approved) and axis assignment Z2. Reported separately for assumptions A and B. 0.9900 does not pass.

D-centre's ICC is reported beside it. It does not decide G3, and a difference triggers a stop (section 12).

## 9. Exploratory threshold sweep
**Exploratory sensitivity analysis only. It does not decide G3.**
- **Thresholds, fixed now:** 101, 130, 150, 200, 250, 272, 300, 350 and 400 HU.
- **Reporting:** every threshold is reported under both assumptions, with both diameters.
- **No selection:** no best threshold is chosen, and no sweep result is presented as validation.
- **Limit:** thresholds near 101 HU sit at the stored mask floor (101 to 105 HU).

## 10. Flagged stones and sensitivity analyses
All 101 stones stay in the main analysis. Flagged stones are listed. These analyses are reported separately and **do not decide G3**:

| Analysis | Change at 272 HU |
|---|---|
| S1 | Leave out reference-flagged stones: IDs 1, 2, 3, 46, 62 |
| S2 | Leave out the largest outliers: IDs 100, 101 |
| S3 | Only the 80 stones where the two spacing fields agree |
| S4 | Leave out the 6 float32 patches: IDs 46, 47, 56, 81, 83, 100 |
| S5 | Use the largest component rather than the union |

**Axis-assignment sensitivity (pre-specified).** Diameter ICC, bias and limits of agreement are reported for assignments Z2, Z1 and Z0, under spacing assumptions A and B and for D-paper and D-centre: 12 combinations in total, using the same bootstrap resamples. Volume is not repeated, because it does not change. These do not decide G3.

## 11. Reporting
Primary and exploratory results are kept in separate files and tables.

| File (in a new timestamped run folder) | Content |
|---|---|
| `a0_primary.csv` | **Primary:** 272 HU, both assumptions, volume, D-paper and D-centre, with the G3 outcome for each assumption |
| `a0_sweep_exploratory.csv` | **Exploratory:** all thresholds, labelled exploratory |
| `a0_sensitivity.csv` | S1 to S5, labelled sensitivity |
| `a0_axis_sensitivity.csv` | Z2, Z1 and Z0 diameter results, labelled sensitivity |
| `a0_stones.csv` | Per-stone measurements and flags for all 101 stones |
| `a0_bootstrap_summary.csv` | Resamples, seed and degenerate count for each metric |
| Bland-Altman PDF and PNG | Primary analysis, both assumptions |

Every reported number must come from these files (Section 12 of the plan). The original Freiburg files are read only.

## 12. Stop rules
After the single primary run, **stop and report to the owner if any of these occurs:**
1. G3 fails for either measurement under either assumption.
2. The two assumptions give different G3 outcomes, or materially different results.
3. G3 is borderline: an ICC point estimate from 0.985 to 0.995, or a bootstrap interval that contains 0.99.
4. D-paper and D-centre would give different G3 outcomes.
5. More than 1% of resamples are degenerate for any primary metric.
6. Any axis assignment would change a diameter G3 outcome, or move it into or out of the borderline range.

**In every case:** no adjusting the gate, choosing the better assumption, diameter or axis assignment, or changing the threshold, connectivity, minimum size, hole filling or diameter convention to force a pass.

If a real software fault is found, it is fixed with a test and disclosed, and all runs are reported. Debugging spacing, orientation, hole filling or the diameter convention (Section 9, Stage 2) happens only after reporting to the owner.

Whatever the result, the owner reviews it before anything further. **Stage 3 and the interval analysis do not start automatically.**

## 13. Build order
1. Measurement modules with Section 11.3 tests on hand-made shapes, plus threshold, agreement, config and logging code. Tests also cover per-axis spacing, all three axis assignments, both diameter calculations, the ICC formula and degenerate resamples.
2. Tests and lint pass before any Freiburg run.
3. Fill in `configs/a0_replicate_freiburg.yaml` from this protocol, with a fixed seed.
4. Run once, write the results, then stop.

## 14. Points approved by the owner
1. **Primary G3 diameter:** D-paper, departing from Section 11.3's anisotropic default (section 6).
2. **Multiple components:** the union of components is the primary stone mask.
3. **"Borderline" definition:** section 12, rule 3.
4. **Threshold rule:** strictly greater than the threshold.
5. **Primary axis assignment:** Z2, with Z1 and Z0 as sensitivity analyses, although the evidence conflicts (section 5).
6. **Uncommitted code:** `docs/experiments.md` has a Git commit column, and no commit is authorized. The run would record `91a2eaf` plus a note that the tree has uncommitted changes, and hashes of the source and config files. A commit before the run would be cleaner, but that is the owner's decision.
