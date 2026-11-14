# Paper Stage E audit (14 November 2026)

**Manuscript:** paper/main.tex; compiled to paper/build/main.pdf (git-ignored). **Not committed.**

## 1. Number trace

- **Method:** every numeric token between the abstract and the acknowledgment was extracted from main.tex (373 occurrences, 170 distinct values). Each was matched to a value recomputed from its source artefact:
  - T1 `evaluation.json` (SHA d8cf65da…);
  - MSWAL `external_check.json` (SHA 6b57e1f6…);
  - the dry-run `summary.json`;
  - the E6 OOF per-stone table;
  - the Freiburg R2 `per_stone.csv`;
  - a hand-traced list for docs/results.md values and code constants.
- **Unmatched tokens:** only non-results ("Li 2022", "CC BY 4.0", a table spacing, the rounded "ICC near 0.99" and "over 60%"). **No result number is untraceable.**
- **Phantom ICC:**
  - The R2 run stores no ICC. ICC(A,1) recomputed from the R2 per-stone table, primary configuration (spacing P, rep1 parameters, M1, 101 stones), is 0.993, and the median absolute relative volume error there is 23.6%.
  - rep1 `primary_all_stone.M1.vol_icc_all` = 0.99296 for the same method and parameters.
  - So the "ICC 0.993 with 23.6% error" pairing is internally consistent.
  - rep1's own 25.0% error uses a different error definition and is not quoted.

## 2. Corrections made (no result changed)

- **Conformal language:**
  - "finite-sample coverage guarantees under exchangeability" is now stated as a theoretical property ("In theory … attains … whenever calibration and test data are exchangeable"), followed by "must be checked empirically".
- **"43% narrower":**
  - It is now tied to its definition: the median per-patient width ratio, 0.573.
  - The ratio of median widths (3,090 / 7,431 = 0.42) is a different quantity and is not used as the headline.
- **Statistics described exactly as computed:**
  - percentile bootstrap over scans for the median per-scan ratio and for the coverage difference;
  - Clopper–Pearson per-stone intervals treat stones as independent;
  - only one formal test, so no multiplicity adjustment;
  - the simulation is described as development design characteristics, not a prediction.
- **Methods made reproducible:**
  - the overlap-based matching rule;
  - the 11 non-trainable development scans predicted by a model not trained on them;
  - the pipeline diameter is the largest voxel-centre distance;
  - the phantom diameters follow the published definition (+0.7 mm);
  - the ICC type is ICC(A,1).
- **Claims softened where evidence is indirect:**
  - mechanism sentence marked "a plausible explanation, not tested here";
  - "needs/needed" for size stratification replaced by "improved small-stone coverage" or "matters";
  - "None of these reports" became "In the sources we examined, none reports";
  - the Reimer 2021 citation is now attached only to the multiplanar-accuracy statement;
  - Andéol 2025 is described as joint coverage of all objects in an image;
  - "only correlation or ICC" for refs 1–3 became "agreement or overlap statistics alone".
- **Labels:**
  - Fig. 3 caption: "Robustness" became "Sensitivity analyses and external check".
  - Table II caption: all rows are descriptive.
  - Table II footnote a: states the comparison base (2,798 vs 2,210 mm³ in the same 200 splits).
- **Bibliography:** the HAL identifier now prints as "HAL".
- **Layout:**
  - `\IEEEtriggeratref` removed (columns now balance naturally);
  - Table I is defined before Results so it sits above its subsection on page 4.

## 3. Checks

- **Build:** 6 pages; 0 LaTeX errors; 0 overfull boxes; 0 undefined references; fonts embedded.
- **Bibliography:** BibTeX 0 errors, 0 warnings. 28 cited entries, all present; schott2025 is in the bib but uncited, so it is not printed.
- **PDF text:** no VERIFY, AI or audit text. Two PENDING placeholders remain: the author block and the acknowledgment/disclosure.

## 4. Stale files (not deleted)

- **Superseded figure renders in paper/figures:**
  - fig_freiburg_r2_* and fig_freiburg_r3_* (.pdf, .png);
  - fig_li_dev_burden_intervals.*, fig_li_dev_detection_by_size.*, fig_li_dev_stone_coverage_by_size.*.
- **Keep:** fig_freiburg_r2_volume_error_by_type_data.csv and fig_freiburg_r3_diameter_agreement_data.csv. They are inputs to Fig. 2 via scripts/make_paper_figures.py.
- **paper/supplement/:** tables S1–S4 from 12 November (development design tables). They are not referenced by the current manuscript and are partly superseded by T1.
- **Scripts:** scripts/make_li_development_figures.py (superseded) and scripts/make_freiburg_figures.py (still produces the Fig. 2 input CSVs).
