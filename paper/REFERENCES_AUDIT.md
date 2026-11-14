# Reference audit trail (paper/references.bib)

This file holds the verification notes that used to be in the `note` fields of `references.bib`. They were moved here on 14 November 2026 so that they no longer print in the PDF bibliography. The bibliography now contains only bibliographic data.

**Sources used in the 14 November 2026 pass:**
- Crossref REST API (`api.crossref.org/works/<DOI>`), the registered metadata for DOI entries;
- arXiv API (`export.arxiv.org`);
- HAL API (`api.archives-ouvertes.fr`);
- the Springer chapter and book pages;
- the NeurIPS proceedings page;
- the QIBA profile PDF.

DBLP was not used: it returned a bot-check page, which was not bypassed.

## Entries corrected in the 14 November 2026 pass (bibliographic data only; no claim changed)

| Key | Correction | Source |
|---|---|---|
| neubauer2023 | **Title corrected.** The previous title ("Volumetric quantification of urinary stones using a semi-automatic segmentation tool") did not match DOI 10.1371/journal.pone.0286016. The registered title is "Effect of surface-partial-volume correction and adaptive threshold on segmentation of uroliths in computed tomography". | Crossref |
| xie2013 | **Title, authors, volume and article number corrected.** The previous title belonged to a different paper (Xie et al., AJR 2014, doi 10.2214/ajr.13.10830). The DOI 10.1259/bjr.20130160 is the BJR 2013 phantom paper that the project's literature audit read (docs/LIT_FULLTEXT_LOCK_20261111.md: "Between-scanner ICC 1.000, while 3 mm nodules vary from −19.3% to +41.6%"), so the DOI was kept and the title fixed to match it. | Crossref |
| obeten2026 | **Title corrected** to the registered title of Research Square preprint 10.21203/rs.3.rs-10604877/v1; full author list added. | Crossref |
| vargas2026 | **Full author names** added: Luisa Vargas, Simone Rossi, Maria A. Zuluaga. **Title and venue:** UNSURE 2026, MICCAI Workshop; HAL hal-05769863v1. | HAL API |
| li2022, molchanova2025, wasserthal2023, reimer2021, xie2013, jaubert2025 | "and others" or "et al." replaced by the full registered author lists. | Crossref / arXiv |
| eatonrosen2018, lambert2024 | Published LNCS chapter details added: pages and DOI. | Crossref |
| jaubert2025 | ISBI 2025 DOI 10.1109/ISBI60581.2025.10980751 added. | Crossref |
| romano2019 | NeurIPS 2019 proceedings record confirmed: title and authors on proceedings.neurips.cc. No volume or page numbers were taken from memory. | NeurIPS proceedings page |
| schott2025 | Article number 115009 added. The entry is not cited in the paper. | Crossref |
| qiba2023nodule | URL now rendered with `\url` (the `url` package was added to main.tex). | — |

## psmatv2026 (resolved 14 November 2026)

- **Verified** on the journal's own search page (jnm.snmjournals.org) as: Michael Treacy, Thomas Ng and Ciprian Catana, "Patient-Level Uncertainty for Automated PSMA Tumor Volume Estimation via Conformal Prediction", *Journal of Nuclear Medicine*, 14 May 2026, 67 (Supplement 1), 262200. The authors were added.
- Crossref and a general web search had not found the abstract.
- **Reading level:** abstract snippet only (record level).

## Original notes (as they were in references.bib before 14 November 2026)

| Key | Original note |
|---|---|
| neubauer2023 | VERIFY exact title against the publisher page |
| li2022 | VERIFY exact title |
| cumpanas2024 | record verified on Europe PMC 13 Nov 2026; abstract only |
| wilhelm2018 | record verified on Europe PMC 13 Nov 2026; abstract only |
| mukherjee2023 | VERIFY title |
| xie2013 | VERIFY title, volume, pages |
| eatonrosen2018 | arXiv:1806.08640 |
| lambert2024 | arXiv:2407.19938 |
| molchanova2025 | VERIFY title; arXiv:2311.08931 |
| isensee2021 | VERIFY (cited from memory; not opened in this project's audits) |
| romano2019 | VERIFY (cited from memory); arXiv:1905.03222 |
| jaubert2025 | arXiv:2506.04030; VERIFY author list |
| alijani2026 | arXiv:2610.09392 (preprint) |
| vargas2026 | HAL hal-05769863; VERIFY first names |
| obeten2026 | VERIFY title |
| kataria2026 | full text read 12 Nov 2026 (AI-assisted); VERIFY |
| andeol2025 | main text read 12 Nov 2026 (AI-assisted); VERIFY venue |
| dunn2023 | bibliographic record only; VERIFY |
| psmatv2026 | conference abstract; authors not yet retrieved: VERIFY authors before submission |
| thongprayoon2026 | record verified on Europe PMC 13 Nov 2026; abstract only (subscription) |
| schott2025 | record verified on Europe PMC 13 Nov 2026; abstract only (subscription); article number to verify |
| wu2026mswal | verified 14 Nov 2026 on the Springer chapter page (citation metadata; first online 20 September 2025; author order as published, which differs from arXiv:2503.13560v1). Dataset licence CC BY-NC 4.0 per the Hugging Face release (zhaodongwu/MSWAL) |
| vovk2005 | bibliographic record verified 14 Nov 2026 on the Springer book page (eBook ISBN 978-0-387-25061-8). Chapter 4, "Modifications of conformal predictors", pp. 97–130. Mondrian conformal predictors: Section 4.5 (with Section 8.4), confirmed 14 Nov 2026 by the authors' own statement in G. Shafer and V. Vovk, "A tutorial on conformal prediction" (www.alrw.net/articles/03.pdf, Section 5.3.1): the book's Sections 4.5 and 8.4 explain Mondrian models, whose conformal predictions are calibrated within categories. The book text itself was not read (paywalled) |
| qiba2023nodule | verified 14 Nov 2026 from the profile PDF, using the document's own required citation format; Claim 1 gives a per-nodule 95% CI Y ± 1.96·Y·CV by size range |
| liden2015 | record verified on Europe PMC 13 Nov 2026; abstract only |
| berkovitz2010 | record verified on Europe PMC 13 Nov 2026; abstract only |

## Verification status after 14 November 2026

- **Every DOI entry matches its Crossref record:** title, authors, journal, volume, issue and pages:
  - neubauer2023, li2022, elton2022, cumpanas2024, wilhelm2018, mukherjee2023, xie2013, molchanova2025, wasserthal2023, isensee2021, obeten2026, reimer2021, dunn2023, thongprayoon2026, liden2015, berkovitz2010, schott2025.
- **arXiv-only preprints match the arXiv API:** alijani2026, kataria2026, andeol2025. No published venue was found for andeol2025, so it is cited as an arXiv preprint.
- **Other sources:**
  - wu2026mswal and vovk2005: Springer pages (see above).
  - qiba2023nodule: profile PDF.
  - vargas2026: HAL.
- **Reading level stays as before:** several items were checked at record or abstract level only. This is the existing OD-20 limitation and is unchanged.

## Checks of 14 November 2026 (final review, items 2c and 2d)

- **obeten2026: now cites the published version.**
  - **Citation:** Obeten OM, Aliu A, Aliu OH, Jimoh A, Aliyu Z, Ezeanya CU. Effect of data leakage on deep learning accuracy for kidney stone classification on CT. Discov Comput Vis. 2026;1(1):2. doi:10.1007/s44584-026-00003-y. Published online 12 October 2026; open access, CC BY-NC-ND 4.0.
  - **Sources:** the Crossref record, and the Springer article page read in the built-in browser.
  - **Same work as the preprint:** the six authors are identical and in the same order, and the study is the same (201 patients, three centres, 3,364 original images). The published abstract lists "conformal prediction" among its analyses of a slice-level classifier. Crossref does not link the two records.
  - **Superseded:** the preprint 10.21203/rs.3.rs-10604877/v1.
  - **Paper wording:** "slice-level stone detection" changed to "slice-level stone classification", to match the published work.
- **wu2026mswal: licence.**
  - **Hugging Face dataset zhaodongwu/MSWAL:** no dataset card (the README returns 404). The API shows no `license` metadata, and gated is false.
  - **Only licence statement found:** the `licence` field of the released `dataset.json`, shown in the Dataset Viewer. It reads "CC BY-NC 4.0", with release date 8 July 2025.
  - **GitHub repository (wuzhaodongaipython/MSWAL-):** no LICENSE file and no licence statement.
  - **Conclusion:** CC BY-NC 4.0 is the licence the dataset release itself declares, in its metadata file, not in a card or LICENSE file. The paper's "(CC BY-NC 4.0)" stands.
- **Yaseen et al. 2026 (medRxiv preprint, doi 10.64898/2026.09.30.26364431; not cited).**
  - **What it is:** "Uncertainty-Aware Multiscale Treatment Response Forecasting in Advanced Non-Small Cell Lung Cancer Using Residual-Variance Conformal Prediction". The verbatim text read covers Sections I–II; later sections were read only as a summary.
  - **What it does:**
    - voxel-level conformal intervals for an SUV response ratio, using nested leave-one-patient-out calibration;
    - median aggregation of voxel bounds to a whole-lesion interval, which the paper says carries no formal guarantee;
    - analysis restricted to the delineated primary tumour.
  - **What it does not do:** model false-positive or missed lesions, or give a patient-level burden interval.
  - **Correction:** the earlier description in NOVELTY_GATE §8 ("voxel → lesion → patient conformal aggregation") overstated its overlap. It does not materially overlap N1 and is not cited.
