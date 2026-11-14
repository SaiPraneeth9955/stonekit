# StoneKit research workstation

`stonekit workstation` is a local, offline web application that presents the frozen StoneKit pipeline and its calibrated uncertainty for individual CT scans.

- **Address:** it listens on `127.0.0.1` only, port 8600 by default.
- **Dependencies:** it needs no internet connection and no package beyond the project environment. The server is Starlette with uvicorn, and the 3D library is the Plotly bundle shipped inside the installed `plotly` package.

```bash
stonekit workstation                 # http://127.0.0.1:8600
stonekit workstation --port 8700
STONEKIT_WORKSTATION_RUNS=/path/to/runs stonekit workstation   # open another folder of run outputs
```

## Screens

| Screen | Content | Source of every value |
|---|---|---|
| Worklist | Studies, matrix, spacing, stones, burden, 90% interval, eligibility, QC | `report.json`, `calibrated_ranges.json` |
| New analysis | MSWAL scan, Li development scan or imported NIfTI file; live progress through the stages | `scripts/demo_case.py` in a background process |
| Overview | Analysis pathway from CT to report; key measurements | saved outputs |
| Imaging | Linked axial, coronal and sagittal views of the kidney region on the original voxel grid (no resampling), with window presets, zoom, pan, kidney and stone overlays with opacity, orientation labels and stone navigation | `roi_crop.nii.gz`, kidney and stone label maps |
| 3D | Kidney and stone surfaces in RAS millimetres; isolate a stone | meshes written by the pipeline |
| Stones | Measurement kept separate from its 90% range; size band; location relative to the kidney segmentation; false-detection probability; 2D axial vs 3D size band; QC flags | `report.json`, `calibrated_ranges.json` |
| Burden & uncertainty | The detection-aware interval built up step by step: retained pieces → expected volume after false-detection modelling → measurement residual bands → missed-burden pool → raw interval → conformal factor *t* → calibrated interval; the CQR comparator | intermediate quantities of the frozen calculation |
| Quality & provenance | PASS / REVIEW / WARNING / FAIL / NOT AVAILABLE for every input check and processing stage; timeline; software versions; SHA-256 of the frozen artefacts | `report.json`, freeze manifest |
| Report | Structured report (print to PDF) and JSON export | the study data above |
| Research evidence | Locked-test, robustness, size-band, phantom and external results | committed aggregate files |
| Methodology | Pipeline, uncertainty method, data roles, limitations | — |

## Architecture

| Module | Role |
|---|---|
| `src/stonekit/workstation/services.py` | Read-only access to run outputs, study evidence and the integrity check. Nothing is calculated here |
| `src/stonekit/workstation/jobs.py` | Background analyses (one at a time, one GPU); progress parsing |
| `src/stonekit/workstation/report.py` | Print-ready HTML report built from the study data |
| `src/stonekit/workstation/server.py` | HTTP API and static front end |
| `src/stonekit/workstation/static/` | Front end: design system, CT viewer, screens |
| `src/stonekit/t1/case_ranges.py` | Calibrated ranges for one scan from the frozen T1 calibration (through `stonekit.t1.external`) |
| `scripts/demo_case.py` | Pipeline → 8-flip TTA → calibrated ranges for one scan |

**API endpoints:**
- `GET /api/worklist`
- `GET /api/studies/{id}`, plus its `volume` and `meshes` endpoints
- `GET /api/evidence`
- `GET /api/integrity`
- `GET /api/sources`
- `PUT /api/uploads/{name}`
- `POST` and `GET /api/analyses`
- `GET /report/{id}`
- `GET /export/{id}.json`

**Deep links:** `#/study/{id}/{tab}/{stone}` opens a study tab with a stone selected.

## Scientific integrity

- **Frozen calibration:** ranges and intervals come only from the T1 calibration file. It is read after its SHA-256 has been checked, and the development pools are checked against their recorded hashes. Nothing is fitted on a new scan.
- **Burden build-up:** it shows intermediate quantities of the same calculation. Tests (`tests/test_case_ranges.py`) check three things:
  - the single-scan pieces reproduce the stored external-check table;
  - the calibrated interval equals the frozen engine's interval;
  - the comparator agrees to 1e-6 relative.
- **Eligibility:** ranges are computed only for scans that the aorta heuristic marks as non-contrast (median below 100 HU).
- **Locked scans:** the Li calibration and test scans, and the six reserved demonstration cases, are refused.
- **Absent artefacts:** in a public clone the model checkpoint and the development tables are absent. The provenance view then reports them as NOT AVAILABLE, never as verified.

## Screenshots

`docs/assets/screenshots/` holds screenshots of two external MSWAL scans (CC BY-NC 4.0), captured from a workstation instance that sees only those scans. They contain no Li 2022 data.
