"""T1 analysis stage (docs/T1_RUNNER_DESIGN.md §2): calibrate -> hash -> only then load test -> evaluate.

`run_analysis` receives the development (fit) and calibration tables, plus a CALLABLE that loads the test tables.
The callable is invoked only after every calibration quantity has been written to `calibration_quantities.json`
and its SHA-256 recorded, so test data cannot influence calibration. Endpoints follow protocol v3 (draft):
- per-stone (descriptive P1 and secondaries): U2 (final; volume and 3D diameter), U0/U1 (baseline), and U4 and
  U2-asymmetric (exploratory), at 0.90 (0.80 descriptive);
- N-B patient-level joint coverage (secondary);
- P2: the single formal test (OD-1) at 0.90 plus descriptive burden results, and the filter ablation (OD-3);
- §5.1 item 5 (OD-17): the same analysis without reserved scans that have a reader-confirmed annotation error,
  refitted with the identical procedure and seed (an estimate only; its calibration is hashed with the primary).
`mode` is "DRESS_REHEARSAL" (development scans standing in for calibration and test; never a result) or "T1".
Only mode "T1" writes the COMPLETED marker, and only the gated runner may pass it.
"""

import csv
import hashlib
import io
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stonekit.uq import confirmatory as cfm
from stonekit.uq import conformal as cf

SEED = 20261105
TARGETS = ("volume_mm3", "max_diameter_3d_mm")
ALPHAS = (0.10, 0.20)
# D-REV answer codes, identical to scripts/apply_radiology_adjudication.py (docs/RADIOLOGY_ADJUDICATION_PROTOCOL.md §5)
Q1_CODES = {"confirmed_urinary_stone", "confirmed_non_stone", "indeterminate", "other_relevant_abnormality"}
Q2_CODES = {"valid", "invalid", "indeterminate", "not_applicable"}
UNLABELLED = "unlabelled region"
PER_STONE = {"U0": "baseline", "U1": "baseline", "U2": "final", "U4": "exploratory", "U2-asymmetric": "exploratory"}


def _tail(x, a):
    x = np.sort(np.asarray(x, float))
    k = math.ceil((len(x) + 1) * (1 - a))
    return float(x[k - 1]) if k <= len(x) else math.inf


def asym_params(cal: pd.DataFrame, target: str, alpha: float) -> dict:
    r = cf.log_ratio(cal[f"ref_{target}"], cal[f"pred_{target}"])
    edges = cf.merge_strata(cal["pred_max_diameter_3d_mm"], cfm.EDGES, cfm.MIN_STRATUM)
    g = np.searchsorted(edges, cal["pred_max_diameter_3d_mm"], side="right")
    return {"edges": list(edges), "lo": [-_tail(-r[g == k], alpha / 2) for k in range(len(edges) + 1)], "hi": [_tail(r[g == k], alpha / 2) for k in range(len(edges) + 1)]}


def nb_params(cal: pd.DataFrame, target: str, alpha: float) -> dict:
    patients = np.sort(cal["case"].unique())
    half = set(np.random.default_rng(SEED).choice(patients, size=len(patients) // 2, replace=False))
    a, b = cal[cal["case"].isin(half)], cal[~cal["case"].isin(half)]
    s = cf.fit_mondrian(np.abs(cf.log_ratio(a[f"ref_{target}"], a[f"pred_{target}"])), a["pred_max_diameter_3d_mm"], cfm.EDGES, alpha, cfm.MIN_STRATUM)
    qs = np.asarray(s.quantiles)
    rb = np.abs(cf.log_ratio(b[f"ref_{target}"], b[f"pred_{target}"])) / qs[s.index(b["pred_max_diameter_3d_mm"])]
    tau = cf.conformal_quantile(pd.Series(rb).groupby(b["case"].to_numpy()).max().to_numpy(), alpha)
    return {"edges": list(s.edges), "q": qs.tolist(), "tau": float(tau)}


def _stone_bounds(method, cal, test, target, alpha, params):
    if method == "U2-asymmetric":
        p = params["asym"][f"{target}|{alpha}"]
        g = np.searchsorted(p["edges"], test["pred_max_diameter_3d_mm"], side="right")
        return np.asarray(p["lo"])[g], np.asarray(p["hi"])[g]
    return cfm.stone_intervals(method, cal, test, target, alpha)


def _summ(covered: np.ndarray, cases: np.ndarray) -> dict:
    n, k = len(covered), int(covered.sum())
    lo, hi = cf.clopper_pearson(k, n)
    blo, bhi = cf.cluster_bootstrap_coverage(covered, cases, 10000, SEED) if n else (math.nan, math.nan)
    return {"n": n, "scans": len(set(cases)), "coverage": k / n if n else None, "cp95": [lo, hi], "cluster_boot95": [blo, bhi]}


def reader_confirmed_errors(sheet: Path, expected_sha256: str) -> frozenset[str]:
    """Protocol v3 §5.1 item 5 (OD-17): scans with a reader-confirmed annotation error, from the HASHED sheet only.

    A scan is marked when any of its blinded first-pass answers is Q2 = invalid, or Q1 = confirmed urinary stone at
    an unlabelled location (a label omission). The sheet must match `expected_sha256` and be complete; anything else
    is refused. No technical-audit output and no screening code is used.
    """
    data = Path(sheet).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("adjudication sheet SHA-256 does not match the recorded hash")
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    marked = set()
    for r in rows:
        if r["q1_finding"] not in Q1_CODES or r["q2_label_validity"] not in Q2_CODES:
            raise ValueError(f"{r['adj_id']}: incomplete or invalid Q1/Q2")
        if r["blinded_first_pass"] != "yes":
            raise ValueError(f"{r['adj_id']}: not a blinded first pass")
        omission = r["q1_finding"] == "confirmed_urinary_stone" and r["location_type"] == UNLABELLED
        if r["q2_label_validity"] == "invalid" or omission:
            marked.add(r["case"])
    return frozenset(marked)


def _calibrate(fit: cfm.Tables, cal: cfm.Tables) -> tuple[dict, dict]:
    mc = cfm.matched(cal)
    params = {"asym": {}, "nb": {}, "u0": {}, "u1": {}, "u2": {}}
    for target in TARGETS:
        r = cf.log_ratio(mc[f"ref_{target}"], mc[f"pred_{target}"])
        for alpha in ALPHAS:
            key = f"{target}|{alpha}"
            params["asym"][key] = asym_params(mc, target, alpha)
            params["nb"][key] = nb_params(mc, target, alpha)
            from scipy import stats

            params["u0"][key] = list(cf.bland_altman_log(r, stats.norm.ppf(1 - alpha / 2)))
            params["u1"][key] = cf.conformal_quantile(np.abs(r), alpha)
            s = cf.fit_mondrian(np.abs(r), mc["pred_max_diameter_3d_mm"], cfm.EDGES, alpha, cfm.MIN_STRATUM)
            params["u2"][key] = {"edges": list(s.edges), "q": list(s.quantiles), "counts": list(s.counts)}
    burden = {(alpha, f): cfm.fit_burden(fit, cal, alpha, filtered=f) for alpha in ALPHAS for f in (True, False)}
    return params, burden


def _quantities(params: dict, burden: dict, cal: cfm.Tables) -> dict:
    return {"per_stone": params, "burden": {f"{a}|filtered={f}": b.quantities() for (a, f), b in burden.items()}, "calibration_scans": sorted(cal.scans.case)}


def _evaluate(params: dict, burden: dict, cal: cfm.Tables, test: cfm.Tables) -> dict:
    mc, mt = cfm.matched(cal), cfm.matched(test)
    ev = {"test_scans": int(test.scans.case.nunique()), "per_stone": {}, "nb": {}, "burden": {}}
    for target in TARGETS:
        r = cf.log_ratio(mt[f"ref_{target}"], mt[f"pred_{target}"])
        band = cfm.stratum(mt["pred_max_diameter_3d_mm"])
        for alpha in ALPHAS:
            key = f"{target}|{1 - alpha:.2f}"
            ev["per_stone"][key] = {}
            for method, role in PER_STONE.items():
                lo, hi = _stone_bounds(method, mc, mt, target, alpha, params)
                cov = ((r >= lo) & (r <= hi)).to_numpy() if hasattr(r, "to_numpy") else (r >= lo) & (r <= hi)
                e = {"role": role, "marginal": _summ(np.asarray(cov), mt.case.to_numpy()), "median_rel_width": float(np.median(np.exp(hi) - np.exp(lo)))}
                e["by_band"] = {n: _summ(np.asarray(cov)[band == k], mt.case.to_numpy()[band == k]) for k, n in enumerate(("<5", "5-<10", ">=10"))}
                ev["per_stone"][key][method] = e
            p = params["nb"][f"{target}|{alpha}"]
            g = np.searchsorted(p["edges"], mt["pred_max_diameter_3d_mm"], side="right")
            half = p["tau"] * np.asarray(p["q"])[g]
            joint = pd.Series(np.abs(np.asarray(r)) <= half).groupby(mt.case.to_numpy()).all()
            ev["nb"][key] = {"patients": len(joint), "joint_coverage": float(joint.mean()), "cp95": list(cf.clopper_pearson(int(joint.sum()), len(joint))),
                             "median_rel_width": float(np.median(np.exp(half) - np.exp(-half)))}  # fmt: skip
    for (alpha, f), b in burden.items():
        iv = cfm.apply_burden(b, test)
        ev["burden"][f"{1 - alpha:.2f}|filtered={f}"] = cfm.p2_test(iv, nominal=1 - alpha)
    ev["formal_P2"] = ev["burden"]["0.90|filtered=True"]["formal"]
    return ev


ITEM5 = "item5_reader_confirmed_errors"


@dataclass(frozen=True)
class Exclusion:
    """A pre-specified sensitivity cohort: `scans` are removed; `reserved_only` leaves the development fit set intact."""

    scans: frozenset[str]
    reserved_only: bool


def run_analysis(fit: cfm.Tables, cal: cfm.Tables, load_test: Callable[[], cfm.Tables], out_dir: Path, mode: str,
                 annotation_error_scans: frozenset[str] | None = None, sensitivity: dict[str, Exclusion] | None = None,
                 item5_unavailable: str | None = None) -> dict:  # fmt: skip
    """Primary analysis plus pre-specified sensitivity cohorts (protocol v3 §5.1).

    `annotation_error_scans`: output of `reader_confirmed_errors` (item 5; reserved scans only). In mode T1, item 5
    is pre-specified, so either it or `item5_unavailable` (a stated reason, e.g. no qualified reader; OD-19) must be
    given. `sensitivity`: further named cohorts (items 1-3). Every sensitivity calibration is computed and hashed
    together with the primary one BEFORE the test loader is called; each is refitted with the identical procedure
    and seed. Sensitivity results are estimates only, never a second formal test.
    """
    if mode not in ("DRESS_REHEARSAL", "T1"):
        raise ValueError("mode must be DRESS_REHEARSAL or T1")
    if mode == "T1" and annotation_error_scans is None and not item5_unavailable:
        raise ValueError("mode T1 requires annotation_error_scans (protocol v3 §5.1 item 5) or a stated item5_unavailable reason")
    specs = dict(sensitivity or {})
    if annotation_error_scans is not None:
        specs[ITEM5] = Exclusion(frozenset(annotation_error_scans), reserved_only=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    # ---- 1. calibrate (no test data exist in this function yet): primary, then every sensitivity calibration
    params, burden = _calibrate(fit, cal)
    quantities = {"mode": mode, **_quantities(params, burden, cal), "fit_scans_n": int(fit.scans.case.nunique())}
    fitted = {}
    for name, spec in specs.items():
        fit_removed = [] if spec.reserved_only else sorted(spec.scans & set(fit.scans.case))
        cal_removed = sorted(spec.scans & set(cal.scans.case))
        rec = {"marked_scans": sorted(spec.scans), "reserved_only": spec.reserved_only, "fit_removed": fit_removed, "calibration_removed": cal_removed}
        if fit_removed or cal_removed:
            fit_s, cal_s = fit.subset(set(fit.scans.case) - set(fit_removed)), cal.subset(set(cal.scans.case) - spec.scans)
            p_s, b_s = _calibrate(fit_s, cal_s)
            fitted[name] = (p_s, b_s, cal_s)
            rec.update(_quantities(p_s, b_s, cal_s))
            rec["fit_scans_n"] = int(fit_s.scans.case.nunique())
        else:
            fitted[name] = (params, burden, cal)
            rec["note"] = "calibration identical to primary"
        quantities.setdefault("sensitivity", {})[name] = rec
    qpath = out_dir / "calibration_quantities.json"
    qpath.write_text(json.dumps(quantities, indent=2, default=float), encoding="utf-8")
    qsha = hashlib.sha256(qpath.read_bytes()).hexdigest()
    (out_dir / "calibration_quantities.sha256").write_text(qsha + "\n", encoding="utf-8")
    # ---- 2. only now load the test tables
    test = load_test()
    ev = {"mode": mode, "calibration_quantities_sha256": qsha, **_evaluate(params, burden, cal, test)}
    if specs or item5_unavailable:
        ev["sensitivity"] = {}
    for name, spec in specs.items():
        q = quantities["sensitivity"][name]
        test_removed = sorted(spec.scans & set(test.scans.case))
        item = {"role": "secondary sensitivity (estimate only; not a formal test)", "marked_scans": q["marked_scans"],
                "fit_removed": q["fit_removed"], "calibration_removed": q["calibration_removed"], "test_removed": test_removed}  # fmt: skip
        if not (q["fit_removed"] or q["calibration_removed"] or test_removed):
            item["result"] = "no scan removed; analysis identical to primary"
        else:
            p, b, c = fitted[name]
            item["evaluation"] = _evaluate(p, b, c, test.subset(set(test.scans.case) - spec.scans))
        ev["sensitivity"][name] = item
    if annotation_error_scans is None and item5_unavailable:
        ev["sensitivity"][ITEM5] = {"result": f"not performed: {item5_unavailable}"}
    (out_dir / "evaluation.json").write_text(json.dumps(ev, indent=2, default=float), encoding="utf-8")
    if mode == "T1":
        (out_dir / "COMPLETED").write_text(qsha + "\n", encoding="utf-8")
    return ev
