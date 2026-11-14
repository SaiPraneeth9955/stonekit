"""Confirmatory (T1) interval engine: calibrate on one set of scans, apply to another, once.

Implements the frozen-arm methods exactly as defined by E6 and E8 (scripts/e6_uncertainty_design.py,
scripts/e8_burden_uncertainty.py) and the owner decisions of 12 November 2026 (docs/decisions.md OD-1/OD-2):

Per-stone (one-to-one matched stones; reference = annotation; log-ratio intervals V_hat * exp([lo, hi])):
- Baseline arm: U0 Bland-Altman (log), U1 global split conformal.
- Final arm: U2 Mondrian by predicted 3D diameter (volume), U4 normalised by TTA spread (3D diameter).
  All are calibrated on the CALIBRATION scans' matched stones only (as in E6, where "cal" = other folds).

Patient burden (B = total annotated volume, B_hat = total predicted volume after the 30 mm filter):
- P2-0 naive: per-piece Mondrian bounds summed; per-stratum quantiles from the CALIBRATION matched stones.
- P2-B direct: split conformal on |log((B + c)/(B_hat + c))| over the CALIBRATION scans.
- P2-B-CQR: quantile regression of log(B + c) on log(B_hat + c), log(1 + sum (1 - pi) V_hat), slice mm,
  FITTED on the development (FIT) scans and conformalised (CQR score) on the CALIBRATION scans.
- P2-A decomposition (master plan v2 amendment 2): per-piece false-positive probability pi from a logistic
  model FITTED on the development scans; per-stone log-ratio residuals drawn from the development
  one-to-one residuals of the same predicted-size stratum; missed burden drawn from the development per-scan
  missed burden of the same slice group (<= 1 mm / > 1 mm; all scans if < 10); 2,000 draws -> central raw
  interval; then ONE log-scale factor t computed on the CALIBRATION scans and applied to the test scans.
So the split is: components that need many examples are fitted on development out-of-fold data; every
conformal quantile or factor uses the calibration scans only; test scans are only evaluated.

Numerical note: QuantReg is run with max_iter = 50000 (E8 used the statsmodels default 1000 and hit the
limit once; a read-only check on 12 November showed coverage and median width unchanged on convergence).
Draws are seeded per scan from the case ID, so results do not depend on scan order.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from stonekit.uq import conformal as cf

SEED = 20261105
EDGES = (5.0, 10.0)
MIN_STRATUM = 20
C_MM3 = 5.0
DRAWS = 2000
FP_FEATURES = ["log_volume", "p_mean", "tta_spread", "threshold_sensitivity", "distance_to_kidney_mm", "hu100"]
FILTER_MM = 30.0
BOOT = 10000


@dataclass
class Tables:
    """OOF-style tables (scripts/build_oof_stone_table.py format) for one set of scans."""

    annotated: pd.DataFrame
    predicted: pd.DataFrame
    scans: pd.DataFrame

    def subset(self, cases) -> "Tables":
        cases = set(cases)
        return Tables(self.annotated[self.annotated.case.isin(cases)].copy(), self.predicted[self.predicted.case.isin(cases)].copy(),
                      self.scans[self.scans.case.isin(cases)].copy())  # fmt: skip


def matched(t: Tables) -> pd.DataFrame:
    a = t.annotated
    return a[(a.group == "one_to_one") & (a.ref_volume_mm3 > 0) & (a.pred_volume_mm3 > 0)].copy().reset_index(drop=True)


def stratum(d) -> np.ndarray:
    return np.searchsorted(EDGES, np.asarray(d, float), side="right")


# ---------------------------------------------------------------- per-stone (identical to E6 `intervals`)
def stone_intervals(method: str, cal: pd.DataFrame, test: pd.DataFrame, target: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    r_cal = cf.log_ratio(cal[f"ref_{target}"], cal[f"pred_{target}"])
    if method == "U0":
        lo, hi = cf.bland_altman_log(r_cal, stats.norm.ppf(1 - alpha / 2))
        return np.full(len(test), lo), np.full(len(test), hi)
    if method == "U1":
        q = cf.conformal_quantile(np.abs(r_cal), alpha)
        return np.full(len(test), -q), np.full(len(test), q)
    if method == "U2":
        strata = cf.fit_mondrian(np.abs(r_cal), cal["pred_max_diameter_3d_mm"], EDGES, alpha, MIN_STRATUM)
        q = np.asarray(strata.quantiles)[strata.index(test["pred_max_diameter_3d_mm"])]
        return -q, q
    if method in ("U4", "U5"):
        signal = {"U4": "pred_tta_spread", "U5": "pred_threshold_sensitivity"}[method]
        floor = float(np.nanquantile(cal[signal], 0.05))
        floor = floor if floor > 0 else float(np.nanmin(cal[signal][cal[signal] > 0])) if (cal[signal] > 0).any() else 1.0
        s_cal = np.maximum(cal[signal].to_numpy(float), floor)
        s_test = np.maximum(test[signal].to_numpy(float), floor)
        q = cf.conformal_quantile(np.abs(r_cal) / s_cal, alpha)
        return -q * s_test, q * s_test
    raise ValueError(f"unknown method {method}")


def stone_coverage(cal: Tables, test: Tables, method: str, target: str, alpha: float) -> pd.DataFrame:
    c, t = matched(cal), matched(test)
    lo, hi = stone_intervals(method, c, t, target, alpha)
    r = cf.log_ratio(t[f"ref_{target}"], t[f"pred_{target}"])
    return pd.DataFrame({"case": t.case.to_numpy(), "pred_d3": t.pred_max_diameter_3d_mm.to_numpy(), "covered": (r >= lo) & (r <= hi),
                         "rel_width": np.exp(hi) - np.exp(lo)})  # fmt: skip


# ---------------------------------------------------------------- burden
def _prepare(t: Tables, filtered: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = t.predicted.copy()
    if filtered:
        p = p[~p["beyond_30mm"].astype(bool)]
    p["stratum"] = stratum(p["max_diameter_3d_mm"])
    p["log_volume"], p["hu100"] = np.log(p["volume_mm3"]), p["mean_hu"] / 100
    p["is_fp"] = (p["group"] == "false_positive").astype(int)
    s = t.scans.copy()
    s["slice_group"] = np.where(s["slice_mm"] <= 1.0, "thin", "thick")
    s["B"] = s["annotated_burden_mm3"]
    s["B_hat"] = s["predicted_burden_filtered_mm3"] if filtered else s["predicted_burden_mm3"]
    missed = t.annotated[t.annotated.group == "missed"].groupby("case")["ref_volume_mm3"].sum()
    s["missed_mm3"] = s["case"].map(missed).fillna(0.0)
    return p, s, matched(t)


def _case_rng(case: str, alpha: float) -> np.random.Generator:
    h = int(hashlib.sha256(f"{case}|{alpha}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng([SEED, h])


@dataclass
class BurdenCalibration:
    """Everything the burden methods need, computed from the development (fit) and calibration scans only.
    Built BEFORE any test scan is loaded (T1 order); `quantities()` is what the T1 runner hashes."""

    alpha: float
    filtered: bool
    feats: list
    model: object
    base_rate: float
    e_fit: np.ndarray
    pool_e: dict
    missed_fit: pd.DataFrame
    q_s: dict
    q_b: float
    lo_m: object
    hi_m: object
    q_cqr: float
    t_fac: float

    def pi_of(self, p: pd.DataFrame) -> np.ndarray:
        out = np.full(len(p), self.base_rate)
        ok = p[self.feats].notna().all(axis=1).to_numpy()
        if ok.any():
            out[ok] = self.model.predict_proba(p.loc[ok, self.feats])[:, 1]
        return out

    def raw_a(self, p: pd.DataFrame, s: pd.DataFrame) -> dict:
        out = {}
        sf = self.missed_fit
        for _, row in s.iterrows():
            rng = _case_rng(row.case, self.alpha)
            pool_m = sf[sf.slice_group == row.slice_group]["missed_mm3"].to_numpy()
            pool_m = pool_m if len(pool_m) >= 10 else sf["missed_mm3"].to_numpy()
            total = np.zeros(DRAWS)
            for _, pc_ in p[p.case == row.case].iterrows():
                e_pool = self.pool_e[pc_.stratum] if len(self.pool_e[pc_.stratum]) else self.e_fit
                keep = rng.random(DRAWS) >= pc_.pi
                total += keep * pc_.volume_mm3 * np.exp(rng.choice(e_pool, DRAWS))
            total = total + rng.choice(pool_m, DRAWS)
            out[row.case] = np.quantile(total, [self.alpha / 2, 1 - self.alpha / 2])
        return out

    def quantities(self) -> dict:
        def h(a) -> str:
            return hashlib.sha256(np.ascontiguousarray(np.asarray(a, float)).tobytes()).hexdigest()

        return {"alpha": self.alpha, "filtered": self.filtered, "fp_features": self.feats,
                "fp_model": {"coef": self.model.coef_.tolist(), "intercept": self.model.intercept_.tolist(), "base_rate": self.base_rate},
                "residual_pools": {str(k): {"n": len(v), "sha256": h(v)} for k, v in self.pool_e.items()},
                "missed_pool": {"n": len(self.missed_fit), "sha256": h(self.missed_fit["missed_mm3"].to_numpy())},
                "p2_0_q_s": {str(k): float(v) for k, v in self.q_s.items()}, "p2_b_q": float(self.q_b),
                "cqr": {"lo_params": np.asarray(self.lo_m.params).tolist(), "hi_params": np.asarray(self.hi_m.params).tolist(), "q": float(self.q_cqr)},
                "p2_a_t": float(self.t_fac)}  # fmt: skip


def _adj(p: pd.DataFrame, s: pd.DataFrame) -> pd.Series:
    return s["case"].map(p.assign(w=(1 - p.pi) * p.volume_mm3).groupby("case")["w"].sum()).fillna(0.0)


def _cqr_x(p: pd.DataFrame, s: pd.DataFrame) -> np.ndarray:
    import statsmodels.api as sm

    return sm.add_constant(np.column_stack([np.log(s.B_hat + C_MM3), np.log1p(_adj(p, s)), s.slice_mm]), has_constant="add")


def fit_burden(fit: Tables, cal: Tables, alpha: float, filtered: bool = True, fp_features: list[str] | None = None) -> BurdenCalibration:
    """Calibrate every burden method from development (fit) and calibration scans. No test data are involved.
    `fp_features` exists only for the N-C ablation (OD-13); the frozen arms use the default FP_FEATURES."""
    import statsmodels.api as sm

    feats = list(fp_features) if fp_features else FP_FEATURES
    pf, sf, mf = _prepare(fit, filtered)
    pc, sc, mc = _prepare(cal, filtered)
    usable = pf.dropna(subset=feats)
    model = LogisticRegression(max_iter=1000).fit(usable[feats], usable.is_fp)
    e_fit = np.log(mf.ref_volume_mm3 / mf.pred_volume_mm3).to_numpy()
    s_fit = stratum(mf.pred_max_diameter_3d_mm)
    cal_obj = BurdenCalibration(alpha=alpha, filtered=filtered, feats=feats, model=model, base_rate=float(pf.is_fp.mean()), e_fit=e_fit,
                                pool_e={k: e_fit[s_fit == k] for k in range(len(EDGES) + 1)},
                                missed_fit=sf[["case", "slice_group", "missed_mm3"]].copy(), q_s={}, q_b=math.nan,
                                lo_m=None, hi_m=None, q_cqr=math.nan, t_fac=math.nan)  # fmt: skip
    pc["pi"], pf["pi"] = cal_obj.pi_of(pc), cal_obj.pi_of(pf)
    raw_cal = cal_obj.raw_a(pc, sc)
    score = [max(np.log((raw_cal[c][0] + C_MM3) / (b + C_MM3)), np.log((b + C_MM3) / (raw_cal[c][1] + C_MM3))) for c, b in zip(sc.case, sc.B, strict=True)]
    cal_obj.t_fac = cf.conformal_quantile(np.array(score), alpha)
    e_cal = np.abs(np.log(mc.ref_volume_mm3 / mc.pred_volume_mm3)).to_numpy()
    s_cal = stratum(mc.pred_max_diameter_3d_mm)
    cal_obj.q_s = {k: cf.conformal_quantile(e_cal[s_cal == k], alpha) if (s_cal == k).any() else math.inf for k in range(len(EDGES) + 1)}
    cal_obj.q_b = cf.conformal_quantile(np.abs(np.log((sc.B + C_MM3) / (sc.B_hat + C_MM3))).to_numpy(), alpha)
    y_fit, y_cal = np.log(sf.B + C_MM3).to_numpy(), np.log(sc.B + C_MM3).to_numpy()
    x_fit, x_cal = _cqr_x(pf, sf), _cqr_x(pc, sc)
    cal_obj.lo_m = sm.QuantReg(y_fit, x_fit).fit(q=alpha / 2, max_iter=50000)
    cal_obj.hi_m = sm.QuantReg(y_fit, x_fit).fit(q=1 - alpha / 2, max_iter=50000)
    cal_obj.q_cqr = cf.conformal_quantile(np.maximum(cal_obj.lo_m.predict(x_cal) - y_cal, y_cal - cal_obj.hi_m.predict(x_cal)), alpha)
    return cal_obj


def apply_burden(c: BurdenCalibration, test: Tables) -> dict[str, pd.DataFrame]:
    """Apply a frozen BurdenCalibration to test scans. Test annotations (B) are carried only for evaluation."""
    pt, st, _ = _prepare(test, c.filtered)
    pt["pi"] = c.pi_of(pt)
    raw_test = c.raw_a(pt, st)
    xt = _cqr_x(pt, st)
    cqr_lo, cqr_hi = c.lo_m.predict(xt) - c.q_cqr, c.hi_m.predict(xt) + c.q_cqr
    res: dict[str, list] = {m: [] for m in ("P2-0 naive", "P2-B direct", "P2-B-CQR", "P2-A decomposition")}
    for i, row in enumerate(st.itertuples()):
        pieces = pt[pt.case == row.case]
        lo0 = sum(v * np.exp(-c.q_s[k]) for v, k in zip(pieces.volume_mm3, pieces.stratum, strict=True))
        hi0 = sum(v * np.exp(c.q_s[k]) for v, k in zip(pieces.volume_mm3, pieces.stratum, strict=True))
        res["P2-0 naive"].append((row.case, row.B, lo0, hi0))
        res["P2-B direct"].append((row.case, row.B, max((row.B_hat + C_MM3) * np.exp(-c.q_b) - C_MM3, 0.0), (row.B_hat + C_MM3) * np.exp(c.q_b) - C_MM3))
        res["P2-B-CQR"].append((row.case, row.B, max(np.exp(cqr_lo[i]) - C_MM3, 0.0), np.exp(cqr_hi[i]) - C_MM3))
        r = raw_test[row.case]
        res["P2-A decomposition"].append((row.case, row.B, max((r[0] + C_MM3) * np.exp(-c.t_fac) - C_MM3, 0.0), (r[1] + C_MM3) * np.exp(c.t_fac) - C_MM3))
    return {m: pd.DataFrame(v, columns=["case", "B", "lo", "hi"]) for m, v in res.items()}


def burden_intervals(fit: Tables, cal: Tables, test: Tables, alpha: float, filtered: bool = True,
                     fp_features: list[str] | None = None) -> dict[str, pd.DataFrame]:  # fmt: skip
    """Intervals for the TEST scans (fit_burden then apply_burden). Kept for the development analyses."""
    return apply_burden(fit_burden(fit, cal, alpha, filtered, fp_features), test)


# ---------------------------------------------------------------- P2 formal test (OD-1)
def p2_test(iv: dict[str, pd.DataFrame], nominal: float = 0.90, seed: int = SEED, resamples: int = BOOT) -> dict:
    """OD-1: single formal test = paired median width ratio P2-A / P2-B-CQR with a 95% scan-level paired
    bootstrap CI excluding 1, AND P2-A coverage CP 95% upper bound >= nominal. Everything else descriptive."""
    a, q = iv["P2-A decomposition"], iv["P2-B-CQR"]
    if list(a.case) != list(q.case):
        raise ValueError("methods must cover the same scans in the same order")
    ratio = (a.hi - a.lo).to_numpy() / np.maximum((q.hi - q.lo).to_numpy(), 1e-9)
    boot = np.random.default_rng(seed).integers(0, len(ratio), size=(resamples, len(ratio)))
    meds = np.median(ratio[boot], axis=1)
    ci = [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]
    out = {"width_ratio_A_over_BCQR": {"median": float(np.median(ratio)), "ci95": ci}}
    for m, d in iv.items():
        cov = ((d.B >= d.lo) & (d.B <= d.hi)).to_numpy()
        out[m] = {"n": len(d), "coverage": float(cov.mean()), "cp95": list(cf.clopper_pearson(int(cov.sum()), len(cov))),
                  "median_width_mm3": float(np.median(d.hi - d.lo)),
                  "coverage_stone_free": float(cov[(d.B == 0).to_numpy()].mean()) if (d.B == 0).any() else None}  # fmt: skip
    ca, cq = ((a.B >= a.lo) & (a.B <= a.hi)).to_numpy(), ((q.B >= q.lo) & (q.B <= q.hi)).to_numpy()
    diff = (ca.astype(float) - cq.astype(float))[boot].mean(axis=1)
    out["paired_coverage_difference_A_minus_BCQR"] = {"estimate": float(ca.mean() - cq.mean()), "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]}
    guard = out["P2-A decomposition"]["cp95"][1] >= nominal
    out["formal"] = {"ratio_ci_excludes_1_below": ci[1] < 1.0, "coverage_check_cp_upper_ge_nominal": bool(guard),
                     "P2_success": bool(ci[1] < 1.0 and guard)}  # fmt: skip
    return out
