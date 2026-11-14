"""External check support (protocol v3 §5.4): rebuild the T1 calibration from its HASHED quantities file.

After T1 completes, the reserved calibration scans are locked again, so the external check cannot refit. It
rebuilds every interval method from `calibration_quantities.json` (written and hashed by the T1 analysis stage)
plus the development fit table, whose residual and missed-burden pools are re-derived and checked against the
hashes stored in that file. Nothing is refitted or tuned on external data. The frozen T1 modules are not
changed; this module only reads their outputs.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from stonekit.uq import confirmatory as cfm


@dataclass(frozen=True)
class _Linear:
    """Stands in for a fitted statsmodels QuantReg result: predict(x) = x @ params."""

    params: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, float) @ self.params


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, float)).tobytes()).hexdigest()


def load_quantities(analysis_dir: Path) -> dict:
    """Read calibration_quantities.json and verify it against the SHA-256 the T1 run recorded."""
    q_path = Path(analysis_dir) / "calibration_quantities.json"
    recorded = (Path(analysis_dir) / "calibration_quantities.sha256").read_text().split()[0]
    if hashlib.sha256(q_path.read_bytes()).hexdigest() != recorded:
        raise ValueError("calibration_quantities.json does not match its recorded SHA-256")
    return json.loads(q_path.read_text(encoding="utf-8"))


def rebuild_burden(q: dict, fit: cfm.Tables) -> cfm.BurdenCalibration:
    """Rebuild a BurdenCalibration from one `burden` entry of the quantities file and the development fit table."""
    _, sf, mf = cfm._prepare(fit, bool(q["filtered"]))
    e_fit = np.log(mf.ref_volume_mm3 / mf.pred_volume_mm3).to_numpy()
    s_fit = cfm.stratum(mf.pred_max_diameter_3d_mm)
    pool_e = {k: e_fit[s_fit == k] for k in range(len(cfm.EDGES) + 1)}
    for k, v in pool_e.items():
        ref = q["residual_pools"][str(k)]
        if len(v) != ref["n"] or _h(v) != ref["sha256"]:
            raise ValueError(f"residual pool {k} differs from the hashed T1 pool")
    missed = sf[["case", "slice_group", "missed_mm3"]].copy()
    if len(missed) != q["missed_pool"]["n"] or _h(missed["missed_mm3"].to_numpy()) != q["missed_pool"]["sha256"]:
        raise ValueError("missed-burden pool differs from the hashed T1 pool")
    model = LogisticRegression()
    model.coef_ = np.asarray(q["fp_model"]["coef"], float)
    model.intercept_ = np.asarray(q["fp_model"]["intercept"], float)
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = model.coef_.shape[1]
    model.feature_names_in_ = np.asarray(q["fp_features"], dtype=object)
    return cfm.BurdenCalibration(alpha=float(q["alpha"]), filtered=bool(q["filtered"]), feats=list(q["fp_features"]), model=model,
                                 base_rate=float(q["fp_model"]["base_rate"]), e_fit=e_fit, pool_e=pool_e, missed_fit=missed,
                                 q_s={int(k): float(v) for k, v in q["p2_0_q_s"].items()}, q_b=float(q["p2_b_q"]),
                                 lo_m=_Linear(np.asarray(q["cqr"]["lo_params"], float)), hi_m=_Linear(np.asarray(q["cqr"]["hi_params"], float)),
                                 q_cqr=float(q["cqr"]["q"]), t_fac=float(q["p2_a_t"]))  # fmt: skip


def stone_bounds(params: dict, method: str, target: str, alpha: float, pred_d3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-stone log-ratio bounds from the hashed T1 per-stone parameters (U0, U1, U2, U2-asymmetric)."""
    key = f"{target}|{alpha}"
    n = len(pred_d3)
    if method == "U0":
        lo, hi = params["u0"][key]
        return np.full(n, lo), np.full(n, hi)
    if method == "U1":
        q = params["u1"][key]
        return np.full(n, -q), np.full(n, q)
    if method == "U2":
        p = params["u2"][key]
        q = np.asarray(p["q"], float)[np.searchsorted(p["edges"], np.asarray(pred_d3, float), side="right")]
        return -q, q
    if method == "U2-asymmetric":
        p = params["asym"][key]
        g = np.searchsorted(p["edges"], np.asarray(pred_d3, float), side="right")
        return np.asarray(p["lo"], float)[g], np.asarray(p["hi"], float)[g]
    raise ValueError(f"{method} cannot be rebuilt from the quantities file")
