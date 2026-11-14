"""Structured, print-ready study report (HTML; the browser's Print saves it as PDF). Built only from the saved
study data returned by services.study(); it calculates nothing. No diagnosis and no recommendation is given."""

import html
from datetime import UTC, datetime

DISCLAIMER = ("Research use. Values are automated research measurements; ranges come from the frozen research "
              "calibration of the StoneKit study (nominal 90%).")  # fmt: skip


def _e(v) -> str:
    return html.escape(str(v))


def _f(v, d=1, unit="") -> str:
    return "—" if v is None else f"{v:,.{d}f}{unit}"


def _when(stamp) -> str:
    """'20261114T094015Z' or an ISO time -> '2026-11-14 09:40 UTC'."""
    t = "".join(ch for ch in str(stamp or "") if ch.isdigit())
    return f"{t[0:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:{t[10:12]} UTC" if len(t) >= 12 else "—"


def render(s: dict, embedded: bool = False) -> str:
    r, rg = s["report"], s["ranges"] or {}
    chk = r.get("input_checks") or {}
    sp = chk.get("spacing_mm") or []
    k = r.get("kidney") or {}
    b = rg.get("burden") or {}
    dec = b.get("decomposition") or {}
    stones = s["stones"]
    rows = "".join(
        f"<tr><td>{st['stone_id']}</td><td>{_e(st['location']['side'])}<br><span class='m'>{_e(st['location']['position'])}</span></td>"
        f"<td class='n'>{_f(st['volume_mm3'], 0)}</td><td class='n'>{_f(st['range']['volume_range_mm3'][0], 0) + '–' + _f(st['range']['volume_range_mm3'][1], 0) if st['range'] else '—'}</td>"
        f"<td class='n'>{_f(st['max_diameter_3d_mm'])}</td><td class='n'>{_f(st['range']['diameter_range_mm'][0]) + '–' + _f(st['range']['diameter_range_mm'][1]) if st['range'] else '—'}</td>"
        f"<td>{_e(st['size_band'])}</td><td class='n'>{_f(st['range']['false_detection_probability'], 2) if st['range'] else '—'}</td><td>{_e(st['qc_status'])}</td></tr>"
        for st in stones
    )  # fmt: skip
    qc = "".join(f"<tr><td>{_e(i['group'])}</td><td>{_e(i['item'])}</td><td><b class='s {i['status'].replace(' ', '')}'>{_e(i['status'])}</b></td><td>{_e(i['detail'])}</td></tr>" for i in s["qc_items"])
    integ = s["integrity"]
    burden = (f"<p><b>Total measured burden:</b> {_f(b.get('predicted_mm3'), 0)} mm³ &nbsp; <b>Detection-aware interval (90%):</b> "
              f"{_f(b['interval_mm3'][0], 0)} – {_f(b['interval_mm3'][1], 0)} mm³</p>"
              f"<p class='m'>Retained pieces {dec.get('retained_pieces', '—')}; expected volume after false-detection modelling "
              f"{_f(dec.get('expected_after_false_detection_mm3'), 0)} mm³; raw Monte Carlo interval {_f(dec.get('raw_interval_mm3', [None])[0], 0)} – "
              f"{_f((dec.get('raw_interval_mm3') or [None, None])[1], 0)} mm³; conformal factor t = {_f(dec.get('conformal_factor_t'), 3)}. "
              f"Comparator (conformalised quantile regression): {_f(b['comparators']['P2-B-CQR'][0], 0)} – {_f(b['comparators']['P2-B-CQR'][1], 0)} mm³.</p>"
              if b else "<p>Calibrated ranges were not computed for this study.</p>")  # fmt: skip
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>StoneKit report · {_e(r.get('case_id'))}</title>
<style>
@page{{size:A4;margin:14mm 13mm}}*{{box-sizing:border-box}}
body{{font:10.5px/1.45 "Segoe UI","Helvetica Neue",Arial,sans-serif;color:#17222b;margin:0;background:#eef1f3}}
.page{{max-width:820px;margin:18px auto;background:#fff;padding:28px 32px;border:1px solid #dfe5ea}}
@media print{{body{{background:#fff}}.page{{margin:0;border:0;padding:0;max-width:none}}.noprint{{display:none}}}}
header{{display:flex;justify-content:space-between;border-bottom:2px solid #0f5e78;padding-bottom:8px;margin-bottom:12px}}
h1{{font-size:16px;margin:0;font-weight:600}}h2{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#0f5e78;margin:16px 0 6px;border-bottom:1px solid #dfe5ea;padding-bottom:3px}}
.m{{color:#5b6874}}table{{width:100%;border-collapse:collapse;font-size:10px}}th,td{{text-align:left;padding:4px 5px;border-bottom:1px solid #e8ecef;vertical-align:top}}
th{{color:#5b6874;font-weight:600;background:#f6f8f9}}.n{{text-align:right;font-variant-numeric:tabular-nums}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:4px 16px}}.grid div b{{display:block;color:#5b6874;font-weight:500;font-size:9.5px}}
.s{{font:600 9px ui-monospace,Consolas,monospace;padding:0 4px;border:1px solid}}.PASS{{color:#2e7d55}}.REVIEW,.WARNING{{color:#a35f00}}.FAIL{{color:#b3261e}}.NOTAVAILABLE{{color:#5b6874}}
.disc{{margin-top:16px;padding:8px 10px;border:1px solid #dfe5ea;background:#f6f8f9;font-size:9.5px;color:#3b4853}}
.btn{{font:12px "Segoe UI",Arial;padding:6px 12px;border:1px solid #0f5e78;background:#0f5e78;color:#fff;cursor:pointer}}
</style></head><body>{'' if embedded else '<div class="noprint" style="max-width:820px;margin:12px auto 0;text-align:right"><button class="btn" onclick="print()">Print / Save as PDF</button></div>'}
<div class="page">
<header><div><h1>Kidney Stone Analysis Report</h1><div class="m">StoneKit research workstation · automated non-contrast CT analysis</div></div>
<div class="m" style="text-align:right">Study <b style="color:#17222b">{_e(r.get('case_id'))}</b><br>Analysed {_when(r.get('created_utc'))}<br>Report generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}</div></header>
<h2>Study</h2><div class="grid">
<div><b>Study identifier</b>{_e(r.get('case_id'))}</div><div><b>Matrix</b>{' × '.join(map(str, chk.get('shape') or [])) or '—'}</div>
<div><b>Voxel spacing</b>{' × '.join(f'{v:.2f}' for v in sp) + ' mm' if sp else '—'}</div><div><b>Analysis eligibility</b>{_e(s['eligibility']['status'])}</div>
<div><b>Contrast check</b>{_f((r.get('contrast_check') or {}).get('aorta_median_hu'), 0, ' HU (aorta median)')}</div><div><b>Overall QC</b>{_e(s['qc'])}</div>
<div><b>Pipeline configuration</b>{_e((r.get('config_sha256') or '')[:12])}…</div><div><b>Calibration</b>{_e((integ.get('calibration_sha256') or '')[:12])}…</div></div>
<h2>Kidneys (segmentation)</h2><p>Right kidney segmentation volume {_f(k.get('right_volume_ml'), 1, ' mL')}; left kidney {_f(k.get('left_volume_ml'), 1, ' mL')}. Kidney region of interest with a {_e((r.get('roi') or {}).get('margin_mm', '—'))} mm margin.</p>
<h2>Stones ({len(stones)} retained after the 30 mm kidney-distance filter)</h2>
<table><tr><th>ID</th><th>Location (segmentation)</th><th class="n">Volume mm³</th><th class="n">90% range</th><th class="n">3D max. diameter mm</th><th class="n">90% range</th><th>Size band</th><th class="n">P(false det.)</th><th>QC</th></tr>{rows or '<tr><td colspan=9>No stone retained.</td></tr>'}</table>
<h2>Patient-level stone burden</h2>{burden}
<h2>Quality control and provenance</h2><table><tr><th>Group</th><th>Step</th><th>Status</th><th>Detail</th></tr>{qc}</table>
<p class="m">Frozen artefacts verified against the T1 freeze manifest: {integ['summary']}. {_e(integ['protocol'])}.</p>
<h2>Method</h2><p class="m">TotalSegmentator kidney segmentation; 40 mm kidney region; nnU-Net v2 stone segmentation (3D full resolution, fold 0, 100 epochs) with 8-flip test-time augmentation; 26-connected pieces of at least 3 voxels; pieces more than 30 mm from a kidney removed; volume from voxels and maximum 3D diameter as the largest voxel-centre distance. Per-stone ranges: size-stratified split conformal (Mondrian) on log ratios. Burden interval: Monte Carlo propagation of false-detection probabilities, size-band measurement residuals and missed burden, followed by one conformal factor. Calibration on 35 scans of Li 2022; evaluated once on 53 locked scans (Kadari, Surepalli and Kolisetty, 2026).</p>
<div class="disc">{DISCLAIMER} The report states measurements and intervals only; it contains no interpretation.</div>
</div></body></html>"""
