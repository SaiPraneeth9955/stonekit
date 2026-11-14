"""One-page HTML report, built only from the JSON report. It calculates nothing itself.

`stone_table` is shared with the app, so the report and the screen show the same values.
"""

import base64
import html
import io
from pathlib import Path

import numpy as np

TABLE_COLUMNS = [
    "Stone ID",
    "Volume (mm3)",
    "Volume (mL)",
    "3D diameter (mm)",
    "Axial diameter (mm)",
    "Extent x, y, z (mm)",
    "Mean HU",
    "Median HU",
    "Peak HU",
    "Voxels",
    "QC flags",
]


def stone_table(method: dict) -> list[dict]:
    """Display rows for one method, formatted from the stored values. No new calculation."""
    rows = []
    for stone in method.get("stones", []):
        extent = (stone["extent_x_mm"], stone["extent_y_mm"], stone["extent_z_mm"])
        rows.append(
            {
                "Stone ID": stone["stone_id"],
                "Volume (mm3)": f"{stone['volume_mm3']:.1f}",
                "Volume (mL)": f"{stone['volume_ml']:.3f}",
                "3D diameter (mm)": f"{stone['max_diameter_3d_mm']:.1f}",
                "Axial diameter (mm)": f"{stone['max_diameter_axial_mm']:.1f}",
                "Extent x, y, z (mm)": ", ".join(f"{v:.1f}" for v in extent),
                "Mean HU": f"{stone['mean_hu']:.0f}",
                "Median HU": f"{stone['median_hu']:.0f}",
                "Peak HU": f"{stone['max_hu']:.0f}",
                "Voxels": str(stone["voxel_count"]),
                "QC flags": ", ".join(stone["qc_flags"]) or "none",
            }
        )
    return rows


def axial_display(ras_slice: np.ndarray) -> np.ndarray:
    """Axial slice for display with origin='lower': patient's right on the image's left."""
    return ras_slice[::-1, :].T


def _rows_html(rows: list[dict]) -> str:
    if not rows:
        return "<p>No stones reported by this method.</p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in TABLE_COLUMNS)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in TABLE_COLUMNS) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _largest_stone_figure(report: dict, out_dir: Path) -> str:
    """An axial CT slice through the primary method's largest stone, drawn from saved files."""
    method = report.get("methods", {}).get(report.get("primary_method"), {})
    stones = method.get("stones") or []
    if not stones:
        return ""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from stonekit.io.nifti import load_volume

    ct = load_volume(Path(report["ct_path"]))
    labels = load_volume(out_dir / method["label_file"])
    z = (stones[0]["voxel_start"][2] + stones[0]["voxel_stop"][2] - 1) // 2
    image = axial_display(ct.data[:, :, z])
    overlay = axial_display((labels.data[:, :, z] > 0).astype(float))
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(image, cmap="gray", vmin=-600, vmax=1200, origin="lower")
    axis.imshow(np.ma.masked_where(overlay == 0, overlay), cmap="autumn", alpha=0.6, origin="lower")
    axis.set_title(f"Largest stone, slice {z} (image left = patient right)", fontsize=9)
    axis.axis("off")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="CT slice with stone overlay" src="data:image/png;base64,{encoded}">'


def model_label(report: dict) -> str:
    """Readable model name from the recorded config, e.g. 'nnU-Net (100 epochs, fold 0)'."""
    model = (report.get("config") or {}).get("model") or {}
    trainer = str(model.get("trainer", ""))
    epochs = trainer.replace("nnUNetTrainer_", "").replace("epochs", " epochs") if trainer else "trainer unknown"
    return f"nnU-Net ({epochs}, fold {model.get('fold', '?')})"


def method_title(name: str, report: dict) -> str:
    if name == "model":
        return f"Trained model: {model_label(report)}"
    if name == "threshold_272hu":
        return "272 HU threshold (locked comparator, not a trained model)"
    return name


def _list(items) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(str(i))}</li>" for i in items) + "</ul>" if items else "<p>None.</p>"


def build_html(report: dict, figure: str) -> str:
    sections = [
        f'<p class="banner">{html.escape(report["disclaimer"])}<br>{html.escape(report["status_line"])}</p>',
        f"<h1>StoneKit case report: {html.escape(report['case_id'])}</h1>",
        f"<p>{html.escape(report['uncertainty_status'])}</p>",
        f"<p>{html.escape(report['qc_note'])}</p>",
    ]
    status = report.get("result_status")
    if status:
        sections.append(f"<p>Result status: <b>{html.escape(status)}</b>.</p>")
    if report.get("failure"):
        sections.append(f"<h2>Processing failure</h2><p>{html.escape(report['failure'])}</p>")
    sections.append("<h2>Case QC flags</h2>" + _list(report["case_qc_flags"]))
    checks = report.get("input_checks")
    if checks:
        sections.append(
            "<h2>Input checks</h2>"
            f"<p>Shape {checks['shape']}, spacing (mm) {[round(s, 4) for s in checks['spacing_mm']]}, "
            f"file orientation {checks['original_orientation']}, HU {checks['hu_min']:.0f} to {checks['hu_max']:.0f}.</p>"
            + _list(f"{c['name']}: {c['status']} ({c['detail']})" for c in checks["checks"])
        )
    sections.append("<h2>Preprocessing</h2>" + _list(report["preprocessing"]))
    if report.get("kidney"):
        sections.append(
            "<h2>Kidney segmentation</h2>"
            f"<p>Right {report['kidney']['right_volume_ml']:.1f} mL, left {report['kidney']['left_volume_ml']:.1f} mL.</p>"
        )
    if report.get("contrast_check"):
        hu = report["contrast_check"]["aorta_median_hu"]
        shown = "not available" if hu is None else f"{hu:.0f} HU"
        sections.append(f"<p>Aorta median: {shown}. {html.escape(report['contrast_check']['note'])}</p>")
    if report.get("roi"):
        size = ", ".join(f"{v:.0f}" for v in report["roi"]["box_size_mm"])
        sections.append(f"<h2>Kidney ROI</h2><p>Box {size} mm, margin {report['roi']['margin_mm']} mm.</p>")
    if figure:
        sections.append(figure)
    post = report.get("postprocessing")
    if post:
        limit = post.get("model_kidney_distance_filter_mm")
        shown = f"{limit:g} mm, model method only" if limit is not None else "not applied"
        sections.append(f"<h2>Post-processing</h2><p>Kidney-distance filter: {html.escape(shown)}. {html.escape(post['note'])}</p>")
    for name, method in report["methods"].items():
        title = method_title(name, report)
        if not method.get("available"):
            sections.append(f"<h2>{html.escape(title)}</h2><p>Not available on this machine. No result is shown.</p>")
            continue
        removed = (method.get("kidney_distance_filter") or {}).get("removed_pieces")
        filter_note = f" Pieces removed by the kidney-distance filter: {len(removed)}." if removed is not None else ""
        sections.append(
            f"<h2>{html.escape(title)}</h2><p>Total {method['total_volume_mm3']:.1f} mm3 "
            f"({method['total_volume_ml']:.3f} mL). Tiny pieces dropped: {method['dropped_tiny_pieces']}.{filter_note} "
            f"Flags: {html.escape(', '.join(method['qc_flags']) or 'none')}.</p>" + _rows_html(stone_table(method))
        )
    comparison = report.get("annotation_comparison")
    if comparison:
        rows = "".join(
            f"<li>{html.escape(method_title(name, report))}: Dice "
            f"{'n/a' if m['dice'] is None else format(m['dice'], '.3f')}, predicted stones "
            f"{m['predicted_stones']}, annotation pieces {m['annotation_components']}</li>"
            for name, m in comparison["methods"].items()
        )
        sections.append(f"<h2>Comparison with annotation</h2><p>{html.escape(comparison['note'])}</p><ul>{rows}</ul>")
    stages = report.get("stages", [])
    sections.append(
        "<h2>Processing</h2>"
        + _list(f"{s['stage']}: {s['status']}, {s['seconds']} s. {s['detail']}" for s in stages)
        + f"<p>Total {report.get('total_seconds')} s. Peak memory (kB): {html.escape(str(report.get('peak_memory_kb')))}.</p>"
        + f"<p>Software: {html.escape(str(report['software']))}.</p>"
        + f"<p>Config file {html.escape(str(report.get('config_file', 'not recorded')))}, SHA-256 {report['config_sha256']}. "
        f"Configuration actually used, SHA-256 {html.escape(str(report.get('effective_config_sha256', 'not recorded')))}.</p>"
        + (
            f"<p>Checkpoint SHA-256 {html.escape(report['model_info']['checkpoint_sha256'])}.</p>"
            if report.get("model_info")
            else ""
        )
    )
    style = (
        "body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;margin:32px auto;max-width:1080px;color:#1d232a;"
        "font-size:14px;line-height:1.5}"
        "h1{font-size:20px;font-weight:600;border-bottom:2px solid #1d232a;padding-bottom:6px}"
        "h2{font-size:15px;font-weight:600;margin-top:26px;border-bottom:1px solid #c9ced4;padding-bottom:3px}"
        ".banner{border:1px solid #8a6d1f;background:#faf6ea;padding:8px 10px;font-size:13px}"
        "table{border-collapse:collapse;font-size:12.5px;margin-top:6px}"
        "th{background:#eef1f4;font-weight:600;border-bottom:1px solid #9aa4ae}"
        "td,th{border:1px solid #d5dade;padding:4px 8px;text-align:left}"
        "td{font-variant-numeric:tabular-nums}img{max-width:460px;border:1px solid #c9ced4}"
        "ul{padding-left:20px}"
        # Print (browser "Save as PDF"): plain black text, no page-splitting inside tables or figures.
        "@media print{body{margin:12mm;max-width:none;font-size:11px}.banner{background:none}"
        "table,img{page-break-inside:avoid}h2{page-break-after:avoid}}"
    )
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>StoneKit {html.escape(report["case_id"])}</title>'
        f"<style>{style}</style></head><body>{''.join(sections)}</body></html>"
    )


def write_html_report(report: dict, path: Path, out_dir: Path) -> None:
    """Write the page. If the figure cannot be drawn, the page says so instead of failing."""
    try:
        figure = _largest_stone_figure(report, out_dir)
    except Exception as error:  # noqa: BLE001 - the figure is optional; report.json is already written
        figure = f"<p>Figure not drawn ({html.escape(type(error).__name__)}).</p>"
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(build_html(report, figure))
