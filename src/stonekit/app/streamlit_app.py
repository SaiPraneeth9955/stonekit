"""StoneKit research workstation: a local viewer for runs written by `stonekit run-case`.

It shows saved outputs only and calculates no measurement. Stone tables come from report.json through
the same function as the HTML report. Unusual runs (failed, incomplete, corrupted, no stones, missing
files) are shown as what they are, and never crash the page or pass as results.
Research use only. Not a diagnostic medical device.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from stonekit.app.viewer import viewer_html, viewer_payload
from stonekit.data.registry import find_project_root
from stonekit.io.nifti import load_volume
from stonekit.measure.mesh import read_obj
from stonekit.pipeline.report_html import (
    TABLE_COLUMNS,
    axial_display,
    method_title,
    model_label,
    stone_table,
)
from stonekit.seg.kidney_roi import KIDNEY_LEFT, KIDNEY_RIGHT, kidney_mask

MAX_MESHES = 30
# Overlay colours each mean one thing: kidney contour, stone fill, selected stone, ROI box.
KIDNEY_RGB = (86, 182, 194)
STONE_RGB = (217, 164, 65)
SELECTED_RGB = (224, 69, 58)
ROI_RGB = (120, 160, 120)
WINDOWS = {"Stone (W 2000 / L 500)": (500, 2000), "Bone (W 1500 / L 300)": (300, 1500), "Soft tissue (W 400 / L 40)": (40, 400)}
FLAG_TEXT = {
    "very_small_component": "Very small piece: near the limit the scan can resolve.",
    "unusually_large_component": "Unusually large piece: check for merged stones or non-stone tissue.",
    "measurement_warning_one_slice_only": "Seen on one slice only: 3D size is poorly defined.",
    "measurement_warning_size_near_voxel_size": "Diameter under three voxels: size is dominated by voxel size.",
    "touches_roi_boundary": "Touches the kidney ROI box edge: the piece may be cut.",
    "outside_kidney_segmentation": "More than 5 mm from the kidney segmentation.",
    "suspicious_long_thin_shape_possible_device": "Long thin shape: possibly a stent or other device.",
    "suspicious_very_high_density": "Very high peak HU: possibly metal.",
    "irregular_shape_possible_merged_stones": "Irregular shape: possibly two stones joined.",
    "no_stone_detected": "No stone piece found by this method.",
    "many_components": "Many pieces: check for false positives.",
    "tiny_components_dropped": "Pieces under the minimum voxel count were dropped.",
    "no_kidney_found": "No kidney found: no ROI and no stone result.",
    "only_one_kidney_found": "Only one kidney found: the ROI covers that side only.",
    "roi_reaches_image_edge": "The ROI box reaches the image edge.",
    "contrast_suspected_bright_aorta": "Bright aorta: contrast may be present. Heuristic only.",
    "contrast_check_not_possible_no_aorta_found": "No aorta found, so the contrast heuristic could not run.",
    "processing_failure": "Processing failed: no stone result exists for this run.",
    "kidney_segmentation_failed": "Kidney segmentation failed; see the failure details.",
    "trained_model_not_available": "No trained model checkpoint was available. The threshold is a comparator, not a substitute.",
    "stone_segmentation_failed": "The trained model's prediction failed or was invalid; see the failure details.",
    "unexpected_error": "An unexpected error stopped processing; see the failure details.",
    "kidney_segmentation_low_memory_fallback": "The normal kidney step ran out of memory; the kidney mask comes from the validated "
    "low-memory fallback (slices resampled to 1.5 mm for the kidney step only). Stones are still measured on original voxels.",
    "pieces_removed_by_kidney_distance_filter": "The kidney-distance filter removed pieces; they may include real stones far from the "
    "kidney (for example ureteric). See Measurements and the saved removed-pieces label map.",
}
CSS = """
<style>
:root {--sk-ink:#1f2933; --sk-muted:#5f6b7a; --sk-rule:#d5dae0; --sk-accent:#1f5fae; --sk-pass:#2f7d4f; --sk-review:#9a6200; --sk-fail:#b42318;}
.block-container {padding-top: 0.8rem; padding-bottom: 1.5rem; max-width: 1680px;}
h1, h2, h3 {font-weight: 600; letter-spacing: 0; color: var(--sk-ink);}
[data-testid="stMarkdownContainer"] h3, h3 {font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: .05em;
   color: var(--sk-muted) !important; margin: 0.9rem 0 0.3rem 0; padding: 0 0 3px 0 !important; border-bottom: 1px solid var(--sk-rule);}
.sk-bar {display:flex; justify-content:space-between; align-items:baseline; border-bottom:1px solid var(--sk-rule); padding-bottom:6px; margin-bottom:6px;}
.sk-title {font-size:1.05rem; font-weight:600; color:var(--sk-ink);} .sk-sub {font-size:0.85rem; color:var(--sk-muted); margin-left:8px;}
.sk-meta {font-size:0.78rem; color:var(--sk-muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;}
.sk-notice {font-size:0.78rem; color:var(--sk-muted); padding:2px 0 8px 0;}
.sk-notice b {color:var(--sk-ink); font-weight:600;}
.sk-study {display:flex; gap:22px; flex-wrap:wrap; font-size:0.8rem; color:var(--sk-muted); border:1px solid var(--sk-rule); background:#fff; padding:6px 10px; margin-bottom:8px;}
.sk-study span b {color:var(--sk-ink); font-weight:600; margin-left:4px;}
.sk-kv {font-size:0.84rem; line-height:1.6;}
.sk-kv b {color:var(--sk-muted); font-weight:500; display:inline-block; min-width:190px;}
.sk-status {display:inline-block; font:600 0.75rem ui-monospace, Menlo, Consolas, monospace; padding:1px 7px; border:1px solid; letter-spacing:.04em;}
.sk-PASS {color:var(--sk-pass); border-color:var(--sk-pass);} .sk-REVIEW {color:var(--sk-review); border-color:var(--sk-review);}
.sk-FAILED {color:var(--sk-fail); border-color:var(--sk-fail);} .sk-NOTE {color:var(--sk-muted); border-color:var(--sk-rule);}
.sk-flag {font-size:0.82rem; margin:2px 0; padding-left:8px; border-left:2px solid var(--sk-review); color:var(--sk-ink);}
div[data-testid="stDataFrame"] {border:1px solid var(--sk-rule);}
</style>
"""
SIZE_EDGES_MM = (5.0, 10.0)
# QC status (research heuristic): FAILED = processing failed or no result; REVIEW = any case or stone QC flag; PASS = none.
FAILED_FLAGS = {"processing_failure", "kidney_segmentation_failed", "stone_segmentation_failed", "unexpected_error", "no_kidney_found"}
UNCERTAINTY_TEXT = ("Not available. The per-stone uncertainty method is still being designed on development data and has "
                    "not been calibrated on the reserved calibration cohort. No interval is shown, so no unvalidated range "
                    "can be mistaken for a calibrated one.")


RANGES_FILE = "calibrated_ranges.json"  # written by scripts/demo_case.py (stonekit.t1.case_ranges)


def load_ranges(folder: Path) -> dict | None:
    """Calibrated ranges saved next to report.json, or None. An unreadable file is reported, never guessed."""
    path = folder / RANGES_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st.error(f"{RANGES_FILE} could not be read; no ranges are shown.")
        return None


def size_category(d_mm: float) -> str:
    """Research size bins by maximum 3D diameter (not a clinical classification)."""
    return "<5 mm" if d_mm < SIZE_EDGES_MM[0] else "5-<10 mm" if d_mm < SIZE_EDGES_MM[1] else ">=10 mm"


def qc_status(report: dict, stones: list[dict]) -> str:
    flags = set(report.get("case_qc_flags", []))
    if report.get("failure") or flags & FAILED_FLAGS or report.get("result_status") in ("failed", "no_result"):
        return "FAILED"
    if flags or any(s.get("qc_flags") for s in stones):
        return "REVIEW"
    return "PASS"


def status_badge(status: str) -> str:
    return f"<span class='sk-status sk-{status}'>{status}</span>"


@st.cache_data(max_entries=4, show_spinner="Preparing views")
def viewer_page(folder: str, ct_path: str, label_file: str | None, selected: int | None, height: int) -> str:
    """Tri-planar viewer HTML for the kidney ROI crop of one run (saved outputs only)."""
    ct, kidneys, labels = load_volumes(folder, ct_path, label_file)
    report = json.loads((Path(folder) / "report.json").read_text(encoding="utf-8"))
    box = tuple(slice(a, b) for a, b in zip(report["roi"]["box_start_voxel"], report["roi"]["box_stop_voxel"], strict=True))
    stone_ids = labels.data[box].astype(np.int32) if labels is not None else None
    focus = None
    if selected is not None and stone_ids is not None:
        hit = np.argwhere(stone_ids == selected)
        if len(hit):
            focus = [int(v) for v in hit.mean(axis=0).round()]
    payload = viewer_payload(ct.data[box], ct.spacing_mm, kidney_mask(kidneys.data[box]), stone_ids, selected, focus,
                             title=report.get("case_id", ""))  # fmt: skip
    return viewer_html(payload, height)



@st.cache_data
def load_report(path: str, modified: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(max_entries=1)
def load_volumes(folder: str, ct_path: str, label_file: str | None):
    ct = load_volume(Path(ct_path))
    kidneys = load_volume(Path(folder) / "kidney_totalsegmentator.nii.gz")
    labels = load_volume(Path(folder) / label_file) if label_file else None
    return ct, kidneys, labels


@st.cache_data
def load_mesh(path: str):
    return read_obj(Path(path))


def discover(runs_folder: Path) -> tuple[list[Path], list[Path], list[tuple[Path, str]]]:
    """Valid runs, incomplete runs (no report.json) and corrupted runs (report.json unreadable)."""
    valid, incomplete, corrupted = [], [], []
    for path in [*runs_folder.glob("*"), *runs_folder.glob("*/*")]:
        if not path.is_dir():
            continue
        report = path / "report.json"
        if report.exists():
            try:
                json.loads(report.read_text(encoding="utf-8"))
                valid.append(path)
            except (OSError, ValueError) as error:
                corrupted.append((path, type(error).__name__))
        elif (path.parent == runs_folder or path.name.startswith("wd")) and not any(
            (child / "report.json").exists() for child in path.iterdir() if child.is_dir()
        ):
            incomplete.append(path)
    return sorted(valid, key=run_time_key, reverse=True), sorted(incomplete), corrupted


def run_time_key(path: Path) -> tuple[str, str]:
    """Sort key: the UTC stamp that ends every run folder name (…_YYYYMMDDTHHMMSSZ), then the name."""
    stamp = path.name.rsplit("_", 1)[-1]
    return (stamp if len(stamp) == 16 and stamp.endswith("Z") else "", path.name)


def window(slice_hu: np.ndarray, level: float, width: float) -> np.ndarray:
    low = level - width / 2
    return (np.clip((slice_hu - low) / width, 0, 1) * 255).astype(np.uint8)


def outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow older than 10.1
        return ImageFont.load_default()


def compose_axial(ct, kidneys, labels, z, level, width, show_kidney, show_stones, box, selected,
                  crop_box=None, out_width: int = 900) -> Image.Image:  # fmt: skip
    """Axial CT slice in radiological convention: patient's right on the image left, anterior up.

    `crop_box` (the ROI record) limits the view to the kidney ROI with a 10-voxel border. Pixels are
    enlarged without interpolation, so every displayed value is an original voxel value.
    """
    base = window(axial_display(ct.data[:, :, z]), level, width)
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)
    if show_kidney:
        kidney = axial_display(np.isin(kidneys.data[:, :, z], [KIDNEY_RIGHT, KIDNEY_LEFT]))
        rgb[outline(kidney)] = KIDNEY_RGB
    shown = axial_display(labels.data[:, :, z]) if labels is not None and show_stones else None
    if shown is not None:
        stone = shown > 0
        rgb[stone] = 0.55 * rgb[stone] + 0.45 * np.array(STONE_RGB)
        rgb[outline(stone)] = STONE_RGB
        if selected:
            rgb[outline(shown == selected)] = SELECTED_RGB
    rgb = np.flipud(rgb)
    height, slice_width = rgb.shape[:2]

    def to_display(x_voxel: float, y_voxel: float) -> tuple[float, float]:
        """RAS voxel (x, y) to the flipped display pixel (column, row)."""
        return slice_width - 1 - x_voxel, height - 1 - y_voxel

    left, top, right, bottom = 0, 0, slice_width, height
    if crop_box is not None:
        c0, r0 = to_display(crop_box["box_stop_voxel"][0] - 1, crop_box["box_stop_voxel"][1] - 1)
        c1, r1 = to_display(crop_box["box_start_voxel"][0], crop_box["box_start_voxel"][1])
        left, top = max(int(c0) - 10, 0), max(int(r0) - 10, 0)
        right, bottom = min(int(c1) + 11, slice_width), min(int(r1) + 11, height)
    view = rgb[top:bottom, left:right]
    s = out_width / view.shape[1]
    image = Image.fromarray(view.astype(np.uint8)).resize((out_width, round(view.shape[0] * s)), Image.NEAREST)
    draw = ImageDraw.Draw(image)

    def pixel(column: float, row: float) -> tuple[float, float]:
        return (column - left) * s, (row - top) * s

    if box is not None and box["box_start_voxel"][2] <= z < box["box_stop_voxel"][2]:
        x0, y0 = pixel(*to_display(box["box_stop_voxel"][0] - 1, box["box_stop_voxel"][1] - 1))
        x1, y1 = pixel(*to_display(box["box_start_voxel"][0], box["box_start_voxel"][1]))
        x1, y1 = x1 + s - 1, y1 + s - 1
        for start in np.arange(x0, x1, 14):
            draw.line([start, y0, min(start + 7, x1), y0], fill=ROI_RGB)
            draw.line([start, y1, min(start + 7, x1), y1], fill=ROI_RGB)
        for start in np.arange(y0, y1, 14):
            draw.line([x0, start, x0, min(start + 7, y1)], fill=ROI_RGB)
            draw.line([x1, start, x1, min(start + 7, y1)], fill=ROI_RGB)
    label_font, marker_font = _font(15), _font(18)
    if shown is not None:
        flipped = np.flipud(shown)
        for stone_id in np.unique(flipped[flipped > 0]):
            rows, columns = np.nonzero(flipped == stone_id)
            x, y = pixel(int(columns.max()) + 2, float(rows.mean()))
            colour = SELECTED_RGB if stone_id == selected else (236, 236, 236)
            draw.text((x + 4, y - 8), str(int(stone_id)), fill=colour, font=label_font)
    w, h = image.size
    for text, xy in (("R", (10, h // 2 - 9)), ("L", (w - 22, h // 2 - 9)), ("A", (w // 2 - 6, 8)), ("P", (w // 2 - 6, h - 28))):
        draw.text(xy, text, fill=(214, 214, 214), font=marker_font)
    bar_mm = 10 if view.shape[1] * ct.spacing_mm[0] < 150 else 50
    bar = bar_mm / ct.spacing_mm[0] * s
    draw.line([w - 24 - bar, h - 18, w - 24, h - 18], fill=(225, 225, 225), width=3)
    draw.text((w - 24 - bar, h - 40), f"{bar_mm} mm", fill=(225, 225, 225), font=label_font)
    return image


def mesh_figure(folder: Path, report: dict, method: dict, selected: int | None, show_kidney: bool) -> go.Figure:
    figure = go.Figure()
    kidney_file = report.get("kidney_mesh_file")
    if show_kidney and kidney_file and (folder / kidney_file).exists():
        v, f = load_mesh(str(folder / kidney_file))
        figure.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                   color="#8fa3b8", opacity=0.15, name="Kidneys", hoverinfo="name", showlegend=True))  # fmt: skip
    for stone in method.get("stones", [])[:MAX_MESHES]:
        path = folder / stone["mesh_file"]
        if not path.exists():
            continue
        v, f = load_mesh(str(path))
        chosen = stone["stone_id"] == selected
        figure.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color="#3d8bfd" if chosen else "#e08a2e", opacity=1.0 if chosen or selected is None else 0.55,
            name=f"Stone {stone['stone_id']}", hovertext=f"Stone {stone['stone_id']}: {stone['volume_mm3']:.1f} mm³, "
            f"3D diameter {stone['max_diameter_3d_mm']:.1f} mm", hoverinfo="text", showlegend=False,
            flatshading=False, lighting={"ambient": 0.45, "diffuse": 0.8, "specular": 0.1},
        ))  # fmt: skip
    axis = {"backgroundcolor": "#f7f8fa", "gridcolor": "#d5dae0", "zerolinecolor": "#c3cad3", "color": "#5f6b7a"}
    # RAS millimetres: x increases toward the patient's Right, y toward Anterior, z toward Superior.
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=560, paper_bgcolor="#ffffff",
        font={"color": "#1f2933", "size": 11}, legend={"x": 0.01, "y": 0.99},
        scene={"aspectmode": "data", "xaxis": {**axis, "title": "L → R (mm)"}, "yaxis": {**axis, "title": "P → A (mm)"},
               "zaxis": {**axis, "title": "I → S (mm)"}},
    )  # fmt: skip
    return figure


def kv(pairs: list[tuple[str, object]]) -> None:
    st.markdown("<div class='sk-kv'>" + "<br>".join(f"<b>{k}</b>{v}" for k, v in pairs) + "</div>", unsafe_allow_html=True)


def method_label(name: str, report: dict) -> str:
    """The same names as the HTML report; the model's name comes from the recorded config."""
    if name == "model":
        return f"{model_label(report)} model"
    return method_title(name, report)


def flag_list(flags: list[str]) -> None:
    if not flags:
        st.caption("No flags.")
        return
    for flag in flags:
        st.markdown(f"- `{flag}`: {FLAG_TEXT.get(flag, 'No description recorded.')}")


def analysis_runs(root: Path) -> list[tuple[Path, dict]]:
    """Readable stone-level analysis runs, newest first. Unreadable folders are skipped."""
    found = []
    for run in sorted((root / "results" / "runs").glob("stone_analysis_*"), key=run_time_key, reverse=True):
        try:
            found.append((run, json.loads((run / "summary.json").read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return found


def analysis_method_label(name: str, summary: dict) -> str:
    """Readable name for a method key in an analysis summary; older summaries carry no labels."""
    labels = summary.get("method_labels") or {}
    if name in labels:
        return labels[name]
    fallback = {"model": summary.get("model_label", "nnU-Net model"), "threshold_272hu": "272 HU threshold"}
    return fallback.get(name, name)


def validation_tab(root: Path, report: dict) -> None:
    comparison = report.get("annotation_comparison")
    st.markdown("### This case against the Li annotation")
    if comparison:
        st.caption(comparison["note"])
        rows = [{"Method": method_label(n, report), "Dice": None if m["dice"] is None else round(m["dice"], 3),
                 "IoU": None if m["iou"] is None else round(m["iou"], 3), "Predicted pieces": m["predicted_stones"],
                 "Annotated pieces": m["annotation_components"], "Predicted voxels": m["predicted_voxels"],
                 "Annotated voxels": m["annotation_voxels"]} for n, m in comparison["methods"].items()]  # fmt: skip
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("No annotation comparison for this run: it is not a Li development scan, or no annotation was given.")
    found = analysis_runs(root)
    st.markdown("### Development evidence across validation scans")
    if not found:
        st.caption("No stone-level analysis run found under results/runs.")
        return
    names = [run.name for run, _ in found]
    chosen = st.selectbox("Analysis run", names, index=0, help="Saved stone-level analysis runs, newest first.")
    run, summary = found[names.index(chosen)]
    st.caption(f"{summary.get('status', '')}. Source: {run.name}. Annotations are references, not physical truth. "
               "Detected = any voxel overlap. Intervals: 95% bootstrap over scans.")  # fmt: skip
    rows = []
    for name, m in summary.get("methods", {}).items():
        d, a = m["detection_rate_all"], m["agreement_one_to_one"]
        rows.append({
            "Method": analysis_method_label(name, summary), "Detected pieces": f"{int(d['numerator'])}/{int(d['denominator'])} ({d['value']:.2f})",
            "False pieces per scan": m["false_positive"]["per_scan"],
            "<3 mm detected": _ratio(m["detection_by_size"].get("<3 mm")),
            "3-<5 mm detected": _ratio(m["detection_by_size"].get("3-<5 mm")),
            "Volume bias (% of mean)": _round(a["volume_percent_difference"].get("mean"), 1),
            "3D diameter bias (mm)": _round(a["diameter_3d_difference_mm"].get("mean"), 2),
            "One-to-one pairs": a["pairs"],
        })  # fmt: skip
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for figure in summary.get("figures", []):
        if (run / figure).exists():
            st.image(str(run / figure), width=760)


def _code_text(code: dict | None) -> str:
    if not code:
        return "not recorded (older run)"
    if not code.get("git_commit"):
        return "unknown (not a git checkout)"
    dirty = " with uncommitted changes" if code.get("uncommitted_changes") else ""
    return f"git {code['git_commit'][:12]}{dirty}"


def _round(value, digits: int):
    return None if value is None else round(float(value), digits)


def _ratio(d: dict | None) -> str:
    if not d or not d.get("denominator"):
        return "n=0"
    return f"{int(d['numerator'])}/{int(d['denominator'])}"


def main() -> None:
    st.set_page_config(page_title="StoneKit workstation", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    # STONEKIT_ROOT lets tests point the viewer at a temporary project.
    root = Path(os.environ["STONEKIT_ROOT"]) if os.environ.get("STONEKIT_ROOT") else find_project_root()
    runs_folder = root / "data" / "interim" / "stonekit_runs"
    valid, incomplete, corrupted = discover(runs_folder) if runs_folder.exists() else ([], [], [])

    with st.sidebar:
        st.markdown("**Study**")
        if not valid:
            st.caption("No processed runs. Create one with: stonekit run-case --case <development scan ID>")
        folder = st.selectbox("Run", valid, format_func=lambda p: str(p.relative_to(runs_folder)), label_visibility="collapsed") if valid else None
        report = None
        method_name = None
        if folder is not None:
            report = load_report(str(folder / "report.json"), (folder / "report.json").stat().st_mtime)
            available = [n for n, m in report.get("methods", {}).items() if m.get("available")]
            primary = report.get("primary_method")
            if available:
                st.markdown("**Segmentation**")
                method_name = st.radio("Segmentation", available, index=available.index(primary) if primary in available else 0,
                                       format_func=lambda n: method_label(n, report), label_visibility="collapsed")  # fmt: skip
        if incomplete or corrupted:
            st.markdown("**Not results**")
            for path in incomplete:
                st.caption(f"Incomplete (no report.json): {path.relative_to(runs_folder)}")
            for path, reason in corrupted:
                st.caption(f"Unreadable report ({reason}): {path.relative_to(runs_folder)}")

    case = report["case_id"] if report else "no study selected"
    created = report.get("created_utc", "") if report else ""
    ranges = load_ranges(folder) if folder is not None else None
    tail = ("calibrated ranges are a research demonstration of the frozen locked-study calibration (nominal 90%)." if ranges
            else "not a validated measurement, and no calibrated uncertainty is reported.")  # fmt: skip
    st.markdown(f"<div class='sk-bar'><span><span class='sk-title'>StoneKit</span><span class='sk-sub'>Kidney stone measurement "
                f"research workstation</span></span><span class='sk-meta'>{case} · run {created}</span></div>"
                "<div class='sk-notice'><b>Research use only. Not a diagnostic medical device.</b> Development-stage results: "
                f"{tail}</div>", unsafe_allow_html=True)  # fmt: skip
    if report is None:
        return

    if report.get("failure"):
        st.error(f"Processing failure. No stone result exists for this run. {report['failure']}")
        for key in ("kidney_segmentation_failure", "stone_segmentation_failure"):
            failure = report.get(key)
            if failure:
                kv([(k, v) for k, v in failure.items() if k != "notes"])
    elif report.get("result_status") == "no_result" or (report.get("methods") == {} and not report.get("failure")):
        st.warning("This run ended without a stone result. See the case flags under QC.")
    if report.get("methods") and report.get("primary_method") is None and not report.get("failure"):
        st.warning("No trained-model result in this run. The 272 HU threshold is shown as a comparator only, not as a substitute.")
    method = report.get("methods", {}).get(method_name, {}) if report.get("methods") and method_name else {}
    stones = method.get("stones", [])
    checks = report.get("input_checks") or {}
    spacing = checks.get("spacing_mm")
    st.markdown("<div class='sk-study'>"
                f"<span>Study<b>{case}</b></span>"
                f"<span>Voxel<b>{' × '.join(f'{s:.2f}' for s in spacing) + ' mm' if spacing else 'not recorded'}</b></span>"
                f"<span>Segmentation<b>{method_label(method_name, report) if method_name else 'none'}</b></span>"
                f"<span>Stones<b>{len(stones)}</b></span>"
                f"<span>QC<b>{qc_status(report, stones)}</b></span></div>", unsafe_allow_html=True)  # fmt: skip

    selected = None
    if stones:
        selected = st.selectbox("Selected stone", [s["stone_id"] for s in stones], key=f"sel_{folder}_{method_name}",
                                format_func=lambda i: next(f"Stone {i} · {s['volume_mm3']:.1f} mm³ · {s['max_diameter_3d_mm']:.1f} mm"
                                                           for s in stones if s["stone_id"] == i))  # fmt: skip
    chosen = next((s for s in stones if s["stone_id"] == selected), None)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        images_ready = report.get("input_checks") and report.get("roi") and (folder / "kidney_totalsegmentator.nii.gz").exists() and Path(report["ct_path"]).exists()
        if not images_ready:
            st.info("No images to show for this run: the CT, the kidney segmentation or the processing result is missing.")
        else:
            try:
                html = viewer_page(str(folder), report["ct_path"], method.get("label_file") if method else None, selected, 700)
                components.html(html, height=712, scrolling=False)
                st.caption("Kidney region of interest on the original voxel grid (no resampling). Outlines: kidney (grey-blue), "
                           "stones (orange), selected stone (blue).")
            except (OSError, ValueError, EOFError, KeyError) as error:
                st.error(f"Saved images could not be read ({type(error).__name__}). Nothing is shown in their place.")
    with right:
        st.markdown("### Stones")
        if stones:
            st.dataframe(pd.DataFrame([{"Stone": s["stone_id"], "Volume (mm³)": round(s["volume_mm3"], 1),
                                        "Max 3D diameter (mm)": round(s["max_diameter_3d_mm"], 1),
                                        "Axial diameter (mm)": round(s["max_diameter_axial_mm"], 1),
                                        "Size bin": size_category(s["max_diameter_3d_mm"]), "Mean HU": round(s["mean_hu"]),
                                        "QC": "REVIEW" if s.get("qc_flags") else "PASS"} for s in stones]),
                         hide_index=True, width="stretch", height=min(38 + 35 * len(stones), 300))  # fmt: skip
        else:
            st.caption("No stone pieces in this result.")
        st.markdown("### Patient summary")
        largest = max(stones, key=lambda s: s["max_diameter_3d_mm"]) if stones else None
        kv([("Stone pieces", len(stones)),
            ("Largest stone (max 3D diameter)", f"{largest['max_diameter_3d_mm']:.1f} mm (stone {largest['stone_id']})" if largest else "none"),
            ("Total stone volume", f"{method['total_volume_mm3']:.1f} mm³" if method else "not available"),
            ("Pieces by size bin", ", ".join(f"{k}: {v}" for k, v in pd.Series([size_category(s['max_diameter_3d_mm']) for s in stones]).value_counts().sort_index().items()) or "none")])  # fmt: skip
        st.markdown(f"### QC status &nbsp;{status_badge(qc_status(report, stones))}", unsafe_allow_html=True)
        flag_list(report.get("case_qc_flags", []))
        st.markdown("### Measurement uncertainty")
        if ranges and method_name == report.get("primary_method"):
            b = ranges["burden"]
            kv([("Total stone burden", f"{b['predicted_mm3']:.0f} mm³"),
                ("Detection-aware interval (90%)", f"{b['interval_mm3'][0]:.0f} – {b['interval_mm3'][1]:.0f} mm³")])  # fmt: skip
            if ranges["stones"]:
                st.dataframe(pd.DataFrame([{"Stone": s["stone_id"], "Volume (mm³)": round(s["volume_mm3"]),
                                            "Volume range (90%)": f"{s['volume_range_mm3'][0]:.0f} – {s['volume_range_mm3'][1]:.0f}",
                                            "3D diameter (mm)": round(s["max_diameter_3d_mm"], 1),
                                            "Diameter range (90%)": f"{s['diameter_range_mm'][0]:.1f} – {s['diameter_range_mm'][1]:.1f}",
                                            "P(false detection)": round(s["false_detection_probability"], 2)} for s in ranges["stones"]]),
                             hide_index=True, width="stretch", height=min(38 + 35 * len(ranges["stones"]), 300))  # fmt: skip
            st.caption(ranges["note"])
        else:
            st.markdown(f"<div class='sk-kv'>{status_badge('NOTE')} {UNCERTAINTY_TEXT}</div>", unsafe_allow_html=True)
        if chosen:
            st.markdown(f"### Stone {chosen['stone_id']}")
            kv([("Volume", f"{chosen['volume_mm3']:.1f} mm³ ({chosen['volume_ml']:.3f} mL)"),
                ("Maximum 3D diameter", f"{chosen['max_diameter_3d_mm']:.1f} mm"),
                ("Maximum axial diameter", f"{chosen['max_diameter_axial_mm']:.1f} mm"),
                ("Extent x, y, z", ", ".join(f"{chosen[k]:.1f}" for k in ("extent_x_mm", "extent_y_mm", "extent_z_mm")) + " mm"),
                ("Attenuation mean / peak", f"{chosen['mean_hu']:.0f} / {chosen['max_hu']:.0f} HU"),
                ("Distance to kidney", f"{chosen['distance_to_kidney_mm']:.1f} mm"), ("Voxels", chosen["voxel_count"])])  # fmt: skip
            flag_list(chosen.get("qc_flags", []))
            st.caption("Diameters are centre-to-centre voxel distances. Attenuation (HU) is not composition.")

    tabs = st.tabs(["3D", "Measurements", "QC and processing", "Validation", "Provenance"])
    with tabs[0]:
        if stones and report.get("roi"):
            st.plotly_chart(mesh_figure(folder, report, method, selected, True), width="stretch")
            note = f" Showing the first {MAX_MESHES} of {len(stones)}." if len(stones) > MAX_MESHES else ""
            st.caption("Surfaces from the segmentation mask by marching cubes, unsmoothed, RAS millimetres." + note)
        else:
            st.info("This method reported no stone pieces, so there is nothing to render in 3D.")

    with tabs[1]:
        if not method:
            st.info("No segmentation result in this run.")
        else:
            st.markdown(f"### {method_label(method_name, report)}")
            kv([("Pieces", len(stones)), ("Total volume", f"{method['total_volume_mm3']:.1f} mm³ ({method['total_volume_ml']:.3f} mL)"),
                ("Tiny pieces dropped", f"{method['dropped_tiny_pieces']} ({method['dropped_tiny_voxels']} voxels)")])  # fmt: skip
            flag_list(method.get("qc_flags", []))
            rows = stone_table(method)
            if rows:
                st.dataframe(pd.DataFrame(rows, columns=TABLE_COLUMNS), hide_index=True, width="stretch")
            st.caption("Measured on the original CT grid with original spacing; no resampling. Not calibrated: no uncertainty range is given.")
            kd = method.get("kidney_distance_filter")
            if kd is not None:
                st.markdown(f"### Kidney-distance filter ({kd['max_distance_mm']:g} mm)")
                removed = kd.get("removed_pieces", [])
                if removed:
                    st.dataframe(pd.DataFrame([{"Piece ID before filtering": r["stone_id"], "Volume (mm3)": round(r["volume_mm3"], 1),
                                                "Distance to kidney (mm)": round(r["distance_to_kidney_mm"], 1)} for r in removed]),
                                 hide_index=True, width="stretch")  # fmt: skip
                else:
                    st.caption("No piece was removed in this run.")
                st.caption("Development-selected filter. It can remove real stones far from the kidney mask, such as some ureteric "
                           "stones, so removed pieces are listed here. Not validated for general use.")  # fmt: skip
        files = st.columns(3)
        files[0].download_button("report.json", (folder / "report.json").read_bytes(), "report.json")
        if (folder / "report.html").exists():
            files[1].download_button("report.html", (folder / "report.html").read_bytes(), "report.html")

    with tabs[2]:
        stages = pd.DataFrame(report.get("stages", []))
        if not stages.empty:
            st.dataframe(stages.rename(columns=str.capitalize), hide_index=True, width="stretch")
        memory = report.get("peak_memory_kb") or {}
        shown = ", ".join(f"{k.replace('_', ' ')} {v / 1024 / 1024:.2f} GB" for k, v in memory.items() if isinstance(v, int | float))
        kv([("Total time", f"{report.get('total_seconds')} s"), ("Peak memory (RSS)", shown or "not recorded")])
        st.markdown("### Case flags")
        flag_list(report.get("case_qc_flags", []))
        checks = report.get("input_checks")
        if checks:
            st.markdown("### Input checks")
            kv([("Shape", checks["shape"]), ("Spacing", ", ".join(f"{s:.4f}" for s in checks["spacing_mm"]) + " mm"),
                ("File orientation", checks["original_orientation"]), ("HU range", f"{checks['hu_min']:.0f} to {checks['hu_max']:.0f}")])  # fmt: skip
            st.dataframe(pd.DataFrame(checks["checks"]), hide_index=True, width="stretch")
        st.markdown("### Preprocessing, kidney and ROI")
        info = [("Preprocessing", " ".join(report.get("preprocessing", [])))]
        if report.get("kidney"):
            info.append(("Kidney volumes", f"right {report['kidney']['right_volume_ml']:.1f} mL, left {report['kidney']['left_volume_ml']:.1f} mL"))
        if report.get("contrast_check"):
            hu = report["contrast_check"]["aorta_median_hu"]
            info.append(("Aorta median", "not available" if hu is None else f"{hu:.0f} HU (heuristic only)"))
        if report.get("roi"):
            info.append(("ROI box", ", ".join(f"{v:.0f}" for v in report["roi"]["box_size_mm"]) + f" mm, margin {report['roi']['margin_mm']} mm"))
        kv(info)

    with tabs[3]:
        validation_tab(root, report)

    with tabs[4]:
        kidney = next((s for s in report.get("stages", []) if s["stage"] == "kidney_segmentation"), {})
        config = report.get("config") or {}
        limit = (report.get("postprocessing") or {}).get("model_kidney_distance_filter_mm",
                                                         (config.get("postprocessing") or {}).get("model_max_distance_to_kidney_mm"))  # fmt: skip
        try:
            run_folder = str(folder.relative_to(root))
        except ValueError:
            run_folder = str(folder)
        kv([("Run folder", run_folder), ("Created (UTC)", report.get("created_utc")),
            ("Result status", report.get("result_status", "not recorded (older run)")),
            ("StoneKit code", _code_text(report.get("code_version"))),
            ("Configuration", f"{config.get('id', 'unknown')} ({report.get('config_file', 'file not recorded')})"),
            ("Config file SHA-256", report.get("config_sha256")),
            ("Configuration used SHA-256", report.get("effective_config_sha256", "not recorded (older run)")),
            ("Model", model_label(report)),
            ("Kidney-distance filter", "not applied" if limit is None else f"{limit:g} mm, model method only"),
            ("Kidney segmentation", kidney.get("detail") or kidney.get("status", "not run")),
            ("Checkpoint", report.get("model_info", {}).get("checkpoint", "none")),
            ("Checkpoint SHA-256", report.get("model_info", {}).get("checkpoint_sha256", "none"))])  # fmt: skip
        st.markdown("### Software")
        kv(list(report.get("software", {}).items()))


if __name__ == "__main__":
    main()
