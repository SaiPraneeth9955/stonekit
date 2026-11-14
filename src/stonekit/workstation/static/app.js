// StoneKit workstation front end. Plain JavaScript, no external network. Every value shown comes from the API.
"use strict";

// ------------------------------------------------------------------ helpers
const $ = (sel, el = document) => el.querySelector(sel);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function fmt(v, d = 1) { return v == null || Number.isNaN(+v) ? "—" : (+v).toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d }); }
const rng = (a, d = 0) => a ? `${fmt(a[0], d)} – ${fmt(a[1], d)}` : "—";
const chip = s => `<span class="chip s-${String(s).replace(/\s/g, "")}">${esc(s)}</span>`;
function when(s) {
  if (!s) return "—";
  const m = String(s).match(/^(\d{4})-?(\d{2})-?(\d{2})[T ]?(\d{2}):?(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]} UTC` : esc(s);
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}
const app = () => $("#app");
const loading = (msg = "Loading…") => `<div class="loading"><div class="spinner"></div>${esc(msg)}</div>`;
const errorBox = e => `<div class="callout err"><b>Could not load.</b> ${esc(e.message || e)}</div>`;
const PLOT = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "toImage"] };
const AX = { gridcolor: "#e9edf0", zerolinecolor: "#dde3e7", linecolor: "#dde3e7", tickfont: { size: 11, color: "#66737e" }, titlefont: { size: 12, color: "#3a4752" } };
const LAYOUT = { paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff", font: { family: "Segoe UI, Helvetica Neue, Arial, sans-serif", size: 12, color: "#16212a" }, margin: { l: 56, r: 18, t: 18, b: 46 } };
const C = { accent: "#0e6a85", soft: "#b7d3dc", cmp: "#9aa6b0", stone: "#d9822b", sel: "#1f6fd1", pass: "#2c7a51", review: "#9a5d00", kr: "#3c9c9a", kl: "#5f86b3" };

let STATE = { worklist: null, studies: {}, evidence: null, viewer: null, job: null, sel: null };

// ------------------------------------------------------------------ router
const routes = [
  [/^#\/worklist$/, renderWorklist], [/^#\/new$/, renderNew], [/^#\/evidence$/, renderEvidence], [/^#\/method$/, renderMethod],
  [/^#\/study\/([^/]+)(?:\/(\w+))?(?:\/(\d+))?$/, (m) => { if (m[3]) STATE.sel = +m[3]; return renderStudy(decodeURIComponent(m[1]), m[2] || "overview"); }],
];
function route() {
  const h = location.hash || "#/worklist";
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", h.startsWith("#/" + a.dataset.n) || (a.dataset.n === "worklist" && h.startsWith("#/study"))));
  for (const [re, fn] of routes) { const m = h.match(re); if (m) { window.scrollTo(0, 0); return fn(m); } }
  location.hash = "#/worklist";
}
window.addEventListener("hashchange", route);

async function checkIntegrity() {
  try {
    const i = await api("/api/integrity");
    $("#sys-integrity").innerHTML = i.status === "PASS"
      ? `<span class="dot ok"></span>Frozen calibration verified (${i.verified}/${i.rows.length})`
      : i.status === "FAIL" ? `<span class="dot bad"></span>Frozen artefact mismatch`
      : `<span class="dot idle"></span>Frozen artefacts: ${i.verified}/${i.rows.length} verified, ${i.missing} not present`;
    $("#sys-integrity").title = i.rows.map(r => `${r.artefact}: ${r.status}`).join("\n");
  } catch (e) { $("#sys-integrity").innerHTML = `<span class="dot bad"></span>Integrity check unavailable`; }
}

// ------------------------------------------------------------------ worklist
async function renderWorklist() {
  app().innerHTML = `<div class="page-head"><div><h1>Worklist</h1><div class="sub">Processed CT studies on this workstation. Open a study to review imaging, stones, uncertainty and quality control.</div></div>
    <div class="toolbar"><input class="input" id="wl-q" placeholder="Filter by study, source or status" style="width:280px"><a class="btn primary" href="#/new">New analysis</a></div></div>
    <div class="panel"><div id="wl">${loading("Loading studies…")}</div></div>`;
  try {
    const data = await api("/api/worklist"); STATE.worklist = data.studies;
    const draw = () => {
      const f = ($("#wl-q").value || "").toLowerCase();
      const rows = data.studies.filter(s => !f || [s.case_id, s.source, s.eligibility.status, s.qc, s.group].join(" ").toLowerCase().includes(f));
      const n = data.studies.length, el = data.studies.filter(s => s.eligibility.status === "ELIGIBLE").length;
      $("#wl").innerHTML = !n ? `<div class="empty"><h3>No processed studies</h3>Start one from <a href="#/new">New analysis</a>.</div>` : `
        <div class="panel-h"><h3>Studies</h3><span class="hint">${n} studies · ${el} with calibrated ranges · ${rows.length} shown</span></div>
        <div style="overflow:auto;max-height:calc(100vh - 250px)"><table class="t"><thead><tr>
          <th>Study</th><th>Source</th><th>Analysed</th><th>Matrix</th><th>Voxel spacing (mm)</th><th class="n">Stones</th><th class="n">Burden (mm³)</th><th class="n">90% burden interval (mm³)</th><th>Eligibility</th><th>QC</th></tr></thead>
          <tbody>${rows.map(s => `<tr class="click" data-id="${esc(s.id)}" title="${esc(s.eligibility.detail)}">
            <td><b>${esc(s.case_id)}</b>${s.group ? `<div class="sub">${esc(s.group)}</div>` : ""}</td><td>${esc(s.source)}</td><td class="mono">${when(s.created_utc)}</td>
            <td class="mono">${s.shape ? s.shape.join(" × ") : "—"}</td><td class="mono">${s.spacing_mm ? s.spacing_mm.map(v => fmt(v, 2)).join(" × ") : "—"}</td>
            <td class="n">${s.stones}</td><td class="n">${fmt(s.total_volume_mm3, 0)}</td><td class="n">${rng(s.burden_interval_mm3)}</td>
            <td>${chip(s.eligibility.status)}</td><td>${chip(s.qc)}</td></tr>`).join("") || `<tr><td colspan="10" class="empty">No study matches the filter.</td></tr>`}</tbody></table></div>`;
      document.querySelectorAll("#wl tr.click").forEach(tr => tr.onclick = () => location.hash = `#/study/${encodeURIComponent(tr.dataset.id)}`);
    };
    $("#wl-q").oninput = draw; draw();
  } catch (e) { $("#wl").innerHTML = errorBox(e); }
}

// ------------------------------------------------------------------ new analysis
const PIPE = ["Load CT and validate input", "Kidney segmentation (TotalSegmentator)", "Kidney region of interest", "Stone segmentation (nnU-Net)",
  "Measurement, QC and 3D surfaces", "8-flip test-time augmentation", "Calibrated ranges (frozen calibration)"];
async function renderNew() {
  app().innerHTML = `<div class="page-head"><div><h1>New analysis</h1><div class="sub">Run the frozen StoneKit pipeline on one non-contrast CT. Processing runs locally on this workstation's GPU (typically about five minutes).</div></div></div>
  <div class="grid g-main-wide">
    <div class="panel"><div class="panel-h"><h3>1 · Select CT study</h3><div class="seg" id="src-tabs"><button class="on" data-t="mswal">External dataset</button><button data-t="li">Li development scan</button><button data-t="up">Import NIfTI file</button></div></div>
      <div class="panel-b" id="src">${loading("Loading available studies…")}</div></div>
    <div class="panel"><div class="panel-h"><h3>2 · Processing</h3><span class="hint" id="job-hint">Idle</span></div><div class="panel-b" id="job"></div></div>
  </div>`;
  let sources; try { sources = await api("/api/sources"); } catch (e) { $("#src").innerHTML = errorBox(e); return; }
  let tab = "mswal", chosen = null;
  const drawJobIdle = () => { $("#job").innerHTML = `<ol class="steps">${PIPE.map((s, i) => `<li class="pending"><span class="ic">${i + 1}</span>${s}</li>`).join("")}</ol>
    <div class="mt14"><div class="kv"><dt>Selected input</dt><dd id="chosen">${chosen ? esc(chosen.label) : "None"}</dd><dt>Run label</dt><dd><input class="input w100" id="label" placeholder="Defaults to the study name"></dd></div></div>
    <div class="mt14 toolbar"><button class="btn primary" id="go" ${chosen ? "" : "disabled"}>Start analysis</button><span class="note">The scan is checked for contrast; ranges are only computed for non-contrast studies.</span></div>`;
    $("#go").onclick = start; };
  const pick = c => { chosen = c; $("#chosen").textContent = c.label; $("#go").disabled = false; };
  const drawSrc = () => {
    document.querySelectorAll("#src-tabs button").forEach(b => b.classList.toggle("on", b.dataset.t === tab));
    if (tab === "mswal") {
      $("#src").innerHTML = sources.mswal.length ? `<div class="note mb14">MSWAL abdominal CT (CC BY-NC 4.0). An external dataset: the model never trained on it. Phase status is from the stored external-check screen.</div>
        <input class="input w100 mb14" id="mq" placeholder="Filter cases"><div style="max-height:52vh;overflow:auto;border:1px solid var(--line-2)"><table class="t"><thead><tr><th>Case</th><th>Phase (stored screen)</th><th></th></tr></thead><tbody id="mrows"></tbody></table></div>`
        : `<div class="empty"><h3>No external CT files found</h3>Expected in data/external/mswal/imagesTr.</div>`;
      const rows = () => { const f = ($("#mq")?.value || "").toLowerCase();
        $("#mrows").innerHTML = sources.mswal.filter(m => m.name.toLowerCase().includes(f)).map(m => `<tr class="click" data-p="${esc(m.path)}" data-n="${esc(m.name)}">
          <td><b>${esc(m.name)}</b></td><td>${m.phase === "eligible" ? chip("NON-CONTRAST") : m.phase.startsWith("ineligible") ? chip("CONTRAST") : `<span class="note">${esc(m.phase)}</span>`}</td><td class="right"><button class="btn sm">Select</button></td></tr>`).join("");
        document.querySelectorAll("#mrows tr").forEach(tr => tr.onclick = () => { document.querySelectorAll("#mrows tr").forEach(x => x.classList.remove("sel")); tr.classList.add("sel");
          pick({ kind: "path", value: tr.dataset.p, label: `${tr.dataset.n} (MSWAL)`, name: tr.dataset.n }); }); };
      if (sources.mswal.length) { $("#mq").oninput = rows; rows(); }
    } else if (tab === "li") {
      $("#src").innerHTML = `<div class="note mb14">Li 2022 development scans only. Locked calibration and test scans are refused by the pipeline, and so are the six reserved demonstration cases.</div>
        <div class="toolbar"><input class="input" id="liid" placeholder="Scan ID, e.g. wd0254" style="width:220px"><button class="btn" id="lisel">Select</button></div><div class="note mt8" id="lierr"></div>`;
      $("#lisel").onclick = () => { const v = $("#liid").value.trim(); if (!/^wd\d{4}$/.test(v)) { $("#lierr").textContent = "A scan ID looks like wd0254."; return; } $("#lierr").textContent = ""; pick({ kind: "li", value: v, label: `${v} (Li development)`, name: v }); };
    } else {
      $("#src").innerHTML = `<div class="note mb14">Import a non-contrast CT volume in NIfTI format (.nii or .nii.gz). The file is stored locally in data/interim/uploads and never leaves this computer.</div>
        <div class="toolbar"><input type="file" id="file" accept=".nii,.gz"><button class="btn" id="upbtn">Import</button></div><div class="mt8 note" id="upmsg"></div>
        ${sources.uploads.length ? `<div class="mt14"><b>Previously imported</b><table class="t mt8"><tbody>${sources.uploads.map(u => `<tr class="click" data-p="${esc(u.path)}" data-n="${esc(u.name)}"><td>${esc(u.name)}</td><td class="right"><button class="btn sm">Select</button></td></tr>`).join("")}</tbody></table></div>` : ""}`;
      $("#upbtn").onclick = async () => { const f = $("#file").files[0]; if (!f) { $("#upmsg").textContent = "Choose a file first."; return; }
        $("#upmsg").innerHTML = `<span class="spinner" style="display:inline-block;vertical-align:-2px"></span> Importing ${esc(f.name)} (${fmt(f.size / 1e6, 1)} MB)…`;
        try { const u = await api(`/api/uploads/${encodeURIComponent(f.name)}`, { method: "PUT", body: f }); $("#upmsg").textContent = `Imported ${u.name}.`; pick({ kind: "path", value: u.path, label: `${u.name} (imported)`, name: u.name }); }
        catch (e) { $("#upmsg").innerHTML = `<span style="color:var(--fail)">${esc(e.message)}</span>`; } };
      document.querySelectorAll("#src tr.click").forEach(tr => tr.onclick = () => pick({ kind: "path", value: tr.dataset.p, label: `${tr.dataset.n} (imported)`, name: tr.dataset.n }));
    }
  };
  async function start() {
    const label = ($("#label").value || "").trim();
    try { const j = await api("/api/analyses", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind: chosen.kind, value: chosen.value, label: label || chosen.name }) }); poll(j.id); }
    catch (e) { $("#job").insertAdjacentHTML("beforeend", `<div class="callout err mt14">${esc(e.message)}</div>`); }
  }
  async function poll(id) {
    let j; try { j = await api(`/api/analyses/${id}`); } catch (e) { return; }
    if (!$("#job")) return; // the user left this page
    $("#job-hint").textContent = j.state === "running" ? `Running · ${Math.floor(j.elapsed_s / 60)} min ${j.elapsed_s % 60} s` : j.state === "done" ? `Completed in ${Math.round(j.elapsed_s / 60 * 10) / 10} min` : "Failed";
    const pct = j.state === "done" ? 100 : Math.round(j.step / j.steps.length * 100);
    $("#job").innerHTML = `<div class="pbar mb14"><span style="width:${pct}%"></span></div>
      <ol class="steps">${j.steps.map((s, i) => `<li class="${i < j.step || j.state === "done" ? "done" : i === j.step && j.state === "running" ? "active" : "pending"}"><span class="ic">${i < j.step || j.state === "done" ? "✓" : ""}</span>${s}</li>`).join("")}</ol>
      ${j.estimated ? `<div class="note mt8">Pipeline stages 1–5 run as one process; the active stage is estimated from elapsed time.</div>` : ""}
      ${j.state === "failed" ? `<div class="callout err mt14"><b>Analysis stopped.</b> ${esc(j.error)}</div>` : ""}
      ${j.state === "done" && j.study_id ? `<div class="mt14 toolbar"><a class="btn primary" href="#/study/${encodeURIComponent(j.study_id)}">Open study</a></div>` : ""}
      <div class="mt14 note">Process output</div><div class="console">${esc(j.log.join("\n")) || "Starting…"}</div>`;
    if (j.state === "running" && location.hash === "#/new") setTimeout(() => poll(id), 2000);
  }
  document.querySelectorAll("#src-tabs button").forEach(b => b.onclick = () => { tab = b.dataset.t; drawSrc(); });
  drawSrc(); drawJobIdle();
  // show a running analysis, or the result of the most recent one in this session
  try { const js = await api("/api/analyses"); const last = js.jobs.find(j => j.state === "running") || js.jobs[0]; if (last) poll(last.id); } catch (e) { /* no job list */ }
}

// ------------------------------------------------------------------ study shell
const TABS = [["overview", "Overview"], ["imaging", "Imaging"], ["3d", "3D"], ["stones", "Stones"], ["burden", "Burden & uncertainty"], ["qc", "Quality & provenance"], ["report", "Report"]];
async function loadStudy(id) {
  if (!STATE.studies[id]) STATE.studies[id] = await api(`/api/studies/${encodeURIComponent(id)}`);
  return STATE.studies[id];
}
async function renderStudy(id, tab) {
  if (STATE.viewer && STATE.viewer.id !== id) STATE.viewer = null;
  app().innerHTML = loading("Opening study…");
  let s; try { s = await loadStudy(id); } catch (e) { app().innerHTML = errorBox(e); return; }
  const r = s.report, chk = r.input_checks || {}, b = (s.ranges || {}).burden;
  app().innerHTML = `<div class="crumbs"><a href="#/worklist">Worklist</a> / ${esc(r.case_id)}</div>
    <div class="banner">
      <div class="cell"><div class="l">Study</div><div class="v">${esc(r.case_id)}</div></div>
      <div class="cell"><div class="l">Matrix</div><div class="v mono">${chk.shape ? chk.shape.join(" × ") : "—"}</div></div>
      <div class="cell"><div class="l">Voxel spacing</div><div class="v mono">${chk.spacing_mm ? chk.spacing_mm.map(v => fmt(v, 2)).join(" × ") + " mm" : "—"}</div></div>
      <div class="cell"><div class="l">Analysed</div><div class="v mono">${when(r.created_utc)}</div></div>
      <div class="cell"><div class="l">Stones</div><div class="v">${s.stones.length}</div></div>
      <div class="cell"><div class="l">Burden · 90% interval</div><div class="v">${b ? `${fmt(b.predicted_mm3, 0)} mm³ · ${rng(b.interval_mm3)}` : "—"}</div></div>
      <div class="cell grow"><div class="l">Eligibility · QC</div><div class="v" title="${esc(s.eligibility.detail)}">${chip(s.eligibility.status)} ${chip(s.qc)}</div></div>
      <div class="actions"><a class="btn sm" href="/report/${encodeURIComponent(id)}" target="_blank">Open report</a><a class="btn sm" href="/export/${encodeURIComponent(id)}.json">Export JSON</a></div>
    </div>
    <nav class="tabs">${TABS.map(([k, n]) => `<a href="#/study/${encodeURIComponent(id)}/${k}" class="${k === tab ? "active" : ""}">${n}</a>`).join("")}</nav>
    <div id="tab"></div>`;
  const T = { overview: tabOverview, imaging: tabImaging, "3d": tab3D, stones: tabStones, burden: tabBurden, qc: tabQC, report: tabReport }[tab] || tabOverview;
  T(s, id, $("#tab"));
}

// ------------------------------------------------------------------ overview
function tabOverview(s, id, el) {
  const r = s.report, b = (s.ranges || {}).burden, d = b && b.decomposition, k = r.kidney || {}, cc = r.contrast_check || {};
  const kd = ((r.methods || {}).model || {}).kidney_distance_filter || {};
  const go = t => `#/study/${encodeURIComponent(id)}/${t}`;
  const flow = [
    ["CT study", `${(r.input_checks || {}).shape ? r.input_checks.shape.join("×") : "—"}`, "imaging"],
    ["Input & phase QC", cc.aorta_median_hu != null ? `aorta ${fmt(cc.aorta_median_hu, 0)} HU` : "not checked", "qc"],
    ["Kidneys", `R ${fmt(k.right_volume_ml, 0)} · L ${fmt(k.left_volume_ml, 0)} mL`, "imaging"],
    ["Region of interest", `${(r.roi || {}).margin_mm ?? "—"} mm margin`, "imaging"],
    ["Stone segmentation", `${s.stones.length + (kd.removed_pieces || []).length} pieces`, "imaging"],
    ["Kidney-distance filter", `${(kd.removed_pieces || []).length} removed`, "qc"],
    ["3D measurement", `${s.stones.length} stones`, "stones"],
    ["Per-stone ranges", s.ranges ? "90% calibrated" : "not available", "stones"],
    ["Burden interval", b ? rng(b.interval_mm3) + " mm³" : "not available", "burden"],
    ["Report", "structured", "report"],
  ];
  el.innerHTML = `
  <div class="panel mb14"><div class="panel-h"><h3>Analysis pathway</h3><span class="hint">Each step of the StoneKit pipeline for this study. Select a step to review it.</span></div>
    <div class="panel-b"><div class="flow">${flow.map(([n, v, t], i) => `<a class="step ${t === "burden" ? "hl" : ""}" href="${go(t)}"><div class="n">${String(i + 1).padStart(2, "0")}</div><div class="name">${n}</div><div class="val">${esc(v)}</div></a>`).join("")}</div></div></div>
  <div class="grid g4 mb14">
    <div class="panel metric"><div class="label">Stones retained</div><div class="value">${s.stones.length}</div><div class="note">${(kd.removed_pieces || []).length} removed by the 30 mm filter</div></div>
    <div class="panel metric"><div class="label">Measured stone burden</div><div class="value">${b ? fmt(b.predicted_mm3, 0) : fmt(((r.methods || {}).model || {}).total_volume_mm3, 0)}<small>mm³</small></div><div class="note">Sum of retained stone volumes</div></div>
    <div class="panel metric primary"><div class="label">Detection-aware burden interval</div><div class="value">${b ? rng(b.interval_mm3) : "—"}<small>${b ? "mm³" : ""}</small></div><div class="note">${b ? "Nominal 90%, frozen calibration" : esc(s.eligibility.detail)}</div></div>
    <div class="panel metric"><div class="label">Largest stone · 3D diameter</div><div class="value">${s.stones.length ? fmt(Math.max(...s.stones.map(x => x.max_diameter_3d_mm)), 1) : "—"}<small>mm</small></div><div class="note">${s.stones.length ? `Stone ${s.stones.reduce((a, x) => x.max_diameter_3d_mm > a.max_diameter_3d_mm ? x : a).stone_id}` : "No stone retained"}</div></div>
  </div>
  <div class="grid g-main">
    <div class="panel"><div class="panel-h"><h3>Stones</h3><a class="hint" href="${go("stones")}">Stone analysis →</a></div><div class="panel-b flush">${stoneTable(s, id, true)}</div></div>
    <div class="panel"><div class="panel-h"><h3>Study information</h3></div><div class="panel-b"><dl class="kv">
      <dt>Study identifier</dt><dd>${esc(r.case_id)}</dd><dt>Eligibility</dt><dd>${chip(s.eligibility.status)}<div class="note mt8">${esc(s.eligibility.detail)}</div></dd>
      <dt>Contrast check</dt><dd>${cc.aorta_median_hu != null ? `Aorta median ${fmt(cc.aorta_median_hu, 0)} HU (threshold ${cc.threshold_hu} HU)` : "Not available"}</dd>
      <dt>Kidney segmentation</dt><dd>Right ${fmt(k.right_volume_ml, 1)} mL · left ${fmt(k.left_volume_ml, 1)} mL</dd>
      <dt>Model</dt><dd>${esc((r.model_info || {}).name || "—")}</dd>
      <dt>Pipeline configuration</dt><dd class="mono">${esc((r.config_sha256 || "").slice(0, 16))}…</dd>
      <dt>Calibration</dt><dd class="mono">${esc((s.integrity.calibration_sha256 || "").slice(0, 16))}… ${chip(s.integrity.status)}</dd>
      <dt>Protocol</dt><dd>${esc(s.integrity.protocol)}</dd></dl>
      ${r.status_line ? `<div class="callout mt14">${esc(r.status_line)}</div>` : ""}</div></div>
  </div>`;
  bindStoneRows(el, id);
}

function stoneTable(s, id, compact) {
  if (!s.stones.length) return `<div class="empty"><h3>No stone retained</h3>The model found no stone piece within 30 mm of a kidney.</div>`;
  return `<div style="overflow:auto"><table class="t"><thead><tr><th>Stone</th><th>Location (segmentation)</th><th class="n">Volume (mm³)</th><th class="n">90% range</th><th class="n">3D diameter (mm)</th><th class="n">90% range</th><th>Size band</th>${compact ? "" : `<th class="n">2D axial (mm)</th>`}<th>P(false detection)</th><th>QC</th></tr></thead>
    <tbody>${s.stones.map(st => { const g = st.range; const p = g ? g.false_detection_probability : null;
      return `<tr class="click ${STATE.sel === st.stone_id ? "sel" : ""}" data-s="${st.stone_id}"><td><span class="stone-pill"><i></i><b>${st.stone_id}</b></span></td>
      <td>${esc(st.location.side)}<div class="sub">${esc(st.location.position)}</div></td><td class="n"><b>${fmt(st.volume_mm3, 0)}</b></td><td class="n">${g ? rng(g.volume_range_mm3) : "—"}</td>
      <td class="n"><b>${fmt(st.max_diameter_3d_mm, 1)}</b></td><td class="n">${g ? rng(g.diameter_range_mm, 1) : "—"}</td><td>${esc(st.size_band)}</td>
      ${compact ? "" : `<td class="n">${fmt(st.max_diameter_axial_mm, 1)}${st.category_changes_2d ? ` <span class="tag" title="The 2D axial diameter falls in a different size band (${esc(st.size_band_2d)})">${esc(st.size_band_2d)}</span>` : ""}</td>`}
      <td>${p == null ? "—" : `<span class="prob ${p >= 0.5 ? "hi" : ""}"><span style="width:${Math.round(p * 100)}%"></span></span>${fmt(p, 2)}`}</td><td>${chip(st.qc_status)}</td></tr>`; }).join("")}</tbody></table></div>`;
}
function bindStoneRows(el, id) {
  el.querySelectorAll("tr[data-s]").forEach(tr => tr.onclick = () => { STATE.sel = +tr.dataset.s; location.hash = `#/study/${encodeURIComponent(id)}/stones`; });
}

// ------------------------------------------------------------------ imaging
async function tabImaging(s, id, el) {
  el.innerHTML = `<div class="grid g-main"><div><div id="vwr">${loading("Loading CT volume and segmentations…")}</div>
      <div class="panel-b note" style="background:var(--surface);border:1px solid var(--line);border-top:0">Wheel: slice (Shift ×5) · Left-drag: window/level · Click: set crosshair / select stone · Right-drag: pan · Ctrl+wheel: zoom · Double-click: reset view. Kidney region of interest on the original voxel grid; no resampling.</div></div>
    <div class="grid" style="align-content:start">
      <div class="panel"><div class="panel-h"><h3>Cursor</h3></div><div class="panel-b"><dl class="kv" id="cur"><dt>Voxel (x, y, z)</dt><dd>—</dd><dt>Position (mm)</dt><dd>—</dd><dt>Attenuation</dt><dd>—</dd><dt>Structure</dt><dd>—</dd></dl></div></div>
      <div class="panel"><div class="panel-h"><h3>Selected stone</h3></div><div class="panel-b" id="selstone"><div class="note">Click a stone in the image or choose one in "Go to".</div></div></div>
      <div class="panel"><div class="panel-h"><h3>Legend</h3></div><div class="panel-b legend">
        <span><i class="sw" style="background:${C.kr}"></i>Right kidney (outline)</span><span><i class="sw" style="background:${C.kl}"></i>Left kidney (outline)</span>
        <span><i class="sw" style="background:${C.stone}"></i>Stone</span><span><i class="sw" style="background:${C.sel}"></i>Selected stone</span></div></div>
    </div></div>`;
  const showSel = sid => { const st = s.stones.find(x => x.stone_id === sid); STATE.sel = sid;
    $("#selstone").innerHTML = st ? `<div class="measure"><div><div class="h">Measurement</div><div class="big">${fmt(st.volume_mm3, 0)}<small>mm³</small></div><div class="note">${fmt(st.max_diameter_3d_mm, 1)} mm 3D max. diameter</div></div>
      <div><div class="h">90% calibrated range</div>${st.range ? `<div class="rng">${rng(st.range.volume_range_mm3)} mm³</div><div class="note">${rng(st.range.diameter_range_mm, 1)} mm</div>` : `<div class="note">Not available</div>`}</div></div>
      <dl class="kv mt8"><dt>Stone</dt><dd>${st.stone_id} · ${esc(st.size_band)}</dd><dt>Location</dt><dd>${esc(st.location.side)}, ${esc(st.location.position)}</dd><dt>Attenuation mean / peak</dt><dd>${fmt(st.mean_hu, 0)} / ${fmt(st.max_hu, 0)} HU</dd><dt>Slices spanned</dt><dd>${st.slices_spanned ?? "—"}</dd></dl>
      <div class="mt8"><a href="#/study/${encodeURIComponent(id)}/stones">Full stone analysis →</a></div>` : `<div class="note">Not a retained stone.</div>`; };
  try {
    const payload = await api(`/api/studies/${encodeURIComponent(id)}/volume`);
    const v = new CTViewer($("#vwr"), payload, s.stones, showSel);
    v.onReadout = o => { $("#cur").innerHTML = `<dt>Voxel (x, y, z)</dt><dd class="mono">${o.voxel.join(", ")}</dd><dt>Position (mm)</dt><dd class="mono">${o.mm.map(x => fmt(x, 1)).join(", ")}</dd><dt>Attenuation</dt><dd class="mono">${o.hu} HU</dd><dt>Structure</dt><dd>${esc(o.label)}</dd>`; };
    await v.init(); STATE.viewer = v;
    if (STATE.sel && s.stones.some(x => x.stone_id === STATE.sel)) { v.goto(STATE.sel); showSel(STATE.sel); }
  } catch (e) { $("#vwr").innerHTML = errorBox(e); }
}

// ------------------------------------------------------------------ 3D
async function tab3D(s, id, el) {
  el.innerHTML = `<div class="grid g-main"><div class="panel"><div class="panel-h"><h3>3D surfaces · RAS millimetres</h3>
      <div class="toolbar"><button class="btn sm on" id="k3">Kidneys</button><label>Kidney opacity</label><input type="range" id="ko" min="5" max="60" value="16"><button class="btn sm" id="iso">Isolate selected</button><button class="btn sm" id="rs">Reset view</button></div></div>
      <div class="panel-b flush"><div id="p3" class="plot tall">${loading("Loading surfaces…")}</div></div></div>
    <div class="grid" style="align-content:start"><div class="panel"><div class="panel-h"><h3>Stones</h3><span class="hint">Select to highlight</span></div><div class="panel-b flush" id="l3"></div></div>
      <div class="panel"><div class="panel-h"><h3>Why 3D measurement</h3></div><div class="panel-b note">Stones are irregular and rarely aligned with the image planes. StoneKit measures volume from all voxels and the maximum diameter in 3D. On the physical phantom, the 3D maximum diameter had a bias of +0.19 mm against calipers, whereas the 2D axial maximum had −1.72 mm (Research evidence).</div></div></div></div>`;
  let m; try { m = await api(`/api/studies/${encodeURIComponent(id)}/meshes`); } catch (e) { $("#p3").innerHTML = errorBox(e); return; }
  let showK = true, kop = 0.16, iso = false;
  const byId = Object.fromEntries(s.stones.map(x => [x.stone_id, x]));
  const draw = (keepCamera) => {
    const tr = [];
    if (m.kidneys && showK && !iso) { const v = m.kidneys.v, f = m.kidneys.f;
      tr.push({ type: "mesh3d", x: v.map(p => p[0]), y: v.map(p => p[1]), z: v.map(p => p[2]), i: f.map(t => t[0]), j: f.map(t => t[1]), k: f.map(t => t[2]), color: "#7fa6b5", opacity: kop, name: "Kidneys", hoverinfo: "name", flatshading: false, lighting: { ambient: 0.55, diffuse: 0.7, specular: 0.05 } }); }
    for (const st of m.stones) { const sel = st.stone_id === STATE.sel; if (iso && STATE.sel && !sel) continue; const info = byId[st.stone_id] || {};
      tr.push({ type: "mesh3d", x: st.v.map(p => p[0]), y: st.v.map(p => p[1]), z: st.v.map(p => p[2]), i: st.f.map(t => t[0]), j: st.f.map(t => t[1]), k: st.f.map(t => t[2]),
        color: sel ? C.sel : C.stone, opacity: sel || !STATE.sel ? 1 : 0.55, name: `Stone ${st.stone_id}`,
        hovertext: `Stone ${st.stone_id}<br>${fmt(info.volume_mm3, 0)} mm³ · ${fmt(info.max_diameter_3d_mm, 1)} mm${info.range ? `<br>90% range ${rng(info.range.volume_range_mm3)} mm³` : ""}`, hoverinfo: "text", lighting: { ambient: 0.45, diffuse: 0.8, specular: 0.15 } }); }
    const ax = t => ({ title: { text: t, font: { size: 11, color: "#66737e" } }, backgroundcolor: "#fbfcfc", gridcolor: "#e9edf0", zerolinecolor: "#dde3e7", showbackground: true, tickfont: { size: 10, color: "#8b96a0" } });
    const lay = { ...LAYOUT, margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false, scene: { aspectmode: "data", xaxis: ax("x: L → R (mm)"), yaxis: ax("y: P → A (mm)"), zaxis: ax("z: I → S (mm)"),
      // anterior view, superior up: the patient's right appears on the screen's left, as in the 2D views
      camera: { eye: { x: 0, y: 1.9, z: 0.35 }, up: { x: 0, y: 0, z: 1 } } } };
    if (keepCamera && $("#p3").layout && $("#p3").layout.scene) lay.scene.camera = $("#p3").layout.scene.camera;
    $("#p3").innerHTML = ""; Plotly.react("p3", tr, lay, PLOT);
  };
  const list = () => { $("#l3").innerHTML = s.stones.length ? `<table class="t"><tbody>${s.stones.map(st => `<tr class="click ${STATE.sel === st.stone_id ? "sel" : ""}" data-s="${st.stone_id}"><td><span class="stone-pill"><i></i><b>Stone ${st.stone_id}</b></span><div class="sub">${esc(st.location.side)}</div></td><td class="n">${fmt(st.volume_mm3, 0)} mm³<div class="sub">${fmt(st.max_diameter_3d_mm, 1)} mm</div></td></tr>`).join("")}</tbody></table>` : `<div class="empty">No stone surfaces.</div>`;
    $("#l3").querySelectorAll("tr").forEach(tr => tr.onclick = () => { STATE.sel = STATE.sel === +tr.dataset.s ? null : +tr.dataset.s; list(); draw(true); }); };
  $("#k3").onclick = () => { showK = !showK; $("#k3").classList.toggle("on", showK); draw(true); };
  $("#ko").oninput = e => { kop = e.target.value / 100; draw(true); };
  $("#iso").onclick = () => { iso = !iso; $("#iso").classList.toggle("on", iso); draw(true); };
  $("#rs").onclick = () => draw(false);
  list(); draw(false);
}

// ------------------------------------------------------------------ stones
function tabStones(s, id, el) {
  if (!s.stones.length) { el.innerHTML = `<div class="panel">${stoneTable(s, id)}</div>`; return; }
  if (!STATE.sel || !s.stones.some(x => x.stone_id === STATE.sel)) STATE.sel = s.stones[0].stone_id;
  el.innerHTML = `<div class="panel mb14"><div class="panel-h"><h3>Stone analysis</h3><span class="hint">Measurement on the left of each range; 90% calibrated ranges from the frozen size-stratified calibration</span></div><div class="panel-b flush" id="stbl"></div></div>
    <div class="grid g2"><div class="panel" id="sdet"></div><div class="panel"><div class="panel-h"><h3>All stones · measurement and 90% range</h3><span class="hint">log scale</span></div><div class="panel-b"><div id="sint" class="plot"></div></div></div></div>`;
  const draw = () => {
    $("#stbl").innerHTML = stoneTable(s, id, false);
    $("#stbl").querySelectorAll("tr[data-s]").forEach(tr => tr.onclick = () => { STATE.sel = +tr.dataset.s; draw(); });
    const st = s.stones.find(x => x.stone_id === STATE.sel), g = st.range;
    const bar = (val, r, lo, hi, unit, d) => { const L = Math.log(lo), H = Math.log(hi), pos = x => (Math.log(x) - L) / (H - L) * 100;
      return `<div class="ibar"><div class="axis"></div>${r ? `<div class="band" style="left:${pos(r[0])}%;width:${pos(r[1]) - pos(r[0])}%"></div>` : ""}<div class="pt" style="left:${pos(val)}%"></div></div><div class="scale"><span>${fmt(lo, d)} ${unit}</span><span>${fmt(hi, d)} ${unit}</span></div>`; };
    const vlo = g ? Math.min(g.volume_range_mm3[0], st.volume_mm3) * 0.7 : st.volume_mm3 / 3, vhi = g ? Math.max(g.volume_range_mm3[1], st.volume_mm3) * 1.4 : st.volume_mm3 * 3;
    const dlo = g ? Math.min(g.diameter_range_mm[0], st.max_diameter_3d_mm) * 0.7 : st.max_diameter_3d_mm / 3, dhi = g ? Math.max(g.diameter_range_mm[1], st.max_diameter_3d_mm) * 1.4 : st.max_diameter_3d_mm * 3;
    $("#sdet").innerHTML = `<div class="panel-h"><h3>Stone ${st.stone_id}</h3><div class="toolbar"><a class="btn sm" href="#/study/${encodeURIComponent(id)}/imaging">Show in images</a><a class="btn sm" href="#/study/${encodeURIComponent(id)}/3d">Show in 3D</a></div></div><div class="panel-b">
      <div class="note" style="margin-bottom:6px">VOLUME</div>
      <div class="measure"><div><div class="h">Measurement</div><div class="big">${fmt(st.volume_mm3, 1)}<small>mm³</small></div><div class="note">${fmt(st.volume_ml, 3)} mL · ${st.voxel_count} voxels</div></div>
        <div><div class="h">90% calibrated range</div>${g ? `<div class="rng">${rng(g.volume_range_mm3, 0)} mm³</div>${bar(st.volume_mm3, g.volume_range_mm3, vlo, vhi, "mm³", 0)}` : `<div class="note">Not available for this study</div>`}</div></div>
      <div class="note" style="margin:12px 0 6px">3D MAXIMUM DIAMETER</div>
      <div class="measure"><div><div class="h">Measurement</div><div class="big">${fmt(st.max_diameter_3d_mm, 1)}<small>mm</small></div><div class="note">Size band ${esc(st.size_band)}</div></div>
        <div><div class="h">90% calibrated range</div>${g ? `<div class="rng">${rng(g.diameter_range_mm, 1)} mm</div>${bar(st.max_diameter_3d_mm, g.diameter_range_mm, dlo, dhi, "mm", 1)}` : `<div class="note">Not available for this study</div>`}</div></div>
      <dl class="kv mt14">
        <dt>Location (segmentation)</dt><dd>${esc(st.location.side)} · ${esc(st.location.position)}${st.location.within_kidney_mask === false ? " · centre not inside the kidney segmentation mask" : ""}</dd>
        <dt>Distance to kidney mask</dt><dd>${fmt(st.distance_to_kidney_mm, 1)} mm</dd>
        <dt>False-detection probability</dt><dd>${g ? `<span class="prob ${g.false_detection_probability >= 0.5 ? "hi" : ""}"><span style="width:${Math.round(g.false_detection_probability * 100)}%"></span></span>${fmt(g.false_detection_probability, 2)}` : "—"} <span class="note">· frozen development model (size, probability, TTA spread, threshold sensitivity, distance, attenuation)</span></dd>
        <dt>2D axial vs 3D diameter</dt><dd>${fmt(st.max_diameter_axial_mm, 1)} mm vs ${fmt(st.max_diameter_3d_mm, 1)} mm${st.category_changes_2d ? ` · <b>size band changes</b> (${esc(st.size_band_2d)} in 2D)` : " · same size band"}</dd>
        <dt>Extent x, y, z</dt><dd>${[st.extent_x_mm, st.extent_y_mm, st.extent_z_mm].map(v => fmt(v, 1)).join(" × ")} mm · ${st.slices_spanned ?? "—"} slices</dd>
        <dt>Attenuation mean / median / peak</dt><dd>${fmt(st.mean_hu, 0)} / ${fmt(st.median_hu, 0)} / ${fmt(st.max_hu, 0)} HU</dd>
        <dt>Shape</dt><dd>Sphericity ${fmt(st.sphericity, 2)} · hull fill ${fmt(st.hull_fill, 2)}</dd>
        <dt>QC</dt><dd>${chip(st.qc_status)} ${(st.qc_flags || []).map(f => `<span class="tag">${esc(f.replace(/_/g, " "))}</span>`).join(" ")}</dd></dl>
      <details class="more"><summary>How the ranges are produced</summary><div class="note mt8">Each range is a split-conformal interval on the log ratio between annotated and measured size, calibrated separately within the stone's predicted-size band (&lt;5, 5–10, ≥10 mm) on 35 locked calibration scans (Mondrian conformal prediction). It covers about 90% of stones of similar predicted size on data like the calibration set. It is not a probability that this stone's true size lies in the range, and attenuation (HU) does not indicate composition.</div></details></div>`;
    const ss = s.stones.filter(x => x.range);
    if (ss.length) Plotly.react("sint", [
      { type: "scatter", mode: "markers", x: ss.map(x => x.volume_mm3), y: ss.map(x => `Stone ${x.stone_id}`), error_x: { type: "data", symmetric: false, array: ss.map(x => x.range.volume_range_mm3[1] - x.volume_mm3), arrayminus: ss.map(x => x.volume_mm3 - x.range.volume_range_mm3[0]), color: C.accent, thickness: 1.5, width: 4 },
        marker: { color: ss.map(x => x.stone_id === STATE.sel ? C.sel : C.accent), size: 8 }, hovertemplate: "%{y}: %{x:.0f} mm³<extra></extra>" }],
      { ...LAYOUT, margin: { l: 70, r: 16, t: 8, b: 44 }, xaxis: { ...AX, type: "log", title: { text: "Volume (mm³), measurement with 90% range" } }, yaxis: { ...AX, autorange: "reversed" } }, PLOT);
    else $("#sint").innerHTML = `<div class="empty">No calibrated ranges for this study.</div>`;
  };
  draw();
}

// ------------------------------------------------------------------ burden & uncertainty
function tabBurden(s, id, el) {
  const b = (s.ranges || {}).burden;
  if (!b) { el.innerHTML = `<div class="panel"><div class="empty"><h3>No calibrated burden interval</h3>${esc(s.eligibility.detail)}</div></div>`; return; }
  const d = b.decomposition, cq = b.comparators["P2-B-CQR"], kept = s.stones.filter(x => x.range);
  const bands = Object.entries(d.measurement_residual_bands).map(([k, v]) => `${k.replace(">=", "≥").replace("5-10", "5–10")}: ×${fmt(v.relative_error_p05_p95[0], 2)}–×${fmt(v.relative_error_p05_p95[1], 2)} (n = ${v.n})`).join("<br>");
  el.innerHTML = `
  <div class="grid g4 mb14">
    <div class="panel metric"><div class="label">Retained stones</div><div class="value">${d.retained_pieces}</div><div class="note">after the 30 mm kidney-distance filter</div></div>
    <div class="panel metric"><div class="label">Measured burden</div><div class="value">${fmt(b.predicted_mm3, 0)}<small>mm³</small></div><div class="note">sum of retained stone volumes</div></div>
    <div class="panel metric primary"><div class="label">Detection-aware interval · 90%</div><div class="value">${rng(b.interval_mm3)}<small>mm³</small></div><div class="note">frozen calibration (P2-A)</div></div>
    <div class="panel metric"><div class="label">Comparator interval · 90%</div><div class="value" style="color:var(--ink-2)">${rng(cq)}<small>mm³</small></div><div class="note">conformalised quantile regression (P2-B-CQR)</div></div>
  </div>
  <div class="panel mb14"><div class="panel-h"><h3>How the burden interval is built</h3><span class="hint">Intermediate quantities of the same frozen calculation for this scan</span></div><div class="panel-b">
    <div class="decomp">
      <div class="dstep"><div class="k">01 · OBSERVED</div><div class="t">Retained pieces</div><div class="v">${fmt(d.measured_mm3, 0)}<small> mm³</small></div><div class="d">${d.retained_pieces} pieces measured in 3D from the segmentation.</div></div>
      <div class="dstep"><div class="k">02 · FALSE DETECTIONS</div><div class="t">False-detection modelling</div><div class="v">${fmt(d.expected_after_false_detection_mm3, 0)}<small> mm³ expected</small></div><div class="d">Each piece kept in a draw with probability 1 − π; expected volume Σ(1 − π)·V.</div></div>
      <div class="dstep"><div class="k">03 · MEASUREMENT</div><div class="t">Measurement uncertainty</div><div class="v" style="font-size:13px;line-height:1.5">${bands}</div><div class="d">Multiplicative error (5th–95th percentile) resampled from development residuals of the same size band.</div></div>
      <div class="dstep"><div class="k">04 · MISSED STONES</div><div class="t">Missed burden</div><div class="v">${fmt(d.missed_burden_pool.p90_mm3, 0)}<small> mm³ (90th pct.)</small></div><div class="d">Development per-scan missed burden, ${esc(d.missed_burden_pool.slice_group)}-slice group (n = ${d.missed_burden_pool.n_scans}; ${fmt(d.missed_burden_pool.share_nonzero * 100, 0)}% non-zero).</div></div>
      <div class="dstep result"><div class="k">05 · CALIBRATED</div><div class="t">Conformal calibration</div><div class="v">${rng(d.calibrated_interval_mm3)}<small> mm³</small></div><div class="d">Raw ${fmt(d.draws, 0)}-draw interval ${rng(d.raw_interval_mm3)} mm³ widened by one factor t = ${fmt(d.conformal_factor_t, 3)} on the log scale (c = ${fmt(d.offset_c_mm3, 0)} mm³).</div></div>
    </div>
    <div id="bline" class="plot mt14" style="height:230px"></div></div></div>
  <div class="grid g2">
    <div class="panel"><div class="panel-h"><h3>Per-stone contribution</h3><span class="hint">measured volume and expected volume after false-detection modelling</span></div><div class="panel-b"><div id="bcontrib" class="plot"></div></div></div>
    <div class="panel"><div class="panel-h"><h3>Method and status</h3></div><div class="panel-b">
      <dl class="kv"><dt>Nominal coverage</dt><dd>90% (α = 0.10)</dd><dt>Calibration</dt><dd>35 locked Li 2022 non-contrast scans; frozen ${chip(s.integrity.status)}</dd>
      <dt>Calibration SHA-256</dt><dd class="mono">${esc((s.integrity.calibration_sha256 || "").slice(0, 24))}…</dd><dt>Protocol</dt><dd>${esc(s.integrity.protocol)}</dd>
      <dt>Locked-test evidence</dt><dd>50 of 53 scans covered; median width ratio to the comparator 0.573 (95% CI 0.540–0.656)</dd>
      <dt>Status</dt><dd>Research demonstration of the frozen method on this scan; not a validated clinical measurement.</dd></dl>
      <details class="more"><summary>Formula</summary><div class="formula mt8">B̃⁽ᵈ⁾ = Σⱼ 1[uⱼ ≥ πⱼ] · V̂ⱼ · exp(εⱼ⁽ᵈ⁾) + M⁽ᵈ⁾,  d = 1…${d.draws}<br>[L, U] = 5th and 95th percentiles of B̃<br>interval = [max((L + c)·e⁻ᵗ − c, 0),  (U + c)·eᵗ − c]</div>
        <div class="note mt8">π: false-detection probability per piece; ε: log residual from the same size band; M: missed burden from the same slice-thickness group. Only t is calibrated; all other components come from development data. Nothing is fitted on this scan.</div></details>
      ${s.eligibility.status !== "ELIGIBLE" ? `<div class="callout warn mt14">${esc(s.eligibility.detail)}</div>` : ""}
    </div></div>
  </div>`;
  const y = ["Comparator (CQR)", "Detection-aware (calibrated)", "Raw Monte Carlo", "Measured"];
  const tr = [
    { type: "scatter", mode: "lines", x: cq, y: [y[0], y[0]], line: { color: C.cmp, width: 8 }, hovertemplate: "Comparator: %{x:.0f} mm³<extra></extra>" },
    { type: "scatter", mode: "lines", x: d.calibrated_interval_mm3, y: [y[1], y[1]], line: { color: C.accent, width: 10 }, hovertemplate: "Calibrated: %{x:.0f} mm³<extra></extra>" },
    { type: "scatter", mode: "lines", x: d.raw_interval_mm3, y: [y[2], y[2]], line: { color: C.soft, width: 8 }, hovertemplate: "Raw: %{x:.0f} mm³<extra></extra>" },
    { type: "scatter", mode: "markers", x: [b.predicted_mm3], y: [y[3]], marker: { color: "#16212a", size: 10, symbol: "line-ns-open", line: { width: 3 } }, hovertemplate: "Measured: %{x:.0f} mm³<extra></extra>" },
    { type: "scatter", mode: "markers", x: [d.expected_after_false_detection_mm3], y: [y[3]], marker: { color: C.review, size: 9, symbol: "diamond" }, hovertemplate: "Expected after false-detection modelling: %{x:.0f} mm³<extra></extra>" },
  ];
  const note = (x, text, ay) => ({ x, y: y[3], text, showarrow: true, arrowhead: 0, arrowcolor: "#8b96a0", ax: 0, ay, font: { size: 11, color: "#3a4752" }, bgcolor: "#ffffff" });
  const xmax = Math.max(cq[1], d.calibrated_interval_mm3[1], d.raw_interval_mm3[1], b.predicted_mm3);
  // a lower-end label next to zero would collide with the row label: write it after the upper end instead
  const ends = (iv, row, col) => iv[0] < 0.04 * xmax ? [{ x: iv[1], y: row, text: `${fmt(iv[0], 0)} – ${fmt(iv[1], 0)}`, showarrow: false, xanchor: "left", xshift: 6, font: { size: 10.5, color: col } }] : [{ x: iv[0], y: row, text: fmt(iv[0], 0), showarrow: false, xanchor: "right", xshift: -6, font: { size: 10.5, color: col } },
    { x: iv[1], y: row, text: fmt(iv[1], 0), showarrow: false, xanchor: "left", xshift: 6, font: { size: 10.5, color: col } }];
  Plotly.react("bline", tr, { ...LAYOUT, showlegend: false, margin: { l: 190, r: 40, t: 44, b: 40 }, xaxis: { ...AX, title: { text: "Stone burden (mm³)" }, rangemode: "tozero" }, yaxis: { ...AX, categoryorder: "array", categoryarray: y },
    annotations: [note(b.predicted_mm3, `measured ${fmt(b.predicted_mm3, 0)}`, -18), note(d.expected_after_false_detection_mm3, `expected after false-detection modelling ${fmt(d.expected_after_false_detection_mm3, 0)}`, -36),
      ...ends(d.calibrated_interval_mm3, y[1], C.accent), ...ends(d.raw_interval_mm3, y[2], "#66737e"), ...ends(cq, y[0], "#66737e")] }, PLOT);
  if (kept.length) Plotly.react("bcontrib", [
    { type: "bar", orientation: "h", y: kept.map(x => `Stone ${x.stone_id}`), x: kept.map(x => x.volume_mm3), name: "Measured", marker: { color: "#c9d6dc" }, hovertemplate: "Measured %{x:.0f} mm³<extra></extra>" },
    { type: "bar", orientation: "h", y: kept.map(x => `Stone ${x.stone_id}`), x: kept.map(x => x.volume_mm3 * (1 - x.range.false_detection_probability)), name: "Expected after false-detection modelling", marker: { color: C.accent }, hovertemplate: "Expected %{x:.0f} mm³<extra></extra>" }],
    { ...LAYOUT, barmode: "overlay", margin: { l: 70, r: 16, t: 34, b: 44 }, legend: { orientation: "h", x: 0, y: 1.1, font: { size: 11 } }, xaxis: { ...AX, title: { text: "Volume (mm³)" } }, yaxis: { ...AX, autorange: "reversed" } }, PLOT);
}

// ------------------------------------------------------------------ quality & provenance
function tabQC(s, id, el) {
  const r = s.report, groups = [...new Set(s.qc_items.map(i => i.group))], st = r.stages || [], sw = r.software || {};
  el.innerHTML = `<div class="grid g-main"><div class="grid" style="align-content:start">
    ${groups.map(g => `<div class="panel"><div class="panel-h"><h3>${esc(g)}</h3></div><div class="panel-b flush"><table class="t"><tbody>${s.qc_items.filter(i => i.group === g).map(i => `<tr><td style="width:34%"><b>${esc(i.item)}</b></td><td style="width:130px">${chip(i.status)}</td><td class="note">${esc(i.detail)}</td></tr>`).join("")}</tbody></table></div></div>`).join("")}
    <div class="panel"><div class="panel-h"><h3>Processing timeline</h3><span class="hint">${fmt(st.reduce((a, x) => a + (x.seconds || 0), 0), 0)} s pipeline</span></div><div class="panel-b"><div id="tl" class="plot" style="height:${60 + st.length * 26}px"></div></div></div></div>
  <div class="grid" style="align-content:start">
    <div class="panel"><div class="panel-h"><h3>Frozen artefacts</h3>${chip(s.integrity.status)}</div><div class="panel-b flush"><table class="t"><tbody>${s.integrity.rows.map(x => `<tr><td>${esc(x.artefact)}<div class="sub mono">${esc((x.actual || "missing").slice(0, 16))}…</div></td><td class="right">${chip(x.status)}</td></tr>`).join("")}</tbody></table>
      <div class="panel-b note">${esc(s.integrity.summary)}. SHA-256 recomputed at start-up and compared with the T1 freeze manifest and the recorded calibration hash.</div></div></div>
    <div class="panel"><div class="panel-h"><h3>Software</h3></div><div class="panel-b"><dl class="kv">${Object.entries(sw).map(([k, v]) => `<dt>${esc(k)}</dt><dd class="mono">${esc(v)}</dd>`).join("")}</dl></div></div>
    <div class="panel"><div class="panel-h"><h3>Model and configuration</h3></div><div class="panel-b"><dl class="kv"><dt>Model</dt><dd>${esc((r.model_info || {}).name || "—")}</dd><dt>Checkpoint SHA-256</dt><dd class="mono">${esc(((r.model_info || {}).checkpoint_sha256 || "").slice(0, 24))}…</dd><dt>Configuration SHA-256</dt><dd class="mono">${esc((r.config_sha256 || "").slice(0, 24))}…</dd><dt>Preprocessing</dt><dd>${(r.preprocessing || []).map(esc).join("<br>")}</dd></dl></div></div>
  </div></div>`;
  if (st.length) Plotly.react("tl", [{ type: "bar", orientation: "h", y: st.map(x => x.stage.replace(/_/g, " ")), x: st.map(x => x.seconds || 0), marker: { color: st.map(x => x.status === "ok" ? C.accent : C.review) }, hovertemplate: "%{y}: %{x:.1f} s<extra></extra>" }],
    { ...LAYOUT, margin: { l: 250, r: 16, t: 4, b: 36 }, xaxis: { ...AX, title: { text: "Seconds" } }, yaxis: { ...AX, autorange: "reversed" } }, PLOT);
}

function tabReport(s, id, el) {
  el.innerHTML = `<div class="panel"><div class="panel-h"><h3>Structured report</h3><div class="toolbar"><a class="btn sm" href="/report/${encodeURIComponent(id)}" target="_blank">Open in new window</a><button class="btn sm primary" id="pr">Print / Save as PDF</button><a class="btn sm" href="/export/${encodeURIComponent(id)}.json">Export JSON</a></div></div>
    <iframe id="rep" title="Structured report" src="/report/${encodeURIComponent(id)}?embed=1" style="width:100%;height:78vh;border:0;background:#eef1f3"></iframe></div>`;
  $("#pr").onclick = () => $("#rep").contentWindow.print();
}

// ------------------------------------------------------------------ research evidence
async function renderEvidence() {
  app().innerHTML = `<div class="page-head"><div><h1>Research evidence</h1><div class="sub">Committed study results behind the StoneKit method (Kadari, Surepalli and Kolisetty, 2026). These are aggregate study results, not results for any single scan on this workstation.</div></div></div><div id="ev">${loading("Loading study results…")}</div>`;
  let e; try { e = STATE.evidence || (STATE.evidence = await api("/api/evidence")); } catch (x) { $("#ev").innerHTML = errorBox(x); return; }
  const lt = e.locked_test, ext = e.external;
  $("#ev").innerHTML = `
  <div class="grid g4 mb14">
    <div class="panel metric primary"><div class="label">Locked test · width ratio</div><div class="value">${fmt(lt.ratio.median, 3)}</div><div class="note">95% CI ${fmt(lt.ratio.ci95[0], 3)}–${fmt(lt.ratio.ci95[1], 3)} · formal test ${lt.formal.P2_success ? "met" : "not met"}</div></div>
    <div class="panel metric"><div class="label">Locked test · coverage</div><div class="value">${Math.round(lt.table[3].coverage * lt.table[3].n)}/${lt.table[3].n}</div><div class="note">nominal 90% · ${lt.n} locked scans, evaluated once</div></div>
    <div class="panel metric"><div class="label">Stone-bearing scans (post hoc)</div><div class="value">${e.posthoc_stone_bearing ? fmt(e.posthoc_stone_bearing.median, 3) : "—"}</div><div class="note">${e.posthoc_stone_bearing ? `ratio, 95% CI ${fmt(e.posthoc_stone_bearing.ci95[0], 3)}–${fmt(e.posthoc_stone_bearing.ci95[1], 3)} (n = ${e.posthoc_stone_bearing.n})` : ""}</div></div>
    <div class="panel metric"><div class="label">External check (MSWAL)</div><div class="value">${Math.round(ext.burden["P2-A decomposition"].coverage * ext.eligible)}/${ext.eligible}</div><div class="note">burden coverage; width ratio ${fmt(ext.burden.width_ratio_A_over_BCQR.median, 2)} (descriptive)</div></div>
  </div>
  <div class="panel mb14"><div class="panel-h"><h3>Locked test · patient-burden intervals at nominal 90%</h3><span class="hint">53 scans, 11 stone-free · confirmatory</span></div><div class="panel-b flush"><table class="t"><thead><tr><th>Method</th><th class="n">Coverage</th><th class="n">Clopper–Pearson 95%</th><th class="n">Stone-free</th><th class="n">Median width (mm³)</th></tr></thead><tbody>
    ${lt.table.map(r => `<tr ${r.key === "P2-A decomposition" ? 'class="sel"' : ""}><td><b>${esc(r.method)}</b></td><td class="n">${Math.round(r.coverage * r.n)}/${r.n} (${fmt(r.coverage, 3)})</td><td class="n">${fmt(r.cp95[0], 3)}–${fmt(r.cp95[1], 3)}</td><td class="n">${r.stone_free == null ? "—" : `${Math.round(r.stone_free * 11)}/11`}</td><td class="n">${fmt(r.median_width_mm3, 0)}</td></tr>`).join("")}</tbody></table></div></div>
  <div class="grid g2 mb14">
    <div class="panel"><div class="panel-h"><h3>Robustness · width ratio to comparator</h3><span class="hint">below 1 = narrower</span></div><div class="panel-b"><div id="evr" class="plot"></div></div></div>
    <div class="panel"><div class="panel-h"><h3>Per-stone coverage by predicted size · locked test</h3><span class="hint">3D diameter, nominal 90%</span></div><div class="panel-b"><div id="evb" class="plot"></div></div></div>
  </div>
  <div class="grid g2 mb14">
    <div class="panel"><div class="panel-h"><h3>Physical phantom · volume error by composition</h3><span class="hint">Freiburg, 101 ex-vivo stones · ICC 0.993</span></div><div class="panel-b"><div id="evp" class="plot"></div></div></div>
    <div class="panel"><div class="panel-h"><h3>Physical phantom · 2D vs 3D diameter</h3><span class="hint">automated minus caliper</span></div><div class="panel-b"><div id="evd" class="plot"></div></div></div>
  </div>
  <div class="panel"><div class="panel-h"><h3>External check · MSWAL (descriptive)</h3><span class="hint">frozen system and calibration applied without refitting</span></div><div class="panel-b"><dl class="kv">
    <dt>Scans</dt><dd>${ext.eligible} non-contrast of ${ext.cases} stone-bearing</dd><dt>Burden coverage</dt><dd>${fmt(ext.burden["P2-A decomposition"].coverage, 3)} (Clopper–Pearson ${fmt(ext.burden["P2-A decomposition"].cp95[0], 3)}–${fmt(ext.burden["P2-A decomposition"].cp95[1], 3)})</dd>
    <dt>Width ratio vs comparator</dt><dd>${fmt(ext.burden.width_ratio_A_over_BCQR.median, 3)} (${fmt(ext.burden.width_ratio_A_over_BCQR.ci95[0], 3)}–${fmt(ext.burden.width_ratio_A_over_BCQR.ci95[1], 3)}): the width advantage did not transfer</dd>
    <dt>Per-stone coverage U2 (vol. / diam.)</dt><dd>${fmt(ext.per_stone["volume_mm3|U2"].coverage, 3)} / ${fmt(ext.per_stone["max_diameter_3d_mm|U2"].coverage, 3)} · global U1 ${fmt(ext.per_stone["volume_mm3|U1"].coverage, 3)} / ${fmt(ext.per_stone["max_diameter_3d_mm|U1"].coverage, 3)}</dd>
    <dt>False positives per scan</dt><dd>${fmt(ext.fp_per_scan, 2)}</dd><dt>Reading</dt><dd>Burden coverage transferred; interval width and per-stone volume calibration did not. Local recalibration is needed at a new site.</dd></dl></div></div>`;
  const rb = e.robustness;
  Plotly.react("evr", [{ type: "scatter", mode: "markers", y: rb.map(r => `${r.analysis} (n=${r.n})`), x: rb.map(r => +r.ratio),
    error_x: { type: "data", symmetric: false, array: rb.map(r => r.ratio_hi - r.ratio), arrayminus: rb.map(r => r.ratio - r.ratio_lo), color: C.accent, thickness: 1.5, width: 4 },
    marker: { size: 8, color: rb.map(r => r.kind === "external" ? "#ffffff" : C.accent), line: { color: C.accent, width: 1.5 } }, hovertemplate: "%{y}: %{x:.3f}<extra></extra>" }],
    { ...LAYOUT, margin: { l: 230, r: 16, t: 8, b: 44 }, shapes: [{ type: "line", x0: 1, x1: 1, yref: "paper", y0: 0, y1: 1, line: { color: "#8b96a0", dash: "dot", width: 1 } }],
      xaxis: { ...AX, type: "log", title: { text: "Paired median width ratio (95% CI)" }, tickvals: [0.4, 0.6, 0.8, 1, 1.4] }, yaxis: { ...AX, autorange: "reversed" } }, PLOT);
  const sb = e.size_bands.filter(r => r.target === "max_diameter_3d_mm" && r.data === "locked test"), bandName = { "<5": "<5 mm", "5-<10": "5–10 mm", ">=10": "≥10 mm" };
  const meth = [["U0", "Bland–Altman", "#8b96a0"], ["U1", "Global conformal", C.stone], ["U2", "Size-stratified (StoneKit)", C.accent]];
  Plotly.react("evb", meth.map(([m, n, col]) => { const rr = sb.filter(r => r.method === m);
    return { type: "scatter", mode: "markers", name: n, x: rr.map(r => bandName[r.band]), y: rr.map(r => +r.coverage), error_y: { type: "data", symmetric: false, array: rr.map(r => r.cp_high - r.coverage), arrayminus: rr.map(r => r.coverage - r.cp_low), color: col, thickness: 1.2, width: 3 }, marker: { color: col, size: 8 }, hovertemplate: `${n}: %{y:.3f}<extra></extra>` }; }),
    { ...LAYOUT, scattermode: "group", margin: { l: 56, r: 16, t: 8, b: 66 }, legend: { orientation: "h", y: -0.22, font: { size: 11 } }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.9, y1: 0.9, line: { color: "#8b96a0", dash: "dot", width: 1 } }],
      xaxis: { ...AX, title: { text: "Predicted size band" } }, yaxis: { ...AX, title: { text: "Per-stone coverage" }, range: [0.35, 1.03] } }, PLOT);
  const ph = e.phantom.filter(r => r.type), names = { "uric acid": "Uric acid", "not recorded": "Not recorded", other: "Other types", cystine: "Cystine", "calciumoxalate-dihydrate": "Ca oxalate dihydrate", brushite: "Brushite" };
  Plotly.react("evp", [{ type: "scatter", mode: "markers", y: ph.map(r => `${names[r.type] || r.type} (${r.n})`), x: ph.map(r => 100 * r.median),
    error_x: { type: "data", symmetric: false, array: ph.map(r => 100 * (r.ci_high - r.median)), arrayminus: ph.map(r => 100 * (r.median - r.ci_low)), color: C.accent, thickness: 1.5, width: 4 }, marker: { color: C.accent, size: 8 }, hovertemplate: "%{y}: %{x:.1f}%<extra></extra>" }],
    { ...LAYOUT, margin: { l: 170, r: 16, t: 8, b: 44 }, shapes: [{ type: "line", x0: 23.6, x1: 23.6, yref: "paper", y0: 0, y1: 1, line: { color: "#8b96a0", dash: "dot", width: 1 } }], xaxis: { ...AX, title: { text: "Median absolute volume error (%) · dotted: all stones 23.6%" } }, yaxis: { ...AX } }, PLOT);
  const p3 = e.phantom.filter(r => r.panel_b === "3D maximum diameter"), p2 = e.phantom.filter(r => r.panel_b === "2D axial maximum diameter");
  Plotly.react("evd", [
    { type: "scatter", mode: "markers", name: "3D maximum", x: p3.map(r => +r.caliper_mm), y: p3.map(r => +r.error_mm), marker: { color: C.accent, size: 6 }, hovertemplate: "3D: %{y:.2f} mm<extra></extra>" },
    { type: "scatter", mode: "markers", name: "2D axial maximum", x: p2.map(r => +r.caliper_mm), y: p2.map(r => +r.error_mm), marker: { color: "#ffffff", size: 6, symbol: "triangle-up", line: { color: C.stone, width: 1.3 } }, hovertemplate: "2D: %{y:.2f} mm<extra></extra>" }],
    { ...LAYOUT, margin: { l: 56, r: 16, t: 8, b: 66 }, legend: { orientation: "h", y: -0.22, font: { size: 11 } }, xaxis: { ...AX, title: { text: "Caliper diameter (mm)" } }, yaxis: { ...AX, title: { text: "Error vs caliper (mm)" }, zeroline: true } }, PLOT);
}

// ------------------------------------------------------------------ methodology
function renderMethod() {
  app().innerHTML = `<div class="page-head"><div><h1>Methodology</h1><div class="sub">How StoneKit measures kidney stones and attaches calibrated uncertainty. Frozen protocol v3; research software.</div></div></div>
  <div class="grid g2">
    <div class="panel"><div class="panel-h"><h3>Measurement pipeline</h3></div><div class="panel-b"><ol class="list-tight">
      <li><b>Input validation.</b> Orientation, spacing and header checks; the CT is reordered to RAS without interpolation, and no HU value is changed.</li>
      <li><b>Contrast check.</b> Aorta median attenuation below 100 HU (heuristic). The calibration covers non-contrast CT only.</li>
      <li><b>Kidney segmentation.</b> TotalSegmentator 2.18.0; a region of interest with a 40 mm margin is cropped around both kidneys.</li>
      <li><b>Stone segmentation.</b> nnU-Net v2 2.8.1 (3D full resolution, 100 epochs, fold 0) with 8-flip test-time augmentation.</li>
      <li><b>Pieces and filter.</b> 26-connected components of at least 3 voxels; pieces more than 30 mm from a kidney are removed (pre-specified).</li>
      <li><b>3D measurement.</b> Volume from voxels; maximum 3D diameter as the largest voxel-centre distance, on the original grid.</li></ol></div></div>
    <div class="panel"><div class="panel-h"><h3>Calibrated uncertainty</h3></div><div class="panel-b">
      <p style="margin-top:0"><b>Per stone.</b> A split-conformal interval on |log(reference / measured)|, calibrated separately within predicted-size bands (&lt;5, 5–10, ≥10 mm; Mondrian conformal prediction). Nominal 90%.</p>
      <p><b>Per patient.</b> The detection-aware burden interval propagates three error sources by Monte Carlo (2,000 draws): each piece's false-detection probability, measurement residuals of its size band, and missed burden from scans with the same slice thickness. The raw interval is then widened by one conformal factor t learnt on calibration scans.</p>
      <div class="formula">interval = [max((L + c)·e⁻ᵗ − c, 0),  (U + c)·eᵗ − c],   c = 5 mm³</div></div></div>
    <div class="panel"><div class="panel-h"><h3>Data roles</h3></div><div class="panel-b"><table class="t"><tbody>
      <tr><td><b>Development</b></td><td>165 Li 2022 scans</td><td class="note">model training, out-of-fold fitting of false-detection model and residual and missed-burden pools</td></tr>
      <tr><td><b>Calibration</b></td><td>35 locked scans</td><td class="note">every conformal quantity, hashed before the test scans were loaded</td></tr>
      <tr><td><b>Locked test</b></td><td>53 locked scans</td><td class="note">evaluated once under the frozen protocol</td></tr>
      <tr><td><b>External check</b></td><td>58 MSWAL scans</td><td class="note">descriptive; no refitting</td></tr>
      <tr><td><b>Physical reference</b></td><td>101 Freiburg stones</td><td class="note">phantom with water-displacement volume and caliper diameter</td></tr></tbody></table></div></div>
    <div class="panel"><div class="panel-h"><h3>Limitations and intended use</h3></div><div class="panel-b"><ul class="list-tight">
      <li>Reference standard is expert annotation from one centre, not physical truth; no radiology-trained reader adjudicated flagged annotations.</li>
      <li>Ranges cover detected stones; the burden interval accounts for missed stones statistically, not by finding them.</li>
      <li>On an external dataset, burden coverage held but the width advantage and per-stone volume calibration did not transfer; recalibrate locally before use at another site.</li>
      <li>Attenuation (HU) is reported as measured and does not indicate stone composition.</li>
      <li>The workstation states measurements and calibrated intervals only; it makes no interpretation of them.</li></ul>
      <div class="note mt14">Source code: github.com/SaiPraneeth9955/stonekit (MIT licence).</div></div></div>
  </div>`;
}

checkIntegrity(); route();
