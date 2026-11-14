// StoneKit CT viewer: axial / coronal / sagittal views of the kidney region on the original voxel grid (no resampling).
// Arrays are RAS (x toward patient Right... index increases as stored), C order of shape (nx, ny, nz).
// Radiological display: patient Right on screen left (axial, coronal); Anterior up (axial); Superior up (coronal, sagittal).
"use strict";

async function unpack(b64, T) {
  if (!b64) return null;
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const s = new Blob([bin]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new T(await new Response(s).arrayBuffer());
}

const PRESETS = [["Stone", 500, 2000], ["Bone", 300, 1500], ["Soft tissue", 40, 400], ["Abdomen", 50, 350], ["Kidney", 30, 300]];

class CTViewer {
  constructor(root, payload, stones, onSelect) {
    this.root = root; this.P = payload; this.stoneInfo = stones || []; this.onSelect = onSelect || (() => {});
    this.L = 500; this.W = 2000; this.op = 0.5; this.showK = true; this.showS = true; this.showX = true; this.sel = 0;
    this.single = false;
  }
  async init() {
    const P = this.P;
    [this.nx, this.ny, this.nz] = P.shape; [this.sx, this.sy, this.sz] = P.spacing;
    [this.CT, this.KD, this.ST] = await Promise.all([unpack(P.ct, Int16Array), unpack(P.kidney, Uint8Array), unpack(P.stones, Uint16Array)]);
    this.cur = [this.nx >> 1, this.ny >> 1, this.nz >> 1];
    this.buildDom(); this.bind(); this.drawAll(); this.readout(this.cur);
    new ResizeObserver(() => this.drawAll()).observe(this.grid);
    return this;
  }
  at(x, y, z) { return (x * this.ny + y) * this.nz + z; }
  buildDom() {
    const { nx, ny, nz, sx, sy, sz } = this;
    this.root.innerHTML = `
    <div class="vtools">
      <span class="note">Window</span><div class="seg" data-r="presets">${PRESETS.map(([n, l, w]) => `<button data-l="${l}" data-w="${w}" title="W ${w} / L ${l}">${n}</button>`).join("")}</div>
      <span class="sep"></span>
      <button class="btn sm on" data-r="tk" title="Show the kidney segmentation outline">Kidneys</button>
      <button class="btn sm on" data-r="ts" title="Show the stone segmentation">Stones</button>
      <button class="btn sm on" data-r="tx" title="Show the linked crosshair">Crosshair</button>
      <label>Opacity</label><input type="range" data-r="op" min="0" max="100" value="50">
      <span class="sep"></span>
      <button class="btn sm" data-r="zin" title="Zoom in (Ctrl + wheel)">Zoom +</button>
      <button class="btn sm" data-r="zout" title="Zoom out">Zoom −</button>
      <button class="btn sm" data-r="reset" title="Reset zoom, pan and window">Reset</button>
      <span class="sep"></span>
      <div class="seg" data-r="layout"><button class="on" data-v="3">Tri-planar</button><button data-v="1">Axial</button></div>
      <span class="sep"></span>
      <label>Go to</label><select class="input" data-r="jump" style="height:26px"><option value="0">Select stone…</option>${this.stoneInfo.map(s => `<option value="${s.stone_id}">Stone ${s.stone_id} · ${fmt(s.volume_mm3, 0)} mm³</option>`).join("")}</select>
    </div>
    <div class="viewer" data-r="grid"><div class="vp ax"></div><div class="vp"></div><div class="vp"></div></div>`;
    const q = r => this.root.querySelector(`[data-r="${r}"]`);
    this.q = q; this.grid = q("grid");
    const vps = this.grid.querySelectorAll(".vp");
    this.V = [
      { name: "Axial", fix: 2, cols: nx, rows: ny, pw: sx, ph: sy, map: (c, r, k) => [nx - 1 - c, ny - 1 - r, k], inv: (x, y) => [nx - 1 - x, ny - 1 - y], lab: ["R", "L", "A", "P"], n: nz },
      { name: "Coronal", fix: 1, cols: nx, rows: nz, pw: sx, ph: sz, map: (c, r, k) => [nx - 1 - c, k, nz - 1 - r], inv: (x, y, z) => [nx - 1 - x, nz - 1 - z], lab: ["R", "L", "S", "I"], n: ny },
      { name: "Sagittal", fix: 0, cols: ny, rows: nz, pw: sy, ph: sz, map: (c, r, k) => [k, ny - 1 - c, nz - 1 - r], inv: (x, y, z) => [ny - 1 - y, nz - 1 - z], lab: ["A", "P", "S", "I"], n: nx },
    ];
    this.V.forEach((v, i) => {
      v.el = vps[i]; v.zoom = 1; v.pan = [0, 0];
      v.cv = document.createElement("canvas"); v.cv.width = v.cols; v.cv.height = v.rows; v.el.appendChild(v.cv);
      v.oc = document.createElement("canvas"); v.oc.className = "ov"; v.el.appendChild(v.oc);
      v.el.insertAdjacentHTML("beforeend", `<div class="c tl"></div><div class="c tr"></div><div class="c bl"></div><div class="c br"></div>
        <div class="o ol">${v.lab[0]}</div><div class="o or">${v.lab[1]}</div><div class="o ot">${v.lab[2]}</div><div class="o ob">${v.lab[3]}</div>`);
      v.img = v.cv.getContext("2d").createImageData(v.cols, v.rows);
    });
    this.setPresetActive();
  }
  setPresetActive() {
    this.q("presets").querySelectorAll("button").forEach(b => b.classList.toggle("on", +b.dataset.l === this.L && +b.dataset.w === this.W));
  }
  layout(v) {
    const r = v.el.getBoundingClientRect(); const wmm = v.cols * v.pw, hmm = v.rows * v.ph;
    const s = Math.min(r.width / wmm, r.height / hmm) * v.zoom; v.scale = s;
    const w = wmm * s, h = hmm * s; v.x0 = (r.width - w) / 2 + v.pan[0]; v.y0 = (r.height - h) / 2 + v.pan[1];
    Object.assign(v.cv.style, { left: v.x0 + "px", top: v.y0 + "px", width: w + "px", height: h + "px" });
    v.oc.width = Math.max(1, r.width); v.oc.height = Math.max(1, r.height);
  }
  draw(i) {
    const v = this.V[i]; if (!v.el.offsetParent) return;
    const k = this.cur[v.fix], d = v.img.data, lo = this.L - this.W / 2, W = this.W, a = this.op, CT = this.CT, KD = this.KD, ST = this.ST, sel = this.sel;
    const KC = [[0, 0, 0], [60, 156, 154], [95, 134, 179]], SC = [217, 130, 43], SEL = [31, 111, 209];
    for (let r = 0; r < v.rows; r++) for (let c = 0; c < v.cols; c++) {
      const [x, y, z] = v.map(c, r, k), q = this.at(x, y, z);
      let g = (CT[q] - lo) / W * 255; g = g < 0 ? 0 : g > 255 ? 255 : g; let R = g, G = g, B = g; const o = (r * v.cols + c) * 4;
      if (this.showS && ST) {
        const s = ST[q];
        if (s) {
          const col = s === sel ? SEL : SC; let edge = false;
          for (const [dc, dr] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const cc = c + dc, rr = r + dr;
            if (cc < 0 || rr < 0 || cc >= v.cols || rr >= v.rows) { edge = true; break; } const p = v.map(cc, rr, k); if (ST[this.at(...p)] !== s) { edge = true; break; } }
          const f = edge ? Math.max(a, 0.9) : a * 0.45; R = R * (1 - f) + col[0] * f; G = G * (1 - f) + col[1] * f; B = B * (1 - f) + col[2] * f;
        }
      }
      if (this.showK && KD && KD[q]) {
        let edge = false;
        for (const [dc, dr] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const cc = c + dc, rr = r + dr;
          if (cc < 0 || rr < 0 || cc >= v.cols || rr >= v.rows) continue; const p = v.map(cc, rr, k); if (KD[this.at(...p)] !== KD[q]) { edge = true; break; } }
        if (edge) { const f = Math.max(a, 0.65), col = KC[KD[q]]; R = R * (1 - f) + col[0] * f; G = G * (1 - f) + col[1] * f; B = B * (1 - f) + col[2] * f; }
      }
      d[o] = R; d[o + 1] = G; d[o + 2] = B; d[o + 3] = 255;
    }
    v.cv.getContext("2d").putImageData(v.img, 0, 0); this.layout(v); this.overlay(i);
  }
  overlay(i) {
    const v = this.V[i], g = v.oc.getContext("2d"); g.clearRect(0, 0, v.oc.width, v.oc.height);
    const [cx, cy] = v.inv(...this.cur);
    if (this.showX) {
      g.strokeStyle = "rgba(120,190,215,0.55)"; g.setLineDash([4, 4]); g.lineWidth = 1;
      const px = v.x0 + (cx + 0.5) * v.pw * v.scale, py = v.y0 + (cy + 0.5) * v.ph * v.scale;
      g.beginPath(); g.moveTo(px, 0); g.lineTo(px, v.oc.height); g.moveTo(0, py); g.lineTo(v.oc.width, py); g.stroke(); g.setLineDash([]);
    }
    const len = 10 * v.scale, x1 = v.oc.width - 18 - len, y1 = v.oc.height - 24;
    g.strokeStyle = "#d7dee4"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 + len, y1);
    g.moveTo(x1, y1 - 4); g.lineTo(x1, y1 + 4); g.moveTo(x1 + len, y1 - 4); g.lineTo(x1 + len, y1 + 4); g.stroke();
    const k = this.cur[v.fix], mm = (k * this.P.spacing[v.fix]).toFixed(1);
    v.el.querySelector(".tl").textContent = `${v.name}   ${k + 1} / ${v.n}`;
    v.el.querySelector(".tr").textContent = `W ${Math.round(this.W)}  L ${Math.round(this.L)}${v.zoom > 1 ? `  ×${v.zoom.toFixed(1)}` : ""}`;
    v.el.querySelector(".bl").textContent = `Position ${mm} mm`;
    v.el.querySelector(".br").textContent = "10 mm";
  }
  drawAll() { for (let i = 0; i < 3; i++) this.draw(i); }
  toVoxel(v, ev) {
    const r = v.el.getBoundingClientRect();
    const c = Math.floor((ev.clientX - r.left - v.x0) / (v.pw * v.scale)), rr = Math.floor((ev.clientY - r.top - v.y0) / (v.ph * v.scale));
    if (c < 0 || rr < 0 || c >= v.cols || rr >= v.rows) return null; return v.map(c, rr, this.cur[v.fix]);
  }
  readout(p) {
    if (!p || !this.onReadout) return; const q = this.at(...p);
    const sp = this.P.spacing, k = this.KD ? this.KD[q] : 0, s = this.ST ? this.ST[q] : 0;
    this.onReadout({ voxel: p, mm: p.map((u, j) => u * sp[j]), hu: this.CT[q], label: s ? `Stone ${s}` : k === 1 ? "Right kidney" : k === 2 ? "Left kidney" : "—" });
  }
  goto(stoneId) {
    const c = this.P.centres && this.P.centres[stoneId]; this.sel = +stoneId || 0;
    if (c) {
      this.cur = [...c];
      // zoom in and centre every view on the stone, so small stones are visible immediately
      for (const v of this.V) {
        if (!v.el.offsetParent) continue;
        v.zoom = Math.max(v.zoom, 3); v.pan = [0, 0]; this.layout(v);
        const [cx, cy] = v.inv(...this.cur), r = v.el.getBoundingClientRect();
        v.pan = [r.width / 2 - (v.x0 + (cx + 0.5) * v.pw * v.scale), r.height / 2 - (v.y0 + (cy + 0.5) * v.ph * v.scale)];
      }
    }
    this.q("jump").value = String(this.sel); this.drawAll(); this.readout(this.cur);
  }
  bind() {
    const q = this.q;
    this.V.forEach((v, i) => {
      let drag = null;
      v.el.addEventListener("contextmenu", e => e.preventDefault());
      v.el.addEventListener("wheel", e => { e.preventDefault();
        if (e.ctrlKey) { v.zoom = Math.min(16, Math.max(1, v.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15))); this.draw(i); return; }
        const s = (e.deltaY > 0 ? 1 : -1) * (e.shiftKey ? 5 : 1); this.cur[v.fix] = Math.min(v.n - 1, Math.max(0, this.cur[v.fix] + s)); this.drawAll(); }, { passive: false });
      v.el.addEventListener("mousedown", e => { drag = { b: e.button, x: e.clientX, y: e.clientY, L: this.L, W: this.W, pan: [...v.pan], moved: false }; });
      window.addEventListener("mouseup", e => {
        if (drag && !drag.moved && drag.b === 0) { const p = this.toVoxel(v, e);
          if (p) { this.cur = p; const s = this.ST ? this.ST[this.at(...p)] : 0; if (s) { this.sel = s; this.q("jump").value = String(s); this.onSelect(s); } this.drawAll(); this.readout(p); } }
        drag = null; });
      v.el.addEventListener("mousemove", e => { this.readout(this.toVoxel(v, e)); if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true; if (!drag.moved) return;
        if (drag.b === 2) { v.pan = [drag.pan[0] + dx, drag.pan[1] + dy]; this.draw(i); }
        else if (drag.b === 0) { this.W = Math.max(1, drag.W + dx * 4); this.L = drag.L + dy * 2; this.setPresetActive(); this.drawAll(); } });
      v.el.addEventListener("dblclick", () => { v.zoom = 1; v.pan = [0, 0]; this.draw(i); });
    });
    q("presets").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; this.L = +b.dataset.l; this.W = +b.dataset.w; this.setPresetActive(); this.drawAll(); });
    const tog = (r, key) => q(r).addEventListener("click", () => { this[key] = !this[key]; q(r).classList.toggle("on", this[key]); this.drawAll(); });
    tog("tk", "showK"); tog("ts", "showS"); tog("tx", "showX");
    if (!this.ST) { q("ts").disabled = true; q("ts").classList.remove("on"); q("ts").textContent = "Stones (none)"; }
    q("op").addEventListener("input", e => { this.op = e.target.value / 100; this.drawAll(); });
    q("zin").addEventListener("click", () => { this.V.forEach(v => v.zoom = Math.min(16, v.zoom * 1.25)); this.drawAll(); });
    q("zout").addEventListener("click", () => { this.V.forEach(v => v.zoom = Math.max(1, v.zoom / 1.25)); this.drawAll(); });
    q("reset").addEventListener("click", () => { this.V.forEach(v => { v.zoom = 1; v.pan = [0, 0]; }); this.L = 500; this.W = 2000; this.setPresetActive(); this.drawAll(); });
    q("layout").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return;
      this.single = b.dataset.v === "1"; q("layout").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
      this.grid.classList.toggle("single", this.single); this.drawAll(); });
    q("jump").addEventListener("change", e => { const s = +e.target.value; if (s) { this.goto(s); this.onSelect(s); } });
  }
}
window.CTViewer = CTViewer;
