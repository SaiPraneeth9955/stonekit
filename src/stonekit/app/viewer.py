"""Tri-planar CT viewer for the StoneKit workstation: one self-contained HTML/JS component.

Python packs a CT region (HU, int16) plus optional kidney and stone label volumes, zlib-compressed and
base64-encoded; the browser decompresses them (DecompressionStream) and draws axial, coronal and
sagittal views on canvases. No network access, no external script.

Conventions (radiological): arrays are in RAS order (index x increases toward patient Right,
y toward Anterior, z toward Superior). Axial and coronal views show patient Right on screen left;
axial shows Anterior at the top; coronal and sagittal show Superior at the top; sagittal shows
Anterior on screen left. Pixel aspect follows the voxel spacing.

Interaction: wheel = slice (Shift = 5 slices); left-drag = window/level; click = move crosshair (all
views follow); right-drag = pan; Ctrl+wheel = zoom; double-click = reset view.
"""

import base64
import json
import zlib

import numpy as np

PRESETS = {"Stone": (500, 2000), "Bone": (300, 1500), "Soft tissue": (40, 400)}


def _pack(array: np.ndarray, dtype) -> str:
    """C-order bytes of the array as `dtype`, zlib-compressed, base64."""
    return base64.b64encode(zlib.compress(np.ascontiguousarray(array, dtype=dtype).tobytes(), 6)).decode("ascii")


def viewer_payload(ct_hu: np.ndarray, spacing_mm, kidney: np.ndarray | None = None, stones: np.ndarray | None = None,
                   selected: int | None = None, focus_voxel=None, window=(500, 2000), title: str = "") -> dict:  # fmt: skip
    """Everything the page needs. `stones` holds per-stone integer IDs (0 = background)."""
    if ct_hu.ndim != 3:
        raise ValueError("CT must be 3D")
    for name, arr in (("kidney", kidney), ("stones", stones)):
        if arr is not None and arr.shape != ct_hu.shape:
            raise ValueError(f"{name} shape {arr.shape} differs from CT shape {ct_hu.shape}")
    if stones is not None and int(stones.max(initial=0)) > 65535:
        raise ValueError("too many stone IDs for uint16")
    shape = [int(s) for s in ct_hu.shape]
    focus = [int(v) for v in focus_voxel] if focus_voxel is not None else [s // 2 for s in shape]
    return {
        "shape": shape, "spacing": [float(s) for s in spacing_mm], "title": title,
        "ct": _pack(np.clip(ct_hu, -32768, 32767), np.int16),
        "kidney": _pack(kidney > 0, np.uint8) if kidney is not None else None,
        "stones": _pack(stones, np.uint16) if stones is not None else None,
        "selected": int(selected) if selected is not None else 0,
        "focus": [min(max(f, 0), s - 1) for f, s in zip(focus, shape, strict=True)],
        "window": [float(window[0]), float(window[1])], "presets": PRESETS,
    }  # fmt: skip


def viewer_html(payload: dict, height_px: int = 720) -> str:
    return _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)).replace("__HEIGHT__", str(int(height_px)))


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
:root{--frame:#d5dae0;--ink:#1f2933;--muted:#5f6b7a;--accent:#1f5fae;--bg:#f7f8fa}
*{box-sizing:border-box}
body{margin:0;font:12px/1.35 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;color:var(--ink);background:var(--bg)}
.sk{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:4px;height:__HEIGHT__px}
.vp{position:relative;background:#000;border:1px solid var(--frame);overflow:hidden;cursor:crosshair;user-select:none;-webkit-user-select:none}
.vp canvas{position:absolute;image-rendering:pixelated}
.ov{position:absolute;inset:0;pointer-events:none}
.c{position:absolute;color:#cfd6dd;font:11px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;text-shadow:0 0 2px #000}
.tl{left:6px;top:4px}.tr{right:6px;top:4px;text-align:right}.bl{left:6px;bottom:4px}.br{right:6px;bottom:4px;text-align:right}
.tl,.tr{max-width:calc(50% - 10px);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bl{max-width:calc(100% - 84px);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.o{position:absolute;color:#e3e8ed;font:600 12px ui-monospace,Menlo,Consolas,monospace;text-shadow:0 0 2px #000}
.ol{left:4px;top:50%}.or{right:4px;top:50%}.ot{top:18px;left:50%}.ob{bottom:18px;left:50%}
.panel{border:1px solid var(--frame);background:#fff;padding:10px 12px;overflow:auto}
.panel h4{margin:0 0 6px;font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.row{display:flex;gap:6px;align-items:center;margin:4px 0 10px;flex-wrap:wrap}
button{font:12px inherit;padding:3px 8px;border:1px solid #c3cad3;background:#f4f6f8;color:var(--ink);border-radius:2px;cursor:pointer}
button.on{border-color:var(--accent);color:var(--accent);background:#eef4fb}
label{color:var(--muted)} input[type=range]{width:120px}
table{border-collapse:collapse;width:100%;font-size:12px}td{padding:2px 4px;border-bottom:1px solid #eef0f3}td:first-child{color:var(--muted);width:46%}
.sw{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px}
.hint{color:var(--muted);font-size:11px;margin-top:8px}
</style></head><body>
<div class="sk" id="sk">
 <div class="vp" id="v0"></div><div class="vp" id="v1"></div><div class="vp" id="v2"></div>
 <div class="panel" id="panel">
  <h4>Display</h4>
  <div class="row" id="presets"></div>
  <div class="row"><label>Overlay opacity</label><input type="range" id="op" min="0" max="100" value="55">
   <button id="tk" class="on">Kidney</button><button id="ts" class="on">Stones</button><button id="tx" class="on">Crosshair</button></div>
  <h4>Cursor</h4>
  <table id="cur"><tr><td>Voxel (x, y, z)</td><td>-</td></tr><tr><td>Position (mm)</td><td>-</td></tr><tr><td>Attenuation</td><td>-</td></tr><tr><td>Label</td><td>-</td></tr></table>
  <h4 style="margin-top:10px">Legend</h4>
  <div><span class="sw" style="background:#8fa3b8"></span>Kidney segmentation (outline)</div>
  <div><span class="sw" style="background:#e08a2e"></span>Stone segmentation</div>
  <div><span class="sw" style="background:#3d8bfd"></span>Selected stone</div>
  <div class="hint">Wheel: slice (Shift ×5) · Drag: window/level · Click: crosshair · Right-drag: pan · Ctrl+wheel: zoom · Double-click: reset</div>
 </div>
</div>
<script>
const P=__PAYLOAD__;
async function unpack(b64,T){if(!b64)return null;const bin=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
 const s=new Blob([bin]).stream().pipeThrough(new DecompressionStream('deflate'));const buf=await new Response(s).arrayBuffer();return new T(buf);}
(async()=>{
const [nx,ny,nz]=P.shape,[sx,sy,sz]=P.spacing;
const CT=await unpack(P.ct,Int16Array),KD=await unpack(P.kidney,Uint8Array),ST=await unpack(P.stones,Uint16Array);
const at=(x,y,z)=>(x*ny+y)*nz+z; // C order of a (nx,ny,nz) array
let L=P.window[0],W=P.window[1],cur=[...P.focus],op=0.55,showK=!!KD,showS=!!ST,showX=true,sel=P.selected;
// view definitions: which axis is fixed, and how screen (col,row) maps to (x,y,z)
const V=[
 {name:'Axial',fix:2,cols:nx,rows:ny,pw:sx,ph:sy,map:(c,r,k)=>[nx-1-c,ny-1-r,k],inv:(x,y,z)=>[nx-1-x,ny-1-y],lab:['R','L','A','P'],n:nz},
 {name:'Coronal',fix:1,cols:nx,rows:nz,pw:sx,ph:sz,map:(c,r,k)=>[nx-1-c,k,nz-1-r],inv:(x,y,z)=>[nx-1-x,nz-1-z],lab:['R','L','S','I'],n:ny},
 {name:'Sagittal',fix:0,cols:ny,rows:nz,pw:sy,ph:sz,map:(c,r,k)=>[k,ny-1-c,nz-1-r],inv:(x,y,z)=>[ny-1-y,nz-1-z],lab:['A','P','S','I'],n:nx}];
for(let i=0;i<3;i++){const v=V[i],el=document.getElementById('v'+i);v.el=el;v.zoom=1;v.pan=[0,0];
 v.cv=document.createElement('canvas');v.cv.width=v.cols;v.cv.height=v.rows;el.appendChild(v.cv);
 v.oc=document.createElement('canvas');v.oc.className='ov';el.appendChild(v.oc);
 el.insertAdjacentHTML('beforeend',`<div class="c tl"></div><div class="c tr"></div><div class="c bl"></div><div class="c br"></div>
 <div class="o ol">${v.lab[0]}</div><div class="o or">${v.lab[1]}</div><div class="o ot">${v.lab[2]}</div><div class="o ob">${v.lab[3]}</div>`);
 v.img=v.cv.getContext('2d').createImageData(v.cols,v.rows);}
function layout(v){const r=v.el.getBoundingClientRect();const wmm=v.cols*v.pw,hmm=v.rows*v.ph;
 const s=Math.min(r.width/wmm,r.height/hmm)*v.zoom;v.scale=s;const w=wmm*s,h=hmm*s;
 v.x0=(r.width-w)/2+v.pan[0];v.y0=(r.height-h)/2+v.pan[1];
 Object.assign(v.cv.style,{left:v.x0+'px',top:v.y0+'px',width:w+'px',height:h+'px'});
 v.oc.width=r.width;v.oc.height=r.height;}
function draw(i){const v=V[i],k=cur[v.fix],d=v.img.data,lo=L-W/2,a=op;
 for(let r=0;r<v.rows;r++)for(let c=0;c<v.cols;c++){const [x,y,z]=v.map(c,r,k),q=at(x,y,z);
  let g=(CT[q]-lo)/W*255;g=g<0?0:g>255?255:g;let R=g,G=g,B=g;const o=(r*v.cols+c)*4;
  if(showS&&ST){const s=ST[q];if(s){const col=s===sel?[61,139,253]:[224,138,46];
    // outline: any in-plane neighbour with a different id
    let edge=false;for(const [dc,dr] of [[1,0],[-1,0],[0,1],[0,-1]]){const cc=c+dc,rr=r+dr;
     if(cc<0||rr<0||cc>=v.cols||rr>=v.rows){edge=true;break}const [x2,y2,z2]=v.map(cc,rr,k);if(ST[at(x2,y2,z2)]!==s){edge=true;break}}
    const f=edge?Math.max(a,0.9):a*(s===sel?0.25:0.35);R=R*(1-f)+col[0]*f;G=G*(1-f)+col[1]*f;B=B*(1-f)+col[2]*f;}}
  if(showK&&KD&&KD[q]){let edge=false;for(const [dc,dr] of [[1,0],[-1,0],[0,1],[0,-1]]){const cc=c+dc,rr=r+dr;
    if(cc<0||rr<0||cc>=v.cols||rr>=v.rows){continue}const [x2,y2,z2]=v.map(cc,rr,k);if(!KD[at(x2,y2,z2)]){edge=true;break}}
   if(edge){const f=Math.max(a,0.6);R=R*(1-f)+143*f;G=G*(1-f)+163*f;B=B*(1-f)+184*f;}}
  d[o]=R;d[o+1]=G;d[o+2]=B;d[o+3]=255;}
 v.cv.getContext('2d').putImageData(v.img,0,0);layout(v);overlay(i);}
function overlay(i){const v=V[i],g=v.oc.getContext('2d');g.clearRect(0,0,v.oc.width,v.oc.height);
 const [cx,cy]=v.inv(...cur);
 if(showX){g.strokeStyle='rgba(120,170,230,0.55)';g.setLineDash([4,4]);g.lineWidth=1;
  const px=v.x0+(cx+0.5)*v.pw*v.scale,py=v.y0+(cy+0.5)*v.ph*v.scale;
  g.beginPath();g.moveTo(px,0);g.lineTo(px,v.oc.height);g.moveTo(0,py);g.lineTo(v.oc.width,py);g.stroke();g.setLineDash([]);}
 // 10 mm scale bar
 const len=10*v.scale,x1=v.oc.width-16-len,y1=v.oc.height-22;g.strokeStyle='#cfd6dd';g.lineWidth=2;
 g.beginPath();g.moveTo(x1,y1);g.lineTo(x1+len,y1);g.moveTo(x1,y1-4);g.lineTo(x1,y1+4);g.moveTo(x1+len,y1-4);g.lineTo(x1+len,y1+4);g.stroke();
 const k=cur[v.fix],mm=(k*P.spacing[v.fix]).toFixed(1);
 v.el.querySelector('.tl').textContent=`${v.name}  ${k+1}/${v.n}`;
 v.el.querySelector('.tr').textContent=`W${Math.round(W)} L${Math.round(L)}`;
 v.el.querySelector('.bl').textContent=`slice at ${mm} mm`;
 v.el.querySelector('.br').textContent='10 mm';}
const all=()=>{for(let i=0;i<3;i++)draw(i)};
function toVoxel(v,ev){const r=v.el.getBoundingClientRect();const c=Math.floor((ev.clientX-r.left-v.x0)/(v.pw*v.scale)),
 rr=Math.floor((ev.clientY-r.top-v.y0)/(v.ph*v.scale));if(c<0||rr<0||c>=v.cols||rr>=v.rows)return null;return v.map(c,rr,cur[v.fix]);}
function readout(p){const t=document.getElementById('cur').rows;if(!p){return}const q=at(...p);
 t[0].cells[1].textContent=p.join(', ');t[1].cells[1].textContent=p.map((u,j)=>(u*P.spacing[j]).toFixed(1)).join(', ');
 t[2].cells[1].textContent=CT[q]+' HU';t[3].cells[1].textContent=(ST&&ST[q]?'Stone '+ST[q]:(KD&&KD[q]?'Kidney':'-'));}
V.forEach((v,i)=>{let drag=null;
 v.el.addEventListener('contextmenu',e=>e.preventDefault());
 v.el.addEventListener('wheel',e=>{e.preventDefault();if(e.ctrlKey){v.zoom=Math.min(16,Math.max(1,v.zoom*(e.deltaY<0?1.15:1/1.15)));draw(i);return}
  const s=(e.deltaY>0?1:-1)*(e.shiftKey?5:1);cur[v.fix]=Math.min(v.n-1,Math.max(0,cur[v.fix]+s));all();},{passive:false});
 v.el.addEventListener('mousedown',e=>{drag={b:e.button,x:e.clientX,y:e.clientY,L,W,pan:[...v.pan],moved:false}});
 window.addEventListener('mouseup',e=>{if(drag&&!drag.moved&&drag.b===0){const p=toVoxel(v,e);if(p){cur=p;all();readout(p)}}drag=null});
 v.el.addEventListener('mousemove',e=>{readout(toVoxel(v,e));if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
  if(Math.abs(dx)+Math.abs(dy)>3)drag.moved=true;if(!drag.moved)return;
  if(drag.b===2){v.pan=[drag.pan[0]+dx,drag.pan[1]+dy];draw(i)}else if(drag.b===0){W=Math.max(1,drag.W+dx*4);L=drag.L+dy*2;all()}});
 v.el.addEventListener('dblclick',()=>{v.zoom=1;v.pan=[0,0];draw(i)});});
const pr=document.getElementById('presets');for(const [n,[l,w]] of Object.entries(P.presets)){const b=document.createElement('button');
 b.textContent=`${n} ${w}/${l}`;b.onclick=()=>{L=l;W=w;all()};pr.appendChild(b);}
document.getElementById('op').oninput=e=>{op=e.target.value/100;all()};
const tog=(id,f)=>{const b=document.getElementById(id);b.onclick=()=>{f();b.classList.toggle('on');all()}};
tog('tk',()=>showK=!showK);tog('ts',()=>showS=!showS);tog('tx',()=>showX=!showX);
for(const [id,arr,txt] of [['tk',KD,'Kidney'],['ts',ST,'Stones']])if(!arr){const b=document.getElementById(id);b.disabled=true;b.classList.remove('on');b.textContent=txt+' (none)';}
new ResizeObserver(all).observe(document.getElementById('sk'));all();readout(cur);
})();
</script></body></html>"""
