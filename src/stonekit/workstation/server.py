"""HTTP layer of the workstation (Starlette). Binds to 127.0.0.1 only; serves the static front end and a JSON API.
No external network access is needed: the 3D library is the Plotly bundle shipped inside the installed package."""

import json
from pathlib import Path

import numpy as np
import plotly
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from stonekit.workstation import jobs, report, services

STATIC = Path(__file__).parent / "static"
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _json(data, status: int = 200) -> Response:
    return Response(json.dumps(data, cls=_Encoder, allow_nan=False, default=str).encode(), status_code=status, media_type="application/json")


def _err(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def index(_: Request):
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


async def plotly_js(_: Request):
    return FileResponse(PLOTLY_JS, media_type="application/javascript")


async def api_worklist(_: Request):
    return _json({"studies": services.worklist(), "integrity_ok": services.integrity()["all_ok"]})


async def api_study(request: Request):
    try:
        return _json(services.study(request.path_params["sid"]))
    except KeyError:
        return _err("study not found", 404)


async def api_volume(request: Request):
    try:
        return _json(services.volume_payload(request.path_params["sid"]))
    except KeyError:
        return _err("study not found", 404)
    except (OSError, ValueError) as error:
        return _err(f"images could not be read: {error}", 422)


async def api_meshes(request: Request):
    try:
        return _json(services.meshes(request.path_params["sid"]))
    except KeyError:
        return _err("study not found", 404)


async def api_evidence(_: Request):
    return _json(services.evidence())


async def api_integrity(_: Request):
    return _json(services.integrity())


async def api_sources(_: Request):
    return _json(jobs.sources())


async def api_upload(request: Request):
    body = await request.body()
    if len(body) > 1_500_000_000:
        return _err("file too large", 413)
    try:
        return _json(jobs.save_upload(request.path_params["name"], body))
    except ValueError as error:
        return _err(str(error), 400)


async def api_start(request: Request):
    data = await request.json()
    try:
        return _json(jobs.start(data.get("kind", ""), data.get("value", ""), data.get("label") or None))
    except (ValueError, RuntimeError) as error:
        return _err(str(error), 409 if isinstance(error, RuntimeError) else 400)


async def api_jobs(_: Request):
    return _json({"jobs": jobs.jobs()})


async def api_job(request: Request):
    try:
        return _json(jobs.status(request.path_params["jid"]))
    except KeyError:
        return _err("job not found", 404)


async def report_page(request: Request):
    try:
        return HTMLResponse(report.render(services.study(request.path_params["sid"]), embedded=request.query_params.get("embed") == "1"))
    except KeyError:
        return _err("study not found", 404)


async def export_json(request: Request):
    try:
        s = services.study(request.path_params["sid"])
    except KeyError:
        return _err("study not found", 404)
    payload = {"disclaimer": report.DISCLAIMER, "study": s["report"].get("case_id"), "eligibility": s["eligibility"], "qc": s["qc_items"],
               "stones": [{k: st[k] for k in ("stone_id", "volume_mm3", "max_diameter_3d_mm", "max_diameter_axial_mm", "size_band", "location", "range", "qc_flags")} for st in s["stones"]],
               "burden": (s["ranges"] or {}).get("burden"), "integrity": s["integrity"]}  # fmt: skip
    return Response(json.dumps(payload, cls=_Encoder, indent=2, default=str), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="stonekit_{payload["study"]}.json"'})


routes = [
    Route("/", index), Route("/vendor/plotly.min.js", plotly_js),
    Route("/api/worklist", api_worklist), Route("/api/studies/{sid}", api_study), Route("/api/studies/{sid}/volume", api_volume),
    Route("/api/studies/{sid}/meshes", api_meshes), Route("/api/evidence", api_evidence), Route("/api/integrity", api_integrity),
    Route("/api/sources", api_sources), Route("/api/uploads/{name}", api_upload, methods=["PUT"]),
    Route("/api/analyses", api_start, methods=["POST"]), Route("/api/analyses", api_jobs), Route("/api/analyses/{jid}", api_job),
    Route("/report/{sid}", report_page), Route("/export/{sid}.json", export_json),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
]  # fmt: skip
class _NoStaleAssets(BaseHTTPMiddleware):
    """Front-end files are small and local: always revalidate them so an updated workstation is never shown stale."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response


app = Starlette(routes=routes, middleware=[Middleware(_NoStaleAssets)])


def serve(port: int = 8600) -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
