"""Background analyses: each job runs scripts/demo_case.py (the frozen pipeline, 8-flip TTA and the frozen-calibration
ranges) in its own process and records its output, so the web server stays responsive. One job at a time (one GPU)."""

import csv
import re
import subprocess
import sys
import threading
import time
import uuid

from stonekit.workstation.services import ROOT, RUNS

UPLOADS = ROOT / "data" / "interim" / "uploads"
MSWAL_IMAGES = ROOT / "data" / "external" / "mswal" / "imagesTr"
MSWAL_ELIGIBILITY = ROOT / "results/runs/mswal_external_check_20261113T132418Z/eligibility.csv"
STEPS = [
    ("Load CT and validate input", 5), ("Kidney segmentation (TotalSegmentator)", 75), ("Kidney region of interest", 10),
    ("Stone segmentation (nnU-Net)", 55), ("Measurement, QC and 3D surfaces", 10), ("8-flip test-time augmentation", 100),
    ("Calibrated ranges (frozen calibration)", 25),
]  # fmt: skip
SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def sources() -> dict:
    """CT inputs available on this workstation. MSWAL phase status comes from the stored external-check eligibility."""
    phase = {}
    if MSWAL_ELIGIBILITY.exists():
        with open(MSWAL_ELIGIBILITY, encoding="utf-8") as f:
            phase = {r["case"]: r["status"] for r in csv.DictReader(f)}
    mswal = [{"name": p.name.replace("_0000.nii.gz", ""), "path": str(p.relative_to(ROOT)), "phase": phase.get(p.name.replace("_0000.nii.gz", ""), "not checked")}
             for p in sorted(MSWAL_IMAGES.glob("*_0000.nii.gz"))] if MSWAL_IMAGES.exists() else []  # fmt: skip
    uploads = [{"name": p.name.split(".")[0], "path": str(p.relative_to(ROOT))} for p in sorted(UPLOADS.glob("*.nii*"))] if UPLOADS.exists() else []
    return {"mswal": mswal, "uploads": uploads}


def save_upload(name: str, body: bytes) -> dict:
    if not SAFE.match(name) or not name.endswith((".nii.gz", ".nii")):
        raise ValueError("file name must be a NIfTI file (.nii or .nii.gz) of letters, digits, '.', '_' or '-'")
    if body[:2] != b"\x1f\x8b" and name.endswith(".gz"):
        raise ValueError("the file is not gzip-compressed although its name ends in .gz")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / name
    if path.exists():
        raise ValueError("a file with this name was already uploaded")
    path.write_bytes(body)
    return {"name": name.split(".")[0], "path": str(path.relative_to(ROOT)), "bytes": len(body)}


def start(kind: str, value: str, label: str | None) -> dict:
    with _lock:
        if any(j["state"] == "running" for j in _jobs.values()):
            raise RuntimeError("an analysis is already running; one at a time on this GPU")
        if kind == "path":
            p = (ROOT / value).resolve()
            if ROOT.resolve() not in p.parents or not p.exists():
                raise ValueError("CT file not found inside the project")
            args, name = ["--ct", str(p)], label or p.name.split(".")[0]
        elif kind == "li":
            if not re.match(r"^wd\d{4}$", value):
                raise ValueError("a Li development scan ID looks like wd0254")
            args, name = ["--case", value], label or value
        else:
            raise ValueError("unknown source")
        if not SAFE.match(name):
            raise ValueError("label must be letters, digits, '.', '_' or '-'")
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "name": name, "state": "running", "started": time.time(), "log": [], "step": 0, "study_id": None, "error": None}
        _jobs[jid] = job
    threading.Thread(target=_run, args=(job, [*args, "--name", name]), daemon=True).start()
    return status(jid)


def _run(job: dict, args: list[str]) -> None:
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "demo_case.py"), *args]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        job["log"].append(line[-400:])
        job["log"] = job["log"][-200:]
        if line.startswith("[2/3]"):
            job["step"] = 5
        elif line.startswith("[3/3]"):
            job["step"] = 6
        m = re.search(r"run folder: (\S+)", line)
        if m:
            folder = (ROOT / m.group(1)).resolve()
            if RUNS.resolve() in folder.parents:
                job["study_id"] = folder.relative_to(RUNS.resolve()).as_posix().replace("/", "~")
    code = proc.wait()
    job["finished"] = time.time()
    if code == 0:
        job["state"], job["step"] = "done", len(STEPS)
    else:
        job["state"] = "failed"
        job["error"] = next((ln for ln in reversed(job["log"]) if ln.strip()), f"exit code {code}")


def status(jid: str) -> dict:
    job = _jobs[jid]
    elapsed = (job.get("finished") or time.time()) - job["started"]
    step = job["step"]
    if job["state"] == "running" and step < 5:  # the pipeline prints one line for steps 0-4: estimate from elapsed time
        acc = 0
        for i, (_, secs) in enumerate(STEPS[:5]):
            acc += secs
            if elapsed < acc:
                step = i
                break
        else:
            step = 4
    return {"id": job["id"], "name": job["name"], "state": job["state"], "elapsed_s": round(elapsed), "step": step,
            "steps": [s for s, _ in STEPS], "estimated": step < 5 and job["state"] == "running", "study_id": job["study_id"],
            "error": job["error"], "log": job["log"][-12:]}  # fmt: skip


def jobs() -> list[dict]:
    return [status(j) for j in sorted(_jobs, key=lambda k: _jobs[k]["started"], reverse=True)]
