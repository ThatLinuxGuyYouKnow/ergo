"""ergo demo/test API: upload videos, run inference live, review flagged clips.

Serves the single-file modern UI at /ui (no CDN, works offline).
Mock backends by default (ERGO_MOCK_*=1): zero AWS cost.
"""
from __future__ import annotations
import glob
import json
import os
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import cv2

from ergo.pipeline import process_video

HERE = os.path.dirname(__file__)
JOBS = os.path.abspath("data/jobs")
os.makedirs(JOBS, exist_ok=True)

app = FastAPI(title="ergo — active-passive ergonomic observation")
app.mount("/files", StaticFiles(directory="data"), name="files")

_registry: dict[str, dict] = {}
_lock = threading.Lock()


class ResolveBody(BaseModel):
    verdict: str  # confirmed | rejected
    note: str = ""


def _job_summary(j: dict) -> dict:
    return {k: j.get(k) for k in
            ("id", "name", "status", "backend", "total", "done", "peak",
             "triggers", "error", "annotated", "score_now", "fps_live",
             "threshold", "n_events")}


def _run_job(jid: str, src: str, threshold: float, height: float | None, model: str | None):
    job = _registry[jid]
    out_dir = os.path.join(JOBS, jid)
    try:
        for p in process_video(src, model_path=model, threshold=threshold,
                               subject_height_cm=height, job=job, out_dir=out_dir):
            job["n_events"] = len(job.get("events", []))
    except Exception as e:  # never kill the server on a bad video
        job.update(status="error", error=str(e)[:300])


@app.get("/health")
def health():
    return {"ok": True, "opencv": cv2.__version__,
            "model": os.path.exists("models/yolov8n-pose.onnx"),
            "mock_bedrock": os.getenv("ERGO_MOCK_BEDROCK", "1"),
            "mock_s3": os.getenv("ERGO_MOCK_S3", "1")}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...),
                     threshold: float = Form(0.65),
                     subject_height_cm: float | None = Form(None),
                     mock_pose: bool = Form(False)):
    jid = uuid.uuid4().hex[:8]
    out_dir = os.path.join(JOBS, jid)
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
    src = os.path.join(out_dir, f"upload{ext}")
    with open(src, "wb") as f:
        f.write(await file.read())
    model = None if mock_pose or not os.path.exists("models/yolov8n-pose.onnx") else "models/yolov8n-pose.onnx"
    job = {"id": jid, "name": file.filename, "status": "queued",
           "threshold": threshold, "events": [], "n_events": 0}
    with _lock:
        _registry[jid] = job
    threading.Thread(target=_run_job, args=(jid, src, threshold, subject_height_cm, model),
                     daemon=True).start()
    return {"job_id": jid}


@app.get("/api/jobs")
def list_jobs():
    with _lock:
        return {"jobs": [_job_summary(j) for j in _registry.values()]}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = _registry.get(jid)
    if not job:
        return JSONResponse({"error": "unknown job"}, 404)
    d = _job_summary(job)
    d["events"] = job.get("events", [])
    return d


@app.get("/api/jobs/{jid}/frame")
def job_frame(jid: str):
    job = _registry.get(jid)
    jpg = (job or {}).get("frame_jpg")
    if not jpg:
        return Response(status_code=204)
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/api/events")
def all_events():
    rev_by_clip: dict[str, str] = {}
    for p in glob.glob("data/review/*.json"):
        try:
            r = json.load(open(p))
            if r.get("status") == "pending" and r.get("clip"):
                rev_by_clip[r["clip"]] = os.path.splitext(os.path.basename(p))[0]
        except Exception:
            pass
    evs = []
    with _lock:
        for j in _registry.values():
            for e in j.get("events", []):
                ev = {"job_id": j["id"], "job_name": j.get("name"), **e}
                if e.get("clip") in rev_by_clip:
                    ev["review_id"] = rev_by_clip[e["clip"]]
                evs.append(ev)
    evs.sort(key=lambda e: e.get("score", 0), reverse=True)
    return {"events": evs}


@app.get("/api/trace")
def trace(limit: int = 200):
    tp = os.getenv("ERGO_TRACE_PATH", "data/trace.jsonl")
    if not os.path.exists(tp):
        return {"rows": []}
    rows = [json.loads(l) for l in open(tp)][-limit:]
    return {"rows": rows}


@app.get("/api/review")
def review_queue():
    out = []
    for p in sorted(glob.glob("data/review/*.json")):
        r = json.load(open(p))
        r["file"] = os.path.basename(p)
        out.append(r)
    return {"pending": out}


@app.post("/api/review/{rid}/resolve")
def resolve(rid: str, body: ResolveBody):
    p = f"data/review/{rid}.json"
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, 404)
    rec = json.load(open(p))
    rec.update(status=body.verdict, note=body.note)
    json.dump(rec, open(p, "w"), indent=2)
    return rec


@app.get("/ui", response_class=HTMLResponse)
def ui():
    with open(os.path.join(HERE, "static", "app.html")) as f:
        return f.read()
