"""FastAPI: passive ingest + active agent events + review queue + trace viewer."""
from __future__ import annotations
import glob, json, os
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="ergo — active-passive ergonomic observation")

class ResolveBody(BaseModel):
    verdict: str  # confirmed | rejected
    note: str = ""

@app.get("/health")
def health():
    import cv2
    return {"ok": True, "opencv": cv2.__version__,
            "mock_bedrock": os.getenv("ERGO_MOCK_BEDROCK", "1"),
            "mock_s3": os.getenv("ERGO_MOCK_S3", "1")}

@app.get("/events")
def events():
    out = []
    for p in sorted(glob.glob("data/s3_mock/clips/*/metadata.json")):
        out.append(json.load(open(p)))
    return {"stored": out,
            "review": [json.load(open(p)) for p in glob.glob("data/review/*.json")]}

@app.get("/trace")
def trace(limit: int = 200):
    tp = os.getenv("ERGO_TRACE_PATH", "data/trace.jsonl")
    if not os.path.exists(tp): return {"rows": []}
    rows = [json.loads(l) for l in open(tp)][-limit:]
    return {"rows": rows}

@app.post("/review/{rid}/resolve")
def resolve(rid: str, body: ResolveBody):
    p = f"data/review/{rid}.json"
    if not os.path.exists(p): return JSONResponse({"error": "not found"}, 404)
    rec = json.load(open(p)); rec.update(status=body.verdict, note=body.note)
    json.dump(rec, open(p, "w"), indent=2)
    return rec

@app.post("/ingest")
async def ingest(f: UploadFile):
    import tempfile, subprocess, sys
    tmp = tempfile.mktemp(suffix=".mp4")
    open(tmp, "wb").write(await f.read())
    # run pipeline as subprocess so API stays responsive in demo
    subprocess.Popen([sys.executable, "-m", "ergo.cli", "watch",
                      "--source", tmp, "--mock-pose"])
    return {"accepted": True}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """<html><head><title>ergo demo</title>
    <style>body{font-family:system-ui;margin:24px;max-width:900px}
    pre{background:#111;color:#0f0;padding:12px;overflow:auto}
    button{padding:8px 16px;margin:4px}</style></head><body>
    <h1>ergo — passive watch, active response</h1>
    <button onclick="load('events')">events</button>
    <button onclick="load('trace')">trace (award evidence)</button>
    <button onclick="load('health')">health</button>
    <pre id=o>click a button…</pre>
    <script>async function load(p){document.getElementById('o').textContent=
    JSON.stringify(await (await fetch('/'+p)).json(),null,1).slice(0,12000)}</script>"""
