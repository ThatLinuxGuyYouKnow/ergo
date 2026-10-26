"""Shared video pipeline: passive OpenCV watch -> active Strands/Bedrock agent.

Used by both the CLI (`cli.py`) and the demo API (`api.py`), so judges see
the exact same code path in terminal and browser. Yields progress dicts;
records events (clip + OpenCV numbers + VLM verdict + decision) into job.
"""
from __future__ import annotations
import os
import time
from typing import Iterable

import cv2

from .pose import PoseEstimator, draw_pose
from .angles import joint_angles, rnle_proxies
from .score import risk_score, Trigger
from .buffer import ClipBuffer
from .agent import make_agent
from .trace import Tracer


def _writer(path: str, fps: float, w: int, h: int):
    for tag in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag), fps, (w, h))
        if vw.isOpened():
            return vw
    raise RuntimeError("no mp4 writer available")


def process_video(src_path: str, *, model_path: str | None = None,
                  threshold: float = 0.65, subject_height_cm: float | None = None,
                  job: dict | None = None, out_dir: str = "data/jobs/manual",
                  max_seconds: float | None = None,
                  preview_width: int = 480) -> Iterable[dict]:
    job = job if job is not None else {}
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(int(src_path) if str(src_path).isdigit() else src_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

    est = PoseEstimator(model_path)
    trig, buf, tracer = Trigger(threshold), ClipBuffer(out_dir=out_dir), Tracer()
    agent = make_agent()
    ann_path = os.path.join(out_dir, "annotated.mp4")
    writer, (W, H) = None, (None, None)
    job.update(status="running", backend=est.backend, total=total,
               done=0, peak=0.0, triggers=0, events=job.get("events", []),
               annotated=os.path.relpath(ann_path, "data").replace(os.sep, "/"))

    t0, n, event_peak = time.time(), 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > 960:
            frame = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))
        buf.push(frame)
        res = est.infer(frame)
        h, w = frame.shape[:2]
        if writer is None:
            W, H = w, h
            writer = _writer(ann_path, min(fps, 30), w, h)
        ang = joint_angles(res.keypoints, w, h)
        ang.update(rnle_proxies(res.keypoints, h, subject_height_cm))
        sc = risk_score(ang)["score"]
        n += 1
        job["done"], job["peak"] = n, max(job["peak"], sc)
        tracer.log("perception", job=job.get("id"), score=sc,
                   trunk_deg=round(ang["trunk_flexion_deg"], 1),
                   H_cm=ang.get("H_cm"), backend=res.backend,
                   mode="active" if buf.active else "passive")

        if trig.update(sc) and not buf.active:
            buf.start()
            event_peak = sc
            job["triggers"] += 1
        event_peak = max(event_peak, sc) if buf.active else event_peak
        clip = buf.feed(frame)
        if clip:
            esc = round(max(sc, event_peak), 3)
            r = agent.run({"score": esc, "angles": ang},
                          buf.frames[-6:], clip)
            event_peak = 0.0
            ev = {"t": round(time.time() - t0, 1),
                  "clip": os.path.relpath(clip, "data").replace(os.sep, "/"),
                  "score": esc,
                  "trunk_deg": round(ang["trunk_flexion_deg"], 1),
                  "H_cm": ang.get("H_cm"), "backend": res.backend,
                  "vlm_label": r["vlm"].get("label"), "vlm_conf": r["vlm"].get("confidence"),
                  "rationale": r["vlm"].get("rationale", ""),
                  "decision": r["decision"]}
            ev["score"] = round(max(sc, 0.0), 3)
            job["events"].append(ev)
            job["last_event"] = ev

        vis = draw_pose(frame, res.keypoints)
        bar = "ACTIVE" if buf.active else "passive"
        cv2.putText(vis, f"{sc:.2f} {bar} trunk {ang['trunk_flexion_deg']:.0f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
        writer.write(vis)
        _, jpg = cv2.imencode(".jpg", cv2.resize(
            vis, (preview_width, int(preview_width * h / w))),
            [cv2.IMWRITE_JPEG_QUALITY, 65])
        job["frame_jpg"] = jpg.tobytes()
        job["score_now"] = sc
        if max_seconds and time.time() - t0 > max_seconds:
            break
        yield {"done": n, "total": total, "score": sc, "peak": job["peak"],
               "status": "running", "triggers": job["triggers"]}
    cap.release()
    if writer is not None:
        writer.release()
    job.update(status="done", fps_live=round(n / max(time.time() - t0, 1e-6), 1))
    yield {"done": n, "total": total, "score": job.get("score_now", 0),
           "peak": job["peak"], "status": "done", "triggers": job["triggers"]}
