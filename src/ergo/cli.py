"""CLI: watch (passive -> active loop) and friends. Same pipeline as the demo UI."""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ergo.pipeline import process_video  # noqa: E402

DEFAULT_MODEL = "models/yolov8n-pose.onnx"


def watch(source: str, model: str | None, threshold: float,
          subject_height_cm: float | None, save_dir: str,
          max_seconds: float | None):
    job: dict = {"id": "cli"}
    t0 = time.time()
    try:
        for p in process_video(source, model_path=model, threshold=threshold,
                               subject_height_cm=subject_height_cm, job=job,
                               out_dir=save_dir, max_seconds=max_seconds):
            if p["done"] % 60 == 0 or p["status"] == "done":
                print(f"  f={p['done']}/{p['total'] or '?'} score={p['score']:.2f} "
                      f"peak={p['peak']:.2f} triggers={p['triggers']}", flush=True)
    except RuntimeError as e:
        print(f"ergo: {e}")
        return 1
    for ev in job.get("events", []):
        print(f"  EVENT t={ev['t']}s OpenCV {ev['score']} "
              f"-> VLM {ev['vlm_label']}@{ev['vlm_conf']} -> {ev['decision']} ({ev['clip']})")
    el = time.time() - t0
    print(f"done: {job['done']} frames in {el:.1f}s "
          f"({job['done'] / max(el, 1e-6):.1f} fps) peak {job['peak']:.2f} "
          f"backend={job.get('backend')} events={len(job.get('events', []))}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ergo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="passive pipeline + active agent")
    w.add_argument("--source", default="0", help="video path, RTSP url, or camera index")
    w.add_argument("--model", default=None, help=f"ONNX pose model (default: {DEFAULT_MODEL} if present)")
    w.add_argument("--mock-pose", action="store_true", help="skip the ONNX model")
    w.add_argument("--threshold", type=float, default=float(os.getenv("ERGO_THRESHOLD", "0.65")))
    w.add_argument("--subject-height-cm", type=float, default=None)
    w.add_argument("--save-dir", default="data/jobs/cli")
    w.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    a = ap.parse_args(argv)

    if a.cmd == "watch":
        model = None if a.mock_pose else (a.model or (DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else None))
        print(f"ergo | backend={'mock' if model is None else model} thr={a.threshold}")
        return watch(a.source, model, a.threshold, a.subject_height_cm, a.save_dir, a.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
