"""CLI: watch (passive -> active loop) and friends."""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import cv2
from ergo.pose import PoseEstimator, draw_pose
from ergo.angles import joint_angles, rnle_proxies
from ergo.score import risk_score, Trigger
from ergo.buffer import ClipBuffer
from ergo.agent import make_agent
from ergo.trace import Tracer

DEFAULT_MODEL = "models/yolov8n-pose.onnx"


def _open(source: str):
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def watch(source: str, model_path: str | None, threshold: float, mock_pose: bool,
          subject_height_cm: float | None, save_annotated: str | None,
          max_seconds: float | None, fps_hint: float = 15.0):
    cap = _open(source)
    if not cap.isOpened():
        print(f"cannot open {source}")
        return 1
    est = PoseEstimator(None if mock_pose else model_path)
    trig, buf, tracer = Trigger(threshold), ClipBuffer(fps=fps_hint), Tracer()
    agent = make_agent()
    writer = None
    print(f"ergo | source={source} backend={est.backend} thr={threshold} "
          f"agent={'bedrock' if os.getenv('ERGO_MOCK_BEDROCK','1')=='0' else 'mock'}")
    t0, n, peaks = time.time(), 0, []
    event_peak = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > 960:
            frame = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))
        buf.push(frame)
        res = est.infer(frame)
        h, w = frame.shape[:2]
        ang = joint_angles(res.keypoints, w, h)
        ang.update(rnle_proxies(res.keypoints, h, subject_height_cm))
        sc = risk_score(ang)["score"]
        n += 1
        peaks.append(sc)
        tracer.log("perception", score=sc, trunk_deg=round(ang["trunk_flexion_deg"], 1),
                   H_cm=ang.get("H_cm"), backend=res.backend,
                   mode="active" if buf.active else "passive")

        if trig.update(sc) and not buf.active:
            buf.start()
            event_peak = sc
            print(f"  TRIGGER score={sc:.2f} trunk={ang['trunk_flexion_deg']:.0f}deg "
                  f"H={ang.get('H_cm')}cm -> waking agent")
        clip = buf.feed(frame)
        if clip:
            r = agent.run({"score": max(sc, event_peak), "angles": ang}, buf.frames[-6:], clip)
            print(f"  AGENT  OpenCV {sc:.2f} -> VLM {r['vlm'].get('label')}"
                  f"@{r['vlm'].get('confidence')} -> {r['decision']}  ({clip})")

        vis = draw_pose(frame, res.keypoints)
        cv2.putText(vis, f"score {sc:.2f} {'ACTIVE' if buf.active else 'passive'} "
                         f"trunk {ang['trunk_flexion_deg']:.0f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
        if save_annotated:
            if writer is None:
                writer = cv2.VideoWriter(save_annotated,
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_hint, (w, h))
            writer.write(vis)
        if max_seconds and time.time() - t0 > max_seconds:
            break
    cap.release()
    if writer:
        writer.release()
    el = time.time() - t0
    print(f"done: {n} frames in {el:.1f}s ({n/max(el,1e-6):.1f} fps) "
          f"peak score {max(peaks) if peaks else 0:.2f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ergo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="passive pipeline + active agent")
    w.add_argument("--source", default="0", help="video path, RTSP url, or camera index")
    w.add_argument("--model", default=None, help=f"ONNX pose model (default: {DEFAULT_MODEL} if present)")
    w.add_argument("--mock-pose", action="store_true", help="skip the ONNX model")
    w.add_argument("--threshold", type=float, default=float(os.getenv("ERGO_THRESHOLD", "0.65")))
    w.add_argument("--subject-height-cm", type=float, default=None,
                   help="enables RNLE H/V in cm (SAFELIFT videos: see annotations.json)")
    w.add_argument("--save", default=None, help="write annotated mp4 here")
    w.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    a = ap.parse_args(argv)

    if a.cmd == "watch":
        model = None if a.mock_pose else (a.model or (DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else None))
        return watch(a.source, model, a.threshold, model is None, a.subject_height_cm,
                     a.save, a.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
