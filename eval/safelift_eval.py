"""SAFELIFT eval: our OpenCV estimates vs RNLE ground truth (Zenodo 18243098).

Ground truth per lift (annotations.json): H (cm, mid-ankles->hands) at
action_start/action_end, V where known, subject_height_cm for our px->cm scale.

Metrics per video (action window only):
  * H-MAE: |our H_cm - annotated H| at start/end second
  * flag: peak risk score >= thr  vs  GT-risk = (start_H > 63 or end_H > 63)
    -> precision / recall / F1 + confusion
  * latency: ms/frame for the passive pipeline
  * failure axes recorded per video: low visibility, no detection, oblique

Usage:
  python eval/safelift_eval.py --annotations data/annotations.json \\
      --videos data/env2 --out eval/report.json [--limit 10] [--model models/yolov8n-pose.onnx]
  (no --model => mock backend; real numbers need OpenCV 5 + the ONNX file)
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2
from ergo.pose import PoseEstimator
from ergo.angles import joint_angles, rnle_proxies
from ergo.score import risk_score

RNLE_H_RISK = 63.0


def find_video(videos_dir: str, name: str) -> str | None:
    for cand in (os.path.join(videos_dir, name),
                 os.path.join(videos_dir, name.lower()),
                 os.path.join(videos_dir, name.replace(" ", "_"))):
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(videos_dir, "**", name), recursive=True)
    return hits[0] if hits else None


def run_video(path: str, est: PoseEstimator, t0: float, t1: float,
              height_cm: float | None, fps_probe: int = 8):
    """Sample the action window; return peak score, H at window edges, latency."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    f0, f1 = int(t0 * fps), int(t1 * fps)
    f1 = max(f1, f0 + 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    peak, hs, lat, n = 0.0, [], [], 0
    for f in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if f != f0 and f != f1 and f % max(int(fps / fps_probe), 1):
            continue
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
        t = time.time()
        res = est.infer(frame)
        h, w = frame.shape[:2]
        ang = joint_angles(res.keypoints, w, h)
        ang.update(rnle_proxies(res.keypoints, h, height_cm))
        sc = risk_score(ang)["score"]
        lat.append((time.time() - t) * 1000)
        peak = max(peak, sc)
        n += 1
        if f == f0 or abs(f - f1) <= 1:
            hs.append(ang.get("H_cm"))
    cap.release()
    return {"peak": round(peak, 3),
            "H_start_cm": hs[0] if hs else None,
            "H_end_cm": hs[-1] if hs else None,
            "frames": n, "ms_per_frame": round(sum(lat) / max(len(lat), 1), 1),
            "visibility_ok": n > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="eval/report.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    ann = json.load(open(a.annotations))
    model = None if a.mock else (a.model or "models/yolov8n-pose.onnx")
    if model and not os.path.exists(model):
        print(f"model {model} missing -> mock backend"); model = None
    est = PoseEstimator(model)

    def known(v): return v != "unknown" and v is not None
    rows, skipped = [], 0
    for item in (ann if isinstance(ann, list) else ann.get("annotations", []))[:a.limit]:
        vp = find_video(a.videos, item["video"])
        if not vp:
            skipped += 1
            continue
        gt_h0 = item["action_start"].get("box_distance_horizontal_from_mid_ankles_cm")
        gt_h1 = item["action_end"].get("box_distance_horizontal_from_mid_ankles_cm")
        r = run_video(vp, est, item["action_start"]["video_second"],
                      item["action_end"]["video_second"], item.get("subject_height_cm"))
        gt_risk = any(isinstance(v, (int, float)) and v > RNLE_H_RISK for v in (gt_h0, gt_h1))
        rows.append({**r, "video": item["video"], "flag": r["peak"] >= a.threshold,
                     "gt_risk": gt_risk, "gt_H0": gt_h0, "gt_H1": gt_h1})

    tp = sum(r["flag"] and r["gt_risk"] for r in rows)
    fp = sum(r["flag"] and not r["gt_risk"] for r in rows)
    fn = sum(not r["flag"] and r["gt_risk"] for r in rows)
    herr = [abs(r["H_start_cm"] - r["gt_H0"]) for r in rows
            if r["H_start_cm"] and isinstance(r["gt_H0"], (int, float))]
    rep = {"backend": est.backend, "n": len(rows), "skipped_missing_video": skipped,
           "precision": round(tp / max(tp + fp, 1), 3),
           "recall": round(tp / max(tp + fn, 1), 3),
           "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": len(rows) - tp - fp - fn},
           "H_MAE_cm": round(sum(herr) / max(len(herr), 1), 1) if herr else None,
           "H_pairs": len(herr),
           "mean_ms_per_frame": round(sum(r["ms_per_frame"] for r in rows) / max(len(rows), 1), 1),
           "rows": rows}
    rep["f1"] = round(2 * rep["precision"] * rep["recall"] / max(rep["precision"] + rep["recall"], 1e-9), 3)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
