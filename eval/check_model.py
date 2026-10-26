"""Validate the real ONNX backend: single-frame inference + overlay dump.

Usage:
  .venv/bin/python eval/check_model.py --video data/env2/.../jv-scrivania-terra-001.mp4 --second 3
Writes eval/check_pose.jpg (skeleton overlay) and prints keypoint confidences.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2
from ergo.pose import PoseEstimator, draw_pose
from ergo.angles import joint_angles, rnle_proxies
from ergo.score import risk_score

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--second", type=float, default=3.0)
ap.add_argument("--model", default="models/yolov8n-pose.onnx")
ap.add_argument("--out", default="eval/check_pose.jpg")
a = ap.parse_args()

est = PoseEstimator(a.model)
print("backend:", est.backend, "| cv2:", cv2.__version__)
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.second * fps))
ok, frame = cap.read()
assert ok, "seek failed"
res = est.infer(frame)
h, w = frame.shape[:2]
ang = joint_angles(res.keypoints, w, h)
print("person_conf:", round(res.person_conf, 2), "| bbox:", res.bbox)
print("mean keypoint conf:", round(float(res.keypoints[:, 2].mean()), 2))
print("angles:", {k: round(v, 1) if isinstance(v, float) else v for k, v in ang.items()})
print("risk:", risk_score(ang))
cv2.imwrite(a.out, draw_pose(frame, res.keypoints))
print("wrote", a.out)
