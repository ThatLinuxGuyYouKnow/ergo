# Technical report — ergo (fill numbers before submitting)

## 1. Problem + users
Manual ergonomic audits miss most risky lifts; continuous human monitoring
doesn't scale and always-on VLM is too costly. Users: EHS managers, warehouse
ops, insurers. Ergo flags only skeletal-risky moments for review/action.

## 2. Architecture
Passive OpenCV 5 + active Strands/Bedrock (see ARCHITECTURE.md diagram).
Agent workflow: perception (pose→angles→score) → decision (VLM confirm) →
action (STORE / HUMAN_REVIEW / DISCARD) with trace.

## 3. OpenCV 5 implementation
- `cv2.dnn.readNetFromONNX("models/yolov8n-pose.onnx")` (YOLOv8n-pose, 13.5 MB,
  COCO-17, letterbox 640 + NMS, top-1 person) → keypoints (`pose.py`,
  incl. `draw_pose` overlay). Heatmap-style ONNX exports also decode.
  Mock backend keeps everything runnable without weights.
- Joint angles: trunk flexion, knee flexion, shoulder elevation, H/V proxies (`angles.py`).
- Risk score + hysteresis trigger (`score.py`); rolling clip buffer (`buffer.py`).
- Pin: `opencv-python==5.0.0.93`. Verify: `python -c "import cv2;print(cv2.__version__)"`.

## 4. AWS deployment
EC2/ECS runs pipeline; S3 stores clips+metadata; Bedrock (Claude/Nova via
Strands BedrockModel) second look; FastAPI judge endpoint `/ui /events /trace`.
Mock env flags allow zero-cost judging. IAM least-privilege.

## 5. Evaluation (SAFELIFT, Zenodo 18243098)
Harness: `eval/safelift_eval.py`. Report: P/R/F1 of score≥thr vs LI>1,
latency (ms/frame passive, s/event active), cost/event, confusion matrix.
TODO — paste your numbers here before submit.

## 6. Failure cases / limitations
Occlusion (forklift/pallet hides hips), oblique view skews H, anonymization
blur lowers conf (score downweighted), multi-person needs tracking (ByteTrack
future), mock pose overestimates bend on wide boxes.

## 7. Responsible use
Worker-facing signage, no PII in metadata (no faces stored, blur option),
review queue human-in-the-loop for unsure cases, misses log for bias audit,
CC BY-NC-SA 4.0 dataset compliance (cite SAFELIFT IUI'26).
