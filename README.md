# ergo — active-passive ergonomic observation for factory / storage floors
# OpenCV AI Competition 2026, powered by AWS — Agentic Vision path

Passive OpenCV 5 pose pipeline watches CCTV/RTSP. Only when a skeletal
risk score crosses threshold does a Strands + Bedrock agent wake up,
look closer (multimodal VLM), and decide: STORE / HUMAN_REVIEW / DISCARD.
Every decision keeps a JSONL trace proving OpenCV output changed what happened next.

## Quickstart (judge / demo, 3 min)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add AWS creds + BEDROCK_MODEL_ID

# 1. run passive pipeline on a SAFELIFT clip or webcam (mock pose if no ONNX yet)
python -m ergo.cli watch --source samples/safelift_demo.mp4 --threshold 0.65 --mock-pose

# with a real ONNX pose model via cv2.dnn:
python -m ergo.cli watch --source 0 --pose-model models/human_pose.onnx

# 2. run the demo UI (upload videos, watch inference live, review flagged clips)
pip install -e .   # or: PYTHONPATH=src uvicorn ergo.api:app
uvicorn ergo.api:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/ui  (Analyze → upload, Flagged clips → review, Trace → award evidence)
# GET  /health  /events  /trace  /review
# POST /ingest (multipart video)  POST /review/{id}/resolve
```

Mock mode (`ERGO_MOCK_BEDROCK=1`, `ERGO_MOCK_S3=1`) runs with zero AWS cost.
Real mode uses Bedrock `converse` (Claude/Nova multimodal) via Strands, S3 via boto3.

## EC2 deployment (judge-live in one command)

```bash
# on an Amazon Linux 2023 instance (t3.medium+, SG opens 22 + 8000):
rsync -avz --exclude .venv --exclude data/jobs -e "ssh -i key.pem" ./ ec2-user@<host>:~/ergo/
ssh -i key.pem ec2-user@<host> "cd ~/ergo && bash infra/ec2-deploy.sh"
# open http://<host>:8000/ui
```

Full guide (instance sizing, IAM role, real S3/Bedrock mode, nginx, TLS,
troubleshooting): [`docs/DEPLOY.md`](docs/DEPLOY.md).
Single worker only — the job registry is in-memory (`--workers 1` is baked in).

## OpenCV 5 pose model (real, downloaded)

- `models/yolov8n-pose.onnx` (13.5 MB) — Ultralytics YOLOv8-nano pose,
  single-shot COCO-17 keypoints, `cv2.dnn.readNetFromONNX` + letterbox 640 +
  NMS, most-confident person kept. sha256
  `04f6d2416268f2aba6c5ba8b26de33ed9eba3279f972ac02e33f9f1366547586`.
- Requires **OpenCV 5** (`opencv-python==5.0.0.93`): OpenCV 4.x cannot parse
  the YOLOv8 `Split` ONNX node. Verify: `python -c "import cv2;print(cv2.__version__)"`.
- No model / CI without weights: `--mock-pose` silhouette heuristic keeps the
  whole pipeline, agent, trace and API runnable with zero downloads.

## Layout

```
src/ergo/
  pose.py       # OpenCV 5 cv2.dnn ONNX pose + fallback detector
  angles.py     # joint angles (trunk, knee, shoulder, H/V/D approx)
  score.py      # posture risk score 0..1 + threshold + hysteresis
  buffer.py     # rolling clip buffer (pre/post roll)
  bedrock.py    # Bedrock multimodal confirm (Strands BedrockModel under the hood)
  tools.py      # Strands @tool defs: confirm_posture, store_clip, request_review, discard
  agent.py      # Strands Agent: perception-decision-action loop
  store.py      # S3 clip + metadata (+ local fallback)
  trace.py      # JSONL trace logger (the Agentic Vision award evidence)
  api.py        # FastAPI: ingest, events, trace, review UI
  cli.py        # watch / eval / demo commands
eval/
  safelift_eval.py  # H/V/D + LI scoring vs SAFELIFT annotations.json
docs/
  ARCHITECTURE.md   # diagram + AWS path
  TECHNICAL_REPORT.md  # submission report skeleton (fill eval numbers)
  VIDEO_SCRIPT.md   # <=5 min judge video shot list
  TRACE_EXAMPLE.jsonl  # real trace sample showing OpenCV -> decision change
```

## The agentic loop (what judges check)

```
frame -> OpenCV5 pose (cv2.dnn) -> angles -> score
  score < thr : stay PASSIVE (log only)
  score >= thr: become ACTIVE ->
    Strands agent pulls clip (pre/post roll) ->
      tool: confirm_posture (Bedrock VLM on 3-5 frames + OpenCV angles) ->
        confident dangerous -> tool: store_clip (S3 + metadata)
        unsure               -> tool: request_review (human queue)
        false alarm          -> tool: discard (log miss for threshold tuning)
    every step appended to trace.jsonl
```

See `docs/TRACE_EXAMPLE.jsonl` — OpenCV `trunk_flexion: 68° + score 0.81`
triggers the agent; Bedrock `confirm: bending_risk p=0.91` flips the
outcome from `would_ignore` to `STORE`. That trace is the award evidence.

## SAFELIFT dataset (Zenodo 18243098, CC BY-NC-SA 4.0)

Annotated lifting videos (floor -> surface) with H/V/D for RNLE RWL/LI.
Download one env zip + `annotations.json`, then:

```bash
python eval/safelift_eval.py --annotations data/annotations.json --videos data/env1/ --out eval/report.json
```

Maps our trunk/knee/H estimates to LI bands and reports precision/recall
+ failure cases (occlusion, oblique view, anonymization blur).

## AWS (meaningful component, required)

- EC2/Graviton or local container runs the OpenCV 5 pipeline (Dockerfile).
- S3 (`s3://$ERGO_BUCKET/clips/`) stores event clips + `metadata.json`.
- Bedrock (`$BEDROCK_MODEL_ID`, e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`
  or `amazon.nova-pro-v1:0`) does the second look via Strands.
- IAM: least-privilege `s3:PutObject/GetObject`, `bedrock:InvokeModel/Converse`.

Infra stub in `infra/` (S3 + IAM). COOL path is optional — swap base image
for the COOL Graviton build and rerun `eval/bench_cool.py` if pursuing it.

## Submission checklist

- [ ] `pip freeze` matches requirements.txt pins
- [ ] `docs/TECHNICAL_REPORT.md` filled (problem, users, arch, OpenCV5 impl, AWS, eval, limits, responsible use)
- [ ] `docs/ARCHITECTURE.md` diagram exported
- [ ] Judge video (<5 min, team + working app + arch + results)
- [ ] Public/private repo link + live endpoint or screenshare slot
