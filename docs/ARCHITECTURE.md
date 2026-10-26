# Architecture — ergo (Agentic Vision path)

```
 CCTV/RTSP ──► [PASSIVE: OpenCV 5 cv2.dnn ONNX pose] ──► angles ──► risk score
   always-on, cheap, edge/EC2              trunk/knee/H/V      thr=0.65 + hysteresis
                                                                     │
                                              score < thr              │ score ≥ thr
                                              log only                 ▼
                                                              [ACTIVE: Strands agent + Bedrock VLM]
                                                               confirm_posture(frames+angles)
                                                                 ┌──────┼──────┐
                                                          p≥.75 danger │ unsure │ false alarm
                                                                 ▼      ▼        ▼
                                                              S3+meta HUMAN_REVIEW DISCARD+miss-log
                                                              clips/  /review Q   threshold tuning
                                                                 └──────┼──────┘
                                                              trace.jsonl (every step)
```

## AWS mapping (meaningful component)

| Piece | AWS service | Notes |
|---|---|---|
| Passive pipeline host | EC2 (Graviton) / ECS + ECR image | `Dockerfile` runs `ergo watch --source $RTSP_URL` |
| Event clips + metadata | S3 `s3://$ERGO_BUCKET/clips/<event>/` | clip.mp4 + metadata.json (label, conf, score, angles, trace id) |
| Second look | Bedrock converse (`$BEDROCK_MODEL_ID`) via Strands `BedrockModel` | Claude 3.5 Sonnet or Nova Pro multimodal |
| Agent orchestration | Strands Agents (`Agent` + `@tool`) | policy in `src/ergo/agent.py` SYSTEM prompt |
| Review UI + ingest | FastAPI on EC2/ECS (`/ui /events /trace /review`) | judge endpoint |
| IAM | least-privilege | `s3:Put/GetObject`, `bedrock:Converse/InvokeModel` |

`infra/` holds a minimal S3+IAM stub. COOL (optional $1k path): rebuild on the
COOL Graviton base image and run `eval/bench_cool.py` for latency/throughput
vs stock OpenCV — not required for the Agentic Vision award.

## Why active-passive

Bedrock per-frame is $$$ and slow; OpenCV per-frame is ¢ and fast.
The trigger is the cost gate: ~99% of frames stay passive, <1% of windows
pay for a VLM call. The trace proves the gate works both ways
(OpenCV can cause STORE, and VLM can still override to DISCARD).
