# Devpost submission draft — ergo

**Tagline:** Passive CCTV watch that only wakes up when backs bend: OpenCV 5
pose gating a Strands + Bedrock agent that confirms risk and stores proof.

**Inspiration:** Ergonomic injuries come from lifts nobody sees. Watching every
frame with a VLM is too expensive; watching none misses everything. ergo stays
passive (cheap OpenCV 5 pose, ~6 ms/frame) and goes active only on skeletal
risk — then a Bedrock VLM takes a second look before anything is stored.

**What it does:** Ingests CCTV/RTSP, runs YOLOv8-pose through `cv2.dnn`,
computes trunk/knee/shoulder angles + RNLE H/V proxies, scores 0..1. Crossing
0.65 wakes a Strands agent: it pulls the pre/post-roll clip, asks Bedrock
(Claude/Nova multimodal) to confirm, then STOREs to S3 with metadata /
requests HUMAN_REVIEW / DISCARDS with a miss log. Every step lands in
`trace.jsonl`.

**How we built it:** OpenCV 5 (`cv2.dnn.readNetFromONNX`, letterbox+NMS
decoder), Strands Agents `@tool`s + `BedrockModel`, boto3 S3/Bedrock,
FastAPI judge UI. Evaluated on SAFELIFT (Zenodo 18243098, CC BY-NC-SA 4.0):
our H vs annotated H (MAE), flag vs LI-risk (P/R/F1), latency. Mock backend
keeps the whole loop runnable with zero AWS cost.

**Challenges:** OpenCV 4.x cannot parse YOLOv8's Split node (hence OpenCV 5);
anonymized/oblique views degrade keypoints (downweighted, escalated to human);
2D H from one camera is approximate (VLM resolves the ambiguity — that's the
point of the agentic design).

**Accomplishments:** Real ONNX pose on real lifting footage; three-way agent
policy verified (STORE / REVIEW / DISCARD); full trace from OpenCV output to
action; reproducible pins + Dockerfile + infra stub.

**What's next:** ByteTrack multi-person, on-camera RNLE LI display, COOL
Graviton bench for the $1k path, pilot with EHS team.

**Built with:** OpenCV 5, AWS (Bedrock, S3, EC2), Strands Agents, FastAPI.
**Dataset:** SAFELIFT, Zenodo 18243098 (cite IUI'26).
