# ergo EC2 deployment

Single-container deploy (Docker Compose) on Amazon Linux 2023.
The demo UI, API, OpenCV 5 pipeline, and Strands/Bedrock agent all run in one container.

## 0. Pick the instance

| Need | Suggestion |
|---|---|
| Demo / judges (mock backends) | `t3.medium` (2 vCPU, 4 GB) — fine |
| Real YOLOv8-pose on video | `t3.large` or `c6i.large` (faster DNN decode) |
| COOL award path | `t4g.medium` (Graviton2) with the COOL base image |

Disk: 20 GB gp3 is plenty (model 14 MB + jobs volume).
Security group inbound: **22** (you), **8000** (demo) — or **80** if you put nginx in front. No other ports.

## 1. Launch + IAM

1. Launch Amazon Linux 2023 with the security group above.
2. Attach an IAM instance profile from `infra/s3-iam.yaml` (CloudFormation):
   `s3:PutObject/GetObject` on the clip bucket + `bedrock:InvokeModel/Converse*`.
   With the role attached you need **no AWS keys on the box** — boto3 uses the role.
3. `ssh -i key.pem ec2-user@<host>`, then copy the repo up:
   ```bash
   # from your laptop:
   rsync -avz --exclude .venv --exclude data/jobs --exclude data/env2 \
     -e "ssh -i key.pem" ./ ec2-user@<host>:~/ergo/
   ```

## 2. Deploy (one command)

```bash
cd ~/ergo
bash infra/ec2-deploy.sh
# → builds, starts, waits for /health, prints the UI URL
```

Open `http://<host>:8000/ui` (root `/` redirects there too):
**Run inference** → upload a clip → watch live scoring → **Flagged clips** tab to play event videos.

## 3. Real AWS mode (S3 + Bedrock)

Mock backends are the default (zero cost). To go live, on the instance:

```bash
cp .env.example .env   # if not already created by the deploy script
nano .env
```

Set:

```
AWS_REGION=us-east-1
ERGO_BUCKET=<bucket-from-s3-iam-stack>
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
ERGO_MOCK_BEDROCK=0
ERGO_MOCK_S3=0
```

Then `docker compose up -d` to restart. Verify `/health` shows
`"mock_bedrock": "0", "mock_s3": "0"`.

Bedrock notes: the model must be enabled in Bedrock console (Model access),
in the same region as `AWS_REGION`. First live call can take ~30 s (cold start).

## 4. Room to grow

- **nginx on port 80** (optional): `infra/nginx-ergo.conf` has a ready
  reverse-proxy with a 200 MB upload limit. Copy to `/etc/nginx/conf.d/`.
- **TLS**: terminate with an ALB + ACM cert in front of the instance, or
  certbot on the box. The app itself stays plain HTTP behind it.
- **Persistence**: `./data` is bind-mounted, so jobs, clips, review queue,
  and `trace.jsonl` survive container rebuilds. Back up `data/` to S3
  (`aws s3 sync data/ s3://<bucket>/ergo-backup/`) before terminating.
- **Logs**: `docker compose logs -f` (uvicorn access + app). Trace evidence
  lives in `data/trace.jsonl` and the UI Trace tab.

## 5. Constraints (do not "fix" by scaling)

- **Exactly 1 worker / 1 replica.** The job registry is in-memory, so
  `--workers 1` is baked into the Dockerfile and compose runs one container.
  Scaling out needs a shared store (Redis/S3) first — out of scope for this package.
- Uploads are buffered to `data/jobs/<id>/upload.*`; allow ~2× the source
  video size in free disk per concurrent job.
- Long videos: inference is ~real-time on CPU; a 5-min clip takes ~5 min
  to score. The UI polls progress — the connection can idle, nginx timeouts
  are set to 600 s.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` shows `"opencv": "4.x"` | You built on the wrong base — rebuild, don't `pip install opencv-python` over it; the Dockerfile pins `opencv-python==5.0.0.93` (YOLOv8 `Split` node needs DNN from OpenCV 5) |
| `cannot open upload...` / 0-frame jobs | Corrupt upload or codec; re-upload an H.264 `.mp4` (the pipeline writes `avc1`, which browsers play) |
| Bedrock `AccessDenied` | Model not enabled in that region, or IAM role missing `bedrock:Converse*` — see `infra/s3-iam.yaml` |
| S3 `NoSuchBucket` | `ERGO_BUCKET` must exist; the CFN stack creates `ergo-clips-<acct>-<region>` |
| Container unhealthy | `docker compose logs --tail=100`; usually missing `.env` or port clash — `PORT=8001 bash infra/ec2-deploy.sh` moves it |
