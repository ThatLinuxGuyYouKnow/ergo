# ergo headless API (no GUI)

Base URL: `https://ergo-cv.duckdns.org` (or `http://localhost:8000` locally).
Interactive docs: `/docs`, `/redoc`, `/openapi.json`. Frozen spec: `docs/openapi.yaml`.

## 1. Submit a video → job

```bash
BASE=https://ergo-cv.duckdns.org
SUBMIT=$(curl -s -X POST $BASE/api/jobs \
  -F "file=@lift.mp4;type=video/mp4" \
  -F "threshold=0.4")
echo $SUBMIT
# {"job_id":"d0b72250","status_url":"/api/jobs/d0b72250","result_url":"/api/jobs/d0b72250/result"}
JID=$(echo $SUBMIT | python3 -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
```

## 2. Poll until done

```bash
while true; do
  S=$(curl -s $BASE/api/jobs/$JID | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "status=$S"; [ "$S" = done ] || [ "$S" = error ] && break
  sleep 5
done
```

## 3. Result + S3 links

```bash
curl -s $BASE/api/jobs/$JID/result | python3 -m json.tool
# annotated_url -> full annotated run (H.264, plays in browser)
# events[] -> {t, score, trunk_deg, decision, vlm_label, vlm_conf, clip_url, s3_uri}
# s3_links[] -> flat STORE URIs (empty unless score >= ~0.8 triggered a STORE)

# download the first flagged clip + annotated run:
curl -s $BASE/api/jobs/$JID/result \
 | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['annotated_url']);[print(e['clip_url']) for e in d['events']]" \
 | while read p; do curl -O $BASE$p; done
```

## Notes

- `clip_url` / `annotated_url` are same-origin `/files/...` downloads (H.264 after
  the transcode fix — playable with `<video>` / VLC / ffplay).
- `s3_uri` is set **only** on `STORE` verdicts (mock VLM needs score ≥ 0.8).
  `DISCARD` / `HUMAN_REVIEW` never touch the bucket by design — `HUMAN_REVIEW`
  goes to `GET /api/review` instead.
- Single worker/replica: the job registry is in-memory; don't scale past 1.
- `GET /api/events` = all jobs' flagged clips, highest score first.
- `GET /api/trace?limit=200` = award-trace evidence (perception → agent → action).
